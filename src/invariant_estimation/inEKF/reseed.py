r"""
inEKF/reseed.py
===============
Touchdown **re-seed** for the world-centric, right-invariant contact InEKF —
the port of Java `InvariantEKF.reseedContact` + `TouchdownReseedLatch`
(CLAUDE.md §2 "Touchdown reseed", G5; spec in `TEST_SUITE_MAP.md`
§`InvariantEKFReseedTest`).

Why this exists (the null-space argument)
-----------------------------------------
`state.build_H` gives ``H_i ξ = ξ_p − ξ_{d_i}``.  Therefore for any ``c ∈ ℝ³``

    ξ_p = ξ_{d_1} = … = ξ_{d_N} = c    ⟹    H ξ = 0.

The **common translation** of base and every contact is exactly in ``null(H)``,
for every ``N``.  Gravity leveling is rank 2 with its null direction along
``e_z`` by construction, so nothing in the filter observes that mode.  A contact
process covariance ``Σ_C`` can only reshape ``K = P Hᵀ S⁻¹`` — it modulates *how
much* of an innovation is shared with the base, and can never inject a signed
correction along a direction where the innovation is identically zero.

A re-seed is therefore **not a measurement**.  It is a re-initialisation: the
contact slot is re-anchored to be exactly consistent with the current base
estimate and the current FK, so the touchdown transient (impact, foot roll, FK
and compliance mismatch — all of it systematically signed) is absorbed as an
*anchor definition* instead of being gain-split into ``p̂``.  That severs the
pathway which `PORT_NOTES.md` Finding 2 measured as ~0.09 m/s of vertical drift,
63 % of it accumulating in the 25 % of ticks around a touchdown.

The congruence (`reseed_contacts`)
----------------------------------
Re-seeding contact ``i`` asserts ``d_i = p + R y_i`` with ``y_i`` the body-frame
FK measurement carrying noise ``N_i``.  In the right-invariant tangent that is
the **linear** map

    ξ_{d_i} ← ξ_p + R w_i ,        w_i ~ N(0, N_i),

every other tangent component untouched.  Writing it as ``ξ⁺ = F ξ + G w`` gives
the covariance congruence

    P⁺ = F P Fᵀ + G blkdiag(N_i) Gᵀ

whose two characteristic blocks are the ones the Java test asserts elementwise:

    P_{d_i d_i}⁺ = P_pp + R N_i Rᵀ          (`testReseedCovarianceConsistency`)
    P_{θ d_i}⁺   = P_{θ p}                   (the ``K_θ = 0`` condition)

and from which the **zero-release** property follows immediately: the innovation
``ν_i = R y_i − (d̄_i − p̄)`` is identically zero right after the re-seed, so
``K ν = 0`` and the base is not moved at all
(`testZeroReleaseAfterReseed`).  ``P_{θd} = P_{θp}`` is the stronger, structural
half — it makes the rotation rows of the gain vanish for that contact, so even a
*slightly* inconsistent follow-up measurement cannot rotate the estimate.

Constant graph (I7)
-------------------
Firing is a per-contact **float mask**, never a Python branch and never a shape
change.  ``F`` and ``G`` are assembled from that mask, so ``fire_i = 0`` yields
``F = I``, ``G = 0`` and hence ``P⁺ = P`` and ``d̄⁺ = d̄`` **bit-for-bit** — the
congruence runs unconditionally every tick and is the identity when it must be.
Because ``fire ∈ {0, 1}``, ``fire² = fire`` and the noise block is exact rather
than merely masked.

The latch (`advance_latch`)
---------------------------
`TouchdownReseedLatch`: fire once on a rising crossing of ``trigger``, then stay
disarmed until the contact signal has been continuously below ``rearm`` for
``dwell_ticks``.  The dwell is the whole point — a foot strike commonly produces
a ``p: 1 → 0 → 1`` pulse mid-strike, and a latch without a sustained-low
requirement fires twice on one touchdown (CLAUDE.md §6, "reseed double-fire").
Any tick at or above ``rearm`` resets the counter to zero.
"""
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp

from ..config import section
from .state import BASE_POSITION_TANGENT_INDEX, CONTACT_TANGENT_OFFSET, InEKFState


__all__ = [
    "ReseedParams",
    "LatchState",
    "default_reseed_params",
    "init_latch",
    "advance_latch",
    "reseed_contacts",
    "pre_reseed_residual",
]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class ReseedParams(NamedTuple):
    """Touchdown re-seed configuration (``reseed:`` in ``config/filter_cfg.yaml``).

    Attributes
    ----------
    enabled : bool
        Build-time flag.  **Static**, never traced: a disabled build simply does
        not emit the congruence, so this is not a data-dependent branch.
    trigger : float
        Rising-crossing threshold on the per-contact contact signal that fires an
        armed latch.  Java: 0.5.
    rearm : float
        The contact signal must sit strictly below this to count toward the
        re-arm dwell.  Java: 0.1.
    dwell_ticks : int
        Consecutive below-``rearm`` ticks required to re-arm.  Java: 100.
    """
    enabled: bool
    trigger: float
    rearm: float
    dwell_ticks: int


def default_reseed_params(
    enabled: bool | None = None,
    trigger: float | None = None,
    rearm: float | None = None,
    dwell_ticks: int | None = None,
) -> ReseedParams:
    """Build `ReseedParams` from the ``reseed`` config section, with overrides."""
    cfg = section("reseed")
    params = ReseedParams(
        enabled=bool(cfg["enabled"] if enabled is None else enabled),
        trigger=float(cfg["latch_trigger"] if trigger is None else trigger),
        rearm=float(cfg["latch_rearm"] if rearm is None else rearm),
        dwell_ticks=int(cfg["latch_dwell_ticks"] if dwell_ticks is None else dwell_ticks),
    )
    if not params.rearm < params.trigger:
        raise ValueError(
            f"reseed latch needs rearm < trigger (hysteresis), got "
            f"rearm={params.rearm}, trigger={params.trigger}"
        )
    if params.dwell_ticks < 1:
        raise ValueError(f"dwell_ticks must be >= 1, got {params.dwell_ticks}")
    return params


# ---------------------------------------------------------------------------
# The latch
# ---------------------------------------------------------------------------

class LatchState(NamedTuple):
    """Per-contact latch carry — float arrays so it lives in a `lax.scan` carry.

    Attributes
    ----------
    armed : Array, shape (N,)
        1.0 if this contact may fire on the next rising crossing, else 0.0.
    low_count : Array, shape (N,)
        Consecutive ticks the contact signal has been below ``rearm``.
    """
    armed: Array
    low_count: Array


def init_latch(N: int) -> LatchState:
    """Fresh latch: every contact **armed**, no low-dwell accumulated.

    Armed-at-init is deliberate — the first touchdown after boot is exactly the
    one worth re-seeding, since the contact slots were seeded from a prior rather
    than from a measured foothold.
    """
    return LatchState(
        armed=jnp.ones(N, dtype=jnp.float64),
        low_count=jnp.zeros(N, dtype=jnp.float64),
    )


def advance_latch(
    latch: LatchState, contact_prob: Array, params: ReseedParams
) -> tuple[LatchState, Array]:
    r"""Advance the fire-once latch one tick.

    Parameters
    ----------
    latch : LatchState
        Previous carry.
    contact_prob : Array, shape (N,)
        Per-contact contact signal in [0, 1] (the Schmitt-trusted mask in sim, a
        contact probability on hardware).
    params : ReseedParams

    Returns
    -------
    latch : LatchState
        Advanced carry.
    fire : Array, shape (N,)
        Float mask, 1.0 on the contacts re-seeding this tick.

    Notes
    -----
    Branch-free.  Order within the tick: accumulate the low dwell, re-arm on a
    completed dwell, then fire on a rising crossing of ``trigger``.  Because
    ``rearm < trigger`` is enforced at construction, "below rearm" and "above
    trigger" are mutually exclusive, so the ordering of the first two steps
    against the third cannot change the result.
    """
    p = jnp.asarray(contact_prob, dtype=jnp.float64)

    low = p < params.rearm
    low_count = jnp.where(low, latch.low_count + 1.0, 0.0)

    # Re-arm on a COMPLETED dwell. `>=` rather than `==` so a latch that is
    # already armed and stays low simply remains armed.
    rearmed = (low_count >= params.dwell_ticks).astype(jnp.float64)
    armed = jnp.maximum(latch.armed, rearmed)

    fire = armed * (p > params.trigger).astype(jnp.float64)

    # Fire-once: firing disarms until the dwell is served again.
    armed = armed * (1.0 - fire)

    return LatchState(armed=armed, low_count=low_count), fire


# ---------------------------------------------------------------------------
# The congruence
# ---------------------------------------------------------------------------

def pre_reseed_residual(state: InEKFState, y: Array) -> Array:
    r"""Per-contact world-frame residual norm ``‖R̄ y_i − (d̄_i − p̄)‖``, shape ``(N,)``.

    The quantity Java's `reseedContact` returns.  Diagnostic only — it is the
    geometric discrepancy the re-seed is about to absorb, and
    `testZeroReleaseAfterReseed` asserts it is genuinely nonzero before the
    re-seed and that the immediately-following identical measurement is not.
    """
    rel = state.d - state.p                       # (N, 3) d̄_i − p̄, world
    return jnp.linalg.norm(y @ state.R.T - rel, axis=-1)


def _selector(N: int) -> Array:
    """Constant ``(m, m)`` map sending ``ξ_p`` into every contact's tangent block.

    ``S[9+3i+a, 6+a] = 1``; all other entries zero.  Static in ``N``, so XLA
    constant-folds it out of the traced graph.
    """
    m = 3 * N + 9
    S = jnp.zeros((m, m), dtype=jnp.float64)
    rows = CONTACT_TANGENT_OFFSET + 3 * jnp.arange(N)[:, None] + jnp.arange(3)[None, :]
    cols = jnp.broadcast_to(BASE_POSITION_TANGENT_INDEX + jnp.arange(3)[None, :], (N, 3))
    return S.at[rows.reshape(-1), cols.reshape(-1)].set(1.0)


def _noise_injector(R: Array, N: int) -> Array:
    """Constant-structure ``(m, 3N)`` map ``w ↦ R w`` into each contact block."""
    m = 3 * N + 9
    G = jnp.zeros((m, 3 * N), dtype=jnp.float64)
    return G.at[CONTACT_TANGENT_OFFSET:, :].set(jnp.kron(jnp.eye(N), R))


def reseed_contacts(
    state: InEKFState, y: Array, Np: Array, fire: Array
) -> InEKFState:
    r"""Re-anchor the fired contacts; identity on the rest (§ module docstring).

    Mean::

        d̄_i ⁺ = (1 − f_i) d̄_i + f_i (p̄ + R̄ y_i)

    Covariance, with ``f_r`` the fire mask broadcast over each contact's three
    tangent rows, ``S`` the constant ``ξ_p``-selector and ``G = f_r ⊙ (I_N ⊗ R̄)``::

        F  = (1 − f_r) ⊙ I + f_r ⊙ S
        P⁺ = F P Fᵀ + G blkdiag(N_i) Gᵀ

    ``f_i = 0`` ⟹ ``F`` is the identity row and ``G`` the zero row, so an unfired
    contact leaves both mean and covariance **bit-for-bit** unchanged.

    Parameters
    ----------
    state : InEKFState
        Prior state (post-propagation, pre-update).
    y : Array, shape (N, 3)
        Body-frame FK measurements ``h_{p,i}(q̂)`` — the same array the contact
        update consumes.
    Np : Array, shape (N, 3, 3)
        Per-contact **body-frame** FK covariances ``N_i = J_{C_i} Σ_q J_{C_i}ᵀ``.
        Body-frame: ``G`` rotates them to world, which is why the asserted block
        is ``P_pp + R N Rᵀ`` and not ``P_pp + N``.
    fire : Array, shape (N,)
        Float mask from `advance_latch`, values in ``{0, 1}``.

    Returns
    -------
    InEKFState
        State with re-anchored contact means and the congruent covariance.
    """
    N = state.N
    R = state.R
    f = jnp.asarray(fire, dtype=jnp.float64)

    # -- mean -------------------------------------------------------------
    anchored = state.p[None, :] + y @ R.T            # (N, 3) p̄ + R̄ y_i
    d_next = (1.0 - f)[:, None] * state.d + f[:, None] * anchored

    # -- covariance congruence -------------------------------------------
    # Fire mask broadcast onto tangent ROWS: zero on [R, v, p], f_i on d_i.
    f_rows = jnp.concatenate([jnp.zeros(9, dtype=jnp.float64), jnp.repeat(f, 3)])

    m = 3 * N + 9
    F = (1.0 - f_rows)[:, None] * jnp.eye(m, dtype=jnp.float64) \
        + f_rows[:, None] * _selector(N)
    G = f_rows[:, None] * _noise_injector(R, N)

    Nblk = _block_diag(Np)
    P_next = F @ state.P @ F.T + G @ Nblk @ G.T
    P_next = 0.5 * (P_next + P_next.T)               # numerical hygiene (I7)

    return state._replace(d=d_next, P=P_next)


def _block_diag(blocks: Array) -> Array:
    """Block-diagonal ``(3N, 3N)`` from an ``(N, 3, 3)`` stack (vectorised).

    Same construction as `correct._block_diag` / `propagate._block_diag_from_stack`;
    kept local so `reseed.py` adds no import edges between filter modules.
    """
    n = blocks.shape[0]
    selector = jnp.einsum("ij,ikl->ijkl", jnp.eye(n), blocks)
    return selector.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
