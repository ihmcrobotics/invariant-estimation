r"""Touchdown re-seed — `reseedContact` plus the `TouchdownReseedLatch` that decides
when it fires.

A contact slot is an *anchor*: the filter's belief about where that foot is planted
in the world.  Over a stance the anchor stays fixed and the base state is propagated
relative to it, so any error the anchor accumulated — slip, sole compliance, the FK
it was seeded with — is carried forward into the next stance instead of being dropped
when the foot lifts and lands again.  `reseed_contacts` re-anchors the slot at
touchdown.

**It does not make anything observable.**  Global ``x``, ``y`` and yaw are
unobservable in a proprioceptive InEKF and stay unobservable after a re-seed —
`p0`'s ``diag(R) = [7.1e-5, 7.1e-5, 1.0]`` is the filter correctly saying so.  What a
re-seed changes is the *rate* at which the unobservable directions drift, by refusing
to propagate a stale anchor across a flight phase.  Report it as drift, never as
observability.

Re-seeding contact ``i`` to the foothold implied by body-frame measurement ``y`` sets

.. math::
    d_i \leftarrow \hat p + \hat R y

and transforms the covariance so the new anchor inherits the base position's
uncertainty plus the FK noise of the measurement that placed it:

.. math::
    P_{d_i d_i} = P_{pp} + \hat R N \hat R^\top, \qquad
    P_{\theta d_i} = P_{\theta p}

Both are exact, and both matter.  The first says a freshly-anchored foot is known
exactly as well as the base is, plus however well FK locates it — nothing better.
The second is the ``K_\theta = 0`` condition, and it is what makes the *zero release*
property hold: re-applying the very same measurement immediately after a re-seed
produces zero residual, hence zero NIS and **zero rotation correction**.  Without
``P_{\theta d} = P_{\theta p}`` the re-seed would inject a spurious attitude
correction at every touchdown, which is precisely the failure mode a re-seed is
supposed to remove.

Implemented as a genuine congruence
``P \leftarrow T P T^\top + G(\hat R N \hat R^\top)G^\top`` rather than by writing
blocks into ``P``, so PSD is preserved structurally for *any* PSD input and any blend
weight, not just at the endpoints
(`InvariantEKFReseedTest.testReseedPreservesPositiveSemiDefiniteness`).

**Why the latch.**  Contact probability is not clean.  On hardware log
``20260717_112516`` the mid-strike signal pulses ``1 -> 0 -> 1`` inside a single foot
strike, and a re-seed on each rising edge would re-anchor twice in one stance — the
second one against a foot that has already loaded and deformed.
`TouchdownReseedLatch` fires **at most once per genuine swing**: it arms only after
``dwell_ticks`` consecutive ticks below ``rearm``, fires on the first tick at or above
``trigger``, and disarms on firing.  A dip shorter than the dwell cannot re-arm it, so
the double pulse cannot double-fire.

Constant-graph form (I7): no Python branches and no data-dependent shapes.  The latch
is two float carries per contact advanced with `jnp.where`; the congruence runs
**every tick** and is blended by the fire mask, with ``fire = 0`` giving exactly the
identity congruence and an exactly unchanged state — asserted, not assumed
(`test_a_tick_that_does_not_fire_changes_nothing_at_all`).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .state import BASE_POSITION_TANGENT_INDEX, InEKFState, contact_tangent_index

__all__ = [
    "ReseedParams",
    "LatchState",
    "default_reseed_params",
    "init_latch",
    "advance_latch",
    "reseed_contacts",
    "reseed_step",
]


class ReseedParams(NamedTuple):
    """`TouchdownReseedLatch` configuration. Build via `default_reseed_params`.

    ``trigger`` (0.5) is the contact probability at or above which an **armed** latch
    fires; ``rearm`` (0.1) the probability below which the dwell counter accumulates;
    ``dwell_ticks`` (100 = 100 ms at 1 kHz) the consecutive sub-`rearm` ticks needed to
    arm.
    """
    trigger: float
    rearm: float
    dwell_ticks: int


def default_reseed_params(trigger: float = 0.5, rearm: float = 0.1,
                          dwell_ticks: int = 100) -> ReseedParams:
    """Validated constructor — Java's constructor guard, kept as a build-time check.

    Both rejections are behavioural, not stylistic. ``trigger == rearm`` leaves no
    hysteresis band, so the latch arms and fires on the same signal and chatters at
    the threshold; ``dwell_ticks == 0`` arms on any single sub-`rearm` tick, which is
    exactly the mid-strike dropout the latch exists to reject.
    """
    if not trigger > rearm:
        raise ValueError(f"reseed: trigger ({trigger}) must exceed rearm ({rearm}); "
                         "equal thresholds leave no hysteresis band")
    if int(dwell_ticks) < 1:
        raise ValueError(f"reseed: dwell_ticks ({dwell_ticks}) must be >= 1; a zero dwell "
                         "arms on any single low tick, which is the mid-strike dropout "
                         "the latch exists to reject")
    return ReseedParams(float(trigger), float(rearm), int(dwell_ticks))


class LatchState(NamedTuple):
    """Per-contact latch carry, as floats so it rides in a `lax.scan` carry (I7).

    ``armed`` ``(N,)`` is 1.0 once `dwell_ticks` consecutive sub-`rearm` ticks have
    been seen, back to 0.0 on the tick it fires.  ``low_count`` ``(N,)`` counts
    consecutive sub-`rearm` ticks and is reset to 0 by **any** tick at or above
    `rearm` — that reset is what makes single-tick dips unable to re-arm.
    """
    armed: Array
    low_count: Array


def init_latch(n_contacts: int) -> LatchState:
    """Disarmed, counters at zero — Java's ``initialArmedFlag = false``.

    Disarmed, so the very first touchdown after initialisation does **not** re-seed.
    That is the right default: at ``t = 0`` the anchors were just placed by
    `init_state` from the same FK a re-seed would use, so firing would be a no-op at
    best, and the filter has no swing history to justify it.
    """
    z = jnp.zeros((int(n_contacts),), dtype=jnp.float64)
    return LatchState(armed=z, low_count=z)


def advance_latch(params: ReseedParams, latch: LatchState,
                  p: Array) -> tuple[LatchState, Array]:
    """One tick on the ``(N,)`` contact probability ``p`` → ``(latch, fire)``, both elementwise.

    ``fire`` is 1.0 on contacts re-seeding this tick.  The state machine, in order:

    1. ``low_count = (low_count + 1) * [p < rearm]`` — accumulate or reset.
    2. ``armed |= low_count >= dwell_ticks``.
    3. ``fire = armed AND p >= trigger``.
    4. ``armed &= NOT fire``.

    No explicit rising-edge test is needed and none is used: arming requires
    ``dwell_ticks`` consecutive ticks below ``rearm``, so the tick that arms always has
    ``p < rearm < trigger``. Any later fire is therefore a rising crossing by
    construction. Adding a stored ``p_prev`` would be state that can only ever agree
    with this.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    low = (p < params.rearm).astype(jnp.float64)
    low_count = (latch.low_count + 1.0) * low
    armed = jnp.maximum(latch.armed, (low_count >= params.dwell_ticks).astype(jnp.float64))
    fire = armed * (p >= params.trigger).astype(jnp.float64)
    return LatchState(armed=armed * (1.0 - fire), low_count=low_count), fire


def reseed_contacts(state: InEKFState, y: Array, fk_cov: Array,
                    fire: Array) -> tuple[InEKFState, Array]:
    r"""Re-anchor every contact whose `fire` is 1, as one masked congruence.

    All ``N`` contacts are handled in a single ``T P Tᵀ`` because the per-contact
    transforms **commute**: ``T_i`` rewrites only the ``d_i`` row/column block and reads
    only the ``p`` block, which no ``T_j`` ever writes. Composing them is therefore the
    same as applying them in any order, and doing it once avoids ``N`` sequential
    ``(m, m)`` products.

    ``y`` ``(N, 3)`` is the body-frame FK measurement per contact — the same ``y`` the
    contact update consumes, so a re-seeded anchor is exactly consistent with the
    measurement that placed it (this is what the zero-release property rests on).
    ``fk_cov`` ``(N, 3, 3)`` is that measurement's body-frame FK covariance ``N_i``,
    rotated to world here; it is the *measurement* noise of the re-seed, not the
    process ``Σ_C``.  ``fire`` ``(N,)`` is 1.0 to re-seed, 0.0 to leave untouched —
    intermediate values are a valid congruence too (PSD is preserved for any weight),
    but the latch only ever emits 0 or 1.

    Returns the new state and ``pre_residual`` ``(N, 3)``: the **pre-re-seed** world
    residual ``R̂ y − (d̂_i − p̂)`` per contact — how far the old anchor had drifted from
    where FK now says the foot is.  This is the diagnostic that says whether
    re-seeding did anything; its norm is what Java's `reseedContact` returns.
    """
    N, m = state.N, state.P.shape[0]
    f = jnp.asarray(fire, dtype=jnp.float64).reshape(N, 1)
    pi = BASE_POSITION_TANGENT_INDEX

    d_new = state.p[None, :] + jnp.einsum("ab,nb->na", state.R, y)
    pre_residual = d_new - state.d
    d = state.d + f * pre_residual

    # T is the identity except on each fired contact's row block, which is blended
    # from "keep my own error" toward "my error IS the base position's error".
    T = jnp.eye(m, dtype=jnp.float64)
    I3 = jnp.eye(3, dtype=jnp.float64)
    for i in range(N):                       # N is static; this unrolls at trace time
        j = contact_tangent_index(i)
        fi = f[i, 0]
        T = T.at[j:j + 3, j:j + 3].set((1.0 - fi) * I3)
        T = T.at[j:j + 3, pi:pi + 3].set(fi * I3)
    P = T @ state.P @ T.T

    # The FK noise of the placing measurement, world frame, weighted by the same
    # mask. A non-negative multiple of a PSD matrix is PSD, so this cannot break
    # the PSD guarantee the congruence provides.
    Nw = jnp.einsum("ab,nbc,dc->nad", state.R, fk_cov, state.R)
    for i in range(N):
        j = contact_tangent_index(i)
        P = P.at[j:j + 3, j:j + 3].add(f[i, 0] * Nw[i])

    # Symmetrise: the congruence is symmetric in exact arithmetic, and the Java
    # suite asserts symmetry to 1e-10, so the float asymmetry is folded out here
    # rather than left for a downstream Cholesky to trip on.
    P = 0.5 * (P + P.T)
    return state._replace(d=d, P=P), pre_residual


def reseed_step(params: ReseedParams, latch: LatchState, state: InEKFState,
                p: Array, y: Array, fk_cov: Array
                ) -> tuple[LatchState, InEKFState, Array, Array]:
    """`advance_latch` then `reseed_contacts` → ``(latch, state, fire, pre_residual)``.

    Runs unconditionally: on a tick where nothing fires the congruence is the identity
    and `state` comes back bit-identical, which is what keeps the traced graph constant
    (I7).
    """
    latch, fire = advance_latch(params, latch, p)
    state, pre_residual = reseed_contacts(state, y, fk_cov, fire)
    return latch, state, fire, pre_residual


def expand_per_foot(prob: Array, n_contacts: int) -> Array:
    """Per-**foot** trust `(K,)` -> per-**contact** `(N,)`, foot-major.

    At ``N == K`` this is the identity. With toe/heel (``N == 2K``) the contact
    ordering is ``(left heel, left toe, right heel, right toe)``
    (`main_estimator.TOE_HEEL_SITES`), so contact ``2f + s`` belongs to foot ``f`` and
    both points of a foot share that foot's trust — they touch down together as far as
    the trust signal can tell, since it is a per-foot force switch.
    """
    K = prob.shape[-1]
    if n_contacts == K:
        return prob
    if n_contacts % K:
        raise ValueError(f"cannot expand {K} feet to {n_contacts} contacts")
    return jnp.repeat(prob, n_contacts // K, axis=-1)
