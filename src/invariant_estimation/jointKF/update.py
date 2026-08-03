r"""
jointKF/update.py
=================
The **one** measurement-update path of the joint-space KF: Joseph-form covariance
update with a masked gain (Java `josephUpdate`, gate G6).  Every measurement the
filter takes — encoder rows, the stacked ``L Σ Lᵀ`` gyro block, the stance
anchors, the optional direct-velocity channel — goes through this function, so
the gating and conditioning semantics cannot drift between channels.

Joseph, not ``(I − KH)P``
-------------------------
The short form is algebraically equal to the Joseph form **only at the exactly
optimal gain**.  Two things here guarantee the gain is not exactly optimal: the
gyro rows carry a Jacobian ``J_ang(q̂)`` linearised at an estimate, and the gate
below deliberately applies ``K = 0`` (a legal but suboptimal gain) whenever ``S``
is ill-conditioned.  The Joseph form

    P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ

is the honest ``L Σ Lᵀ`` pushforward of the posterior error
``e⁺ = (I − KH) e⁻ − K v`` over *both* error sources, so it is correct for
whatever ``K`` was actually applied and is a sum of two congruences — structurally
PSD.  The short form under a suboptimal gain is neither.

Conditioning gate — skip, never latch
-------------------------------------
``cond(S)`` is estimated from the Cholesky factor's diagonal as
``(max L_ii / min L_ii)²``: an eigendecomposition would cost more than the update
and is not differentiable-friendly, whereas the factor is already computed for the
gain solve.  Above `cond_s_max` the entire update is dropped by zeroing ``K``.

This matters more than it looks.  A finite but ill-conditioned ``S`` inverts to a
huge gain, and the ``K R Kᵀ`` term squares it every tick — that is the covariance
divergence mechanism a plain `isfinite` guard is blind to.  Dropping the update
loses information for one tick; taking it poisons ``P`` permanently.

The gate is a **float mask**, never a Python branch (I7): the graph is identical
whether or not the update is taken, which is what lets the whole step live inside
one jitted `lax.scan`.  A gated update must leave ``(x, P)`` **bit-identical**
(`testSingularInnovationIsSkippedNotLatched`, tolerance exactly 0.0), so the
result is selected with `jnp.where` on the *whole* carry rather than merely
zeroing the gain: with `K = 0`, ``(I−KH) P (I−KH)ᵀ`` re-derives ``P`` through two
matrix products and a symmetrisation, and neither is obliged to return the input
bit-for-bit if ``P`` is not already exactly symmetric.

NaN hardening
-------------
A single non-finite sensor sample must be *skipped*, not propagated, and must not
latch — `testTransientNonFiniteInputRecovers` restores clean input and demands the
filter track again with no intervention.  So ``H, z, R`` are checked for
finiteness and **sanitised before use**, not merely multiplied by a zero gate:
``0.0 * NaN = NaN``, and a NaN reaching ``P`` is permanent.  The sanitised
``R → I`` (never ``0``) keeps ``S`` non-singular so the Cholesky itself stays
finite; the gate independently records that nothing was applied.

NIS is computed on the **prior** ``P`` and the **prior** residual
(CLAUDE.md §6): ``ν ᵀ S⁻¹ ν`` with ``S = H P⁻ Hᵀ + R``.  Computing it after the
update passes every easy test and fails the quadratic-form oracle.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_factor, cho_solve
from jax.typing import ArrayLike

from .state import JointKFParams, JointKFState


def joseph_covariance(P: ArrayLike, K: ArrayLike, H: ArrayLike, R: ArrayLike) -> Array:
    r"""``P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ``, symmetrised.

    Exposed as its own function because it is the only part of the update that is
    correct for an **arbitrary** gain, and therefore the only part a test can
    constrain independently: at the optimal ``K`` the short form ``(I − KH) P`` is
    algebraically identical, so no test driven through `joseph_update` can tell
    the two apart.  Feed this a deliberately suboptimal ``K`` and they diverge —
    which is exactly the regime the filter enters whenever ``H`` is a linearised
    Jacobian or the conditioning gate has zeroed the gain.

    Parameters
    ----------
    P : Array, shape (dim, dim)
        Prior covariance.
    K : Array, shape (dim, k)
        The gain **actually applied** — not necessarily the optimal one.
    H : Array, shape (k, dim)
    R : Array, shape (k, k)

    Returns
    -------
    Array, shape (dim, dim)
    """
    P, K, H, R = (jnp.asarray(a) for a in (P, K, H, R))
    IKH = jnp.eye(P.shape[0], dtype=P.dtype) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    return 0.5 * (P_new + P_new.T)


class UpdateInfo(NamedTuple):
    r"""Per-update diagnostics — part of the seam surface, not optional logging.

    CLAUDE.md §4: the Java filter publishes these as YoVariables and the ported
    tests read them, so they are returned as a pytree of arrays.

    Attributes
    ----------
    nu : Array, shape (k,)
        Prior innovation ``z − H x⁻`` actually used.  When the finiteness gate
        fires this is the sanitised (zero) residual; `was_applied` is the flag to
        read, not `nu`.
    S : Array, shape (k, k)
        Symmetrised innovation covariance ``H P⁻ Hᵀ + R``.
    nis : Array, scalar
        Normalised innovation squared ``νᵀ S⁻¹ ν`` on the **prior** ``P`` and the
        **prior** residual.  ``NaN`` whenever the update was gated out — a skipped
        update has no meaningful consistency statistic, and NaN cannot be
        mistaken for "in band".
    condition_proxy : Array, scalar
        ``(max L_ii / min L_ii)²`` from the Cholesky factor of ``S``.  ``inf`` or
        ``NaN`` for an exactly singular ``S`` (either gates).
    was_applied : Array, scalar float
        ``1.0`` if the gain was applied, ``0.0`` if the update was skipped.
        Java `wasLastUpdateApplied`.
    nis_per_row : Array, shape (k,)
        Per-row ``ν_i² / S_ii``.  ``NaN`` when gated, same convention as `nis`.

        The aggregate `nis` cannot localise a fault: the whole point of the Java
        per-joint `jointKF_encNIS_<joint>` diagnostic is to say *which* encoder
        went bad, and a single scalar over all rows cannot.  For a diagonal
        channel (encoders, direct velocity) each row is an independent
        ``chi²₁``, which is exactly what the NIS-consistency tests assert.

        Note this is the per-row *marginal* statistic, not a decomposition of
        `nis` — they agree only when ``S`` is diagonal.  For the correlated
        stacked gyro measurement the marginals are still individually
        interpretable, but they do not sum to `nis`.
    """

    nu: Array
    S: Array
    nis: Array
    condition_proxy: Array
    was_applied: Array
    nis_per_row: Array


def joseph_update(
    state: JointKFState,
    H: ArrayLike,
    z: ArrayLike,
    R: ArrayLike,
    params: JointKFParams,
    label: str | None = None,
) -> tuple[JointKFState, UpdateInfo]:
    r"""Gated Joseph-form measurement update — Java `josephUpdate(H, z, R)`.

    ::

        ν  = z − H x⁻                          (prior residual)
        S  = H P⁻ Hᵀ + R                       (symmetrised)
        K  = P⁻ Hᵀ S⁻¹                         (Cholesky solve — never `inv`)
        x⁺ = x⁻ + K ν
        P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ    (Joseph — mandatory)

    all multiplied by the float gate ``was_applied = finite(H, z, R) ∧
    cond(S) < cond_s_max``.  A gated update returns ``(x, P)`` bit-identically.

    Parameters
    ----------
    state : JointKFState
        Prior (post-predict) carry.
    H : Array, shape (k, dim)
        Measurement Jacobian.  Fixed-shape: inactive rows are masked by giving
        them ``R_LARGE`` (`JointKFParams.r_large`), never by dropping them (I7).
    z : Array, shape (k,)
        Measurement.
    R : Array, shape (k, k)
        Measurement covariance.
    params : JointKFParams
        Supplies ``cond_s_max``.
    label : str, optional
        Static tag for host-side diagnostic attribution (Java passes a label into
        `describeSingularInnovation`).  Not returned in `UpdateInfo`: strings
        cannot cross a jit boundary, and the constant-graph rule (I7) forbids
        branching on it.

    Returns
    -------
    state : JointKFState
        Posterior carry, or the prior bit-for-bit if gated.
    info : UpdateInfo
    """
    del label  # host-side attribution only; see the docstring.

    # numpy in, jax out: the ported oracles build `H`/`z`/`R` with numpy on purpose (an
    # independent path from the filter's own jax algebra is what makes them oracles).
    # `asarray` without a dtype PRESERVES dtype, so this cannot launder a float32 leak
    # into float64 -- I8 still bites where it should.
    H, z, R = jnp.asarray(H), jnp.asarray(z), jnp.asarray(R)

    x, P = state.x, state.P
    k = R.shape[0]

    # -- finiteness sanitisation (BEFORE any arithmetic: 0.0 * NaN = NaN) -----
    finite = (
        jnp.all(jnp.isfinite(H)) & jnp.all(jnp.isfinite(z)) & jnp.all(jnp.isfinite(R))
    )
    Hs = jnp.where(finite, H, jnp.zeros_like(H))
    zs = jnp.where(finite, z, jnp.zeros_like(z))
    # R -> I, not 0: a zero R on a zero H would make S exactly singular and the
    # Cholesky NaN, re-poisoning the very quantities this branch is sanitising.
    Rs = jnp.where(finite, R, jnp.eye(k, dtype=R.dtype))

    # -- innovation and its covariance ---------------------------------------
    PHt = P @ Hs.T                                   # (dim, k)
    S = Hs @ PHt + Rs
    S = 0.5 * (S + S.T)
    factor = cho_factor(S)

    # -- condition proxy, over INFORMATIVE rows only -------------------------
    # A row deliberately masked to `R_LARGE` (an inactive stance anchor,
    # CLAUDE.md §4) is structurally decoupled: its `H` row is zero, so `S` is
    # block-diagonal there and its Cholesky diagonal is exactly `sqrt(R_LARGE)`.
    # Counting it would make `cond(S) ~ R_LARGE / lambda_min(pair block) ~ 4e11`,
    # far above `cond_s_max = 1e9` — so the gate would drop the ENTIRE stacked
    # update, gyro rows included, on every tick any foot is in swing. That is to
    # say: the filter would stop updating for the whole of walking.
    #
    # Java never meets this because its stacked measurement literally has no
    # anchor rows when no foot is trusted; the fixed-shape port has to say the
    # same thing with a mask, and `R_LARGE` and `cond_s_max` are otherwise
    # mutually destructive as configured.
    #
    # Excluding them is not a fudge, it is the gate's own semantics: the gate
    # exists to catch an `S` that inverts to a HUGE gain, and a row we have
    # declared uninformative contributes gain ~1/R_LARGE ~ 0. It is the safest
    # row in the matrix, not the most dangerous. Deriving the mask from `R`
    # rather than an extra argument keeps this true for any caller that follows
    # the masking rule, with no plumbing to forget.
    diag = jnp.abs(jnp.diag(factor[0]))
    informative = jnp.diag(Rs) < 0.5 * params.r_large
    any_informative = jnp.any(informative)
    d_max = jnp.max(jnp.where(informative, diag, -jnp.inf))
    d_min = jnp.min(jnp.where(informative, diag, jnp.inf))
    condition_proxy = jnp.where(any_informative, (d_max / d_min) ** 2, 1.0)

    # K = P Hᵀ S⁻¹  ⟺  S Kᵀ = (P Hᵀ)ᵀ   (S symmetric)
    K = cho_solve(factor, PHt.T).T                   # (dim, k)
    nu = zs - Hs @ x

    # A singular S makes the factor non-finite, so `condition_proxy` is NaN or
    # inf and BOTH comparisons below are False — the gate closes either way.
    applied = finite & (condition_proxy < params.cond_s_max)
    was_applied = applied.astype(jnp.float64)

    # Zero the gain rather than multiplying by the mask: a non-finite K (singular
    # S) times 0.0 is still NaN.
    K = jnp.where(applied, K, jnp.zeros_like(K))
    nis = jnp.where(applied, nu @ cho_solve(factor, nu), jnp.nan)
    nis_per_row = jnp.where(applied, nu ** 2 / jnp.diag(S), jnp.nan)

    x_new = x + K @ nu
    P_new = joseph_covariance(P, K, Hs, Rs)

    # Bit-identity when gated: `K = 0` makes the algebra an identity, but not
    # necessarily the floating-point evaluation of it.
    posterior = JointKFState(
        x=jnp.where(applied, x_new, x),
        P=jnp.where(applied, P_new, P),
    )
    return posterior, UpdateInfo(
        nu=nu,
        S=S,
        nis=nis,
        condition_proxy=condition_proxy,
        was_applied=was_applied,
        nis_per_row=nis_per_row,
    )
