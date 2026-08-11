r"""Contact zero-velocity constraint — the measurement block that observes base velocity.

Motivation, in one paragraph. The existing contact-FK block has ``H^p = [0|0|+I|-I]``:
**no columns for velocity**. Base velocity is corrected only through ``P``'s
cross-covariance, and the influence map attributes 88.7% of the vertical sink to that
path (severing it is worse, not better: masking the rotation/velocity rows of ``K``
takes height error -2.07 -> -6.53 m). Σ_C can scale that correction but not redirect
it, which is why four separate Σ_C-shaped arms have now hit the same ceiling. This
module changes ``H`` instead.

The kinematic identity, from ``d_i = p + R h_i(q)``::

    ḋ_i = v + R([ω]× h_i + J_i q̇),        J_i ≡ ∂h_i/∂q

is a *fact*, not a model. A world-static contact has ``ḋ_i = 0``, giving

    y^v_i ≡ -([ω]× h_i + J_i q̇) = Rᵀ v          (the constraint)

Every term on the right is measured: ``ω`` bias-corrected (I1), ``h_i`` and ``J_i``
from FK on filtered encoders, ``q̇`` from the joint KF. Physically the stance foot is a
temporary world-fixed reference and the leg is a ruler measuring how fast the base
recedes from it — leg odometry.

**This does not compete with the Σ_C process model; it observes it.** Setting the
identity equal to the process model ``ḋ_i = R w_C`` and rearranging gives
``y^v_i = Rᵀ v - w_C``: the pseudo-measurement's noise IS the contact process noise,
transported. They are the same random variable. The consequence is a hard requirement
rather than an option — ``N^v`` must be built *from* Σ_C (see `velocity_noise`), or the
same "this foot is planted" evidence is used twice and the filter goes overconfident.

**World frame, never body.** Comparing in the body frame gives ν = y^v - R̂ᵀv̂ whose
Jacobian carries ``R̂ᵀ`` — state-dependent, breaking I3 and I7. Rotating the
measurement instead, ``ν^v = R̂ y^v - v̂``, makes the attitude error enter both terms
identically so it cancels, leaving ``ν^v ≈ -ξ_v`` and a constant ``H^v``.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .state import InEKFState

# ===========================================================================
# *** J̇ = 0 IS A LARGE, UNJUSTIFIED ASSUMPTION AND A DEVIATION FROM THE
# *** DERIVATION. READ THIS BEFORE TRUSTING ANY NOISE NUMBER FROM THIS MODULE.
# ===========================================================================
#
# `main_estimator._make_contact_kinematics` returns `J_dot = jnp.zeros_like(J)`
# with `q_dot` accepted and unused. That is a PORT STUB, not a modelling choice:
# J̇ vanishes only if h(q) were linear in q -- forward kinematics is trigonometric
# in joint angles -- or if q̇ = 0.
#
# MEASURED on the real model (9 filtered + 4 aux joints, 8 contacts, a plausible
# walking pose, q̇ ~ 2 rad/s):
#
#     ||J||        = 1.342  m/rad
#     ||J_dot||    = 3.513  m/(rad s)      <- stubbed to zero
#     ||[w] J||    = 0.497  m/(rad s)
#     ratio        = 7.06                  <- J_dot DOMINATES
#
# `contact-zero-velocity.pdf` carries an encoder term J̇ Σ_q J̇ᵀ and no [w]J term;
# an earlier note in this module claimed the write-up had "omitted" the coupling.
# Both are halves of ONE sensitivity, and neither statement was complete:
#
#     ∂y^v/∂q = -( [w]x J + J̇ )
#
# (the δJ q̇ term equals J̇ δq exactly, by symmetry of ∂²h/∂q² in its two q-slots).
# The write-up has the larger half; the earlier note here had the smaller one.
#
# WHY IT IS TOLERATED FOR NOW, AND EXACTLY HOW FAR THAT GOES. The whole encoder
# channel is negligible against foot roll at this Σ_q = 3.3e-8 rad²:
#
#     encoder, J̇ only      std 6.4e-4 m/s
#     encoder, [w]J only    std 9.0e-5 m/s
#     encoder, both         std 6.4e-4 m/s
#     ROLL TERM (measured)  std 7e-2 .. 2e-1 m/s     <- 100-300x larger
#
# So it does not change whether the zero-velocity constraint helps: that is decided
# by the roll model. It WOULD become load-bearing if Σ_q ever grows (e.g. the
# contact_meas_var floor is folded into it -- at 1e-3 rad² this term reaches
# 0.11 m/s, i.e. the same size as roll), or if J̇ is used for anything else.
#
# FIXING IT IS THREE LINES and stays constant-graph:
#     _, J_dot = jax.jvp(lambda x: _foot_y_jacobian(x), (q,), (q_dot,))
# one forward-mode pass, about the cost of J itself.
#
# NOTE FOR WHOEVER IMPLEMENTS IT: `test_measurement_actually_holds_the_contact_
# world_static` uses a synthetic FK h = h0 + A q, which has J̇ = 0 BY CONSTRUCTION
# and is therefore blind to this term. It needs a nonlinear h to have an opinion.
J_DOT_IS_ZERO = True
"""Whether `ContactFrames.J_dot` is the zero stub. See the block comment above.
Kept as a named constant so the assumption is greppable and so a future fix has an
obvious thing to flip and a test to hang off."""


def velocity_jacobian(n_contacts: int) -> Array:
    r"""The zero-velocity block ``H^v = [0 | -I₃ | 0 | 0]``, shape ``(3, 3N+9)``.

    Constant and state-independent, like `correct.contact_jacobian` — the argument is a
    *count*, not a state. Two properties are asserted rather than assumed
    (`tests/inEKF/test_zero_velocity.py`):

    * **zero columns in the rotation block**, so this constraint cannot manufacture yaw.
      Yaw is unobservable by design (``enableYawSeeding=false``) and the gravity update
      carries the identical requirement (rank 2, null along e_z).
    * **zero columns in the position and contact blocks**, so it adds no absolute
      position information either. Stacked with ``H^p`` the rank goes 24 -> 27 of 33 at
      N=8; common-mode translation stays in the null space. This constraint *starves*
      the sink mode rather than observing it.

    The sign follows the suite's convention that the innovation linearises to ``+Hξ``
    (`correct.innovation`): with ``X̂ = exp(ξ)X``, ``ν^v = R̂y^v - v̂ ≈ -ξ_v``.
    """
    if n_contacts < 0:
        raise ValueError(f"n_contacts must be non-negative, got {n_contacts}")
    H = jnp.zeros((3, 3 * n_contacts + 9))
    return H.at[:, 3:6].set(-jnp.eye(3))


def velocity_measurement(omega: Array, h: Array, J: Array, q_dot: Array) -> Array:
    r"""``y^v_i = -([ω]× h_i + J_i q̇)`` for every contact, shape ``(N, 3)``.

    Body frame, and equal to ``Rᵀv`` exactly when contact *i* is world-static.

    ``omega`` is the **bias-corrected** gyro (I1: bias lives in the joint KF, the InEKF
    consumes ``ω̄``), ``h`` the body-frame base->contact vectors (``ContactFrames.y``,
    the same quantity the position block measures), ``J`` their joint Jacobians, ``q̇``
    the joint-KF velocity estimate.

    Note this needs ``J``, **not** ``J̇``. `filter.contact_velocity_noise` and the TODO
    that points at it are built on ``ContactFrames.J_dot``, which
    `main_estimator` currently stubs to zeros — anything resting on that seam is
    silently zero. See `J_DOT_IS_ZERO` below.
    """
    return -(jnp.cross(jnp.broadcast_to(omega, h.shape), h) + J @ q_dot)


def velocity_residual(state: InEKFState, y_v: Array) -> Array:
    r"""``ν^v_i = R̂ y^v_i - v̂`` stacked over contacts, shape ``(3N,)``.

    World frame — see the module docstring for why the body frame is not an option.
    """
    return (y_v @ state.R.T - state.v).reshape(-1)


def velocity_noise(state: InEKFState, sigma_c: Array, J: Array, sigma_q_dot: Array,
                   h: Array, sigma_omega: Array, dt: float,
                   nv_scale: float = 1.0) -> Array:
    r"""``N^v_i``, shape ``(N, 3, 3)`` — the noise on the zero-velocity constraint.

    Propagating first-order uncertainty through the constraint, with
    ``∂([ω]×h)/∂ω = -[h]×``::

        N^v_i = R̂ [ J_i Σ_q̇ J_iᵀ  +  [h_i]× Σ_ω [h_i]×ᵀ  +  Σ_C,i ] R̂ᵀ
                    └ joint vel ┘     └ gyro × lever arm ┘    └ slip ┘

    The ``R̂(·)R̂ᵀ`` congruence mirrors the position block's ``R̂ N R̂ᵀ``, already locked
    by `ContactUpdaterTest`.

    **Σ_C is the slip term, not a separate tunable.** The derivation leaves a slot
    ``N^roll = (κ|ω_f| r)²I`` for foot roll; the algebra in the module docstring says
    that slot *is* the contact process noise, since ``y^v_i = Rᵀv - w_C``. Units agree:
    ``ḋ = R w_C`` makes ``w_C`` a velocity, so Σ_C is (m/s)², commensurate with ``N^v``.
    Feeding a hand-tuned roll constant here *alongside* a learned Σ_C would count the
    same physical evidence twice — I6's error (block-diagonal ``R_g`` on a shared
    source) in a new place.

    This is also what makes the constraint safe during heel->toe roll, which is the
    failure mode that decides whether the whole thing helps or hurts: a rolling contact
    has large Σ_C, so its rows are down-weighted exactly when ``ḋ_i = 0`` stops being
    true. The rule from `anchor_rate_gain` carries over unchanged — **it must inflate,
    never disable**: an over-large Σ_C removes the only absolute velocity observation
    and reintroduces the drift the constraint was added for.

    ``nv_scale`` (κ) is a **diagnostic** multiplier on the whole block, not part of the
    model: ``N^v ← κ N^v``. It exists because the derivation fixes the trust level with
    no free parameter, and that claim needs a falsification — if the constraint only
    helps at some κ ≠ 1 then ``N^v`` as derived is not the right noise. ``κ → ∞`` must
    reproduce the no-ZV baseline exactly (``K → 0``): that is the graceful-degradation
    check, and it failing would be a bug in the update, not a modelling result.
    ``κ = 1`` is bit-identical to the unscaled path, so every recorded number stands.
    """
    h_x = _skew(h)                                             # (N, 3, 3)
    joint = J @ sigma_q_dot @ jnp.swapaxes(J, -1, -2)
    gyro = h_x @ sigma_omega @ jnp.swapaxes(h_x, -1, -2)
    # UNITS. Sigma_C is a spectral DENSITY: `propagate.build_Qd` forms
    # `Qd = M Qc M^T * dt`, so Sigma_C carries m^2/s and the per-tick contact position
    # variance is Sigma_C*dt. N^v is a VELOCITY covariance, (m/s)^2. The average
    # velocity over one tick of a white-noise-driven contact has variance Sigma_C/dt --
    # NOT Sigma_C. Using Sigma_C raw understates the slip term by 1/dt = 1000x at
    # dt = 1 ms, which is what made the first Z6 run trust the constraint ~1000x too
    # much during stance.
    body = joint + gyro + sigma_c / dt
    return nv_scale * (state.R @ body @ state.R.T)


def velocity_position_cross(state: InEKFState, J: Array, sigma_q: Array,
                            omega: Array) -> Array:
    r"""Cross-covariance ``N^{pv}_i = E[δν^p_i δν^{v}_iᵀ]``, shape ``(N, 3, 3)``.

    ``ν^p`` and ``ν^v`` are **not** independent: both read the same encoder error. The
    position residual carries ``δy^p = J δq``; differentiating the constraint,
    ``δy^v ⊇ -[ω]× J δq``. Hence, in the world frame,

        N^{pv}_i = R̂ ( J_i Σ_q J_iᵀ [ω]×ᵀ )ᵀ R̂ᵀ = R̂ ( -[ω]× J_i Σ_q J_iᵀ )ᵀ R̂ᵀ

    A block-diagonal ``R`` over the stacked ``[ν^p; ν^v]`` would drop this and
    double-count the encoder information — invariant **I6** in a new place, with
    `testBiasColumnsOfHgAreExactlyL` as the precedent for asserting the exact joint
    ``L Σ Lᵀ`` rather than assuming independence.

    **This is only HALF the encoder sensitivity — see the `J_DOT_IS_ZERO` block at the
    top of the module.** The full term is ``∂y^v/∂q = -([ω]× J + J̇)``; this function
    carries the ``[ω]J`` half because ``J̇`` is stubbed to zero upstream. Measured, the
    missing half is the *larger* one by 7×. Both are negligible against foot roll at
    today's Σ_q, which is the only reason this is tolerable.
    """
    JSJ = J @ sigma_q @ jnp.swapaxes(J, -1, -2)       # (N, 3, 3)
    body = -_skew(jnp.broadcast_to(omega, (J.shape[0], 3))) @ JSJ
    return state.R @ jnp.swapaxes(body, -1, -2) @ state.R.T


def fuse_contacts(nu_v: Array, N_v: Array) -> tuple[Array, Array]:
    r"""Fold ``N`` estimates of the same 3-vector into one, information-weighted.

    Operates on the **world-frame residuals** ``ν^v_i`` (shape ``(N, 3)``), not on the
    body-frame ``y^v_i``: `velocity_noise` already applies the ``R̂(·)R̂ᵀ`` congruence,
    so fusing the body-frame measurements against world-frame information mixes frames
    and silently produces a different posterior. (It does — that mistake failed
    `test_fused_block_equals_the_stacked_update` before this signature was fixed.)

    Every stance contact contributes the *identical* three rows ``[0|-I|0|0]``, so
    stacking them is ``N`` redundant observations of one 3-vector. That is well posed
    but it is the same redundancy that already measures ``cond(S) = 1.09e9`` against a
    ``cond_s_max`` of 1e9 at N=8 — the margin is one part in ten. Fusing first keeps
    the update at three rows and the conditioning at N=1's::

        N_f⁻¹ = Σ_i (N^v_i)⁻¹,      y_f = N_f Σ_i (N^v_i)⁻¹ y^v_i

    Mathematically identical to the stacked update for independent blocks — every
    ``H_i`` is the same, so the information contributions ``Hᵀ N_i⁻¹ H`` and
    ``Hᵀ N_i⁻¹ ν_i`` simply sum (asserted in `test_zero_velocity.py`). It is also where
    Σ_C does its work: a rolling or swinging corner has large Σ_C, so it contributes
    almost nothing to the fused estimate.

    Returns ``(ν_fused, N_fused)`` with shapes ``(3,)`` and ``(3, 3)``.
    """
    info = jnp.linalg.inv(N_v)                                  # (N, 3, 3), SPD
    N_f = jnp.linalg.inv(info.sum(axis=0))
    nu_f = N_f @ jnp.einsum("nij,nj->i", info, nu_v)
    return nu_f, N_f


def _skew(v: Array) -> Array:
    """``[v]×`` for a stack of vectors, ``(..., 3) -> (..., 3, 3)``."""
    z = jnp.zeros(v.shape[:-1])
    return jnp.stack([
        jnp.stack([z, -v[..., 2], v[..., 1]], axis=-1),
        jnp.stack([v[..., 2], z, -v[..., 0]], axis=-1),
        jnp.stack([-v[..., 1], v[..., 0], z], axis=-1),
    ], axis=-2)
