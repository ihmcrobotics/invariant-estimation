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
    silently zero.
    """
    return -(jnp.cross(jnp.broadcast_to(omega, h.shape), h) + J @ q_dot)


def velocity_residual(state: InEKFState, y_v: Array) -> Array:
    r"""``ν^v_i = R̂ y^v_i - v̂`` stacked over contacts, shape ``(3N,)``.

    World frame — see the module docstring for why the body frame is not an option.
    """
    return (y_v @ state.R.T - state.v).reshape(-1)


def velocity_noise(state: InEKFState, sigma_c: Array, J: Array, sigma_q_dot: Array,
                   h: Array, sigma_omega: Array) -> Array:
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
    """
    h_x = _skew(h)                                             # (N, 3, 3)
    joint = J @ jnp.diag(sigma_q_dot) @ jnp.swapaxes(J, -1, -2)
    gyro = h_x @ sigma_omega @ jnp.swapaxes(h_x, -1, -2)
    body = joint + gyro + sigma_c
    return state.R @ body @ state.R.T


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

    **Deviation from `contact-zero-velocity.pdf`, deliberate.** The write-up's ``N^v``
    lists an encoder term ``J̇ Σ_q J̇ᵀ``; the ``-[ω]× J δq`` coupling above is not in it.
    That term is first order in Σ_q and does not vanish when ``J̇`` does — and ``J̇``
    *is* currently zero (`main_estimator` stubs it), so building the encoder
    contribution on ``J̇`` alone would make it identically zero. Implemented as derived;
    flagged here so the discrepancy is not silently absorbed.
    """
    JSJ = J @ jnp.diag(sigma_q) @ jnp.swapaxes(J, -1, -2)       # (N, 3, 3)
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
