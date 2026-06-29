"""
inEKF/propagate.py
==================
Propagation (prediction) step of the world-centric, right-invariant contact-aided
InEKF on ``SE_{N+2}(3)`` (CLAUDE.md §3).

The step is the textbook split: a **nonlinear mean** propagated on the group, and
a **linear covariance** propagated in the tangent.  What makes the right-invariant
+ world-centric corner special is that the covariance side is *state-independent*:

* the transition ``Φ`` is the precomputed constant of `state.build_Phi` (§3.2) —
  ``expm`` is never called in the loop;
* the process noise ``Q̄_d`` is the **exact closed-form** integral of the constant
  nilpotent error dynamics (§3.3), not the ``Φ Q̄ Φᵀ dt`` approximation (which
  over-counts the cross terms and corrupts NEES).

Inputs
------
* ``ω̃, ã`` — the **bias-corrected** IMU (the pelvis-IMU bias is removed upstream
  by the joint-KF / Mahony pre-filter; it is *never* in this state — invariant 2).
* ``Σ_c`` — the per-contact noise **densities already in world frame** (i.e.
  ``R̄ Σ_{C_i} R̄ᵀ``), shape ``(N, 3, 3)``.  The Cholesky digest, noise floor and
  rotate-to-world that produce them live in `inekf/contact.py` (§5); propagation
  only multiplies them by ``dt`` to form the ``Q̄_d`` contact blocks.

Everything here is ``jax.jit``-able and differentiable (§8): the mean uses the
``Γ`` closed forms from `group.py` (branch-free near ``θ→0``), and the per-contact
``Q̄_d`` blocks are assembled by a single vectorised scatter rather than a Python
loop over the contacts.

Mean (exact under constant IMU over ``dt``, §3.1)::

    R̄_{k+1} = R̄_k Γ_0(ω̄ dt)
    v̄_{k+1} = v̄_k + R̄_k Γ_1(ω̄ dt) ā dt + g dt
    p̄_{k+1} = p̄_k + v̄_k dt + R̄_k Γ_2(ω̄ dt) ā dt² + ½ g dt²
    d̄_{i,k+1} = d̄_{i,k}                       (contacts: pure noise, §3.3)

Covariance (§3.2-3.3)::

    P_{k+1} = Φ P_k Φᵀ + Q̄_d
"""
from jax import Array

import jax.numpy as jnp

from .group import Gamma0, Gamma1, Gamma2, skew
from .state import InEKFState, InEKFParams


# ---------------------------------------------------------------------------
# Mean propagation (§3.1)
# ---------------------------------------------------------------------------

def propagate_mean(
    state: InEKFState,
    omega: Array,
    accel: Array,
    params: InEKFParams,
) -> InEKFState:
    r"""Propagate the group mean ``(R̄, v̄, p̄, d̄)`` one step (§3.1).

    Exact integration under the constant-IMU-over-``dt`` assumption: ``Γ_1`` and
    ``Γ_2`` are the once/twice integrals of the rotating accelerometer signal, so
    this is *not* an Euler step.  Contact means are unchanged (their dynamics is
    all noise, §3.3).  The covariance ``P`` is carried through untouched.

    Parameters
    ----------
    state : InEKFState
        Prior state.
    omega : Array, shape (3,)
        Bias-corrected gyro measurement ``ω̃`` [rad/s].
    accel : Array, shape (3,)
        Bias-corrected accelerometer measurement ``ã`` [m/s²].
    params : InEKFParams
        Carries ``g`` and ``dt``.

    Returns
    -------
    InEKFState
        State with updated ``(R, v, p)``; ``d`` and ``P`` unchanged.
    """
    dt = params.dt
    g = params.g

    phi = omega * dt
    R = state.R

    R_next = R @ Gamma0(phi)
    v_next = state.v + R @ (Gamma1(phi) @ accel) * dt + g * dt
    p_next = (
        state.p
        + state.v * dt
        + R @ (Gamma2(phi) @ accel) * dt * dt
        + 0.5 * g * dt * dt
    )
    return state._replace(R=R_next, v=v_next, p=p_next)


# ---------------------------------------------------------------------------
# Process noise Q̄_d — exact closed form (§3.3)
# ---------------------------------------------------------------------------

def inertial_Qd(params: InEKFParams) -> Array:
    r"""Exact inertial (``R, v, p``) block of ``Q̄_d``, shape ``(9, 9)`` (§3.3).

    Closed form of ``∫₀^{dt} e^{A^r s} Q̄_c e^{A^rᵀ s} ds`` restricted to the
    inertial states.  Because ``A^r`` is nilpotent (``(A^r)³ = 0``) the integrand
    is a degree-≤4 polynomial in ``s``, so the integral is exact (no truncation,
    no quadrature).  With ``G ≡ (g)_×``, ``Q_g = σ_g² I``, ``Q_a = σ_a² I``::

        [R,R] = Q_g dt
        [R,v] = −½ Q_g G dt²              [v,R] = ½ G Q_g dt²   = [R,v]ᵀ
        [R,p] = −⅙ Q_g G dt³              [p,R] = ⅙ G Q_g dt³   = [R,p]ᵀ
        [v,v] = Q_a dt − ⅓ G Q_g G dt³
        [v,p] = ½ Q_a dt² − ⅛ G Q_g G dt⁴   ( = [p,v], symmetric)
        [p,p] = ⅓ Q_a dt³ − (1/20) G Q_g G dt⁵

    The off-diagonals are the cross terms: gyro noise leaking R→v→p and accel
    noise leaking v→p through the ``A^r`` coupling, accumulated over the step.
    ``−G Q_g G = −σ_g² (g)_×² ⪰ 0`` keeps the result PSD.
    """
    dt = params.dt
    I3 = jnp.eye(3)
    Qg = params.sigma_gyro ** 2 * I3
    Qa = params.sigma_accel ** 2 * I3
    G = skew(params.g)

    QgG = Qg @ G            # [R, v]/[R, p] carrier
    GQgG = G @ Qg @ G       # PSD: −GQgG = −σ_g² (g)_×² ⪰ 0

    RR = Qg * dt
    vv = Qa * dt - (1.0 / 3.0) * GQgG * dt ** 3
    pp = (1.0 / 3.0) * Qa * dt ** 3 - (1.0 / 20.0) * GQgG * dt ** 5

    Rv = -0.5 * QgG * dt ** 2                                 # [R, v]
    Rp = -(1.0 / 6.0) * QgG * dt ** 3                         # [R, p]
    vp = 0.5 * Qa * dt ** 2 - (1.0 / 8.0) * GQgG * dt ** 4    # [v, p] = [p, v]

    Q = jnp.zeros((9, 9))
    Q = Q.at[0:3, 0:3].set(RR)
    Q = Q.at[3:6, 3:6].set(vv)
    Q = Q.at[6:9, 6:9].set(pp)
    Q = Q.at[0:3, 3:6].set(Rv)
    Q = Q.at[3:6, 0:3].set(Rv.T)
    Q = Q.at[0:3, 6:9].set(Rp)
    Q = Q.at[6:9, 0:3].set(Rp.T)
    Q = Q.at[3:6, 6:9].set(vp)
    Q = Q.at[6:9, 3:6].set(vp.T)
    return Q


def _block_diag_from_stack(blocks: Array) -> Array:
    """Block-diagonal ``(3N, 3N)`` matrix from a ``(N, 3, 3)`` stack.

    Vectorised (no Python loop over contacts): block ``(i, i)`` is ``blocks[i]``
    and every off-diagonal block is zero.  Handles ``N = 0`` (returns ``(0, 0)``).
    """
    N = blocks.shape[0]
    # selector[i, j] = δ_ij · blocks[i]  → (N, N, 3, 3); regroup to (3N, 3N).
    selector = jnp.einsum("ij,ikl->ijkl", jnp.eye(N), blocks)
    return selector.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


def build_Qd(sigma_c: Array, params: InEKFParams) -> Array:
    r"""Full process-noise matrix ``Q̄_d``, shape ``(3N+9, 3N+9)`` (§3.3).

    Inertial ``9x9`` block from `inertial_Qd`; the ``N`` contact blocks are the
    decoupled (no propagation cross-terms, no leakage into the base states)

        ``Q̄_d[d_i, d_i] = Σ_c[i] · dt``

    where ``Σ_c[i]`` is the world-frame contact noise density ``R̄ Σ_{C_i} R̄ᵀ``
    supplied by `inekf/contact.py`'s digest (§5).

    Parameters
    ----------
    sigma_c : Array, shape (N, 3, 3)
        Per-contact world-frame noise densities ``R̄ Σ_{C_i} R̄ᵀ``.
    params : InEKFParams
        Carries ``dt`` and the IMU densities.

    Returns
    -------
    Array, shape (3N+9, 3N+9)
    """
    N = sigma_c.shape[0]
    Qd = jnp.zeros((3 * N + 9, 3 * N + 9))
    Qd = Qd.at[0:9, 0:9].set(inertial_Qd(params))
    contact_blocks = _block_diag_from_stack(sigma_c * params.dt)
    Qd = Qd.at[9:, 9:].set(contact_blocks)
    return Qd


# ---------------------------------------------------------------------------
# Covariance propagation (§3.2)
# ---------------------------------------------------------------------------

def propagate_cov(P: Array, sigma_c: Array, params: InEKFParams) -> Array:
    r"""Right-invariant covariance step ``P⁺ = Φ P Φᵀ + Q̄_d`` (§3.2-3.3).

    ``Φ`` is the precomputed constant transition (`InEKFParams.Phi`); ``Q̄_d`` is
    the exact closed form (`build_Qd`).  The result is symmetrised for numerical
    hygiene (invariant 7).

    Parameters
    ----------
    P : Array, shape (3N+9, 3N+9)
        Prior covariance.
    sigma_c : Array, shape (N, 3, 3)
        World-frame contact noise densities (see `build_Qd`).
    params : InEKFParams

    Returns
    -------
    Array, shape (3N+9, 3N+9)
    """
    Phi = params.Phi
    Pn = Phi @ P @ Phi.T + build_Qd(sigma_c, params)
    return 0.5 * (Pn + Pn.T)


# ---------------------------------------------------------------------------
# Full propagation step
# ---------------------------------------------------------------------------

def propagate(
    state: InEKFState,
    omega: Array,
    accel: Array,
    sigma_c: Array,
    params: InEKFParams,
) -> InEKFState:
    r"""One full InEKF prediction step: mean (§3.1) then covariance (§3.2-3.3).

    Parameters
    ----------
    state : InEKFState
        Prior state.
    omega, accel : Array, shape (3,)
        Bias-corrected IMU ``(ω̃, ã)``.
    sigma_c : Array, shape (N, 3, 3)
        Per-contact world-frame noise densities ``R̄ Σ_{C_i} R̄ᵀ`` (§5 digest).
    params : InEKFParams

    Returns
    -------
    InEKFState
        Predicted state ``(R̄⁺, v̄⁺, p̄⁺, d̄, P⁺)``.
    """
    mean = propagate_mean(state, omega, accel, params)
    P_next = propagate_cov(state.P, sigma_c, params)
    return mean._replace(P=P_next)
