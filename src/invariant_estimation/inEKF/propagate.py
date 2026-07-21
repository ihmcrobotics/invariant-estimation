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

from .group import Adjoint, Gamma0, Gamma1, Gamma2
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

def continuous_Qc(sigma_c: Array, params: InEKFParams) -> Array:
    r"""Continuous **body-frame** error density ``Q_c``, shape ``(3N+9, 3N+9)``.

    Block-diagonal in the tangent ordering ``[R, v, p, d_1 … d_N]``::

        Q_c = blkdiag( gyro_var·I₃ , accel_var·I₃ , 0₃ , Σ_{C_1} … Σ_{C_N} )

    The position block is zero: position picks up noise only through the
    ``A``-coupling from velocity, which the ``Φ`` conjugation in `build_Qd`
    supplies.  Contact covariances enter in the **body / contact frame** — the
    ``Ad_X̂`` conjugation rotates them to world (see `build_Qd`).

    Parameters
    ----------
    sigma_c : Array, shape (N, 3, 3)
        Per-contact body-frame covariances ``Σ_{C_i}`` (§5 digest).
    params : InEKFParams
        Carries the IMU variance densities.
    """
    N = sigma_c.shape[0]
    I3 = jnp.eye(3)
    Qc = jnp.zeros((3 * N + 9, 3 * N + 9))
    Qc = Qc.at[0:3, 0:3].set(params.gyro_var * I3)
    Qc = Qc.at[3:6, 3:6].set(params.accel_var * I3)
    # position block stays zero
    Qc = Qc.at[9:, 9:].set(_block_diag_from_stack(sigma_c))
    return Qc


def _block_diag_from_stack(blocks: Array) -> Array:
    """Block-diagonal ``(3N, 3N)`` matrix from a ``(N, 3, 3)`` stack.

    Vectorised (no Python loop over contacts): block ``(i, i)`` is ``blocks[i]``
    and every off-diagonal block is zero.  Handles ``N = 0`` (returns ``(0, 0)``).
    """
    N = blocks.shape[0]
    # selector[i, j] = δ_ij · blocks[i]  → (N, N, 3, 3); regroup to (3N, 3N).
    selector = jnp.einsum("ij,ikl->ijkl", jnp.eye(N), blocks)
    return selector.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


def build_Qd(sigma_c: Array, Ad: Array, params: InEKFParams) -> Array:
    r"""Discrete process noise ``Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt`` (paper Eq. 38).

    This is CLAUDE.md **I3** verbatim: the ``Ad_X̂`` conjugation stays.  IMU noise
    is measured in the *body* frame, so the right-invariant error picks it up as
    ``Ad_X̂ w``; dropping the adjoint to match Hartley's convention is the named
    trap of §6.  Note this is *not* trivial even for isotropic ``Q_g, Q_a``:
    ``Ad_X̂`` carries ``(v)_× R̂`` and ``(p)_× R̂`` in its first block-column, so the
    conjugation generates genuine cross terms.

    ``Q_d`` is therefore **state-dependent but error-independent** — it depends on
    the estimate ``X̂`` through ``Ad_X̂``, never on the error ``ξ``, which is exactly
    what leaves the log-linear property of I3 intact.

    The ``Δt`` is the paper's first-order discretisation of
    ``∫₀^{Δt} e^{As} Ad Q_c Adᵀ e^{Aᵀs} ds``, and matches the working Java
    estimator.  TODO(van-loan): the exact closed-form integral is available if
    NEES/NIS at G10 shows the cross terms are overstated — ``A`` is nilpotent, so
    the integrand is a degree-≤4 polynomial in ``s`` and the integral is
    closed-form.  Deliberately not done for v1: the Java reference does not.

    Parameters
    ----------
    sigma_c : Array, shape (N, 3, 3)
        Per-contact **body-frame** covariances ``Σ_{C_i}``.
    Ad : Array, shape (3N+9, 3N+9)
        Adjoint ``Ad_X̂`` of the current estimate (`group.Adjoint`).
    params : InEKFParams
        Carries ``Φ``, ``dt`` and the IMU variance densities.

    Returns
    -------
    Array, shape (3N+9, 3N+9)
        Symmetric PSD discrete process noise.
    """
    Qc = continuous_Qc(sigma_c, params)
    M = params.Phi @ Ad
    Qd = M @ Qc @ M.T * params.dt
    return 0.5 * (Qd + Qd.T)


# ---------------------------------------------------------------------------
# Covariance propagation (§3.2)
# ---------------------------------------------------------------------------

def propagate_cov(P: Array, sigma_c: Array, Ad: Array, params: InEKFParams) -> Array:
    r"""Right-invariant covariance step ``P⁺ = Φ P Φᵀ + Q_d``.

    ``Φ`` is the precomputed constant transition (`InEKFParams.Phi`); ``Q_d`` is
    Eq. 38 (`build_Qd`).  The result is symmetrised for numerical hygiene.

    Parameters
    ----------
    P : Array, shape (3N+9, 3N+9)
        Prior covariance.
    sigma_c : Array, shape (N, 3, 3)
        Per-contact body-frame covariances (see `build_Qd`).
    Ad : Array, shape (3N+9, 3N+9)
        Adjoint of the **prior** estimate.
    params : InEKFParams

    Returns
    -------
    Array, shape (3N+9, 3N+9)
    """
    Phi = params.Phi
    Pn = Phi @ P @ Phi.T + build_Qd(sigma_c, Ad, params)
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
        Per-contact **body-frame** covariances ``Σ_{C_i}`` (§5 digest).
    params : InEKFParams

    Returns
    -------
    InEKFState
        Predicted state ``(R̄⁺, v̄⁺, p̄⁺, d̄, P⁺)``.
    """
    mean = propagate_mean(state, omega, accel, params)
    # Ad of the PRIOR estimate: Q_d is evaluated at the state entering the step,
    # matching the Java predict ordering.  The difference is O(dt).
    Ad = Adjoint(state.as_matrix)
    P_next = propagate_cov(state.P, sigma_c, Ad, params)
    return mean._replace(P=P_next)
