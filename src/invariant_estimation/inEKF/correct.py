r"""
inEKF/correct.py
================
Correction (measurement-update) step of the world-centric, right-invariant
contact-aided InEKF on ``SE_{N+2}(3)`` (CLAUDE.md §4).

Forward kinematics gives the body-frame vector base→contact,
``y_i = h_{p,i}(q̂) = {}^{B}p_{BC_i}``.  In the world-centric state this is a
**right-invariant observation** (``Y = X⁻¹ b + V``), which is exactly why the
observation matrix ``H`` is *state-independent* and can be precomputed once
(`state.build_H`, invariant 6) — the corner chosen in §0.

Per-contact observation (§4.1)
------------------------------
With ``b`` the homogeneous selector ``[0_3 ; 0 ; 1 ; −1]`` (the ``+1`` in the
``p`` slot, the ``−1`` in this contact's ``d_i`` slot) and the state's own
prediction ``h_pred_i = R̄ᵀ(d̄_i − p̄)``, the right-invariant innovation mapped
into the world frame is (§4.3)

    \nu_i = Π(X̄ Y_i) = R̄ y_i − (d̄_i − p̄) = R̄ (y_i − h_pred_i)  (measurement − model).

**Sign convention (CLAUDE.md I5).**  The precomputed ``H`` (`state.build_H`) is
``H_i = [0  0  +I  …  −I(col d_i)  …]`` ⟹ ``H_i ξ = ξ_p − ξ_{d_i}``, matching the
Java `ContactUpdater.computeJacobian` element-for-element.  The innovation
linearises the *same* way, ``\nu_i ≈ ξ_p − ξ_{d_i} = +H_i ξ``, so ``ξ⁺ = K \nu``
estimates the error and is removed by ``X̂⁺ = exp(−(Kν)^∧) X̂``.  Then
``ξ_err⁺ = (I − KH) ξ_err`` (error reduces) and the Joseph form is exact.

Measurement noise (§4.2)
------------------------
The position FK noise ``N^p_i = J_{C_i}(q̂) Σ_q J_{C_i}ᵀ`` is routed in **already
assembled** as ``Np`` (shape ``(N, 3, 3)``): the kinematic Jacobian ``J_{C_i}``
(from ``robot/``) and the filtered joint covariance ``Σ_q`` (from ``joint_kf``)
are multiplied upstream — this module imports neither, mirroring how
`propagate.py` consumes the already-digested ``sigma_c``.  The boundary contract
(§6) is honoured: joint-KF outputs reach the filter only here, on the correction
side, always pre-multiplied by a kinematic Jacobian.

    # TODO(N^v / zero-velocity): the contact zero-velocity constraint with its
    # own noise N^v_i = J_{Ċ_i} Σ_q̇ J_{Ċ_i}ᵀ (§4.2) is a *separate* measurement
    # block, never folded into N^p.  Deferred for v1 (§10); when it lands it
    # stacks below the position block with its own H rows and its own Np-like
    # input — it does not mix into this Jacobian.

Gain + update (§4.3)
--------------------
    S  = H P Hᵀ + N
    K  = P Hᵀ S⁻¹                         (via cho_solve — never `inv`)
    ξ⁺ = K \nu
    P⁺ = (I − K H) P (I − K H)ᵀ + K N Kᵀ  (Joseph form — mandatory)
    X̂⁺ = exp(−ξ⁺) X̂                       (right-invariant: exp on the LEFT, I5)

Everything is ``jax.jit``-able and differentiable (§8): per-contact work is
vectorised (no Python loop over contacts), ``S⁻¹`` is a Cholesky solve, and the
covariance is kept symmetric.  ``N`` is static, so the ``N = 0`` (no candidates)
case is a plain early return.
"""
from jax import Array
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from .group import exp_SEn3
from .state import (
    InEKFParams,
    InEKFState,
    _check_contact_index,
    contact_tangent_index,
)


# ---------------------------------------------------------------------------
# Observation model (§4.1)
# ---------------------------------------------------------------------------

def predicted_contact(state: InEKFState) -> Array:
    r"""State-predicted body-frame FK vectors ``h_pred_i = R̄ᵀ(d̄_i − p̄)`` (§4.1).

    The model counterpart of the measured ``y_i = h_{p,i}(q̂)``; also a natural
    ContactNet trust feature.  Vectorised over the ``N`` contacts.

    Parameters
    ----------
    state : InEKFState

    Returns
    -------
    Array, shape (N, 3)
        Predicted base→contact vectors in the body frame.
    """
    rel = state.d - state.p                       # (N, 3): d̄_i − p̄ in world
    return rel @ state.R                          # (R̄ᵀ rel_i) stacked = rel @ R


def innovation(state: InEKFState, y: Array) -> Array:
    r"""Right-invariant innovation stacked over contacts (§4.1), shape ``(3N,)``.

    ``\nu_i = Π(X̄ Y_i) = R̄ y_i − (d̄_i − p̄)`` (measurement − model, world frame);
    it linearises to ``ξ_p − ξ_{d_i} = +H_i ξ``, so ``ξ⁺ = K \nu`` is the error
    estimate and is applied as ``exp(−ξ⁺)`` (I5).

    Parameters
    ----------
    state : InEKFState
    y : Array, shape (N, 3)
        Measured body-frame FK vectors ``h_{p,i}(q̂)`` from ``robot/`` FK.

    Returns
    -------
    Array, shape (3N,)
        ``[\nu_1 ; … ; \nu_N]`` in the same block order as ``H``.
    """
    rel = state.d - state.p                       # (N, 3): d̄_i − p̄
    nu = y @ state.R.T - rel                       # (N, 3): R̄ y_i − (d̄_i − p̄)
    return nu.reshape(-1)


def _block_diag(blocks: Array) -> Array:
    """Block-diagonal ``(3N, 3N)`` from a ``(N, 3, 3)`` stack (vectorised).

    Block ``(i, i)`` is ``blocks[i]``; every off-diagonal block is zero.  Handles
    ``N = 0`` (returns ``(0, 0)``).  Same construction as `propagate`'s contact
    assembly — kept local so `correct.py` stays decoupled.
    """
    N = blocks.shape[0]
    selector = jnp.einsum("ij,ikl->ijkl", jnp.eye(N), blocks)   # δ_ij blocks[i]
    return selector.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


def measurement_noise(Np: Array) -> Array:
    r"""Stacked FK measurement noise ``N`` (block-diagonal), shape ``(3N, 3N)``.

    The per-contact position noises ``N^p_i = J_{C_i} Σ_q J_{C_i}ᵀ`` (§4.2) are
    independent across contacts, so the stacked noise is block-diagonal.

    Parameters
    ----------
    Np : Array, shape (N, 3, 3)
        Per-contact position FK covariances (pre-assembled upstream).

    Returns
    -------
    Array, shape (3N, 3N)
    """
    return _block_diag(Np)


# ---------------------------------------------------------------------------
# Gain and Joseph update (§4.3)
# ---------------------------------------------------------------------------

def kalman_gain(P: Array, H: Array, N: Array) -> tuple[Array, Array]:
    r"""Right-invariant Kalman gain ``K = P Hᵀ S⁻¹`` with ``S = H P Hᵀ + N``.

    ``S⁻¹`` is applied through a Cholesky solve (``S`` is SPD: ``H`` is full row
    rank and ``P`` SPD ⟹ ``H P Hᵀ`` SPD, ``N`` PSD) — never an explicit inverse
    (invariant 7).

    Parameters
    ----------
    P : Array, shape (3N+9, 3N+9)
        Predicted covariance.
    H : Array, shape (3N, 3N+9)
        Constant FK observation (`InEKFParams.H`).
    N : Array, shape (3N, 3N)
        Measurement noise (`measurement_noise`).

    Returns
    -------
    K : Array, shape (3N+9, 3N)
    S : Array, shape (3N, 3N)
        Innovation covariance (returned for downstream NIS / consistency checks).
    """
    PHt = P @ H.T                                 # (3N+9, 3N)
    S = H @ PHt + N                               # (3N, 3N)
    cho = cho_factor(S)
    # K = PHt S⁻¹  ⟺  S Kᵀ = PHtᵀ  (S symmetric) ⟹ Kᵀ = cho_solve(S, PHtᵀ).
    K = cho_solve(cho, PHt.T).T
    return K, S


def joseph_update(P: Array, K: Array, H: Array, N: Array) -> Array:
    r"""Joseph-form covariance update ``P⁺ = (I−KH) P (I−KH)ᵀ + K N Kᵀ`` (§4.3).

    Joseph form is mandatory: ``K`` comes from a linearised ``H`` and is never
    exactly optimal, so the short ``(I−KH)P`` is not guaranteed PSD; the Joseph
    form is.  The result is symmetrised (invariant 7).

    Parameters
    ----------
    P : Array, shape (3N+9, 3N+9)
    K : Array, shape (3N+9, 3N)
    H : Array, shape (3N, 3N+9)
    N : Array, shape (3N, 3N)

    Returns
    -------
    Array, shape (3N+9, 3N+9)
    """
    IKH = jnp.eye(P.shape[0]) - K @ H
    Pp = IKH @ P @ IKH.T + K @ N @ K.T
    return 0.5 * (Pp + Pp.T)


def apply_correction(state: InEKFState, xi: Array) -> InEKFState:
    r"""Apply the tangent correction ``X̂⁺ = exp(−ξ⁺) X̂`` (left multiply, I5).

    Right-invariant ⟹ ``exp`` multiplies on the **left** of ``X̂``, so base *and*
    every contact move consistently — the off-diagonal covariance coupling is
    what lets a foot measurement sharpen the base and vice versa.

    **Sign** — with the Java/I5 ``H`` (`state.build_H`) the residual linearises
    to ``ν ≈ +H ξ``, so ``ξ⁺ = Kν`` is an estimate *of the error itself* and must
    be subtracted: hence ``exp(−ξ⁺)``, CLAUDE.md I5 verbatim. Getting this
    backwards passes the easy tests and diverges under transients (§6).

    Parameters
    ----------
    state : InEKFState
        Predicted state (covariance left untouched here; updated separately).
    xi : Array, shape (3N+9,)
        Correction ``ξ⁺ = K \nu`` in the fixed ``[ξ_R ; ξ_v ; ξ_p ; ξ_{d_i}]``
        order.  Applied as ``exp(−ξ⁺)``.

    Returns
    -------
    InEKFState
        State with corrected ``(R, v, p, d)``; ``P`` unchanged.
    """
    N = state.N
    Xi = exp_SEn3(-xi, N)                         # (N+5, N+5) — I5 sign
    Xnew = Xi @ state.as_matrix
    return state._replace(
        R=Xnew[0:3, 0:3],
        v=Xnew[0:3, 3],
        p=Xnew[0:3, 4],
        d=Xnew[0:3, 5:].T,
    )


# ---------------------------------------------------------------------------
# ContactUpdater seams (Java `ContactUpdater`, ported suite)
#
# The per-contact views of the machinery above.  `correct` is the vectorised
# hot path over all N contacts; these are the single-contact entry points the
# Java class exposes and the ported `ContactUpdaterTest` exercises directly
# (CLAUDE.md I10 — the test seams ARE the public surface).
# ---------------------------------------------------------------------------

def contact_jacobian(N: int, contact_index: int) -> Array:
    r"""Single-contact observation Jacobian ``H_i``, shape ``(3, 3N+9)``.

    ``H_i = [ 0_{3x6} | +I_3 | … −I_3 (own d_i block) … ]`` — Java
    `ContactUpdater.computeJacobian`.  **State-independent by construction**:
    the argument is the contact *index*, not the state, which is exactly the
    property `testJacobianStructureAndStateIndependence` asserts (it calls the
    Java form with two different random states and demands bit-equality).

    Parameters
    ----------
    N : int
        Number of contact candidates (static).
    contact_index : int
        Which contact; `IndexError` if out of range.

    Returns
    -------
    Array, shape (3, 3N+9)
    """
    _check_contact_index(contact_index, N)
    H = jnp.zeros((3, 3 * N + 9))
    H = H.at[:, 6:9].set(jnp.eye(3))
    j = contact_tangent_index(contact_index)
    return H.at[:, j:j + 3].set(-jnp.eye(3))


def contact_residual(state: InEKFState, contact_index: int, y: Array) -> Array:
    r"""Single-contact world residual ``r = R̂ y − (d̂_i − p̂)``, shape ``(3,)``.

    Java `ContactUpdater.computeResidual`.  The per-contact slice of
    `innovation`; ``y`` is the measured body-frame FK vector.
    """
    _check_contact_index(contact_index, state.N)
    return state.R @ y - (state.d[contact_index] - state.p)


def rotate_measurement_covariance(state: InEKFState, body_cov: Array) -> Array:
    r"""Rotate a body-frame measurement covariance to world: ``R̂ N R̂ᵀ``.

    Java `ContactUpdater.computeMeasurementCovariance`.  The residual lives in
    the world frame (`contact_residual`), so the body-frame FK noise must be
    conjugated by the estimated attitude before it enters ``S``.
    """
    return state.R @ body_cov @ state.R.T


def map_encoder_noise(contact_jac: Array, joint_cov: Array) -> Array:
    r"""Encoder noise through the kinematics: ``N = J Σ_q Jᵀ`` (§4.2).

    Java `ContactUpdater.mapEncoderNoise` (static).  ``J`` is the contact-point
    position Jacobian at ``q̂`` and ``Σ_q`` the filtered joint covariance from the
    joint KF; the result is the **body-frame** FK covariance, which
    `rotate_measurement_covariance` then takes to world.

    Parameters
    ----------
    contact_jac : Array, shape (3, n_joints)
    joint_cov : Array, shape (n_joints, n_joints)

    Returns
    -------
    Array, shape (3, 3)
    """
    return contact_jac @ joint_cov @ contact_jac.T


def contact_update(
    state: InEKFState,
    contact_index: int,
    measurement: Array,
    body_covariance: Array,
    learned: bool = False,
) -> tuple[InEKFState, Array]:
    r"""One single-contact FK update — Java `InvariantUpdater.update(...)`.

    Composes the seams above: residual → rotate noise to world → gain → Joseph →
    ``exp(−ξ)`` (I5).  Returns the corrected state and the residual actually used.

    Parameters
    ----------
    state : InEKFState
    contact_index : int
        Which contact; `IndexError` if out of range.
    measurement : Array, shape (3,)
        Measured body-frame FK vector ``y_i = h_{p,i}(q̂)``.
    body_covariance : Array, shape (3, 3)
        Body-frame measurement covariance (e.g. from `map_encoder_noise`).
    learned : bool
        The learned-measurement branch.  Raises `NotImplementedError` while
        ContactNet is unlanded — the Java contract raises
        `NotImplementedException` and the ported test asserts it (§7).

    Returns
    -------
    state : InEKFState
    residual : Array, shape (3,)
    """
    if learned:
        raise NotImplementedError(
            "learned contact measurement module is not implemented; "
            "see CLAUDE.md §7 (ContactNet socket)"
        )
    _check_contact_index(contact_index, state.N)

    H = contact_jacobian(state.N, contact_index)
    residual = contact_residual(state, contact_index, measurement)
    Nmat = rotate_measurement_covariance(state, body_covariance)

    K, _ = kalman_gain(state.P, H, Nmat)
    corrected = apply_correction(state, K @ residual)
    P_new = joseph_update(state.P, K, H, Nmat)
    return corrected._replace(P=P_new), residual


# ---------------------------------------------------------------------------
# Full correction step
# ---------------------------------------------------------------------------

def correct(
    state: InEKFState,
    y: Array,
    Np: Array,
    params: InEKFParams,
) -> tuple[InEKFState, Array]:
    r"""One full FK measurement update (§4): innovation → gain → Joseph → exp.

    The observation matrix ``H`` is the precomputed constant (`InEKFParams.H`);
    nothing here is rebuilt per step.  With ``N = 0`` candidates there is no
    measurement, so this is a static early return (``N`` is not traced).

    Parameters
    ----------
    state : InEKFState
        Predicted (post-propagation) state.
    y : Array, shape (N, 3)
        Measured body-frame FK vectors ``h_{p,i}(q̂)`` (from ``robot/`` FK).
    Np : Array, shape (N, 3, 3)
        Per-contact position FK covariances ``J_{C_i} Σ_q J_{C_i}ᵀ`` (§4.2),
        assembled upstream.
    params : InEKFParams
        Carries the constant ``H``.

    Returns
    -------
    state : InEKFState
        Corrected state ``(R⁺, v⁺, p⁺, d⁺, P⁺)``.
    nu : Array, shape (3N,)
        The innovation, emitted for the filter outputs (NIS / ContactNet feature).
    """
    if state.N == 0:                              # static: no contacts, no update
        return state, jnp.zeros(0)

    H = params.H
    nu = innovation(state, y)                     # (3N,)
    Nmat = measurement_noise(Np)                  # (3N, 3N)

    K, _ = kalman_gain(state.P, H, Nmat)          # (3N+9, 3N)
    xi = K @ nu                                   # (3N+9,)

    corrected = apply_correction(state, xi)
    P_new = joseph_update(state.P, K, H, Nmat)
    return corrected._replace(P=P_new), nu
