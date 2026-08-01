r"""Correction (measurement-update) step of the InEKF on ``SE_{N+2}(3)``.

Forward kinematics gives the body-frame vector base→contact,
``y_i = h_{p,i}(q̂) = {}^{B}p_{BC_i}``.  In the world-centric state this is a
**right-invariant observation** (``Y = X⁻¹ b + V``), which is exactly why the
observation matrix ``H`` is *state-independent* and can be precomputed once
(`state.build_H`).

With ``b`` the homogeneous selector ``[0_3 ; 0 ; 1 ; −1]`` (the ``+1`` in the ``p``
slot, the ``−1`` in this contact's ``d_i`` slot) and the state's own prediction
``h_pred_i = R̄ᵀ(d̄_i − p̄)``, the right-invariant innovation mapped into the world
frame is

    \nu_i = Π(X̄ Y_i) = R̄ y_i − (d̄_i − p̄) = R̄ (y_i − h_pred_i)  (measurement − model).

**Sign convention (I5).**  ``H_i = [0  0  +I  …  −I(col d_i)  …]`` ⟹
``H_i ξ = ξ_p − ξ_{d_i}``, matching the Java `ContactUpdater.computeJacobian`
element-for-element.  The innovation linearises the *same* way,
``\nu_i ≈ ξ_p − ξ_{d_i} = +H_i ξ``, so ``ξ⁺ = K \nu`` estimates the error and is
removed by ``X̂⁺ = exp(−(Kν)^∧) X̂``.  Then ``ξ_err⁺ = (I − KH) ξ_err`` (error
reduces) and the Joseph form is exact.

The position FK noise ``N^p_i = J_{C_i}(q̂) Σ_q J_{C_i}ᵀ`` arrives **already
assembled** as ``Np`` ``(N, 3, 3)``: the kinematic Jacobian (from ``robot/``) and the
filtered joint covariance (from ``joint_kf``) are multiplied upstream, so this
module imports neither.  Joint-KF outputs reach the filter only here, on the
correction side, always pre-multiplied by a kinematic Jacobian.

Gain + update (§4.3)::

    S  = H P Hᵀ + N
    K  = P Hᵀ S⁻¹                         (via cho_solve — never `inv`)
    ξ⁺ = K \nu
    P⁺ = (I − K H) P (I − K H)ᵀ + K N Kᵀ  (Joseph form — mandatory)
    X̂⁺ = exp(−ξ⁺) X̂                       (right-invariant: exp on the LEFT, I5)

``N`` is static, so the ``N = 0`` (no candidates) case is a plain early return.
"""
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from ..config import section

from .group import exp_SEn3
from .state import (
    InEKFParams,
    InEKFState,
    _check_contact_index,
    contact_tangent_index,
)


def predicted_contact(state: InEKFState) -> Array:
    r"""State-predicted body-frame FK vectors ``h_pred_i = R̄ᵀ(d̄_i − p̄)``, ``(N, 3)``.

    The model counterpart of the measured ``y_i = h_{p,i}(q̂)``; also a natural
    ContactNet trust feature.
    """
    rel = state.d - state.p                       # (N, 3): d̄_i − p̄ in world
    return rel @ state.R                          # (R̄ᵀ rel_i) stacked = rel @ R


def innovation(state: InEKFState, y: Array) -> Array:
    r"""Right-invariant innovation ``\nu_i = R̄ y_i − (d̄_i − p̄)`` stacked over contacts, ``(3N,)``.

    ``y`` is ``(N, 3)`` measured body-frame FK vectors; the result is world-frame and
    in the same block order as ``H``.  It linearises to ``ξ_p − ξ_{d_i} = +H_i ξ``, so
    ``ξ⁺ = K \nu`` is the error estimate and is applied as ``exp(−ξ⁺)`` (I5).
    """
    rel = state.d - state.p                       # (N, 3): d̄_i − p̄
    nu = y @ state.R.T - rel                       # (N, 3): R̄ y_i − (d̄_i − p̄)
    return nu.reshape(-1)


def _block_diag(blocks: Array) -> Array:
    """``(N,3,3)`` stack → block-diagonal ``(3N, 3N)``, vectorised; handles ``N = 0``.

    Same construction as `propagate`'s contact assembly — kept local so `correct.py`
    stays decoupled.
    """
    N = blocks.shape[0]
    selector = jnp.einsum("ij,ikl->ijkl", jnp.eye(N), blocks)   # δ_ij blocks[i]
    return selector.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


def measurement_noise(Np: Array) -> Array:
    r"""``(N,3,3)`` per-contact ``N^p_i`` → stacked ``(3N, 3N)`` noise, block-diagonal because contacts are independent."""
    return _block_diag(Np)


def kalman_gain(P: Array, H: Array, N: Array) -> tuple[Array, Array]:
    r"""``(K, S)`` with ``K = P Hᵀ S⁻¹``, ``S = H P Hᵀ + N`` — the innovation covariance is returned for NIS.

    ``S⁻¹`` is applied through a Cholesky solve (``S`` is SPD: ``H`` full row rank and
    ``P`` SPD ⟹ ``H P Hᵀ`` SPD, ``N`` PSD) — never an explicit inverse.
    """
    PHt = P @ H.T                                 # (3N+9, 3N)
    S = H @ PHt + N                               # (3N, 3N)
    cho = cho_factor(S)
    # K = PHt S⁻¹  ⟺  S Kᵀ = PHtᵀ  (S symmetric) ⟹ Kᵀ = cho_solve(S, PHtᵀ).
    K = cho_solve(cho, PHt.T).T
    return K, S


def joseph_update(P: Array, K: Array, H: Array, N: Array) -> Array:
    r"""Joseph-form covariance update ``P⁺ = (I−KH) P (I−KH)ᵀ + K N Kᵀ``, symmetrised.

    Joseph form is mandatory: ``K`` comes from a linearised ``H`` and is never exactly
    optimal, so the short ``(I−KH)P`` is not guaranteed PSD; the Joseph form is.
    """
    IKH = jnp.eye(P.shape[0]) - K @ H
    Pp = IKH @ P @ IKH.T + K @ N @ K.T
    return 0.5 * (Pp + Pp.T)


def apply_correction(state: InEKFState, xi: Array) -> InEKFState:
    r"""Apply the tangent correction ``X̂⁺ = exp(−ξ⁺) X̂`` (left multiply, I5); ``P`` unchanged.

    Right-invariant ⟹ ``exp`` multiplies on the **left** of ``X̂``, so base *and* every
    contact move consistently — the off-diagonal covariance coupling is what lets a
    foot measurement sharpen the base and vice versa.

    **Sign** — with the Java/I5 ``H`` the residual linearises to ``ν ≈ +H ξ``, so
    ``ξ⁺ = Kν`` is an estimate *of the error itself* and must be subtracted: hence
    ``exp(−ξ⁺)``.  Getting this backwards passes the easy tests and diverges under
    transients.
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


class UpdateDiagnostics(NamedTuple):
    """Published per-update diagnostics — the Java `InvariantUpdater` / `InvariantEKF`
    introspection getters, and part of the seam surface (I10), not optional logging.

    Attributes
    ----------
    applied : Array, scalar float
        1.0 if the gain was applied, 0.0 if gated out.  A gated update leaves
        ``(X̂, P)`` bit-for-bit unchanged (masked ``K``, never a Python branch).
    nis : Array, scalar
        ``rᵀ S⁻¹ r``, computed on the **prior** ``P`` and the **prior** residual — do
        not compute it on the posterior.  NaN before any update.
    condition_proxy : Array, scalar
        ``(max L_ii / min L_ii)²`` from the Cholesky of ``S`` — the §4 gate proxy.
    correction_rotation_norm : Array, scalar
        ``‖(Kν)_rotation‖``; zero-release checks read this.
    logdet_S : Array, scalar
        ``log det S`` on the **prior** ``P``, from the same Cholesky factor that
        produces ``nis`` — so the pair ``(nis, logdet_S)`` is a complete Gaussian NLL,
        ``0.5 (nis + logdet_S)``, with no second numerical path.  Exposed for
        ContactNet's β-NLL objective, which needs the ``logdet`` term to constrain the
        *absolute* scale of ``S``; the quadratic term alone fixes only ratios.  NaN
        before any update, like ``nis``.
    """
    applied: Array
    nis: Array
    condition_proxy: Array
    correction_rotation_norm: Array
    logdet_S: Array


def no_update_diagnostics() -> UpdateDiagnostics:
    """Diagnostics before any update has run — ``nis`` is **NaN**.

    Java initialises NIS to NaN so a never-updated value cannot read as "in-band";
    `testNormalizedInnovationSquaredIsNaNBeforeAnyUpdate` locks it.
    """
    return UpdateDiagnostics(
        applied=jnp.array(0.0),
        nis=jnp.array(jnp.nan),
        condition_proxy=jnp.array(jnp.nan),
        correction_rotation_norm=jnp.array(jnp.nan),
        logdet_S=jnp.array(jnp.nan),
    )


def linear_update(
    state: InEKFState,
    H: Array,
    residual: Array,
    R: Array,
    cond_max: float | None = None,
    gate: Array | float = 1.0,
) -> tuple[InEKFState, UpdateDiagnostics]:
    r"""Generic linear measurement update — Java ``InvariantUpdater.update``.

    The one code path every update in the filter goes through (contact FK, gravity
    leveling, and anything the orchestrator adds), so they cannot drift apart::

        S  = H P Hᵀ + R                       (symmetrised)
        K  = P Hᵀ S⁻¹                         (Cholesky solve — never `inv`)
        X̂⁺ = exp(−(Kν)^∧) X̂                   (I5)
        P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ    (Joseph — mandatory)

    ``H`` is ``(z, 3N+9)``, ``residual`` ``(z,)``, ``R`` ``(z, z)`` in the same frame as
    the residual.  ``cond_max`` defaults to ``inekf.cond_max`` in the config; ``gate``
    is an external mask (e.g. a quasi-static gate).

    Gating: the conditioning proxy ``(max L_ii / min L_ii)²`` of ``S`` and the
    caller's ``gate`` multiply into ``K``.  A gated update therefore leaves ``(X̂, P)``
    **bit-for-bit unchanged** rather than latching a bad correction — which is what
    `testSingularInnovationIsSkippedNotLatched` requires — and it does so without a
    data-dependent Python branch (I7).
    """
    if cond_max is None:
        cond_max = section("inekf")["cond_max"]

    S = H @ state.P @ H.T + R
    S = 0.5 * (S + S.T)
    factor = cho_factor(S)

    diag = jnp.abs(jnp.diag(factor[0]))
    condition_proxy = (jnp.max(diag) / jnp.min(diag)) ** 2
    # log det S = 2 Σ log L_ii — free from the factor already computed, so it
    # cannot drift from `nis` the way a separate `slogdet` call could.
    logdet_S = 2.0 * jnp.sum(jnp.log(diag))

    K = cho_solve(factor, H @ state.P).T           # P Hᵀ S⁻¹
    nis = residual @ cho_solve(factor, residual)   # prior P, prior residual

    applied = jnp.asarray(gate, dtype=jnp.float64) * (
        condition_proxy < cond_max
    ).astype(jnp.float64)
    K = applied * K

    xi = K @ residual
    corrected = apply_correction(state, xi)
    P_new = joseph_update(state.P, K, H, R)

    return (
        corrected._replace(P=P_new),
        UpdateDiagnostics(
            applied=applied,
            nis=nis,
            condition_proxy=condition_proxy,
            correction_rotation_norm=jnp.linalg.norm(xi[0:3]),
            logdet_S=logdet_S,
        ),
    )


# ContactUpdater seams (Java `ContactUpdater`): the single-contact entry points the
# Java class exposes and the ported `ContactUpdaterTest` exercises directly (I10 —
# the test seams ARE the public surface).  `correct` is the vectorised hot path.

def contact_jacobian(N: int, contact_index: int) -> Array:
    r"""Single-contact observation Jacobian ``H_i = [ 0_{3x6} | +I_3 | … −I_3 (own d_i block) … ]``, ``(3, 3N+9)``.

    **State-independent by construction**: the argument is the contact *index*, not
    the state, which is exactly the property
    `testJacobianStructureAndStateIndependence` asserts (it calls the Java form with
    two different random states and demands bit-equality).  `IndexError` if the index
    is out of range.
    """
    _check_contact_index(contact_index, N)
    H = jnp.zeros((3, 3 * N + 9))
    H = H.at[:, 6:9].set(jnp.eye(3))
    j = contact_tangent_index(contact_index)
    return H.at[:, j:j + 3].set(-jnp.eye(3))


def contact_residual(state: InEKFState, contact_index: int, y: Array) -> Array:
    r"""Single-contact world residual ``r = R̂ y − (d̂_i − p̂)``, ``(3,)`` — the per-contact slice of `innovation`."""
    _check_contact_index(contact_index, state.N)
    return state.R @ y - (state.d[contact_index] - state.p)


def rotate_measurement_covariance(state: InEKFState, body_cov: Array) -> Array:
    r"""Body-frame measurement covariance → world: ``R̂ N R̂ᵀ``.

    The residual lives in the world frame (`contact_residual`), so the body-frame FK
    noise must be conjugated by the estimated attitude before it enters ``S``.
    """
    return state.R @ body_cov @ state.R.T


def map_encoder_noise(contact_jac: Array, joint_cov: Array) -> Array:
    r"""``(3, n)`` contact Jacobian at ``q̂`` and ``(n, n)`` joint covariance → ``(3,3)`` **body-frame** FK noise ``J Σ_q Jᵀ``.

    `rotate_measurement_covariance` then takes it to world.
    """
    return contact_jac @ joint_cov @ contact_jac.T


def contact_update(
    state: InEKFState,
    contact_index: int,
    measurement: Array,
    body_covariance: Array,
    learned: bool = False,
) -> tuple[InEKFState, Array, UpdateDiagnostics]:
    r"""One single-contact FK update: residual → rotate noise to world → `linear_update`.

    ``measurement`` ``(3,)`` is the body-frame FK vector ``y_i``; ``body_covariance``
    ``(3,3)`` its body-frame covariance (e.g. from `map_encoder_noise`).  Returns
    ``(state, prior residual, diagnostics)``.  Going through the shared update path is
    what makes the EKF-delegation tests bit-exact.

    ``learned=True`` raises `NotImplementedError` while ContactNet is unlanded — the
    Java contract raises `NotImplementedException` and the ported test asserts it.
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

    updated, diagnostics = linear_update(state, H, residual, Nmat)
    return updated, residual, diagnostics


def correct(
    state: InEKFState,
    y: Array,
    Np: Array,
    params: InEKFParams,
) -> tuple[InEKFState, Array]:
    r"""One full FK measurement update: innovation → gain → Joseph → exp.

    ``y`` ``(N, 3)`` measured body-frame FK vectors, ``Np`` ``(N, 3, 3)`` the
    per-contact position FK covariances assembled upstream.  Returns the corrected
    state and the ``(3N,)`` innovation (emitted for NIS / ContactNet features).
    ``H`` is the precomputed constant (`InEKFParams.H`); nothing here is rebuilt per
    step.  With ``N = 0`` this is a static early return (``N`` is not traced).
    """
    if state.N == 0:                              # static: no contacts, no update
        return state, jnp.zeros(0)

    H = params.H
    nu = innovation(state, y)                     # (3N,)
    Nmat = measurement_noise(Np)                  # (3N, 3N)

    corrected, _ = linear_update(state, H, nu, Nmat)
    return corrected, nu
