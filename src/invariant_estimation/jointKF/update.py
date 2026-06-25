"""
joint_kf/update.py
==================
Measurement-correction (update) step of the joint-chain KF, with the
**Joseph-form** covariance update (CLAUDE.md §4):

    ν  = z − H x⁻                              (pre-fit innovation)
    S  = H P⁻ Hᵀ + R                           (innovation covariance)
    K  = P⁻ Hᵀ S⁻¹                             (gain)
    x⁺ = x⁻ + K ν
    P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ        (Joseph form)

Why Joseph, not (I−KH)P⁻ (CLAUDE.md §4): `H` carries the linearized relative-FK
Jacobian `J^k_q̇(q̂)`, so the gain is computed from an *approximate* `H` and is
never exactly optimal.  The short form is exact only at the optimal gain; the
Joseph form is the honest `LΣLᵀ` transform of the posterior error over both error
sources, so it is correct for whatever `K` was applied and symmetric-PSD by
construction.

`update` is a pure linear-algebra step over `(state, z, H, R)`, decoupled from how
the measurement was built: `filter.py` assembles `z, H` via `measurement.py` and
`R` via `noise.build_R`, then calls this.  Pre-filter invariant (§0/§8): this
correction is internal to the joint KF — nothing here feeds InEKF propagation.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .state import JointKFState, split_x


class UpdateInfo(NamedTuple):
    """Transient byproducts of one update step (not stored in the state).

    The posterior covariance `P⁺` and its marginals (`Σ_q`, `Σ_q̇`, `b̂_ω`) live on
    the returned `JointKFState`; only the quantities that would otherwise be lost
    are surfaced here.

    Attributes
    ----------
    nu : Array, shape (n+3m,)
        Pre-fit innovation `z − H x⁻`.  `nu[n:]` is the per-pair IMU innovation
        whose magnitude `‖ν_ω‖` is a ContactNet kinematic-consistency feature
        (CLAUDE.md §6).
    S : Array, shape (n+3m, n+3m)
        Innovation covariance `H P⁻ Hᵀ + R`.  Enables a normalized-innovation
        squared `NIS = νᵀ S⁻¹ ν` for scale-invariant gating/trust downstream.
    """
    nu: Array
    S: Array


def update(
    state: JointKFState,
    z: Array,
    H: Array,
    R: Array,
) -> tuple[JointKFState, UpdateInfo]:
    """Apply one measurement correction; return `(state⁺, UpdateInfo)`.

    Parameters
    ----------
    state : JointKFState
        Predicted (prior) estimate `(x⁻, P⁻)`.
    z : Array, shape (n+3m,)
        Stacked measurement `[q̃ ; z_ω]` (`measurement.build_z`).
    H : Array, shape (n+3m, 2n+3m)
        Measurement Jacobian (`measurement.build_H`).
    R : Array, shape (n+3m, n+3m)
        Measurement noise `blkdiag(R_enc, R_ω)` (`noise.build_R`).

    Returns
    -------
    (JointKFState, UpdateInfo)
        Posterior estimate `(x⁺, P⁺)` and the transient `(ν, S)`.
    """
    x = state.x
    P = state.P
    n = state.n_joints
    dim = x.shape[0]

    nu = z - H @ x
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)                       # symmetrize before factoring

    # Gain K = P⁻ Hᵀ S⁻¹ without forming S⁻¹.  S is SPD (R is PD, H P⁻ Hᵀ is PSD),
    # so solve via Cholesky:  K = (S⁻¹ (P⁻ Hᵀ)ᵀ)ᵀ.
    PHt = P @ H.T                             # (dim, dim_z)
    cho = jax.scipy.linalg.cho_factor(S)
    K = jax.scipy.linalg.cho_solve(cho, PHt.T).T #NOTE: cholesky solves makes it significantly faster, we might also want to think about making this SR variant if it's slow

    x_new = x + K @ nu

    # Joseph form — symmetric-PSD by construction (sum of two LΣLᵀ sandwiches).
    ImKH = jnp.eye(dim) - K @ H
    P_new = ImKH @ P @ ImKH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    q_hat, q_dot_hat, b_omega = split_x(x_new, n)
    new_state = JointKFState(q_hat=q_hat, q_dot_hat=q_dot_hat,
                             b_omega=b_omega, P=P_new)
    return new_state, UpdateInfo(nu=nu, S=S)
