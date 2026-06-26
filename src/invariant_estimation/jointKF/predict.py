"""
joint_kf/predict.py
===================
EKF time-update (predict step) for the joint-chain KF (CLAUDE.md §4):

    x⁻ = F x
    P⁻ = F P Fᵀ + Q_d(M(q̂))

`F` (exact nilpotent transition) and `Q_d` (exact Van Loan process noise) come
from `noise.py`, so this step carries **no approximation** — the only EKF
nonlinearity in the filter lives later in `measurement.py`.

Pre-filter invariant (CLAUDE.md §0/§8): this propagation is internal to the joint
KF.  Nothing here ever feeds the InEKF SE₂(3) propagation `Φ` / `Q̄`.
"""
from jax import Array

from . import noise
from .state import JointKFParams, JointKFState, split_x


def predict(
    state: JointKFState,
    params: JointKFParams,
    M: Array | None = None,
) -> JointKFState:
    """Propagate the state mean and covariance one timestep.

    Parameters
    ----------
    state : JointKFState
        Prior estimate (x, P).
    params : JointKFParams
        Filter parameters (provides dt and the process-noise scales).
    M : Array of shape (n, n), optional
        Mass matrix M(q̂) for the inertia-shaped process noise
        `Q_a = σ_τ² M⁻²`.  Pass `None` (default) to use the diagonal early-dev
        stand-in `Q_a = σ_acc² I_n`.  The caller (filter.py) sources `M` from a
        `robot.RobotModel`; this function stays pure and array-only.

    Returns
    -------
    JointKFState
        Predicted estimate (x⁻, P⁻).  Mean: positions advance by Δt·q̇, while
        velocity and the residual bias are unchanged (F is identity on those
        blocks); only their covariance grows via `Q_d`.
    """
    n, m = state.n_joints, state.n_pairs

    F = noise.build_F(n, m, params.dt)
    Q_d = noise.build_process_noise(params, n, m, M)

    x_pred = F @ state.x
    P_pred = F @ state.P @ F.T + Q_d
    # Symmetrize to suppress float round-off drift; F P Fᵀ + Q_d is symmetric in
    # exact arithmetic (covariance LΣLᵀ discipline, CLAUDE.md §7).
    P_pred = 0.5 * (P_pred + P_pred.T)

    q_hat, q_dot_hat, b_omega = split_x(x_pred, n)
    return JointKFState(q_hat=q_hat, q_dot_hat=q_dot_hat, b_omega=b_omega, P=P_pred)
