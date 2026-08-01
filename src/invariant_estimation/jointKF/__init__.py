"""jointKF — joint-space Kalman pre-filter (CLAUDE.md §1 deliverable 1, gates G6-G8 green).

A bias-augmented filter over ``x = [q ; q_dot ; b_omega]`` fusing joint encoders
with a *stacked* distributed-IMU relative-gyro measurement and stance anchors,
exporting honest joint estimates and covariances to the InEKF.  Design record:
`CLAUDE.md` in this package; authoritative spec: the repo-root `CLAUDE.md`.
`state.py` is the frozen contract.

Bias is **per-IMU** (`m` = distinct IMUs), not per-pair — invariant I6 requires
the exact ``L Sigma L^T`` cross-covariance on the shared-base-IMU star, and the
bias columns of ``H_g`` must *be* the mixing operator ``L``.
"""
from ..robot import RobotModel
from .state import (
    SEAM_MAP,
    JointKFBuild,
    JointKFParams,
    JointKFState,
    alpha_for_name,
    default_params,
    encoder_var_for_name,
    init_state,
    rotor_inertia_for_name,
    split_x,
)

__all__ = [
    "JointKFState",
    "JointKFParams",
    "JointKFBuild",
    "init_state",
    "default_params",
    "split_x",
    "rotor_inertia_for_name",
    "alpha_for_name",
    "encoder_var_for_name",
    "SEAM_MAP",
    "RobotModel",
]
