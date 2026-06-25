"""
jointKF — linear joint-chain Kalman filter (pre-filter).

A bias-augmented EKF that fuses joint encoders with distributed-IMU relative
angular velocity, producing honest joint estimates and covariances for the
downstream InEKF / ContactNet (see `CLAUDE.md` for the full design record).

Typical use
-----------
>>> from invariant_estimation.jointKF import init_state, default_params, run, SensorInputs
>>> state0 = init_state(n_joints, n_pairs)
>>> params = default_params(dt=1e-3)
>>> final, states, infos = run(state0, params, sensor_seq, robot)   # robot: RobotModel

The pipeline (state → noise → predict → measurement → update → filter) is laid out
one concern per module; `filter.step` / `filter.run` are the usual entry points.
The robot dynamics seam is `invariant_estimation.robot.RobotModel`, re-exported here.
"""
from ..robot import RobotModel
from .filter import SensorInputs, run, step
from .measurement import (
    build_H,
    build_measurement,
    build_z,
    relative_gyro_measurement,
)
from .noise import (
    acceleration_cov_diag,
    acceleration_cov_mass,
    build_F,
    build_process_noise,
    build_Q_d,
    build_R,
)
from .predict import predict
from .state import (
    JointKFParams,
    JointKFState,
    default_params,
    init_state,
    split_x,
)
from .update import UpdateInfo, update

__all__ = [
    # state + params
    "JointKFState",
    "JointKFParams",
    "init_state",
    "default_params",
    "split_x",
    # noise / model builders
    "build_F",
    "build_Q_d",
    "build_process_noise",
    "build_R",
    "acceleration_cov_diag",
    "acceleration_cov_mass",
    # predict
    "predict",
    # measurement
    "relative_gyro_measurement",
    "build_z",
    "build_H",
    "build_measurement",
    # update
    "update",
    "UpdateInfo",
    # filter (entry points)
    "SensorInputs",
    "step",
    "run",
    # robot seam
    "RobotModel",
]
