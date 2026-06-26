"""
joint_kf/filter.py
==================
Thin orchestrator for the joint-chain KF (CLAUDE.md §4/§7).  This is the single
layer that resolves a `robot.RobotModel` into the raw arrays the pure steps
consume, runs one `predict → update` cycle (`step`), and scans over a trajectory
(`run`, via `jax.lax.scan`) for ContactNet BPTT training.

Linearization points (EKF):
* the process noise `Q_d` is shaped by the mass matrix `M(q̂)` at the **prior**
  estimate — only `q̂` is available before `predict`;
* the IMU measurement Jacobian `J^k_q̇(q̂)` is evaluated at the **predicted** state
  `x⁻` — the standard EKF measurement-linearization point.

Pre-filter invariant (CLAUDE.md §0/§8): everything here is internal to the joint
KF; no output ever feeds the InEKF SE₂(3) propagation.

The whole package runs in float64 (see `invariant_estimation/__init__.py`).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from . import measurement, noise
from .predict import predict
from .state import JointKFParams, JointKFState
from .update import UpdateInfo, update
from ..robot import RobotModel


class SensorInputs(NamedTuple):
    """Per-step measurements fed to the filter.

    When passed to `run`, each field carries a leading time axis of length T
    (e.g. `q_tilde` is `(T, n)`); `step` consumes a single-step slice.

    Attributes
    ----------
    q_tilde : Array, shape (n,)
        Encoder position reading.
    omega_a, omega_b : Array, shape (m, 3)
        Per-pair Mahony-cleaned gyro readings, each in its own IMU frame.
    R_ba : Array, shape (m, 3, 3)
        Per-pair inter-IMU rotation `R^b_a` (a-frame → b-frame).
    """
    q_tilde: Array
    omega_a: Array
    omega_b: Array
    R_ba: Array


def step(
    state: JointKFState,
    params: JointKFParams,
    sensors: SensorInputs,
    robot: RobotModel | None = None,
) -> tuple[JointKFState, UpdateInfo]:
    """One `predict → update` cycle.

    Parameters
    ----------
    state : JointKFState
        Prior estimate.
    params : JointKFParams
        Filter parameters.
    sensors : SensorInputs
        This step's encoder + IMU measurements.
    robot : RobotModel, optional
        Source of `M(q̂)` and the relative-gyro Jacobian.  If `None`, the process
        noise falls back to the diagonal early-dev `Q_a` and the IMU rows observe
        only the residual bias (encoder-driven; typically used with `m = 0`).

    Returns
    -------
    (JointKFState, UpdateInfo)
        Posterior estimate and the update byproducts `(ν, S)`.
    """
    n, m = state.n_joints, state.n_pairs

    # Predict — Q_d shaped by M at the prior estimate (diagonal if no robot).
    M = None if robot is None else robot.mass_matrix(state.q_hat)
    pred = predict(state, params, M)

    # Update — linearize the IMU Jacobian about the predicted state x⁻.
    if robot is None:
        J_omega = jnp.zeros((m, 3, n))
    else:
        J_omega = robot.relative_gyro_jacobian(pred.q_hat)

    z, H = measurement.build_measurement(
        sensors.q_tilde, sensors.omega_a, sensors.omega_b, sensors.R_ba, J_omega, n
    )
    R = noise.build_R(params, n, m)
    return update(pred, z, H, R)


def run(
    state0: JointKFState,
    params: JointKFParams,
    sensor_seq: SensorInputs,
    robot: RobotModel | None = None,
) -> tuple[JointKFState, JointKFState, UpdateInfo]:
    """Filter a whole trajectory with `jax.lax.scan`.

    Parameters
    ----------
    state0 : JointKFState
        Initial estimate.
    params : JointKFParams
        Filter parameters (closure constant of the scan).
    sensor_seq : SensorInputs
        Measurements with a leading time axis of length T.
    robot : RobotModel, optional
        Closure constant (a static Python object whose methods trace to arrays).
        To `jit(run)`, capture `robot` (e.g. `functools.partial`) rather than
        passing it as a traced argument.

    Returns
    -------
    (final_state, states, infos)
        `final_state` is the carry after the last step; `states` and `infos` are
        the full per-step trajectories (each `JointKFState` / `UpdateInfo` field
        stacked along a new leading T axis).
    """
    def body(carry: JointKFState, sensors: SensorInputs):
        new_state, info = step(carry, params, sensors, robot)
        return new_state, (new_state, info)

    final_state, (states, infos) = jax.lax.scan(body, state0, sensor_seq)
    return final_state, states, infos
