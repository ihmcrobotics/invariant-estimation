"""Compose the existing JointKF and InEKF ticks without copying their math.

ModelInputs must be supplied in the same convention as jointKF.filter.step;
this adapter is not a robot-log decoder or a Java-policy parity claim.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..jointKF import filter as joint_filter
from ..inEKF import filter as base_filter
from .noise import apply_joint_noise, apply_inekf_noise, apply_inekf_inputs


class TwoStageCarry(NamedTuple):
    joint: joint_filter.FilterCarry
    base: base_filter.InEKFCarry


class TwoStageInputs(NamedTuple):
    sensors: joint_filter.SensorInputs
    model: joint_filter.ModelInputs
    accel_body: jax.Array  # already accel-bias corrected, in pelvis/body frame
    contact_chol: jax.Array  # external frozen heuristic/provider, not GT labels


class TwoStageOutputs(NamedTuple):
    joint_state: object
    joint_diagnostics: joint_filter.TickDiagnostics
    base: base_filter.InEKFOutputs


def make_step(theta, spec, build, joint_params, ekf, kinematics, imu_to_body):
    """Build inside the differentiated loss, not once with a fixed theta.

    imu_to_body is explicitly the base gyro measurement-frame -> pelvis/body
    rotation. The base gyro bias is estimated in that measurement frame, so
    subtraction MUST precede rotation. Accelerometer bias correction remains
    the caller's responsibility; JointKF estimates gyro bias only.
    """
    spec.validate_build(build)
    imu_to_body = jnp.asarray(imu_to_body, dtype=jnp.float64)
    if imu_to_body.shape != (3, 3):
        raise ValueError("imu_to_body must be a 3x3 rotation")
    scales = spec.scales(theta)
    learned_build = apply_joint_noise(build, scales)
    base_step = base_filter.make_step(apply_inekf_noise(ekf, scales), kinematics)

    def step(carry, inputs):
        joint, diag = joint_filter.step(
            carry.joint, inputs.sensors, inputs.model, learned_build, joint_params
        )
        n = build.n_joints
        bias = joint.state.x[2*n + 3*build.base_imu:2*n + 3*build.base_imu + 3]
        raw = inputs.sensors.gyros[build.base_imu]
        observations = base_filter.InEKFInputs(
            omega=imu_to_body @ (raw - bias),
            accel=inputs.accel_body,
            raw_omega=imu_to_body @ raw,
            joint=base_filter.JointFilterOutput(
                q=joint.state.x[:n], q_dot=joint.state.x[n:2*n],
                sigma_q=joint.state.P[:n, :n],
                sigma_q_dot=joint.state.P[n:2*n, n:2*n],
            ),
            contact_chol=inputs.contact_chol,
        )
        base, outputs = base_step(carry.base, apply_inekf_inputs(observations, scales))
        return TwoStageCarry(joint, base), TwoStageOutputs(joint.state, diag, outputs)

    return step


def run(theta, spec, build, joint_params, ekf, kinematics, imu_to_body, carry, inputs):
    """Differentiable complete trajectory; independent sessions need fresh carries."""
    return jax.lax.scan(
        make_step(theta, spec, build, joint_params, ekf, kinematics, imu_to_body),
        carry, inputs,
    )
