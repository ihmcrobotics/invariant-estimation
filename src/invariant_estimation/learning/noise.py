"""Differentiable, bounded VARIANCE scales around a frozen baseline.

Joint gyro scales belong to raw IMUs, NOT independent pair blocks. Rebuilding
the existing L Sigma L.T stack preserves shared-IMU and anchor correlations.
Scale after the baseline's acquisition/flooring; no noise floor is relearned.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class NoiseScales(NamedTuple):
    imu_gyro: Array
    base_gyro_q: Array
    base_accel_q: Array
    contact_q: Array
    contact_fk_r: Array
    gravity_roll_r: Array
    gravity_pitch_r: Array


INEKF_CHANNELS = (
    "base_gyro_q", "base_accel_q", "contact_q", "contact_fk_r",
    "gravity_roll_r", "gravity_pitch_r",
)


@dataclass(frozen=True)
class NoiseSpec:
    """Static schema: IMU order must equal JointKFBuild.imu_names exactly.

    theta=0 gives scale=1. exp(b*tanh(theta/b)) bounds every variance
    multiplier between 1/max_scale and max_scale without a hard clip.
    Arms 4/5/6/7 train neither/joint/inekf/both groups respectively.
    """
    imu_names: tuple[str, ...]
    arm: int = 7
    max_scale: float = 100.0

    def __post_init__(self):
        object.__setattr__(self, "imu_names", tuple(self.imu_names))
        if not self.imu_names or any(not n for n in self.imu_names):
            raise ValueError("IMU names must be nonempty")
        if len(set(self.imu_names)) != len(self.imu_names):
            raise ValueError("duplicate IMU name")
        if self.arm not in (4, 5, 6, 7):
            raise ValueError("noise-learning arms are 4, 5, 6, 7")
        if not math.isfinite(self.max_scale) or self.max_scale <= 1:
            raise ValueError("max_scale must be finite and > 1")

    @property
    def names(self):
        return tuple(f"imu_gyro:{n}" for n in self.imu_names) + INEKF_CHANNELS

    def initial_theta(self):
        return jnp.zeros(len(self.names), dtype=jnp.float64)

    def scales(self, theta: Array) -> NoiseScales:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        if theta.shape != (len(self.names),):
            raise ValueError(f"theta must have shape {(len(self.names),)}")
        m = len(self.imu_names)
        mask = jnp.asarray(
            [self.arm in (5, 7)] * m + [self.arm in (6, 7)] * 6,
            dtype=jnp.float64,
        )
        bound = math.log(self.max_scale)
        values = jnp.exp(bound * jnp.tanh(theta * mask / bound))
        return NoiseScales(values[:m], *values[m:])

    def validate_build(self, build):
        if tuple(build.imu_names) != self.imu_names:
            raise ValueError("IMU order differs from noise schema; do not reorder silently")


def apply_joint_noise(build, scales: NoiseScales):
    """Retain the baseline anisotropy and ALL existing covariance assembly."""
    if scales.imu_gyro.shape != (build.n_imus,):
        raise ValueError("one variance scale per raw IMU is required")
    return build._replace(
        gyro_sigma=jnp.asarray(build.gyro_sigma) * scales.imu_gyro[:, None, None]
    )


def apply_inekf_noise(ekf, scales: NoiseScales):
    """Scale process/gravity noise; do not alter gates, dt, H, Phi or priors.

    sigma_c is consumed by the low-level ekf.predict API. The trajectory driver
    consumes contact_chol instead, which apply_inekf_inputs scales separately.
    """
    return ekf._replace(
        params=ekf.params._replace(
            gyro_var=ekf.params.gyro_var * scales.base_gyro_q,
            accel_var=ekf.params.accel_var * scales.base_accel_q,
        ),
        sigma_c=ekf.sigma_c * scales.contact_q,
        gravity_params=ekf.gravity_params._replace(
            roll_var=ekf.gravity_params.roll_var * scales.gravity_roll_r,
            pitch_var=ekf.gravity_params.pitch_var * scales.gravity_pitch_r,
        ),
    )


def apply_inekf_inputs(inputs, scales: NoiseScales):
    """FK R := s_R J Sigma_q J.T, contact Q factor := sqrt(s_Q) L.

    Scaling a correction-only COPY of Sigma_q uses the existing Jacobian path;
    the JointKF's actual covariance and its reported outputs are not modified.
    This is a global FK covariance multiplier, not an additive constant R.
    Contact covariance floors in contact.digest stay fixed.
    Works with one tick or a leading time/batch dimension.
    """
    return inputs._replace(
        joint=inputs.joint._replace(
            sigma_q=inputs.joint.sigma_q * scales.contact_fk_r
        ),
        contact_chol=inputs.contact_chol * jnp.sqrt(scales.contact_q),
    )


def run_inekf(theta, spec, ekf, kinematics, state, inputs):
    """BPTT through the EXISTING scan, with theta a dynamic JAX argument."""
    from ..inEKF.filter import run
    scales = spec.scales(theta)
    return run(apply_inekf_noise(ekf, scales), kinematics, state,
               apply_inekf_inputs(inputs, scales))
