"""`EstimatorRuntime` — the fused estimator driven from a running MuJoCo sim.

This is the estimator half of the closed loop and knows nothing about policies,
viewers or command handling: it takes `FusedSensors` in and publishes an estimate
out. `run_estimator.py` at the repo root joins it to `run_policy`'s sim loop.

Rate and phase
--------------
The estimator runs at the PHYSICS rate (one step per `mj_step`), the policy at
the control rate. One control tick is therefore

    [advance the estimator over the substeps that just happened]
    -> [build obs from the fresh estimate] -> [policy] -> [DECIMATION * mj_step]

so the estimate the policy reads is current, not one control period stale. The
substeps are advanced in ONE jitted `lax.scan` call rather than N Python-level
calls: same arithmetic, one dispatch (and the same constant graph, I7).

What the policy actually consumes
---------------------------------
`base_ang_vel` and `projected_gravity` — nothing else (the 98-term observation's
other entries are commands, raw encoders and the last action). Both come from the
parts of the estimator that are hardware-validated: the bias-corrected pelvis
gyro rotated into the body frame, and `R̂ᵀ·down` from the InEKF attitude which
gravity leveling keeps observable. Base position/velocity never reach the policy,
which is why contact-FK drift shows up in the scoring but not in the gait.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..pipeline import main_estimator as me

__all__ = ["EstimatorRuntime", "EstimateView"]

_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class EstimateView:
    """The estimator's published output for one tick, in NumPy.

    `omega_body` is recomputed here rather than read out of `FusedOutputs`,
    which does not carry it: it is the boundary quantity
    `R_mount · (gyro_base − b_base)` (`main_estimator._boundary`, I1). Doing it in
    NumPy from the published bias keeps the jitted step untouched.
    """

    R: np.ndarray            # (3,3) ^W R_B
    v: np.ndarray            # (3,)  world velocity
    p: np.ndarray            # (3,)  world position
    q: np.ndarray            # (n,)  filtered joint positions
    q_dot: np.ndarray        # (n,)
    bias: np.ndarray         # (3m,) per-IMU gyro bias, each in its own IMU frame
    omega_body: np.ndarray   # (3,)  bias-corrected base gyro, body frame
    nis: float               # InEKF contact-update NIS
    trusted: np.ndarray      # (K,)  the contact mask that was fed in

    @property
    def projected_gravity(self) -> np.ndarray:
        """`R̂ᵀ · down` — the policy's gravity term, from the ESTIMATED attitude."""
        return self.R.T @ _DOWN

    @property
    def rpy(self) -> np.ndarray:
        """Roll/pitch/yaw [rad] of `R`, for scoring."""
        return np.array([
            np.arctan2(self.R[2, 1], self.R[2, 2]),
            -np.arcsin(np.clip(self.R[2, 0], -1.0, 1.0)),
            np.arctan2(self.R[1, 0], self.R[0, 0]),
        ])


class EstimatorRuntime:
    """Owns the fused estimator, its carry, and the jitted multi-substep advance."""

    def __init__(self, fused, reader, *, substeps: int):
        self.fused = fused
        self.reader = reader
        self.substeps = int(substeps)
        self.n = fused.n_joints
        self.K = fused.n_contacts
        self.base_imu = int(fused.build.base_imu)
        self.R_mount = np.asarray(fused.R_mount)
        step = me.make_fused_step(fused)

        def advance(carry, sensors):
            return jax.lax.scan(step, carry, sensors)

        self._advance = jax.jit(advance)
        self.carry = None
        self.last: EstimateView | None = None

    # -- lifecycle ----------------------------------------------------------

    def seed(self, d, *, rotation=None, position=None):
        """Seed the carry from the sim's current state.

        The robot boots knowing its own attitude and where it is standing, so the
        pose is seeded from truth. Yaw is unobservable to this filter either way
        (`enableYawSeeding=false`), and a deliberately wrong seed is a separate
        experiment, not the default.
        """
        t = self.reader.truth(d)
        R0 = t["R"] if rotation is None else np.asarray(rotation)
        p0 = t["p"] if position is None else np.asarray(position)
        self.carry = me.init_fused_carry(
            self.fused,
            q0=jnp.asarray(t["q"], dtype=jnp.float64),
            rotation=jnp.asarray(R0, dtype=jnp.float64),
            position=jnp.asarray(p0, dtype=jnp.float64),
            # Read straight off `MjData`, not through `reader.read`: that call would
            # advance the contact-trust state machine a tick before the loop starts.
            q0_unfiltered=jnp.asarray(d.qpos[self.reader.unf_qadr], dtype=jnp.float64)
            if self.fused.n_aux else None,
        )
        return self.carry

    # -- per control tick ---------------------------------------------------

    def advance(self, batch) -> EstimateView:
        """Run the estimator over a list of `FusedSensors` (one per physics step)."""
        if self.carry is None:
            raise RuntimeError("call seed() before advance()")
        stacked = jax.tree.map(
            lambda *xs: jnp.asarray(np.stack(xs), dtype=jnp.float64), *batch)
        self.carry, out = self._advance(self.carry, stacked)
        self.last = self._view(out, batch[-1])
        return self.last

    def _view(self, out, last_sensors) -> EstimateView:
        """Last tick of a scanned `FusedOutputs` → NumPy `EstimateView`."""
        take = lambda a: np.asarray(a)[-1]                                   # noqa: E731
        bias = take(out.bias)
        b_base = bias[3 * self.base_imu: 3 * self.base_imu + 3]
        gyro_base = np.asarray(last_sensors.gyros)[self.base_imu]
        nis = np.asarray(out.inekf.contact_diagnostics.nis)
        return EstimateView(
            R=take(out.R), v=take(out.v), p=take(out.p),
            q=take(out.q), q_dot=take(out.q_dot), bias=bias,
            omega_body=self.R_mount @ (gyro_base - b_base),
            nis=float(nis[-1]) if nis.ndim else float(nis),
            trusted=np.asarray(last_sensors.contact),
        )


def attitude_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    c = (np.trace(R_est.T @ R_true) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def tilt_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Angle between the two estimates of the gravity direction in body frame.

    This is the error that reaches the policy (`projected_gravity`) and the one
    gravity leveling is responsible for; it excludes yaw, which this filter does
    not observe and the policy does not use.
    """
    a, b = R_est.T @ _DOWN, R_true.T @ _DOWN
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
