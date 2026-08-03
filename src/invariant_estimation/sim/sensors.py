"""The plant → estimator boundary: MuJoCo `MjData` → `FusedSensors`.

Everything here is plain NumPy and runs OUTSIDE the jitted step, so it may branch
and allocate freely (I7 constrains the filter, not the sensor harness).

Three conventions are load-bearing and each is checked by a test in
`tests/sim/test_sensors.py`:

1. **A MuJoCo `gyro` sensor reports in its SITE frame**, which is exactly the
   estimator's per-IMU "own measurement frame" (`FusedSensors.gyros`). No
   rotation is applied here — `R_mount` at the fused boundary does that, and
   rotating twice was the frame bug the G9 session already paid for once.
2. **A MuJoCo `accelerometer` reports SPECIFIC FORCE** in the site frame: at rest
   it reads `+9.81` along the site's up axis, not zero. That is precisely what
   `FusedSensors.accel_base` wants ("gravity is added inside the InEKF
   propagation — feed it as read").
3. **The contact signal is one tick delayed by the filter, not by us.** We hand
   `jkf.step` this tick's trust and it stores it for the next tick
   (`jointKF/filter.py` §1). Delaying it here too would double-delay it.

The sim's advantage over the robot is a real normal force per foot, so the
contact probability is `f_n / (0.5 · m·g)` rather than a foot-switch voltage; the
Schmitt/dwell state machine on top of it uses the Java thresholds from
`config/filter_cfg.yaml` (`contact_trust`), which is the part that ports back.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Sequence

import mujoco
import numpy as np

__all__ = ["add_imu_sensors", "SimSensorReader", "ContactTrust", "IMUNoise"]


# ---------------------------------------------------------------------------
# MJCF: put real IMUs on the estimator's sites
# ---------------------------------------------------------------------------

def add_imu_sensors(root: ET.Element, imu_sites: Sequence[str]) -> ET.Element:
    """Append a `<sensor>` block with a gyro + accelerometer on each IMU site.

    Sensors are massless and stateless: adding them cannot change the dynamics,
    so a model with and without them integrates identically (asserted by
    `test_sensors_do_not_change_the_dynamics`).
    """
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for site in imu_sites:
        ET.SubElement(sensor, "gyro", {"name": f"gyro_{site}", "site": site})
        ET.SubElement(sensor, "accelerometer", {"name": f"acc_{site}", "site": site})
    return root


# ---------------------------------------------------------------------------
# Sensor corruption (optional)
# ---------------------------------------------------------------------------

@dataclass
class IMUNoise:
    """Additive sensor corruption, so the filter has something to actually do.

    Defaults are the hardware scale: gyro white noise at the config's
    `sigma_gyro_floor` (1e-6 (rad/s)² ⇒ 1e-3 rad/s), encoder noise at the
    measured per-joint values (~2e-4 rad), and a CONSTANT per-IMU gyro bias — the
    quantity the joint KF exists to estimate and hand to the InEKF (I1).
    """

    gyro_std: float = 1.0e-3          # [rad/s]
    accel_std: float = 3.0e-2         # [m/s^2]
    encoder_std: float = 2.0e-4       # [rad]
    encoder_vel_std: float = 5.0e-3   # [rad/s]
    gyro_bias_std: float = 1.0e-2     # [rad/s] one draw per IMU, then constant
    torque_std: float = 5.0e-1        # [N.m] not hardware-grounded; ~1% standing knee torque
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _bias: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def bias(self, n_imus: int) -> np.ndarray:
        """The per-IMU constant gyro bias, `(m, 3)`. Drawn once, then frozen."""
        if self._bias is None:
            self._bias = self.gyro_bias_std * self._rng.standard_normal((n_imus, 3))
        return self._bias

    def corrupt_gyros(self, gyros: np.ndarray) -> np.ndarray:
        return (gyros + self.bias(gyros.shape[0])
                + self.gyro_std * self._rng.standard_normal(gyros.shape))

    def corrupt_accel(self, accel: np.ndarray) -> np.ndarray:
        return accel + self.accel_std * self._rng.standard_normal(accel.shape)

    def corrupt_encoders(self, q: np.ndarray) -> np.ndarray:
        return q + self.encoder_std * self._rng.standard_normal(q.shape)

    def corrupt_velocities(self, qd: np.ndarray) -> np.ndarray:
        return qd + self.encoder_vel_std * self._rng.standard_normal(qd.shape)

    def corrupt_torques(self, tau: np.ndarray) -> np.ndarray:
        return tau + self.torque_std * self._rng.standard_normal(tau.shape)


# ---------------------------------------------------------------------------
# Contact trust
# ---------------------------------------------------------------------------

@dataclass
class ContactTrust:
    """Schmitt trigger + on-ground dwell per foot (the Java thresholds).

    `enter`/`stay` are `contact_trust.schmitt_enter` / `schmitt_stay` from the
    config and act on a normalised load `p = f_n / (0.5·m·g)`; `dwell` is the
    sustained-high time required before a foot is trusted. Release is immediate:
    a foot that unloads must stop anchoring the same tick, whereas a foot that
    lands must prove it (the asymmetry is the point of the debounce — a bouncing
    touchdown that anchors early poisons the bias gauge).
    """

    n_feet: int
    dt: float
    enter: float = 0.35
    stay: float = 0.25
    dwell: float = 0.04
    ema_tau: float = 0.01
    p: np.ndarray = field(init=False)
    trusted: np.ndarray = field(init=False)
    high_time: np.ndarray = field(init=False)

    def __post_init__(self):
        self.p = np.zeros(self.n_feet)
        self.trusted = np.zeros(self.n_feet)
        self.high_time = np.zeros(self.n_feet)

    def update(self, load: np.ndarray) -> np.ndarray:
        """Advance one tick on the normalised per-foot load; return the trust mask."""
        a = self.dt / max(self.ema_tau, self.dt)
        self.p += min(a, 1.0) * (np.asarray(load, float) - self.p)
        # Schmitt: a trusted foot holds down to `stay`, an untrusted one must clear `enter`.
        high = np.where(self.trusted > 0.0, self.p > self.stay, self.p > self.enter)
        self.high_time = np.where(high, self.high_time + self.dt, 0.0)
        self.trusted = np.where(
            self.trusted > 0.0,
            high.astype(float),                          # release immediately
            (self.high_time >= self.dwell).astype(float),  # enter only after the dwell
        )
        return self.trusted.copy()


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

class SimSensorReader:
    """Index maps from a sim `MjModel` to one `FusedSensors` per call.

    Every lookup is by NAME and resolved once here, never by assuming the sim
    model and the estimator model share index order — they are built from the
    same MJCF but the sim adds a floor, collision geoms, actuators and visual
    meshes, and an index coincidence that holds today is not a contract.
    """

    def __init__(
        self,
        m: mujoco.MjModel,
        fused,
        *,
        foot_geoms: Sequence[str],
        dt: float,
        noise: IMUNoise | None = None,
        stance_chol: float = 1.0e-4,
        swing_chol: float = 1.0e1,
    ):
        self.m = m
        self.fused = fused
        self.noise = noise
        self.stance_chol = stance_chol
        self.swing_chol = swing_chol

        build = fused.build
        self.imu_names = tuple(build.imu_names)
        self.n_imus = len(self.imu_names)
        self.base_imu = int(build.base_imu)

        def sid(name, obj):
            i = mujoco.mj_name2id(m, obj, name)
            if i < 0:
                raise KeyError(f"no {obj} named {name!r} in the sim model")
            return i

        # -- IMU sensor addresses (site frame, see module docstring) ---------
        self.gyro_adr = np.array(
            [m.sensor_adr[sid(f"gyro_{s}", mujoco.mjtObj.mjOBJ_SENSOR)] for s in self.imu_names])
        self.acc_adr = np.array(
            [m.sensor_adr[sid(f"acc_{s}", mujoco.mjtObj.mjOBJ_SENSOR)] for s in self.imu_names])

        # -- encoders: the 9 filtered joints, in filter state order ----------
        self.enc_qadr = np.array(
            [m.jnt_qposadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in build.joint_names])

        # -- the unfiltered anchor-chain joints (Alex's 4 ankles) ------------
        self.unfiltered_names = _dof_joint_names(
            fused.model.mj_model, np.asarray(build.dof_anchor_unfiltered, dtype=int))
        self.unf_dofadr = np.array(
            [m.jnt_dofadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in self.unfiltered_names],
            dtype=int)
        self.unf_qadr = np.array(
            [m.jnt_qposadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in self.unfiltered_names],
            dtype=int)

        # -- contact --------------------------------------------------------
        self.foot_gids = np.array(
            [sid(g, mujoco.mjtObj.mjOBJ_GEOM) for g in foot_geoms], dtype=int)
        self.weight = float(m.body_mass.sum()) * 9.81
        self.trust = ContactTrust(n_feet=len(self.foot_gids), dt=dt)

        # -- ground truth, for scoring ---------------------------------------
        self.base_bid = sid("PELVIS_LINK", mujoco.mjtObj.mjOBJ_BODY)
        self.base_site = sid("base_body", mujoco.mjtObj.mjOBJ_SITE)

    # -- pieces ------------------------------------------------------------

    def foot_loads(self, d: mujoco.MjData) -> np.ndarray:
        """Normalised per-foot normal load, `f_n / (0.5·m·g)`, clipped to [0, 1]."""
        f = np.zeros(len(self.foot_gids))
        frc = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for k, gid in enumerate(self.foot_gids):
                if c.geom1 == gid or c.geom2 == gid:
                    mujoco.mj_contactForce(self.m, d, i, frc)
                    f[k] += abs(frc[0])
        return np.clip(f / (0.5 * self.weight), 0.0, 1.0)

    def read(self, d: mujoco.MjData):
        """One `FusedSensors` (NumPy leaves) from the current `MjData`."""
        from ..pipeline.main_estimator import FusedSensors

        gyros = d.sensordata[self.gyro_adr[:, None] + np.arange(3)].copy()
        accel = d.sensordata[self.acc_adr[self.base_imu] + np.arange(3)].copy()
        enc = d.qpos[self.enc_qadr].copy()
        enc_vel = d.qvel[self.enc_dofadr].copy()
        qd_u = d.qvel[self.unf_dofadr].copy()
        # Only read when the estimator was built with `contact_fk_unfiltered`; an empty array
        # otherwise, which is the "field absent" encoding `FusedSensors` expects.
        q_u = (d.qpos[self.unf_qadr].copy() if self.fused.n_aux else np.zeros(0))
        # ContactNet feature channel only -- the estimator never reads it. `qfrc_actuator` is in
        # GENERALISED coords, so it indexes by dofadr and lines up with the encoder ordering.
        # Ordered concat(filtered, unfiltered), matching how the contact FK widens q̂.
        tau = d.qfrc_actuator[np.concatenate([self.enc_dofadr, self.unf_dofadr])].copy()
        if self.noise is not None:
            gyros = self.noise.corrupt_gyros(gyros)
            accel = self.noise.corrupt_accel(accel)
            enc = self.noise.corrupt_encoders(enc)
            enc_vel = self.noise.corrupt_velocities(enc_vel)
            qd_u = self.noise.corrupt_velocities(qd_u)
            q_u = self.noise.corrupt_encoders(q_u)
            tau = self.noise.corrupt_torques(tau)

        trusted = self.trust.update(self.foot_loads(d))
        # The InEKF has NO contact mask: contact condition rides ENTIRELY in
        # Sigma_C (`inEKF/filter.py`, the DECISION note). A swing foot therefore
        # needs a LARGE factor here, or the filter keeps believing it is planted.
        chol = np.where(trusted[:, None, None] > 0.0, self.stance_chol, self.swing_chol)
        return FusedSensors(
            encoders=enc,
            encoders_vel=enc_vel,
            gyros=gyros,
            accel_base=accel,
            qd_unfiltered=qd_u,
            contact=trusted,
            contact_chol=chol * np.tile(np.eye(3), (len(self.foot_gids), 1, 1)),
            q_unfiltered=q_u,
            torques=tau,
        )

    # -- ground truth --------------------------------------------------------

    def truth(self, d: mujoco.MjData) -> dict:
        """The sim's own answer to what the estimator is estimating.

        `omega` uses `mjOBJ_XBODY`: with `mjOBJ_BODY` MuJoCo resolves the twist in
        the body's INERTIAL frame, which on Alex's pelvis permutes the axes — the
        bug that made every policy fall (`run_policy.build_obs`, EXPERIMENTS.md).
        """
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_XBODY, self.base_bid, v6, 1)
        R = d.xmat[self.base_bid].reshape(3, 3).copy()
        return {
            "R": R,
            "omega": v6[:3].copy(),                 # body frame
            "v": (R @ v6[3:]).copy(),               # world frame, to match the InEKF state
            "p": d.xpos[self.base_bid].copy(),
            "q": d.qpos[self.enc_qadr].copy(),
            "q_dot": d.qvel[[self.m.jnt_dofadr[mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.fused.build.joint_names]].copy(),
        }


def _dof_joint_names(mj_model: mujoco.MjModel, dofs: np.ndarray) -> tuple[str, ...]:
    """Joint names owning the given DoF indices, in the order given."""
    out = []
    for dof in dofs:
        j = int(np.flatnonzero(mj_model.jnt_dofadr == int(dof))[0])
        out.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j))
    return tuple(out)
