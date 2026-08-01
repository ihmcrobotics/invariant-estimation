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

__all__ = ["add_imu_sensors", "SimSensorReader", "ContactTrust", "EarlyRelease",
           "IMUNoise"]


def add_imu_sensors(root: ET.Element, imu_sites: Sequence[str]) -> ET.Element:
    """Append a `<sensor>` block with a gyro + accelerometer on each IMU site.

    Sensors are massless and stateless, so a model with and without them integrates identically
    (`test_sensors_do_not_change_the_dynamics`).
    """
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for site in imu_sites:
        ET.SubElement(sensor, "gyro", {"name": f"gyro_{site}", "site": site})
        ET.SubElement(sensor, "accelerometer", {"name": f"acc_{site}", "site": site})
    return root


@dataclass
class IMUNoise:
    """Additive sensor corruption, so the filter has something to actually do.

    Defaults are the hardware scale: gyro white noise at the config's
    `sigma_gyro_floor` (1e-6 (rad/s)² ⇒ 1e-3 rad/s), encoder noise at the
    measured per-joint values (~2e-4 rad), and a CONSTANT per-IMU gyro bias — the
    quantity the joint KF exists to estimate and hand to the InEKF (I1).

    `torque_std` is the exception: **not** grounded in hardware. Plain AWGN at 0.5 N·m,
    chosen so the ContactNet torque channel is not the one clean signal in an
    otherwise-corrupted bundle. For scale that is ~1% of Alex's standing knee torque
    (measured: −50.9 N·m), against 0.02–0.5% relative noise on the other channels.

    Revisit before trusting any absolute ContactNet calibration result: is the logged
    `tau` measured (a sensor, so this model is the right shape) or commanded (a
    controller output, which carries no sensor noise at all)?
    """

    gyro_std: float = 1.0e-3          # [rad/s]
    accel_std: float = 3.0e-2         # [m/s^2]
    encoder_std: float = 2.0e-4       # [rad]
    encoder_vel_std: float = 5.0e-3   # [rad/s]
    gyro_bias_std: float = 1.0e-2     # [rad/s] one draw per IMU, then constant
    torque_std: float = 5.0e-1        # [N.m] PLACEHOLDER — see below
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _bias: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def bias(self, n_imus: int) -> np.ndarray:
        """Per-IMU constant gyro bias, `(m, 3)`. Drawn once, then frozen."""
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
        """AWGN only — no bias term, unlike the gyro (see the class docstring)."""
        return tau + self.torque_std * self._rng.standard_normal(tau.shape)


@dataclass
class ContactTrust:
    """Schmitt trigger + on-ground dwell per foot (the Java thresholds).

    `enter`/`stay` are `contact_trust.schmitt_enter` / `schmitt_stay` from the config and act
    on a normalised load `p = f_n / (0.5·m·g)`; `dwell` is the sustained-high time required
    before a foot is trusted. Release is immediate, entry must be earned — a bouncing
    touchdown that anchors early poisons the bias gauge.
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


@dataclass
class EarlyRelease:
    r"""Causal anchor early-release: loosen ``Sigma_C`` while the foot is still loaded.

    **Why.** `docs/theory/anchor_release_timing.md` derives the vertical sink: at end of
    stance the foot unloads and rolls while `ContactTrust` still says "planted", so ``nu_z``
    turns positive while the apportionment fraction
    ``f = (P_pp - P_pd)/(Sigma_rel + N)`` is still at its tight stance value (~0.54) and the
    velocity gain is 2584x its swing value -- one rectified downward dose per liftoff per
    foot. **A threshold on load LEVEL, however low, is structurally late**; the anticipatory
    signal has to be the load's ratio to its own stance peak.

    The live counterpart of `experiments.process_socket_ablation.causal_early_release`
    (arm E'), whose non-causal ceiling is arm B (``loosen_early``, which reads liftoff from
    the future and is not deployable). Measured offline on `data/dr5` at N=4: baseline slope
    ``e_pz`` -0.01486 m/s, arm B at 100 ticks of lead -0.00344 (4.3x), arm E' at
    ``frac = 0.5`` -0.01101 (1.35x).

    Three design choices, each of which cost a failed arm to learn:

    * **The impact spike is blanked** (`blank_ticks`). Touchdown peak normal force is
      2.0-10.8x the stance median and lands 1-16% into the stance, so a running peak from
      tick 0 locks onto the impact, ``frac * peak`` sits above where the foot spends the rest
      of its stance, and the latch fires almost immediately -- 70-86% of stance released.
      That is a constant-loose anchor, not early release, and it scored monotonically WORSE
      than baseline. Still causal: the window looks backwards, never forwards.
    * **It only ever loosens.** The caller takes the max with the heuristic, so this moves
      liftoff earlier and can never move touchdown earlier.
    * **Latched within a stance.** Ground reaction force is double-humped; a mid-stance dip
      below ``frac * peak`` would otherwise release and re-tighten, chattering. Same
      fire-once logic as `inEKF/reseed.py`'s latch, for the same reason.

    ``peak_mode`` picks the reference stance. ``"current"`` is the running post-blank peak of
    the stance in progress: reproduces arm E' exactly, needs the blanking window and cannot
    act during it. ``"prev"`` is the previous completed stance's peak for this point, so the
    reference is available from tick 0 and does not depend on this stance's impact at all
    (falling back to ``"current"`` until one stance has completed). ``"prev"`` is the
    lead-VARIANCE lever -- arm E' at ``frac = 0.7`` had the better median lead and the worse
    score.

    ``rate_frac`` ORs in a falling-rate test: release when the load drops faster than
    ``rate_frac * peak`` per second. Zero disables it, which is the default and the
    configuration every recorded number was produced under.

    ``clock_lead`` ORs in a **stance clock**: release once this stance has run to
    ``prev_len - lead_ticks``. It is the only member of the family that can reproduce arm B's
    *fixed* lead, and it is here because of a measurement:

        Arm B (non-causal oracle, 100 ticks) gives every liftoff a lead of ~100 ticks, std
        23, and misses **0.7%** of them. Every load-threshold predictor measured on
        `data/dr5` -- level on the running peak, level on the previous stance's peak, rate,
        at fractions 0.35 to 0.85 -- misses **31-50%** of liftoffs, and so does a *perfect*
        contact-point-speed sensor at the same loose fraction (22% missed at 0.02 m/s). The
        lead arm B uses is not in the load or in the contact's motion at that instant; it is
        in the GAIT PLAN, which a walking controller has and a load sensor does not.

    So ``clock_lead`` is the causal predictor with the right *shape*, at the price of
    assuming stride-to-stride regularity: on a dataset containing standing (where a "stance"
    lasts thousands of ticks) it releases far too early. Set ``frac = 0`` for the clock alone.

    All state is plain NumPy and lives outside the jitted step (I7 constrains the
    filter, not the sensor harness).
    """

    n_points: int
    dt: float
    frac: float = 0.5
    blank_ticks: int = 150
    peak_mode: str = "current"
    rate_frac: float = 0.0
    clock_lead: int = 0
    off_dwell: int = 0
    """Consecutive UNLOADED ticks required to end a stance episode.

    **Measured, closed loop at vx = 0.6:** with ``off_dwell = 0`` a 30 s walk reports **306
    "liftoffs"** on two feet -- about 10/s against a real cadence near 2 steps/s. The per-foot
    normal load momentarily reads zero mid-stance (MuJoCo re-solves the contact set every step and
    a foot can have no qualifying contact for a tick or two), so a stance fragments into several
    episodes and every fragment resets the peak reference AND clears the latch -- exactly the
    state this mechanism needs to keep. A short off-dwell coalesces them.

    Asymmetric on purpose, in the same direction as `ContactTrust`: entering a stance is
    immediate, leaving it must be sustained -- here because a spurious *end* is the expensive
    error, whereas `ContactTrust` debounces the entry because a bouncing touchdown that anchors
    early poisons the bias gauge.

    ``0`` is the default and reproduces every number recorded before this field existed.
    """
    k: np.ndarray = field(init=False)
    peak: np.ndarray = field(init=False)
    prev_peak: np.ndarray = field(init=False)
    prev_len: np.ndarray = field(init=False)
    released: np.ndarray = field(init=False)
    lead_ticks: np.ndarray = field(init=False)
    leads: list = field(init=False)
    _on: np.ndarray = field(init=False)
    _off: np.ndarray = field(init=False)
    _last: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.peak_mode not in ("current", "prev"):
            raise ValueError(f"peak_mode must be 'current' or 'prev', got {self.peak_mode!r}")
        if not 0.0 <= self.frac < 1.0:
            raise ValueError(f"frac must be in [0, 1), got {self.frac}")
        if self.frac == 0.0 and self.clock_lead <= 0 and self.rate_frac <= 0.0:
            raise ValueError("every test is disabled (frac = rate_frac = clock_lead = 0); "
                             "the OFF switch is `SimSensorReader(early_release=0.0)`")
        n = self.n_points
        self.k = np.zeros(n, dtype=int)
        self.peak = np.zeros(n)
        self.prev_peak = np.zeros(n)
        self.prev_len = np.zeros(n, dtype=int)
        # Unloaded is unambiguously not planted, so the initial state is released.
        self.released = np.ones(n)
        self.lead_ticks = np.zeros(n, dtype=int)
        # One entry per completed stance: the LEAD, i.e. the length of the released run
        # that ended at liftoff. Not "the first release tick in the stance" -- a release
        # that fires and then re-tightens is worth nothing, because the anchor has to be
        # loose AT the moment the liftoff residual arrives, and the two metrics disagree
        # by 10-20 percentage points of miss rate on real data.
        self.leads = []
        self._on = np.zeros(n, dtype=bool)
        self._off = np.full(n, self.off_dwell + 1, dtype=int)
        self._last = np.zeros(n)

    def update(self, load: np.ndarray) -> np.ndarray:
        """Advance one tick on the per-point load; return the release mask (1 = loose).

        `load` must be UNCLIPPED -- a peak reference taken from a signal clipped at 1.0
        is not a peak, and at N=4 a fully-loaded toe reads ~1.0 (see `point_loads`).
        """
        f = np.asarray(load, float)
        raw_on = f > 0.0
        # Coalesce sub-`off_dwell` gaps in the load: a stance is still in progress until the
        # load has been absent for `off_dwell` consecutive ticks (see the field docstring).
        self._off = np.where(raw_on, 0, self._off + 1)
        on = raw_on | (self._on & (self._off <= self.off_dwell))
        fresh = on & ~self._on
        # Read BEFORE `k` is advanced: at the first unloaded tick `self.k` still holds the
        # last loaded index, so the stance length is `k + 1`.
        ending = self._on & ~on
        for c in np.flatnonzero(ending):
            self.leads.append(int(self.lead_ticks[c]))
        self.prev_len = np.where(ending, self.k + 1, self.prev_len)

        # A new stance retires the old peak into `prev_peak` and restarts the counter.
        self.prev_peak = np.where(fresh & (self.peak > 0.0), self.peak, self.prev_peak)
        self.k = np.where(fresh, 0, np.where(on, self.k + 1, 0))
        self.peak = np.where(fresh, 0.0, self.peak)
        self.released = np.where(fresh, 0.0, self.released)

        past = on & (self.k >= self.blank_ticks)
        self.peak = np.where(past, np.maximum(self.peak, f), self.peak)

        if self.peak_mode == "prev":
            # Use the previous stance where there is one; otherwise the running peak,
            # which is `"current"` behaviour and keeps the blanking guard.
            have = self.prev_peak > 0.0
            ref = np.where(have, self.prev_peak, self.peak)
            armed = on & (have | past)
        else:
            ref, armed = self.peak, past
        ref = np.maximum(ref, 1e-9)

        rel = armed & (f < self.frac * ref)
        if self.rate_frac > 0.0:
            drop = (self._last - f) / self.dt
            rel |= armed & (drop > self.rate_frac * ref)
        if self.clock_lead > 0:
            # The stance clock. Needs a completed stance for this point; until then the
            # level test is the only thing armed.
            rel |= on & (self.prev_len > 0) & (self.k >= self.prev_len - self.clock_lead)

        self.released = np.where(on, np.maximum(self.released, rel.astype(float)), 1.0)
        # Ticks this point has been released while still carrying load -- the LEAD, which
        # is the quantity the theory note says matters, and it is per-liftoff rather than
        # a median. Reads 0 in swing and at a fresh touchdown.
        self.lead_ticks = np.where(on & (self.released > 0.0), self.lead_ticks + 1, 0)
        self._on, self._last = on, f
        return self.released.copy()


def _contact_site_names(fused) -> tuple[str, ...]:
    """The estimator's InEKF contact-site names in slot order, read off the filter's own build so
    the sim's toe/heel split cannot disagree with it."""
    names = tuple(fused.model.site_names)
    return tuple(names[int(o)] for o in fused.contact_site_ords)


class SimSensorReader:
    """Index maps from a sim `MjModel` to one `FusedSensors` per call.

    Every lookup is by NAME and resolved once here, never by assuming the sim model and the
    estimator model share index order — the sim adds a floor, collision geoms, actuators and
    visual meshes, and an index coincidence that holds today is not a contract.
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
        sole_centre_x: float = 0.197 / 2.0 - 0.052,
        early_release: float = 0.0,
        early_release_blank: int = 150,
        early_release_mode: str = "current",
        early_release_source: str = "point",
        early_release_rate: float = 0.0,
        early_release_lead: int = 0,
        early_release_off_dwell: int = 0,
    ):
        self.m = m
        self.fused = fused
        self.noise = noise
        self.stance_chol = stance_chol
        self.swing_chol = swing_chol
        # `early_release = 0.0` is OFF and OFF is the default deliberately: every number on record
        # was produced without it. See `EarlyRelease`.
        self.early_release_frac = float(early_release)
        self.early_release_blank = int(early_release_blank)
        self.early_release_mode = str(early_release_mode)
        self.early_release_rate = float(early_release_rate)
        self.early_release_lead = int(early_release_lead)
        self.early_release_off_dwell = int(early_release_off_dwell)
        if early_release_source not in ("point", "foot"):
            raise ValueError("early_release_source must be 'point' or 'foot', got "
                             f"{early_release_source!r}")
        self.early_release_source = early_release_source
        self.release = None

        build = fused.build
        self.imu_names = tuple(build.imu_names)
        self.n_imus = len(self.imu_names)
        self.base_imu = int(build.base_imu)

        def sid(name, obj):
            i = mujoco.mj_name2id(m, obj, name)
            if i < 0:
                raise KeyError(f"no {obj} named {name!r} in the sim model")
            return i

        # IMU sensor addresses (site frame, see module docstring).
        self.gyro_adr = np.array(
            [m.sensor_adr[sid(f"gyro_{s}", mujoco.mjtObj.mjOBJ_SENSOR)] for s in self.imu_names])
        self.acc_adr = np.array(
            [m.sensor_adr[sid(f"acc_{s}", mujoco.mjtObj.mjOBJ_SENSOR)] for s in self.imu_names])

        # Encoders: the 9 filtered joints, in filter state order.
        self.enc_qadr = np.array(
            [m.jnt_qposadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in build.joint_names])
        # DOF addresses for the same joints: qposadr != dofadr in general, and torque is a
        # generalised force, so it indexes by DOF.
        self.enc_dofadr = np.array(
            [m.jnt_dofadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in build.joint_names],
            dtype=int)

        # The unfiltered anchor-chain joints (Alex's 4 ankles).
        self.unfiltered_names = _dof_joint_names(
            fused.model.mj_model, np.asarray(build.dof_anchor_unfiltered, dtype=int))
        self.unf_dofadr = np.array(
            [m.jnt_dofadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in self.unfiltered_names],
            dtype=int)
        self.unf_qadr = np.array(
            [m.jnt_qposadr[sid(n, mujoco.mjtObj.mjOBJ_JOINT)] for n in self.unfiltered_names],
            dtype=int)

        self.foot_gids = np.array(
            [sid(g, mujoco.mjtObj.mjOBJ_GEOM) for g in foot_geoms], dtype=int)
        self.weight = float(m.body_mass.sum()) * 9.81
        self.trust = ContactTrust(n_feet=len(self.foot_gids), dt=dt)

        # Multi-point contacts (toe/heel): `N > K` means the InEKF has more contact points than
        # feet and `contact_chol` must be sized N. The split is resolved HERE, from the
        # estimator's own site names.
        self.n_points = int(fused.n_contacts)
        self.point_trust = None
        self.gid_to_foot = {int(g): k for k, g in enumerate(self.foot_gids)}
        self.foot_bids = np.array(
            [int(m.geom_bodyid[g]) for g in self.foot_gids], dtype=int)
        self.sole_centre_x = float(sole_centre_x)
        self.point_slot: dict[tuple[int, bool], int] = {}
        if self.n_points != len(self.foot_gids):
            names = _contact_site_names(fused)
            if len(names) != self.n_points:
                raise ValueError(
                    f"estimator has N={self.n_points} contacts but its site table "
                    f"resolved {len(names)} names ({names}); cannot split the load")
            for j, nm in enumerate(names):
                low = nm.lower()
                fore = "toe" in low
                if not fore and "heel" not in low:
                    raise ValueError(
                        f"contact site {nm!r} is neither a 'toe' nor a 'heel'; "
                        f"`point_loads` splits the per-foot normal force fore/aft "
                        f"and has no rule for it")
                # Foot by geom-name overlap: 'left_toe' -> the geom whose name
                # contains 'LEFT'. Explicit rather than positional, so reordering
                # either table cannot silently swap the feet.
                side = "LEFT" if low.startswith("left") else "RIGHT"
                cand = [k for k, g in enumerate(foot_geoms) if side in g.upper()]
                if len(cand) != 1:
                    raise ValueError(
                        f"contact site {nm!r} matched {len(cand)} foot geoms for "
                        f"side {side!r}: {foot_geoms}")
                self.point_slot[(cand[0], fore)] = j
            if len(self.point_slot) != self.n_points:
                raise ValueError(
                    f"the (foot, toe/heel) split is not one-to-one: "
                    f"{self.n_points} sites collapsed to {len(self.point_slot)} slots")
            self.point_trust = ContactTrust(n_feet=self.n_points, dt=dt)

        if self.early_release_frac > 0.0 or self.early_release_lead > 0:
            self.release = EarlyRelease(
                n_points=self.n_points, dt=dt, frac=self.early_release_frac,
                blank_ticks=self.early_release_blank, peak_mode=self.early_release_mode,
                rate_frac=self.early_release_rate, clock_lead=self.early_release_lead,
                off_dwell=self.early_release_off_dwell)

        # Ground truth, for scoring.
        self.base_bid = sid("PELVIS_LINK", mujoco.mjtObj.mjOBJ_BODY)
        self.base_site = sid("base_body", mujoco.mjtObj.mjOBJ_SITE)

    def foot_loads_raw(self, d: mujoco.MjData) -> np.ndarray:
        """Normalised per-foot normal load, `f_n / (0.5·m·g)`, **unclipped**.

        `EarlyRelease` needs the unclipped signal: its reference is the stance PEAK, and
        a peak read off a signal clipped at 1.0 is not a peak. `foot_loads` clips, because
        `ContactTrust` wants a probability.
        """
        f = np.zeros(len(self.foot_gids))
        frc = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for k, gid in enumerate(self.foot_gids):
                if c.geom1 == gid or c.geom2 == gid:
                    mujoco.mj_contactForce(self.m, d, i, frc)
                    f[k] += abs(frc[0])
        return f / (0.5 * self.weight)

    def foot_loads(self, d: mujoco.MjData) -> np.ndarray:
        """Normalised per-foot normal load, `f_n / (0.5·m·g)`, clipped to [0, 1]."""
        return np.clip(self.foot_loads_raw(d), 0.0, 1.0)

    def point_loads_raw(self, d: mujoco.MjData) -> np.ndarray:
        r"""Normalised load per CONTACT POINT, ``(N,)``, heel/toe split fore-aft, unclipped.

        Only meaningful when the estimator was built with more contact points than
        feet (`build_fused_estimator(contact_sites=...)`); `read` falls back to
        `foot_loads` otherwise.

        The split needs no change to the collision geometry, which is one box per
        foot. MuJoCo reports each contact's world position, so a contact is
        attributed to the toe or the heel by the sign of its position **in the foot
        frame** relative to the sole-plate centre:

            x_local = (ᵂR_F)ᵀ (c.pos − p_F) · x̂        toe if x_local > x_centre

        `abs(frc[0])` is the normal component in the contact frame, the same
        quantity `foot_loads` sums.

        Normalisation is per point against the same ``0.5·m·g``, so a *fully*
        loaded toe reads ~1.0 and a flat-footed stance reads ~0.5 at each point —
        nearer the ``enter = 0.35`` threshold than a whole foot is, so the Schmitt
        trigger has less margin and may chatter where the per-foot signal would
        not. Watch `trusted` if a run looks like it is losing anchors.
        """
        f = np.zeros(self.n_points)
        frc = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            k = self.gid_to_foot.get(int(c.geom1), self.gid_to_foot.get(int(c.geom2)))
            if k is None:
                continue
            bid = self.foot_bids[k]
            R = d.xmat[bid].reshape(3, 3)
            x_local = float((R.T @ (np.asarray(c.pos) - d.xpos[bid]))[0])
            j = self.point_slot.get((k, x_local > self.sole_centre_x))
            if j is None:
                continue
            mujoco.mj_contactForce(self.m, d, i, frc)
            f[j] += abs(frc[0])
        return f / (0.5 * self.weight)

    def point_loads(self, d: mujoco.MjData) -> np.ndarray:
        """`point_loads_raw` clipped to [0, 1] -- what `ContactTrust` consumes."""
        return np.clip(self.point_loads_raw(d), 0.0, 1.0)

    def read(self, d: mujoco.MjData):
        """One `FusedSensors` (NumPy leaves) from the current `MjData`."""
        from ..pipeline.main_estimator import FusedSensors

        gyros = d.sensordata[self.gyro_adr[:, None] + np.arange(3)].copy()
        accel = d.sensordata[self.acc_adr[self.base_imu] + np.arange(3)].copy()
        enc = d.qpos[self.enc_qadr].copy()
        qd_u = d.qvel[self.unf_dofadr].copy()
        # Only read when the estimator was built with `contact_fk_unfiltered`; an empty array
        # otherwise, which is the "field absent" encoding `FusedSensors` expects.
        q_u = (d.qpos[self.unf_qadr].copy() if self.fused.n_aux else np.zeros(0))
        # ContactNet feature channel only — the estimator never reads it. `qfrc_actuator` is in
        # GENERALISED coordinates, so it indexes by dofadr and lines up with the encoder ordering;
        # `actuator_force` would be per-actuator and need the transmission map. Ordered
        # concat(filtered, unfiltered), matching how `fused_inputs` widens q̂ for the contact FK.
        tau = d.qfrc_actuator[np.concatenate([self.enc_dofadr, self.unf_dofadr])].copy()
        if self.noise is not None:
            gyros = self.noise.corrupt_gyros(gyros)
            accel = self.noise.corrupt_accel(accel)
            enc = self.noise.corrupt_encoders(enc)
            qd_u = self.noise.corrupt_velocities(qd_u)
            q_u = self.noise.corrupt_encoders(q_u)
            tau = self.noise.corrupt_torques(tau)

        # `contact` feeds the joint KF's K stance anchors (one per foot);
        # `contact_chol` feeds the InEKF's N contact points. They are the same
        # thing only when N == K -- see `build_fused_estimator(contact_sites=...)`.
        raw_foot = self.foot_loads_raw(d)
        trusted = self.trust.update(np.clip(raw_foot, 0.0, 1.0))
        if self.point_trust is None:
            point_trusted, raw_point = trusted, raw_foot
        else:
            raw_point = self.point_loads_raw(d)
            point_trusted = self.point_trust.update(np.clip(raw_point, 0.0, 1.0))
        # The InEKF has NO contact mask: contact condition rides ENTIRELY in
        # Sigma_C (`inEKF/filter.py`, the DECISION note). A swing foot therefore
        # needs a LARGE factor here, or the filter keeps believing it is planted.
        loose = point_trusted <= 0.0
        if self.release is not None:
            # ONLY ever loosens (`EarlyRelease`): OR with the heuristic's swing state, so
            # this can move liftoff earlier and can never move touchdown earlier.
            src = raw_point if self.early_release_source == "point" else \
                np.repeat(raw_foot, self.n_points // len(raw_foot))
            loose = loose | (self.release.update(src) > 0.0)
        chol = np.where(loose[:, None, None], self.swing_chol, self.stance_chol)
        return FusedSensors(
            encoders=enc,
            gyros=gyros,
            accel_base=accel,
            qd_unfiltered=qd_u,
            contact=trusted,
            contact_chol=chol * np.tile(np.eye(3), (len(point_trusted), 1, 1)),
            q_unfiltered=q_u,
            torques=tau,
        )

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
