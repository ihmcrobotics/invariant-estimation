"""The plant → estimator boundary: MuJoCo sensors, encoder maps, contact trust.

These are the tests that would have caught a frame or an index error at the boundary — the
class of bug that costs the most, because a rotated gyro or a mis-ordered encoder vector still
produces a filter that runs, converges, and is quietly wrong (see `R_mount`, PORT_NOTES G9).

Each sensor property is checked against an INDEPENDENT computation (`mj_objectVelocity` on the
site's body, an explicit `R^T g`), never against another call to the same helper.
"""

import numpy as np
import mujoco
import pytest

import run_policy as rp
from invariant_estimation.pipeline import main_estimator as me
from invariant_estimation.sim.sensors import (
    ContactTrust,
    IMUNoise,
    SimSensorReader,
    add_imu_sensors,
)

DT = rp.DT


@pytest.fixture(scope="module")
def policy():
    return rp.load_policy("baseline")


@pytest.fixture(scope="module")
def model_with_sensors(policy):
    return rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True)


@pytest.fixture(scope="module")
def rest_data(model_with_sensors, policy):
    """The robot standing on the floor, HELD UP BY THE POLICY, in steady state.

    Two things this fixture has to get right, both learned the hard way:

    * `Loop` seeds the robot 1 mm above the floor, where it is in FREE FALL — and an
      accelerometer in free fall correctly reads zero. A fixture that only calls `mj_forward`
      on the seed pose measures the one state in which specific force is not `+g`.
    * The home-pose position servos alone do NOT hold Alex up: left passive, the robot sinks
      from z=0.92 to z=0.19 over ~5 s (measured). Only the policy stands it up, so "at rest"
      has to mean "closed loop, settled", not "stepped for a while".
    """
    m = model_with_sensors
    maps = rp.make_maps(m, policy)
    loop = rp.Loop(m, policy, maps)
    for _ in range(100):                     # 2 s of policy control -> steady stance
        loop.control_tick()
    assert loop.d.ncon >= 4, "the robot did not settle onto the floor"
    assert loop.d.qpos[2] > 0.8, f"the robot did not stay standing (z={loop.d.qpos[2]:.2f})"
    return m, loop.d, maps


# ---------------------------------------------------------------------------
# The sensors themselves
# ---------------------------------------------------------------------------

def test_sensors_do_not_change_the_dynamics(policy):
    """Adding IMUs must be observationally free: same states, bit for bit.

    If this ever fails, every A/B against `run_policy.py` is invalid.
    """
    m_plain = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=False)
    m_imu = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True)
    assert m_imu.nsensor > m_plain.nsensor

    maps = rp.make_maps(m_plain, policy)
    states = []
    for m in (m_plain, m_imu):
        d = mujoco.MjData(m)
        d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
        d.qpos[0:3] = [0.0, 0.0, rp.foot_rest_height(m, maps)]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        for _ in range(200):
            mujoco.mj_step(m, d)
        states.append(d.qpos.copy())
    np.testing.assert_array_equal(states[0], states[1])


def test_gyro_sensors_are_in_their_own_site_frames(model_with_sensors, policy):
    """`gyro_<site>` == `^W R_site^T · ω_world(site's body)`, for every IMU, in motion.

    The independent reference is `mj_objectVelocity` in WORLD (flg_local=0) rotated by the site
    rotation — a different code path from the sensor. A gyro reported in the body/inertial frame
    (the `mjOBJ_BODY` trap) or unrotated fails here.
    """
    m = model_with_sensors
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    d.qpos[:] = m.qpos0
    d.qpos[3:7] = [0.83, 0.21, -0.39, 0.34]         # a generic base attitude
    d.qpos[7:] += 0.3 * rng.standard_normal(m.nq - 7)
    d.qvel[:] = 0.7 * rng.standard_normal(m.nv)     # everything moving
    mujoco.mj_forward(m, d)

    v6 = np.zeros(6)
    for site in me.ALEX_IMU_SITES:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"gyro_{site}")]
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, v6, 0)   # world frame
        R_site = d.site_xmat[sid].reshape(3, 3)
        np.testing.assert_allclose(d.sensordata[adr:adr + 3], R_site.T @ v6[:3], atol=1e-9)


def test_accelerometer_reads_specific_force_not_acceleration(rest_data):
    """At rest the pelvis IMU reads +g along its own up axis, not zero.

    This is the convention `FusedSensors.accel_base` requires (the InEKF adds gravity itself).
    An accelerometer that reported coordinate acceleration would read ~0 here and the InEKF
    would integrate a 9.81 m/s² phantom.
    """
    m, d, _ = rest_data
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pelvis_imu")
    adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "acc_pelvis_imu")]
    a = d.sensordata[adr:adr + 3]
    assert abs(np.linalg.norm(a) - 9.81) < 0.3, "a standing robot's IMU must read ~1 g"
    R_site = d.site_xmat[sid].reshape(3, 3)
    np.testing.assert_allclose(R_site @ a, [0.0, 0.0, 9.81], atol=0.3)


# ---------------------------------------------------------------------------
# The reader's index maps
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fused():
    return me.build_alex_fused_estimator_from_urdf(rp.cycloid_forearm_urdf(rp.URDF), dt=DT)


@pytest.fixture(scope="module")
def reader(model_with_sensors, fused):
    return SimSensorReader(model_with_sensors, fused, foot_geoms=rp.FOOT_GEOMS, dt=DT)


def test_encoders_are_gathered_in_filter_state_order(rest_data, reader, fused):
    """A DISTINCT value per joint, so a permuted gather cannot pass by symmetry."""
    m, d, _ = rest_data
    d = mujoco.MjData(m)
    want = {}
    for i, name in enumerate(fused.build.joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        value = 0.11 * (i + 1)
        d.qpos[m.jnt_qposadr[jid]] = value
        want[name] = value
    mujoco.mj_forward(m, d)
    got = reader.read(d).encoders
    np.testing.assert_allclose(got, [want[n] for n in fused.build.joint_names], atol=0)
    # and the ordering is not accidentally sorted/reversed
    assert not np.allclose(got, np.sort(got)[::-1])


def test_unfiltered_anchor_joints_are_the_four_ankles(reader):
    """The joint-KF anchor chain's non-state joints on Alex (no foot IMUs)."""
    assert set(reader.unfiltered_names) == {
        "LEFT_ANKLE_Y", "LEFT_ANKLE_X", "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X"}


def test_unfiltered_velocities_follow_the_build_column_order(rest_data, reader):
    """`qd_unfiltered` must be in `anchor_unfiltered_mask` column order, not model order."""
    m, _, _ = rest_data
    d = mujoco.MjData(m)
    for i, name in enumerate(reader.unfiltered_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qvel[m.jnt_dofadr[jid]] = 0.37 * (i + 1)
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(reader.read(d).qd_unfiltered,
                               [0.37 * (i + 1) for i in range(len(reader.unfiltered_names))])


def test_sensor_shapes_match_what_the_filter_declares(rest_data, reader, fused):
    m, d, _ = rest_data
    s = reader.read(d)
    assert s.encoders.shape == (fused.n_joints,)
    assert s.gyros.shape == (fused.build.n_imus, 3)
    assert s.accel_base.shape == (3,)
    assert s.contact.shape == (fused.n_contacts,)
    assert s.contact_chol.shape == (fused.n_contacts, 3, 3)


def test_noise_is_reproducible_and_bias_is_constant():
    """The gyro bias is drawn ONCE per run: it is a bias, not another white channel."""
    n = IMUNoise(seed=3)
    b1, b2 = n.bias(8).copy(), n.bias(8).copy()
    np.testing.assert_array_equal(b1, b2)
    assert np.linalg.norm(b1) > 0
    np.testing.assert_array_equal(IMUNoise(seed=3).bias(8), b1)
    assert not np.allclose(IMUNoise(seed=4).bias(8), b1)


# ---------------------------------------------------------------------------
# Contact trust
# ---------------------------------------------------------------------------

def _drive(trust, load, ticks):
    out = None
    for _ in range(ticks):
        out = trust.update(np.array([load]))
    return out


def test_a_lightly_loaded_foot_is_never_trusted():
    t = ContactTrust(n_feet=1, dt=DT)
    assert _drive(t, 0.30, 400)[0] == 0.0        # below `enter` = 0.35, however long


def test_entering_requires_the_dwell():
    """Sustained load must clear the dwell; a shorter pulse must not anchor."""
    t = ContactTrust(n_feet=1, dt=DT, ema_tau=0.0)
    short = t.dwell / DT - 1
    assert _drive(t, 1.0, int(short))[0] == 0.0
    assert t.update(np.array([1.0]))[0] == 1.0


def test_hysteresis_holds_a_trusted_foot_between_stay_and_enter():
    """A foot that has landed keeps anchoring while unloading down to `stay`."""
    t = ContactTrust(n_feet=1, dt=DT, ema_tau=0.0)
    _drive(t, 1.0, 20)
    assert t.trusted[0] == 1.0
    assert _drive(t, 0.30, 50)[0] == 1.0          # between stay (0.25) and enter (0.35)
    assert _drive(t, 0.20, 1)[0] == 0.0           # below stay -> released immediately


def test_release_is_immediate_but_re_entry_is_not():
    """The asymmetry that protects the bias gauge from a bouncing touchdown."""
    t = ContactTrust(n_feet=1, dt=DT, ema_tau=0.0)
    _drive(t, 1.0, 20)
    assert t.update(np.array([0.0]))[0] == 0.0
    assert t.update(np.array([1.0]))[0] == 0.0    # must serve the dwell again
    assert _drive(t, 1.0, int(t.dwell / DT))[0] == 1.0


def test_feet_are_independent(rest_data, reader):
    t = ContactTrust(n_feet=2, dt=DT, ema_tau=0.0)
    for _ in range(20):
        out = t.update(np.array([1.0, 0.0]))
    np.testing.assert_array_equal(out, [1.0, 0.0])


def test_standing_robot_trusts_both_feet_and_swing_inflates_sigma_c(rest_data, reader, fused):
    """Both feet loaded at rest ⇒ both trusted, and the InEKF factor is the STANCE one.

    The InEKF has no contact mask (contact condition rides entirely in Σ_C), so this mapping is
    the only thing telling it a foot is airborne.
    """
    m, d, _ = rest_data
    reader.trust.__init__(n_feet=2, dt=DT)
    for _ in range(30):
        s = reader.read(d)
    np.testing.assert_array_equal(s.contact, [1.0, 1.0])
    np.testing.assert_allclose(s.contact_chol[0], np.eye(3) * reader.stance_chol)

    d2 = mujoco.MjData(m)                              # no contacts at all: robot in the air
    d2.qpos[:] = d.qpos
    d2.qpos[2] += 1.0
    mujoco.mj_forward(m, d2)
    for _ in range(30):
        s = reader.read(d2)
    np.testing.assert_array_equal(s.contact, [0.0, 0.0])
    np.testing.assert_allclose(s.contact_chol[0], np.eye(3) * reader.swing_chol)
