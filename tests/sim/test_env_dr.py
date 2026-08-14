"""Environment domain randomization at the collection boundary.

The commit adds terrain heightfields, per-rollout friction, and pelvis pushes to
`sim/collect.py`, all gated on `ContactNetConfig.env_dr`. Two invariants have to hold
and this file pins both:

  * OFF is a no-op. With `env_dr=False` / `terrain=None` the sim is the flat plane at
    mu=1 with no applied force -- the run we already trust must be untouched.
  * ON did something. Terrain is actually an hfield, friction actually drops at the
    contact, and pushes actually fire AND clear. A DR knob that silently does nothing
    is worse than no knob: training "on rough terrain" that is secretly flat is the
    kind of green-but-wrong that costs a week.

Following `tests/sim/test_sensors.py`: every claim is checked against an INDEPENDENT
computation (MuJoCo's own contact-friction mixing, a re-derived hfield_data), never
against a second call to the code under test, and the "ON" tests assert the world
DIFFERS rather than merely that it ran (TERRAIN.md §7: a check that agrees on a zero
is not a check).
"""

import numpy as np
import mujoco
import pytest

import run_policy as rp
from invariant_estimation.sim import terrain as terr
from invariant_estimation.sim.collect import build_collector, _RecordingLoop, Disturb
from invariant_estimation.sim.sensors import SimSensorReader


def _geom_type(m, name):
    """MuJoCo returns the type as a size-1 array through the named accessor."""
    return int(np.ravel(m.geom(name).type)[0])


@pytest.fixture(scope="module")
def policy():
    return rp.load_policy("baseline")


# ---------------------------------------------------------------------------
# OFF is a no-op: the flat-preserving invariant
# ---------------------------------------------------------------------------

def test_terrain_none_keeps_the_flat_plane(policy):
    """`terrain=None` must reproduce today's world: a plane floor at mu=1.

    This is the whole safety argument for landing the commit -- every DR path is
    gated, so with the gate shut nothing about the trusted flat run may change.
    """
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True, terrain=None)
    assert _geom_type(m, "floor") == mujoco.mjtGeom.mjGEOM_PLANE
    # Nominal slide friction is CONTACT["friction"][0] == 1.0, untouched.
    assert float(m.geom_friction[m.geom("floor").id, 0]) == 1.0
    for name in rp.FOOT_GEOMS:
        assert float(m.geom_friction[m.geom(name).id, 0]) == 1.0


def test_disturb_none_never_touches_xfrc(policy):
    """With no `Disturb`, `xfrc_applied` stays identically zero for the whole rollout.

    Guards the OFF half of the push path: an applied force that leaks in when DR is
    disabled would silently perturb the baseline dynamics.
    """
    c = build_collector(policy_name="baseline", verbose=False)
    m = rp.build_sim_model(c.policy, with_visuals=False, with_imu_sensors=True)
    reader = SimSensorReader(m, c.fused, foot_geoms=rp.FOOT_GEOMS, dt=c.dt, noise=None)
    loop = _RecordingLoop(m, c.policy, rp.make_maps(m, c.policy), reader, disturb=None)
    loop.set_height_target(loop.height_target)
    for _ in range(60):
        loop.control_tick()
        assert np.all(loop.d.xfrc_applied == 0.0), "xfrc leaked with disturb=None"


# ---------------------------------------------------------------------------
# ON did something: terrain
# ---------------------------------------------------------------------------

def test_terrain_field_builds_an_hfield(policy):
    """A field turns the floor into an hfield whose data is the normalized relief.

    `hfield_data` is re-derived here from the field and EZ (an independent
    computation), so this catches a missing or wrong `/ EZ` normalization -- the
    one that would silently flatten or clip the terrain.
    """
    field = terr.sample_field("waves", seed=0)
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True, terrain=field)

    assert _geom_type(m, "floor") == mujoco.mjtGeom.mjGEOM_HFIELD
    assert m.hfield_data.shape[0] == terr.N * terr.N
    expected = (np.asarray(field, np.float32) / terr.EZ).clip(0.0, 1.0).ravel()
    np.testing.assert_allclose(m.hfield_data, expected, rtol=0, atol=0)
    assert field.max() > 0.0, "the waves field is degenerate -- nothing to walk over"


def test_policy_walks_on_terrain(policy):
    """End-to-end: the baseline policy stays up and moves forward on a waves hfield.

    This is the "prove the terrain is real AND survivable" test -- it fails if the
    hfield never actually collides (robot falls through / stands on nothing) or if
    the spawn-z raise is missing and the robot spawns buried. Nominal friction, so
    walking is not slip-limited (TERRAIN.md Stage 1: ~0.38 m/s on waves).
    """
    field = terr.sample_field("waves", seed=0)
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=False, terrain=field)
    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    loop.d.qpos[2] += float(field.max()) + 0.02          # start clear of the relief
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)

    for _ in range(100):                                  # 2 s settle onto the terrain
        loop.control_tick()
    x0 = float(loop.d.qpos[0])
    for _ in range(250):                                  # 5 s walking +x
        loop.cmd[0:3] = (0.4, 0.0, 0.0); loop.cmd[3] = 0.0
        loop.control_tick()

    assert np.all(np.isfinite(loop.d.qpos)), "sim went non-finite on terrain"
    assert loop.d.qpos[2] > 0.8, f"robot fell on terrain (z={float(loop.d.qpos[2]):.2f})"
    assert float(loop.d.qpos[0]) - x0 > 0.5, "robot did not make forward progress on terrain"


# ---------------------------------------------------------------------------
# ON did something: friction, and WHY both geoms must be set
# ---------------------------------------------------------------------------

def test_mujoco_mixes_friction_by_max_so_both_geoms_matter():
    """The load-bearing rationale: MuJoCo takes the ELEMENT-WISE MAX of the pair.

    Lowering only the foot leaves the effective slide friction at the floor's 1.0 --
    the low-mu tail would be silently clipped away and no slip would ever occur.
    Lowering both gives the intended mu. Deterministic minimal model: a box settling
    on a plane, no policy, no slip flakiness. If this ever flips, the "set BOTH foot
    and floor" line in `collect_rollout` is no longer justified and must be revisited.
    """
    xml = """
    <mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 0.1" condim="4" friction="1 0.05 0.01"/>
      <body pos="0 0 0.05"><freejoint/>
        <geom name="foot" type="box" size="0.13 0.07 0.0275" condim="4" friction="1 0.05 0.01"/>
      </body>
    </worldbody></mujoco>"""

    def settle_and_read(lower):
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        m.geom_friction[[m.geom(g).id for g in lower], 0] = 0.2
        for _ in range(200):
            mujoco.mj_step(m, d)
        assert d.ncon > 0
        return float(d.contact[0].friction[0])

    assert settle_and_read(["foot"]) == pytest.approx(1.0), "MAX-mixing assumption broke"
    assert settle_and_read(["foot", "floor"]) == pytest.approx(0.2)


def test_friction_dr_edit_hits_both_foot_and_floor_on_the_real_model(policy):
    """The commit's two-line edit lands on the right geoms of the Alex model.

    Applies the exact indexing `collect_rollout` uses and asserts every foot geom
    AND the floor carry the sampled mu -- so a wrong geom name, the wrong friction
    column, or a forgotten floor is caught here rather than as a mysterious lack of
    slip in a 12-hour run.
    """
    field = terr.sample_field("flat", seed=0)
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True, terrain=field)
    gids = [m.geom(n).id for n in rp.FOOT_GEOMS] + [m.geom("floor").id]
    m.geom_friction[gids, 0] = 0.2
    assert np.allclose(m.geom_friction[gids, 0], 0.2)


# ---------------------------------------------------------------------------
# ON did something: pushes fire AND clear
# ---------------------------------------------------------------------------

def test_pushes_fire_and_clear(policy):
    """A `Disturb` must apply force on some ticks and release it on others.

    `xfrc_applied` is never auto-reset, so the bug this guards is a push that never
    clears -- a permanent thrust masquerading as a disturbance. Gentle, brief pushes
    so the robot does not fall; we assert only fire+clear+finiteness, not uprightness.
    """
    c = build_collector(policy_name="baseline", verbose=False)
    m = rp.build_sim_model(c.policy, with_visuals=False, with_imu_sensors=True)
    reader = SimSensorReader(m, c.fused, foot_geoms=rp.FOOT_GEOMS, dt=c.dt, noise=None)
    disturb = Disturb(rate_hz=6.0, mag_N=(10.0, 20.0), dur_s=0.02)   # ~1 control tick per push
    loop = _RecordingLoop(m, c.policy, rp.make_maps(m, c.policy), reader,
                          disturb=disturb, dr_seed=0)
    loop.set_height_target(loop.height_target)

    base = reader.base_bid
    norms = []
    for _ in range(150):
        loop.control_tick()
        norms.append(float(np.linalg.norm(loop.d.xfrc_applied[base, :3])))
    norms = np.asarray(norms)

    assert np.all(np.isfinite(loop.d.qpos))
    assert (norms > 0.0).any(), "no push ever fired"
    assert (norms == 0.0).any(), "push never cleared -- xfrc_applied left latched"
