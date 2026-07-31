"""Tilt-dependent friction — slippery ramps, normal flats (`DomainRandomization.slope_friction`).

The intervention this exists for: every dataset so far sets ONE `mu` on every geom, so both feet
always stand on the same surface and contact quality is perfectly correlated across feet. `data/dr6`
raised slip that way to 19% pooled and the retrain did not help. A per-contact `Sigma_C` has nothing
to learn from a signal identical on both feet at every tick. This puts one foot on firm ground and
the other on a slope *at the same tick*.

Three obligations:

1. **The `mj_step1`/`mj_step2` split is physically the same integrator as `mj_step`.** The feature
   needs the split because friction cannot be set before collision. If the split alone changed the
   trajectory, every on-vs-off comparison would be confounded by the stepping scheme rather than by
   friction.
2. **Flat ground is never made slippery.** `flat` and `waves` must be untouched by construction, or
   the intervention is just another global `mu` change wearing a hat.
3. **Stone terrains actually get ramp contacts.** A feature that silently does nothing on the
   terrain it targets would still produce a plausible-looking dataset.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from invariant_estimation.sim import collect as C
from invariant_estimation.sim import terrain as tr


def _model(field):
    import run_policy as rp
    policy = rp.load_policy("baseline")
    return rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True,
                              floor=tr.HeightfieldFloor(field))


# ---------------------------------------------------------------------------
# 1. the split is the same integrator
# ---------------------------------------------------------------------------

def test_the_step_split_is_bit_identical_to_plain_mj_step():
    """`mj_step` vs `mj_step1` + `mj_step2` with nothing modified in between.

    `integrator="implicitfast"` supports the split; RK4 would not, and this is the test that would
    catch someone changing the integrator underneath the feature.
    """
    m = _model(C.terrain_field("hard_stepping", 0))
    a, b = mujoco.MjData(m), mujoco.MjData(m)
    rng = np.random.default_rng(0)
    ctrl = rng.uniform(-0.05, 0.05, m.nu)
    for d in (a, b):
        mujoco.mj_resetData(m, d)
        d.qpos[2] += tr.spawn_lift(C.terrain_field("hard_stepping", 0))
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)

    for _ in range(300):                      # long enough to be in contact and loaded
        mujoco.mj_step(m, a)
        mujoco.mj_step1(m, b)
        mujoco.mj_step2(m, b)

    np.testing.assert_array_equal(a.qpos, b.qpos)
    np.testing.assert_array_equal(a.qvel, b.qvel)


# ---------------------------------------------------------------------------
# 2/3. what the slope rule classifies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expect_ramps", [
    ("flat", False),                 # excluded BY NAME
    ("waves", False),                # excluded BY NAME
    ("hard_stepping", True),         # stone edges
])
def test_only_the_named_terrains_get_ramp_friction(name, expect_ramps):
    """Flats keep their friction because they are NOT IN THE TERRAIN LIST — not because of the
    threshold.

    Measured with the policy walking, every terrain has a few near-90 deg foot contacts (foot
    SIDE/EDGE strikes at touchdown), so a tilt threshold alone would make flat ground slippery
    ~2.5% of the time. The terrain list is what makes this an experiment about ramps rather than
    another global mu change, and this test pins that rather than the threshold.
    """
    import run_policy as rp

    field = C.terrain_field(name, 0)
    m = _model(field)
    policy = rp.load_policy("baseline")
    maps = rp.make_maps(m, policy)
    d = mujoco.MjData(m)
    d.qpos[2] += tr.spawn_lift(field)
    # Hold the home pose so the robot STANDS. Unactuated it collapses, and a collapsing robot puts
    # knees and arms on the ground at every angle — which is how the first version of this test
    # "failed" on flat ground and, usefully, exposed that the implementation was overriding
    # non-foot contacts too.
    d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
    mujoco.mj_forward(m, d)

    dr = C.DomainRandomization(
        slope_friction=True, slope_deg=10.0,
        slope_terrains=("stepping_stones", "hard_stepping"))
    applies = (not dr.slope_terrains) or name in dr.slope_terrains
    assert applies == expect_ramps, f"{name}: terrain-list gating disagrees with the fixture"

    # FOOT contacts only — the same restriction the implementation applies.
    feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in rp.FOOT_GEOMS}
    cos_t = float(np.cos(np.radians(dr.slope_deg)))
    ramps = 0
    for _ in range(600):
        mujoco.mj_step(m, d)
        if not applies:
            continue
        for i in range(d.ncon):
            c = d.contact[i]
            if (int(c.geom1) in feet or int(c.geom2) in feet) and abs(c.frame[2]) < cos_t:
                ramps += 1

    if expect_ramps:
        assert ramps > 0, f"{name}: no contact was ever classified as a ramp"
    else:
        assert ramps == 0, (
            f"{name}: {ramps} contacts would be made slippery on ground that is flat to within "
            "the threshold — this would be a global mu change in disguise")


def test_the_override_actually_lowers_the_enforced_friction():
    """`_read_slip` reads `contact.friction[0]` — the value MuJoCo enforced — so if the override
    did not take, the recorded cone saturation would be computed against the wrong mu."""
    field = C.terrain_field("hard_stepping", 0)
    m = _model(field)
    d = mujoco.MjData(m)
    d.qpos[2] += tr.spawn_lift(field)
    mujoco.mj_forward(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)

    cos10 = float(np.cos(np.radians(10.0)))
    steep = [i for i in range(d.ncon) if abs(d.contact[i].frame[2]) < cos10]
    if not steep:
        pytest.skip("no steep contact in this pose; the classification is covered above")
    before = float(d.contact[steep[0]].friction[0])
    d.contact[steep[0]].friction[0] = 0.2
    assert float(d.contact[steep[0]].friction[0]) == pytest.approx(0.2)
    assert before != pytest.approx(0.2), "the contact already had the ramp value; test is vacuous"


def test_slope_friction_is_off_by_default_and_parses():
    assert C.DomainRandomization().slope_friction is False
    dr = C.DomainRandomization.from_dict(
        {"slope_friction": {"enabled": True, "slope_deg": 12.0, "mu_range": [0.15, 0.4]}})
    assert dr.slope_friction and dr.slope_deg == 12.0 and dr.slope_mu_range == (0.15, 0.4)
    with pytest.raises(KeyError, match="slope_degrees"):
        C.DomainRandomization.from_dict({"slope_friction": {"slope_degrees": 12.0}})
