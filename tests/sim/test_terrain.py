"""`invariant_estimation.sim.terrain` — the rasterisers, the floor spec, and the shared builder.

The walk itself is `experiments/terrain_stage1_walk.py` (20 s x 4 terrains, far too slow for CI).
What is worth locking down here is everything that could make that walk silently meaningless: a
rasteriser that returns flat ground, a normalisation that clips the relief away, an `hfield_data`
that never reaches the compiled model, or a terrain build that stops sharing `run_policy`'s
assembly and starts drifting from it.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

import run_policy as rp
from invariant_estimation.sim import terrain as tr

SMALL = 64          # px, for the rasteriser tests -- shape is a parameter, so keep them cheap


# --- rasterisers -------------------------------------------------------------------------------
def test_flat_is_flat():
    z = tr.flat(SMALL)
    assert z.shape == (SMALL, SMALL) and z.dtype == np.float32
    assert not z.any()


def test_waves_wavelength_is_extent_independent():
    """`num_waves` counts waves per TILE, not per field, so widening the field adds crests rather
    than stretching them. Getting this wrong turns a 64 m field of waves into one gentle ramp."""
    a, hscale = 0.10, tr.HSCALE
    small = tr.waves(a, 2.0, n=SMALL, hscale=hscale)[0]
    big = tr.waves(a, 2.0, n=4 * SMALL, hscale=hscale)[0]
    np.testing.assert_allclose(big[:SMALL], small, atol=1e-7)   # same terrain, just more of it

    # One period is tile / num_waves = 4 m = 40 px, and the profile repeats exactly.
    period = int(round(tr.TILE / 2.0 / hscale))
    np.testing.assert_allclose(big[period:2 * period], big[:period], atol=1e-6)


def test_waves_amplitude_and_orientation():
    z = tr.waves(0.10, 2.0, n=SMALL)
    assert z.min() == pytest.approx(0.0, abs=1e-6)
    assert z.max() == pytest.approx(0.10, abs=1e-3)
    # Axis 0 is the row index (world +y), axis 1 the column index (world +x). The robot walks +x,
    # so the ripple must live along axis 1; a transposed field is terrain it never crosses.
    assert z.std(axis=0).max() == pytest.approx(0.0, abs=1e-7)
    assert z.std(axis=1).min() > 0.01


def test_stepping_stones_quantises_to_the_grid_and_is_seeded():
    grid, hi = 0.45, 0.03
    z = tr.stepping_stones(grid, hi, seed=3, n=SMALL)
    assert 0.0 <= z.min() and z.max() <= hi
    assert z.max() > 0.5 * hi                      # actual relief, not a rounding artefact
    np.testing.assert_array_equal(z, tr.stepping_stones(grid, hi, seed=3, n=SMALL))
    assert not np.array_equal(z, tr.stepping_stones(grid, hi, seed=4, n=SMALL))

    blk = int(round(grid / tr.HSCALE))             # each stone is constant over blk x blk px
    c = SMALL // 2 + 3 * blk                       # off the cleared centre pad
    assert np.ptp(z[c:c + blk, c:c + blk]) == 0.0


def test_stepping_stones_always_clears_a_spawn_pad():
    """The robot has to settle somewhere level, or the first tick is a fall instead of a step."""
    for platform in (0.0, 0.5):
        z = tr.stepping_stones(0.75, 0.07, seed=5, platform=platform, n=SMALL)
        c = SMALL // 2
        assert not z[c - 3:c + 3, c - 3:c + 3].any()


def test_terrains_registry_covers_isaaclab_and_the_entries_differ():
    assert set(tr.TERRAINS) == {"flat", "waves", "stepping_stones", "hard_stepping"}
    fields = {n: f() for n, f in tr.TERRAINS.items()}
    assert all(f.shape == (tr.N, tr.N) for f in fields.values())
    # TERRAIN.md §1's reliefs. A registry whose entries are all flat passes every "it ran" check.
    assert fields["flat"].max() == 0.0
    assert fields["waves"].max() == pytest.approx(0.10, abs=1e-3)
    assert fields["stepping_stones"].max() == pytest.approx(0.03, abs=2e-3)
    assert fields["hard_stepping"].max() == pytest.approx(0.07, abs=4e-3)


def test_extent_is_big_enough_for_a_useful_episode():
    """16 m gave ~21 s from a centre spawn at the policy's 0.38 m/s -- not enough once a warm-up
    prefix is discarded. An hfield is finite; walking off the edge is a real failure mode."""
    assert tr.EXTENT / 2 / 0.38 > 60.0
    assert tr.N == int(round(tr.EXTENT / tr.HSCALE))


def test_sample_maps_world_coordinates_to_the_right_pixel():
    z = np.zeros((SMALL, SMALL), np.float32)
    extent = SMALL * tr.HSCALE
    z[SMALL // 2, SMALL // 2 + 20] = 1.0            # row = +y, col = +x
    assert tr.sample(z, 2.0, 0.0, extent=extent) == 1.0     # +2 m in x  = +20 px in col
    assert tr.sample(z, 0.0, 2.0, extent=extent) == 0.0     # +2 m in y is a different cell
    assert tr.sample(z, 1e6, 1e6, extent=extent) == 0.0     # clamps instead of raising


# --- the floor spec ----------------------------------------------------------------------------
def test_heightfield_floor_normalises_by_ez_without_clipping():
    field = tr.waves(0.10, 2.0, n=SMALL)
    floor = tr.HeightfieldFloor(field, extent=SMALL * tr.HSCALE)
    n = floor.normalized()
    assert n.shape == (SMALL * SMALL,)
    assert n.max() == pytest.approx(0.10 / tr.EZ, rel=1e-5)
    np.testing.assert_allclose(n.reshape(field.shape) * tr.EZ, field, atol=1e-7)


def test_heightfield_floor_rejects_relief_it_would_have_to_clip():
    """Silent clipping is how a run reports 12 cm of terrain and walks over 15 cm-capped ground."""
    with pytest.raises(ValueError, match="ez"):
        tr.HeightfieldFloor(tr.waves(0.30, 2.0, n=SMALL))
    with pytest.raises(ValueError, match="square"):
        tr.HeightfieldFloor(np.zeros((4, 7), np.float32))


def test_heightfield_floor_declares_a_solid_field():
    root = ET.fromstring("<mujoco><worldbody/></mujoco>")
    floor = tr.HeightfieldFloor(tr.flat(SMALL), extent=6.4)
    attrs = floor.geom_attrs(root, with_visuals=False)
    assert attrs["type"] == "hfield" and attrs["hfield"] == "terrain"
    hf = root.find("asset/hfield")
    assert (hf.get("nrow"), hf.get("ncol")) == (str(SMALL), str(SMALL))
    rx, ry, ez, bz = (float(v) for v in hf.get("size").split())
    assert (rx, ry) == (3.2, 3.2)                  # RADII, not extent
    assert ez == tr.EZ
    assert bz > 0.0                                # or the field is a shell, not solid ground


def test_floor_data_reaches_the_compiled_model():
    """`finalize` is the half of the protocol that runs AFTER compilation. If it silently no-ops,
    every terrain run is a flat-ground run with a terrain-shaped label."""
    field = tr.stepping_stones(0.45, 0.03, seed=7, n=SMALL)
    floor = tr.HeightfieldFloor(field, extent=SMALL * tr.HSCALE)
    root = ET.fromstring("<mujoco><worldbody/></mujoco>")
    attrs = floor.geom_attrs(root, with_visuals=False)
    g = ET.SubElement(root.find("worldbody"), "geom")
    for k, v in attrs.items():
        g.set(k, v)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # An `<hfield>` with no `file` compiles to all zeros -- i.e. FLAT. That is precisely why
    # `finalize` has to run, and why a no-op there would be invisible without this test.
    assert not m.hfield_data.any()
    floor.finalize(m)
    assert m.hfield_data.std() > 0.0
    np.testing.assert_allclose(m.hfield_data.reshape(field.shape) * tr.EZ, field, atol=1e-6)


# --- the shared builder ------------------------------------------------------------------------
def test_terrain_model_shares_run_policys_assembly():
    """The whole point of the refactor: a terrain model differs from a flat one in the FLOOR and
    nothing else. Any drift in the collision set, actuators or contact parameters shows up here."""
    policy = rp.load_policy("baseline")
    field = tr.stepping_stones(0.45, 0.03, seed=1, n=128)
    plane = rp.build_sim_model(policy, with_visuals=False)
    rough = tr.build_terrain_model(policy, field, extent=128 * tr.HSCALE)

    assert (rough.nq, rough.nv, rough.nu, rough.ngeom) == (plane.nq, plane.nv, plane.nu, plane.ngeom)
    np.testing.assert_allclose(rough.body_mass, plane.body_mass)
    np.testing.assert_allclose(rough.actuator_gainprm, plane.actuator_gainprm)
    np.testing.assert_allclose(rough.actuator_forcerange, plane.actuator_forcerange)
    np.testing.assert_allclose(rough.dof_damping, plane.dof_damping)

    fid = mujoco.mj_name2id(rough, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pid = mujoco.mj_name2id(plane, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert rough.geom_type[fid] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert plane.geom_type[pid] == mujoco.mjtGeom.mjGEOM_PLANE
    # ... and everything the floor carries besides its shape is still SCS2's.
    np.testing.assert_allclose(rough.geom_friction[fid], plane.geom_friction[pid])
    np.testing.assert_allclose(rough.geom_solref[fid], plane.geom_solref[pid])
    assert rough.geom_condim[fid] == plane.geom_condim[pid]
    assert (rough.geom_contype[fid], rough.geom_conaffinity[fid]) == \
           (plane.geom_contype[pid], plane.geom_conaffinity[pid])
    np.testing.assert_allclose(rough.hfield_data.reshape(field.shape) * tr.EZ, field, atol=1e-6)


def test_default_floor_reproduces_the_pre_refactor_plane():
    """`floor=None` must be EXACTLY what `build_sim_model` inlined before it grew the parameter.

    Pinned against the literal values, not against `PlaneFloor()` -- comparing the default path to
    itself would agree no matter what either of them became (TERRAIN.md §7, last bullet).
    """
    policy = rp.load_policy("baseline")
    m = rp.build_sim_model(policy, with_visuals=False)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert m.geom_type[fid] == mujoco.mjtGeom.mjGEOM_PLANE
    np.testing.assert_allclose(m.geom_size[fid], [20.0, 20.0, 0.1])
    np.testing.assert_allclose(m.geom_pos[fid], [0.0, 0.0, 0.0])
    assert m.geom_condim[fid] == int(rp.CONTACT["condim"])
    assert m.geom_contype[fid] == int(rp.TERRAIN_GROUP["contype"])
    assert m.geom_conaffinity[fid] == int(rp.TERRAIN_GROUP["conaffinity"])
    np.testing.assert_allclose(rp.build_sim_model(policy, with_visuals=False,
                                                 floor=rp.PlaneFloor()).geom_size[fid],
                               m.geom_size[fid])
