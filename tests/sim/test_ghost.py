"""The estimator ghost: a translucent robot drawn at the estimated state.

Two properties matter, and they pull in opposite directions:

* the ghost must show the estimate FAITHFULLY (otherwise it lies about the error you are trying
  to see), and
* it must not touch the simulation AT ALL (otherwise every A/B in `EXPERIMENTS.md` taken with the
  viewer open is invalid).

The physics-free test is the decisive one and is written the same way as
`test_sensors_do_not_change_the_dynamics`: same seed, bit-identical `qpos`, `atol=0`.

None of these tests build the estimator -- the ghost consumes an `EstimateView`-shaped object and
does not care where it came from, so a tiny stub keeps the whole file at a few seconds. Every
positive assertion is paired with a MUTATION CHECK, because "the ghost matches the estimate" is
trivially true of a ghost that ignores the estimate and sits at `qpos0`.
"""

import mujoco
import numpy as np
import pytest

import run_policy as rp
from invariant_estimation.sim.ghost import Ghost


class _Est:
    """An `EstimateView`-shaped stub: the ghost only reads `.p`, `.R`, `.q`."""

    def __init__(self, p, R, q):
        self.p, self.R, self.q = np.asarray(p), np.asarray(R), np.asarray(q)


def _rot_y(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), 0.0, np.sin(t)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(t), 0.0, np.cos(t)]])


@pytest.fixture(scope="module")
def policy():
    return rp.load_policy("baseline")


@pytest.fixture(scope="module")
def model(policy):
    # `with_visuals=True` deliberately: the ghost is a RENDERING feature and the geoms it appends
    # are the visual meshes, so a visuals-free model would leave the drawing tests asserting
    # nothing (they did, on the first run -- 0 dynamic geoms). Visual meshes add no mass, geom or
    # collision, so the physics-free test below is unaffected by the choice.
    return rp.build_sim_model(policy, with_visuals=True, with_imu_sensors=False)


@pytest.fixture(scope="module")
def maps(model, policy):
    return rp.make_maps(model, policy)


# The 9 joints the Alex estimator filters are a leg-shaped subset; for the ghost's purposes any
# fixed 9 policy slots exercise the same scatter, so the tests stay independent of the filter build.
SLOTS = np.arange(9)


@pytest.fixture
def ghost(model, maps):
    return Ghost(model, maps, SLOTS, mode="full")


def _rest_est(model, maps, d):
    """An estimate that is simply the current true state, as a starting point to perturb."""
    return _Est(d.xpos[maps["BASE_BID"]].copy(),
                d.xmat[maps["BASE_BID"]].reshape(3, 3).copy(),
                d.qpos[maps["QADR"]][SLOTS].copy())


# ---------------------------------------------------------------------------
# The decisive property: the ghost cannot touch the simulation
# ---------------------------------------------------------------------------

def test_ghost_does_not_change_the_dynamics(model, policy, maps):
    """Same seed, ghost on vs ghost off -> bit-identical `qpos`.

    `mj_kinematics` on a SECOND `MjData` cannot write anything the integrator reads, but that is
    an argument, not a test. This is the test.
    """
    states = []
    for use_ghost in (False, True):
        d = mujoco.MjData(model)
        d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
        d.qpos[0:3] = [0.0, 0.0, rp.foot_rest_height(model, maps)]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        g = Ghost(model, maps, SLOTS, mode="full") if use_ghost else None
        scn = mujoco.MjvScene(model, maxgeom=100000)
        for _ in range(200):
            mujoco.mj_step(model, d)
            if g is not None:
                # A deliberately WILD estimate: if the ghost leaked into the sim at all, an
                # estimate this far from truth would make the difference unmistakable.
                est = _Est([5.0, -3.0, 2.0], _rot_y(60.0),
                           d.qpos[maps["QADR"]][SLOTS] + 0.5)
                g.update(est, d)
                scn.ngeom = 0
                g.draw(scn)
        states.append(d.qpos.copy())
    np.testing.assert_array_equal(states[0], states[1])


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def test_ghost_tracks_the_estimate(model, maps, ghost):
    """Pelvis pose == est.p/est.R, filtered joints == est.q, to round-off."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)

    est = _Est([0.3, -0.2, 0.95], _rot_y(45.0),
               d.qpos[maps["QADR"]][SLOTS] + 0.07)
    ghost.update(est, d)

    bid = maps["BASE_BID"]
    np.testing.assert_allclose(ghost.data.xpos[bid], est.p, atol=1e-12)
    np.testing.assert_allclose(ghost.data.xmat[bid].reshape(3, 3), est.R, atol=1e-12)
    np.testing.assert_allclose(ghost.data.qpos[ghost.filtered_qadr], est.q, atol=1e-12)


def test_ghost_geoms_move_when_the_estimate_moves(model, maps, ghost):
    """MUTATION CHECK for the test above: perturb the attitude by 15 deg, the drawn geoms move.

    Without this, a ghost that silently ignored `est` and drew the robot at `qpos0` would still
    pass a "ghost matches estimate" test written against a rest-pose estimate.
    """
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    scn = mujoco.MjvScene(model, maxgeom=100000)

    def drawn(est):
        ghost.update(est, d)
        scn.ngeom = 0
        n = ghost.draw(scn)
        return np.array([scn.geoms[i].pos.copy() for i in range(n)])

    base = _rest_est(model, maps, d)
    a = drawn(base)
    b = drawn(_Est(base.p, _rot_y(15.0) @ base.R, base.q))
    assert a.shape == b.shape and len(a) > 0
    assert np.abs(a - b).max() > 1e-3, "the ghost ignored a 15 deg attitude change"

    # ... and to the joints, which is a different code path (scatter, not the free joint).
    c = drawn(_Est(base.p, base.R, base.q + 0.3))
    assert np.abs(a - c).max() > 1e-3, "the ghost ignored a 0.3 rad joint change"


def test_unfiltered_joints_come_from_truth(model, maps, ghost):
    """The ghost knows the 9 filtered joints from the filter and the other 20 from the encoders."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    d.qpos[maps["ALL_QADR"]] += 0.11          # move every joint away from qpos0

    est = _rest_est(model, maps, d)
    est.q = est.q + 0.4                        # the filtered ones disagree with truth
    ghost.update(est, d)

    filt = set(ghost.filtered_qadr.tolist())
    other = [a for a in maps["ALL_QADR"] if a not in filt]
    np.testing.assert_allclose(ghost.data.qpos[other], d.qpos[other], atol=0)
    np.testing.assert_allclose(ghost.data.qpos[ghost.filtered_qadr], est.q, atol=1e-12)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_attitude_mode_pins_position_but_keeps_estimated_orientation(model, maps):
    """Isolates orientation error from the height drift: position exact, rotation still estimated."""
    g = Ghost(model, maps, SLOTS, mode="attitude")
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)

    bid = maps["BASE_BID"]
    est = _Est([9.0, 9.0, 9.0], _rot_y(30.0),          # a position the ghost must IGNORE
               d.qpos[maps["QADR"]][SLOTS].copy())
    g.update(est, d)

    np.testing.assert_allclose(g.data.xpos[bid], d.xpos[bid], atol=0)
    np.testing.assert_allclose(g.data.xmat[bid].reshape(3, 3), est.R, atol=1e-12)

    # And the same estimate in `full` mode DOES take the position -- so the pin is mode-specific.
    g.mode = "full"
    g.update(est, d)
    np.testing.assert_allclose(g.data.xpos[bid], est.p, atol=1e-12)


def test_off_mode_draws_nothing_and_cycle_visits_every_mode(model, maps):
    g = Ghost(model, maps, SLOTS)
    assert g.mode == "off" and not g.on

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    scn = mujoco.MjvScene(model, maxgeom=100000)
    scn.ngeom = 0
    g.update(_rest_est(model, maps, d), d)
    assert g.draw(scn) == 0 and scn.ngeom == 0

    assert [g.cycle(), g.cycle(), g.cycle()] == ["full", "attitude", "off"]


def test_offset_displaces_the_ghost_laterally_only(model, maps):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    est = _rest_est(model, maps, d)

    a = Ghost(model, maps, SLOTS, mode="full")
    b = Ghost(model, maps, SLOTS, mode="full", offset=0.8)
    a.update(est, d)
    b.update(est, d)
    delta = b.data.xpos[maps["BASE_BID"]] - a.data.xpos[maps["BASE_BID"]]
    np.testing.assert_allclose(delta, [0.0, 0.8, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_draw_appends_the_robot_without_sites_or_the_floor(model, maps, ghost):
    """The dedicated `MjvOption` must suppress the 20 model sites; the floor is static, so the
    dynamic pass excludes it for free."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    ghost.update(_rest_est(model, maps, d), d)

    scn = mujoco.MjvScene(model, maxgeom=100000)
    scn.ngeom = 0
    n_ghost = ghost.draw(scn)

    scn.ngeom = 0
    mujoco.mjv_addGeoms(model, ghost.data, mujoco.MjvOption(), mujoco.MjvPerturb(),
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
    n_stock = scn.ngeom

    assert n_ghost > 0
    assert n_stock - n_ghost == model.nsite, (
        f"expected the stock option to add {model.nsite} extra site geoms, "
        f"got {n_stock} vs {n_ghost}")


# ---------------------------------------------------------------------------
# Wiring: the existing command plumbing must reach the ghost
# ---------------------------------------------------------------------------

def test_the_keypad_and_terminal_bindings_reach_the_ghost(model, policy, maps):
    """One BINDINGS entry should give the keypad, the terminal letter and the help text at once."""
    assert rp._BY_KEY[332] == "ghost"          # keypad *, a code MuJoCo does not reserve
    assert rp._BY_CHAR["g"] == "ghost"
    assert "ghost" in rp.KEYMAP_HELP

    loop = rp.Loop(model, policy, maps)
    assert loop.ghost is None
    loop.command("ghost")                       # must be a harmless no-op with nothing attached

    loop.ghost = Ghost(model, maps, SLOTS)
    assert loop.ghost.mode == "off"
    loop.key(332)                               # the viewer's key_callback path
    assert loop.ghost.mode == "full"
    loop.command(rp._BY_CHAR["g"])              # the stdin_commands path
    assert loop.ghost.mode == "attitude"


def test_ghost_binding_does_not_collide_with_a_mujoco_render_toggle():
    """The reason this binding is on the keypad at all: MuJoCo reserves every letter A-Z."""
    assert 332 >= 128, "a sub-128 code would be a letter and collide with a render toggle"
    assert "G" in rp.RESERVED_KEYS, (
        "if MuJoCo ever frees up G, the keypad indirection could be dropped -- but not before")


def test_draw_appends_rather_than_clobbering_and_tints_only_its_own(model, maps, ghost):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    ghost.update(_rest_est(model, maps, d), d)

    scn = mujoco.MjvScene(model, maxgeom=100000)
    scn.ngeom = 0
    mujoco.mjv_addGeoms(model, d, mujoco.MjvOption(), mujoco.MjvPerturb(),
                        mujoco.mjtCatBit.mjCAT_STATIC, scn)     # the floor
    n_before = scn.ngeom
    pre = scn.geoms[0].rgba.copy()

    n = ghost.draw(scn)
    assert scn.ngeom == n_before + n
    np.testing.assert_array_equal(scn.geoms[0].rgba, pre)        # the floor is untouched
    for i in range(n_before, scn.ngeom):
        assert scn.geoms[i].rgba[3] == pytest.approx(0.35)
