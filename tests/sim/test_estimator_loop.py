"""The closed loop: estimator in the sim, policy on the estimate (G10).

`test_sensors.py` checks the boundary; this checks the loop that runs on top of it. The tests
that matter here are the ones that fail when the wiring is dead rather than wrong:

* `test_the_estimate_actually_reaches_the_policy` — a deliberately corrupted estimate MUST change
  the robot's behaviour. Without it, every "the robot stands" assertion below would pass just as
  happily with the estimator computed and thrown away (the failure mode called out in
  `estimator-port-testing-discipline`: a suite that verifies "present", not "right").
* `test_the_scanned_step_is_one_compiled_program` — I7 in the loop, across a contact-state change.

Building the Alex estimator traces MJX kinematics over 49 links (~45 s), so the fixtures are
module-scoped and the module is marked `slow`.
"""

import numpy as np
import pytest

import run_estimator as re_
import run_policy as rp
from invariant_estimation.sim.estimator_loop import attitude_error_deg, tilt_error_deg

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def loop():
    lp = re_.make_estimated_loop("baseline", with_visuals=False, verbose=False)
    for _ in range(60):                    # 1.2 s: touchdown transient over, stance settled
        lp.control_tick()
    return lp


# ---------------------------------------------------------------------------
# It runs, and it is right
# ---------------------------------------------------------------------------

def test_the_robot_stands_on_its_own_estimate(loop):
    """The headline: closed loop, upright, finite, both feet found."""
    assert np.all(np.isfinite(loop.d.qpos))
    assert loop.d.qpos[2] > 0.80, f"fell to z={loop.d.qpos[2]:.3f}"
    assert loop.tilt_deg() < 10.0
    assert loop.history[-1]["trusted"] == 2.0, "a standing robot must trust both feet"


def test_attitude_tracks_truth(loop):
    """Tilt is the component the policy consumes; it is the one gravity leveling owns."""
    tail = loop.history[-25:]
    assert np.mean([h["tilt_deg"] for h in tail]) < 1.0
    assert np.mean([h["att_deg"] for h in tail]) < 1.0


def test_base_gyro_tracks_truth(loop):
    """`omega_body` = R_mount·(gyro − b) must land in the pelvis BODY frame.

    Ground truth is `mj_objectVelocity(mjOBJ_XBODY)` — an independent path. A wrong `R_mount`
    (or the `mjOBJ_BODY` inertial-frame trap) permutes axes and fails here even though the norm
    would still match, so compare componentwise.
    """
    est, truth = loop.rt.last, loop.reader.truth(loop.d)
    assert np.linalg.norm(est.omega_body - truth["omega"]) < 0.05


def test_joint_states_track_truth(loop):
    est, truth = loop.rt.last, loop.reader.truth(loop.d)
    assert np.abs(est.q - truth["q"]).max() < 5.0e-3
    assert np.abs(est.q_dot - truth["q_dot"]).max() < 0.5


def test_bias_stays_small_without_injected_bias(loop):
    """With clean sensors there is no bias to find; the estimate must not invent one."""
    assert loop.rt.last.bias.max() < 0.05


# ---------------------------------------------------------------------------
# The wiring is live (mutation checks)
# ---------------------------------------------------------------------------

def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_the_estimate_actually_reaches_the_policy():
    """Corrupt the published attitude by 15° of roll; the robot must react.

    This is the test that distinguishes "the estimator drives the policy" from "the estimator
    runs next to the policy". Both loops are otherwise identical and deterministic, so any
    divergence is attributable to the injected error.
    """
    clean = re_.make_estimated_loop("baseline", with_visuals=False, verbose=False)
    tilted = re_.make_estimated_loop("baseline", with_visuals=False, verbose=False)

    real_advance = tilted.rt.advance

    def lying_advance(batch):
        est = real_advance(batch)
        est.R = _rot_x(np.radians(15.0)) @ est.R
        return est

    tilted.rt.advance = lying_advance
    for _ in range(40):
        clean.control_tick()
        tilted.control_tick()

    assert np.abs(clean.last_action - tilted.last_action).max() > 1e-3, \
        "the policy's action did not change when the estimated attitude did"
    assert abs(clean.d.qpos[2] - tilted.d.qpos[2]) > 1e-4, "the robot's motion did not change"


def test_sources_truth_bypasses_the_estimate():
    """`--source truth` must reproduce `run_policy` exactly: same plant, same obs, same action.

    The A/B control. If this drifts, the estimator is leaking into the loop through some path
    other than the declared `sources`.
    """
    bypass = re_.make_estimated_loop("baseline", with_visuals=False, sources=(), verbose=False)
    policy = rp.load_policy("baseline")
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True)
    plain = rp.Loop(m, policy, rp.make_maps(m, policy))
    for _ in range(30):
        bypass.control_tick()
        plain.control_tick()
    np.testing.assert_allclose(bypass.d.qpos, plain.d.qpos, atol=0, rtol=0)
    assert bypass.history, "the estimator must still run (and be scored) under --source truth"


# ---------------------------------------------------------------------------
# I7 in the loop
# ---------------------------------------------------------------------------

def test_the_scanned_step_is_one_compiled_program(loop):
    """A contact-state change must not retrace (CLAUDE.md §4, I7).

    Measured by the lowered HLO, never by `_cache_size()` — a full-suite run evicts that global
    LRU (`jit-cache-size-lru-trap`).
    """
    import jax
    import jax.numpy as jnp

    def stacked(trust):
        batch = [loop.reader.read(loop.d) for _ in range(rp.DECIMATION)]
        batch = [b._replace(
            contact=np.full_like(b.contact, trust),
            contact_chol=np.tile(np.eye(3) * (1e-4 if trust else 1e1), (len(b.contact), 1, 1)))
            for b in batch]
        return jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs), dtype=jnp.float64), *batch)

    both = loop.rt._advance.lower(loop.rt.carry, stacked(1.0)).as_text()
    none = loop.rt._advance.lower(loop.rt.carry, stacked(0.0)).as_text()
    assert both == none, "the contact mask changed the compiled program"


def test_seeding_starts_from_truth(loop):
    """The seed is the sim's own pose, so the very first estimate is not a transient."""
    first = loop.history[0]
    assert first["att_deg"] < 1e-6
    assert first["p_err"] < 5e-2


def test_error_metrics_are_what_they_claim():
    """`tilt_error_deg` ignores yaw; `attitude_error_deg` does not."""
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 90° about z
    assert tilt_error_deg(yaw, np.eye(3)) < 1e-9
    assert abs(attitude_error_deg(yaw, np.eye(3)) - 90.0) < 1e-9
    assert abs(tilt_error_deg(_rot_x(np.radians(7.0)), np.eye(3)) - 7.0) < 1e-9
