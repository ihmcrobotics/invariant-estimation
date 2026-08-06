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

import time

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


# ---------------------------------------------------------------------------
# Threaded mode (--realtime)
# ---------------------------------------------------------------------------

def test_threaded_mode_agrees_with_synchronous_within_noise():
    """The acceptance gate for `--realtime`: a stale estimate must not be a WORSE estimate.

    Deliberately NOT a bit-equality check -- the whole point of threading is that the estimate the
    policy reads is a few ticks old, so the trajectories diverge. What must hold is that the error
    SUMMARY is unchanged within noise, because a threaded run that quietly doubled the tilt error
    would still look fine on screen.

    The 0.3 deg gate is what set the default `max_backlog_ticks`. Measured on this machine at
    vx=0.6: backlog 1 -> 0.099 deg, 2 -> 0.112, 3 -> 0.456, 5 -> 1.598. Staleness tracks the
    allowed backlog almost exactly (age ~= backlog + 1 ticks), and attitude error degrades sharply
    past ~3 ticks -- which is why the default is 2 and not the 5 originally planned.

    250 ticks, not fewer: `summarise` scores the TAIL HALF, and at 120 ticks that window
    (1.2-2.4 s) is still inside the gait transient, where the two trajectories differ by more than
    the estimator does -- a 120-tick version of this test read 1.157 vs 0.738 deg and failed on
    startup noise alone. By 250 ticks the tail is settled walking and both numbers are
    reproducible to 1e-3 across repeats.
    """
    ticks = 250
    out = {}
    for threaded in (False, True):
        lp = re_.make_estimated_loop("baseline", with_visuals=False, verbose=False,
                                     threaded=threaded)
        lp.cmd[0:3] = (0.6, 0.0, 0.0)
        lp.cmd[3] = 0.0
        lp.start_thread()
        try:
            for _ in range(ticks):
                lp.control_tick()
        finally:
            lp.stop_thread()
        out[threaded] = (re_.summarise(lp.history), lp)

    sync_s, thr_s = out[False][0], out[True][0]
    assert abs(sync_s["tilt_deg_tail_rms"] - thr_s["tilt_deg_tail_rms"]) < 0.3, (
        f"threaded tilt {thr_s['tilt_deg_tail_rms']:.3f} vs sync "
        f"{sync_s['tilt_deg_tail_rms']:.3f} deg -- staleness is costing accuracy")

    # Staleness must be REPORTED, not silent: it is the number that explains any divergence.
    ages = np.array([h["age_ticks"] for h in out[True][1].history])
    assert ages.max() > 0, "threaded mode reported zero staleness -- the age is not being tracked"
    assert ages.max() <= out[True][1]._max_backlog_ticks + 2, f"age ran away to {ages.max()}"
    # ... and the synchronous path must report exactly zero, since its estimate IS this tick's.
    assert all(h["age_ticks"] == 0.0 for h in out[False][1].history)


def test_threaded_mode_loses_no_sensor_sample():
    """Through the REAL loop, not the stub: everything submitted is consumed or still queued."""
    lp = re_.make_estimated_loop("baseline", with_visuals=False, verbose=False, threaded=True)
    lp.start_thread()
    try:
        for _ in range(60):
            lp.control_tick()
        te = lp.te
        # Drain so nothing is in flight, then account exactly.
        for _ in range(500):
            if te.backlog < te.substeps:
                break
            time.sleep(0.005)
        time.sleep(0.2)
        consumed, backlog = te.latest().seq, te.backlog
        submitted = lp._submitted
    finally:
        lp.stop_thread()
    assert consumed + backlog == submitted, (
        f"submitted {submitted} but accounted {consumed}+{backlog}; a sample was lost")
    assert consumed % te.substeps == 0, "the worker consumed a partial chunk (XLA retrace)"


def test_headless_stays_synchronous_by_default():
    """The determinism gate depends on this: nothing threads unless asked."""
    lp = re_.make_estimated_loop("baseline", with_visuals=False, verbose=False)
    assert lp.threaded is False and lp.te is None
    lp.start_thread()                      # must be a no-op, not an accidental opt-in
    assert lp.te is None
    lp.control_tick()
    assert lp.history[-1]["age_ticks"] == 0.0
