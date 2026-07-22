"""The filter tick and the scan — `jointKF/filter.py`.

No single Java class corresponds to this file: the Java suite exercises the
orchestration through `JointLevelKFFilterTest` and `JointLevelKFTrajectoryTest`
(gate G8, behavioural) rather than structurally. What is tested here is the
wiring the port has to get right *because* it is fixed-shape and pure — the
things that have no Java analogue because Java simply reshapes:

* the one-tick delay on the trusted-feet mask (CLAUDE.md §4 phase ordering),
* that the two channels gate independently, so a bad gyro does not cost the
  encoders,
* and the constant-graph property (I7).

On what the graph tests prove. Traced arrays cannot change a jaxpr, so equality
across contact patterns is nearly automatic; what these genuinely catch is a
data-dependent branch, which raises at trace time. `_cache_size() == 1` is the
load-bearing assertion — it says no *recompilation* happened, which is the port's
analogue of the Java allocation guard (CLAUDE.md §3, "Skip"). Stated as what it
proves, not what one might hope (JOINTKF_PORT_PLAN §4 lesson 3).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.jointKF import anchors as anchors_mod, measure
from invariant_estimation.jointKF.build import build_joint_kf
from invariant_estimation.jointKF.filter import (
    FilterCarry,
    ModelInputs,
    SensorInputs,
    init_carry,
    run,
    step,
)
from invariant_estimation.jointKF.state import default_params

from . import _fixture as fx
from ._fixture import kinematic_tree
from ._oracles import SHAPES, assert_positive_semidefinite, assert_symmetric

SHAPE = SHAPES[0]


@pytest.fixture(scope="module")
def scene():
    """A real MJX chain with one anchor slot, and one tick of consistent motion."""
    f = fx.fixture(SHAPE["name"])
    tree = kinematic_tree(f)
    build = build_joint_kf(
        tree,
        imu_sites=list(f.imu_names),
        pairs=[tuple(p) for p in f.pairs],
        foot_sites=[f.foot_site],
    )
    params = default_params()
    motion = f.apply_consistent_motion(np.zeros(f.n), np.zeros(f.n))
    q = jnp.asarray(motion.q, dtype=jnp.float64)
    J_rel, R_rel = measure.pair_frames(f.model, q)
    ev = f.model.evaluate(q)
    # `anchor_jacobians` indexes into `ev.J_ang` / `ev.site_rot`, which are in
    # `model.site_names` order -- IMU sites first, then the foot.
    names = list(f.model.site_names)
    jac = anchors_mod.anchor_jacobians(
        build, ev.J_ang, ev.site_rot,
        base_site=names.index(f.imu_names[build.base_imu]),
        foot_sites=np.array([names.index(f.foot_site)]),
    )
    # The fixture MJCF carries `armature`, so `ev.M` already includes the rotor
    # inertia pre-Schur -- which is why `filter.step` passes the
    # ROTOR_IN_MASS_MATRIX sentinel and never adds it again (CLAUDE.md §6).
    model = ModelInputs(J_rel=J_rel, R_rel=R_rel, anchor_jac=jac, M=ev.M)
    return f, build, params, model, motion


def sensors_for(build, motion, contact):
    n_u = build.anchor_unfiltered_mask.shape[1]
    return SensorInputs(
        encoders=jnp.asarray(motion.q, dtype=jnp.float64),
        gyros=jnp.asarray(motion.gyro, dtype=jnp.float64),
        qd_unfiltered=jnp.zeros(n_u, dtype=jnp.float64),
        contact=jnp.asarray(contact, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------

def test_one_tick_keeps_the_covariance_symmetric_psd(scene):
    f, build, params, model, motion = scene
    carry = init_carry(build, params, jnp.asarray(motion.q))
    carry, diag = step(carry, sensors_for(build, motion, [1.0]), model, build, params)

    P = np.asarray(carry.state.P)
    assert np.all(np.isfinite(np.asarray(carry.state.x)))
    assert_symmetric(P, 1.0e-9, "P after one tick")
    assert_positive_semidefinite(P, "P after one tick")


def test_trusted_feet_are_delayed_by_exactly_one_tick(scene):
    """The mask is written at the end of step k and read at the start of k+1.

    Using *this* tick's contact would correlate the gating decision with the
    measurement it gates, through the shared sensor noise — which biases the very
    bias estimate the anchor exists to make observable. The delay is the fix, and
    it is invisible unless asserted directly.
    """
    f, build, params, model, motion = scene
    carry = init_carry(build, params, jnp.asarray(motion.q))
    assert float(carry.trusted_feet[0]) == 0.0, "feet start untrusted"

    # Tick 0 reports contact; the anchor must NOT yet be active this tick.
    carry, diag0 = step(carry, sensors_for(build, motion, [1.0]), model, build, params)
    assert float(diag0.active_anchors) == 0.0
    assert float(carry.trusted_feet[0]) == 1.0, "mask carried forward"

    # Tick 1 now sees it.
    carry, diag1 = step(carry, sensors_for(build, motion, [0.0]), model, build, params)
    assert float(diag1.active_anchors) == 1.0
    assert float(carry.trusted_feet[0]) == 0.0, "release also delayed"


def test_a_nan_gyro_does_not_cost_the_encoder_update(scene):
    """Channels gate independently — the reason they are two Joseph updates.

    A single stacked block would mean one bad IMU throws the encoders away too.
    That is a behavioural difference, not a numerical one, and it is exactly what
    `testTransientNonFiniteInputRecovers` cares about.
    """
    f, build, params, model, motion = scene
    carry = init_carry(build, params, jnp.asarray(motion.q))
    s = sensors_for(build, motion, [0.0])
    s = s._replace(gyros=s.gyros.at[0].set(jnp.nan))

    carry, diag = step(carry, s, model, build, params)
    assert float(diag.stacked_applied) == 0.0, "poisoned gyro update must be skipped"
    assert float(diag.encoder_applied) == 1.0, "encoders must survive it"
    assert np.all(np.isfinite(np.asarray(carry.state.x))), "no NaN may propagate"
    assert np.all(np.isfinite(np.asarray(carry.state.P)))


def test_recovery_is_automatic_after_a_bad_window(scene):
    """No latch: once the input is clean again the channel applies immediately."""
    f, build, params, model, motion = scene
    carry = init_carry(build, params, jnp.asarray(motion.q))
    bad = sensors_for(build, motion, [0.0])
    bad = bad._replace(gyros=bad.gyros.at[0].set(jnp.nan))
    good = sensors_for(build, motion, [0.0])

    for _ in range(5):
        carry, diag = step(carry, bad, model, build, params)
        assert float(diag.stacked_applied) == 0.0
    carry, diag = step(carry, good, model, build, params)
    assert float(diag.stacked_applied) == 1.0, "gate latched — it must not"


# ---------------------------------------------------------------------------
# Constant graph (I7)
# ---------------------------------------------------------------------------

def test_step_does_not_recompile_across_contact_patterns(scene):
    """The load-bearing constant-graph assertion: no recompilation.

    This is the port's analogue of the Java allocation guard, which is skipped as
    JVM-specific (CLAUDE.md §3). A data-dependent branch would raise; a changed
    shape would recompile. `_cache_size() == 1` rules out both.
    """
    f, build, params, model, motion = scene
    jstep = jax.jit(lambda c, s: step(c, s, model, build, params))

    carry = init_carry(build, params, jnp.asarray(motion.q))
    for contact in ([0.0], [1.0], [0.0], [1.0]):
        carry, _ = jstep(carry, sensors_for(build, motion, contact))
    assert jstep._cache_size() == 1, "contact pattern must not change the graph"


def test_run_scans_a_trajectory(scene):
    """`run` over 50 ticks: finite, PSD, and one compiled tick regardless of length."""
    f, build, params, model, motion = scene
    ticks = 50
    s1 = sensors_for(build, motion, [1.0])
    contact = np.zeros((ticks, build.n_anchors))
    contact[20:] = 1.0                                  # a touchdown mid-trajectory
    traj = SensorInputs(
        encoders=jnp.tile(s1.encoders, (ticks, 1)),
        gyros=jnp.tile(s1.gyros, (ticks, 1, 1)),
        qd_unfiltered=jnp.tile(s1.qd_unfiltered, (ticks, 1)),
        contact=jnp.asarray(contact),
    )
    models = jax.tree.map(lambda a: jnp.broadcast_to(a, (ticks,) + a.shape), model)

    carry = init_carry(build, params, jnp.asarray(motion.q))
    final, diag = run(carry, traj, models, build, params)

    P = np.asarray(final.state.P)
    assert np.all(np.isfinite(np.asarray(final.state.x)))
    assert_symmetric(P, 1.0e-9, "P after 50 ticks")
    assert_positive_semidefinite(P, "P after 50 ticks")
    # Anchors switch on one tick AFTER contact is reported (phase ordering).
    active = np.asarray(diag.active_anchors)
    assert active[20] == 0.0 and active[21] == 1.0


def test_covariance_stays_bounded_over_the_scan(scene):
    """A measurement-free-ish run must not let P run away (Java testCovarianceBounded)."""
    f, build, params, model, motion = scene
    ticks = 200
    s1 = sensors_for(build, motion, [1.0])
    traj = SensorInputs(
        encoders=jnp.tile(s1.encoders, (ticks, 1)),
        gyros=jnp.tile(s1.gyros, (ticks, 1, 1)),
        qd_unfiltered=jnp.tile(s1.qd_unfiltered, (ticks, 1)),
        contact=jnp.ones((ticks, build.n_anchors)),
    )
    models = jax.tree.map(lambda a: jnp.broadcast_to(a, (ticks,) + a.shape), model)
    carry = init_carry(build, params, jnp.asarray(motion.q))
    initial_trace = float(jnp.trace(carry.state.P))
    final, _ = run(carry, traj, models, build, params)
    final_trace = float(jnp.trace(final.state.P))

    assert np.isfinite(final_trace)
    assert final_trace <= initial_trace * 10.0 + 1.0, (
        f"trace grew {initial_trace:.3e} -> {final_trace:.3e}"
    )
