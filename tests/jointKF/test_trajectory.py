"""Port of `JointLevelKFTrajectoryTest` (8 tests) plus
`JointLevelKFPreFilterAllocationTest.testHotPathStaysFinite`.

Gate G8's behavioural half.  Where `test_whole_filter.py` drives the filter with
constant sensors, this class drives it with an **analytically consistent**
per-joint sinusoid: the encoders and every IMU gyro are derived from the *same*
true `(q, q_dot)` through `_fixture.apply_consistent_motion`, whose base twist is
exactly zero, so

    gyro_child - {}^{c}R_{p} gyro_parent  ==  J_ang(q) q_dot        exactly

(not to first order).  That exactness is the premise every tolerance below rests
on: the filter is being asked to recover a state it *could* recover perfectly, so
a 5e-3 position error is entirely the filter's, never the fixture's.

Phase 1 only — hence no trusted feet
------------------------------------
Java's `JointLevelKFTrajectoryTest` calls `computeJointState()` and **not**
`computeImuBiases(feet)`.  The port has one fused tick, so the faithful spelling
is `contact = 0` throughout: the anchor rows stay structurally present and
numerically inert (`R -> r_large`), which is exactly the fixed-shape mask design
(CLAUDE.md §4) and also keeps this file exercising it on every tick.  The
consequence is that the common-mode gyro bias is *unobservable* here, which is
what makes `test_bias_stays_small_with_zero_true_bias` a statement about drift
rather than about convergence.

Fixture and constants
---------------------
`singlePair(seed, 8, 1, 7)` — an 8-link chain, IMUs on `link1`/`link7`, foot on
`link7`, so `n = 7 - 1 = 6` filtered joints and `m = 2` IMUs (see
`test_whole_filter.py`'s module docstring for the `n` convention and for why all
tests share one geometry rather than one per Java seed).  The scalar-CWNA
process-noise path is used, per `testMassMatrixPathEnabledOnlyWithModel`.

Trajectory constants are verbatim: `AMP = 0.10` rad, `FREQ_HZ = 0.5`,
`OMEGA = 2*pi*0.5`, `DT = 1e-3`, and for joint `i`,
`phase = OMEGA*tick*DT + i*pi/n`, `q[i] = AMP sin(phase)`,
`q_dot[i] = AMP OMEGA cos(phase)`.  The per-joint phase offset matters: it makes
the joints move out of phase, so a filter that merely tracked a single scalar
would fail.

Performance
-----------
The trajectory is open loop — `q(t)` does not depend on the estimate — so every
model quantity for the whole run is precomputed **outside** the scan with a
single eager `vmap` over the tick axis and fed in as a stacked `ModelInputs`.
That is the reason `filter.step` takes model quantities as arguments (its module
docstring), and it is what keeps a 3000-tick run at ~9 s of model evaluation
(dominated by one MJX trace, essentially independent of the tick count) plus
~0.5 s of scan, instead of 3000 separate MJX passes.

`testSingularInnovationIsSkippedNotLatched` is NOT here
-------------------------------------------------------
It exercises `predict()` + `josephUpdate(H, z, R)` directly with a rank-deficient
`H` and `R = 0`, and is already ported in
`tests/jointKF/test_update.py` (the masked-`K` gate, asserted bit-identical).
Duplicating it here would test `update.py` twice and the trajectory not at all.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.jointKF import anchors as anchors_mod, predict as predict_mod, process
from invariant_estimation.jointKF.build import build_joint_kf
from invariant_estimation.jointKF.filter import ModelInputs, SensorInputs, init_carry, step
from invariant_estimation.jointKF.state import default_params

from . import _fixture as fx
from ._fixture import kinematic_tree
from ._oracles import (
    SHAPES,
    assert_all_close,
    assert_all_finite,
    assert_positive_semidefinite,
    assert_symmetric,
)

# -- trajectory constants, verbatim from the Java class ---------------------
AMP = 0.10                      # [rad]
FREQ_HZ = 0.5
OMEGA = 2.0 * np.pi * FREQ_HZ   # [rad/s]
DT = 1.0e-3                     # [s]
PEAK_VELOCITY = AMP * OMEGA     # [rad/s] — the `0.5*peak` observability floor

#: Java `singlePair(seed, numChainJoints=8, imuAfterJoint=1, footAfterJoint=7)`.
SINGLE_PAIR = {"name": "trajectory_single_pair_8_1_7", "chain": 8, "imus": (1, 7),
               "pairs": ((0, 1),), "n": 6, "m": 2}

#: Java `JointLevelKFPreFilterAllocationTest.Fixture`: a 10-joint chain with the
#: parent IMU on `joints[1].successor` and the child IMU on `joints[9].successor`,
#: one pair, `feet = [childLink]`.  That is `_oracles.SHAPES[0]` exactly.
HOT_PATH_SHAPE = SHAPES[0]

#: The Java fixture's IMUs report `diag(1e-4)` for every noise covariance — see
#: `test_whole_filter.GYRO_SIGMA` for why this must be passed explicitly.
GYRO_SIGMA = 1.0e-4


# ---------------------------------------------------------------------------
# Scene wiring
# ---------------------------------------------------------------------------

class Scene(NamedTuple):
    """A built filter plus the callable that turns a configuration into inputs.

    Deliberately duplicated from `test_whole_filter.py` rather than shared: the
    two files are read independently against two different Java classes, and a
    helper that drifted would silently change what one of them is testing.  The
    duplication is ~30 lines and is covered by both files failing loudly.
    """

    fixture: fx.ChainFixture
    build: object
    params: object
    base_site: int
    foot_sites: np.ndarray


def _scene(shape: dict) -> Scene:
    f = fx.build_fixture(shape)
    build = build_joint_kf(
        kinematic_tree(f),
        imu_sites=list(f.imu_names),
        pairs=[tuple(p) for p in f.pairs],
        foot_sites=[f.foot_site],
        use_mass_matrix=False,                       # scalar CWNA path
        gyro_sigma=lambda name: GYRO_SIGMA * np.eye(3),
    )
    names = list(f.model.site_names)
    return Scene(
        fixture=f,
        build=build,
        params=default_params(),
        base_site=names.index(f.imu_names[build.base_imu]),
        foot_sites=np.array([names.index(f.foot_site)]),
    )


def _model_inputs(scene: Scene, q):
    """`ModelInputs` at one configuration — vmappable over a whole trajectory.

    `M=None` selects the scalar-CWNA process noise, matching the Java fixture's
    no-elevator construction (`isUsingMassMatrixProcessNoise() == false`).
    """
    f, build = scene.fixture, scene.build
    ev = f.model.evaluate(jnp.asarray(q, dtype=jnp.float64))
    R = ev.site_rot[jnp.asarray(f.model.pair_sites)]
    R_rel = jnp.einsum("eji,ejk->eik", R[:, 1], R[:, 0])      # ^c R_p
    jac = anchors_mod.anchor_jacobians(
        build, ev.J_ang, ev.site_rot,
        base_site=scene.base_site, foot_sites=scene.foot_sites,
    )
    return ModelInputs(J_rel=ev.J_rel, R_rel=R_rel, anchor_jac=jac, M=None)


@pytest.fixture(scope="module")
def scene() -> Scene:
    return _scene(SINGLE_PAIR)


# ---------------------------------------------------------------------------
# Trajectory + consistent sensors
# ---------------------------------------------------------------------------

def trajectory(ticks, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Java `trajectory(tick, n, q, qd)`, vectorised over ticks.

    `phase = OMEGA*tick*DT + i*pi/n` — the per-joint offset spreads the chain
    over half a period so no two joints move together.
    """
    t = np.asarray(ticks, dtype=float)[:, None]
    i = np.arange(n, dtype=float)[None, :]
    phase = OMEGA * t * DT + i * np.pi / n
    return AMP * np.sin(phase), AMP * OMEGA * np.cos(phase)


def consistent_sensors(scene: Scene, q: np.ndarray, qd: np.ndarray) -> SensorInputs:
    """Encoders and gyros derived from the SAME `(q, q_dot)` — the whole premise.

    `contact` is zero for every tick: this class is the phase-1 port (module
    docstring), so no anchor is ever trusted.
    """
    build, f = scene.build, scene.fixture
    T = q.shape[0]
    gyro = np.stack([f.apply_consistent_motion(q[t], qd[t]).gyro for t in range(T)])
    return SensorInputs(
        encoders=jnp.asarray(q, dtype=jnp.float64),
        gyros=jnp.asarray(gyro, dtype=jnp.float64),
        qd_unfiltered=jnp.zeros((T, build.anchor_unfiltered_mask.shape[1]), dtype=jnp.float64),
        contact=jnp.zeros((T, build.n_anchors), dtype=jnp.float64),
    )


def roll(scene: Scene, ticks: int, *, poison=None):
    """Precompute the trajectory + model, scan the filter, return `(q, qd, x, P)`.

    `poison` is an optional callable `(encoders, gyros) -> (encoders, gyros)`
    applied to the stacked sensor arrays before the scan — the NaN-window test's
    only extra machinery.

    Everything model-derived is evaluated once, eagerly vmapped over the tick
    axis, and passed in stacked; the scan itself compiles a single tick.
    """
    build, params, n = scene.build, scene.params, scene.fixture.n
    q, qd = trajectory(np.arange(ticks), n)
    sensors = consistent_sensors(scene, q, qd)
    if poison is not None:
        enc, gyr = poison(np.asarray(sensors.encoders).copy(), np.asarray(sensors.gyros).copy())
        sensors = sensors._replace(encoders=jnp.asarray(enc), gyros=jnp.asarray(gyr))

    models = jax.vmap(lambda qq: _model_inputs(scene, qq))(jnp.asarray(q))

    def body(carry, inputs):
        s, mdl = inputs
        carry, _diag = step(carry, s, mdl, build, params)
        return carry, (carry.state.x, carry.state.P)

    carry = init_carry(build, params, jnp.asarray(q[0], dtype=jnp.float64))
    _, (xs, Ps) = jax.jit(lambda c: jax.lax.scan(body, c, (sensors, models)))(carry)
    return q, qd, np.asarray(xs), np.asarray(Ps)


# ---------------------------------------------------------------------------
# testPositionTracksTrajectory
# ---------------------------------------------------------------------------

def test_position_tracks_trajectory(scene):
    """300 ticks of consistent sinusoid: `q` tracks to 5e-3 rad at the final tick."""
    n = scene.fixture.n
    q, _qd, xs, _Ps = roll(scene, 300)
    assert_all_close(xs[-1, :n], q[-1], 5.0e-3, "position tracking at tick 299")


# ---------------------------------------------------------------------------
# testVelocityConverges
# ---------------------------------------------------------------------------

def test_velocity_converges(scene):
    """3000 ticks: `q_dot` tracks to 3e-2 rad/s **and** is actually excited.

    The second clause is the one with teeth.  `q_dot` tracking alone is passed
    trivially by an estimate that stays near zero for most of the cycle, since
    the true velocity is a cosine that spends time near zero anyway; the
    `max |v_est| > 0.5 * AMP * OMEGA` floor (measured after a 500-tick warmup) is
    what says the velocity is *observed* rather than merely small.

    Mutation-checked: zeroing the `q_dot` columns of the stacked gyro Jacobian
    leaves the position tracking test green and fails this one on both clauses.
    """
    n = scene.fixture.n
    warmup, total = 500, 3000
    q, qd, xs, _Ps = roll(scene, total)

    assert_all_close(xs[-1, n:2 * n], qd[-1], 3.0e-2, "velocity tracking at tick 2999")

    max_abs_v = np.max(np.abs(xs[warmup:, n:2 * n]), axis=0)
    assert np.all(max_abs_v > 0.5 * PEAK_VELOCITY), (
        f"velocity is not observed: per-joint max |v_est| = {max_abs_v} "
        f"vs floor {0.5 * PEAK_VELOCITY:.4f}"
    )


# ---------------------------------------------------------------------------
# testBiasStaysSmallWithZeroTrueBias
# ---------------------------------------------------------------------------

def test_bias_stays_small_with_zero_true_bias(scene):
    """1500 ticks of perfectly consistent motion leave every IMU bias < 5e-3.

    With no anchor the common-mode bias direction is unobservable (see the
    module docstring), so this is a *drift* statement, not a convergence one:
    the differenced pair rows have no reason to move the bias at all, and any
    systematic motion of it means the gyro model and the Jacobian disagree —
    the residual is being explained by a fictitious bias instead of by `q_dot`.
    """
    n = scene.fixture.n
    _q, _qd, xs, _Ps = roll(scene, 1500)
    bias = xs[-1, 2 * n:].reshape(-1, 3)
    norms = np.linalg.norm(bias, axis=1)
    assert np.all(norms < 5.0e-3), f"bias drifted with zero true bias: |b| = {norms}"


# ---------------------------------------------------------------------------
# testCovarianceStaysPsdAndBounded
# ---------------------------------------------------------------------------

def test_covariance_stays_psd_and_bounded(scene):
    """Symmetric + PSD every 50th tick to 2000, and `trace_final <= 10*trace_warmup + 1`.

    The warmup reference is taken at tick 200 rather than at init because the
    initial `P` is diagonal by construction and therefore trivially bounded; the
    interesting question is whether the *converged* covariance is stationary.
    """
    warmup, total = 200, 2000
    _q, _qd, _xs, Ps = roll(scene, total)

    for t in range(warmup, total, 50):
        assert_symmetric(Ps[t], 1.0e-6, f"P at tick {t}")
        assert_positive_semidefinite(Ps[t], f"P at tick {t}")

    trace_warmup = float(np.trace(Ps[warmup]))
    trace_final = float(np.trace(Ps[-1]))
    assert np.isfinite(trace_final)
    assert trace_final <= 10.0 * trace_warmup + 1.0, (
        f"trace(P) grew {trace_warmup:.3e} (tick {warmup}) -> {trace_final:.3e}"
    )


# ---------------------------------------------------------------------------
# testTransientNonFiniteInputRecovers — the NaN-hardening regression
# ---------------------------------------------------------------------------

def test_transient_non_finite_input_recovers(scene):
    """100 clean ticks, 5 poisoned ticks, 1000 clean ticks — skipped, never latched.

    Ticks 100..104 carry `NaN` on IMU0's whole gyro vector and on joint 0's
    encoder.  Two things must hold, and they are different claims:

    * **during** the bad window `x` and `P` stay finite — the update was
      *skipped*, not multiplied by a zero gate.  `0.0 * NaN` is `NaN`, so this
      only passes if `update.py` sanitises `H, z, R` **before** the arithmetic;
      a gate applied afterwards poisons `P` permanently on the first bad tick.
    * **after** the window, tracking returns to full accuracy with no
      intervention — no latch, no reset, no accumulated damage.

    Note the two channels are gated independently (`filter.py` §2), so the NaN
    encoder costs only the encoder update and the NaN gyro only the stacked one.
    A single concatenated measurement would drop both channels for five ticks
    and still pass this test — which is why `test_filter.py` asserts the
    independence directly.

    Mutation-checked: removing the `jnp.where(finite, ...)` sanitisation in
    `update.py` makes the bad window produce an all-NaN `x` and `P` on the very
    first poisoned tick.
    """
    n = scene.fixture.n
    clean, bad, recover = 100, 5, 1000
    total = clean + bad + recover

    def poison(encoders, gyros):
        gyros[clean:clean + bad, 0, :] = np.nan          # IMU0's gyro
        encoders[clean:clean + bad, 0] = np.nan          # joint 0's encoder
        return encoders, gyros

    q, qd, xs, Ps = roll(scene, total, poison=poison)

    for t in range(clean, clean + bad):
        assert_all_finite(xs[t], f"x during the bad window, tick {t}")
        assert_all_finite(Ps[t], f"P during the bad window, tick {t}")
    assert_all_finite(xs, "x over the whole run")
    assert_all_finite(Ps, "P over the whole run")

    assert_all_close(xs[-1, :n], q[-1], 5.0e-3, f"position after recovery, tick {total - 1}")
    assert_all_close(xs[-1, n:2 * n], qd[-1], 3.0e-2, f"velocity after recovery, tick {total - 1}")


# ---------------------------------------------------------------------------
# testPoisonedBiasCovarianceDoesNotLatchNaN  (adapted — see the docstring)
# ---------------------------------------------------------------------------

def test_poisoned_imu_covariance_does_not_latch_nan():
    """A non-finite per-IMU noise covariance is caught at BUILD, not at runtime.

    Java's `singlePairPoisonBias(..., imuIndex=0)` makes IMU 0's bias
    process-noise covariance non-finite *before* the filter is constructed, then
    asserts `getProcessNoise()` and `getTransitionMatrix()` are finite up front
    and that 200 ticks stay finite — i.e. that a construction-time guard drops
    the poisoned covariance instead of letting it reach `Q`.

    **Adaptation.**  The port has no per-IMU bias process noise: CONTRACT_CARD §2
    makes `imu_bias_process_var` a single scalar read from config, so the Java
    scenario has no literal spelling.  What it *tests* — a construction-time
    guard on a poisoned per-IMU noise covariance — does exist, on the per-IMU
    gyro `Sigma` that `build.py` floors, so the observable is ported there.
    Recorded as a deviation rather than silently reshaped, per CONTRACT_CARD §9.

    Mutation-checked: dropping the `np.all(np.isfinite(S))` clause from
    `build.py`'s floor makes `R_g = L Sigma L^T` non-finite, and this test fails
    at the `gyro_sigma` assertion.
    """
    f = fx.build_fixture(SINGLE_PAIR)
    poisoned = {"imu0": np.full((3, 3), np.nan), "imu1": GYRO_SIGMA * np.eye(3)}
    build = build_joint_kf(
        kinematic_tree(f),
        imu_sites=list(f.imu_names),
        pairs=[tuple(p) for p in f.pairs],
        foot_sites=[f.foot_site],
        use_mass_matrix=False,
        gyro_sigma=lambda name: poisoned[name],
    )
    params = default_params()

    # -- up front: the build guard replaced the poison, so Q and F are finite --
    assert_all_finite(build.gyro_sigma, "per-IMU gyro Sigma after the build guard")
    assert_all_finite(process.build_process_noise(build, params, None,
                                                 rotor=process.ROTOR_IN_MASS_MATRIX),
                      "process noise Q at construction")
    assert_all_finite(predict_mod.build_transition(build, params),
                      "transition matrix F at construction")

    # -- and 200 ticks of consistent motion stay finite ----------------------
    scene = Scene(fixture=f, build=build, params=params,
                  base_site=list(f.model.site_names).index(f.imu_names[build.base_imu]),
                  foot_sites=np.array([list(f.model.site_names).index(f.foot_site)]))
    _q, _qd, xs, Ps = roll(scene, 200)
    assert_all_finite(xs, "x after 200 ticks with a poisoned IMU covariance")
    assert_all_finite(Ps, "P after 200 ticks with a poisoned IMU covariance")


# ---------------------------------------------------------------------------
# testCovarianceEmbedsKinematicCoupling
# ---------------------------------------------------------------------------

def test_covariance_embeds_kinematic_coupling(scene):
    """`P`'s off-diagonal structure is physics, not noise — two distinct claims.

    **After one predict** (from rest, so `P` is the seeded diagonal pushed
    through `F` and `Q`):

    * within-joint `q <-> q_dot` coupling exists, `|P[i, n+i]| > 1e-4`.  This is
      the Van Loan `dt^2/2 * Qa` block: the double integrator correlates a
      joint's position with its own velocity, and a filter that discretised the
      process noise as a plain `diag(Q) * dt` would have exactly zero here.
    * cross-joint velocity coupling is *absent*, `|P[n+i, n+j]| < 1e-10` for
      `i < j`.  On the scalar-CWNA path `Qa = sigma_accel^2 I` is diagonal, so
      this is exact — and it is what makes the second half of the test mean
      something: any cross-joint correlation seen later was **created by the
      measurement**, not carried in by the process model.

    **After 200 ticks** of consistent motion the pair rows have been applied 200
    times, each one a rank-3 constraint through the shared Jacobian `J(q)`.
    Because one `J` row touches every joint on the chain, correcting it
    correlates all of them, and the cross-joint velocity *correlation* rises to
    ~0.99 against a required floor of 0.02.  That is the kinematic tree showing
    up in the covariance, which is the entire reason for a coupled tree filter
    over `n` independent per-joint filters (`jointKF/CLAUDE.md` §2).
    """
    build, params, n = scene.build, scene.params, scene.fixture.n

    # -- one predict from rest ---------------------------------------------
    q0, _ = trajectory([0], n)
    carry = init_carry(build, params, jnp.asarray(q0[0], dtype=jnp.float64))
    F = predict_mod.build_transition(build, params)
    Q = process.build_process_noise(build, params, None, rotor=process.ROTOR_IN_MASS_MATRIX)
    P0 = np.asarray(predict_mod.predict(carry.state, F, Q).P)

    within = np.abs(np.diag(P0[:n, n:2 * n]))
    assert np.all(within > 1.0e-4), f"no within-joint q<->q_dot coupling after predict: {within}"

    vel = P0[n:2 * n, n:2 * n]
    cross = np.abs(vel - np.diag(np.diag(vel)))
    assert cross.max() < 1.0e-10, (
        f"predict created cross-joint velocity coupling ({cross.max():.3e}); the scalar "
        "CWNA Qa is diagonal, so this must come from the measurement alone"
    )

    # -- 200 ticks of consistent motion -------------------------------------
    _q, _qd, _xs, Ps = roll(scene, 200)
    P1 = Ps[-1]
    assert_symmetric(P1, 1.0e-6, "P after 200 ticks")
    assert_positive_semidefinite(P1, "P after 200 ticks")

    vel = P1[n:2 * n, n:2 * n]
    sd = np.sqrt(np.diag(vel))
    corr = np.abs(vel / np.outer(sd, sd))
    np.fill_diagonal(corr, 0.0)
    assert corr.max() > 0.02, (
        f"the shared Jacobian did not correlate joint velocities: max |corr| = {corr.max():.3e}"
    )

    within = np.abs(np.diag(P1[:n, n:2 * n]))
    assert np.all(within > 1.0e-6), f"within-joint coupling vanished: {within}"


# ---------------------------------------------------------------------------
# JointLevelKFPreFilterAllocationTest.testHotPathStaysFinite
# ---------------------------------------------------------------------------

def test_hot_path_stays_finite():
    """1000 ticks on the allocation fixture: every published output is finite.

    The other two tests in `JointLevelKFPreFilterAllocationTest` count JVM
    thread-allocated bytes through `com.sun.management.ThreadMXBean` and are not
    portable (CONTRACT_CARD §7); the port's analogue of "no per-tick allocation"
    is "no recompilation", asserted in
    `test_filter.py::test_step_does_not_recompile_across_contact_patterns`.

    This one is portable and is kept as written: the Java fixture is a 10-joint
    chain with IMUs on `joints[1]` and `joints[9]` and `feet = [childLink]` —
    `_oracles.SHAPES[0]` exactly — built **without** the elevator, i.e. the
    scalar process-noise path, with every IMU reporting a fixed
    `(0.01, -0.02, 0.03)`.  It is a smoke test and nothing more: it says the hot
    path does not produce non-finite outputs, not that the outputs are right.
    """
    scene = _scene(HOT_PATH_SHAPE)
    f, build, params = scene.fixture, scene.build, scene.params
    n = f.n

    model = _model_inputs(scene, jnp.zeros(n, dtype=jnp.float64))
    sensors = SensorInputs(
        encoders=jnp.zeros(n, dtype=jnp.float64),
        gyros=jnp.tile(jnp.asarray([0.01, -0.02, 0.03], dtype=jnp.float64), (f.m, 1)),
        qd_unfiltered=jnp.zeros(build.anchor_unfiltered_mask.shape[1], dtype=jnp.float64),
        contact=jnp.ones(build.n_anchors, dtype=jnp.float64),
    )

    def body(carry, _):
        carry, _diag = step(carry, sensors, model, build, params)
        return carry, None

    carry = init_carry(build, params, jnp.zeros(n, dtype=jnp.float64))
    final, _ = jax.jit(lambda c: jax.lax.scan(body, c, None, length=1000))(carry)

    x = np.asarray(final.state.x)
    assert_all_finite(x[:n], "estimated joint positions after 1000 ticks")
    assert_all_finite(x[n:2 * n], "estimated joint velocities after 1000 ticks")
    assert_all_finite(np.asarray(final.state.b_omega_imus(n))[build.base_imu],
                      "base-IMU gyro bias after 1000 ticks")
