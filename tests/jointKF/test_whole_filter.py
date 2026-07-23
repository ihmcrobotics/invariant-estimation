"""Port of `JointLevelKFFilterTest` (6 tests) — the whole filter, end to end.

This is gate G8's first half: the Java class drives the full orchestration
(`computeJointState()` **and** `computeImuBiases(feet)`) over trajectories and
asserts the properties that only exist once every module is composed —
finiteness, covariance PSD-ness and boundedness, determinism, encoder tracking,
and the one genuinely numerical claim in the class, **stance-phase bias
convergence over 20,000 ticks**.

What the port maps onto what
----------------------------
Java's `tick(f)` is `computeJointState()` then `computeImuBiases(f.feet)`.  The
port fuses both into a single `filter.step`, with `feet` expressed as the
`SensorInputs.contact` mask — so "feet passed to `computeImuBiases`" becomes
`contact = 1` and, because the mask is delayed by exactly one tick
(`filter.py` §1), the anchor is live from tick 1 onwards.  Every scenario in
this class holds its sensors constant, so a whole run is a `lax.scan` over a
*closed-over* constant `ModelInputs`; nothing here needs the model re-evaluated
per tick, which is what makes a 20,000-tick run cost well under a second.

Fixture: `singlePair(seed, 8, 1, 7)`
------------------------------------
Four of the six tests use Java's `singlePair(seed, 8, 1, 7)`: an 8-link chain
with IMUs on `link1` and `link7` and the foot on `link7`.  `_oracles.SHAPES`
does not carry that topology (it is the `shapes(...)` parametrisation), so it is
built here as a one-off shape dict through the *same* `_fixture.build_fixture`,
which keys its RNG stream on the shape **name** — so this chain's geometry is
independent of every `SHAPES` entry and stable across runs.

`n = child_link - parent_link = 7 - 1 = 6`.  `TEST_SUITE_MAP.md`'s prose says
`n = child - parent - 1` (and annotates this very fixture "n=5"), but its own
shape table says `singlePair(10, 1, 9) -> n = 8`.  PORT_NOTES already resolved
that contradiction in favour of the shape table, and `_fixture.py` implements
it; this file follows.

Per-test seeds collapse to one geometry
---------------------------------------
Java gives each test its own fixture seed (5000, 5100, ... 5500).  Nothing in
this class is statistical — there is no RNG anywhere in the scenarios, only in
the chain's link offsets — and CLAUDE.md §5 explicitly permits substituting any
deterministic geometry.  Rebuilding the MJX model per test would cost a model
evaluation per test (~8 s at this chain depth) to buy nothing, so all four
`singlePair` tests share one module-scoped scene.  The trial-count-bearing
constants (tick counts, tolerances, the gyro offset) are preserved verbatim.

The scalar-CWNA path, not the mass-matrix path
----------------------------------------------
`isUsingMassMatrixProcessNoise()` is `false` for a plain `singlePair(...)`
fixture and `true` only for `singlePairMassMatrix(...)`
(`JointLevelKFMassMatrixNoiseTest.testMassMatrixPathEnabledOnlyWithModel`), so
this class runs the **scalar** path: `Qa = sigma_accel^2 I`, `M = None`.  That
is not a convenience — `test_covariance_embeds_kinematic_coupling` in
`test_trajectory.py` asserts that one predict produces *exactly zero*
cross-joint velocity covariance, which is true of the diagonal scalar `Qa` and
false of the dense Schur `Qa`.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.jointKF import anchors as anchors_mod
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

#: Java `singlePair(seed, numChainJoints=8, imuAfterJoint=1, footAfterJoint=7)`.
#: The foot site always sits on the last link (`_fixture.chain_geometry`), which
#: is `link7` here — exactly `footAfterJoint=7`.
SINGLE_PAIR = {"name": "filter_single_pair_8_1_7", "chain": 8, "imus": (1, 7),
               "pairs": ((0, 1),), "n": 6, "m": 2}

#: The Java fixture's `TestIMU` reports `diag(1e-4)` for every noise covariance
#: (`TEST_SUITE_MAP.md`, "Filter constants the tests mirror"), and `stub_build`
#: in `_oracles.py` uses the same value.  Passed explicitly because `build.py`'s
#: default is the *floor* (1e-6), which is two orders tighter and would make the
#: gyro rows pin `q_dot` far harder than the Java fixture does.
GYRO_SIGMA = 1.0e-4

#: `testStancePhaseBiasConvergence`: the constant offset every IMU reports, in
#: its own measurement frame — which is the frame `b_omega` is stored in
#: (CONTRACT_CARD §1), so the expected converged bias is this vector itself.
BIAS_OFFSET = np.array([0.01, -0.02, 0.03])


# ---------------------------------------------------------------------------
# Scene wiring — the same pattern as `test_filter.py::scene`
# ---------------------------------------------------------------------------

class Scene(NamedTuple):
    """A built filter plus its model quantities at one configuration.

    `model` is evaluated once and closed over by every scan in this file.  That
    is legitimate rather than a shortcut: every scenario in `JointLevelKFFilterTest`
    holds its encoders fixed, so the true configuration never moves and the
    Jacobians are genuinely constant.  It is also the only way a 20,000-tick run
    is affordable — MJX tracing cost grows sharply with chain depth (PORT_NOTES
    G1), and `filter.step` takes the model as an *argument* precisely so the
    caller can hoist it.
    """

    fixture: fx.ChainFixture
    build: object
    params: object
    model: ModelInputs


def _scene(shape: dict) -> Scene:
    f = fx.build_fixture(shape)
    build = build_joint_kf(
        kinematic_tree(f),
        imu_sites=list(f.imu_names),
        pairs=[tuple(p) for p in f.pairs],
        foot_sites=[f.foot_site],
        use_mass_matrix=False,                       # scalar CWNA — see module docstring
        gyro_sigma=lambda name: GYRO_SIGMA * np.eye(3),
    )
    params = default_params()

    # `anchor_jacobians` indexes into `ev.J_ang` / `ev.site_rot`, which are in
    # `model.site_names` order — IMU sites first, then the foot.
    names = list(f.model.site_names)
    q0 = jnp.zeros(f.n, dtype=jnp.float64)
    ev = f.model.evaluate(q0)
    R = ev.site_rot[jnp.asarray(f.model.pair_sites)]
    R_rel = jnp.einsum("eji,ejk->eik", R[:, 1], R[:, 0])     # ^c R_p = ^W R_c^T ^W R_p
    jac = anchors_mod.anchor_jacobians(
        build, ev.J_ang, ev.site_rot,
        base_site=names.index(f.imu_names[build.base_imu]),
        foot_sites=np.array([names.index(f.foot_site)]),
    )
    # M=None selects the scalar-CWNA process noise; `use_mass_matrix=False` above
    # is what makes that legal rather than a silently unwired mass matrix
    # (`process.acceleration_covariance` raises otherwise).
    return Scene(f, build, params, ModelInputs(J_rel=ev.J_rel, R_rel=R_rel,
                                               anchor_jac=jac, M=None))


@pytest.fixture(scope="module")
def single_pair() -> Scene:
    """Java `singlePair(..., 8, 1, 7)` — shared by four tests (see docstring)."""
    return _scene(SINGLE_PAIR)


@pytest.fixture(scope="module")
def shape_scenes() -> tuple[Scene, ...]:
    """Java `shapes(seed)` — the four-fixture parametrisation."""
    return tuple(_scene(s) for s in SHAPES)


def _sensors(scene: Scene, *, encoders=None, gyros=None, contact=1.0) -> SensorInputs:
    """One constant tick of proprioception.

    `contact=1.0` is the port's spelling of Java passing `f.feet` into
    `computeImuBiases`: the stance anchor is what makes the base gyro bias
    observable at all, so it is on for every scenario in this class.
    """
    f, build = scene.fixture, scene.build
    n_u = build.anchor_unfiltered_mask.shape[1]
    return SensorInputs(
        encoders=jnp.zeros(f.n, dtype=jnp.float64) if encoders is None
        else jnp.asarray(encoders, dtype=jnp.float64),
        gyros=jnp.zeros((f.m, 3), dtype=jnp.float64) if gyros is None
        else jnp.asarray(gyros, dtype=jnp.float64),
        qd_unfiltered=jnp.zeros(n_u, dtype=jnp.float64),
        contact=jnp.full(build.n_anchors, float(contact), dtype=jnp.float64),
    )


def _run(scene: Scene, sensors: SensorInputs, ticks: int, *, q0=None, record: bool = False):
    """`ticks` scans of `filter.step` against a constant model and constant sensors.

    Returns `(final_carry, history)`; `history` is `(x, P)` stacked over time
    when `record=True` and `None` otherwise.  Recording 20,000 covariances would
    be 77 MB for no gain, so it is opt-in.

    The scan (rather than a Python loop) is the point: it compiles ONE tick and
    reuses it, which is both what makes the long runs affordable and a standing
    demonstration of I7 — no gate, mask or anchor decision in the tick can be a
    Python branch or the trace would fail here.
    """
    build, params, model = scene.build, scene.params, scene.model

    def body(carry, _):
        carry, _diag = step(carry, sensors, model, build, params)
        return carry, ((carry.state.x, carry.state.P) if record else jnp.float64(0.0))

    carry = init_carry(build, params, q0)
    return jax.jit(lambda c: jax.lax.scan(body, c, None, length=ticks))(carry)


# ---------------------------------------------------------------------------
# testTrajectoryFiniteAndShapes
# ---------------------------------------------------------------------------

def test_trajectory_finite_and_shapes(shape_scenes):
    """50 ticks on every shape: `x` and `P` keep their dimensions and stay finite.

    A smoke test, and honest about it — what it actually rules out is a shape
    mismatch between modules (which would raise) and an outright NaN/inf leak
    from the mass-matrix-free path.  It does not constrain any value.
    """
    for scene in shape_scenes:
        final, _ = _run(scene, _sensors(scene), ticks=50)
        dim = scene.build.dim
        x, P = np.asarray(final.state.x), np.asarray(final.state.P)
        assert x.shape == (dim,), f"{scene.fixture.name}: x has {x.shape}, expected ({dim},)"
        assert P.shape == (dim, dim), f"{scene.fixture.name}: P has {P.shape}"
        assert_all_finite(x, f"{scene.fixture.name}: x after 50 ticks")
        assert_all_finite(P, f"{scene.fixture.name}: P after 50 ticks")


# ---------------------------------------------------------------------------
# testCovariancePSDAlongTrajectory
# ---------------------------------------------------------------------------

def test_covariance_psd_along_trajectory(shape_scenes):
    """`P` is symmetric (1e-6) and PSD at **every** one of 20 ticks, on every shape.

    Checked per tick rather than at the end because the Joseph form's failure
    mode is transient: a gain applied through an ill-conditioned `S` can push
    `P` indefinite for a few ticks and be re-symmetrised away later, so only the
    per-tick check sees it.
    """
    for scene in shape_scenes:
        _, (xs, Ps) = _run(scene, _sensors(scene), ticks=20, record=True)
        for t, P in enumerate(np.asarray(Ps)):
            assert_symmetric(P, 1.0e-6, f"{scene.fixture.name}: P at tick {t}")
            assert_positive_semidefinite(P, f"{scene.fixture.name}: P at tick {t}")


# ---------------------------------------------------------------------------
# testDeterministic
# ---------------------------------------------------------------------------

def test_deterministic(single_pair):
    """Two independent builds from the same spec run 30 ticks to the same numbers.

    Java builds two fixtures from one seed and asserts `assertAllClose(..., 0.0)`
    — literal bit-equality.  CONTRACT_CARD §7 says to render `tol = 0.0` as
    "assert determinism against the port's own repeat run", which is exactly
    what Java is doing, so the tolerance stays 0.0 here.

    Two claims, asserted separately so a failure says which one broke:

    1. the *geometry* is reproducible — `_fixture.chain_geometry` is seeded on
       the shape name, so a second `build_fixture` must produce bit-identical
       link offsets, inertias and site poses;
    2. the *filter* has no hidden state or RNG — same geometry, same inputs,
       same trajectory to the last bit.

    Splitting them matters because (1) failing would make (2) vacuous.
    """
    gyros = np.array([[0.02, -0.01, 0.03], [0.01, 0.02, -0.02]])

    rebuilt = fx.build_fixture(SINGLE_PAIR)
    for field, a, b in zip(rebuilt.geometry._fields, rebuilt.geometry,
                           single_pair.fixture.geometry):
        assert_all_close(a, b, 0.0, f"chain geometry field {field!r} is not reproducible")

    a_final, _ = _run(single_pair, _sensors(single_pair, gyros=gyros), ticks=30)
    b_final, _ = _run(single_pair, _sensors(single_pair, gyros=gyros), ticks=30)
    assert_all_close(a_final.state.x, b_final.state.x, 0.0, "x is not deterministic")
    assert_all_close(a_final.state.P, b_final.state.P, 0.0, "P is not deterministic")


# ---------------------------------------------------------------------------
# testEncoderTracking
# ---------------------------------------------------------------------------

def test_encoder_tracking(single_pair):
    """Position tracks a constant encoder set to 5e-3 rad over 100 ticks.

    Java sets the encoders, *then* calls `initialize()`, which seeds `q` from
    the encoders (`JointLevelKFStateTest.testQ0Seed`) — so the scenario as
    written starts at the answer and asks that 100 ticks of predict + update do
    not drift off it.  That is a real constraint (a runaway process noise, or an
    encoder residual with the wrong sign feeding a non-zero gain, both move it)
    but it is a weak one, and it is worth saying so rather than claiming more
    (JOINTKF_PORT_PLAN §4 lesson 3).

    So the second half adds the direction the test's *name* implies and its
    scenario does not test: from a deliberately wrong seed (`q = 0`), the
    estimate must actually walk to the encoders.  The 20x floor is read off the
    mechanism, not tuned: one pair gives 3 gyro rows against 6 joint velocities,
    so three velocity directions are pinned near zero by the gyro and their
    positions can only be corrected by the encoder gain itself, which is why the
    error falls fast at first (~24x in 100 ticks) and then crawls.  A bound of
    20x sits just under the observed behaviour and far above anything a broken
    encoder channel could produce.

    Deliberately NOT asserted: monotone decrease.  The error is a max over
    joints, and joints in the gyro-pinned subspace converge on a different
    timescale from the free ones, so the max legitimately ticks back up by ~1e-3
    late in the window as the dominant joint changes.  Two decreasing
    checkpoints say the same thing without asserting something false.
    """
    n = single_pair.fixture.n
    target = 0.2 * (np.arange(n) + 1.0) - 0.5
    sensors = _sensors(single_pair, encoders=target)

    final, _ = _run(single_pair, sensors, ticks=100, q0=jnp.asarray(target))
    assert_all_close(np.asarray(final.state.x)[:n], target, 5.0e-3,
                     "position drifted off a constant encoder")

    # -- the non-trivial direction: converge to the encoders from a wrong seed --
    _, (xs, _Ps) = _run(single_pair, sensors, ticks=100,
                        q0=jnp.zeros(n, dtype=jnp.float64), record=True)
    err = np.max(np.abs(np.asarray(xs)[:, :n] - target), axis=1)
    start = float(np.max(np.abs(target)))
    assert err[49] < start / 5.0, f"too slow at tick 50: {start:.3e} -> {err[49]:.3e}"
    assert err[-1] < start / 20.0, (
        f"position did not converge toward the encoders: {start:.3e} -> {err[-1]:.3e}"
    )


# ---------------------------------------------------------------------------
# testStancePhaseBiasConvergence — THE numerical property of this class
# ---------------------------------------------------------------------------

def test_stance_phase_bias_convergence(single_pair):
    """20,000 ticks with a constant gyro offset: the base bias converges to it.

    Every IMU reports a constant `(0.01, -0.02, 0.03)` in its own measurement
    frame, the encoders read zero, and the foot is trusted.  The consistent
    interpretation of that sensor set is `q_dot = 0` with `b_k = offset` on
    every IMU, and the filter must find it.

    This is the one test in the class that can only pass if the whole chain is
    right, and specifically **it cannot pass without the stance anchor**.  The
    pair rows only ever see `b_child - {}^{c}R_{p} b_parent`, so the common-mode
    bias direction lies in their nullspace: absent an anchor the base bias is
    unobservable and 20,000 ticks converge to nothing in particular.  The anchor
    row is what fixes the gauge — it reads the base IMU's own rate back through
    its `+I3` bias column against a foot whose absolute angular rate is asserted
    to be ~zero.  Mutation-checked: zeroing the anchor `H` block leaves the
    filter finite, PSD and bounded, and fails only here.

    The tolerance is 2e-3 per axis, per the Java class; the port converges to
    ~5e-7, i.e. with three orders of margin, so this is not a threshold that
    could be met by accident.
    """
    f = single_pair.fixture
    n = f.n
    sensors = _sensors(single_pair, gyros=np.tile(BIAS_OFFSET, (f.m, 1)))

    final, _ = _run(single_pair, sensors, ticks=20_000,
                    q0=jnp.zeros(n, dtype=jnp.float64))

    bias = np.asarray(final.state.b_omega_imus(n))       # Java getAngularVelocityBiasInIMUFrame
    base = bias[single_pair.build.base_imu]
    assert_all_close(base, BIAS_OFFSET, 2.0e-3,
                     "base-IMU residual bias did not converge to the injected offset")


# ---------------------------------------------------------------------------
# testCovarianceBounded
# ---------------------------------------------------------------------------

def test_covariance_bounded(single_pair):
    """2000 ticks: `trace(P)` stays finite and within `10 * initial + 1`.

    The mechanism this guards is the Joseph `K R K^T` term under a nearly
    singular `S`: a gain that is too large is squared into `P` every tick, so
    divergence here is geometric and shows up long before any single tick looks
    wrong.
    """
    n = single_pair.fixture.n
    carry = init_carry(single_pair.build, single_pair.params,
                       jnp.zeros(n, dtype=jnp.float64))
    initial_trace = float(jnp.trace(carry.state.P))

    final, _ = _run(single_pair, _sensors(single_pair), ticks=2000,
                    q0=jnp.zeros(n, dtype=jnp.float64))
    final_trace = float(jnp.trace(final.state.P))

    assert np.isfinite(final_trace)
    assert final_trace <= initial_trace * 10.0 + 1.0, (
        f"trace(P) grew {initial_trace:.3e} -> {final_trace:.3e}"
    )
