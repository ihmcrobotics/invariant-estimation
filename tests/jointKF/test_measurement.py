"""Port of `JointLevelKFMeasurementTest` (11 tests) + the port-specific oracles
that actually *constrain* invariant I6.

Java -> Python mapping
----------------------
Java iterates `shapes(base)`; here that is `_oracles.SHAPES`, driven through the
real MJX chain fixture (`_fixture.py`) so every rotation in `L` is a genuine
kinematic quantity rather than a stub.  Java's `singlePair(seed, 8, 1, 7)` becomes
a one-off `ChainFixture` with the same topology and a seed-tagged name, so each
Java seed still gets its own geometry (the Java geometry itself is not
reproducible across languages -- `TEST_SUITE_MAP.md`, "Hard dependency").

Why there are more than 11 tests here
-------------------------------------
`PORT_NOTES.md` measured it: on three of the four shapes -- every single-pair
shape, with the fixture's isotropic `Sigma = 1e-4 I3` -- a block-diagonal `R_g`
is not an approximation of `L Sigma L^T`, it is an **identity** (deviation
2.6e-20).  The ported suite therefore *exercises* I6 nine times and *constrains*
it twice.  The extra tests below close that: an anisotropic two-pair congruence
check with an explicit anti-block-diagonal assertion, and the marginalized
raw-gyro reference, which is also what decides the frame `b_omega` lives in.
"""
import jax
import numpy as np
import pytest

from invariant_estimation.jointKF import measure
from invariant_estimation.jointKF.state import default_params

from ._fixture import ChainFixture, build_fixture
from ._oracles import (
    SHAPES,
    assert_all_close,
    assert_positive_semidefinite,
    assert_symmetric,
    reference_marginalized,
    reference_update,
    seeded_prior_update,
    stub_build,
)

PARAMS = default_params()

#: Java's fixture default: `angularVelocityNoiseCovariance = 1e-4 * I3` per IMU.
ISOTROPIC_SIGMA = 1.0e-4 * np.eye(3)


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------

_SINGLE_PAIR_CACHE: dict[str, tuple[dict, ChainFixture]] = {}


def single_pair(seed: int, chain: int = 8, parent: int = 1, child: int = 7):
    """Java `JointLevelKFTestFixture.singlePair(seed, chain, parent, child)`.

    The Java seed selects a random chain geometry; here it selects *a* geometry
    (via the fixture's name-keyed stream), which is all the tests need -- they
    assert self-consistency between the fixture's kinematics and the filter's,
    never particular numbers.
    """
    key = f"sp{seed}_{chain}_{parent}_{child}"
    if key not in _SINGLE_PAIR_CACHE:
        shape = {"name": key, "chain": chain, "imus": (parent, child),
                 "pairs": ((0, 1),), "n": child - parent, "m": 2}
        _SINGLE_PAIR_CACHE[key] = (shape, build_fixture(shape))
    return _SINGLE_PAIR_CACHE[key]


def shapes():
    """`_oracles.SHAPES` paired with their built fixtures -- Java `shapes(base)`."""
    return [(s, build_fixture(s)) for s in SHAPES]


def kf_build(shape: dict, fx: ChainFixture, **overrides):
    """A `JointKFBuild` whose pair structure is the fixture's REAL graph.

    `stub_build` supplies dimensions and the parameter vectors; everything that
    encodes geometry (`pair_velocity_mask`, and through `build_stacked` the
    rotations and Jacobians) comes from the MJX model, so a disagreement between
    the graph resolution and the kinematics cannot hide.
    """
    fields = dict(pair_velocity_mask=np.asarray(fx.model.pair_joint_mask, dtype=float))
    fields.update(overrides)
    return stub_build(shape, **fields)


def stacked(fx: ChainFixture, build, *, q=None, gyros=None, seed: float = 0.0, **kwargs):
    """`buildStackedMeasurementForTest` at a definite configuration.

    `q` defaults to a seeded random configuration rather than zero: at `q = 0`
    the chain's link rotations are its (already random) offsets, but a spread of
    joint angles is what keeps `R_rel` far from anything special, so a test that
    accidentally assumes `R = I` fails.
    """
    rng = np.random.default_rng(int(seed) if seed else 12345)
    q = fx.random_q(rng) if q is None else np.asarray(q, dtype=float)
    gyros = rng.normal(size=(fx.m, 3)) if gyros is None else np.asarray(gyros, dtype=float)
    J_rel, R_rel = measure.pair_frames(fx.model, q)
    return measure.build_stacked(build, PARAMS, gyros=gyros, J_rel=J_rel, R_rel=R_rel, **kwargs)


def pair_jacobian(sm, build, pair: int) -> np.ndarray:
    """Java `pairJacobian(f, pairIndex)`: the 3 x dim slice for one pair."""
    r = build.stacked_row_for_pair(pair)
    return np.asarray(sm.H)[r:r + 3, :]


def pair_residual(sm, build, pair: int) -> np.ndarray:
    """Java `pairResidual(f, pairIndex)`."""
    r = build.stacked_row_for_pair(pair)
    return np.asarray(sm.z)[r:r + 3]


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------

def test_encoder_jacobian_structure():
    """`testEncoderJacobianStructure` -- `shapes(4000L)`, tol 1e-12.

    Encoder rows observe `q` and are blind to `q_dot` and bias.
    """
    for shape, fx in shapes():
        build = kf_build(shape, fx)
        n = build.n_joints
        H = np.asarray(measure.encoder_jacobian(build))
        assert H.shape == (n, build.dim)
        assert_all_close(H[:, :n], np.eye(n), 1.0e-12, f"{shape['name']} q block")
        assert_all_close(H[:, n:], np.zeros((n, build.dim - n)), 1.0e-12,
                         f"{shape['name']} q_dot/bias block")


def test_encoder_predicts_position():
    """`testEncoderPredictsPosition` -- `singlePair(4050L, 8, 1, 7)`, tol 1e-12."""
    shape, fx = single_pair(4050)
    build = kf_build(shape, fx)
    n = build.n_joints
    x = 0.1 * np.arange(1, build.dim + 1)
    hx = np.asarray(measure.encoder_jacobian(build)) @ x
    assert_all_close(hx, x[:n], 1.0e-12, "H_enc x == position segment")


def test_encoder_noise_is_the_per_joint_variance():
    """Port-specific: `getEncoderNoise` is `diag(build.encoder_var)`, off-diagonal
    exactly 0.

    Java's per-joint wiring is locked by `JointLevelKFEncoderNISConsistencyTest`
    (another agent's class); what belongs here is that this module reads the
    per-joint array and never collapses it to one scalar (invariant I9).
    """
    shape, fx = single_pair(4050)
    var = 1.0e-5 * np.arange(1, shape["n"] + 1)
    build = kf_build(shape, fx, encoder_var=var)
    R = np.asarray(measure.encoder_noise(build))
    assert_all_close(np.diag(R), var, 0.0, "encoder variances")
    assert_all_close(R - np.diag(np.diag(R)), np.zeros_like(R), 0.0, "off-diagonal")


# ---------------------------------------------------------------------------
# Pair rows: structure
# ---------------------------------------------------------------------------

def test_pair_jacobian_structure():
    """`testPairJacobianStructure` -- `shapes(4100L)`.

    Sparsity: no `q` dependence at all, velocity support exactly the pair's chain,
    bias support exactly the parent and child blocks. The off-support entries are
    checked at tol 0.0 (Java's `assertEquals(0.0, v)`), which is meaningful here
    because they are never written rather than merely small.
    """
    for shape, fx in shapes():
        build = kf_build(shape, fx)
        sm = stacked(fx, build, seed=4100)
        n, dim = build.n_joints, build.dim
        H = pair_jacobian(sm, build, 0)
        assert H.shape == (3, dim)
        assert_all_close(H[:, :n], np.zeros((3, n)), 1.0e-12, f"{shape['name']} q columns")

        vel = set(build.pair_velocity_cols(0))
        off_vel = [c for c in range(n, 2 * n) if c not in vel]
        assert_all_close(H[:, off_vel], np.zeros((3, len(off_vel))), 0.0,
                         f"{shape['name']} off-chain velocity columns")
        assert vel, "a pair with no velocity columns would measure nothing but bias"

        bias = set(range(build.pair_parent_bias_col(0), build.pair_parent_bias_col(0) + 3))
        bias |= set(range(build.pair_child_bias_col(0), build.pair_child_bias_col(0) + 3))
        off_bias = [c for c in range(2 * n, dim) if c not in bias]
        assert_all_close(H[:, off_bias], np.zeros((3, len(off_bias))), 0.0,
                         f"{shape['name']} unrelated bias columns")


def test_bias_columns_of_hg_are_exactly_l():
    """`testBiasColumnsOfHgAreExactlyL` -- `shapes(4150L)`, tol **0.0**.

    The central Rev.2 identity (invariant I6). Bit-identity is achievable because
    `L` is built once and scattered into `H_g`; if the two were computed by
    separate expressions this would be a rounding-level assertion instead of a
    structural one.
    """
    for shape, fx in shapes():
        build = kf_build(shape, fx)
        sm = stacked(fx, build, seed=4150)
        n, m = build.n_joints, build.n_imus
        H, L = np.asarray(sm.H), np.asarray(sm.L)
        assert H.shape[0] == L.shape[0], f"{shape['name']}: H and L must span the same rows"
        assert_all_close(H[:, 2 * n:2 * n + 3 * m], L, 0.0, f"{shape['name']} bias columns == L")


def test_child_bias_block_is_identity():
    """`testChildBiasBlockIsIdentity` -- `singlePair(4200L, 8, 1, 7)`, tol 1e-9.

    The child IMU's frame IS the Jacobian frame, so its bias needs no rotation.
    This is the observable consequence of `b_omega` living in each IMU's own
    frame; see the module docstring of `measure.py`.
    """
    shape, fx = single_pair(4200)
    build = kf_build(shape, fx)
    sm = stacked(fx, build, seed=4200)
    col = build.pair_child_bias_col(0)
    block = pair_jacobian(sm, build, 0)[:, col:col + 3]
    assert_all_close(block, np.eye(3), 1.0e-9, "child bias block")


def test_parent_bias_block_is_negative_rotation():
    """`testParentBiasBlockIsNegativeRotation` -- `singlePair(4300L, 8, 1, 7)`, tol 1e-9.

    `-R^T R == I` only says the block is a negated *orthonormal* matrix; the test
    below (`..._matches_the_relative_rotation`) pins WHICH rotation, which is the
    part a transposed frame convention would break.
    """
    shape, fx = single_pair(4300)
    build = kf_build(shape, fx)
    sm = stacked(fx, build, seed=4300)
    col = build.pair_parent_bias_col(0)
    rotation = -pair_jacobian(sm, build, 0)[:, col:col + 3]
    assert_all_close(rotation.T @ rotation, np.eye(3), 1.0e-9, "parent block orthonormality")


def test_parent_bias_block_matches_the_relative_rotation():
    """Port-specific: the parent block is `-{}^{child}R_{parent}`, not its transpose.

    Orthonormality (the Java assertion) is invariant under transposition, so the
    Java test cannot distinguish `R` from `R^T` -- and a transposed convention is
    exactly the plausible mistake. Independent route: the fixture's own site
    rotations, composed in NumPy.
    """
    shape, fx = single_pair(4300)
    build = kf_build(shape, fx)
    rng = np.random.default_rng(4301)
    q = fx.random_q(rng)
    sm = stacked(fx, build, q=q, seed=4301)

    motion = fx.apply_consistent_motion(q, np.zeros(fx.n))
    R_parent_w, R_child_w = motion.site_rot[0], motion.site_rot[1]
    expected = -(R_child_w.T @ R_parent_w)

    col = build.pair_parent_bias_col(0)
    assert_all_close(pair_jacobian(sm, build, 0)[:, col:col + 3], expected, 1.0e-9,
                     "parent bias block == -R(child<-parent)")


# ---------------------------------------------------------------------------
# Pair rows: the residual
# ---------------------------------------------------------------------------

def test_relative_gyro_parent_zero():
    """`testRelativeGyroParentZero` -- `singlePair(4400L, 8, 1, 7)`, tol 1e-9."""
    shape, fx = single_pair(4400)
    build = kf_build(shape, fx)
    child = np.array([0.1, -0.2, 0.3])
    sm = stacked(fx, build, gyros=np.stack([np.zeros(3), child]), seed=4400)
    assert_all_close(pair_residual(sm, build, 0), child, 1.0e-9, "z == omega_child")


def test_relative_gyro_child_zero_preserves_norm():
    """`testRelativeGyroChildZeroPreservesNorm` -- `singlePair(4500L, 8, 1, 7)`, tol 1e-9.

    `omega_child = 0 => z = -R omega_parent`, and a rotation preserves norm.
    """
    shape, fx = single_pair(4500)
    build = kf_build(shape, fx)
    parent = np.array([0.1, -0.2, 0.3])
    sm = stacked(fx, build, gyros=np.stack([parent, np.zeros(3)]), seed=4500)
    z = pair_residual(sm, build, 0)
    assert abs(np.linalg.norm(z) - np.linalg.norm(parent)) <= 1.0e-9


def test_relative_gyro_equals_j_qdot_under_consistent_motion():
    """Port-specific: on the `applyConsistentMotion` oracle, `z == J_rel qdot` exactly.

    Zero base twist makes the pair measurement exact, not first-order (see
    `_fixture.apply_consistent_motion`). This is the assertion every downstream
    tracking tolerance silently assumes, and it constrains the residual's *sign
    and frame together* -- unlike the two Java residual tests, which each hold one
    gyro at zero and so cannot see a frame error in the other.
    """
    for shape, fx in shapes():
        build = kf_build(shape, fx)
        rng = np.random.default_rng(4550 + shape["n"])
        q, qd = fx.random_q(rng), rng.normal(size=fx.n) * 0.4
        motion = fx.apply_consistent_motion(q, qd)
        sm = stacked(fx, build, q=q, gyros=motion.gyro)
        n = build.n_joints
        J = np.asarray(sm.H)[:3 * build.n_pairs, n:2 * n]
        assert_all_close(np.asarray(sm.z)[:3 * build.n_pairs], J @ qd, 1.0e-10,
                         f"{shape['name']}: relative gyro == J_rel qdot")


# ---------------------------------------------------------------------------
# The stacked noise -- invariant I6
# ---------------------------------------------------------------------------

def test_measurement_noise_symmetric_psd():
    """`testMeasurementNoiseSymmetricPSD` -- `shapes(4600L)`.

    Java asserts `R.numRows == 3 * numberOfPairs` because its stacked build emits
    no rows for untrusted anchors. This port keeps anchor rows always (invariant
    I7), so the Java shape assertion is checked in the `n_anchors = 0`
    configuration -- Java's "no feet configured" case -- and the masked case gets
    its own test (`test_inactive_anchor_rows_are_masked_not_removed`).
    """
    for shape, fx in shapes():
        build = kf_build(shape, fx)
        sm = stacked(fx, build, seed=4600)
        R = np.asarray(sm.R)
        assert R.shape[0] == 3 * build.n_pairs
        assert_symmetric(R, 1.0e-12, f"{shape['name']} R")
        assert_positive_semidefinite(R, f"{shape['name']} R")


def test_measurement_noise_uses_gyro_measurement_covariance():
    """`testMeasurementNoiseUsesGyroMeasurementCovariance` -- `singlePair(4700L)`, tol 1e-12.

    ANISOTROPIC covariances on purpose: with the fixture's isotropic default,
    `R Sigma R^T = sigma^2 I` exactly and the rotation cancels, so an isotropic
    version of this test would pass against a wrong `R_g` (`PORT_NOTES.md`). The
    trace guard is Java's: an `R` built from the bias process covariance
    (`1e-9`-scale here) would have trace ~6e-9.
    """
    shape, fx = single_pair(4700)
    sigma_parent = np.diag([4.0e-4, 1.0e-6, 2.5e-5])
    sigma_child = np.diag([9.0e-4, 1.6e-5, 4.9e-6])
    build = kf_build(shape, fx, gyro_sigma=np.stack([sigma_parent, sigma_child]))

    rng = np.random.default_rng(4700)
    q = fx.random_q(rng)
    sm = stacked(fx, build, q=q, seed=4700)

    motion = fx.apply_consistent_motion(q, np.zeros(fx.n))
    R_rel = motion.site_rot[1].T @ motion.site_rot[0]        # child <- parent
    expected = sigma_child + R_rel @ sigma_parent @ R_rel.T

    assert_all_close(np.asarray(sm.R), expected, 1.0e-12, "R == Sigma_c + R Sigma_p R^T")
    assert float(np.trace(np.asarray(sm.R))) > 1.0e-5, "R looks like it was built from bias noise"


def test_measurement_noise_independent_of_bias_process_covariance():
    """`testMeasurementNoiseIndependentOfBiasProcessCovariance` -- `singlePair(4800L)`, tol 0.0.

    Java scales every IMU's bias process covariance by 1000x and rebuilds. In this
    port the bias random walk is a scalar in `JointKFParams` (it belongs to
    `process.py`), so the port of the scenario is to scale *that* by 1000x -- and
    the fact that `build_stacked` never reads it is the property under test.
    """
    shape, fx = single_pair(4800)
    build = kf_build(shape, fx)
    before = np.asarray(stacked(fx, build, seed=4800).R)

    poisoned = default_params(imu_bias_process_var=1000.0 * PARAMS.imu_bias_process_var)
    rng = np.random.default_rng(4800)
    q = fx.random_q(rng)
    J_rel, R_rel = measure.pair_frames(fx.model, q)
    after = np.asarray(measure.build_stacked(
        build, poisoned, gyros=rng.normal(size=(fx.m, 3)), J_rel=J_rel, R_rel=R_rel).R)

    assert_all_close(after, before, 0.0, "R must not depend on the bias process covariance")


def _dense_sigma(gyro_sigma: np.ndarray) -> np.ndarray:
    """`blkdiag(Sigma_k)` in plain NumPy -- the oracle's own route."""
    m = gyro_sigma.shape[0]
    out = np.zeros((3 * m, 3 * m))
    for k in range(m):
        out[3 * k:3 * k + 3, 3 * k:3 * k + 3] = gyro_sigma[k]
    return out


def test_stacked_noise_is_the_exact_congruence_on_a_shared_imu_star():
    """Port-specific, and the FIRST of only two things that constrain I6.

    `SHAPES[3]` is the two-pair star sharing the middle IMU. Both pairs inherit
    IMU 1's noise -- with `+I3` (as pair 0's child) and `-R` (as pair 1's parent)
    -- so `R_g` carries an off-diagonal block that a per-pair assembly drops
    entirely. Anisotropic `Sigma` on top, because with isotropic noise the
    *diagonal* blocks agree exactly and only the cross block would differ.

    The second assertion is the mutation baked into the test: the block-diagonal
    alternative must be numerically distinguishable, or this test would be
    exercising I6 without constraining it (JOINTKF_PORT_PLAN §4, lesson 1).
    """
    shape = SHAPES[3]
    assert shape["m"] == 3 and len(shape["pairs"]) == 2
    fx = build_fixture(shape)

    rng = np.random.default_rng(4900)
    sigma = np.stack([np.diag(v) for v in
                      [[4.0e-4, 1.0e-6, 2.5e-5], [9.0e-4, 1.6e-5, 4.9e-6], [2.0e-4, 7.0e-5, 3.0e-6]]])
    build = kf_build(shape, fx, gyro_sigma=sigma)
    q = fx.random_q(rng)
    sm = stacked(fx, build, q=q, seed=4900)

    L = np.asarray(sm.L)
    expected = L @ _dense_sigma(sigma) @ L.T
    assert_all_close(np.asarray(sm.R), expected, 1.0e-15, "R == L Sigma L^T")

    # ... and the congruence is not vacuous: the shared IMU couples the pairs.
    cross = np.asarray(sm.R)[0:3, 3:6]
    assert np.max(np.abs(cross)) > 1.0e-5, (
        "the shared-IMU cross-covariance vanished -- R_g was assembled per pair")

    block_diagonal = np.zeros_like(expected)
    for e in range(build.n_pairs):
        r = 3 * e
        R_rel = -L[r:r + 3, 3 * int(build.pair_parent[e]):3 * int(build.pair_parent[e]) + 3]
        block_diagonal[r:r + 3, r:r + 3] = (sigma[int(build.pair_child[e])]
                                            + R_rel @ sigma[int(build.pair_parent[e])] @ R_rel.T)
    deviation = np.max(np.abs(expected - block_diagonal))
    assert deviation > 1.0e-5, (
        f"block-diagonal R_g deviates by only {deviation:.2e} -- this shape cannot constrain I6")


def test_isotropic_single_pair_cannot_constrain_i6():
    """Documents the blind spot rather than pretending it does not exist.

    On a single pair with `Sigma = sigma^2 I`, `R Sigma R^T = sigma^2 I` exactly,
    so `L Sigma L^T` and the block-diagonal form coincide to machine noise. Three
    of the four `SHAPES` are in exactly that configuration. If this assertion ever
    fails, the fixture changed and the two tests above are no longer the only
    coverage of I6 -- which is worth knowing, hence the test.
    """
    shape, fx = single_pair(4901)
    build = kf_build(shape, fx, gyro_sigma=np.stack([ISOTROPIC_SIGMA, ISOTROPIC_SIGMA]))
    sm = stacked(fx, build, seed=4901)
    assert_all_close(np.asarray(sm.R), 2.0 * ISOTROPIC_SIGMA, 1.0e-18,
                     "isotropic single-pair R degenerates to Sigma_c + Sigma_p")


# ---------------------------------------------------------------------------
# The frame arbiter: agreement with the marginalized raw-gyro reference
# ---------------------------------------------------------------------------

def _raw_gyro_reference_inputs(fx: ChainFixture, q: np.ndarray, base: int = 0):
    """`(imu_omega_base_rot, imu_joint_jacobian)` for `reference_marginalized`.

    Derivation. Every IMU sees the same base rate plus the chain below it::

        omega_k^W = omega_base^W + (J_k^W - J_base^W) qdot

    Expressing IMU `k`'s reading in its own frame and the shared unknown in the
    BASE IMU's frame gives exactly the oracle's two arguments::

        raw_k = (R_k^T R_base) omega_base^base + R_k^T (J_k^W - J_base^W) qdot

    Computed here from `model.evaluate` (world Jacobians + site rotations), a
    different route from `relative_gyro_jacobian`'s pair difference.
    """
    ev = fx.model.evaluate(q)
    R = np.asarray(ev.site_rot)[:fx.m]                       # (m, 3, 3) world
    J = np.asarray(ev.J_ang)[:fx.m][:, :, np.asarray(fx.model.joint_dof)]   # (m, 3, n)
    rot = np.einsum("kji,jl->kil", R, R[base])               # R_k^T R_base
    jac = np.einsum("kji,kjn->kin", R, J - J[base])          # R_k^T (J_k - J_base)
    return rot, jac


@pytest.mark.parametrize("shape_index", [0, 3])
def test_stacked_pair_rows_match_the_marginalized_raw_gyro_reference(shape_index):
    """THE decisive check for this module -- and the arbiter of the frame question.

    `reference_marginalized` measures the RAW per-IMU gyros with independent
    (block-diagonal) noise over a state augmented by the shared unknown
    `omega_base`, then integrates that nuisance out in the information form. It
    puts `+I3` on each IMU's bias **in that IMU's own frame**. If `b_omega` lived
    in any other frame -- world, or the base IMU's -- the two posteriors could not
    agree, because the bias columns would differ by a rotation that the
    marginalisation does not undo.

    It is also the only route that *derives* the shared-IMU correlation rather
    than asserting it: on shape 3 the two pairs share IMU 1, and the reference
    reaches the same posterior with a strictly block-diagonal input noise.
    Anisotropic `Sigma` again, for the reason in `PORT_NOTES.md`.

    Tolerance, and why it is 1e-8 rather than 1e-12
    ----------------------------------------------
    The oracle inverts an information matrix in which the nuisance carries a
    ZERO prior block, so its condition number is set by the measurement weight:
    `cond(Lambda) ~ sigma^-2 / lambda_min(P^-1)`. Measured, holding everything
    else fixed and sweeping the gyro STD:

        sigma  1e-2   1e-3   1e-4   1e-5   1e-6
        dev    8e-11  2e-08  5e-07  5e-04  1e-02

    -- exactly `eps * cond`, i.e. the oracle's own arithmetic, not a modelling
    disagreement (JOINTKF_PORT_PLAN §4, lesson 2: find the mechanism before
    touching the threshold). `Sigma` is therefore drawn in the 1e-2 band, where
    the oracle is trustworthy, and the tolerance is set an order above the
    residual noise. The magnitude of `Sigma` is irrelevant to what is being
    tested: a wrong bias frame perturbs the posterior by O(0.1-1), seven orders
    above this bound, at any noise level.
    """
    shape = SHAPES[shape_index]
    fx = build_fixture(shape)
    rng = np.random.default_rng(4950 + shape_index)

    sigma = np.stack([np.diag(rng.uniform(2.0e-3, 1.8e-2, size=3)) for _ in range(fx.m)])
    build = kf_build(shape, fx, gyro_sigma=sigma)
    q = fx.random_q(rng)
    raw_gyro = rng.normal(size=(fx.m, 3)) * 0.3

    sm = stacked(fx, build, q=q, gyros=raw_gyro)
    mu, P = seeded_prior_update(build.dim, 4950.0 + shape_index)

    got = reference_update(mu, P, np.asarray(sm.H), np.asarray(sm.z), np.asarray(sm.R))

    rot, jac = _raw_gyro_reference_inputs(fx, q)
    want = reference_marginalized(
        mu, P, n=build.n_joints, imu_bias_col=build.bias_col, raw_gyro=raw_gyro,
        imu_omega_base_rot=rot, imu_joint_jacobian=jac, imu_sigma=sigma,
    )

    assert_all_close(got[0], want[0], 1.0e-8, f"{shape['name']} posterior mean")
    assert_all_close(got[1], want[1], 1.0e-9, f"{shape['name']} posterior covariance")


# ---------------------------------------------------------------------------
# Constant-graph discipline (invariant I7) and the anchor seam
# ---------------------------------------------------------------------------

def test_inactive_anchor_rows_are_masked_not_removed():
    """CLAUDE.md §4: an untrusted anchor keeps its rows, zeroes its residual, and
    takes `r_large * I3` -- never a zeroed `R` block, which makes `S` singular.

    Also pins the row layout `anchors.py` is being written against: pair rows
    first, anchors from `build.anchor_row0`, total `build.n_stacked_rows`.
    """
    shape, fx = single_pair(4960)
    K = 2
    build = kf_build(shape, fx, n_anchors=K,
                     anchor_filtered_mask=np.zeros((K, shape["n"])),
                     anchor_unfiltered_mask=np.zeros((K, 0)),
                     anchor_imu=np.zeros(K, dtype=int))
    rng = np.random.default_rng(4960)
    anchor = measure.AnchorBlock(
        H=rng.normal(size=(3 * K, build.dim)),
        z=rng.normal(size=3 * K),
        R=np.eye(3 * K) * 4.0e-4,
    )
    trusted = np.array([1.0, 0.0])
    sm = stacked(fx, build, seed=4960, trusted_feet=trusted, anchor=anchor)

    assert np.asarray(sm.H).shape == (build.n_stacked_rows, build.dim)
    r0 = build.anchor_row0
    assert r0 == 3 * build.n_pairs and build.n_stacked_rows == 3 * (build.n_pairs + K)

    z = np.asarray(sm.z)
    assert_all_close(z[r0:r0 + 3], np.asarray(anchor.z)[:3], 0.0, "trusted anchor residual kept")
    assert_all_close(z[r0 + 3:], np.zeros(3), 0.0, "untrusted anchor residual zeroed")

    R = np.asarray(sm.R)
    assert_all_close(R[r0:r0 + 3, r0:r0 + 3], 4.0e-4 * np.eye(3), 0.0, "trusted anchor noise")
    assert_all_close(R[r0 + 3:, r0 + 3:], PARAMS.r_large * np.eye(3), 0.0, "untrusted anchor -> R_LARGE")
    assert np.min(np.linalg.eigvalsh(R)) > 0.0, "masked R must stay strictly PD"
    assert_all_close(R[:r0, r0:], np.zeros((r0, 3 * K)), 0.0,
                     "gyro and anchor noise are independent sources")


def test_shapes_are_constant_across_gyros_and_trust_masks():
    """Invariant I7: the jaxpr must not depend on the data.

    What this actually proves (JOINTKF_PORT_PLAN §4, lesson 3) is narrow: traced
    values cannot change a jaxpr, so what would fail here is a Python branch on a
    traced array -- which raises -- or a data-dependent shape. That is exactly the
    failure mode being guarded, so the narrow claim is the useful one.
    """
    shape, fx = single_pair(4970)
    K = 1
    build = kf_build(shape, fx, n_anchors=K,
                     anchor_filtered_mask=np.zeros((K, shape["n"])),
                     anchor_unfiltered_mask=np.zeros((K, 0)),
                     anchor_imu=np.zeros(K, dtype=int))
    rng = np.random.default_rng(4970)
    q = fx.random_q(rng)
    J_rel, R_rel = measure.pair_frames(fx.model, q)
    anchor = measure.AnchorBlock(H=np.zeros((3 * K, build.dim)), z=np.zeros(3 * K),
                                R=np.eye(3 * K) * 4.0e-4)

    fn = jax.jit(lambda g, t: measure.build_stacked(
        build, PARAMS, gyros=g, trusted_feet=t, J_rel=J_rel, R_rel=R_rel, anchor=anchor))

    jaxprs = {
        str(jax.make_jaxpr(fn)(rng.normal(size=(fx.m, 3)), np.array(t)))
        for t in ([0.0], [1.0])
    }
    assert len(jaxprs) == 1, "the trust mask changed the graph -- it must be a value, not a branch"


def test_float32_input_is_rejected():
    """Invariant I8. The check reads the RAW argument: casting first would upcast
    the leak and make the assertion vacuous.
    """
    shape, fx = single_pair(4980)
    build = kf_build(shape, fx)
    rng = np.random.default_rng(4980)
    J_rel, R_rel = measure.pair_frames(fx.model, fx.random_q(rng))
    with pytest.raises(TypeError, match="float64"):
        measure.build_stacked(build, PARAMS, gyros=np.zeros((fx.m, 3), dtype=np.float32),
                              J_rel=J_rel, R_rel=R_rel)
