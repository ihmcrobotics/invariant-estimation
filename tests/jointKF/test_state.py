r"""Port of `JointLevelKFStateTest.java` (10 tests) — gate G6.

What this class pins is the **state layout and the prior**, nothing dynamic:

* ``x = [q (n) ; q̇ (n) ; b_ω (3m)]`` with ``dim = 2n + 3m`` and ``m`` the number
  of **distinct IMUs** (`CONTRACT_CARD.md` §1) — the ordering is asserted by
  seeding the three segments to values that cannot be confused (`testXOrdering`);
* ``q`` seeded from the encoders, ``q̇`` and ``b_ω`` from exact zero;
* the diagonal prior ``(1e-6, 1.0, 2.5e-3)`` and its ordering
  ``pos < bias < vel`` — encoders are trusted at init, velocity is genuinely
  unknown, bias sits between.  That ordering is *intent*, so it gets its own test
  rather than being implied by the numbers.

These are pure linear algebra over ``(n, m)``: the Java fixtures differ only in
dimension here, so the dimension-only `stub_build` stands in for the geometry
(`_oracles.stub_build`).
"""
import numpy as np
import pytest

from invariant_estimation.jointKF.state import default_params, init_state

from ._oracles import (
    SHAPES,
    assert_all_close,
    assert_positive_semidefinite,
    assert_symmetric,
    shape_dims,
    stub_build,
)

# Java `JointLevelKFStateTest` constants.
INIT_POS_VAR = 1.0e-6
INIT_VEL_VAR = 1.0
INIT_BIAS_VAR = 2.5e-3

PARAMS = default_params()


def single_pair(num_chain_joints: int, imu_after_joint: int, foot_after_joint: int) -> dict:
    """Java `JointLevelKFTestFixture.singlePair(seed, chain, imuAfter, footAfter)`.

    Two IMUs bracketing a sub-chain, so ``m = 2`` and ``n`` is the number of
    joints strictly between them.  The Java seed only drives the random link
    geometry, which these tests never touch — hence no seed argument.
    """
    n = foot_after_joint - imu_after_joint - 1
    return {
        "name": f"single_pair_n{n}",
        "chain": num_chain_joints,
        "imus": (imu_after_joint, foot_after_joint),
        "pairs": ((0, 1),),
        "n": n,
        "m": 2,
    }


def _init(shape, q0=None):
    build = stub_build(shape)
    return build, init_state(build, PARAMS, q0=q0)


# ---------------------------------------------------------------------------
# Shapes and dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_init_state_shapes(shape):
    """`testInitStateShapes`: `dim == 2n+3m`, `x` is (dim,), `P` is (dim, dim)."""
    n, m, dim = shape_dims(shape)
    build, state = _init(shape)
    assert build.dim == 2 * n + 3 * m
    assert state.x.shape == (dim,)
    assert state.P.shape == (dim, dim)


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_inferred_dims(shape):
    """`testInferredDims`: `n > 0`, `m >= 2` (a pair needs two IMUs), `2n+3m == dim`."""
    n, m, dim = shape_dims(shape)
    build, _ = _init(shape)
    assert n > 0
    assert m >= 2
    assert 2 * n + 3 * m == build.dim == dim


# ---------------------------------------------------------------------------
# Mean: encoder seeding and segment ordering
# ---------------------------------------------------------------------------

def test_init_state_defaults_zero():
    """`testInitStateDefaultsZero`: all encoders 0 ⇒ every entry of `x` exactly 0."""
    shape = single_pair(8, 1, 7)                       # n=5, m=2
    n, m, dim = shape_dims(shape)
    _, state = _init(shape, q0=np.zeros(n))
    assert_all_close(state.x, np.zeros(dim), 0.0, "x")


def test_q0_seed():
    """`testQ0Seed`: encoder `i` = `0.11*(i+1)` lands in `x[i]`, tol 1e-12."""
    shape = single_pair(8, 1, 7)
    n, _, _ = shape_dims(shape)
    encoders = 0.11 * np.arange(1, n + 1)
    _, state = _init(shape, q0=encoders)
    assert_all_close(state.q(n), encoders, 1.0e-12, "q segment")


def test_x_ordering():
    """`testXOrdering`: `q = 1` (1e-12), `q̇ = 0` and `b_ω = 0` **exactly**.

    The decisive check of the `[q ; q̇ ; b_ω]` ordering: any permutation of the
    segments moves a 1.0 into a segment asserted to be exactly zero.
    """
    shape = single_pair(8, 1, 7)
    n, m, _ = shape_dims(shape)
    _, state = _init(shape, q0=np.ones(n))
    assert_all_close(state.q(n), np.ones(n), 1.0e-12, "q segment")
    assert_all_close(state.q_dot(n), np.zeros(n), 0.0, "q_dot segment")
    assert_all_close(state.b_omega(n), np.zeros(3 * m), 0.0, "bias segment")


# ---------------------------------------------------------------------------
# Prior covariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_marginal_blocks_match_p(shape):
    """`testMarginalBlocksMatchP`: prior diagonal is `(1e-6, 1.0, 2.5e-3)` by segment."""
    n, m, _ = shape_dims(shape)
    _, state = _init(shape)
    d = np.diag(np.asarray(state.P))
    assert_all_close(d[:n], np.full(n, INIT_POS_VAR), 1.0e-12, "position variances")
    assert_all_close(d[n:2 * n], np.full(n, INIT_VEL_VAR), 1.0e-12, "velocity variances")
    assert_all_close(d[2 * n:], np.full(3 * m, INIT_BIAS_VAR), 1.0e-12, "bias variances")


def test_velocity_block_excludes_bias_block():
    """`testVelocityBlockExcludesBiasBlock`: the velocity marginal is pure velocity.

    An off-by-`n` slice would fold bias rows into `sigma_q_dot`; the second
    assertion is what catches it, since `2.5e-3` is far from `1.0`.
    """
    shape = single_pair(6, 1, 5)                       # n=3, m=2
    n, _, _ = shape_dims(shape)
    _, state = _init(shape)
    vel = np.diag(np.asarray(state.sigma_q_dot(n)))
    assert vel.shape == (n,)
    assert_all_close(vel, np.full(n, INIT_VEL_VAR), 1.0e-12, "velocity marginal")
    assert np.all(np.abs(vel - INIT_BIAS_VAR) > 1.0e-6)


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_p_symmetric_and_psd(shape):
    """`testPSymmetricAndPSD`."""
    _, state = _init(shape)
    assert_symmetric(state.P, 1.0e-12, "prior P")
    assert_positive_semidefinite(state.P, "prior P")


def test_prior_confidence_ordering():
    """`testPriorConfidenceOrdering`: `pos < bias < vel`.

    Intent, not coincidence — see `init_state`'s docstring.
    """
    shape = single_pair(8, 1, 7)
    n, _, _ = shape_dims(shape)
    P = np.asarray(_init(shape)[1].P)
    pos, vel, bias = P[0, 0], P[n, n], P[2 * n, 2 * n]
    assert pos < bias
    assert bias < vel


def test_prior_variances_positive():
    """`testPriorVariancesPositive`: every prior diagonal entry is strictly positive."""
    shape = single_pair(8, 1, 7)
    P = np.asarray(_init(shape)[1].P)
    assert np.all(np.diag(P) > 0.0)
