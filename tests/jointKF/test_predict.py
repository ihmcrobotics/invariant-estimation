r"""Port of `JointLevelKFPredictTest.java` (7 tests) plus the two ``F`` structure
tests from `JointLevelKFTransitionNoiseTest` — gate G6.

The time update has exactly two moving parts and both are pinned here:

* ``F = I + A Δt`` is **exact**, not a first-order truncation (``A`` is nilpotent
  on the ``(q, q̇)`` block and zero on the bias block).  `test_mean_propagation_exact`
  therefore asserts the closed form ``q⁺ᵢ = qᵢ + Δt q̇ᵢ`` to 1e-9 with no
  discretisation slack, and `test_build_f_equals_i_plus_a_dt` asserts the matrix
  itself against an independently-written ``I`` with one off-diagonal band;
* ``P⁻ = F P Fᵀ + Q`` — checked against a NumPy recomputation, and, structurally,
  by `test_bias_block_growth_diagonal`: the bias marginal must grow by *exactly*
  ``Δt σ_b² I`` and nothing else, which is only true if ``F`` has no bias↔joint
  coupling at all.

``Q`` is an **argument** to `predict` (`CONTRACT_CARD.md` §4 / I10), so these
tests build it themselves from the closed form rather than depending on
`jointKF.process`: the exact Van-Loan blocks of a double integrator driven by an
acceleration covariance ``Q_a`` (`_oracles.van_loan_blocks`) plus the bias
random-walk block ``Δt σ_b² I``.
"""
import numpy as np
import pytest

from invariant_estimation.jointKF.predict import build_transition, predict
from invariant_estimation.jointKF.state import JointKFState, default_params, init_state

from ._oracles import (
    SHAPES,
    assert_all_close,
    assert_positive_semidefinite,
    assert_symmetric,
    seeded_prior_predict,
    shape_dims,
    spd,
    stub_build,
    van_loan_blocks,
)
from .test_state import single_pair

PARAMS = default_params()
DT = PARAMS.dt                       # 1e-3
IMU_BIAS_VAR = PARAMS.imu_bias_process_var    # 1e-4


def process_noise(shape, params=PARAMS) -> np.ndarray:
    """Closed-form ``Q`` on the scalar-CWNA path — independent of `jointKF.process`.

    ``Q_a = σ_a² I`` (the fallback used when no mass matrix is wired), discretised
    exactly over one step, with the per-IMU gyro-bias random walk on the diagonal
    block.  Written in NumPy from the closed form so it is an oracle for the
    covariance propagation rather than a restatement of the code under test.
    """
    n, m, dim = shape_dims(shape)
    qa = params.sigma_accel ** 2 * np.eye(n)
    b = van_loan_blocks(qa, params.dt)
    Q = np.zeros((dim, dim))
    Q[:n, :n] = b["qq"]
    Q[:n, n:2 * n] = b["qqd"]
    Q[n:2 * n, :n] = b["qdq"]
    Q[n:2 * n, n:2 * n] = b["qdqd"]
    Q[2 * n:, 2 * n:] = params.dt * params.imu_bias_process_var * np.eye(3 * m)
    return Q


def _prior(shape, seed):
    """Java `seededPrior(f, seed)` → a `JointKFState` carry."""
    n, m, _ = shape_dims(shape)
    x, P = seeded_prior_predict(n, m, seed)
    return JointKFState(x=x, P=P)


# ---------------------------------------------------------------------------
# The transition matrix (`JointLevelKFTransitionNoiseTest`, F half)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_build_f_structure_and_exactness(shape):
    """`testBuildFStructureAndExactness`: identity blocks, one `dt·I` band, no coupling.

    The bias↔joint blocks being exactly zero is the load-bearing part: the gyro
    bias is observable only through the measurement (I6), never through the
    dynamics.
    """
    n, m, dim = shape_dims(shape)
    F = np.asarray(build_transition(stub_build(shape), PARAMS))
    assert F.shape == (dim, dim)
    assert_all_close(F[:n, :n], np.eye(n), 1.0e-12, "F_qq")
    assert_all_close(F[n:2 * n, n:2 * n], np.eye(n), 1.0e-12, "F_qdqd")
    assert_all_close(F[2 * n:, 2 * n:], np.eye(3 * m), 1.0e-12, "F_bb")
    assert_all_close(F[:n, n:2 * n], DT * np.eye(n), 1.0e-12, "F_q_qd")
    assert_all_close(F[n:2 * n, :n], np.zeros((n, n)), 1.0e-12, "F_qd_q")
    assert_all_close(F[:2 * n, 2 * n:], np.zeros((2 * n, 3 * m)), 1.0e-12, "F joint->bias")
    assert_all_close(F[2 * n:, :2 * n], np.zeros((3 * m, 2 * n)), 1.0e-12, "F bias->joint")


def test_build_f_equals_i_plus_a_dt():
    """`testBuildFEqualsIPlusADt`: `F == I(dim)` with `F[i, n+i] = dt`, tol 1e-12."""
    shape = single_pair(6, 1, 5)                       # n=3, m=2
    n, _, dim = shape_dims(shape)
    expected = np.eye(dim)
    expected[np.arange(n), n + np.arange(n)] = DT
    F = build_transition(stub_build(shape), PARAMS)
    assert_all_close(F, expected, 1.0e-12, "F")


def test_build_f_is_exact_for_any_dt():
    """Port-specific: `F(Δt)` is `expm(A Δt)` for a **large** Δt, not just 1 ms.

    `I + A Δt` being exact rather than truncated is the claim the module docstring
    makes; a Δt of 0.5 s makes any second-order term visible at 1e-12 (it would be
    `½Δt² = 0.125`), so this is the test that actually constrains it.
    """
    from scipy.linalg import expm

    shape = single_pair(8, 1, 7)
    n, _, dim = shape_dims(shape)
    dt = 0.5
    A = np.zeros((dim, dim))
    A[np.arange(n), n + np.arange(n)] = 1.0
    F = build_transition(stub_build(shape), default_params(dt=dt))
    assert_all_close(F, expm(A * dt), 1.0e-12, "F vs expm(A dt)")


# ---------------------------------------------------------------------------
# The time update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_shapes_preserved(shape):
    """`testShapesPreserved`."""
    _, _, dim = shape_dims(shape)
    build = stub_build(shape)
    out = predict(_prior(shape, 0), build_transition(build, PARAMS), process_noise(shape))
    assert out.x.shape == (dim,)
    assert out.P.shape == (dim, dim)


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_mean_propagation_exact(shape):
    """`testMeanPropagationExact`: `q⁺ᵢ = qᵢ + dt·q̇ᵢ` (1e-9); `q̇`, `b_ω` unchanged (1e-12).

    The seeded prior separates the three segments by two orders of magnitude
    (`i+1` / `i+1+100` / `i+1+1000`), so a transposed or mis-indexed `dt` band
    shows up as a gross error rather than as rounding.
    """
    n, m, _ = shape_dims(shape)
    prior = _prior(shape, 1)
    x = np.asarray(prior.x)
    out = predict(prior, build_transition(stub_build(shape), PARAMS), process_noise(shape))
    xp = np.asarray(out.x)
    assert_all_close(xp[:n], x[:n] + DT * x[n:2 * n], 1.0e-9, "position propagation")
    assert_all_close(xp[n:2 * n], x[n:2 * n], 1.0e-12, "velocity segment")
    assert_all_close(xp[2 * n:], x[2 * n:], 1.0e-12, "bias segment")


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_covariance_matches_builders(shape):
    """`testCovarianceMatchesBuilders`: `P⁻ == F P Fᵀ + Q`, tol 1e-6.

    Java reads `getTransitionMatrix()` / `getProcessNoise()` off the filter; here
    `F` comes from the same builder and `Q` from the independent closed form, and
    the product is recomputed in NumPy.
    """
    prior = _prior(shape, 2)
    F = np.asarray(build_transition(stub_build(shape), PARAMS))
    Q = process_noise(shape)
    expected = F @ np.asarray(prior.P) @ F.T + Q
    out = predict(prior, F, Q)
    assert_all_close(out.P, expected, 1.0e-6, "P after predict")


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_covariance_symmetric_psd(shape):
    """`testCovarianceSymmetricPSD`."""
    out = predict(
        _prior(shape, 3), build_transition(stub_build(shape), PARAMS), process_noise(shape)
    )
    assert_symmetric(out.P, 1.0e-6, "P after predict")
    assert_positive_semidefinite(out.P, "P after predict")


def test_bias_block_growth_diagonal():
    """`testBiasBlockGrowthDiagonal`: bias marginal grows by exactly `dt·1e-4·I`.

    This is the structural test for "`F` is the identity on the bias block and
    couples nothing into it": any bias↔joint entry in `F` would add a term from
    the (order-1) joint covariance, swamping the 1e-7 increment.
    """
    shape = single_pair(8, 1, 7)
    n, m, _ = shape_dims(shape)
    prior = _prior(shape, 4)
    before = np.asarray(prior.P)[2 * n:, 2 * n:]
    out = predict(prior, build_transition(stub_build(shape), PARAMS), process_noise(shape))
    after = np.asarray(out.P)[2 * n:, 2 * n:]
    expected = DT * IMU_BIAS_VAR * np.eye(3 * m)
    assert_all_close(after - before, expected, 1.0e-9, "bias block increment")


def test_deterministic():
    """`testDeterministic`: repeated predicts from the same carry agree bit-for-bit.

    Java asserts `tol = 0.0` against a second run of the same filter; per
    CLAUDE.md §5 the port asserts determinism against its **own** repeat run.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    x0 = 0.01 * np.arange(1, dim + 1)
    P0 = spd(dim, 5)
    F = build_transition(stub_build(shape), PARAMS)
    Q = process_noise(shape)
    a = predict(JointKFState(x=x0, P=P0), F, Q)
    b = predict(JointKFState(x=x0, P=P0), F, Q)
    assert_all_close(a.x, b.x, 0.0, "x determinism")
    assert_all_close(a.P, b.P, 0.0, "P determinism")


def test_diffuse_position_variance_grows():
    """`testDiffusePositionVarianceGrows`: 5 measurement-free predicts never shrink `σ²_q`.

    Free diffusion: with no measurement the only covariance sources are the
    congruence `F P Fᵀ` (which cannot reduce the marginal here — the velocity
    variance feeds forward into position) and `+Q` (PSD).
    """
    shape = single_pair(8, 1, 7)
    n, _, _ = shape_dims(shape)
    build = stub_build(shape)
    state = init_state(build, PARAMS)
    F, Q = build_transition(build, PARAMS), process_noise(shape)
    prev = np.diag(np.asarray(state.P))[:n]
    for _ in range(5):
        state = predict(state, F, Q)
        assert_positive_semidefinite(state.P, "P along free diffusion")
        cur = np.diag(np.asarray(state.P))[:n]
        assert np.all(cur >= prev - 1.0e-9)
        prev = cur


def test_jit_matches_eager():
    """Port-specific (I7): the step is jit-safe and the graph does not depend on data."""
    import jax

    shape = single_pair(8, 1, 7)
    prior = _prior(shape, 6)
    F = build_transition(stub_build(shape), PARAMS)
    Q = process_noise(shape)
    jitted = jax.jit(predict)(prior, F, Q)
    eager = predict(prior, F, Q)
    assert_all_close(jitted.x, eager.x, 0.0, "jit x")
    assert_all_close(jitted.P, eager.P, 0.0, "jit P")
