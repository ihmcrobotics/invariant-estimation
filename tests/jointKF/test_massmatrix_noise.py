"""Port of `JointLevelKFMassMatrixNoiseTest.java` (10 tests).

Locks in the Schur-complement process noise: unmodeled joint torque `w_tau`
acting through **floating-base** dynamics gives `dq_ddot = Lambda^-1 w_tau` with
`Lambda = M_jj - M_jb M_bb^-1 M_bj`; with per-joint `sigma_tau` the acceleration
covariance is the Gram form `Qa = Lambda_eff^-1 diag(sigma_tau^2) Lambda_eff^-T`,
Van-Loan discretised into `dt^3/3 Qa`, `dt^2/2 Qa`, `dt Qa`.

Deviation from the Java, and why it is still an oracle test
-----------------------------------------------------------
Java drives these tests with a randomly generated revolute chain and a *second*
`CompositeRigidBodyMassMatrixCalculator` as the reference. This port drives them
with a **synthetic symmetric-PD `M`** (`spd(size, seed)`, the suite's own
bit-for-bit generator) plus explicit filtered / nuisance DoF index arrays.

That loses nothing the class was testing. `JointLevelKFMassMatrixNoiseTest`
asserts *algebra* -- Schur, Gram, Van Loan, block layout -- against a reference
built from the same `M` by an independent linear-algebra path (LU inverse here,
Cholesky solve in the filter). Whether `M` came from a CRB algorithm or from
`sin`-fills is irrelevant to every assertion in the class. What a synthetic `M`
cannot test is that `M` is the *right* inertia for the robot -- and that is the
job of the separate MJX armature-equivalence oracle (CLAUDE.md G3), not of this
class.

The DoF index split is deliberately **non-contiguous** -- base 6 DoF plus two
interleaved "gap" joints -- so an implementation that slices instead of gathering
fails here rather than at G9.

`config/filter_cfg.yaml` supplies every constant; `stub_build` supplies
`alpha = 0.15`, `tau_max = NaN` (so `sigma_tau` takes the 5.0 fallback, exactly
as Java's effort-limit-free random chain does) and `rotor = 0.005` (the default
floor, exactly as Java's unmatched random joint names do).
"""
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  -- enables float64
from invariant_estimation.jointKF import process
from invariant_estimation.jointKF.state import default_params

from ._oracles import (
    SHAPES,
    assert_all_close,
    assert_positive_semidefinite,
    assert_symmetric,
    reference_qa,
    reference_schur,
    scaled_identity,
    shape_dims,
    spd,
    stub_build,
)

PARAMS = default_params()

#: Gap joints on the chain that are NOT filter states (Alex: the ankles). They
#: are eliminated alongside the floating base's 6 DoF -- the Java "considered
#: subsystem" trick -- because from the filter's point of view they are equally
#: unmodelled recoil paths.
N_GAP = 2


# ---------------------------------------------------------------------------
# Synthetic fixture -- shared with test_standing_stability.py
# ---------------------------------------------------------------------------

def dof_split(n: int, n_gap: int = N_GAP) -> tuple[np.ndarray, np.ndarray, int]:
    """`(filtered_idx, nuisance_idx, D)` with the MJX DoF convention.

    Floating base occupies DoF 0..5 (MJX puts the free joint's 6 DoF first), then
    hinges. Two hinges are marked as gap joints and land in the nuisance set, so
    `filtered_idx` is non-contiguous and `nuisance_idx` is not a prefix.
    """
    n_nu = 6 + n_gap
    D = n + n_nu
    hinges = np.arange(6, D)
    gap = hinges[np.linspace(1, len(hinges) - 2, n_gap).astype(int)]
    filtered = np.array([d for d in hinges if d not in set(gap.tolist())], dtype=int)
    assert filtered.size == n
    return filtered, np.concatenate([np.arange(6), gap]).astype(int), D


def synthetic_mass_matrix(D: int, seed: float, light_dofs=(), light_scale: float = 1.0) -> np.ndarray:
    """`spd(D, seed)`, optionally congruence-scaled to create a light mode.

    `diag(d) M diag(d)` preserves symmetric positive-definiteness exactly, so
    shrinking one DoF's scale is a legitimate inertia -- it is what a distal
    link with tiny link-side inertia looks like. Used only by
    `test_standing_stability` to reach the near-singular regime.
    """
    M = spd(D, seed)
    d = np.ones(D)
    if len(light_dofs):
        d[list(light_dofs)] = light_scale
    return (d[:, None] * M) * d[None, :]


def mass_fixture(shape: dict, seed: float, *, light_dofs=(), light_scale: float = 1.0,
                 rotor: float | None = None):
    """`(build, M)` on the mass-matrix path.

    `rotor` overrides `stub_build`'s 0.005 (the *unmatched-name default*, which is
    what Java's randomly-named chain gets). Pass a real table value when the test
    needs the rotor term to be numerically visible -- see `near_singular_fixture`.
    """
    n = shape["n"]
    filtered, nuisance, D = dof_split(n)
    M = synthetic_mass_matrix(D, seed, light_dofs, light_scale)
    build = stub_build(
        shape, dof_joint=filtered, dof_nuisance=nuisance, use_mass_matrix=True
    )
    if rotor is not None:
        build = build._replace(rotor_inertia=np.full(n, rotor))
    return build, M


#: Congruence scale applied to the two lightest FILTERED hinge DoF, manufacturing
#: a near-singular articulated mode -- the regime the rotor floor exists for.
LIGHT_SCALE = 0.08

#: Real Alex KNEE / HIP_Y reflected rotor inertia from `config/filter_cfg.yaml`.
#: `stub_build`'s 0.005 default is ~30x smaller and, against the map's 3e-3
#: relative tolerance on a well-conditioned `Lambda`, is numerically INVISIBLE --
#: a double-add of it shifts `Qa` by ~7e-4 relative and slips through. Every test
#: that needs to *constrain* the rotor term rather than merely exercise it uses
#: this fixture (JOINTKF_PORT_PLAN §4, lesson 1).
ROTOR_KNEE = 0.167


def near_singular_fixture(shape: dict, seed: float):
    """`(build, M)` with a light articulated mode and a realistic rotor inertia.

    The light DoF are drawn from the FILTERED set, not the gap joints, so the
    near-singular mode lands in `Lambda` where the rotor term can act on it.
    """
    filtered, _, _ = dof_split(shape["n"])
    return mass_fixture(
        shape, seed, light_dofs=tuple(filtered[:2]), light_scale=LIGHT_SCALE,
        rotor=ROTOR_KNEE,
    )


def numpy_sigma_tau(build) -> np.ndarray:
    """Java `referenceSigmaTau` -- independent of `process.sigma_tau_per_joint`."""
    alpha = np.asarray(build.alpha, dtype=float)
    tau = np.asarray(build.tau_max, dtype=float)
    ok = np.isfinite(tau) & (tau > 0.0)
    return np.where(ok, alpha * np.where(ok, tau, 0.0), PARAMS.sigma_tau)


def ref_qa(build, M: np.ndarray) -> np.ndarray:
    """Java `referenceQa`: LU Schur -> post-Schur rotor add -> dense sandwich."""
    lam = reference_schur(M, np.asarray(build.dof_joint), np.asarray(build.dof_nuisance))
    lam_eff = lam + np.diag(np.asarray(build.rotor_inertia, dtype=float))
    return reference_qa(lam_eff, numpy_sigma_tau(build))


def rel_tol(expected: np.ndarray) -> float:
    """Java `relTol(expected) = 3e-3 * max(1e-30, elementMaxAbs(expected))`.

    Loosened from 1e-8 in the Java because the filter takes a Cholesky/Gram path
    and the reference an LU one. This port keeps the number verbatim rather than
    tightening it: the tolerance is a statement about how much two independent
    linear-algebra routes may disagree, which does not change with the language.
    """
    return 3.0e-3 * max(1e-30, float(np.max(np.abs(expected))))


def Q_of(build, M):
    """`getProcessNoise()` on the mass-matrix path.

    `rotor=build.rotor_inertia` is passed **explicitly**: the synthetic `M` has
    no MJCF `armature` folded in, so this is the Java post-Schur add. Production
    passes `M` from MJX and leaves `rotor` at its `ROTOR_IN_MASS_MATRIX` default
    -- doing both is the double-add trap (CLAUDE.md §6).
    """
    return np.asarray(process.build_process_noise(build, PARAMS, M, rotor=build.rotor_inertia))


# ---------------------------------------------------------------------------
# 1. testMassMatrixPathEnabledOnlyWithModel
# ---------------------------------------------------------------------------

def test_mass_matrix_path_enabled_only_with_model():
    """The path is selected by whether `M` is supplied, and the build declares it.

    In the Java the flag tracks "was a robot model handed to the constructor".
    Here `M` is an argument, so the port asserts the two stay consistent: a build
    that declares `use_mass_matrix` and is then called with `M=None` is a wiring
    bug and raises rather than silently degrading to the 50 rad/s^2 fallback.
    """
    shape = SHAPES[0]
    build_mass, M = mass_fixture(shape, 6000.0)
    build_scalar = stub_build(shape, use_mass_matrix=False)

    assert build_mass.use_mass_matrix is True
    assert build_scalar.use_mass_matrix is False

    with pytest.raises(ValueError, match="use_mass_matrix"):
        process.build_process_noise(build_mass, PARAMS, M=None)

    # sigma_tau closed form agrees with what build.py resolves into the build
    assert_all_close(
        process.sigma_tau_per_joint(build_mass, PARAMS),
        np.asarray(build_mass.sigma_tau),
        1e-15,
        "sigma_tau vs build",
    )
    assert_all_close(
        process.sigma_tau_per_joint(build_mass, PARAMS), numpy_sigma_tau(build_mass),
        1e-15, "sigma_tau vs oracle",
    )


# ---------------------------------------------------------------------------
# 2. testProcessNoiseEqualsVanLoanOfSchurComplementInverseSquared
# ---------------------------------------------------------------------------

def test_process_noise_equals_van_loan_of_schur_complement_inverse_squared():
    """All four joint blocks of `Q` against the independent LU/dense reference."""
    fixtures = [(sh, mass_fixture(sh, 6100.0)) for sh in SHAPES]
    # A second pass on the near-singular / realistic-rotor fixture. Without it
    # this test is BLIND to a rotor double-add: at rotor = 0.005 on a
    # well-conditioned Lambda the doubling moves Qa by ~7e-4 relative, an order
    # under the map's 3e-3 tolerance. Verified by mutation.
    fixtures += [(sh, near_singular_fixture(sh, 6101.0)) for sh in SHAPES]

    for shape, (build, M) in fixtures:
        n, _, _ = shape_dims(shape)
        Q = Q_of(build, M)
        qa = ref_qa(build, M)

        for name, block, expected in (
            ("qq", Q[:n, :n], (PARAMS.dt ** 3 / 3.0) * qa),
            ("q-qd", Q[:n, n:2 * n], (PARAMS.dt ** 2 / 2.0) * qa),
            ("qd-q", Q[n:2 * n, :n], (PARAMS.dt ** 2 / 2.0) * qa),
            ("qdqd", Q[n:2 * n, n:2 * n], PARAMS.dt * qa),
        ):
            assert_all_close(block, expected, rel_tol(expected), f"{shape['name']} {name}")


# ---------------------------------------------------------------------------
# 3. testSchurComplementIsSymmetricPDAndDominatedByLockedInertia
# ---------------------------------------------------------------------------

def test_schur_complement_is_symmetric_pd_and_dominated_by_locked_inertia():
    """`Lambda` symmetric PD, and `M_jj - Lambda = M_jb M_bb^-1 M_bj` is PSD.

    The PSD ordering `Lambda <= M_jj` is the physics: a free base recoils, so the
    joints accelerate MORE per unit torque than a locked base predicts, and the
    honest process noise `Lambda^-2` is correspondingly LARGER than `M_jj^-2`.
    Getting the sign of the recoil term wrong inverts this.

    (Java's "bent configuration" is a second `q`; here a second `spd` seed plays
    the same role -- the assertion is a property of any SPD `M`.)
    """
    for shape in SHAPES:
        build, M = mass_fixture(shape, 6800.0)
        lam = np.asarray(process.schur_complement(M, build.dof_joint, build.dof_nuisance))
        M_jj = M[np.ix_(np.asarray(build.dof_joint), np.asarray(build.dof_joint))]

        assert_symmetric(lam, 1e-9 * max(1.0, float(np.max(np.abs(lam)))), "Lambda")
        assert_positive_semidefinite(lam, "Lambda")
        assert_positive_semidefinite(M_jj - lam, "M_jj - Lambda (recoil term)")
        # strict: the recoil term is nonzero, i.e. the Schur term actually fired
        assert np.max(np.abs(M_jj - lam)) > 1e-6


# ---------------------------------------------------------------------------
# 4. testVanLoanBlocksAreExactlySymmetric
# ---------------------------------------------------------------------------

def test_van_loan_blocks_are_exactly_symmetric():
    """`tol = 0.0` -- the Gram form's structural symmetry, propagated into `Q`.

    THE DECISIVE ASSERTION for the Gram-vs-dense choice. `Qa = Y Y^T` is
    bit-symmetric on XLA (`Qa[i,j]` and `Qa[j,i]` are the same reduction over the
    same summands in the same order); the dense sandwich
    `Lambda_eff^-1 diag(s^2) Lambda_eff^-T` agrees to 1e-17 but is NOT. This test
    is only able to see the difference because `process` applies no
    `0.5(A + A^T)` anywhere on the `Qa -> Q` path.
    """
    for shape in SHAPES:
        n, _, _ = shape_dims(shape)
        build, M = mass_fixture(shape, 6900.0)
        Q = Q_of(build, M)

        qq, qdqd = Q[:n, :n], Q[n:2 * n, n:2 * n]
        qqd, qdq = Q[:n, n:2 * n], Q[n:2 * n, :n]

        assert np.array_equal(qq, qq.T), f"{shape['name']}: qq not bit-symmetric"
        assert np.array_equal(qdqd, qdqd.T), f"{shape['name']}: qdqd not bit-symmetric"
        assert np.array_equal(qqd, qdq.T), f"{shape['name']}: qqd != qdq^T bit-exactly"


# ---------------------------------------------------------------------------
# 5. testBiasBlockUntouchedByMassMatrixPath
# ---------------------------------------------------------------------------

def test_bias_block_untouched_by_mass_matrix_path():
    """Bias block is the plain random walk; joint<->bias cross blocks are zero."""
    for shape in SHAPES:
        n, m, _ = shape_dims(shape)
        m_dof = 3 * m
        build, M = mass_fixture(shape, 6200.0)
        Q = Q_of(build, M)

        assert_all_close(
            Q[2 * n:, 2 * n:],
            scaled_identity(m_dof, PARAMS.dt * PARAMS.imu_bias_process_var),
            1e-12,
            "Q bias",
        )
        assert_all_close(Q[:2 * n, 2 * n:], np.zeros((2 * n, m_dof)), 1e-12, "Q joint-bias")
        assert_all_close(Q[2 * n:, :2 * n], np.zeros((m_dof, 2 * n)), 1e-12, "Q bias-joint")


# ---------------------------------------------------------------------------
# 6. testProcessNoiseSymmetricPSDOnMassMatrixPath
# ---------------------------------------------------------------------------

def test_process_noise_symmetric_psd_on_mass_matrix_path():
    for shape in SHAPES:
        build, M = mass_fixture(shape, 6300.0)
        Q = Q_of(build, M)
        assert_symmetric(Q, 1e-12, f"Q {shape['name']}")
        assert_positive_semidefinite(Q, f"Q {shape['name']}")


# ---------------------------------------------------------------------------
# 7. testProcessNoiseCouplesJointsThroughInertia
# ---------------------------------------------------------------------------

def test_process_noise_couples_joints_through_inertia():
    """Dense `Lambda_eff^-2` couples joints; the scalar fallback provably cannot.

    This is the whole reason for carrying an inertia: a diagonal torque
    uncertainty comes out as CORRELATED acceleration uncertainty across the
    kinematic tree, and that correlation is what makes the exported `Sigma_q`
    honest for the InEKF's `N = J_C Sigma_q J_C^T` pushforward.
    """
    shape = SHAPES[0]                                   # n = 8
    n, _, _ = shape_dims(shape)
    build, M = mass_fixture(shape, 6400.0)
    qdqd = Q_of(build, M)[n:2 * n, n:2 * n]
    off = qdqd - np.diag(np.diag(qdqd))
    assert np.max(np.abs(off)) > 0.0

    scalar = np.asarray(
        process.build_process_noise(stub_build(shape, use_mass_matrix=False), PARAMS, M=None)
    )[n:2 * n, n:2 * n]
    scalar_off = scalar - np.diag(np.diag(scalar))
    assert np.count_nonzero(scalar_off) == 0, "scalar-CWNA path must be exactly diagonal"


# ---------------------------------------------------------------------------
# 8. testProcessNoiseIsConfigurationDependent
# ---------------------------------------------------------------------------

def test_process_noise_is_configuration_dependent():
    """`Q` tracks `M`, and equals `dt * Qa` at the NEW inertia.

    Java changes `q` to a large alternating bend; here a second `spd` seed is the
    second configuration. The assertion that matters is the second one --
    recomputing at the new inertia, not merely changing.
    """
    shape = SHAPES[0]
    n, _, _ = shape_dims(shape)
    build, M_before = mass_fixture(shape, 6500.0)
    _, M_after = mass_fixture(shape, 6501.0)

    before = Q_of(build, M_before)[n:2 * n, n:2 * n]
    after = Q_of(build, M_after)[n:2 * n, n:2 * n]

    assert np.max(np.abs(after - before)) > 1e-6 * float(np.max(np.abs(before)))
    expected = PARAMS.dt * ref_qa(build, M_after)
    assert_all_close(after, expected, rel_tol(expected), "qdqd at new configuration")


# ---------------------------------------------------------------------------
# 9. testPredictRefreshesQAndKeepsCovarianceSymmetricPSD
# ---------------------------------------------------------------------------

def test_predict_refreshes_q_and_keeps_covariance_symmetric_psd():
    """`P^- = F P F^T + Q` stays symmetric PSD with a seeded SPD prior.

    Deviation: `predict()` lives in `jointKF/predict.py` (agent A2), so `F` is
    written out here in three lines from its closed form `I + A dt`. That keeps
    the test self-contained AND independent -- the point being asserted is that
    the `Q` this module produces cannot break the covariance, not that A2's
    predict is correct.
    """
    shape = SHAPES[0]
    n, m, dim = shape_dims(shape)
    build, M = mass_fixture(shape, 6600.0)

    Q = Q_of(build, M)
    expected = PARAMS.dt * ref_qa(build, M)
    assert_all_close(Q[n:2 * n, n:2 * n], expected, rel_tol(expected), "qdqd")

    F = np.eye(dim)
    F[np.arange(n), n + np.arange(n)] = PARAMS.dt
    P = F @ spd(dim, 42.0) @ F.T + Q

    assert_symmetric(P, 1e-9 * max(1.0, float(np.max(np.abs(P)))), "P^-")
    assert_positive_semidefinite(P, "P^-")


# ---------------------------------------------------------------------------
# 10. testMassMatrixAndScalarPathsDiffer
# ---------------------------------------------------------------------------

def test_mass_matrix_and_scalar_paths_differ():
    """Scalar `qdqd[0,0] == sigma_accel^2 * dt` exactly; the two paths differ materially."""
    shape = SHAPES[0]
    n, _, _ = shape_dims(shape)
    sa2dt = PARAMS.sigma_accel ** 2 * PARAMS.dt

    build, M = mass_fixture(shape, 6700.0)
    mass_qdqd = Q_of(build, M)[n:2 * n, n:2 * n]
    scalar_qdqd = np.asarray(
        process.build_process_noise(stub_build(shape, use_mass_matrix=False), PARAMS, M=None)
    )[n:2 * n, n:2 * n]

    assert abs(scalar_qdqd[0, 0] - sa2dt) <= 1e-12
    assert np.max(np.abs(mass_qdqd - scalar_qdqd)) > 1e-3 * sa2dt
