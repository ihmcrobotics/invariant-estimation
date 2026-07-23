"""Port of `JointLevelKFStandingStabilityTest.java` (5 tests).

The Java class is a **reconciliation harness** written around the Alex002
hardware finding that joint VELOCITY covariance blew up inside a single
`predict()` while POSITION covariance stayed sane, driven by the Schur process
noise `Qa = sigma_tau^2 Lambda^-2`. Several of its methods only `System.out`
their diagnostics; this port keeps every assertion that states a real property
and drops the printing (noted per test).

The load-bearing content, in order of importance:

1. `testSinglePredictVelocityVarianceInjectionIsPhysicallyBounded` -- the
   regression gate. One `predict()` may not inject more than
   `ONE_TICK_QDD_VARIANCE_BOUND = 1.0 (rad/s)^2` of velocity variance into any
   joint. This port **strengthens** it with an explicitly near-singular fixture
   where the bound is violated without the rotor term and satisfied with it, so
   the assertion constrains the fix rather than merely exercising it
   (`JOINTKF_PORT_PLAN.md` §4, lesson 1).
2. `testRotorInertiaFloorBoundsQaForNearSingularLambda` -- fully self-contained,
   hand-built, and the tightest statement of the Weyl bound in the suite.
3. `testCandidateFixesBoundQa` -- the PSD ordering `Lambda <= M_jj`, i.e. that
   the floating base genuinely inflates the honest process noise.
4. `testAccelerationEqualizedSigmaTauFloorsAtTargetAndEqualizesDominantTerm` --
   the calibration identity behind `alpha_overrides`.

Fixtures come from `test_massmatrix_noise` (synthetic SPD inertia + non-contiguous
DoF split); see that module's docstring for why a synthetic `M` is a valid oracle
for these algebra-level properties.
"""
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  -- enables float64
from invariant_estimation.jointKF import process
from invariant_estimation.jointKF.state import JointKFState, default_params

from ._oracles import (
    SHAPES,
    assert_positive_semidefinite,
    assert_symmetric,
    cond_spd,
    shape_dims,
    symmetric_eigenvalues,
)
from .test_massmatrix_noise import (
    LIGHT_SCALE,
    ROTOR_KNEE,
    mass_fixture,
    near_singular_fixture,
    numpy_sigma_tau,
)

PARAMS = default_params()

#: The Alex002 regression gate: a single 1 kHz tick may not inject more than
#: 1 (rad/s)^2 of velocity variance. Physically, a joint whose velocity
#: uncertainty grows by 1 rad/s in one millisecond is not being filtered.
ONE_TICK_QDD_VARIANCE_BOUND = 1.0

#: Java's "current" diagnostic sigma_tau in `testCandidateFixesBoundQa`.
SIGMA_TAU_CURRENT = 50.0

def _qa(build, M, rotor):
    """`Qa` from a synthetic `M` with an explicit post-Schur rotor add (or none)."""
    lam = process.schur_complement(M, build.dof_joint, build.dof_nuisance)
    lam_eff = process._apply_rotor(lam, rotor)
    return np.asarray(process.qa_from_lambda_eff(lam_eff, numpy_sigma_tau(build)))


def _one_predict_velocity_injection(build, M, rotor):
    """`diag(P^-)_qd - diag(P)_qd` after one `predict()` from `P = 1e-4 I`.

    Uses agent A2's `predict` when present. It is a genuine end-to-end check:
    with `F = I + A dt`, row `n+i` of `F` is `e_{n+i}`, so `F P F^T` leaves the
    velocity diagonal untouched and the entire increment is `dt * Qa[i,i]` --
    which is exactly what makes this a clean measurement of the `Q` this module
    produces, uncontaminated by the prior.
    """
    n, _, dim = build.n_joints, build.n_imus, build.dim
    Q = process.build_process_noise(build, PARAMS, M, rotor=rotor)

    predict_mod = pytest.importorskip(
        "invariant_estimation.jointKF.predict",
        reason="jointKF/predict.py is owned by agent A2",
    )
    F = predict_mod.build_transition(build, PARAMS)
    P0 = 1e-4 * np.eye(dim)
    state = JointKFState(x=np.zeros(dim), P=P0)
    post = predict_mod.predict(state, F, Q)
    P1 = np.asarray(post.P)
    return P1, np.diag(P1)[n:2 * n] - np.diag(P0)[n:2 * n]


# ---------------------------------------------------------------------------
# 1. testSchurConditioningAndPredictInflatesVelocityCovariance
# ---------------------------------------------------------------------------

def test_schur_conditioning_and_predict_inflates_velocity_covariance():
    """One predict's velocity-variance increment at the worst joint `== dt Qa[w,w]`.

    Proves that `predict()`/`Qa` -- not the gyro update -- is the inflation
    source, which is the whole diagnostic point of the Java class. Tolerance
    `1e-4 * max(1, expected)` per the map (loosened there for Gram-vs-LU numerics
    on an ill-conditioned `Lambda_eff`).

    The Java also prints `cond(Lambda)`; `cond_spd` is evaluated here so a
    pathological fixture would surface, but only the increment is asserted.
    """
    for shape in SHAPES:
        build, M = near_singular_fixture(shape, 9100.0)
        qa = _qa(build, M, build.rotor_inertia)
        worst = int(np.argmax(np.diag(qa)))

        lam = np.asarray(process.schur_complement(M, build.dof_joint, build.dof_nuisance))
        assert np.isfinite(cond_spd(lam))

        _, injection = _one_predict_velocity_injection(build, M, build.rotor_inertia)
        expected = PARAMS.dt * qa[worst, worst]
        assert abs(injection[worst] - expected) <= 1e-4 * max(1.0, expected), (
            f"{shape['name']}: injection {injection[worst]:.6e} != dt*Qa[{worst},{worst}] "
            f"= {expected:.6e}"
        )


# ---------------------------------------------------------------------------
# 2. testCandidateFixesBoundQa
# ---------------------------------------------------------------------------

def test_candidate_fixes_bound_qa():
    """Locked-base `Qa` <= Schur `Qa`: the free base makes the noise LARGER.

    `Lambda = M_jj - M_jb M_bb^-1 M_bj <= M_jj` (the subtracted term is a
    congruence of `M_bb^-1 >= 0`), hence `Lambda^-2 >= M_jj^-2`. Dropping the
    recoil term -- the tempting "simplification" -- therefore *understates* the
    process noise, which is the failure direction that makes a filter overconfident.

    Java prints four other candidate fixes (sigma_tau = 5, sigma_tau = 1,
    eigenvalue-floored `Lambda`, locked base) and asserts only this ordering; the
    prints are dropped.
    """
    for shape in SHAPES:
        n, _, _ = shape_dims(shape)
        build, M = mass_fixture(shape, 9200.0)
        sig = np.full(n, SIGMA_TAU_CURRENT)

        lam = process.schur_complement(M, build.dof_joint, build.dof_nuisance)
        curr = np.max(np.diag(np.asarray(process.qa_from_lambda_eff(lam, sig))))

        M_jj = M[np.ix_(np.asarray(build.dof_joint), np.asarray(build.dof_joint))]
        locked = np.max(np.diag(np.asarray(process.qa_from_lambda_eff(M_jj, sig))))

        assert locked <= curr * (1.0 + 1e-6), f"{shape['name']}: {locked:.6e} > {curr:.6e}"
        # the ordering must be STRICT here, else the recoil term did nothing
        assert locked < curr * (1.0 - 1e-6), f"{shape['name']}: recoil term had no effect"


# ---------------------------------------------------------------------------
# 3. testAccelerationEqualizedSigmaTauFloorsAtTargetAndEqualizesDominantTerm
# ---------------------------------------------------------------------------

def test_acceleration_equalized_sigma_tau_floors_at_target_and_equalizes_dominant_term():
    r"""`sigma_i = TARGET / |inv_ii|` equalises the dominant term and floors the STD.

    `sqrt(Qa_ii) = sqrt(sum_j (inv_ij sigma_j)^2) >= |inv_ii| sigma_i = TARGET`,
    with equality only if the off-diagonal inertial coupling vanishes. So the
    achieved unmodeled-acceleration STD is a *floor* at the target, never below
    it -- which is exactly why CLAUDE.md prescribes 2-3 calibration iterations
    rather than a one-shot solve.
    """
    target = PARAMS.target_qdd_std
    for shape in SHAPES:
        build, M = mass_fixture(shape, 9500.0)
        lam = np.asarray(process.schur_complement(M, build.dof_joint, build.dof_nuisance))
        lam_eff = np.asarray(process.lambda_eff(lam, build.rotor_inertia))
        inv = np.linalg.inv(lam_eff)                        # independent of the code under test

        sig_eq = np.asarray(process.equalized_sigma_tau(lam_eff, target))
        std_eq = np.sqrt(np.diag(np.asarray(process.qa_from_lambda_eff(lam_eff, sig_eq))))

        # (a) FLOOR
        assert np.all(std_eq >= target * (1.0 - 1e-6)), f"{shape['name']}: {std_eq}"
        # (b) DOMINANT-TERM EQUALISATION, exact
        assert np.max(np.abs(np.abs(np.diag(inv)) * sig_eq - target)) <= 1e-9 * target
        # sanity: the fixture has nontrivial inertia spread, so this is not vacuous
        d = np.abs(np.diag(inv))
        assert d.max() / d.min() > 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 4. testSinglePredictVelocityVarianceInjectionIsPhysicallyBounded
# ---------------------------------------------------------------------------

def test_single_predict_velocity_variance_injection_is_physically_bounded():
    """THE regression gate: one tick may inject <= 1.0 (rad/s)^2 of velocity variance.

    Strengthened beyond the Java: the fixture carries a deliberately light
    articulated mode (`LIGHT_SCALE`), and the test asserts BOTH directions --
    without the rotor term the bound is blown by orders of magnitude, with it the
    bound holds. A test that only checks the passing side would stay green if
    `lambda_eff` silently became the identity.
    """
    worst_with = 0.0
    worst_without = 0.0
    for shape in SHAPES:
        build, M = near_singular_fixture(shape, 9300.0)

        P1, injection = _one_predict_velocity_injection(build, M, build.rotor_inertia)
        assert_symmetric(P1, 1e-9 * float(np.max(np.abs(P1))) + 1e-12, "P^-")
        assert_positive_semidefinite(P1, "P^-")
        worst_with = max(worst_with, float(injection.max()))

        _, bare = _one_predict_velocity_injection(build, M, process.ROTOR_IN_MASS_MATRIX)
        worst_without = max(worst_without, float(bare.max()))

    assert worst_with <= ONE_TICK_QDD_VARIANCE_BOUND, (
        f"one-tick velocity-variance injection {worst_with:.4e} > {ONE_TICK_QDD_VARIANCE_BOUND}"
    )
    assert worst_without > ONE_TICK_QDD_VARIANCE_BOUND, (
        "fixture is not near-singular enough to constrain the rotor floor "
        f"(bare injection {worst_without:.4e}); the gate would pass without Lambda_eff"
    )


# ---------------------------------------------------------------------------
# 5. testRotorInertiaFloorBoundsQaForNearSingularLambda
# ---------------------------------------------------------------------------

def test_rotor_inertia_floor_bounds_qa_for_near_singular_lambda():
    """Hand-built near-singular `Lambda` -- the Weyl bound, exactly.

    `Lambda = diag(2.0, 0.02, 1.5)` has one nearly-massless articulated mode.
    Unfloored, `max diag(Qa) = sigma_tau^2 / 0.02^2 = 62500`, ~70x over QA_MAX.
    Adding `rotor = 0.05` moves `lambda_min` to `0.07` and Weyl bounds the result
    by `sigma_tau^2 / (lambda_min + rotor)^2` -- a theorem, not a measurement.

    Self-contained: no fixture, no mass matrix, no kinematics.
    """
    lam = np.diag([2.0, 0.02, 1.5])
    sigma_tau = np.full(3, PARAMS.sigma_tau)
    rotor = 0.05

    qa_unfloored = np.asarray(process.qa_from_lambda_eff(lam, sigma_tau))
    lam_eff = np.asarray(process.lambda_eff(lam, np.full(3, rotor)))
    qa_floored = np.asarray(process.qa_from_lambda_eff(lam_eff, sigma_tau))

    lam_min = float(symmetric_eigenvalues(lam).min())
    weyl_bound = PARAMS.sigma_tau ** 2 / (lam_min + rotor) ** 2

    assert np.max(np.diag(qa_unfloored)) > PARAMS.qa_max * 1.5
    assert np.max(np.diag(qa_floored)) <= weyl_bound * (1.0 + 1e-9)
    assert np.max(np.diag(qa_floored)) < np.max(np.diag(qa_unfloored))


# ---------------------------------------------------------------------------
# Tripwire -- the QA_MAX surfacing behaviour these tests exist to protect
# ---------------------------------------------------------------------------

def test_qa_max_tripwire_surfaces_and_never_rescales():
    """The tripwire reports and counts; it must NOT touch `Qa`.

    Not a Java test -- Java's tripwire is a YoVariable counter read off the
    hardware. It is asserted here because the *old* behaviour (uniform rescale to
    bring the outlier under the cap) is the documented cause of the min-side `S`
    singularity, and nothing else in the suite would notice its return.
    """
    lam = np.diag([2.0, 0.02, 1.5])
    sigma_tau = np.full(3, PARAMS.sigma_tau)
    qa = process.qa_from_lambda_eff(lam, sigma_tau)

    diag = process.qa_tripwire(qa, PARAMS)
    assert float(diag.qa_max_diag) > PARAMS.qa_max
    assert int(diag.qa_argmax) == 1
    np.testing.assert_array_equal(np.asarray(diag.qa_tripped), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(diag.qa_trip_count), [0.0, 1.0, 0.0])

    # counter accumulates across ticks, purely functionally
    diag2 = process.qa_tripwire(qa, PARAMS, diag.qa_trip_count)
    np.testing.assert_array_equal(np.asarray(diag2.qa_trip_count), [0.0, 2.0, 0.0])

    # and Qa itself is untouched: the diagnostics path is observation-only
    np.testing.assert_array_equal(
        np.asarray(qa), np.asarray(process.qa_from_lambda_eff(lam, sigma_tau))
    )
