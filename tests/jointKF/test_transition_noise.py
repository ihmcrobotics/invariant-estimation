"""Port of `JointLevelKFTransitionNoiseTest.java` (6 tests).

Locks in the **scalar-CWNA fallback** process-noise path (`M = None`, no robot
model): `Qa = sigma_accel^2 I_n`, Van-Loan discretised, plus the untouched bias
block. The transition matrix `F` and the encoder measurement model are asserted
in the same Java class but live in other modules of this port
(`jointKF/predict.py`, `jointKF/measure.py`, owned by agents A2/B1); those three
tests are written here against the real seams and **skip cleanly until those
modules land**, so the class is ported whole rather than silently truncated.

Constants: `DT = 1e-3`, `SIGMA_ACCEL = 50.0`, `ENCODER_VAR = 5e-5`,
`IMU_BIAS_PROCESS_VAR = 1e-4` -- all read from `config/filter_cfg.yaml` via
`default_params()`, never hard-coded (CONTRACT_CARD §3).
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
    scaled_identity,
    shape_dims,
    stub_build,
)

PARAMS = default_params()


def _scalar_build(shape):
    """A build on the fallback path: `use_mass_matrix=False`, so `M=None` is legal."""
    return stub_build(shape, use_mass_matrix=False)


# ---------------------------------------------------------------------------
# testBuildFStructureAndExactness / testBuildFEqualsIPlusADt
#   -> jointKF/predict.py::build_transition (agent A2). Skipped until it lands.
# ---------------------------------------------------------------------------

def _build_transition():
    predict = pytest.importorskip(
        "invariant_estimation.jointKF.predict",
        reason="jointKF/predict.py is owned by agent A2; F tests activate when it lands",
    )
    return predict.build_transition


def test_build_f_structure_and_exactness():
    """`F = [[I, dt I, 0], [0, I, 0], [0, 0, I]]` -- block structure, tol 1e-12."""
    build_transition = _build_transition()
    for shape in SHAPES:
        n, m, dim = shape_dims(shape)
        F = np.asarray(build_transition(_scalar_build(shape), PARAMS))
        assert F.shape == (dim, dim)
        assert_all_close(F[:n, :n], np.eye(n), 1e-12, "F qq")
        assert_all_close(F[n:2 * n, n:2 * n], np.eye(n), 1e-12, "F qdqd")
        assert_all_close(F[2 * n:, 2 * n:], np.eye(3 * m), 1e-12, "F bias")
        assert_all_close(F[:n, n:2 * n], PARAMS.dt * np.eye(n), 1e-12, "F q<-qd")
        assert_all_close(F[n:2 * n, :n], np.zeros((n, n)), 1e-12, "F qd<-q")
        assert_all_close(F[:2 * n, 2 * n:], np.zeros((2 * n, 3 * m)), 1e-12, "F joint<-bias")
        assert_all_close(F[2 * n:, :2 * n], np.zeros((3 * m, 2 * n)), 1e-12, "F bias<-joint")


def test_build_f_equals_i_plus_a_dt():
    """`F = I + A dt` exactly -- `A` is nilpotent on `(q, qd)`, so no truncation."""
    build_transition = _build_transition()
    shape = SHAPES[2]                       # n=3, m=2 -- the Java `singlePair(1001L, 6, 1, 4)`
    n, _, dim = shape_dims(shape)
    expected = np.eye(dim)
    expected[np.arange(n), n + np.arange(n)] = PARAMS.dt
    F = np.asarray(build_transition(_scalar_build(shape), PARAMS))
    assert_all_close(F, expected, 1e-12, "F = I + A dt")


# ---------------------------------------------------------------------------
# testProcessNoiseVanLoanJointBlocks
# ---------------------------------------------------------------------------

def test_process_noise_van_loan_joint_blocks():
    """Fallback `Qa = sigma_accel^2 I`, Van-Loan discretised, tol 1e-9.

    The three factors `dt^3/3`, `dt^2/2`, `dt` are the *exact* closed form of the
    Van-Loan integral for a double integrator, not a first-order approximation:
    `A` is nilpotent on the `(q, qd)` block so the series terminates.
    """
    sa2 = PARAMS.sigma_accel ** 2
    for shape in SHAPES:
        n, m, dim = shape_dims(shape)
        build = _scalar_build(shape)
        Q = np.asarray(process.build_process_noise(build, PARAMS, M=None))
        assert Q.shape == (dim, dim)

        assert_all_close(Q[:n, :n], (PARAMS.dt ** 3 / 3.0) * sa2 * np.eye(n), 1e-9, "Q qq")
        assert_all_close(Q[:n, n:2 * n], (PARAMS.dt ** 2 / 2.0) * sa2 * np.eye(n), 1e-9, "Q q-qd")
        assert_all_close(Q[n:2 * n, :n], (PARAMS.dt ** 2 / 2.0) * sa2 * np.eye(n), 1e-9, "Q qd-q")
        assert_all_close(Q[n:2 * n, n:2 * n], PARAMS.dt * sa2 * np.eye(n), 1e-9, "Q qdqd")

        qa = np.asarray(process.acceleration_covariance(build, PARAMS, M=None))
        assert qa[0, 0] > 0.0


# ---------------------------------------------------------------------------
# testProcessNoiseBiasBlockAndNoCrossCoupling
# ---------------------------------------------------------------------------

def test_process_noise_bias_block_and_no_cross_coupling():
    """Bias block `= dt * imu_bias_process_var * I_3m`; joint<->bias cross EXACTLY zero.

    The zeros are structural (invariant I1): joint torque does not drive the gyro
    bias random walk, so a nonzero cross block would let the bias estimator
    absorb joint modelling error.
    """
    for shape in SHAPES:
        n, m, _ = shape_dims(shape)
        m_dof = 3 * m
        Q = np.asarray(process.build_process_noise(_scalar_build(shape), PARAMS, M=None))

        assert_all_close(
            Q[2 * n:, 2 * n:],
            scaled_identity(m_dof, PARAMS.dt * PARAMS.imu_bias_process_var),
            1e-12,
            "Q bias",
        )
        assert_all_close(Q[:2 * n, 2 * n:], np.zeros((2 * n, m_dof)), 1e-12, "Q joint-bias")
        assert_all_close(Q[2 * n:, :2 * n], np.zeros((m_dof, 2 * n)), 1e-12, "Q bias-joint")


# ---------------------------------------------------------------------------
# testProcessNoiseSymmetricPSD
# ---------------------------------------------------------------------------

def test_process_noise_symmetric_psd():
    """`Q` symmetric to 1e-12 and PSD on the fallback path."""
    for shape in SHAPES:
        Q = np.asarray(process.build_process_noise(_scalar_build(shape), PARAMS, M=None))
        assert_symmetric(Q, 1e-12, f"Q {shape['name']}")
        assert_positive_semidefinite(Q, f"Q {shape['name']}")


# ---------------------------------------------------------------------------
# testEncoderMeasurementModel
#   -> jointKF/measure.py (agent B1). Skipped until it lands.
# ---------------------------------------------------------------------------

def test_encoder_measurement_model():
    """`H_enc = [I_n | 0]` (position only) and `R_enc = encoder_var * I_n`."""
    measure = pytest.importorskip(
        "invariant_estimation.jointKF.measure",
        reason="jointKF/measure.py is owned by agent B1; encoder test activates when it lands",
    )
    for shape in SHAPES:
        n, _, dim = shape_dims(shape)
        build = _scalar_build(shape)
        H = np.asarray(measure.encoder_jacobian(build, PARAMS))
        R = np.asarray(measure.encoder_noise(build, PARAMS))
        assert H.shape == (n, dim)
        assert_all_close(H[:, :n], np.eye(n), 1e-12, "H_enc position")
        assert_all_close(H[:, n:], np.zeros((n, dim - n)), 1e-12, "H_enc non-position")
        assert_all_close(R, scaled_identity(n, PARAMS.encoder_var), 1e-12, "R_enc")
