"""1:1 port of ``ContactUpdaterTest.java`` (TEST_SUITE_MAP.md §invariant_estimator
contact / support tests) onto the `inEKF.correct` ContactUpdater seams.

Java class constants preserved verbatim: ``CONTACTS = 1``, ``GROUP_SIZE = 6``,
``TANGENT_SIZE = 12``, ``POSITION_BLOCK = 6``, ``CONTACT_BLOCK = 9``, and the
per-test seeds 101 / 202 / 303 / 404 / 505 / 606 / 707 / 808 / 909.

Measurement model: body-frame contact position ``y = R̂ᵀ(d − p)``; the
right-invariant world residual is ``r = R̂ y − (d̂ − p̂)``.

Java's RNG cannot be reproduced in NumPy, but every oracle here is recomputed
from the same draw (FK, residual, covariance rotation are all self-consistent),
so the port is tolerance-based exactly as the map prescribes.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import importlib

from invariant_estimation.inEKF import group as gr
from invariant_estimation.inEKF import state as s

# `inEKF.__init__` re-exports the function `correct`, shadowing the module of the
# same name — import the module explicitly.
co = importlib.import_module("invariant_estimation.inEKF.correct")

from ._oracles import assert_symmetric, next_rotation_matrix, next_vector3d

CONTACTS = 1
GROUP_SIZE = 6
TANGENT_SIZE = 12
POSITION_BLOCK = 6
CONTACT_BLOCK = 9


# ---------------------------------------------------------------------------
# Java helper/oracle methods
# ---------------------------------------------------------------------------

def _random_state(rng, P=None):
    """Java ``randomState``: InvariantState(1), random rotation/vel/pos/contact."""
    return s.InEKFState(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(next_vector3d(rng)[0]),
        p=jnp.asarray(next_vector3d(rng)[0]),
        d=jnp.asarray(next_vector3d(rng, CONTACTS)),
        P=jnp.eye(TANGENT_SIZE) if P is None else P,
    )


def _forward_kinematics_from_truth(truth, i=0):
    """Java ``forwardKinematicsFromTruth``: y = Rᵀ(d − p), exact and noise-free."""
    return truth.R.T @ (truth.d[i] - truth.p)


def _random_error(rng, scale):
    """Java ``randomError``: 12-vector, each component uniform in ±scale."""
    return jnp.asarray(rng.uniform(-scale, scale, size=TANGENT_SIZE))


def _perturb(truth, xi):
    """Java ``perturb``: estimate = exp(ξ) · truth (left multiplication)."""
    X = gr.exp_SEk3(xi) @ truth.as_matrix
    return truth._replace(R=X[0:3, 0:3], v=X[0:3, 3], p=X[0:3, 4], d=X[0:3, 5:].T)


def _world_residual_norm(estimate, i, measurement):
    """Java ``worldResidualNorm``: ‖R̂ y − (d̂ − p̂)‖."""
    return float(jnp.linalg.norm(co.contact_residual(estimate, i, measurement)))


def _symmetric_matrix(rng, size):
    """Java ``symmetricMatrix``: ½(A + Aᵀ) for A random in ±1."""
    A = rng.uniform(-1.0, 1.0, size=(size, size))
    return jnp.asarray(0.5 * (A + A.T))


# ---------------------------------------------------------------------------
# testContactUpdateDrivesResidualToZero
# ---------------------------------------------------------------------------

def test_contact_update_drives_residual_to_zero():
    rng = np.random.default_rng(101)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 1.0e-4))
    body_cov = jnp.eye(3) * 1.0e-10

    before = _world_residual_norm(estimate, 0, measurement)
    updated, _, _ = co.contact_update(estimate, 0, measurement, body_cov, False)
    after = _world_residual_norm(updated, 0, measurement)

    assert after < 1.0e-6
    assert after < before


# ---------------------------------------------------------------------------
# testContactUpdateReducesResidualForLargerError
# ---------------------------------------------------------------------------

def test_contact_update_reduces_residual_for_larger_error():
    """>10x reduction at moderate error — rules out a sign flip."""
    rng = np.random.default_rng(202)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 0.05))
    body_cov = jnp.eye(3) * 1.0e-8

    before = _world_residual_norm(estimate, 0, measurement)
    updated, _, _ = co.contact_update(estimate, 0, measurement, body_cov, False)
    after = _world_residual_norm(updated, 0, measurement)

    assert after < 0.1 * before


# ---------------------------------------------------------------------------
# testCovarianceShrinksAndStaysSymmetric
# ---------------------------------------------------------------------------

def test_covariance_shrinks_and_stays_symmetric():
    rng = np.random.default_rng(303)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 0.02))
    trace_before = float(jnp.trace(estimate.P))

    updated, _, _ = co.contact_update(estimate, 0, measurement, jnp.eye(3) * 1.0e-4, False)

    assert_symmetric(updated.P, 1.0e-9)
    assert float(jnp.trace(updated.P)) < trace_before


# ---------------------------------------------------------------------------
# testJacobianStructureAndStateIndependence
# ---------------------------------------------------------------------------

def test_jacobian_structure_and_state_independence():
    """H = [0(3x6) | +I | −I], bit-identical across two different states."""
    rng = np.random.default_rng(404)
    _random_state(rng)                      # draw and discard: two different states
    _random_state(rng)

    H1 = co.contact_jacobian(CONTACTS, 0)
    H2 = co.contact_jacobian(CONTACTS, 0)

    assert H1.shape == (3, TANGENT_SIZE)
    # (1) State independence — exact equality (Java asserts tol 0.0).
    assert jnp.array_equal(H1, H2)
    # (2) Structure — element-wise, exact.
    for r in range(3):
        for c in range(TANGENT_SIZE):
            if c == POSITION_BLOCK + r:
                expected = 1.0
            elif c == CONTACT_BLOCK + r:
                expected = -1.0
            else:
                expected = 0.0
            assert float(H1[r, c]) == expected


# ---------------------------------------------------------------------------
# testResidualFormula
# ---------------------------------------------------------------------------

def test_residual_formula():
    """r = R̂·y − (d̂ − p̂), against an independently computed oracle."""
    rng = np.random.default_rng(505)
    state = _random_state(rng)
    measurement = jnp.asarray(next_vector3d(rng)[0])    # arbitrary, not FK-consistent

    actual = co.contact_residual(state, 0, measurement)
    expected = (np.asarray(state.R) @ np.asarray(measurement)
                - (np.asarray(state.d[0]) - np.asarray(state.p)))

    assert actual.shape == (3,)
    assert np.max(np.abs(np.asarray(actual) - expected)) < 1.0e-12


# ---------------------------------------------------------------------------
# testMeasurementCovarianceIsRotatedToWorld
# ---------------------------------------------------------------------------

def test_measurement_covariance_is_rotated_to_world():
    rng = np.random.default_rng(606)
    state = _random_state(rng)
    body_cov = _symmetric_matrix(rng, 3)

    actual = co.rotate_measurement_covariance(state, body_cov)
    R = np.asarray(state.R)
    expected = R @ np.asarray(body_cov) @ R.T

    assert np.max(np.abs(np.asarray(actual) - expected)) < 1.0e-12


# ---------------------------------------------------------------------------
# testMapEncoderNoise
# ---------------------------------------------------------------------------

def test_map_encoder_noise():
    """N = J Σ Jᵀ (§4.2 routing of the joint-KF covariance)."""
    rng = np.random.default_rng(707)
    n_joints = 6
    contact_jac = jnp.asarray(rng.uniform(-1.0, 1.0, size=(3, n_joints)))
    joint_cov = _symmetric_matrix(rng, n_joints)

    actual = co.map_encoder_noise(contact_jac, joint_cov)
    expected = np.asarray(contact_jac) @ np.asarray(joint_cov) @ np.asarray(contact_jac).T

    assert actual.shape == (3, 3)
    assert np.max(np.abs(np.asarray(actual) - expected)) < 1.0e-12


# ---------------------------------------------------------------------------
# testUpdateWithoutContactUpdaterThrows  (adapted — see PORT_NOTES.md)
# ---------------------------------------------------------------------------

def test_update_on_nonexistent_contact_raises():
    """Java asserts IllegalStateException when no ContactUpdater was installed.

    The port has no installable collaborator — `contact_update` is a free
    function, so that state is unreachable. The analogous "there is no contact
    updater for this measurement" failure is an out-of-range contact index,
    which raises `IndexError` (matching `InvariantStateTest`'s bounds contract).
    """
    rng = np.random.default_rng(808)
    state = _random_state(rng)
    measurement = jnp.asarray(next_vector3d(rng)[0])
    body_cov = jnp.eye(3) * 1.0e-6

    with pytest.raises(IndexError):
        co.contact_update(state, CONTACTS, measurement, body_cov, False)
    with pytest.raises(IndexError):
        co.contact_update(state, -1, measurement, body_cov, False)


# ---------------------------------------------------------------------------
# testLearnedModuleNotImplementedThrows
# ---------------------------------------------------------------------------

def test_learned_module_not_implemented_raises():
    """The ContactNet branch must raise until Lucas lands the module (§7)."""
    rng = np.random.default_rng(909)
    state = _random_state(rng)
    measurement = jnp.asarray(next_vector3d(rng)[0])
    body_cov = jnp.eye(3) * 1.0e-6

    with pytest.raises(NotImplementedError):
        co.contact_update(state, 0, measurement, body_cov, True)


# ---------------------------------------------------------------------------
# Port-specific oracle (CLAUDE.md G4: "programmatic-H-from-b ≡ Table I closed form")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 4])
def test_single_contact_jacobians_stack_into_the_precomputed_H(N):
    """Row-stacking the per-contact H_i must reproduce `state.build_H` exactly.

    Ties the ContactUpdater seam to the vectorised hot path: the two are the
    same operator, so a sign or block-offset drift in either is caught here.
    """
    stacked = jnp.concatenate([co.contact_jacobian(N, i) for i in range(N)], axis=0)
    assert jnp.array_equal(stacked, s.build_H(N))


def test_contact_update_matches_vectorised_correct_at_one_contact():
    """`contact_update` ≡ `correct` when N = 1 — one filter, two entry points."""
    rng = np.random.default_rng(1234)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 0.01))
    body_cov = _symmetric_matrix(rng, 3) @ _symmetric_matrix(rng, 3).T + jnp.eye(3)

    params = s.default_params(CONTACTS, dt=1e-3)
    # `correct` consumes world-frame noise; `contact_update` rotates it itself.
    world_cov = co.rotate_measurement_covariance(estimate, body_cov)

    via_seam, _, _ = co.contact_update(estimate, 0, measurement, body_cov, False)
    via_correct, _ = co.correct(estimate, measurement[None, :], world_cov[None], params)

    assert jnp.allclose(via_seam.as_matrix, via_correct.as_matrix, atol=1e-12)
    assert jnp.allclose(via_seam.P, via_correct.P, atol=1e-12)
