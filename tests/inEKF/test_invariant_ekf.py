"""1:1 port of ``InvariantEKFTest.java`` (TEST_SUITE_MAP.md §invariant_estimator
core tests) onto `inEKF.ekf`.

Java class constants preserved verbatim: ``CONTACTS = 1``, ``GROUP_SIZE = 6``,
``TANGENT_SIZE = 12``; ``GYRO_VARIANCE = 1e-4``, ``ACCEL_VARIANCE = 1e-3``,
``CONTACT_VARIANCE = 1e-6``; seeds 11 / 22 / 33 / 44; delegation tolerance 1e-12.

The orchestrator is pure wiring, so most of this class is delegation: `predict`
and `update` must reproduce the standalone propagator and updater bit-for-bit.
In the port that holds by construction — `ekf.predict` *calls*
`propagate.propagate` — so these tests are really guarding against a future
refactor that inlines or "optimises" one of the two paths apart from the other.
"""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import group as gr
from invariant_estimation.inEKF import state as s
from invariant_estimation.inEKF.propagate import propagate

co = importlib.import_module("invariant_estimation.inEKF.correct")
ekf_mod = importlib.import_module("invariant_estimation.inEKF.ekf")

from ._oracles import next_rotation_matrix, next_vector3d

CONTACTS = 1
GROUP_SIZE = 6
TANGENT_SIZE = 12
GYRO_VARIANCE = 1.0e-4
ACCEL_VARIANCE = 1.0e-3
CONTACT_VARIANCE = 1.0e-6
DT = 1.0e-3


# ---------------------------------------------------------------------------
# Java helper/oracle methods
# ---------------------------------------------------------------------------

def _create(n=CONTACTS):
    return ekf_mod.create(n, GYRO_VARIANCE, ACCEL_VARIANCE, CONTACT_VARIANCE, dt=DT)


def _random_state(rng, n=CONTACTS, P=None):
    return s.InEKFState(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(next_vector3d(rng)[0]),
        p=jnp.asarray(next_vector3d(rng)[0]),
        d=jnp.asarray(next_vector3d(rng, n)),
        P=jnp.eye(9 + 3 * n) if P is None else P,
    )


def _initialize_random(ekf, rng):
    """Java ``initializeRandom``: random components, covariance = I."""
    source = _random_state(rng, ekf.N)
    return ekf_mod.initialize_from_state(ekf, source, jnp.eye(ekf.tangent_size))


def _forward_kinematics_from_truth(truth, i=0):
    return truth.R.T @ (truth.d[i] - truth.p)


def _random_error(rng, scale):
    return jnp.asarray(rng.uniform(-scale, scale, size=TANGENT_SIZE))


def _perturb(truth, xi):
    X = gr.exp_SEk3(xi) @ truth.as_matrix
    return truth._replace(R=X[0:3, 0:3], v=X[0:3, 3], p=X[0:3, 4], d=X[0:3, 5:].T)


def _world_residual_norm(state, i, measurement):
    return float(jnp.linalg.norm(co.contact_residual(state, i, measurement)))


def _assert_states_equal(a, b, tol=1.0e-12):
    """Java ``assertStatesEqual``: X and P element-wise."""
    assert np.max(np.abs(np.asarray(a.as_matrix) - np.asarray(b.as_matrix))) < tol
    assert np.max(np.abs(np.asarray(a.P) - np.asarray(b.P))) < tol


def _body_scaled_identity(scale):
    return jnp.eye(3) * scale


# ---------------------------------------------------------------------------
# testCreateWiresConsistentSizes
# ---------------------------------------------------------------------------

def test_create_wires_consistent_sizes():
    """Sizes follow N, and `update` works without any separate wiring step."""
    ekf = _create(2)
    state = ekf_mod.initialize(
        ekf, jnp.eye(3), jnp.zeros(3), jnp.zeros(3),
        [jnp.zeros(3), jnp.zeros(3)], jnp.eye(15),
    )

    assert ekf.number_of_contacts == 2
    assert ekf.group_size == 7
    assert state.group_size == 7
    assert ekf.tangent_size == 15
    assert state.tangent_size == 15

    # Must not raise: the contact updater is wired by construction (see
    # PORT_NOTES.md — Java's IllegalStateException has no port analogue).
    updated, _ = ekf_mod.update(
        ekf, state, 0, jnp.array([0.1, 0.0, -0.5]), _body_scaled_identity(1.0e-6)
    )
    assert updated.P.shape == (15, 15)


# ---------------------------------------------------------------------------
# testInitializeSetsEstimate
# ---------------------------------------------------------------------------

def test_initialize_sets_estimate():
    rng = np.random.default_rng(11)
    ekf = _create()
    R = jnp.asarray(next_rotation_matrix(rng))
    v = jnp.asarray(next_vector3d(rng)[0])
    p = jnp.asarray(next_vector3d(rng)[0])
    contacts = jnp.asarray(next_vector3d(rng, CONTACTS))
    covariance = jnp.eye(TANGENT_SIZE) * 2.5

    state = ekf_mod.initialize(ekf, R, v, p, contacts, covariance)

    assert jnp.allclose(state.R, R, atol=1.0e-12)
    assert jnp.allclose(state.v, v, atol=1.0e-12)
    assert jnp.allclose(state.p, p, atol=1.0e-12)
    assert jnp.allclose(state.d, contacts, atol=1.0e-12)
    assert jnp.allclose(state.P, covariance, atol=1.0e-12)


# ---------------------------------------------------------------------------
# testInitializeRejectsWrongContactCount
# ---------------------------------------------------------------------------

def test_initialize_rejects_wrong_contact_count():
    ekf = _create()
    with pytest.raises(ValueError):
        ekf_mod.initialize(
            ekf, jnp.eye(3), jnp.zeros(3), jnp.zeros(3), [], jnp.eye(TANGENT_SIZE)
        )


# ---------------------------------------------------------------------------
# testInitializeRejectsWrongCovarianceSize
# ---------------------------------------------------------------------------

def test_initialize_rejects_wrong_covariance_size():
    ekf = _create()
    with pytest.raises(ValueError):
        ekf_mod.initialize(
            ekf, jnp.eye(3), jnp.zeros(3), jnp.zeros(3), [jnp.zeros(3)], jnp.eye(3)
        )


# ---------------------------------------------------------------------------
# testPredictDelegatesToPropagator
# ---------------------------------------------------------------------------

def test_predict_delegates_to_propagator():
    """EKF.predict must equal the standalone propagator bit-for-bit (1e-12)."""
    rng = np.random.default_rng(22)
    ekf = _create()
    state = _initialize_random(ekf, rng)
    reference = state                                   # pytrees are immutable

    angular_velocity = jnp.asarray(next_vector3d(rng)[0])
    linear_acceleration = jnp.asarray(next_vector3d(rng)[0])

    via_ekf = ekf_mod.predict(ekf, state, angular_velocity, linear_acceleration)
    via_propagator = propagate(
        reference, angular_velocity, linear_acceleration, ekf.sigma_c, ekf.params
    )

    _assert_states_equal(via_propagator, via_ekf, 1.0e-12)


# ---------------------------------------------------------------------------
# testUpdateDelegatesToUpdater
# ---------------------------------------------------------------------------

def test_update_delegates_to_updater():
    """EKF.update must equal the standalone updater + contact updater (1e-12)."""
    rng = np.random.default_rng(33)
    ekf = _create()
    state = _initialize_random(ekf, rng)
    reference = state

    measurement = jnp.asarray(next_vector3d(rng)[0])
    body_covariance = _body_scaled_identity(1.0e-6)

    via_ekf, _ = ekf_mod.update(ekf, state, 0, measurement, body_covariance)
    via_updater, _, _ = co.contact_update(
        reference, 0, measurement, body_covariance, learned=False
    )

    _assert_states_equal(via_updater, via_ekf, 1.0e-12)


# ---------------------------------------------------------------------------
# testPredictThenUpdateReducesError
# ---------------------------------------------------------------------------

def test_predict_then_update_reduces_error():
    """The assembled predict -> update loop reduces estimation error."""
    rng = np.random.default_rng(44)
    truth = _random_state(rng)
    estimate = _perturb(truth, _random_error(rng, 1.0e-3))

    ekf = _create()
    state = ekf_mod.initialize_from_state(ekf, estimate, jnp.eye(TANGENT_SIZE))

    angular_velocity = jnp.asarray(next_vector3d(rng)[0])
    linear_acceleration = jnp.asarray(next_vector3d(rng)[0])

    # Propagate truth and estimate with identical inputs.
    truth = propagate(
        truth, angular_velocity, linear_acceleration, ekf.sigma_c, ekf.params
    )
    state = ekf_mod.predict(ekf, state, angular_velocity, linear_acceleration)

    # Measurement is taken from the truth AFTER it moved.
    measurement = _forward_kinematics_from_truth(truth)
    before = _world_residual_norm(state, 0, measurement)

    state, _ = ekf_mod.update(
        ekf, state, 0, measurement, _body_scaled_identity(1.0e-10)
    )
    after = _world_residual_norm(state, 0, measurement)

    assert after < before


# ---------------------------------------------------------------------------
# Port-specific — the introspection surface and the reseed TODO
# ---------------------------------------------------------------------------

def test_update_publishes_diagnostics():
    """Java's introspection getters become the returned diagnostics pytree (§4)."""
    rng = np.random.default_rng(55)
    ekf = _create()
    state = _initialize_random(ekf, rng)

    _, diagnostics = ekf_mod.update(
        ekf, state, 0, jnp.asarray(next_vector3d(rng)[0]), _body_scaled_identity(1e-6)
    )

    # wasLastUpdateApplied / getLastNormalizedInnovationSquared /
    # getLastConditionProxy / getLastCorrectionRotationNorm
    assert float(diagnostics.applied) == 1.0
    assert np.isfinite(float(diagnostics.nis))
    assert np.isfinite(float(diagnostics.condition_proxy))
    assert float(diagnostics.correction_rotation_norm) >= 0.0

    # …and before any update, NIS is NaN rather than a plausible in-band value.
    assert np.isnan(float(ekf_mod.initial_diagnostics().nis))


def test_reseed_is_wired_but_off_by_default():
    """Re-seed exists (`inEKF/reseed.py`) and is **disabled** in the shipped config.

    Replaces the old `test_reseed_is_not_implemented`, which guarded the
    2026-07-21 deferral. The deferral is over, but the shipped default must not
    move silently: every recorded gate number was measured with the re-seed off,
    so flipping `reseed.enabled` has to be a deliberate config edit.
    """
    ekf = ekf_mod.create(2)
    assert ekf.reseed.enabled is False
    assert ekf.reseed.trigger == 0.5
    assert ekf.reseed.rearm == 0.1
    assert ekf.reseed.dwell_ticks == 100
