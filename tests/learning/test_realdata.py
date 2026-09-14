import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.learning.realdata import (
    accel_body_from_raw, contact_chol_heuristic, estimate_static_accel_bias,
)


def test_firm_and_swing_probabilities_reproduce_the_named_endpoints():
    L = contact_chol_heuristic(jnp.array([1.0, 0.0]), firm_variance=1e-6, swing_variance=4.0)
    assert L.shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(L[0]), np.sqrt(1e-6) * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.asarray(L[1]), np.sqrt(4.0) * np.eye(3), atol=1e-12)


def test_intermediate_probability_interpolates_linearly_in_variance():
    L = contact_chol_heuristic(jnp.array([0.5]), firm_variance=1.0, swing_variance=3.0)
    expected_variance = 3.0 + 0.5 * (1.0 - 3.0)  # = 2.0
    np.testing.assert_allclose(np.asarray(L[0]), np.sqrt(expected_variance) * np.eye(3), atol=1e-12)


def test_probability_outside_unit_interval_is_clipped_not_extrapolated():
    L_over = contact_chol_heuristic(jnp.array([1.5]), firm_variance=1e-6, swing_variance=4.0)
    L_under = contact_chol_heuristic(jnp.array([-0.5]), firm_variance=1e-6, swing_variance=4.0)
    np.testing.assert_allclose(np.asarray(L_over[0]), np.sqrt(1e-6) * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.asarray(L_under[0]), np.sqrt(4.0) * np.eye(3), atol=1e-12)


def test_rejects_a_firm_variance_that_is_not_strictly_smaller_than_swing():
    with pytest.raises(ValueError, match="firm_variance"):
        contact_chol_heuristic(jnp.array([1.0]), firm_variance=4.0, swing_variance=1e-6)


def test_reconstructed_covariance_matches_the_requested_variance_through_the_real_digest():
    from invariant_estimation.inEKF.contact import reconstruct_cov
    L = contact_chol_heuristic(jnp.array([1.0, 0.0]), firm_variance=1e-6, swing_variance=4.0)
    Sigma = reconstruct_cov(L)
    np.testing.assert_allclose(np.asarray(Sigma[0]), 1e-6 * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.asarray(Sigma[1]), 4.0 * np.eye(3), atol=1e-9)


def test_static_bias_estimate_recovers_a_known_offset_from_noisy_stationary_samples():
    rng = np.random.default_rng(0)
    true_bias = np.array([0.02, -0.01, 0.05])
    gravity_body_at_rest = np.array([0.0, 0.0, -9.81])
    samples = gravity_body_at_rest + true_bias + rng.normal(0, 1e-4, size=(2000, 3))
    bias = estimate_static_accel_bias(jnp.array(samples), jnp.array(gravity_body_at_rest))
    np.testing.assert_allclose(np.asarray(bias), true_bias, atol=5e-4)


def test_static_bias_rejects_an_empty_or_malformed_sample_window():
    with pytest.raises(ValueError, match="shape"):
        estimate_static_accel_bias(jnp.zeros((10, 2)), jnp.zeros(3))
    with pytest.raises(ValueError, match="at least one"):
        estimate_static_accel_bias(jnp.zeros((0, 3)), jnp.zeros(3))


def test_accel_body_correction_is_a_plain_subtraction_usable_inside_a_scan():
    raw = jnp.array([1.0, 2.0, 3.0])
    bias = jnp.array([0.1, 0.1, 0.1])
    np.testing.assert_allclose(np.asarray(accel_body_from_raw(raw, bias)), [0.9, 1.9, 2.9])
