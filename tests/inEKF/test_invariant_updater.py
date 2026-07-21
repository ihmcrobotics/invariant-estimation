"""1:1 port of ``InvariantUpdaterTest.java`` (TEST_SUITE_MAP.md §invariant_estimator
core tests) onto `inEKF.correct.linear_update`.

Java class constants preserved verbatim: ``CONTACTS = 1``, ``GROUP_SIZE = 6``,
``TANGENT_SIZE = 12``, ``POSITION_BLOCK = 6``, ``CONTACT_BLOCK = 9``; seeds
101 / 202 / 303 / 404 / 505; ``samples = 4000`` for the NIS mean.

This exercises the **generic** update — `update(state, H, residual, R)` — with a
synthetic single-contact model, as distinct from `ContactUpdaterTest` which
drives the contact-specialised entry point. Both go through the same
`linear_update`, by construction.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import group as gr
from invariant_estimation.inEKF import state as s

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
    return s.InEKFState(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(next_vector3d(rng)[0]),
        p=jnp.asarray(next_vector3d(rng)[0]),
        d=jnp.asarray(next_vector3d(rng, CONTACTS)),
        P=jnp.eye(TANGENT_SIZE) if P is None else P,
    )


def _build_contact_jacobian():
    """Java ``buildContactJacobian``: 3x12 with H(r, 6+r)=+1, H(r, 9+r)=−1."""
    H = np.zeros((3, TANGENT_SIZE))
    for r in range(3):
        H[r, POSITION_BLOCK + r] = 1.0
        H[r, CONTACT_BLOCK + r] = -1.0
    return jnp.asarray(H)


def _forward_kinematics_from_truth(truth, i=0):
    """y = Rᵀ(d − p) — the body-frame contact measurement."""
    return truth.R.T @ (truth.d[i] - truth.p)


def _random_error(rng, scale):
    return jnp.asarray(rng.uniform(-scale, scale, size=TANGENT_SIZE))


def _perturb(truth, xi):
    X = gr.exp_SEk3(xi) @ truth.as_matrix
    return truth._replace(R=X[0:3, 0:3], v=X[0:3, 3], p=X[0:3, 4], d=X[0:3, 5:].T)


def _compute_residual(estimate, y, i=0):
    """Java ``computeResidual``: r = R̂·y − (d̂ − p̂); returns (residual, norm)."""
    r = estimate.R @ y - (estimate.d[i] - estimate.p)
    return r, float(jnp.linalg.norm(r))


def _quadratic_form_nis(H, P, R, r):
    """Java ``quadraticFormNIS``: rᵀ(HPHᵀ+R)⁻¹r, via an explicit inverse.

    Deliberately uses `np.linalg.inv` — an independent reference for the port's
    Cholesky-solve NIS, exactly as Java uses EJML `invert` here.
    """
    H, P, R, r = (np.asarray(a) for a in (H, P, R, r))
    S = H @ P @ H.T + R
    return float(r @ np.linalg.inv(S) @ r)


def _scaled_identity(size, scale):
    return jnp.eye(size) * scale


# ---------------------------------------------------------------------------
# testUpdateDrivesResidualToZero
# ---------------------------------------------------------------------------

def test_update_drives_residual_to_zero():
    rng = np.random.default_rng(101)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 1.0e-4))
    H = _build_contact_jacobian()

    residual, before = _compute_residual(estimate, measurement)
    updated, _ = co.linear_update(estimate, H, residual, _scaled_identity(3, 1.0e-10))
    _, after = _compute_residual(updated, measurement)

    assert after < 1.0e-6
    assert after < before


# ---------------------------------------------------------------------------
# testUpdateReducesResidualForLargerError
# ---------------------------------------------------------------------------

def test_update_reduces_residual_for_larger_error():
    """>10x reduction at moderate error — rules out a sign flip."""
    rng = np.random.default_rng(202)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 0.05))
    H = _build_contact_jacobian()

    residual, before = _compute_residual(estimate, measurement)
    updated, _ = co.linear_update(estimate, H, residual, _scaled_identity(3, 1.0e-8))
    _, after = _compute_residual(updated, measurement)

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
    H = _build_contact_jacobian()

    residual, _ = _compute_residual(estimate, measurement)
    updated, _ = co.linear_update(estimate, H, residual, _scaled_identity(3, 1.0e-4))

    assert_symmetric(updated.P, 1.0e-9)
    assert float(jnp.trace(updated.P)) < trace_before


# ---------------------------------------------------------------------------
# testNormalizedInnovationSquaredMatchesQuadraticForm
# ---------------------------------------------------------------------------

def test_normalized_innovation_squared_matches_quadratic_form():
    """NIS = rᵀ(HPHᵀ+R)⁻¹r on the PRIOR P and the PRIOR residual (§6 trap)."""
    rng = np.random.default_rng(404)
    truth = _random_state(rng)
    measurement = _forward_kinematics_from_truth(truth)
    estimate = _perturb(truth, _random_error(rng, 0.03))

    # Non-trivial SPD diagonal prior: P(i,i) = 0.05 + 0.02*i.
    prior = jnp.diag(jnp.array([0.05 + 0.02 * i for i in range(TANGENT_SIZE)]))
    estimate = estimate._replace(P=prior)

    H = _build_contact_jacobian()
    R = _scaled_identity(3, 1.0e-3)
    residual, _ = _compute_residual(estimate, measurement)

    _, diagnostics = co.linear_update(estimate, H, residual, R)

    expected = _quadratic_form_nis(H, prior, R, residual)
    assert float(diagnostics.nis) == pytest.approx(expected, abs=1.0e-9)


# ---------------------------------------------------------------------------
# testNormalizedInnovationSquaredIsNaNBeforeAnyUpdate
# ---------------------------------------------------------------------------

def test_normalized_innovation_squared_is_nan_before_any_update():
    """A never-updated NIS must not read as 'in-band'."""
    assert np.isnan(float(co.no_update_diagnostics().nis))


# ---------------------------------------------------------------------------
# testNormalizedInnovationSquaredAveragesToMeasurementDegreesOfFreedom
# ---------------------------------------------------------------------------

def test_normalized_innovation_squared_averages_to_measurement_dof():
    """Innovation ~ N(0,S) ⇒ NIS ~ χ²(3), mean 3.

    With P = I and H = [+I, −I], S = H P Hᵀ + R = (2 + σ²) I, so the innovation
    std per axis is √(2 + σ²). 4000 samples; the std of the mean is
    √(6/4000) ≈ 0.039, so Java's 0.25 envelope is ~6σ.

    The Java loop over samples becomes a `vmap` — same 4000 draws, no Python
    loop over a data dimension.
    """
    rng = np.random.default_rng(505)
    samples = 4000
    measurement_variance = 1.0e-2
    innovation_std = np.sqrt(2.0 + measurement_variance)

    truth = _random_state(rng)
    H = _build_contact_jacobian()
    R = _scaled_identity(3, measurement_variance)

    # Fresh estimate == truth each sample (exp(0) = identity), P = I.
    residuals = jnp.asarray(innovation_std * rng.standard_normal((samples, 3)))

    def one_sample(residual):
        _, diagnostics = co.linear_update(truth, H, residual, R)
        return diagnostics.nis

    nis = jax.vmap(one_sample)(residuals)

    assert nis.shape == (samples,)
    assert float(jnp.mean(nis)) == pytest.approx(3.0, abs=0.25)


# ---------------------------------------------------------------------------
# Port-specific — the §4 gating contract that G8/G5 rely on
# ---------------------------------------------------------------------------

def test_gated_update_leaves_state_bit_for_bit_unchanged():
    """A gated-out update must be a no-op on (X̂, P), not a latched bad correction.

    CLAUDE.md §4: `K <- gate*K`, so gating is exact rather than approximate.
    This is the mechanism `testSingularInnovationIsSkippedNotLatched` (G8) needs,
    and it is cheaper to pin here than to rediscover there.
    """
    rng = np.random.default_rng(606)
    estimate = _random_state(rng)
    H = _build_contact_jacobian()
    residual = jnp.asarray(next_vector3d(rng)[0])

    updated, diagnostics = co.linear_update(
        estimate, H, residual, _scaled_identity(3, 1.0e-4), gate=0.0
    )

    assert float(diagnostics.applied) == 0.0
    assert jnp.array_equal(updated.as_matrix, estimate.as_matrix)
    assert jnp.array_equal(updated.P, estimate.P)
    # NIS is still reported — the measurement was evaluated, just not applied.
    assert np.isfinite(float(diagnostics.nis))
