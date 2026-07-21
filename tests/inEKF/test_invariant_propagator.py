"""1:1 port of ``InvariantPropagatorTest.java`` (TEST_SUITE_MAP.md
§invariant_estimator core tests) onto `inEKF.propagate`.

Java constants preserved verbatim: ``EPSILON = 1.0e-10``, ``GRAVITY = -9.81``,
per-test seeds 12345 / 2024, step counts 1000 / 500 / 200 / 50 / 20, and the
per-check tolerance loosenings (1e-9 on the integrated free-fall quantities and
on the composed rotation).

Signature mapping — Java `InvariantPropagator(N, gyroNoise, accelNoise,
contactNoise)` + `predict(state, omega, accel, dt)` becomes
`propagate(state, omega, accel, sigma_c, params)`, with the noise scalars folded
into `InEKFParams` / the per-contact `sigma_c` stack (see `_propagator`).

Sign convention (Java note, load-bearing): ``accel`` is the IMU **specific
force**; gravity is added internally, so at rest ``accel = -g = (0, 0, +9.81)``.

Multi-step scenarios are run through `lax.scan` rather than a Python loop —
identical arithmetic, and it exercises the scan path the filter actually uses.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import group as gr
from invariant_estimation.inEKF import state as s
from invariant_estimation.inEKF.propagate import propagate

from ._oracles import (
    assert_symmetric,
    hat,
    next_rotation_matrix,
    next_vector3d,
    so3_exp_reference,
    so3_step_integrals,
)

EPSILON = 1.0e-10
GRAVITY = -9.81
G_VEC = jnp.array([0.0, 0.0, GRAVITY])


# ---------------------------------------------------------------------------
# Java-constructor shim
# ---------------------------------------------------------------------------

def _propagator(N, gyro_noise, accel_noise, contact_noise, dt):
    """`InvariantPropagator(N, gyroNoise, accelNoise, contactNoise)` + `dt`.

    Returns ``(params, sigma_c)``. Per the repo-wide convention the Java scalars
    are read as **variances**, mapping straight onto `InEKFParams.gyro_var` /
    `accel_var`; the contact scalar becomes an isotropic per-contact variance
    density ``contact_noise · I₃``.
    """
    params = s.default_params(
        N, dt=dt, g=G_VEC,
        gyro_var=gyro_noise, accel_var=accel_noise, contact_floor=0.0,
    )
    sigma_c = jnp.tile(contact_noise * jnp.eye(3), (N, 1, 1))
    return params, sigma_c


def _run(state, omega, accel, sigma_c, params, steps):
    """`steps` predicts with constant inputs; returns the final state."""
    def body(st, _):
        return propagate(st, omega, accel, sigma_c, params), None

    final, _ = jax.lax.scan(body, state, None, length=steps)
    return final


def _state_from_matrix(X, P):
    """`InEKFState` from a dense ``(N+5, N+5)`` group element."""
    return s.InEKFState(R=X[0:3, 0:3], v=X[0:3, 3], p=X[0:3, 4], d=X[0:3, 5:].T, P=P)


# ---------------------------------------------------------------------------
# testFreeFall
# ---------------------------------------------------------------------------

def test_free_fall():
    """v_z = gT, p_z = ½gT² exactly after 1000 steps of dt = 1e-3."""
    dt, steps = 1.0e-3, 1000
    T = dt * steps
    params, sigma_c = _propagator(0, 0.0, 0.0, 0.0, dt)

    final = _run(
        s.InEKFState.identity(0),
        omega=jnp.zeros(3), accel=jnp.zeros(3),
        sigma_c=sigma_c, params=params, steps=steps,
    )

    assert abs(float(final.v[0])) < EPSILON
    assert abs(float(final.v[1])) < EPSILON
    assert float(final.v[2]) == pytest.approx(GRAVITY * T, abs=1.0e-9)
    assert float(final.p[2]) == pytest.approx(0.5 * GRAVITY * T**2, abs=1.0e-9)
    assert jnp.max(jnp.abs(final.R - jnp.eye(3))) < EPSILON


# ---------------------------------------------------------------------------
# testStationaryWithGravityCompensation
# ---------------------------------------------------------------------------

def test_stationary_with_gravity_compensation():
    """Specific force -g holds the base at rest — pins internal g = (0,0,-9.81)."""
    dt, steps = 1.0e-3, 1000
    params, sigma_c = _propagator(0, 0.0, 0.0, 0.0, dt)

    final = _run(
        s.InEKFState.identity(0),
        omega=jnp.zeros(3), accel=jnp.array([0.0, 0.0, -GRAVITY]),
        sigma_c=sigma_c, params=params, steps=steps,
    )

    assert jnp.max(jnp.abs(final.v)) < 1.0e-9
    assert jnp.max(jnp.abs(final.p)) < 1.0e-9


# ---------------------------------------------------------------------------
# testConstantAngularVelocityComposesRotation
# ---------------------------------------------------------------------------

def test_constant_angular_velocity_composes_rotation():
    """Constant-axis increments compose exactly: R_N = exp(ω T)."""
    dt, steps = 1.0e-3, 500
    T = dt * steps
    omega = jnp.array([0.3, -0.2, 0.5])
    params, sigma_c = _propagator(0, 0.0, 0.0, 0.0, dt)

    final = _run(
        s.InEKFState.identity(0),
        omega=omega, accel=jnp.zeros(3),
        sigma_c=sigma_c, params=params, steps=steps,
    )

    expected = so3_exp_reference(np.asarray(omega) * T)
    assert np.max(np.abs(np.asarray(final.R) - expected)) < 1.0e-9


# ---------------------------------------------------------------------------
# testContactsRemainStatic
# ---------------------------------------------------------------------------

def test_contacts_remain_static():
    """Contact columns are world-static under predict, noise notwithstanding."""
    dt = 1.0e-2
    params, sigma_c = _propagator(2, 1.0e-4, 1.0e-2, 1.0e-6, dt)

    d0 = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    state = s.InEKFState.identity(2)._replace(d=d0)

    final = propagate(
        state,
        omega=jnp.array([0.1, 0.2, -0.1]),
        accel=jnp.array([0.0, 0.0, -GRAVITY]),
        sigma_c=sigma_c, params=params,
    )

    assert jnp.max(jnp.abs(final.d - d0)) < EPSILON


# ---------------------------------------------------------------------------
# testCovarianceStaysSymmetric
# ---------------------------------------------------------------------------

def test_covariance_stays_symmetric():
    """P symmetric after every one of 50 steps with random IMU inputs."""
    rng = np.random.default_rng(12345)
    dt, steps, N = 1.0e-2, 50, 2
    params, sigma_c = _propagator(N, 1.0e-3, 1.0e-2, 1.0e-4, dt)

    state = s.InEKFState(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(next_vector3d(rng)[0]),
        p=jnp.asarray(next_vector3d(rng)[0]),
        d=jnp.asarray(next_vector3d(rng, N)),
        P=jnp.eye(9 + 3 * N),                    # start symmetric PSD
    )

    omegas = jnp.asarray(next_vector3d(rng, steps))
    accels = jnp.asarray(next_vector3d(rng, steps))

    def body(st, uk):
        st = propagate(st, uk[0], uk[1], sigma_c, params)
        return st, st.P

    _, P_history = jax.lax.scan(body, state, (omegas, accels))

    assert P_history.shape == (steps, 9 + 3 * N, 9 + 3 * N)
    for P in P_history:                          # every step, as in Java
        assert_symmetric(P, 1.0e-9)


# ---------------------------------------------------------------------------
# testCovarianceGrowsFromZero
# ---------------------------------------------------------------------------

def test_covariance_grows_from_zero():
    """One step from P = 0 leaves P = Q_d: symmetric with positive trace."""
    dt, N = 1.0e-2, 1
    params, sigma_c = _propagator(N, 1.0e-3, 1.0e-2, 1.0e-4, dt)

    final = propagate(
        s.InEKFState.identity(N),
        omega=jnp.zeros(3), accel=jnp.array([0.0, 0.0, -GRAVITY]),
        sigma_c=sigma_c, params=params,
    )

    assert_symmetric(final.P, 1.0e-12)
    assert float(jnp.trace(final.P)) > 0.0


# ---------------------------------------------------------------------------
# testZeroNoiseKeepsCovarianceZero
# ---------------------------------------------------------------------------

def test_zero_noise_keeps_covariance_zero():
    """Q_d = 0 and P₀ = 0 ⇒ Φ·0·Φᵀ + 0 stays exactly zero over 20 steps."""
    dt, steps, N = 1.0e-2, 20, 1
    params, sigma_c = _propagator(N, 0.0, 0.0, 0.0, dt)

    final = _run(
        s.InEKFState.identity(N),
        omega=jnp.zeros(3), accel=jnp.array([0.0, 0.0, -GRAVITY]),
        sigma_c=sigma_c, params=params, steps=steps,
    )

    assert jnp.max(jnp.abs(final.P)) < EPSILON


# ---------------------------------------------------------------------------
# testLogLinearErrorPropagation  — the deepest correctness test
# ---------------------------------------------------------------------------

def _build_error_transition(time: float, N: int) -> np.ndarray:
    """Java ``buildErrorTransition``: Φ(T) = exp(A·T), built analytically.

    Independent NumPy oracle for the exact Φ of CLAUDE.md I3 — deliberately not
    `state.build_Phi`. Blocks, with ``hatGravity = (g)_×``:
        (velocity rows 3:6, rotation cols 0:3) = T · hatGravity
        (position rows 6:9, rotation cols 0:3) = ½T² · hatGravity
        (position rows 6:9, velocity cols 3:6) += T · I
    Contact blocks stay identity — exp(A·T) = I + A·T + ½A²T² exactly, since A
    is nilpotent.
    """
    m = 9 + 3 * N
    Phi = np.eye(m)
    hat_g = hat(np.array([0.0, 0.0, GRAVITY]))
    Phi[3:6, 0:3] = time * hat_g
    Phi[6:9, 0:3] = 0.5 * time**2 * hat_g
    Phi[6:9, 3:6] += time * np.eye(3)
    return Phi


def test_log_linear_error_propagation():
    """η = X̂X⁻¹ propagates *exactly* linearly, even for large ξ₀ (not just 1st order)."""
    rng = np.random.default_rng(2024)
    dt, steps, N = 1.0e-3, 200, 2
    T = dt * steps
    params, sigma_c = _propagator(N, 0.0, 0.0, 0.0, dt)   # no noise

    # Random true state.
    truth = s.InEKFState(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(next_vector3d(rng)[0]),
        p=jnp.asarray(next_vector3d(rng)[0]),
        d=jnp.asarray(next_vector3d(rng, N)),
        P=jnp.zeros((9 + 3 * N, 9 + 3 * N)),
    )

    # LARGE initial error: entries ~ U(-0.5, 0.5). This is the whole point —
    # a first-order-only propagation would fail badly at this magnitude.
    xi0 = jnp.asarray(rng.uniform(-0.5, 0.5, size=9 + 3 * N))
    estimate = _state_from_matrix(
        gr.exp_SEn3(xi0, N) @ truth.as_matrix, truth.P,      # X̂₀ = exp(ξ₀)·X₀
    )

    omega = jnp.array([0.4, -0.3, 0.6])
    accel = jnp.array([0.5, 0.2, -GRAVITY])

    truth_f = _run(truth, omega, accel, sigma_c, params, steps)
    est_f = _run(estimate, omega, accel, sigma_c, params, steps)

    # Measured: ξ_N = log(X̂_N · X_N⁻¹).  Predicted: exp(A·T) ξ₀.
    xi_measured = gr.log_SEn3(est_f.as_matrix @ jnp.linalg.inv(truth_f.as_matrix))
    xi_predicted = _build_error_transition(T, N) @ np.asarray(xi0)

    assert xi_measured.shape == (9 + 3 * N,)
    assert np.max(np.abs(np.asarray(xi_measured) - xi_predicted)) < EPSILON


# ---------------------------------------------------------------------------
# Port-specific — closes a coverage gap in the Java class (see PORT_NOTES.md)
# ---------------------------------------------------------------------------

def test_mean_integration_exact_under_simultaneous_rotation_and_acceleration():
    """Γ_1/Γ_2 are the once/twice integrals of the *rotating* accelerometer signal.

    No test in `InvariantPropagatorTest` constrains this: every mean scenario
    there has either ω = 0 (free fall, stationary) or a = 0 (rotation
    composition), and `testLogLinearErrorPropagation` — which does drive both —
    is blind to it, because log-linearity follows from the propagation being
    group-affine, not from it being accurate, and both trajectories share the
    integrator. A plain Euler mean passes all eight Java tests.

    Oracle: Gauss-Legendre quadrature of the two integrals against the reference
    Rodrigues formula, independent of `Γ_1`/`Γ_2`. Measured discrimination —
    exact integrator ~1e-14, Euler ~2e-4, i.e. 10 orders.
    """
    dt, steps = 1.0e-3, 50
    omega = np.array([0.9, -0.7, 1.3])          # both nonzero — the untested case
    accel = np.array([2.0, -1.5, -GRAVITY])
    g_np = np.array([0.0, 0.0, GRAVITY])
    params, sigma_c = _propagator(0, 0.0, 0.0, 0.0, dt)

    # Reference trajectory, stepped with the quadrature integrals.
    I1, I2 = so3_step_integrals(omega, dt)
    R_ref, v_ref, p_ref = np.eye(3), np.zeros(3), np.zeros(3)
    step_rotation = so3_exp_reference(omega * dt)
    for _ in range(steps):
        p_ref = p_ref + v_ref * dt + R_ref @ (I2 @ accel) + 0.5 * g_np * dt**2
        v_ref = v_ref + R_ref @ (I1 @ accel) + g_np * dt
        R_ref = R_ref @ step_rotation

    final = _run(
        s.InEKFState.identity(0),
        omega=jnp.asarray(omega), accel=jnp.asarray(accel),
        sigma_c=sigma_c, params=params, steps=steps,
    )

    assert np.max(np.abs(np.asarray(final.R) - R_ref)) < EPSILON
    assert np.linalg.norm(np.asarray(final.v) - v_ref) < EPSILON
    assert np.linalg.norm(np.asarray(final.p) - p_ref) < EPSILON
