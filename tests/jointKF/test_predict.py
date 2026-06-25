"""Tests for joint_kf/predict.py — the EKF time-update.

Locks in the design-§4 predict step:  x⁻ = F x,  P⁻ = F P Fᵀ + Q_d.
"""
import jax
import jax.numpy as jnp
import pytest

from invariant_estimation.jointKF import noise
from invariant_estimation.jointKF.predict import predict
from invariant_estimation.jointKF.state import (
    JointKFState,
    default_params,
    init_state,
    split_x,
)


# (n_joints, n_pairs), including the encoder-only edge case m = 0.
SHAPES = [(6, 2), (1, 1), (4, 0), (12, 3)]


def _spd(n, seed=0):
    """A random symmetric positive-definite (n, n) matrix."""
    A = jnp.sin(jnp.arange(1.0, n * n + 1.0).reshape(n, n) + seed)
    return A @ A.T + n * jnp.eye(n)


def _seeded_state(n, m):
    """A state with distinct, recognizable mean segments and an SPD P."""
    q = jnp.arange(1.0, n + 1.0)
    qd = jnp.arange(1.0, n + 1.0) + 100.0
    b = jnp.arange(1.0, 3 * m + 1.0) + 1000.0
    P = _spd(2 * n + 3 * m)
    return JointKFState(q_hat=q, q_dot_hat=qd, b_omega=b, P=P)


@pytest.mark.parametrize("n, m", SHAPES)
def test_shapes_preserved(n, m):
    p = default_params()
    out = predict(_seeded_state(n, m), p)
    assert isinstance(out, JointKFState)
    assert out.n_joints == n and out.n_pairs == m
    assert out.P.shape == (2 * n + 3 * m, 2 * n + 3 * m)


@pytest.mark.parametrize("n, m", SHAPES)
def test_mean_propagation_exact(n, m):
    p = default_params(dt=1e-2)
    st = _seeded_state(n, m)
    out = predict(st, p)
    assert jnp.allclose(out.q_hat, st.q_hat + p.dt * st.q_dot_hat)
    assert jnp.allclose(out.q_dot_hat, st.q_dot_hat)      # F identity on q̇
    assert jnp.allclose(out.b_omega, st.b_omega)          # F identity on b_ω


@pytest.mark.parametrize("n, m", SHAPES)
def test_covariance_matches_builders(n, m):
    p = default_params()
    st = _seeded_state(n, m)
    out = predict(st, p)
    F = noise.build_F(n, m, p.dt)
    Q_d = noise.build_process_noise(p, n, m, None)
    expected = F @ st.P @ F.T + Q_d
    expected = 0.5 * (expected + expected.T)
    assert jnp.allclose(out.P, expected, atol=1e-5)


@pytest.mark.parametrize("n, m", SHAPES)
def test_covariance_symmetric_psd(n, m):
    p = default_params()
    out = predict(_seeded_state(n, m), p)
    assert jnp.allclose(out.P, out.P.T)
    eig = jnp.linalg.eigvalsh(out.P)
    assert eig.min() >= -1e-6 * eig.max()


@pytest.mark.parametrize("n, m", SHAPES)
def test_mass_path_differs_and_psd(n, m):
    p = default_params()
    st = _seeded_state(n, m)
    M = _spd(n, seed=7)
    out_diag = predict(st, p, M=None)
    out_mass = predict(st, p, M=M)
    # Different Q_d contributions ⇒ different predicted covariance.
    assert not jnp.allclose(out_diag.P, out_mass.P)
    # Both stay symmetric PSD.
    for out in (out_diag, out_mass):
        assert jnp.allclose(out.P, out.P.T)
        eig = jnp.linalg.eigvalsh(out.P)
        assert eig.min() >= -1e-6 * eig.max()


def test_bias_block_growth_diagonal():
    """With M=None and F identity on the bias block, Σ_b grows by exactly Q_d^bb."""
    n, m = 4, 2
    p = default_params()
    st = _seeded_state(n, m)
    out = predict(st, p)
    increment = out.sigma_b - st.sigma_b
    assert jnp.allclose(increment, p.sigma_b ** 2 * p.dt * jnp.eye(3 * m), atol=1e-7)


@pytest.mark.parametrize("n, m", SHAPES)
def test_jit_matches_eager(n, m):
    p = default_params()
    st = _seeded_state(n, m)
    M = _spd(n, seed=3)
    jitted = jax.jit(predict)
    out_eager = predict(st, p, M)
    out_jit = jitted(st, p, M)
    assert jnp.allclose(out_eager.P, out_jit.P)
    assert jnp.allclose(out_eager.x, out_jit.x)


def test_deterministic():
    n, m = 5, 1
    p = default_params()
    st = _seeded_state(n, m)
    a = predict(st, p)
    b = predict(st, p)
    assert jnp.array_equal(a.P, b.P)
    assert jnp.array_equal(a.x, b.x)


@pytest.mark.parametrize("n, m", SHAPES)
def test_split_x_roundtrips(n, m):
    st = _seeded_state(n, m)
    q, qd, b = split_x(st.x, n)
    assert jnp.array_equal(q, st.q_hat)
    assert jnp.array_equal(qd, st.q_dot_hat)
    assert jnp.array_equal(b, st.b_omega)


def test_diffuse_position_variance_grows():
    """A few measurement-free predicts from init_state keep P PSD and grow Σ_q."""
    n, m = 6, 2
    p = default_params()
    st = init_state(n, m)
    prev = jnp.diag(st.sigma_q)
    for _ in range(5):
        st = predict(st, p)
        eig = jnp.linalg.eigvalsh(st.P)
        assert eig.min() >= -1e-6 * eig.max()
        cur = jnp.diag(st.sigma_q)
        assert jnp.all(cur >= prev - 1e-9)
        prev = cur
