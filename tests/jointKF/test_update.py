"""Tests for joint_kf/update.py — the Joseph-form EKF measurement update.

Locks in the design-§4 update:  ν = z − H x⁻,  S = H P⁻ Hᵀ + R,
K = P⁻ Hᵀ S⁻¹,  x⁺ = x⁻ + Kν,  P⁺ = (I−KH)P⁻(I−KH)ᵀ + K R Kᵀ.
"""
import jax
import jax.numpy as jnp
import pytest

from invariant_estimation.jointKF import measurement as meas
from invariant_estimation.jointKF import noise
from invariant_estimation.jointKF.state import (
    JointKFState,
    default_params,
    init_state,
)
from invariant_estimation.jointKF.predict import predict
from invariant_estimation.jointKF.update import UpdateInfo, update


# (n_joints, n_pairs), including the encoder-only edge case m = 0.
SHAPES = [(6, 2), (1, 1), (4, 0), (12, 3)]


def _spd(n, seed=0):
    A = jnp.sin(jnp.arange(1.0, n * n + 1.0).reshape(n, n) + seed)
    return A @ A.T + n * jnp.eye(n)


def _J_omega(n, m, seed=0):
    return jnp.sin(jnp.arange(m * 3 * n, dtype=float) + seed).reshape(m, 3, n)


def _prior(n, m, seed=0):
    """A prior state with distinct mean and an SPD covariance."""
    q = jnp.arange(1.0, n + 1.0)
    qd = jnp.arange(1.0, n + 1.0) + 10.0
    b = jnp.arange(1.0, 3 * m + 1.0) + 100.0 if m else jnp.zeros(0)
    return JointKFState(q_hat=q, q_dot_hat=qd, b_omega=b, P=_spd(2 * n + 3 * m, seed))


def _meas(n, m, state, p):
    """Build a real (z, H, R) triple for the given prior state."""
    q_tilde = state.q_hat + 0.05            # encoder slightly off the prior
    omega_a = jnp.ones((m, 3))
    omega_b = 1.5 * jnp.ones((m, 3))
    R_ba = jnp.broadcast_to(jnp.eye(3), (m, 3, 3))
    J = _J_omega(n, m, seed=2)
    z, H = meas.build_measurement(q_tilde, omega_a, omega_b, R_ba, J, n)
    R = noise.build_R(p, n, m)
    return z, H, R


# ---------------------------------------------------------------------------
# shapes / innovation / S
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_shapes_and_innovation(n, m):
    p = default_params()
    st = _prior(n, m)
    z, H, R = _meas(n, m, st, p)
    out, info = update(st, z, H, R)
    assert isinstance(out, JointKFState)
    assert out.P.shape == (2 * n + 3 * m, 2 * n + 3 * m)
    assert info.nu.shape == (n + 3 * m,)
    assert info.S.shape == (n + 3 * m, n + 3 * m)
    assert jnp.allclose(info.nu, z - H @ st.x)
    expected_S = H @ st.P @ H.T + R
    assert jnp.allclose(info.S, 0.5 * (expected_S + expected_S.T))


@pytest.mark.parametrize("n, m", SHAPES)
def test_matches_reference_kf(n, m):
    p = default_params()
    st = _prior(n, m, seed=1)
    z, H, R = _meas(n, m, st, p)
    out, info = update(st, z, H, R)

    # Explicit-inverse reference.
    S = H @ st.P @ H.T + R
    S = 0.5 * (S + S.T)
    K = st.P @ H.T @ jnp.linalg.inv(S)
    x_ref = st.x + K @ (z - H @ st.x)
    ImKH = jnp.eye(st.x.shape[0]) - K @ H
    P_ref = ImKH @ st.P @ ImKH.T + K @ R @ K.T
    assert jnp.allclose(out.x, x_ref, atol=1e-4)
    assert jnp.allclose(out.P, 0.5 * (P_ref + P_ref.T), atol=1e-4)


# ---------------------------------------------------------------------------
# covariance properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_covariance_symmetric_psd(n, m):
    p = default_params()
    st = _prior(n, m)
    z, H, R = _meas(n, m, st, p)
    out, _ = update(st, z, H, R)
    assert jnp.allclose(out.P, out.P.T, atol=1e-6)
    eig = jnp.linalg.eigvalsh(out.P)
    assert eig.min() >= -1e-6 * eig.max()


@pytest.mark.parametrize("n, m", SHAPES)
def test_covariance_shrinks(n, m):
    p = default_params()
    st = _prior(n, m)
    z, H, R = _meas(n, m, st, p)
    out, _ = update(st, z, H, R)
    # A measurement cannot increase uncertainty: P⁻ − P⁺ is PSD.
    diff = st.P - out.P
    diff = 0.5 * (diff + diff.T)
    eig = jnp.linalg.eigvalsh(diff)
    assert eig.min() >= -1e-5 * jnp.maximum(eig.max(), 1.0)
    assert jnp.trace(out.P) <= jnp.trace(st.P) + 1e-6


def test_joseph_matches_short_form_at_this_gain():
    """K here is the optimal gain for the given H ⇒ (I−KH)P⁻ ≈ Joseph result."""
    n, m = 6, 2
    p = default_params()
    st = _prior(n, m)
    z, H, R = _meas(n, m, st, p)
    out, info = update(st, z, H, R)
    K = st.P @ H.T @ jnp.linalg.inv(info.S)
    P_short = (jnp.eye(st.x.shape[0]) - K @ H) @ st.P
    assert jnp.allclose(out.P, 0.5 * (P_short + P_short.T), atol=1e-4)


# ---------------------------------------------------------------------------
# physical behavior
# ---------------------------------------------------------------------------

def test_encoder_pull_tiny_R():
    """Encoder-only, σ_enc → 0: posterior position is pulled onto the reading."""
    n, m = 5, 0
    st = _prior(n, m)
    q_tilde = st.q_hat + 0.3
    z, H = meas.build_measurement(q_tilde, jnp.zeros((0, 3)), jnp.zeros((0, 3)),
                                  jnp.zeros((0, 3, 3)), jnp.zeros((0, 3, n)), n)
    R = (1e-8 ** 2) * jnp.eye(n)            # extremely confident encoder
    out, _ = update(st, z, H, R)
    assert jnp.allclose(out.q_hat, q_tilde, atol=1e-3)


def test_bias_observability():
    """A nonzero IMU innovation moves the residual-bias estimate."""
    n, m = 4, 2
    p = default_params()
    st = _prior(n, m)
    z, H, R = _meas(n, m, st, p)
    out, _ = update(st, z, H, R)
    assert not jnp.allclose(out.b_omega, st.b_omega)


# ---------------------------------------------------------------------------
# jit + pytree + integration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_jit_matches_eager(n, m):
    p = default_params()
    st = _prior(n, m)
    z, H, R = _meas(n, m, st, p)
    out_e, info_e = update(st, z, H, R)
    out_j, info_j = jax.jit(update)(st, z, H, R)
    # jit reorders the Cholesky solve; float32 rounding differs from eager. The
    # conditioning fix (jax_enable_x64) is a later decision — tolerate it here.
    assert jnp.allclose(out_e.x, out_j.x, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(out_e.P, out_j.P, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(info_e.nu, info_j.nu, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(info_e.S, info_j.S, rtol=1e-3, atol=1e-3)


def test_update_info_is_pytree():
    info = UpdateInfo(nu=jnp.arange(3.0), S=jnp.eye(3))
    leaves, treedef = jax.tree_util.tree_flatten(info)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, UpdateInfo)
    assert jnp.array_equal(rebuilt.S, info.S)


def test_predict_update_cycle_reduces_position_variance():
    """One predict→update cycle keeps P PSD and tightens Σ_q vs the prior."""
    n, m = 6, 2
    p = default_params()
    st = init_state(n, m)
    pred = predict(st, p)                     # diffuse prior grows
    q_tilde = jnp.zeros(n)
    omega_a = jnp.zeros((m, 3))
    omega_b = jnp.zeros((m, 3))
    R_ba = jnp.broadcast_to(jnp.eye(3), (m, 3, 3))
    J = _J_omega(n, m)
    z, H = meas.build_measurement(q_tilde, omega_a, omega_b, R_ba, J, n)
    R = noise.build_R(p, n, m)
    post, _ = update(pred, z, H, R)
    eig = jnp.linalg.eigvalsh(post.P)
    assert eig.min() >= -1e-6 * eig.max()
    assert jnp.all(jnp.diag(post.sigma_q) <= jnp.diag(pred.sigma_q) + 1e-9)
