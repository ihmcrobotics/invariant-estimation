"""Tests for joint_kf/measurement.py — stacked z / H (encoder + IMU fusion).

Locks in the design-§3 measurement model:
    z = [q̃ ; z_ω],  H = [[I_n, 0, 0], [0, J_stack, I_3m]],  z_ω = ω_b − R^b_a ω_a.
"""
import jax
import jax.numpy as jnp
import pytest

from invariant_estimation.jointKF import measurement as meas
from invariant_estimation.jointKF import noise
from invariant_estimation.jointKF.state import JointKFState, default_params


# (n_joints, n_pairs), including the encoder-only edge case m = 0.
SHAPES = [(6, 2), (1, 1), (4, 0), (12, 3)]


def _J_omega(n, m, seed=0):
    """A deterministic (m, 3, n) stacked relative-gyro Jacobian."""
    return jnp.sin(jnp.arange(m * 3 * n, dtype=float) + seed).reshape(m, 3, n)


def _rot_z(theta):
    """Rotation matrix about z by theta."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# build_H
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_build_H_structure(n, m):
    J = _J_omega(n, m)
    H = meas.build_H(J, n)
    assert H.shape == (n + 3 * m, 2 * n + 3 * m)
    # Encoder rows.
    assert jnp.array_equal(H[:n, :n], jnp.eye(n))
    assert jnp.array_equal(H[:n, n:], jnp.zeros((n, n + 3 * m)))
    if m > 0:
        # IMU rows: no position dependence, J on q̇, identity on bias.
        assert jnp.array_equal(H[n:, :n], jnp.zeros((3 * m, n)))
        assert jnp.allclose(H[n:, n:2 * n], J.reshape(3 * m, n))
        assert jnp.array_equal(H[n:, 2 * n:], jnp.eye(3 * m))


def test_build_H_zero_jacobian_is_pure_bias():
    """J_omega = 0 ⇒ IMU rows reduce to [0 0 I_3m] (bias-only observability)."""
    n, m = 4, 2
    H = meas.build_H(jnp.zeros((m, 3, n)), n)
    assert jnp.array_equal(H[n:, n:2 * n], jnp.zeros((3 * m, n)))
    assert jnp.array_equal(H[n:, 2 * n:], jnp.eye(3 * m))


# ---------------------------------------------------------------------------
# relative_gyro_measurement
# ---------------------------------------------------------------------------

def test_diff_identity_rotation():
    m = 3
    wa = jnp.arange(1.0, 3 * m + 1.0).reshape(m, 3)
    wb = 2.0 * wa
    R = jnp.broadcast_to(jnp.eye(3), (m, 3, 3))
    z = meas.relative_gyro_measurement(wa, wb, R)
    assert jnp.allclose(z, wb - wa)


def test_diff_known_rotation():
    # 90° about z sends a's reading (1,0,0) -> (0,1,0) in b's frame.
    R = _rot_z(jnp.pi / 2)[None]            # (1,3,3)
    wa = jnp.array([[1.0, 0.0, 0.0]])
    wb = jnp.array([[0.0, 0.0, 0.0]])
    z = meas.relative_gyro_measurement(wa, wb, R)
    assert jnp.allclose(z, jnp.array([[0.0, -1.0, 0.0]]), atol=1e-6)


def test_diff_per_pair_independent():
    # Each pair uses its own rotation; pair 0 identity, pair 1 = 90° about z.
    R = jnp.stack([jnp.eye(3), _rot_z(jnp.pi / 2)])     # (2,3,3)
    wa = jnp.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
    wb = jnp.zeros((2, 3))
    z = meas.relative_gyro_measurement(wa, wb, R)
    assert jnp.allclose(z[0], -wa[0], atol=1e-6)
    assert jnp.allclose(z[1], jnp.array([0.0, -1.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# build_z
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_build_z_shape_and_order(n, m):
    q_tilde = jnp.arange(1.0, n + 1.0)
    z_omega = jnp.arange(1.0, 3 * m + 1.0).reshape(m, 3) if m else jnp.zeros((0, 3))
    z = meas.build_z(q_tilde, z_omega)
    assert z.shape == (n + 3 * m,)
    assert jnp.array_equal(z[:n], q_tilde)
    assert jnp.array_equal(z[n:], z_omega.reshape(-1))


# ---------------------------------------------------------------------------
# predicted measurement  H x  and innovation consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_predicted_measurement_is_fusion(n, m):
    q = jnp.arange(1.0, n + 1.0)
    qd = jnp.arange(1.0, n + 1.0) + 50.0
    b = jnp.arange(1.0, 3 * m + 1.0) + 500.0
    state = JointKFState(q_hat=q, q_dot_hat=qd, b_omega=b,
                         P=jnp.eye(2 * n + 3 * m))
    J = _J_omega(n, m, seed=4)
    H = meas.build_H(J, n)
    hx = H @ state.x
    assert jnp.allclose(hx[:n], q)                       # encoder: predicts q
    if m > 0:
        assert jnp.allclose(hx[n:], J.reshape(3 * m, n) @ qd + b)  # J q̇ + b_ω


@pytest.mark.parametrize("n, m", SHAPES)
def test_innovation_shape_matches_R(n, m):
    p = default_params()
    q_tilde = jnp.zeros(n)
    omega_a = jnp.zeros((m, 3))
    omega_b = jnp.zeros((m, 3))
    R_ba = jnp.broadcast_to(jnp.eye(3), (m, 3, 3))
    J = _J_omega(n, m)
    z, H = meas.build_measurement(q_tilde, omega_a, omega_b, R_ba, J, n)
    x = jnp.zeros(2 * n + 3 * m)
    nu = z - H @ x
    R = noise.build_R(p, n, m)
    assert nu.shape == (n + 3 * m,)
    assert R.shape == (n + 3 * m, n + 3 * m)


# ---------------------------------------------------------------------------
# jit + robot seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_jit_matches_eager(n, m):
    q_tilde = jnp.arange(1.0, n + 1.0)
    omega_a = jnp.ones((m, 3))
    omega_b = 2.0 * jnp.ones((m, 3))
    R_ba = jnp.broadcast_to(jnp.eye(3), (m, 3, 3))
    J = _J_omega(n, m, seed=9)

    H_jit = jax.jit(meas.build_H, static_argnums=1)(J, n)
    assert jnp.allclose(H_jit, meas.build_H(J, n))

    bm = jax.jit(meas.build_measurement, static_argnums=5)
    z_j, H_j = bm(q_tilde, omega_a, omega_b, R_ba, J, n)
    z_e, H_e = meas.build_measurement(q_tilde, omega_a, omega_b, R_ba, J, n)
    assert jnp.allclose(z_j, z_e)
    assert jnp.allclose(H_j, H_e)


def test_robot_seam_relative_gyro_jacobian():
    from invariant_estimation import robot

    class Dummy:
        def mass_matrix(self, q):
            return jnp.eye(q.shape[0])

        def relative_gyro_jacobian(self, q):
            return jnp.zeros((2, 3, q.shape[0]))

    assert isinstance(Dummy(), robot.RobotModel)
