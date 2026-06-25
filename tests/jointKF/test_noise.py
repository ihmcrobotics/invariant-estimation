"""Tests for joint_kf/noise.py — F, Q_d, R builders.

Locks in the design-§2/§3 closed forms and the mass-matrix injection seam
(diagonal early-dev fallback vs σ_τ² M⁻² mass path).
"""
import jax.numpy as jnp
import pytest

from invariant_estimation.jointKF import noise
from invariant_estimation.jointKF.state import default_params


# (n_joints, n_pairs), including the encoder-only edge case m = 0.
SHAPES = [(6, 2), (1, 1), (4, 0), (12, 3)]


def _spd(n, seed=0):
    """A random symmetric positive-definite (n, n) matrix."""
    key = jnp.arange(1.0, n * n + 1.0).reshape(n, n) + seed
    A = jnp.sin(key)
    return A @ A.T + n * jnp.eye(n)


# ---------------------------------------------------------------------------
# build_F
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_build_F_structure_and_exactness(n, m):
    dt = 1e-3
    F = noise.build_F(n, m, dt)
    dim = 2 * n + 3 * m
    assert F.shape == (dim, dim)
    # Diagonal blocks are identity.
    assert jnp.array_equal(F[:n, :n], jnp.eye(n))
    assert jnp.array_equal(F[n:2 * n, n:2 * n], jnp.eye(n))
    assert jnp.array_equal(F[2 * n:, 2 * n:], jnp.eye(3 * m))
    # Off-diagonal q->q̇ coupling is exactly dt·I.
    assert jnp.allclose(F[:n, n:2 * n], dt * jnp.eye(n))
    # Everything below the block-diagonal is zero (q̇->q, bias couplings).
    assert jnp.array_equal(F[n:2 * n, :n], jnp.zeros((n, n)))
    assert jnp.array_equal(F[2 * n:, :2 * n], jnp.zeros((3 * m, 2 * n)))
    assert jnp.array_equal(F[:2 * n, 2 * n:], jnp.zeros((2 * n, 3 * m)))


def test_build_F_equals_I_plus_A_dt():
    n, m, dt = 3, 1, 0.01
    F = noise.build_F(n, m, dt)
    dim = 2 * n + 3 * m
    A = jnp.zeros((dim, dim)).at[:n, n:2 * n].set(jnp.eye(n))
    assert jnp.allclose(F, jnp.eye(dim) + A * dt)


# ---------------------------------------------------------------------------
# acceleration covariance
# ---------------------------------------------------------------------------

def test_acceleration_cov_diag():
    Q_a = noise.acceleration_cov_diag(5, sigma_acc=2.0)
    assert jnp.allclose(Q_a, 4.0 * jnp.eye(5))


def test_acceleration_cov_mass_identity():
    Q_a = noise.acceleration_cov_mass(jnp.eye(4), sigma_tau=3.0)
    assert jnp.allclose(Q_a, 9.0 * jnp.eye(4))


def test_acceleration_cov_mass_diagonal():
    d = jnp.array([1.0, 2.0, 4.0])
    M = jnp.diag(d)
    Q_a = noise.acceleration_cov_mass(M, sigma_tau=1.0)
    assert jnp.allclose(Q_a, jnp.diag(1.0 / d ** 2))  # σ_τ² M⁻²


def test_acceleration_cov_mass_couples_and_psd():
    M = _spd(4)
    Q_a = noise.acceleration_cov_mass(M, sigma_tau=1.5)
    # Symmetric, PSD.
    assert jnp.allclose(Q_a, Q_a.T)
    assert jnp.all(jnp.linalg.eigvalsh(Q_a) >= -1e-9)
    # Dense coupling: non-diagonal M produces off-diagonal acceleration cov.
    off = Q_a - jnp.diag(jnp.diag(Q_a))
    assert jnp.any(jnp.abs(off) > 1e-6)
    # Matches the analytic σ_τ² M⁻².
    Minv = jnp.linalg.inv(M)
    assert jnp.allclose(Q_a, 1.5 ** 2 * Minv @ Minv)


# ---------------------------------------------------------------------------
# build_Q_d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_build_Q_d_blocks(n, m):
    p = default_params(dt=2e-3)
    dt = p.dt
    Q_a = _spd(n)
    Q_d = noise.build_Q_d(Q_a, p, m)
    dim = 2 * n + 3 * m
    assert Q_d.shape == (dim, dim)
    # Van Loan closed-form sub-blocks.
    assert jnp.allclose(Q_d[:n, :n], (dt ** 3 / 3.0) * Q_a)
    assert jnp.allclose(Q_d[:n, n:2 * n], (dt ** 2 / 2.0) * Q_a)
    assert jnp.allclose(Q_d[n:2 * n, :n], (dt ** 2 / 2.0) * Q_a)
    assert jnp.allclose(Q_d[n:2 * n, n:2 * n], dt * Q_a)
    # Bias block.
    if m > 0:
        bb = Q_d[2 * n:, 2 * n:]
        assert jnp.allclose(bb, p.sigma_b ** 2 * dt * jnp.eye(3 * m))
        # No cross-coupling between joint and bias blocks.
        assert jnp.array_equal(Q_d[:2 * n, 2 * n:], jnp.zeros((2 * n, 3 * m)))


@pytest.mark.parametrize("n, m", SHAPES)
def test_build_Q_d_symmetric_psd(n, m):
    p = default_params()
    Q_d = noise.build_Q_d(_spd(n), p, m)
    assert jnp.allclose(Q_d, Q_d.T)
    # The double-integrator block is PD but ill-conditioned (eigenvalues ~Δt³),
    # so use a scale-relative tolerance for the float32 PSD check.
    eig = jnp.linalg.eigvalsh(Q_d)
    assert eig.min() >= -1e-6 * eig.max()


# ---------------------------------------------------------------------------
# build_process_noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_build_process_noise_diagonal_path(n, m):
    p = default_params()
    got = noise.build_process_noise(p, n, m, M=None)
    expected = noise.build_Q_d(noise.acceleration_cov_diag(n, p.sigma_acc), p, m)
    assert jnp.allclose(got, expected)


@pytest.mark.parametrize("n, m", SHAPES)
def test_build_process_noise_mass_path(n, m):
    p = default_params()
    M = _spd(n)
    got = noise.build_process_noise(p, n, m, M=M)
    expected = noise.build_Q_d(noise.acceleration_cov_mass(M, p.sigma_tau), p, m)
    assert jnp.allclose(got, expected)


# ---------------------------------------------------------------------------
# build_R
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_build_R(n, m):
    p = default_params()
    R = noise.build_R(p, n, m)
    dim = n + 3 * m
    assert R.shape == (dim, dim)
    assert jnp.allclose(R, R.T)
    # Encoder block.
    assert jnp.allclose(R[:n, :n], p.sigma_enc ** 2 * jnp.eye(n))
    if m > 0:
        # Gyro block + no cross terms.
        assert jnp.allclose(R[n:, n:], p.sigma_omega ** 2 * jnp.eye(3 * m))
        assert jnp.array_equal(R[:n, n:], jnp.zeros((n, 3 * m)))
    # PSD.
    assert jnp.all(jnp.linalg.eigvalsh(R) >= 0.0)


def test_robot_seam_imports_without_simulator():
    """robot.py defines the RobotModel Protocol with no IsaacLab dependency."""
    from invariant_estimation import robot
    assert hasattr(robot, "RobotModel")

    class Dummy:
        def mass_matrix(self, q):
            return jnp.eye(q.shape[0])

        def relative_gyro_jacobian(self, q):
            return jnp.zeros((1, 3, q.shape[0]))

    # runtime_checkable Protocol: a duck-typed provider satisfies it.
    assert isinstance(Dummy(), robot.RobotModel)
