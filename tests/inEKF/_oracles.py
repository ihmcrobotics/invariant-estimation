"""Independent oracles shared by the ported invariant_estimator test classes.

Everything here stands in for a Java dependency the port cannot use
(``EuclidCoreRandomTools``, ``SE3LieGroupTools``, ``SO3LieGroupTools``,
``MatrixFeatures_DDRM.isSymmetric``).

**Deliberately written in NumPy from the closed forms, never by calling
`invariant_estimation.inEKF.group`** — an oracle that delegates to the code under
test proves nothing.
"""
import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``(w)_×``."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def so3_exp_reference(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential (Rodrigues) — stands in for ``SO3LieGroupTools.exp``."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < 1e-12:
        return np.eye(3) + K
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / theta**2) * (K @ K))


def so3_left_jacobian_reference(phi: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian ``V(φ)`` — the translation carrier of the SE(3) exp."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < 1e-12:
        return np.eye(3) + 0.5 * K
    return (np.eye(3)
            + ((1.0 - np.cos(theta)) / theta**2) * K
            + ((theta - np.sin(theta)) / theta**3) * (K @ K))


def se3_exp_reference(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential — stands in for ``SE3LieGroupTools.exp``.

    Rotation-first ordering ``ξ = [φ ; v]``; ``t = V(φ) v``.
    """
    xi = np.asarray(xi, dtype=float)
    T = np.eye(4)
    T[0:3, 0:3] = so3_exp_reference(xi[:3])
    T[0:3, 3] = so3_left_jacobian_reference(xi[:3]) @ xi[3:6]
    return T


def se3_adjoint_reference(T: np.ndarray) -> np.ndarray:
    """SE(3) adjoint under rotation-first ordering: ``[[R, 0], [(t)_× R, R]]``."""
    R, t = T[0:3, 0:3], T[0:3, 3]
    Ad = np.zeros((6, 6))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = hat(t) @ R
    Ad[3:6, 3:6] = R
    return Ad


def so3_step_integrals(omega: np.ndarray, dt: float, order: int = 40):
    """The two rotating-accelerometer integrals over one step, by quadrature.

    Returns ``(I1, I2)`` with::

        I1 = ∫₀^dt exp((ω)_× s) ds              (once-integrated  → Γ_1 · dt)
        I2 = ∫₀^dt (dt - u) exp((ω)_× u) du     (twice-integrated → Γ_2 · dt²)

    Evaluated by Gauss-Legendre quadrature on the reference Rodrigues formula —
    machine-precision for a smooth integrand over a short step, and completely
    independent of `group.Gamma1` / `group.Gamma2`, which is the point: it is the
    oracle for the exact mean integration under *simultaneous* rotation and
    acceleration, the case the Java propagator suite never exercises.
    """
    omega = np.asarray(omega, dtype=float)
    x, w = np.polynomial.legendre.leggauss(order)
    s = 0.5 * dt * (x + 1.0)
    weights = 0.5 * dt * w
    R_s = np.stack([so3_exp_reference(omega * si) for si in s])
    I1 = np.einsum("i,ijk->jk", weights, R_s)
    I2 = np.einsum("i,ijk->jk", weights * (dt - s), R_s)
    return I1, I2


def next_rotation_vector(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Euclid ``nextRotationVector``: random axis, angle ~ U(-π, π). ``(size, 3)``.

    The π bound keeps ``exp`` inside the injectivity radius so ``log`` inverts it.
    """
    axis = rng.normal(size=(size, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis * rng.uniform(-np.pi, np.pi, size=(size, 1))


def next_vector3d(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Euclid ``nextVector3D``: components ~ U(-1, 1). ``(size, 3)``."""
    return rng.uniform(-1.0, 1.0, size=(size, 3))


def next_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Euclid ``nextRotationMatrix``, via the reference Rodrigues formula."""
    return so3_exp_reference(next_rotation_vector(rng)[0])


def random_algebra_vectors(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """Java ``randomAlgebraVector``, batched: ``(n, 3+3k)``.

    Slots 0:3 a bounded rotation vector, then ``k`` blocks of ``nextVector3D``.
    """
    phi = next_rotation_vector(rng, n)
    rho = rng.uniform(-1.0, 1.0, size=(n, 3 * k))
    return np.concatenate([phi, rho], axis=1)


def assert_symmetric(M, epsilon: float) -> None:
    """Java ``assertSymmetric``: ``M(r,c) == M(c,r)`` for every upper-triangle pair."""
    M = np.asarray(M)
    assert np.max(np.abs(M - M.T)) < epsilon, (
        f"matrix not symmetric to {epsilon}: max |M - Mᵀ| = {np.max(np.abs(M - M.T))}"
    )


def yaw_pitch_roll_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """IHMC ``yawPitchRoll`` → rotation matrix: ``R = R_z(yaw) R_y(pitch) R_x(roll)``."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def matrix_to_yaw_pitch_roll(R) -> tuple:
    """Inverse of `yaw_pitch_roll_to_matrix` (Z-Y-X Euler extraction)."""
    R = np.asarray(R)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def assert_symmetric_psd(P, epsilon: float = 1.0e-9) -> None:
    """Java ``assertSymmetricPSD``: symmetric to eps, and Cholesky of P + 1e-12·I works."""
    P = np.asarray(P)
    assert_symmetric(P, epsilon)
    np.linalg.cholesky(0.5 * (P + P.T) + 1e-12 * np.eye(P.shape[0]))
