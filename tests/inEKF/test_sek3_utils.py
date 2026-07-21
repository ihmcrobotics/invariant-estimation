"""1:1 port of ``SEK3UtilsTest.java`` (TEST_SUITE_MAP.md §invariant_estimator
core tests) onto `inEKF.group`.

Java constants preserved verbatim: ``EPSILON = 1.0e-10``, ``ITERATIONS = 1000``,
per-test seeds 1234 / 5768 / 7777 / 8888 / 9999, and the per-test tolerance
loosenings (1e-9 on the homomorphism, 1e-8 on the conjugation identity).

The Java loop over 1000 trials becomes a single `vmap` over a batch of 1000
draws — the trial count is preserved exactly, but there is no Python loop over a
data dimension. Everything asserted is a property recomputed from the same draw,
so exact draw-matching against Java's RNG is neither possible nor needed.

The k=1 oracle (`SE3LieGroupTools`) is replaced by ``_se3_exp_reference`` /
``_se3_adjoint_reference``: hand-rolled NumPy Rodrigues + left-Jacobian ``V``,
deliberately written independently of `group.py` so the comparison has content.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import group as g

EPSILON = 1.0e-10
ITERATIONS = 1000


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def _random_algebra_vectors(rng: np.random.Generator, k: int, n: int) -> jnp.ndarray:
    """Java ``randomAlgebraVector``, batched: ``(n, 3+3k)``.

    Slots 0:3 are a rotation vector with magnitude bounded by π (Euclid's
    ``nextRotationVector``) so exp/log stays inside the injectivity radius; the
    remaining k blocks are ``nextVector3D``, components ~ U(-1, 1).
    """
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.uniform(-np.pi, np.pi, size=(n, 1))
    phi = axis * angle
    rho = rng.uniform(-1.0, 1.0, size=(n, 3 * k))
    return jnp.asarray(np.concatenate([phi, rho], axis=1))


def _se3_exp_reference(xi: np.ndarray) -> np.ndarray:
    """Independent SE(3) exponential, NumPy: ``R = Rodrigues(φ)``, ``t = V(φ) v``.

    Stands in for the Java ``SE3LieGroupTools.exp``. Written from the closed
    forms directly rather than reusing `group.Gamma0` / `group.Gamma1`, so this
    is a genuine cross-check and not a tautology.
    """
    phi, v = xi[:3], xi[3:6]
    theta = np.linalg.norm(phi)
    K = np.array([
        [0.0, -phi[2], phi[1]],
        [phi[2], 0.0, -phi[0]],
        [-phi[1], phi[0], 0.0],
    ])
    if theta < 1e-12:
        R = np.eye(3) + K
        V = np.eye(3) + 0.5 * K
    else:
        R = (np.eye(3)
             + (np.sin(theta) / theta) * K
             + ((1.0 - np.cos(theta)) / theta**2) * (K @ K))
        V = (np.eye(3)
             + ((1.0 - np.cos(theta)) / theta**2) * K
             + ((theta - np.sin(theta)) / theta**3) * (K @ K))
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = V @ v
    return T


def _se3_adjoint_reference(T: np.ndarray) -> np.ndarray:
    """SE(3) adjoint under rotation-first ordering: ``[[R, 0], [(t)_× R, R]]``."""
    R, t = T[0:3, 0:3], T[0:3, 3]
    t_hat = np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ])
    Ad = np.zeros((6, 6))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = t_hat @ R
    Ad[3:6, 3:6] = R
    return Ad


# ---------------------------------------------------------------------------
# testExpLogRoundTrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_exp_log_round_trip(k):
    """log(exp(ξ)) = ξ to 1e-10, 1000 trials per k."""
    rng = np.random.default_rng(1234)
    xi = _random_algebra_vectors(rng, k, ITERATIONS)

    X = jax.vmap(g.exp_SEk3)(xi)
    recovered = jax.vmap(g.log_SEn3)(X)

    assert X.shape == (ITERATIONS, 3 + k, 3 + k)
    assert recovered.shape == xi.shape
    assert jnp.max(jnp.abs(recovered - xi)) < EPSILON


# ---------------------------------------------------------------------------
# testExpMatchesSE3ForK1
# ---------------------------------------------------------------------------

def test_exp_matches_se3_for_k1():
    """At k=1 the SE_k(3) exp equals the trusted SE(3) exp block-for-block."""
    rng = np.random.default_rng(5768)
    xi = _random_algebra_vectors(rng, 1, ITERATIONS)

    X = jax.vmap(g.exp_SEk3)(xi)
    expected = np.stack([_se3_exp_reference(np.asarray(x)) for x in xi])

    assert X.shape == (ITERATIONS, 4, 4)
    # Rotation block (all 9 entries) and the translation column.
    assert np.max(np.abs(np.asarray(X)[:, 0:3, 0:3] - expected[:, 0:3, 0:3])) < EPSILON
    assert np.max(np.abs(np.asarray(X)[:, 0:3, 3] - expected[:, 0:3, 3])) < EPSILON


# ---------------------------------------------------------------------------
# testLogRejectsWrongSizedOutput  (adapted — see group.log_SEn3 docstring)
# ---------------------------------------------------------------------------

def test_log_rejects_wrong_sized_input():
    with pytest.raises(ValueError):
        g.log_SEn3(jnp.eye(5)[:, :4])       # not square
    with pytest.raises(ValueError):
        g.log_SEn3(jnp.eye(3))              # k = 0, below the minimum 4x4


def test_exp_rejects_inconsistent_tangent_length():
    """The dual guard: ξ length must be 3 + 3k / 3N + 9."""
    with pytest.raises(ValueError):
        g.exp_SEk3(jnp.zeros(7))            # 7 != 3 + 3k
    with pytest.raises(ValueError):
        g.exp_SEn3(jnp.zeros(6), N=2)       # 6 != 3*2 + 9


# ---------------------------------------------------------------------------
# testAdjointMatchesSE3ForK1
# ---------------------------------------------------------------------------

def test_adjoint_matches_se3_for_k1():
    """All 36 entries of the k=1 adjoint match the SE(3) adjoint."""
    rng = np.random.default_rng(7777)
    xi = _random_algebra_vectors(rng, 1, ITERATIONS)

    Ad = jax.vmap(g.Adjoint)(jax.vmap(g.exp_SEk3)(xi))
    expected = np.stack([
        _se3_adjoint_reference(_se3_exp_reference(np.asarray(x))) for x in xi
    ])

    assert Ad.shape == (ITERATIONS, 6, 6)
    assert np.max(np.abs(np.asarray(Ad) - expected)) < EPSILON


# ---------------------------------------------------------------------------
# testAdjointHomomorphism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_adjoint_homomorphism(k):
    """Ad_{X_A X_B} = Ad_{X_A} Ad_{X_B}, tol 1e-9 (Java loosens it here)."""
    rng = np.random.default_rng(8888)
    xi_a = _random_algebra_vectors(rng, k, ITERATIONS)
    xi_b = _random_algebra_vectors(rng, k, ITERATIONS)

    X_a = jax.vmap(g.exp_SEk3)(xi_a)
    X_b = jax.vmap(g.exp_SEk3)(xi_b)

    Ad_ab = jax.vmap(g.Adjoint)(X_a @ X_b)
    Ad_product = jax.vmap(g.Adjoint)(X_a) @ jax.vmap(g.Adjoint)(X_b)

    assert Ad_ab.shape == (ITERATIONS, 3 + 3 * k, 3 + 3 * k)
    assert jnp.max(jnp.abs(Ad_product - Ad_ab)) < 1.0e-9


# ---------------------------------------------------------------------------
# testAdjointConjugationIdentity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_adjoint_conjugation_identity(k):
    """log(X exp(ξ) X⁻¹) = Ad_X ξ, tol 1e-8 — ties Ad to exp/log."""
    rng = np.random.default_rng(9999)
    eta = _random_algebra_vectors(rng, k, ITERATIONS)
    xi = _random_algebra_vectors(rng, k, ITERATIONS)
    # Full-π draws are safe here: conjugation is a similarity on the rotation
    # block, so the conjugated angle equals ‖φ_ξ‖ ≤ π and `log` stays unique.

    X = jax.vmap(g.exp_SEk3)(eta)
    X_inv = jnp.linalg.inv(X)
    conj = X @ jax.vmap(g.exp_SEk3)(xi) @ X_inv

    lhs = jax.vmap(g.log_SEn3)(conj)
    rhs = jnp.einsum("nij,nj->ni", jax.vmap(g.Adjoint)(X), xi)

    assert jnp.max(jnp.abs(lhs - rhs)) < 1.0e-8
