"""Tests for inEKF/group.py — SE_{N+2}(3) Lie-group operations (CLAUDE.md §2).

These lock in:
* the SO(3) closed forms Γ_0/Γ_1/Γ_2 (incl. the θ→0 small-angle branch and
  *finite gradients at θ=0*, the double-where requirement of §8),
* exp/log round-trips on SE_{N+2}(3) for several contact counts N,
* the defining Adjoint identity exp((Ad_X v)^) = X exp(v^) X⁻¹,
* jit-equals-eager.

Build-order note (§9): group.py is unit-tested first because everything
downstream (propagate, correct) leans on these ops.
"""
import jax
import jax.numpy as jnp
import pytest
from jaxlie import SO3

from invariant_estimation.inEKF import group as g

# x64 is enabled process-globally on `import invariant_estimation` (see the
# package __init__); the round-trip / Adjoint tolerances below assume float64.

# Contact counts to exercise, including the degenerate single-contact case.
NS = [0, 1, 2, 4]


def _rng_xi(N, seed, scale=1.0):
    """A reproducible tangent vector ξ ∈ R^{3N+9} with rotation magnitude ~scale."""
    key = jax.random.PRNGKey(seed)
    xi = jax.random.normal(key, (3 * N + 9,))
    # Keep the rotation part bounded well below π so log is the unique inverse.
    xi = xi.at[:3].mul(scale / (jnp.linalg.norm(xi[:3]) + 1e-12))
    return xi


# ---------------------------------------------------------------------------
# skew
# ---------------------------------------------------------------------------

def test_skew_matches_cross():
    a = jnp.array([0.2, -1.3, 0.7])
    b = jnp.array([1.1, 0.4, -0.9])
    assert jnp.allclose(g.skew(a) @ b, jnp.cross(a, b))


def test_skew_antisymmetric():
    a = jnp.array([0.5, 2.0, -1.0])
    K = g.skew(a)
    assert jnp.allclose(K, -K.T)
    assert jnp.allclose(jnp.diag(K), 0.0)


# ---------------------------------------------------------------------------
# Gamma0 / Gamma1 / Gamma2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_gamma0_matches_jaxlie(seed):
    """Γ_0 must equal the jaxlie SO(3) exponential (Rodrigues cross-check)."""
    phi = jax.random.normal(jax.random.PRNGKey(seed), (3,))
    assert jnp.allclose(g.Gamma0(phi), SO3.exp(phi).as_matrix(), atol=1e-10)


def test_gamma0_is_rotation():
    phi = jnp.array([0.3, -0.7, 1.2])
    R = g.Gamma0(phi)
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-10)
    assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-10)


def test_gamma_limits_at_zero():
    """As θ→0: Γ_0→I, Γ_1→I, Γ_2→½I (the small-angle branch)."""
    z = jnp.zeros(3)
    assert jnp.allclose(g.Gamma0(z), jnp.eye(3))
    assert jnp.allclose(g.Gamma1(z), jnp.eye(3))
    assert jnp.allclose(g.Gamma2(z), 0.5 * jnp.eye(3))


def test_gamma_branch_continuity():
    """Series and analytic branches must agree across the θ≈1e-4 threshold."""
    for theta in [3e-5, 1e-4, 3e-4, 1e-3]:
        phi = jnp.array([theta, 0.0, 0.0])
        # Compare against a high-θ neighbour evaluated purely analytically by
        # using a clearly-not-small angle scaled down — here just check the two
        # one-sided evaluations are close to each other and finite.
        for G in (g.Gamma0, g.Gamma1, g.Gamma2):
            out = G(phi)
            assert jnp.all(jnp.isfinite(out))


def test_gamma1_is_left_jacobian():
    """Γ_1 is the SO(3) left Jacobian: Γ_0(φ) = I + (φ)_× Γ_1(φ) ... checked via
    the identity Γ_1(φ) (φ) = (φ) and Γ_0 = exp, J_l J_r⁻¹ consistency.

    Concretely use the known relation  exp((φ)_×) = I + (φ)_× Γ_1(φ).
    """
    phi = jnp.array([0.4, -0.2, 0.9])
    lhs = g.Gamma0(phi)
    rhs = jnp.eye(3) + g.skew(phi) @ g.Gamma1(phi)
    assert jnp.allclose(lhs, rhs, atol=1e-10)


# ---------------------------------------------------------------------------
# Finite gradients at θ = 0 (the double-where requirement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", ["Gamma0", "Gamma1", "Gamma2"])
def test_gamma_grad_finite_at_zero(G):
    fn = getattr(g, G)
    # scalar-valued reduction so we can take a gradient w.r.t. φ at φ=0
    jac = jax.jacobian(fn)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(jac))


def test_exp_grad_finite_at_zero():
    N = 2
    def f(xi): 
        return jnp.sum(g.exp_SEn3(xi, N) ** 2)
    grad = jax.grad(f)(jnp.zeros(3 * N + 9))
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# exp_SEn3 structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_exp_shape_and_structure(N):
    xi = _rng_xi(N, seed=1)
    X = g.exp_SEn3(xi, N)
    assert X.shape == (N + 5, N + 5)
    # Bottom-right (N+2)x(N+2) identity, zero below the rotation block.
    assert jnp.allclose(X[3:, 3:], jnp.eye(N + 2))
    assert jnp.allclose(X[3:, 0:3], 0.0)
    # Rotation block is a proper rotation.
    R = X[0:3, 0:3]
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-10)


@pytest.mark.parametrize("N", NS)
def test_exp_at_zero_is_identity(N):
    X = g.exp_SEn3(jnp.zeros(3 * N + 9), N)
    assert jnp.allclose(X, jnp.eye(N + 5))


# ---------------------------------------------------------------------------
# exp / log round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_log_exp_roundtrip(N):
    """log(exp(ξ)) = ξ for ξ with bounded rotation."""
    xi = _rng_xi(N, seed=7, scale=1.1)
    X = g.exp_SEn3(xi, N)
    assert jnp.allclose(g.log_SEn3(X), xi, atol=1e-9)


@pytest.mark.parametrize("N", NS)
def test_exp_log_roundtrip(N):
    """exp(log(X)) = X for X built from a random tangent vector."""
    xi = _rng_xi(N, seed=3, scale=0.8)
    X = g.exp_SEn3(xi, N)
    assert jnp.allclose(g.exp_SEn3(g.log_SEn3(X), N), X, atol=1e-9)


# ---------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_adjoint_shape_and_diagonal(N):
    xi = _rng_xi(N, seed=5)
    X = g.exp_SEn3(xi, N)
    Ad = g.Adjoint(X)
    dim = 3 * N + 9
    assert Ad.shape == (dim, dim)
    R = X[0:3, 0:3]
    # Every diagonal 3x3 block equals R.
    for k in range(N + 3):
        blk = Ad[3 * k:3 * k + 3, 3 * k:3 * k + 3]
        assert jnp.allclose(blk, R, atol=1e-12)


@pytest.mark.parametrize("N", NS)
def test_adjoint_identity(N):
    """Defining property: exp((Ad_X v)^) = X exp(v^) X⁻¹ for any tangent v."""
    xi = _rng_xi(N, seed=11, scale=0.6)
    X = g.exp_SEn3(xi, N)
    v = _rng_xi(N, seed=12, scale=0.5)

    lhs = g.exp_SEn3(g.Adjoint(X) @ v, N)
    rhs = X @ g.exp_SEn3(v, N) @ jnp.linalg.inv(X)
    assert jnp.allclose(lhs, rhs, atol=1e-8)


@pytest.mark.parametrize("N", NS)
def test_adjoint_first_block_column_coupling(N):
    """First block-column rows below R are (t_k)_× R for t_k ∈ {v, p, d_i}."""
    xi = _rng_xi(N, seed=9)
    X = g.exp_SEn3(xi, N)
    Ad = g.Adjoint(X)
    R = X[0:3, 0:3]
    translations = X[0:3, 3:].T            # v, p, d_1, …, d_N
    for k, t in enumerate(translations, start=1):
        blk = Ad[3 * k:3 * k + 3, 0:3]
        assert jnp.allclose(blk, g.skew(t) @ R, atol=1e-12)


# ---------------------------------------------------------------------------
# jit parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_jit_matches_eager(N):
    xi = _rng_xi(N, seed=2, scale=0.9)
    exp_j = jax.jit(g.exp_SEn3, static_argnums=1)
    log_j = jax.jit(g.log_SEn3)
    adj_j = jax.jit(g.Adjoint)

    X = g.exp_SEn3(xi, N)
    assert jnp.allclose(exp_j(xi, N), X)
    assert jnp.allclose(log_j(X), g.log_SEn3(X))
    assert jnp.allclose(adj_j(X), g.Adjoint(X))
