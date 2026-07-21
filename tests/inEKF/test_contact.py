r"""Tests for inEKF/contact.py — the contact-covariance digest (CLAUDE.md §5).

`contact.py` is a pure *consumer* of ContactNet's lower-triangular Cholesky
factors; these tests pin exactly that and nothing more:

* `reconstruct_cov` produces ``Σ = L Lᵀ`` (symmetric PSD), consuming only the
  lower triangle of the incoming factor;
* `apply_floor` is the additive variance floor (min eigenvalue ``≥ floor``);
* `rotate_to_world` is the conjugation ``R̄ Σ R̄ᵀ`` (preserves eigenvalues / floor);
* `digest` composes them into the ``(N, 3, 3)`` world-frame densities that
  `propagate.build_Qd` consumes — verified by feeding the output straight into
  the propagation;
* the JAX invariants (§8): the ``N = 0`` edge, jit-matches-eager, and finite
  gradients through the digest (BPTT into ContactNet), including at a singular
  factor where the additive floor must keep the gradient finite.
"""
import importlib

import jax
import jax.numpy as jnp
import pytest

from invariant_estimation.inEKF import state as s

co = importlib.import_module("invariant_estimation.inEKF.contact")
pr = importlib.import_module("invariant_estimation.inEKF.propagate")
g = importlib.import_module("invariant_estimation.inEKF.group")

# x64 is enabled process-globally on `import invariant_estimation`.

NS = [0, 1, 2, 4]
NS_POS = [1, 2, 4]


def _params(N, floor=1e-4):
    grav = jnp.array([0.1, -0.2, -9.7])
    return s.InEKFParams(
        g=grav, dt=2e-3, gyro_var=9e-6, accel_var=4e-4, contact_floor=floor,
        Phi=s.build_Phi(grav, 2e-3, N), H=s.build_H(N),
    )


def _L(N, key=0):
    """Random lower-triangular Cholesky factors with positive diagonal, (N,3,3)."""
    if N == 0:
        return jnp.zeros((0, 3, 3))
    A = jax.random.normal(jax.random.PRNGKey(key), (N, 3, 3))
    L = jnp.tril(A)
    # Positive diagonal (ContactNet's SPD-safe parameterisation guarantees this).
    diag = jnp.abs(jnp.diagonal(L, axis1=-2, axis2=-1)) + 0.5
    idx = jnp.arange(3)
    return L.at[:, idx, idx].set(diag)


def _R(key=1):
    return g.Gamma0(jax.random.normal(jax.random.PRNGKey(key), (3,)) * 0.5)


# ---------------------------------------------------------------------------
# reconstruct_cov (§5 step 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS_POS)
def test_reconstruct_is_L_Lt(N):
    L = _L(N)
    Sigma = co.reconstruct_cov(L)
    assert Sigma.shape == (N, 3, 3)
    expected = jnp.stack([L[i] @ L[i].T for i in range(N)])
    assert jnp.allclose(Sigma, expected, atol=1e-12)


@pytest.mark.parametrize("N", NS_POS)
def test_reconstruct_symmetric_psd(N):
    Sigma = co.reconstruct_cov(_L(N))
    assert jnp.allclose(Sigma, jnp.swapaxes(Sigma, -1, -2), atol=1e-12)
    eig = jnp.linalg.eigvalsh(Sigma)
    assert jnp.all(eig >= -1e-12)


def test_reconstruct_ignores_upper_triangle():
    """Only the lower triangle of the incoming factor is consumed (§5 contract)."""
    L = _L(2, key=3)
    # Pollute the strict upper triangle; tril must make this a no-op.
    polluted = L.at[:, 0, 1].add(7.0).at[:, 0, 2].add(-3.0).at[:, 1, 2].add(5.0)
    assert jnp.allclose(co.reconstruct_cov(L), co.reconstruct_cov(polluted))


# ---------------------------------------------------------------------------
# apply_floor (§5 step 2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS_POS)
def test_apply_floor_is_additive(N):
    Sigma = co.reconstruct_cov(_L(N))
    floored = co.apply_floor(Sigma, 1e-3)
    assert jnp.allclose(floored, Sigma + 1e-3 * jnp.eye(3), atol=1e-14)


@pytest.mark.parametrize("N", NS_POS)
def test_apply_floor_min_eigenvalue(N):
    floor = 2e-3
    floored = co.apply_floor(co.reconstruct_cov(_L(N)), floor)
    assert jnp.all(jnp.linalg.eigvalsh(floored) >= floor - 1e-12)


def test_apply_floor_singular_factor():
    """A zero factor floors to exactly floor·I (the 'no contact' / degenerate case)."""
    Sigma = co.reconstruct_cov(jnp.zeros((2, 3, 3)))
    floored = co.apply_floor(Sigma, 5e-3)
    assert jnp.allclose(floored, 5e-3 * jnp.broadcast_to(jnp.eye(3), (2, 3, 3)))


# ---------------------------------------------------------------------------
# rotate_to_world (§5 step 3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS_POS)
def test_rotate_matches_conjugation(N):
    Sigma = co.reconstruct_cov(_L(N))
    R = _R()
    rot = co.rotate_to_world(Sigma, R)
    expected = jnp.stack([R @ Sigma[i] @ R.T for i in range(N)])
    assert jnp.allclose(rot, expected, atol=1e-12)


def test_rotate_identity_is_noop():
    Sigma = co.reconstruct_cov(_L(3))
    assert jnp.allclose(co.rotate_to_world(Sigma, jnp.eye(3)), Sigma, atol=1e-12)


@pytest.mark.parametrize("N", NS_POS)
def test_rotate_preserves_eigenvalues(N):
    """Conjugation by a rotation preserves the spectrum (and hence the floor)."""
    Sigma = co.apply_floor(co.reconstruct_cov(_L(N)), 1e-3)
    rot = co.rotate_to_world(Sigma, _R())
    assert jnp.allclose(
        jnp.sort(jnp.linalg.eigvalsh(rot)),
        jnp.sort(jnp.linalg.eigvalsh(Sigma)),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# digest (§5) — composition + downstream wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_digest_composition(N):
    p = _params(N)
    L = _L(N)
    R = _R()
    out = co.digest(L, p)
    assert out.shape == (N, 3, 3)
    manual = co.apply_floor(co.reconstruct_cov(L), p.contact_floor)
    assert jnp.allclose(out, manual, atol=1e-14)
    # Body frame: the digest must NOT pre-rotate — Ad_X̂ in build_Qd does that.
    if N:
        assert not jnp.allclose(out, co.rotate_to_world(manual, R), atol=1e-9)


@pytest.mark.parametrize("N", NS_POS)
def test_digest_symmetric_psd_floored(N):
    p = _params(N, floor=1e-3)
    out = co.digest(_L(N), p)
    assert jnp.allclose(out, jnp.swapaxes(out, -1, -2), atol=1e-12)
    assert jnp.all(jnp.linalg.eigvalsh(out) >= p.contact_floor - 1e-10)


@pytest.mark.parametrize("N", NS)
def test_digest_feeds_build_Qd(N):
    """digest output is exactly the sigma_c that build_Qd consumes (I3 seam)."""
    p = _params(N)
    sigma_c = co.digest(_L(N), p)
    Ad = jnp.eye(3 * N + 9)
    Qd = pr.build_Qd(sigma_c, Ad, p)
    assert Qd.shape == (3 * N + 9, 3 * N + 9)
    # At Ad = Φ = I the contact blocks reduce to Σ_{C_i} dt.
    p_identity = p._replace(Phi=jnp.eye(3 * N + 9))
    Qd_plain = pr.build_Qd(sigma_c, Ad, p_identity)
    for i in range(N):
        sl = slice(9 + 3 * i, 12 + 3 * i)
        assert jnp.allclose(Qd_plain[sl, sl], sigma_c[i] * p.dt, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Qd) >= -1e-9)


def test_digest_no_contacts():
    p = _params(0)
    out = co.digest(jnp.zeros((0, 3, 3)), p)
    assert out.shape == (0, 3, 3)


# ---------------------------------------------------------------------------
# JAX invariants (§8)
# ---------------------------------------------------------------------------

def test_digest_jit_matches_eager():
    p = _params(2)
    L = _L(2)
    assert jnp.allclose(co.digest(L, p), jax.jit(co.digest)(L, p), atol=1e-12)


def test_digest_differentiable():
    """BPTT into ContactNet: grad wrt the Cholesky factors must be finite."""
    p = _params(2)

    def loss(L):
        return jnp.sum(co.digest(L, p) ** 2)

    grad = jax.grad(loss)(_L(2))
    assert jnp.all(jnp.isfinite(grad))


def test_digest_grad_finite_at_singular_factor():
    """The additive floor keeps the gradient finite even at a zero factor."""
    p = _params(2, floor=1e-3)

    def loss(L):
        # trace of the digested covariance: smooth in L through the floor.
        return jnp.sum(jnp.trace(co.digest(L, p), axis1=-2, axis2=-1))

    grad = jax.grad(loss)(jnp.zeros((2, 3, 3)))
    assert jnp.all(jnp.isfinite(grad))
