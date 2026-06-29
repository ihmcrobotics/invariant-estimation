"""Tests for inEKF/propagate.py — the prediction step (CLAUDE.md §3).

Three things are pinned here:

* the **exact mean** integration (§3.1) — checked against its defining physical
  cases (free fall, pure rotation, rotating-accel) rather than re-deriving it;
* the **state-independent constant Φ** covariance step ``P⁺ = Φ P Φᵀ + Q̄_d``
  (§3.2);
* the **exact closed-form Q̄_d** (§3.3) — checked numerically against the defining
  integral ``∫₀^{dt} e^{A^r s} Q̄_c e^{A^rᵀ s} ds`` (the thing the closed form
  replaces), plus PSD, the decoupled contact blocks, and the symmetry/jit/grad
  invariants.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest

import importlib

from invariant_estimation.inEKF import group as g
from invariant_estimation.inEKF import state as s

# The package re-exports the `propagate` *function*, which shadows the
# `propagate` submodule attribute; grab the module directly from sys.modules.
pr = importlib.import_module("invariant_estimation.inEKF.propagate")

# x64 is enabled process-globally on `import invariant_estimation`.

NS = [0, 1, 2, 4]


def _params(N, dt=2e-3, grav=None, sg=3e-3, sa=2e-2):
    grav = jnp.array([0.1, -0.2, -9.7]) if grav is None else grav
    return s.InEKFParams(
        g=grav, dt=dt, sigma_gyro=sg, sigma_accel=sa, contact_floor=1e-4,
        Phi=s.build_Phi(grav, dt, N), H=s.build_H(N),
    )


def _A_r_inertial(grav):
    """The 9x9 inertial part of the constant error-dynamics matrix A^r (§3.2)."""
    A = jnp.zeros((9, 9))
    A = A.at[3:6, 0:3].set(g.skew(grav))   # (g)_× : R → v
    A = A.at[6:9, 3:6].set(jnp.eye(3))     # I    : v → p
    return A


def _seed_state(N, key=0):
    k = jax.random.PRNGKey(key)
    kR, kv, kp, kd, kP = jax.random.split(k, 5)
    R0 = g.Gamma0(jax.random.normal(kR, (3,)) * 0.4)
    v0 = jax.random.normal(kv, (3,))
    p0 = jax.random.normal(kp, (3,))
    d0 = jax.random.normal(kd, (N, 3))
    A = jax.random.normal(kP, (3 * N + 9, 3 * N + 9))
    P = A @ A.T + jnp.eye(3 * N + 9)        # SPD
    return s.InEKFState(R=R0, v=v0, p=p0, d=d0, P=P)


def _sigma_c(N, key=1):
    """Random SPD per-contact world-frame densities, shape (N, 3, 3)."""
    if N == 0:
        return jnp.zeros((0, 3, 3))
    A = jax.random.normal(jax.random.PRNGKey(key), (N, 3, 3))
    return jnp.einsum("nij,nkj->nik", A, A) + jnp.eye(3)


# ---------------------------------------------------------------------------
# Mean propagation (§3.1)
# ---------------------------------------------------------------------------

def test_mean_free_fall():
    """ω = a = 0: orientation frozen, base in free fall under g only."""
    p = _params(2)
    st = _seed_state(2)
    out = pr.propagate_mean(st, jnp.zeros(3), jnp.zeros(3), p)
    dt = p.dt
    assert jnp.allclose(out.R, st.R)
    assert jnp.allclose(out.v, st.v + p.g * dt)
    assert jnp.allclose(out.p, st.p + st.v * dt + 0.5 * p.g * dt ** 2)


def test_mean_pure_rotation():
    """a = 0: R right-multiplies by Γ_0(ω dt); no extra translation beyond g."""
    p = _params(2)
    st = _seed_state(2)
    omega = jnp.array([0.3, -0.5, 0.2])
    out = pr.propagate_mean(st, omega, jnp.zeros(3), p)
    assert jnp.allclose(out.R, st.R @ g.Gamma0(omega * p.dt))
    assert jnp.allclose(out.v, st.v + p.g * p.dt)


def test_mean_zero_omega_accel():
    """ω = 0 ⇒ Γ_0 = Γ_1 = I, Γ_2 = ½I: the closed form is the const-accel step."""
    p = _params(2)
    st = _seed_state(2)
    a = jnp.array([0.7, -1.1, 0.4])
    dt = p.dt
    out = pr.propagate_mean(st, jnp.zeros(3), a, p)
    v_exp = st.v + st.R @ a * dt + p.g * dt
    p_exp = st.p + st.v * dt + st.R @ (0.5 * a) * dt ** 2 + 0.5 * p.g * dt ** 2
    assert jnp.allclose(out.v, v_exp)
    assert jnp.allclose(out.p, p_exp)


@pytest.mark.parametrize("N", NS)
def test_mean_contacts_and_cov_frozen(N):
    """Mean step leaves contact positions and P untouched (§3.1)."""
    p = _params(N)
    st = _seed_state(N)
    out = pr.propagate_mean(st, jnp.array([0.1, 0.2, 0.3]), jnp.ones(3), p)
    assert jnp.array_equal(out.d, st.d)
    assert jnp.array_equal(out.P, st.P)
    assert jnp.allclose(out.R @ out.R.T, jnp.eye(3), atol=1e-12)   # stays SO(3)


# ---------------------------------------------------------------------------
# Process noise Q̄_d (§3.3)
# ---------------------------------------------------------------------------

def test_inertial_Qd_matches_integral():
    """Closed-form inertial Q̄_d == ∫₀^{dt} e^{A s} Q̄_c e^{Aᵀ s} ds (§3.3)."""
    p = _params(0, dt=5e-3)
    A = _A_r_inertial(p.g)
    Qc = jsl.block_diag(
        p.sigma_gyro ** 2 * jnp.eye(3),
        p.sigma_accel ** 2 * jnp.eye(3),
        jnp.zeros((3, 3)),
    )
    # Dense Simpson quadrature of the matrix integrand (reference, not used in prod).
    n = 2000
    sgrid = jnp.linspace(0.0, p.dt, n + 1)

    def integrand(sv):
        E = jsl.expm(A * sv)
        return E @ Qc @ E.T

    vals = jax.vmap(integrand)(sgrid)                 # (n+1, 9, 9)
    w = jnp.ones(n + 1).at[1:-1:2].set(4.0).at[2:-1:2].set(2.0)
    ref = (p.dt / n) / 3.0 * jnp.einsum("k,kij->ij", w, vals)

    assert jnp.allclose(pr.inertial_Qd(p), ref, atol=1e-12, rtol=1e-9)


def test_inertial_Qd_double_integrator_limit():
    """Drop gyro noise ⇒ textbook double-integrator block (§3.3 sanity check)."""
    p = _params(0, dt=3e-3, sg=0.0, sa=0.0)
    p = p._replace(sigma_accel=0.5)
    dt, q = p.dt, p.sigma_accel ** 2
    Q = pr.inertial_Qd(p)
    # gyro off ⇒ R block zero, and v/p reduce to [[q dt³/3, q dt²/2],[·, q dt]].
    assert jnp.allclose(Q[0:3, 0:3], 0.0)
    assert jnp.allclose(Q[3:6, 3:6], q * dt * jnp.eye(3))
    assert jnp.allclose(Q[6:9, 6:9], q * dt ** 3 / 3.0 * jnp.eye(3))
    assert jnp.allclose(Q[3:6, 6:9], q * dt ** 2 / 2.0 * jnp.eye(3))
    assert jnp.allclose(Q[6:9, 3:6], q * dt ** 2 / 2.0 * jnp.eye(3))


def test_inertial_Qd_symmetric_psd():
    p = _params(0)
    Q = pr.inertial_Qd(p)
    assert jnp.allclose(Q, Q.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Q) >= -1e-12)


@pytest.mark.parametrize("N", NS)
def test_build_Qd_structure(N):
    """Inertial block in the corner; decoupled contact blocks = Σ_c dt; PSD."""
    p = _params(N)
    sig = _sigma_c(N)
    Qd = pr.build_Qd(sig, p)
    assert Qd.shape == (3 * N + 9, 3 * N + 9)
    assert jnp.allclose(Qd[0:9, 0:9], pr.inertial_Qd(p))
    # No cross terms between inertial and contact blocks (decoupled in A^r).
    assert jnp.allclose(Qd[0:9, 9:], 0.0)
    assert jnp.allclose(Qd[9:, 0:9], 0.0)
    for i in range(N):
        sl = slice(9 + 3 * i, 12 + 3 * i)
        assert jnp.allclose(Qd[sl, sl], sig[i] * p.dt)
        for j in range(N):
            if i != j:
                cl = slice(9 + 3 * j, 12 + 3 * j)
                assert jnp.allclose(Qd[sl, cl], 0.0)
    assert jnp.allclose(Qd, Qd.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Qd) >= -1e-10)


# ---------------------------------------------------------------------------
# Covariance propagation (§3.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_propagate_cov_formula(N):
    """P⁺ = Φ P Φᵀ + Q̄_d, symmetrised (§3.2)."""
    p = _params(N)
    st = _seed_state(N)
    sig = _sigma_c(N)
    Pn = pr.propagate_cov(st.P, sig, p)
    expected = p.Phi @ st.P @ p.Phi.T + pr.build_Qd(sig, p)
    assert jnp.allclose(Pn, expected, atol=1e-10)
    assert jnp.allclose(Pn, Pn.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Pn) >= -1e-9)


def test_propagate_cov_grows_uncertainty():
    """Process noise is additive ⇒ predicted covariance is no smaller."""
    p = _params(2)
    st = _seed_state(2)
    sig = _sigma_c(2)
    Pn = pr.propagate_cov(st.P, sig, p)
    assert jnp.trace(Pn) >= jnp.trace(st.P)


# ---------------------------------------------------------------------------
# Full step + JAX invariants (§8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_propagate_full_step(N):
    p = _params(N)
    st = _seed_state(N)
    sig = _sigma_c(N)
    omega = jnp.array([0.2, -0.1, 0.3])
    accel = jnp.array([0.5, 0.4, -0.3])
    out = pr.propagate(st, omega, accel, sig, p)
    mean = pr.propagate_mean(st, omega, accel, p)
    assert jnp.allclose(out.R, mean.R)
    assert jnp.allclose(out.v, mean.v)
    assert jnp.allclose(out.p, mean.p)
    assert jnp.array_equal(out.d, st.d)
    assert jnp.allclose(out.P, pr.propagate_cov(st.P, sig, p))
    assert out.P.shape == (3 * N + 9, 3 * N + 9)


def test_propagate_jit_matches_eager():
    p = _params(2)
    st = _seed_state(2)
    sig = _sigma_c(2)
    omega = jnp.array([0.2, -0.1, 0.3])
    accel = jnp.array([0.5, 0.4, -0.3])
    eager = pr.propagate(st, omega, accel, sig, p)
    jitted = jax.jit(pr.propagate)(st, omega, accel, sig, p)
    assert jnp.allclose(eager.R, jitted.R)
    assert jnp.allclose(eager.P, jitted.P)


def test_propagate_differentiable():
    """BPTT must flow through the filter: grad wrt IMU and contact noise is finite."""
    p = _params(2)
    st = _seed_state(2)
    sig = _sigma_c(2)

    def loss(omega, accel, sigma_c):
        out = pr.propagate(st, omega, accel, sigma_c, p)
        return jnp.sum(out.v ** 2) + jnp.sum(out.p ** 2) + jnp.trace(out.P)

    grads = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.array([0.1, 0.2, 0.3]), jnp.array([0.4, -0.2, 0.1]), sig
    )
    for grad in grads:
        assert jnp.all(jnp.isfinite(grad))


def test_propagate_grad_finite_at_zero_omega():
    """θ → 0 in the Γ closed forms must not poison the gradient (double-where)."""
    p = _params(2)
    st = _seed_state(2)
    sig = _sigma_c(2)

    def loss(omega):
        out = pr.propagate(st, omega, jnp.ones(3), sig, p)
        return jnp.sum(out.p ** 2)

    grad = jax.grad(loss)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(grad))
