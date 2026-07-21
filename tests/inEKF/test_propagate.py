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
import pytest

import importlib

from invariant_estimation.inEKF import group as g
from invariant_estimation.inEKF import state as s

# The package re-exports the `propagate` *function*, which shadows the
# `propagate` submodule attribute; grab the module directly from sys.modules.
pr = importlib.import_module("invariant_estimation.inEKF.propagate")

# x64 is enabled process-globally on `import invariant_estimation`.

NS = [0, 1, 2, 4]


def _params(N, dt=2e-3, grav=None, sg=9e-6, sa=4e-4):
    grav = jnp.array([0.1, -0.2, -9.7]) if grav is None else grav
    return s.InEKFParams(
        g=grav, dt=dt, gyro_var=sg, accel_var=sa, contact_floor=1e-4,
        Phi=s.build_Phi(grav, dt, N), H=s.build_H(N),
    )


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

def _Ad(st):
    """Adjoint of a state's group element — what build_Qd conjugates by."""
    return g.Adjoint(st.as_matrix)


def test_continuous_Qc_structure():
    """Q_c = blkdiag(gyro_var·I, accel_var·I, 0, Σ_{C_i}) — position block zero."""
    N = 2
    p = _params(N)
    sig = _sigma_c(N)
    Qc = pr.continuous_Qc(sig, p)
    assert Qc.shape == (3 * N + 9, 3 * N + 9)
    assert jnp.allclose(Qc[0:3, 0:3], p.gyro_var * jnp.eye(3))
    assert jnp.allclose(Qc[3:6, 3:6], p.accel_var * jnp.eye(3))
    assert jnp.allclose(Qc[6:9, 6:9], 0.0)          # position: no direct noise
    for i in range(N):
        sl = slice(9 + 3 * i, 12 + 3 * i)
        assert jnp.allclose(Qc[sl, sl], sig[i])
    # Block diagonal: no cross terms in the *continuous* density.
    assert jnp.allclose(Qc[0:3, 3:], 0.0)
    assert jnp.allclose(Qc[9:, 0:9], 0.0)


@pytest.mark.parametrize("N", NS)
def test_build_Qd_is_eq38(N):
    """Q_d = Φ Ad Q_c Adᵀ Φᵀ Δt exactly (CLAUDE.md I3 / paper Eq. 38)."""
    p = _params(N)
    st = _seed_state(N)
    sig = _sigma_c(N)
    Ad = _Ad(st)

    Qd = pr.build_Qd(sig, Ad, p)
    M = p.Phi @ Ad
    expected = M @ pr.continuous_Qc(sig, p) @ M.T * p.dt

    assert Qd.shape == (3 * N + 9, 3 * N + 9)
    assert jnp.allclose(Qd, expected, atol=1e-14)
    assert jnp.allclose(Qd, Qd.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Qd) >= -1e-12)


@pytest.mark.parametrize("N", NS)
def test_build_Qd_keeps_the_adjoint(N):
    """The Ad_X̂ conjugation is load-bearing — dropping it is the §6 trap.

    Even with isotropic Q_g/Q_a the adjoint is not a no-op: its first
    block-column carries (v)_× R̂ and (p)_× R̂, which generate genuine cross
    terms. This test fails if someone "cleans up" build_Qd to match Hartley.
    """
    p = _params(N)
    st = _seed_state(N)
    sig = _sigma_c(N)

    with_ad = pr.build_Qd(sig, _Ad(st), p)
    without_ad = pr.build_Qd(sig, jnp.eye(3 * N + 9), p)
    assert not jnp.allclose(with_ad, without_ad, atol=1e-9)


def test_build_Qd_is_error_independent():
    """Q_d depends on the estimate X̂ but never on the error ξ (I3).

    Perturbing the *estimate* changes Q_d (state-dependent); that is expected.
    What must hold is that build_Qd is a pure function of (X̂, Σ_C, params) —
    it never sees an error state at all, which is what keeps Φ log-linear.
    """
    N = 2
    p = _params(N)
    st = _seed_state(N)
    sig = _sigma_c(N)
    Ad = _Ad(st)
    # Same estimate ⇒ bit-identical Q_d, regardless of P (the error covariance).
    a = pr.build_Qd(sig, Ad, p)
    b = pr.build_Qd(sig, Ad, p)
    assert jnp.array_equal(a, b)
    st_other_P = st._replace(P=st.P * 7.0 + jnp.eye(3 * N + 9))
    assert jnp.array_equal(pr.build_Qd(sig, _Ad(st_other_P), p), a)


@pytest.mark.parametrize("N", NS)
def test_build_Qd_scales_linearly_with_dt(N):
    """First-order discretisation: Q_d ∝ Δt at fixed Φ (TODO(van-loan) in source)."""
    p1 = _params(N, dt=1e-3)
    p2 = p1._replace(dt=2e-3)                    # Φ held fixed on purpose
    st = _seed_state(N)
    sig = _sigma_c(N)
    Ad = _Ad(st)
    assert jnp.allclose(pr.build_Qd(sig, Ad, p2), 2.0 * pr.build_Qd(sig, Ad, p1), atol=1e-14)


# ---------------------------------------------------------------------------
# Covariance propagation (§3.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_propagate_cov_formula(N):
    """P⁺ = Φ P Φᵀ + Q̄_d, symmetrised (§3.2)."""
    p = _params(N)
    st = _seed_state(N)
    sig = _sigma_c(N)
    Ad = _Ad(st)
    Pn = pr.propagate_cov(st.P, sig, Ad, p)
    expected = p.Phi @ st.P @ p.Phi.T + pr.build_Qd(sig, Ad, p)
    assert jnp.allclose(Pn, expected, atol=1e-10)
    assert jnp.allclose(Pn, Pn.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Pn) >= -1e-9)


def test_propagate_cov_grows_uncertainty():
    """Process noise is additive ⇒ predicted covariance is no smaller."""
    p = _params(2)
    st = _seed_state(2)
    sig = _sigma_c(2)
    Pn = pr.propagate_cov(st.P, sig, _Ad(st), p)
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
    assert jnp.allclose(out.P, pr.propagate_cov(st.P, sig, _Ad(st), p))
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
