r"""Tests for inEKF/correct.py — the FK measurement update (CLAUDE.md §4).

What is pinned here:

* the **right-invariant observation** model (§4.1): the predicted body-frame FK
  vector ``h_pred_i = R̄ᵀ(d̄_i − p̄)``, a zero innovation when the measurement
  matches the state, and — the load-bearing check — that the innovation
  **linearises to the precomputed constant ``H``** (``\nu ≈ −H ξ``).  That single
  test fixes the sign convention independent of any hand derivation;
* the **block-diagonal measurement noise** assembly (§4.2);
* the **gain + Joseph update** (§4.3): the ``S = HPHᵀ+N`` / ``K = PHᵀS⁻¹``
  identities, a symmetric-PSD ``P⁺``, and the **left**-multiplied ``exp(ξ⁺) X̄``;
* a behavioural check that a trusted (small-noise) measurement **reduces the
  innovation** and pulls the state toward FK consistency — the real guard against
  a flipped sign;
* the JAX invariants (§8): ``N = 0`` no-op, jit-matches-eager, finite gradients
  (BPTT must flow through the update).
"""
import importlib

import jax
import jax.numpy as jnp
import pytest

from invariant_estimation.inEKF import group as g
from invariant_estimation.inEKF import state as s

# `correct` the function shadows the submodule on the package; import directly.
co = importlib.import_module("invariant_estimation.inEKF.correct")

# x64 is enabled process-globally on `import invariant_estimation`.

NS = [0, 1, 2, 4]
NS_POS = [1, 2, 4]


def _params(N, dt=2e-3, grav=None):
    grav = jnp.array([0.1, -0.2, -9.7]) if grav is None else grav
    return s.InEKFParams(
        g=grav, dt=dt, sigma_gyro=3e-3, sigma_accel=2e-2, contact_floor=1e-4,
        Phi=s.build_Phi(grav, dt, N), H=s.build_H(N),
    )


def _seed_state(N, key=0, p_scale=1.0):
    """Random SPD-covariance state on SE_{N+2}(3)."""
    k = jax.random.PRNGKey(key)
    kR, kv, kp, kd, kP = jax.random.split(k, 5)
    R0 = g.Gamma0(jax.random.normal(kR, (3,)) * 0.4)
    v0 = jax.random.normal(kv, (3,))
    p0 = jax.random.normal(kp, (3,))
    d0 = jax.random.normal(kd, (N, 3))
    A = jax.random.normal(kP, (3 * N + 9, 3 * N + 9))
    P = p_scale * (A @ A.T) + jnp.eye(3 * N + 9)
    return s.InEKFState(R=R0, v=v0, p=p0, d=d0, P=P)


def _Np(N, key=7, scale=1e-3):
    """Random SPD per-contact position FK covariances, shape (N, 3, 3)."""
    if N == 0:
        return jnp.zeros((0, 3, 3))
    A = jax.random.normal(jax.random.PRNGKey(key), (N, 3, 3))
    return scale * (jnp.einsum("nij,nkj->nik", A, A) + jnp.eye(3))


def _fk_from_state(state):
    """The FK measurement that exactly matches the state (zero innovation)."""
    return co.predicted_contact(state)


# ---------------------------------------------------------------------------
# Observation model (§4.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS_POS)
def test_predicted_contact_matches_definition(N):
    """h_pred_i = R̄ᵀ(d̄_i − p̄), per contact (§4.1)."""
    st = _seed_state(N)
    pred = co.predicted_contact(st)
    assert pred.shape == (N, 3)
    expected = jnp.stack([st.R.T @ (st.d[i] - st.p) for i in range(N)])
    assert jnp.allclose(pred, expected, atol=1e-12)


@pytest.mark.parametrize("N", NS_POS)
def test_innovation_zero_when_measurement_matches(N):
    """Measurement = state prediction ⇒ innovation is exactly zero."""
    st = _seed_state(N)
    y = _fk_from_state(st)
    nu = co.innovation(st, y)
    assert nu.shape == (3 * N,)
    assert jnp.allclose(nu, 0.0, atol=1e-12)


@pytest.mark.parametrize("N", NS_POS)
def test_innovation_linearises_to_minus_H(N):
    r"""\nu(exp(ξ) X_true, y_true) ≈ −H ξ — fixes the sign vs the precomputed H.

    With the measurement taken from the *true* state, perturbing the mean by a
    small right-invariant error ξ makes the (measurement − model) innovation
    reproduce ``−H ξ`` to first order; the ``+K \nu`` update then drives the error
    to ``(I−KH) ξ`` (§4.1, §4.3).  This is the definitive sign check.
    """
    p = _params(N)
    true = _seed_state(N, key=3)
    y = _fk_from_state(true)                      # measurement from the truth

    key = jax.random.PRNGKey(99)
    xi = jax.random.normal(key, (3 * N + 9,)) * 1e-5
    Xbar = g.exp_SEn3(xi, N) @ true.as_matrix     # X̄ = exp(ξ) X_true
    pert = true._replace(
        R=Xbar[0:3, 0:3], v=Xbar[0:3, 3], p=Xbar[0:3, 4], d=Xbar[0:3, 5:].T,
    )
    nu = co.innovation(pert, y)
    assert jnp.allclose(nu, -p.H @ xi, atol=1e-9)


# ---------------------------------------------------------------------------
# Measurement noise (§4.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_measurement_noise_block_diagonal(N):
    """N is block-diagonal with the per-contact N^p_i on the diagonal (§4.2)."""
    Np = _Np(N)
    Nmat = co.measurement_noise(Np)
    assert Nmat.shape == (3 * N, 3 * N)
    for i in range(N):
        sl = slice(3 * i, 3 * i + 3)
        assert jnp.allclose(Nmat[sl, sl], Np[i])
        for j in range(N):
            if i != j:
                assert jnp.allclose(Nmat[sl, slice(3 * j, 3 * j + 3)], 0.0)


# ---------------------------------------------------------------------------
# Gain and Joseph update (§4.3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS_POS)
def test_kalman_gain_identities(N):
    """S = H P Hᵀ + N and K S = P Hᵀ (the gain solves the normal equations)."""
    p = _params(N)
    st = _seed_state(N)
    Nmat = co.measurement_noise(_Np(N))
    K, S = co.kalman_gain(st.P, p.H, Nmat)
    assert K.shape == (3 * N + 9, 3 * N)
    assert S.shape == (3 * N, 3 * N)
    assert jnp.allclose(S, p.H @ st.P @ p.H.T + Nmat, atol=1e-10)
    assert jnp.allclose(K @ S, st.P @ p.H.T, atol=1e-8)


@pytest.mark.parametrize("N", NS_POS)
def test_joseph_update_symmetric_psd(N):
    """Joseph form yields a symmetric PSD covariance (§4.3, invariant 7)."""
    p = _params(N)
    st = _seed_state(N)
    Nmat = co.measurement_noise(_Np(N))
    K, _ = co.kalman_gain(st.P, p.H, Nmat)
    Pp = co.joseph_update(st.P, K, p.H, Nmat)
    assert Pp.shape == st.P.shape
    assert jnp.allclose(Pp, Pp.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(Pp) >= -1e-9)


@pytest.mark.parametrize("N", NS_POS)
def test_joseph_matches_standard_form(N):
    """Joseph form equals (I−KH)P at the optimal gain (numerical cross-check)."""
    p = _params(N)
    st = _seed_state(N)
    Nmat = co.measurement_noise(_Np(N))
    K, _ = co.kalman_gain(st.P, p.H, Nmat)
    joseph = co.joseph_update(st.P, K, p.H, Nmat)
    short = (jnp.eye(st.dim) - K @ p.H) @ st.P
    assert jnp.allclose(joseph, 0.5 * (short + short.T), atol=1e-7)


@pytest.mark.parametrize("N", NS_POS)
def test_apply_correction_left_multiply(N):
    """X̄⁺ = exp(ξ⁺) X̄ — exp multiplies on the LEFT (invariant 9)."""
    st = _seed_state(N)
    xi = jax.random.normal(jax.random.PRNGKey(5), (3 * N + 9,)) * 0.05
    out = co.apply_correction(st, xi)
    Xexp = g.exp_SEn3(xi, N) @ st.as_matrix
    assert jnp.allclose(out.as_matrix, Xexp, atol=1e-10)
    assert jnp.allclose(out.R @ out.R.T, jnp.eye(3), atol=1e-10)   # stays SO(3)
    assert jnp.array_equal(out.P, st.P)                            # P untouched here


# ---------------------------------------------------------------------------
# Full correction step (§4) — structure & behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS_POS)
def test_correct_shapes_and_psd(N):
    p = _params(N)
    st = _seed_state(N)
    y = _fk_from_state(st) + 0.01 * jax.random.normal(jax.random.PRNGKey(2), (N, 3))
    out, nu = co.correct(st, y, _Np(N), p)
    assert out.R.shape == (3, 3) and out.v.shape == (3,) and out.p.shape == (3,)
    assert out.d.shape == (N, 3)
    assert out.P.shape == (3 * N + 9, 3 * N + 9)
    assert nu.shape == (3 * N,)
    assert jnp.allclose(out.P, out.P.T, atol=1e-12)
    assert jnp.all(jnp.linalg.eigvalsh(out.P) >= -1e-9)
    assert jnp.allclose(out.R @ out.R.T, jnp.eye(3), atol=1e-10)


@pytest.mark.parametrize("N", NS_POS)
def test_correct_reduces_innovation(N):
    """A trusted (small-noise) measurement pulls the state toward FK consistency.

    The post-update innovation (recomputed against the same measurement) must be
    smaller than the prior one — the behavioural guarantee that the sign of the
    update is correct.
    """
    p = _params(N)
    st = _seed_state(N, p_scale=2.0)
    # A measurement displaced from the prediction; very small noise ⇒ trusted.
    y = _fk_from_state(st) + 0.05 * jax.random.normal(jax.random.PRNGKey(4), (N, 3))
    Np = _Np(N, scale=1e-6)

    nu_before = co.innovation(st, y)
    out, _ = co.correct(st, y, Np, p)
    nu_after = co.innovation(out, y)
    assert jnp.linalg.norm(nu_after) < jnp.linalg.norm(nu_before)


def test_correct_reduces_covariance_trace():
    """Information added ⇒ corrected covariance trace is no larger."""
    p = _params(2)
    st = _seed_state(2)
    y = _fk_from_state(st)
    out, _ = co.correct(st, y, _Np(2), p)
    assert jnp.trace(out.P) <= jnp.trace(st.P) + 1e-9


def test_correct_no_contacts_is_noop():
    """N = 0: nothing to measure, state returned unchanged (static early return)."""
    p = _params(0)
    st = _seed_state(0)
    out, nu = co.correct(st, jnp.zeros((0, 3)), jnp.zeros((0, 3, 3)), p)
    assert nu.shape == (0,)
    assert jnp.array_equal(out.R, st.R)
    assert jnp.array_equal(out.v, st.v)
    assert jnp.array_equal(out.p, st.p)
    assert jnp.array_equal(out.P, st.P)


# ---------------------------------------------------------------------------
# JAX invariants (§8)
# ---------------------------------------------------------------------------

def test_correct_jit_matches_eager():
    p = _params(2)
    st = _seed_state(2)
    y = _fk_from_state(st) + 0.01
    Np = _Np(2)
    e_st, e_nu = co.correct(st, y, Np, p)
    j_st, j_nu = jax.jit(co.correct)(st, y, Np, p)
    assert jnp.allclose(e_st.R, j_st.R)
    assert jnp.allclose(e_st.p, j_st.p)
    assert jnp.allclose(e_st.d, j_st.d)
    assert jnp.allclose(e_st.P, j_st.P, atol=1e-10)
    assert jnp.allclose(e_nu, j_nu)


def test_correct_differentiable():
    """BPTT must flow through the update: grad wrt measurement and noise is finite."""
    p = _params(2)
    st = _seed_state(2)

    def loss(y, Np):
        out, nu = co.correct(st, y, Np, p)
        return jnp.sum(out.p ** 2) + jnp.trace(out.P) + jnp.sum(nu ** 2)

    y0 = _fk_from_state(st) + 0.02
    g_y, g_Np = jax.grad(loss, argnums=(0, 1))(y0, _Np(2))
    assert jnp.all(jnp.isfinite(g_y))
    assert jnp.all(jnp.isfinite(g_Np))


def test_correct_grad_through_state_finite():
    """Gradient back into the prior state mean/covariance is finite."""
    p = _params(2)
    base = _seed_state(2)
    y = _fk_from_state(base) + 0.02
    Np = _Np(2)

    def loss(p_vec):
        st = base._replace(p=p_vec)
        out, _ = co.correct(st, y, Np, p)
        return jnp.sum(out.p ** 2) + jnp.trace(out.P)

    grad = jax.grad(loss)(base.p)
    assert jnp.all(jnp.isfinite(grad))
