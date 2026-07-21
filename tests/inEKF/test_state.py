"""Tests for inEKF/state.py — the SE_{N+2}(3) state/params and the precomputed
constants Φ (CLAUDE.md §3.2) and H (§4.1).

These lock in the design-§1 layout (R, v, p, d, P) with ordering
ξ = [ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; …], the dense-matrix builder, and — crucially —
that Φ equals the exact matrix exponential of the constant nilpotent A^r and that
H is the state-independent FK pattern.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest

from invariant_estimation.inEKF import group as g
from invariant_estimation.inEKF import state as s

# x64 is enabled process-globally on `import invariant_estimation`.

# Contact counts, including the pure-inertial edge case N = 0.
NS = [0, 1, 2, 4]


def _A_r(grav, N):
    """The constant right-invariant error-dynamics matrix A^r (§3.2)."""
    dim = 3 * N + 9
    A = jnp.zeros((dim, dim))
    A = A.at[3:6, 0:3].set(g.skew(grav))   # (g)_× : R → v
    A = A.at[6:9, 3:6].set(jnp.eye(3))     # I    : v → p
    return A


# ---------------------------------------------------------------------------
# State layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_init_state_shapes(N):
    st = s.init_state(N)
    dim = 3 * N + 9
    assert st.R.shape == (3, 3)
    assert st.v.shape == (3,)
    assert st.p.shape == (3,)
    assert st.d.shape == (N, 3)
    assert st.P.shape == (dim, dim)


@pytest.mark.parametrize("N", NS)
def test_inferred_dims(N):
    st = s.init_state(N)
    assert st.N == N
    assert st.dim == 3 * N + 9


def test_init_defaults():
    st = s.init_state(3)
    assert jnp.allclose(st.R, jnp.eye(3))
    assert jnp.all(st.v == 0.0)
    assert jnp.all(st.p == 0.0)
    assert jnp.all(st.d == 0.0)


def test_init_seeds():
    R0 = g.Gamma0(jnp.array([0.1, -0.2, 0.3]))
    v0 = jnp.array([1.0, 2.0, 3.0])
    p0 = jnp.array([4.0, 5.0, 6.0])
    d0 = jnp.arange(6.0).reshape(2, 3)
    st = s.init_state(2, R0=R0, v0=v0, p0=p0, d0=d0)
    assert jnp.allclose(st.R, R0)
    assert jnp.array_equal(st.v, v0)
    assert jnp.array_equal(st.p, p0)
    assert jnp.array_equal(st.d, d0)


@pytest.mark.parametrize("N", NS)
def test_P_symmetric_psd(N):
    st = s.init_state(N)
    assert jnp.allclose(st.P, st.P.T)
    assert jnp.all(jnp.linalg.eigvalsh(st.P) >= 0.0)


def test_contacts_diffuse_relative_to_position():
    """Default contact prior is looser than the base position prior (CoCo 'off')."""
    st = s.init_state(2, p_p=1e-2, p_d=1.0)
    d_block = jnp.diag(st.P)[9:]
    p_block = jnp.diag(st.P)[6:9]
    assert jnp.all(d_block > p_block.max())


# ---------------------------------------------------------------------------
# Dense matrix builder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_as_matrix_structure(N):
    R0 = g.Gamma0(jnp.array([0.2, 0.1, -0.3]))
    v0 = jnp.array([1.0, 0.0, -1.0])
    p0 = jnp.array([0.5, 0.5, 0.5])
    d0 = jnp.arange(3.0 * N).reshape(N, 3) if N else jnp.zeros((0, 3))
    st = s.init_state(N, R0=R0, v0=v0, p0=p0, d0=d0)
    X = st.as_matrix
    assert X.shape == (N + 5, N + 5)
    assert jnp.allclose(X[0:3, 0:3], R0)
    assert jnp.allclose(X[0:3, 3], v0)
    assert jnp.allclose(X[0:3, 4], p0)
    assert jnp.allclose(X[0:3, 5:], d0.T)
    # Bottom-right identity, zero below the rotation block.
    assert jnp.allclose(X[3:, 3:], jnp.eye(N + 2))
    assert jnp.allclose(X[3:, 0:3], 0.0)


def test_as_matrix_log_roundtrip():
    """as_matrix is the genuine group element: log∘exp consistency via group.py."""
    N = 2
    xi = jax.random.normal(jax.random.PRNGKey(0), (3 * N + 9,)) * 0.3
    X = g.exp_SEn3(xi, N)
    st = s.InEKFState(
        R=X[0:3, 0:3], v=X[0:3, 3], p=X[0:3, 4], d=X[0:3, 5:].T,
        P=jnp.eye(3 * N + 9),
    )
    assert jnp.allclose(st.as_matrix, X, atol=1e-12)


# ---------------------------------------------------------------------------
# Φ  (precomputed transition)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_phi_matches_expm(N):
    """Φ must equal expm(A^r dt) exactly (nilpotent ⇒ closed form, §3.2)."""
    grav = jnp.array([0.0, 0.0, -9.81])
    dt = 1e-3
    Phi = s.build_Phi(grav, dt, N)
    expected = jsl.expm(_A_r(grav, N) * dt)
    assert Phi.shape == (3 * N + 9, 3 * N + 9)
    assert jnp.allclose(Phi, expected, atol=1e-10)


def test_phi_block_values():
    grav = jnp.array([0.1, -0.2, -9.7])
    dt = 2e-3
    N = 2
    Phi = s.build_Phi(grav, dt, N)
    G = g.skew(grav)
    assert jnp.allclose(Phi[3:6, 0:3], G * dt)              # [v, R]
    assert jnp.allclose(Phi[6:9, 3:6], jnp.eye(3) * dt)     # [p, v]
    assert jnp.allclose(Phi[6:9, 0:3], 0.5 * G * dt * dt)   # [p, R]
    # Diagonal identity everywhere.
    assert jnp.allclose(jnp.diag(Phi), 1.0)


def test_phi_contacts_uncoupled():
    """The d-block of Φ is identity: contacts have no deterministic coupling."""
    grav = jnp.array([0.0, 0.0, -9.81])
    N = 3
    Phi = s.build_Phi(grav, 1e-3, N)
    d_block = Phi[9:, :]
    expected = jnp.zeros((3 * N, 3 * N + 9)).at[:, 9:].set(jnp.eye(3 * N))
    assert jnp.allclose(d_block, expected)


def test_A_r_nilpotent():
    """(A^r)³ = 0 — the property that makes Φ and Q̄_d exact closed forms."""
    A = _A_r(jnp.array([0.0, 0.0, -9.81]), 2)
    assert jnp.allclose(A @ A @ A, 0.0)


# ---------------------------------------------------------------------------
# H  (precomputed FK observation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_H_shape_and_pattern(N):
    H = s.build_H(N)
    assert H.shape == (3 * N, 3 * N + 9)
    # R and v blocks are zero.
    assert jnp.allclose(H[:, 0:6], 0.0)
    for i in range(N):
        rows = slice(3 * i, 3 * i + 3)
        assert jnp.allclose(H[rows, 6:9], -jnp.eye(3))                 # −I in p
        # +I in this contact's own d_i block, zero in the others.
        for j in range(N):
            cols = slice(9 + 3 * j, 9 + 3 * j + 3)
            expected = jnp.eye(3) if i == j else jnp.zeros((3, 3))
            assert jnp.allclose(H[rows, cols], expected)


# ---------------------------------------------------------------------------
# Params + pytree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", NS)
def test_default_params(N):
    p = s.default_params(N, dt=2e-3)
    expected = {"g", "dt", "gyro_var", "accel_var", "contact_floor", "Phi", "H"}
    assert set(p._fields) == expected
    assert p.dt == 2e-3
    assert p.Phi.shape == (3 * N + 9, 3 * N + 9)
    assert p.H.shape == (3 * N, 3 * N + 9)
    assert jnp.allclose(p.g, jnp.array([0.0, 0.0, -9.81]))
    for v in (p.gyro_var, p.accel_var, p.contact_floor):
        assert v > 0.0


def test_default_params_phi_matches_builder():
    N = 2
    p = s.default_params(N, dt=1e-3)
    assert jnp.allclose(p.Phi, s.build_Phi(p.g, p.dt, N))
    assert jnp.allclose(p.H, s.build_H(N))


@pytest.mark.parametrize("N", NS)
def test_state_is_pytree(N):
    st = s.init_state(N)
    leaves, treedef = jax.tree_util.tree_flatten(st)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, s.InEKFState)
    assert jnp.array_equal(rebuilt.P, st.P)
    assert rebuilt.N == N


def test_as_matrix_jit():
    st = s.init_state(2, v0=jnp.array([1.0, 2.0, 3.0]))
    eager = st.as_matrix
    jitted = jax.jit(lambda x: x.as_matrix)(st)
    assert jnp.allclose(eager, jitted)
