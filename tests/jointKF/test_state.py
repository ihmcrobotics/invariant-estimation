"""Tests for joint_kf/state.py — the bias-augmented joint KF state/params.

These lock in the design-§1 state layout  x = [q ; q_dot ; b_omega] ∈ R^{2n+3m}
and the marginal-block accessors the rest of the filter (and the InEKF /
ContactNet boundary) build on.
"""
import jax.numpy as jnp
import pytest

from invariant_estimation.jointKF import state as s


# (n_joints, n_pairs) combos, including the encoder-only edge case (m = 0).
SHAPES = [(6, 2), (1, 1), (12, 4), (3, 0)]


@pytest.mark.parametrize("n, m", SHAPES)
def test_init_state_shapes(n, m):
    st = s.init_state(n_joints=n, n_pairs=m)
    dim = 2 * n + 3 * m
    assert st.q_hat.shape == (n,)
    assert st.q_dot_hat.shape == (n,)
    assert st.b_omega.shape == (3 * m,)
    assert st.P.shape == (dim, dim)
    assert st.x.shape == (dim,)


@pytest.mark.parametrize("n, m", SHAPES)
def test_inferred_dims(n, m):
    st = s.init_state(n_joints=n, n_pairs=m)
    assert st.n_joints == n
    assert st.n_pairs == m


def test_init_state_defaults_zero():
    st = s.init_state(n_joints=4, n_pairs=2)
    assert jnp.all(st.q_hat == 0.0)
    assert jnp.all(st.q_dot_hat == 0.0)
    assert jnp.all(st.b_omega == 0.0)


def test_q0_seed():
    q0 = jnp.arange(5.0)
    st = s.init_state(n_joints=5, n_pairs=1, q0=q0)
    assert jnp.array_equal(st.q_hat, q0)


def test_x_ordering():
    """x must be exactly [q ; q_dot ; b_omega] in that order."""
    n, m = 3, 2
    st = s.init_state(n_joints=n, n_pairs=m)
    # Seed each segment with a distinct constant so we can locate it in x.
    st = st._replace(
        q_hat=jnp.full(n, 1.0),
        q_dot_hat=jnp.full(n, 2.0),
        b_omega=jnp.full(3 * m, 3.0),
    )
    x = st.x
    assert jnp.all(x[:n] == 1.0)
    assert jnp.all(x[n:2 * n] == 2.0)
    assert jnp.all(x[2 * n:] == 3.0)


@pytest.mark.parametrize("n, m", SHAPES)
def test_marginal_blocks_match_P(n, m):
    st = s.init_state(n_joints=n, n_pairs=m)
    assert jnp.array_equal(st.sigma_q, st.P[:n, :n])
    assert jnp.array_equal(st.sigma_q_dot, st.P[n:2 * n, n:2 * n])
    assert jnp.array_equal(st.sigma_b, st.P[2 * n:, 2 * n:])
    assert st.sigma_q.shape == (n, n)
    assert st.sigma_q_dot.shape == (n, n)
    assert st.sigma_b.shape == (3 * m, 3 * m)


def test_sigma_q_dot_excludes_bias_block():
    """Regression guard: sigma_q_dot must NOT fold in the bias block.

    The old implementation sliced P[n:, n:]; with the bias block now appended
    that would wrongly include bias rows/cols.  Distinguish the blocks by
    giving velocity and bias different variances and checking the marginal is
    pure velocity.
    """
    n, m = 2, 2
    st = s.init_state(n_joints=n, n_pairs=m)
    sqd = st.sigma_q_dot
    assert sqd.shape == (n, n)
    # Every diagonal entry is the velocity prior (10.0), none the bias prior.
    assert jnp.allclose(jnp.diag(sqd), 10.0)
    assert not jnp.any(jnp.isclose(jnp.diag(sqd), st.P[2 * n, 2 * n]))


def test_b_omega_pairs_view():
    n, m = 3, 2
    st = s.init_state(n_joints=n, n_pairs=m)
    st = st._replace(b_omega=jnp.arange(3.0 * m))
    pairs = st.b_omega_pairs
    assert pairs.shape == (m, 3)
    assert jnp.array_equal(pairs.reshape(-1), st.b_omega)


@pytest.mark.parametrize("n, m", SHAPES)
def test_P_symmetric_and_psd(n, m):
    st = s.init_state(n_joints=n, n_pairs=m)
    assert jnp.allclose(st.P, st.P.T)
    eig = jnp.linalg.eigvalsh(st.P)
    assert jnp.all(eig >= 0.0)


def test_bias_block_is_tight():
    """Bias prior variance must be far tighter than the joint priors."""
    n, m = 4, 2
    st = s.init_state(n_joints=n, n_pairs=m)
    bias_var = jnp.diag(st.sigma_b)
    pos_var = jnp.diag(st.sigma_q)
    vel_var = jnp.diag(st.sigma_q_dot)
    assert jnp.all(bias_var < pos_var.min())
    assert jnp.all(bias_var < vel_var.min())


def test_default_params_fields():
    p = s.default_params()
    expected = {"sigma_enc", "sigma_omega", "sigma_tau", "sigma_acc", "sigma_b", "dt"}
    assert set(p._fields) == expected
    assert "sigma_vel" not in p._fields  # removed: no direct velocity measurement


def test_default_params_values():
    p = s.default_params(dt=2e-3)
    assert p.dt == 2e-3
    # Residual bias random walk must be tight relative to the accel process std.
    assert p.sigma_b < p.sigma_acc
    for v in (p.sigma_enc, p.sigma_omega, p.sigma_tau, p.sigma_acc, p.sigma_b):
        assert v > 0.0


def test_state_is_pytree():
    """JointKFState round-trips through jax.tree_util (scan compatibility)."""
    import jax
    st = s.init_state(n_joints=3, n_pairs=1)
    leaves, treedef = jax.tree_util.tree_flatten(st)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, s.JointKFState)
    assert jnp.array_equal(rebuilt.P, st.P)
