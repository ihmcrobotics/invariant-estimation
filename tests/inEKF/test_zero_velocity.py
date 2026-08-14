"""Gates Z1-Z3 for the contact zero-velocity constraint.

Z1 algebra: H^v is exactly [0|-I|0|0], state-independent, adds no yaw and no absolute
position; stacked rank 24 -> 27 of 33 at N=8 with common-mode translation still in the
null space.
Z2 FD oracle: finite-difference dnu^v/dxi against H^v, linearised AT THE ZERO-RESIDUAL
POINT -- the same caveat the Gate C contact oracle carries.
Z3 noise: N^v reduces to R Sigma_C R^T when the sensing terms vanish; the fused block
equals the stacked one; the cross-covariance with the position block is not zero.
"""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.inEKF import group as gr, zero_velocity as zv
from invariant_estimation.inEKF.state import InEKFState

from ._oracles import next_rotation_matrix

# `inEKF.__init__` re-exports the function `correct`, shadowing the module of the
# same name — import the module explicitly (same workaround as test_contact_updater).
correct = importlib.import_module("invariant_estimation.inEKF.correct")

N_C, N_J = 8, 6


def _state(seed=0, n=N_C):
    rng = np.random.default_rng(seed)
    return InEKFState(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(rng.standard_normal(3)),
        p=jnp.asarray(rng.standard_normal(3)),
        d=jnp.asarray(rng.standard_normal((n, 3))),
        P=jnp.eye(3 * n + 9))


def _kin(seed=1, n=N_C):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.standard_normal(3) * 0.5),            # omega
            jnp.asarray(rng.standard_normal((n, 3)) * 0.3),       # h
            jnp.asarray(rng.standard_normal((n, 3, N_J)) * 0.4),  # J
            jnp.asarray(rng.standard_normal(N_J)))                # q_dot


# --------------------------------------------------------------------------- Z1

def test_jacobian_is_exactly_minus_identity_on_velocity():
    H = np.asarray(zv.velocity_jacobian(N_C))
    assert H.shape == (3, 3 * N_C + 9)
    assert np.array_equal(H[:, 3:6], -np.eye(3)), "velocity block must be -I"
    rest = np.delete(H, np.s_[3:6], axis=1)
    assert np.array_equal(rest, np.zeros_like(rest)), (
        "every other block must be exactly zero: rotation columns would manufacture "
        "yaw, position/contact columns would claim absolute position")


def test_jacobian_is_state_independent():
    """The `testJacobianStructureAndStateIndependence` pattern: two different states,
    bit-identical H. This is invariant I3, and it is what keeps the XLA graph constant."""
    a = zv.velocity_jacobian(N_C)
    b = zv.velocity_jacobian(N_C)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    # and it does not take a state at all -- the signature is the proof
    with pytest.raises(TypeError):
        zv.velocity_jacobian(N_C, _state())  # type: ignore[call-arg]


def test_stacking_raises_rank_but_leaves_common_mode_unobservable():
    """The load-bearing claim: this constraint STARVES the sink mode, it does not
    observe it. If a future change makes common-mode translation observable, that is a
    different (and much stronger) filter -- and this test should be the one that says so.
    """
    Hp = jnp.concatenate([correct.contact_jacobian(N_C, i) for i in range(N_C)])
    Hv = zv.velocity_jacobian(N_C)
    stacked = np.asarray(jnp.concatenate([Hp, Hv]))

    assert np.linalg.matrix_rank(np.asarray(Hp)) == 3 * N_C            # 24
    assert np.linalg.matrix_rank(stacked) == 3 * N_C + 3               # 27 of 33

    dim = 3 * N_C + 9
    # common-mode translation: base position and every contact move together
    xi = np.zeros(dim)
    xi[6:9] = [0.0, 0.0, 1.0]
    for i in range(N_C):
        xi[9 + 3 * i: 12 + 3 * i] = [0.0, 0.0, 1.0]
    assert np.abs(stacked @ xi).max() == 0.0, (
        "common-mode translation must stay exactly unobservable")

    # base-only vertical motion IS observed (the control for the above)
    xi_base = np.zeros(dim)
    xi_base[6:9] = [0.0, 0.0, 1.0]
    assert np.abs(stacked @ xi_base).max() == pytest.approx(1.0)

    # rotation, including yaw, is untouched by the velocity rows
    xi_yaw = np.zeros(dim)
    xi_yaw[0:3] = [0.0, 0.0, 1.0]
    assert np.abs(np.asarray(Hv) @ xi_yaw).max() == 0.0


# --------------------------------------------------------------------------- Z2

def test_finite_difference_matches_H_at_the_zero_residual_point():
    """FD of dnu^v/dxi against H^v.

    Linearised AT THE ZERO-RESIDUAL POINT, i.e. with v set so that R y^v = v exactly.
    The cancellation of the attitude error holds iff that condition is met -- the same
    caveat the Gate C contact-Jacobian oracle carries, where an FD at a random y
    disagrees in the rotation block *correctly*. An oracle written the other way
    reports a convention mismatch as a bug.
    """
    omega, h, J, q_dot = _kin()
    base = _state()
    y_v = zv.velocity_measurement(omega, h, J, q_dot)
    # put the state exactly on the constraint for contact 0
    state = base._replace(v=base.R @ y_v[0])
    assert np.abs(np.asarray(zv.velocity_residual(state, y_v)[:3])).max() < 1e-14

    H = np.asarray(zv.velocity_jacobian(N_C))
    dim = 3 * N_C + 9
    eps = 1e-6
    fd = np.zeros((3, dim))
    for k in range(dim):
        xi = np.zeros(dim)
        xi[k] = eps
        X = gr.exp_SEk3(jnp.asarray(xi)) @ state.as_matrix       # X̂ = exp(xi)X (I5)
        pert = state._replace(R=X[0:3, 0:3], v=X[0:3, 3], p=X[0:3, 4],
                              d=X[0:3, 5:].T)
        nu = np.asarray(zv.velocity_residual(pert, y_v)[:3])
        fd[:, k] = nu / eps
    assert np.abs(fd - H).max() < 1e-6, f"max |FD - H| = {np.abs(fd - H).max():.2e}"


def test_residual_is_zero_when_the_contact_is_truly_static():
    """The constraint must be exactly satisfied by a state that satisfies it -- the
    zero-release property, in the same spirit as the reseed test."""
    omega, h, J, q_dot = _kin(seed=3)
    base = _state(seed=3)
    y_v = zv.velocity_measurement(omega, h, J, q_dot)
    state = base._replace(v=base.R @ y_v[2])
    nu = np.asarray(zv.velocity_residual(state, y_v)).reshape(N_C, 3)
    assert np.abs(nu[2]).max() < 1e-14


def test_measurement_actually_holds_the_contact_world_static():
    """The oracle that pins the FORMULA, not just its algebra.

    Every other test here either checks H's structure (independent of y) or feeds a
    self-consistent y, so all of them pass with the gyro lever-arm term deleted --
    measured, and that term is ~0.45 m/s at |h| = 0.9 m and |w| = 0.5 rad/s, i.e.
    comparable to the signal. This test instead integrates the contact's WORLD position
    forward and asserts it does not move:

        d_i(t) = p(t) + R(t) h_i(q(t)),  p += v dt,  R += R[w] dt,  q += qdot dt

    with a synthetic FK h(q) = h0 + A q, so J = A exactly and nothing is circular. Set
    v = R y^v and d_i must be stationary to O(dt^2). This is `applyConsistentMotion`'s
    discipline: build the motion so the measurement model must hold, then check it does.
    """
    rng = np.random.default_rng(11)
    R = jnp.asarray(next_rotation_matrix(rng))
    omega = jnp.asarray(rng.standard_normal(3) * 0.5)
    A = jnp.asarray(rng.standard_normal((3, N_J)) * 0.4)          # J, exactly
    h0 = jnp.asarray(rng.standard_normal(3) * 0.3)
    q = jnp.asarray(rng.standard_normal(N_J))
    q_dot = jnp.asarray(rng.standard_normal(N_J))
    p = jnp.asarray(rng.standard_normal(3))

    h = h0 + A @ q
    y_v = zv.velocity_measurement(omega, h[None], A[None], q_dot)[0]
    v = R @ y_v                                                   # on the constraint

    def contact_world(dt):
        R_t = R @ gr.Gamma0(omega * dt)          # Gamma0 IS the SO(3) exponential
        return (p + v * dt) + R_t @ (h0 + A @ (q + q_dot * dt))

    d0 = np.asarray(contact_world(0.0))
    for dt in (1e-5, 1e-6):
        drift = np.linalg.norm(np.asarray(contact_world(dt)) - d0) / dt
        assert drift < 1e-4, (
            f"contact moves at {drift:.3e} m/s while the state satisfies the "
            f"constraint — the measurement formula is wrong")


# --------------------------------------------------------------------------- Z3

def test_noise_reduces_to_rotated_sigma_c():
    """With the sensing terms zeroed, N^v IS the transported contact process noise.
    That identity is the whole reason the constraint does not double-count the
    'foot is planted' evidence."""
    state = _state(seed=5)
    _, h, J, _ = _kin(seed=5)
    rng = np.random.default_rng(5)
    L = np.tril(rng.standard_normal((N_C, 3, 3))) * 0.1 + np.eye(3) * 1e-3
    sigma_c = jnp.asarray(L @ np.swapaxes(L, -1, -2))

    N_v = zv.velocity_noise(state, sigma_c, J, jnp.zeros((N_J, N_J)), h, jnp.zeros((3, 3)), 1.0)
    want = np.asarray(state.R) @ np.asarray(sigma_c) @ np.asarray(state.R).T
    assert np.allclose(np.asarray(N_v), want, rtol=1e-12, atol=1e-14)


def test_noise_is_spd_and_grows_with_every_term():
    state = _state(seed=6)
    omega, h, J, _ = _kin(seed=6)
    sigma_c = jnp.broadcast_to(jnp.eye(3) * 1e-6, (N_C, 3, 3))
    small = zv.velocity_noise(state, sigma_c, J, jnp.zeros((N_J, N_J)), h, jnp.zeros((3, 3)), 1.0)
    big = zv.velocity_noise(state, sigma_c, J, jnp.eye(N_J) * 1e-2, h,
                            jnp.eye(3) * 1e-4, 1.0)
    for M in (np.asarray(small), np.asarray(big)):
        assert np.allclose(M, np.swapaxes(M, -1, -2), atol=1e-14)
        assert (np.linalg.eigvalsh(M) > 0).all()
    assert (np.trace(np.asarray(big), axis1=-2, axis2=-1)
            > np.trace(np.asarray(small), axis1=-2, axis2=-1)).all()


def test_fused_block_equals_the_stacked_update():
    """Z4's conditioning remedy must be an identity, not an approximation.

    Every contact contributes the same three rows, so stacking is N redundant
    observations of one 3-vector; fusing first keeps the update at 3 rows and the
    conditioning at N=1's. This asserts the two give the same posterior.
    """
    state = _state(seed=7)
    omega, h, J, q_dot = _kin(seed=7)
    rng = np.random.default_rng(7)
    L = np.tril(rng.standard_normal((N_C, 3, 3))) * 0.05 + np.eye(3) * 0.02
    sigma_c = jnp.asarray(L @ np.swapaxes(L, -1, -2))
    y_v = zv.velocity_measurement(omega, h, J, q_dot)
    N_v = zv.velocity_noise(state, sigma_c, J, jnp.eye(N_J) * 1e-4, h, jnp.eye(3) * 1e-5, 1.0)

    Hv = zv.velocity_jacobian(N_C)
    H_stack = jnp.tile(Hv, (N_C, 1))
    R_stack = correct.measurement_noise(N_v)
    nu_stack = zv.velocity_residual(state, y_v)
    s_state, _ = correct.linear_update(state, H_stack, nu_stack, R_stack)

    nu_f, N_f = zv.fuse_contacts(nu_stack.reshape(N_C, 3), N_v)
    f_state, _ = correct.linear_update(state, Hv, nu_f, N_f)

    for a, b, name in ((s_state.R, f_state.R, "R"), (s_state.v, f_state.v, "v"),
                       (s_state.p, f_state.p, "p"), (s_state.d, f_state.d, "d"),
                       (s_state.P, f_state.P, "P")):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-11), name


def test_cross_covariance_with_the_position_block_is_not_zero():
    """A block-diagonal R over [nu^p; nu^v] would drop this and double-count the
    encoder -- invariant I6 in a new place. This asserts the term is real, i.e. that
    assuming independence is a measurable error rather than a harmless one."""
    state = _state(seed=8)
    omega, _, J, _ = _kin(seed=8)
    sigma_q = jnp.eye(N_J) * 5.0e-5
    cross = np.asarray(zv.velocity_position_cross(state, J, sigma_q, omega))
    assert cross.shape == (N_C, 3, 3)
    assert np.abs(cross).max() > 0.0, "the coupling must not be identically zero"
    # it is driven by omega: a stationary base decouples the two blocks
    zero_omega = np.asarray(
        zv.velocity_position_cross(state, J, sigma_q, jnp.zeros(3)))
    assert np.abs(zero_omega).max() == 0.0


# ------------------------------------------------- nv_scale (the trust knob)

def test_nv_scale_is_exactly_a_multiplier_and_defaults_to_identity():
    """kappa scales N^v and nothing else, and kappa=1 is BIT-identical to the
    unscaled call — so adding the knob cannot move a single recorded number."""
    state = _state(seed=11)
    omega, h, J, _ = _kin(seed=11)
    sigma_c = jnp.broadcast_to(jnp.eye(3) * 1e-6, (N_C, 3, 3))
    args = (state, sigma_c, J, jnp.eye(N_J) * 1e-4, h, jnp.eye(3) * 1e-5, 1.0e-3)

    base = np.asarray(zv.velocity_noise(*args))
    assert np.array_equal(np.asarray(zv.velocity_noise(*args, 1.0)), base), (
        "kappa=1 must be bit-identical to the default path")
    for kappa in (0.1, 10.0, 100.0):
        got = np.asarray(zv.velocity_noise(*args, kappa))
        assert np.allclose(got, kappa * base, rtol=1e-14, atol=0.0)


def test_large_nv_scale_degrades_to_the_no_zero_velocity_baseline():
    """kappa -> inf must switch the constraint OFF, not do something else.

    An untrusted measurement has K -> 0, so the posterior must converge back to the
    prior. If it did not, the sweep in the morning report would be measuring a bug
    rather than a trust level, so this is asserted rather than assumed.
    """
    state = _state(seed=12)
    omega, h, J, q_dot = _kin(seed=12)
    sigma_c = jnp.broadcast_to(jnp.eye(3) * 1e-6, (N_C, 3, 3))
    y_v = zv.velocity_measurement(omega, h, J, q_dot)
    nu = zv.velocity_residual(state, y_v).reshape(N_C, 3)
    Hv = zv.velocity_jacobian(N_C)

    def posterior(kappa):
        N_v = zv.velocity_noise(state, sigma_c, J, jnp.eye(N_J) * 1e-4, h,
                                jnp.eye(3) * 1e-5, 1.0e-3, kappa)
        nu_f, N_f = zv.fuse_contacts(nu, N_v)
        return correct.linear_update(state, Hv, nu_f, N_f)[0]

    moved = np.linalg.norm(np.asarray(posterior(1.0).v) - np.asarray(state.v))
    assert moved > 1e-3, "at kappa=1 the constraint must actually do something"

    # Once kappa N^v dominates H P Hᵀ the gain is K ≈ P Hᵀ (kappa N)⁻¹, so the
    # correction must fall as exactly 1/kappa. Asserting the LAW, not just a bound:
    # a knob that saturated at some floor would still shrink monotonically.
    gaps = {k: np.linalg.norm(np.asarray(posterior(k).v) - np.asarray(state.v))
            for k in (1e6, 1e9, 1e12)}
    products = [k * g for k, g in gaps.items()]
    assert max(products) / min(products) < 1.01, (
        f"correction does not fall as 1/kappa: kappa*|dv| = {products}")
    assert gaps[1e12] < 1e-6, (
        f"kappa=1e12 still moves v by {gaps[1e12]:.3e} — the block does not switch "
        "itself off, so 'no zero-velocity' is not the limit of the sweep")
