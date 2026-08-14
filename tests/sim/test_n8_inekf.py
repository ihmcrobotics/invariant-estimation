"""Gate C: the InEKF at N=8 -- group axioms, Jacobians, and CONDITIONING.

Most of N=8 falls out of the site dict (nothing in `inEKF/` hardcodes 2), so the
interesting content here is the last section: four rigid coplanar corners in full
flat contact are a REDUNDANT measurement set, so `S` is close to singular and the
`condition_proxy` gate may fire. That is expected physics, not a bug, and the plan
forbids widening the gate to hide it -- so these tests MEASURE the conditioning and
record it rather than asserting it away.

Group axioms are checked at N=8 with the trial counts the ported Java suite uses,
against independently recomputed oracles (composition vs matrix product, adjoint
vs conjugation), never against a second call to the code under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import importlib

from invariant_estimation.inEKF import group
from invariant_estimation.inEKF import state as st

# `inEKF/__init__` exports a `correct` FUNCTION that shadows the `correct` module,
# and since 3.7 `import pkg.mod as x` resolves through getattr, so it picks up the
# function. import_module goes through sys.modules and gets the real module.
correct = importlib.import_module("invariant_estimation.inEKF.correct")

N8 = 8
DIM = 3 * N8 + 9          # 33
GROUP = N8 + 5            # 13


def _rand_tangent(rng, scale=0.4):
    return jnp.asarray(rng.normal(size=DIM) * scale)


# ---------------------------------------------------------------------------
# group axioms at N=8
# ---------------------------------------------------------------------------

def test_exp_log_round_trip_at_n8():
    """`log(exp(xi)) == xi` over 500 draws (the ported suite's round-trip count)."""
    rng = np.random.default_rng(80081)
    worst = 0.0
    for _ in range(500):
        xi = _rand_tangent(rng)
        back = group.log_SEn3(group.exp_SEn3(xi, N8))
        worst = max(worst, float(jnp.max(jnp.abs(back - xi))))
    assert worst < 1e-9, f"exp/log round-trip broke at N=8 (worst {worst:.2e})"


def test_group_shapes_at_n8():
    rng = np.random.default_rng(1)
    X = group.exp_SEn3(_rand_tangent(rng), N8)
    assert X.shape == (GROUP, GROUP), f"expected SE_{N8 + 2}(3) as {GROUP}x{GROUP}"
    assert group.Adjoint(X).shape == (DIM, DIM)


def test_adjoint_is_the_conjugation_operator_at_n8():
    """Ad_X xi == log(X exp(xi) X^-1): the adjoint's DEFINING property.

    Recomputed from matrix conjugation, which is independent of however `Adjoint`
    builds its blocks -- a block laid out for translation-first ordering fails here.
    """
    rng = np.random.default_rng(4242)
    for _ in range(200):
        X = group.exp_SEn3(_rand_tangent(rng), N8)
        xi = _rand_tangent(rng, scale=0.05)
        lhs = group.Adjoint(X) @ xi
        rhs = group.log_SEn3(X @ group.exp_SEn3(xi, N8) @ jnp.linalg.inv(X))
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-8)


def test_adjoint_is_a_homomorphism_at_n8():
    """Ad_{XY} == Ad_X Ad_Y."""
    rng = np.random.default_rng(777)
    for _ in range(200):
        X = group.exp_SEn3(_rand_tangent(rng), N8)
        Y = group.exp_SEn3(_rand_tangent(rng), N8)
        np.testing.assert_allclose(
            np.asarray(group.Adjoint(X @ Y)),
            np.asarray(group.Adjoint(X) @ group.Adjoint(Y)), atol=1e-8)


# ---------------------------------------------------------------------------
# Jacobians at N=8
# ---------------------------------------------------------------------------

def test_contact_jacobian_shapes_and_state_independence_at_n8():
    for i in range(N8):
        H = correct.contact_jacobian(N8, i)
        assert H.shape == (3, DIM)
    stacked = jnp.concatenate([correct.contact_jacobian(N8, i) for i in range(N8)])
    assert stacked.shape == (3 * N8, DIM) == (24, 33)


def test_contact_jacobian_matches_finite_differences_at_n8():
    """THE Gate C oracle: FD the contact residual w.r.t. a tangent perturbation.

    `h(exp(eps) X)` differentiated numerically must equal the analytic `H` (up to
    the residual's sign convention). Central differences at eps=1e-6 in float64.
    Catches a contact block written at the wrong tangent offset -- which shifts
    silently to a NEIGHBOURING contact at N=8, where the blocks are adjacent.
    """
    rng = np.random.default_rng(31337)
    state = st.InEKFState.identity(N8)
    state = state._replace(
        R=jnp.asarray(group.exp_SEn3(jnp.zeros(DIM).at[:3].set(
            jnp.asarray(rng.normal(size=3) * 0.2)), N8)[:3, :3]),
        v=jnp.asarray(rng.normal(size=3)),
        p=jnp.asarray(rng.normal(size=3)),
        d=jnp.asarray(rng.normal(size=(N8, 3))),
    )

    def perturbed_residual(xi, i, y):
        # X_hat = exp(xi) X  -- the LEFT perturbation every `perturb` helper uses (I5).
        # `InEKFState` has no from_matrix, so unpack the columns directly:
        # R = X[:3,:3], v = col 3, p = col 4, d_i = col 5+i.
        X = group.exp_SEn3(xi, N8) @ state.as_matrix
        s2 = state._replace(R=X[:3, :3], v=X[:3, 3], p=X[:3, 4], d=X[:3, 5:].T)
        return correct.contact_residual(s2, i, y)

    # Linearise where the framework does: the ZERO-RESIDUAL point, y = R^T(d_i - p).
    # This is the whole content of "H is constant" (I3). Writing
    #   nu = R_hat y - (d_hat_i - p_hat),  X_hat = exp(xi) X
    # and expanding to first order, the rotation terms are
    #   xi_R^ (R y)  -  xi_R^ (d_i - p)
    # which cancel IFF R y == d_i - p. At any other y they do not, the residual
    # picks up a genuine rotation sensitivity, and a finite difference taken at a
    # random y disagrees with the analytic H in the rotation block -- correctly.
    y_all = np.asarray(correct.predicted_contact(state))

    eps = 1e-6
    for i in range(N8):
        y = jnp.asarray(y_all[i])
        H_analytic = np.asarray(correct.contact_jacobian(N8, i))
        J = np.zeros((3, DIM))
        for k in range(DIM):
            e = jnp.zeros(DIM).at[k].set(eps)
            plus = perturbed_residual(e, i, y)
            minus = perturbed_residual(-e, i, y)
            J[:, k] = np.asarray((plus - minus) / (2 * eps))
        # `innovation`'s docstring: nu linearises to +H_i xi (xi_p - xi_d_i)
        np.testing.assert_allclose(J, H_analytic, atol=1e-5, err_msg=(
            f"contact {i}: finite-difference Jacobian disagrees with the analytic H"))


# ---------------------------------------------------------------------------
# the update runs at N=8 and dof reaches the metrics
# ---------------------------------------------------------------------------

def _spd(n, seed):
    """The ported suite's deterministic SPD fill: m m^T + n I with sin() entries."""
    i = np.arange(n * n).reshape(n, n)
    m = np.sin(i + 1 + seed)
    return m @ m.T + n * np.eye(n)


def test_stacked_update_shapes_and_dof_at_n8():
    """Stacked 8-contact update: H 24x33, S 24x24, dof 24 reaching diagnostics."""
    state = st.InEKFState.identity(N8)._replace(P=jnp.asarray(_spd(DIM, 3)))
    H = jnp.concatenate([correct.contact_jacobian(N8, i) for i in range(N8)])
    R = jnp.asarray(_spd(3 * N8, 7)) * 1e-4
    residual = jnp.asarray(np.sin(np.arange(3 * N8) + 0.5)) * 1e-3

    new, diag = correct.linear_update(state, H, residual, R)

    assert H.shape == (24, DIM)
    assert new.P.shape == (DIM, DIM)
    assert np.isfinite(float(diag.nis))
    assert np.isfinite(float(diag.logdet_s))
    # P must stay symmetric PSD through a Joseph update at N=8
    P = np.asarray(new.P)
    np.testing.assert_allclose(P, P.T, atol=1e-9)
    assert np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -1e-9


# ---------------------------------------------------------------------------
# THE KNOWN TRAP -- measured, not asserted away
# ---------------------------------------------------------------------------

def test_four_coplanar_corners_are_a_redundant_measurement_set():
    """Record the rank deficiency: it is real, expected, and the plan's key risk.

    Four corners of ONE rigid foot pin 12 numbers but the foot has 6 DoF, so the
    stacked contact measurement is structurally redundant. This documents the
    conditioning that Gate C says not to paper over -- if `condition_proxy` gates
    the update off on flat ground, THIS is why, and the learned Sigma_C is the
    intended fix rather than a wider gate.
    """
    H = np.asarray(jnp.concatenate([correct.contact_jacobian(N8, i) for i in range(N8)]))
    # H itself is full row rank (each contact has its own -I block)...
    assert np.linalg.matrix_rank(H) == 24

    # ...but with a rigid foot the four corner ERRORS move together: the physical
    # constraint is 6-DoF per foot, so a covariance that reflects rigidity makes
    # S ill-conditioned. Model that with a near-rigid contact block.
    P = np.eye(33) * 1e-4
    rigid = np.ones((4, 4)) * (1 - 1e-9) + np.eye(4) * 1e-9
    for foot in range(2):
        for a in range(4):
            for b in range(4):
                ia = 9 + 3 * (foot * 4 + a)
                ib = 9 + 3 * (foot * 4 + b)
                P[ia:ia + 3, ib:ib + 3] = np.eye(3) * 1e-4 * rigid[a, b]

    S = H @ P @ H.T + np.eye(24) * 1e-12
    cond = np.linalg.cond(S)
    # This is a MEASUREMENT, recorded for results.md -- not a threshold to tune.
    print(f"\n[Gate C] cond(S) with rigid-foot corner correlation: {cond:.3e}")
    assert cond > 1e6, (
        "rigid coplanar corners came out well-conditioned; if this is genuinely "
        "true the N=8 conditioning risk is smaller than the plan assumed -- "
        "verify before relying on it")


@pytest.mark.parametrize("n", [2, 8])
def test_independent_contacts_stay_well_conditioned(n):
    """Control for the test above: with INDEPENDENT contacts, S is fine at both N.

    Without this, the redundancy test could be passing for a reason that has
    nothing to do with corner correlation (e.g. any N=8 S being ill-conditioned).
    """
    dim = 3 * n + 9
    H = np.asarray(jnp.concatenate([correct.contact_jacobian(n, i) for i in range(n)]))
    S = H @ (np.eye(dim) * 1e-4) @ H.T + np.eye(3 * n) * 1e-6
    assert np.linalg.cond(S) < 1e4, "independent contacts should be well conditioned"
