"""THE decisive oracle — `JointLevelKFStackedOracleTest`, feet-active half.

The pairs-only half lives in `test_measurement.py`
(`test_stacked_pair_rows_match_the_marginalized_raw_gyro_reference`). This file
adds the case the plan reserves for the parent: **with a stance anchor active**,
because that is where `measure.py` (pair rows), `anchors.py` (anchor rows) and
`update.py` (the Joseph update) must *compose*, and a wrong answer is not locally
detectable by any one of them.

The claim
---------
The stacked Joseph update must equal a reference KF that measures the **raw**
per-IMU gyros with block-diagonal (independent) noise, plus a near-zero
absolute-rate constraint per trusted foot, over a state augmented with a nuisance
base angular velocity `omega_base` — which is then marginalised out under an
improper prior (the `gamma -> infinity` limit, i.e. a zero information block).

Why the two must agree, and what it pins down
---------------------------------------------
Differencing two IMUs is *exactly* elimination of the shared unknown
`omega_base`. Eliminating a variable you hold no prior on is marginalisation in
the information form. So the correlations that differencing induces — between
pairs sharing an IMU, **and between the anchor row and every pair row that
touches the base IMU** — are forced, not modelled. Reproducing them is what
invariant I6 is about, and the anchor extends it: the anchor row is written in
terms of the base IMU's own measured rate, so it inherits that IMU's noise and is
correlated with every pair row through it.

Deriving the anchor row from the reference
------------------------------------------
The reference's base-IMU row has a zero joint Jacobian, so

    z_base = omega_base + b_base + v_base,     v_base ~ N(0, Sigma_base)

and its anchor row asserts

    J_leg qdot + omega_base = 0 + v_anchor,    v_anchor ~ N(0, Sigma_eps)

Eliminating `omega_base` between the two gives the port's anchor row

    z_base = -J_leg qdot + b_base + (v_anchor - v_base)

i.e. `H = [0 | -J_F | +I3 at the base bias]`, `z = omega_base_measured`, and a
noise covariance of `Sigma_eps + Sigma_base` — plus, when the base->foot chain
carries unfiltered joints, the `J_U diag(sigma_qd^2) J_U^T` congruence for the
measured velocities that moved into `z`.

That `+ Sigma_base` is the term CLAUDE.md §2's formula omits, and it is what this
test exists to catch.
"""
import numpy as np
import pytest

from invariant_estimation.jointKF import anchors as anchors_mod, measure
from invariant_estimation.jointKF.build import build_joint_kf
from invariant_estimation.jointKF.state import default_params

from . import _fixture as fx
from ._fixture import kinematic_tree
from ._oracles import (
    SHAPES,
    assert_all_close,
    reference_marginalized,
    reference_update,
    seeded_prior_update,
)

# Shape 3 is the two-pair shared-middle-IMU star, and its foot sits at the far
# end of the chain so every anchor-chain joint is a filter state (U is empty).
# That keeps this test about the base-IMU noise coupling rather than about the
# unfiltered-velocity congruence, which `test_bias_observability.py` already
# covers on its own.
SHAPE = SHAPES[3]


@pytest.fixture(scope="module")
def scene():
    f = fx.fixture(SHAPE["name"])
    build = build_joint_kf(
        kinematic_tree(f),
        imu_sites=list(f.imu_names),
        pairs=[tuple(p) for p in f.pairs],
        foot_sites=[f.foot_site],
    )
    return f, build


def oracle_inputs(f, build, q):
    """`(rot, jac, foot_jac)` in `reference_marginalized`'s convention."""
    ev = f.model.evaluate(q)
    names = list(f.model.site_names)
    site_rot = np.asarray(ev.site_rot)
    J_ang = np.asarray(ev.J_ang)

    base = names.index(f.imu_names[build.base_imu])
    R_base = site_rot[base]

    rot, jac = [], []
    for k, name in enumerate(f.imu_names):
        s = names.index(name)
        # R(base measurement frame -> IMU k frame), and the absolute angular
        # Jacobian base->IMU expressed in IMU k's frame.
        rot.append(site_rot[s].T @ R_base)
        jac.append(site_rot[s].T @ (J_ang[s] - J_ang[base])[:, build.dof_joint])
    foot = names.index(f.foot_site)
    foot_jac = R_base.T @ (J_ang[foot] - J_ang[base])[:, build.dof_joint]
    return np.stack(rot), np.stack(jac), foot_jac[None]


def port_posterior(f, build, params, q, gyros, sigma, trusted):
    """The port's stacked update, through the real modules."""
    import jax.numpy as jnp

    ev = f.model.evaluate(q)
    names = list(f.model.site_names)
    jac = anchors_mod.anchor_jacobians(
        build, ev.J_ang, ev.site_rot,
        base_site=names.index(f.imu_names[build.base_imu]),
        foot_sites=np.array([names.index(f.foot_site)]),
    )
    anchor = anchors_mod.anchor_block(
        build, params, jac,
        gyro_base=jnp.asarray(gyros[build.base_imu]),
        qd_unfiltered=jnp.zeros(build.anchor_unfiltered_mask.shape[1]),
        trusted_feet=jnp.asarray(trusted, dtype=jnp.float64),
    )
    J_rel, R_rel = measure.pair_frames(f.model, q)
    return measure.build_stacked(
        build, params, gyros=jnp.asarray(gyros),
        trusted_feet=jnp.asarray(trusted, dtype=jnp.float64),
        J_rel=J_rel, R_rel=R_rel, anchor=anchor,
    )


@pytest.mark.parametrize("trial", range(12))
def test_stacked_update_matches_the_nuisance_marginalized_reference(scene, trial):
    """12 trials, tol 1e-5 (Java `testStackedUpdateMatchesNuisanceMarginalizedReference`).

    Anisotropic `Sigma` in the 1e-2 band, for the conditioning reason recorded in
    `test_measurement.py`: the oracle's nuisance carries a zero information block,
    so its accuracy is set by `eps * cond(Lambda) ~ eps / sigma^2`.
    """
    f, build = scene
    rng = np.random.default_rng(90100 + trial)

    sigma = np.stack([np.diag(rng.uniform(2.0e-3, 1.8e-2, size=3)) for _ in range(f.m)])
    b = build_joint_kf(
        kinematic_tree(f), imu_sites=list(f.imu_names),
        pairs=[tuple(p) for p in f.pairs], foot_sites=[f.foot_site],
        gyro_sigma=lambda name, _s=sigma, _n=list(f.imu_names): _s[_n.index(name)],
    )
    params = default_params()
    q = f.random_q(rng)
    gyros = rng.normal(size=(f.m, 3)) * 0.4

    sm = port_posterior(f, b, params, q, gyros, sigma, trusted=[1.0])
    mu, P = seeded_prior_update(b.dim, 700.0 + trial)
    got = reference_update(mu, P, np.asarray(sm.H), np.asarray(sm.z), np.asarray(sm.R))

    rot, jac, foot_jac = oracle_inputs(f, b, q)
    want = reference_marginalized(
        mu, P, n=b.n_joints, imu_bias_col=b.bias_col, raw_gyro=gyros,
        imu_omega_base_rot=rot, imu_joint_jacobian=jac, imu_sigma=sigma,
        foot_joint_jacobian=foot_jac, anchor_var=params.anchor_var,
    )

    assert_all_close(got[0], want[0], 1.0e-5, f"trial {trial} posterior mean")
    assert_all_close(got[1], want[1], 1.0e-5, f"trial {trial} posterior covariance")
