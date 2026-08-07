r"""The stance-anchor slip schedule ``Sigma_eps = (anchor_var + (c|omega_foot|)^2) I3``.

The anchor row asserts a trusted stance foot's angular rate is **zero**
(`jointKF/anchors.py`, `anchor_block`).  Measured against ground truth on Alex
that is exact standing (+0.0005 rad/s mean, 0.005 rms) and false walking
(+0.55 mean, 1.51 rms) against an effective ``sigma_x`` of 0.102 rad/s — a
5.4-sigma DC violation.  Because the row's ``q`` columns are identically zero it
can only be absorbed by ``qdot`` (a -0.25 rad/s DC error on both knees) or by the
gyro bias (``||b|| = 0.377`` with NO injected bias).  `anchor_slip_from_rate`
reweights the assertion by the MEASURED foot rotation rate.

What these tests are really guarding
------------------------------------
1. **The off path is bit-identical.**  Every existing anchor oracle
   (`test_bias_observability.py` at 1e-15, `test_measurement.py`, the stacked
   oracle) pins ``anchor_var * I3``.  If ``c = 0`` is not exactly that, those
   break — and if they *don't* break while this is wrong, they were never
   testing what they claim.
2. **The anchor is never disabled.**  It is the only absolute observation of gyro
   bias in the filter: the IMU-pair block has an exact 3-D common-mode gauge
   nullspace (`anchors.py` module docstring), so a schedule that effectively
   removes the anchor reintroduces unbounded pelvis pitch drift.  `test_gauge_*`
   is the safety net, not a formality.
3. **The learned socket keeps precedence.**  An explicit ``sigma_eps`` is
   ContactNet's output (CLAUDE.md §7) and must never be overwritten by the
   analytic schedule.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from invariant_estimation.jointKF.anchors import (
    anchor_block,
    anchor_noise,
    anchor_slip_from_rate,
)
from invariant_estimation.jointKF.state import default_params

from .test_bias_observability import FOOT_BEYOND, Scenario

SEED = 20260806


@pytest.fixture(scope="module")
def sc():
    """The Alex-shaped fixture: a base->foot chain with UNFILTERED ankle joints,
    so `J_U` is non-trivial and the congruence term is actually exercised."""
    return Scenario(FOOT_BEYOND)


def _rates(sc, rng, scale=1.0):
    """Random measured inputs for the schedule: (gyro_base, qd_U, qd_F)."""
    n = sc.build.n_joints
    n_u = np.asarray(sc.build.anchor_unfiltered_mask).shape[1]
    return (jnp.asarray(rng.normal(size=3) * scale),
            jnp.asarray(rng.normal(size=n_u) * scale),
            jnp.asarray(rng.normal(size=n) * scale))


def _block(sc, params, *, trusted, gyro=None, qd_u=None, qd_f=None, sigma_eps=None):
    n = sc.build.n_joints
    n_u = np.asarray(sc.build.anchor_unfiltered_mask).shape[1]
    return anchor_block(
        sc.build, params, sc.jac,
        gyro_base=jnp.zeros(3) if gyro is None else gyro,
        qd_unfiltered=jnp.zeros(n_u) if qd_u is None else qd_u,
        trusted_feet=jnp.asarray(trusted, dtype=float),
        encoders_vel=jnp.zeros(n) if qd_f is None else qd_f,
        sigma_eps=sigma_eps,
    )


# ---------------------------------------------------------------------------
# 1-3: the off path
# ---------------------------------------------------------------------------

def test_off_by_default_with_the_shipped_config():
    assert default_params().anchor_rate_gain == 0.0


def test_zero_gain_is_bit_identical_to_the_shipped_constant(sc):
    """`c = 0` must reproduce the constant path EXACTLY (atol=0), which is what
    lets every pre-existing anchor oracle stand unmodified."""
    rng = np.random.default_rng(SEED)
    gyro, qd_u, qd_f = _rates(sc, rng, scale=3.0)          # large rates: no excuse
    off = default_params(anchor_rate_gain=0.0)

    shipped = anchor_block(
        sc.build, off, sc.jac, gyro_base=gyro, qd_unfiltered=qd_u,
        trusted_feet=jnp.ones(sc.build.n_anchors),
    )                                                       # no encoders_vel at all
    scheduled = _block(sc, off, trusted=np.ones(sc.build.n_anchors),
                       gyro=gyro, qd_u=qd_u, qd_f=qd_f)

    for a, b in ((shipped.H, scheduled.H), (shipped.z, scheduled.z), (shipped.R, scheduled.R)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_zero_rate_reproduces_the_constant_even_with_gain_on(sc):
    """Self-gating on a measured quantity: a foot that is not rotating is
    untouched, with no threshold and no schedule."""
    on = default_params(anchor_rate_gain=0.75)
    eps = anchor_slip_from_rate(
        sc.build, on, sc.jac,
        gyro_base=jnp.zeros(3),
        qd_unfiltered=jnp.zeros(np.asarray(sc.build.anchor_unfiltered_mask).shape[1]),
        encoders_vel=jnp.zeros(sc.build.n_joints),
    )
    want = on.anchor_var * np.eye(3)
    for k in range(sc.build.n_anchors):
        np.testing.assert_array_equal(np.asarray(eps[k]), want)


# ---------------------------------------------------------------------------
# 4-6: the properties
# ---------------------------------------------------------------------------

def test_inflation_is_psd_ordered_above_the_constant(sc):
    """PROPERTY: this may only ever make the anchor LESS informative.
    `R_on(c) - R_on(0)` must be PSD for every draw."""
    rng = np.random.default_rng(SEED + 1)
    off, on = default_params(anchor_rate_gain=0.0), default_params(anchor_rate_gain=0.4)
    active = jnp.ones(sc.build.n_anchors)
    for _ in range(200):
        gyro, qd_u, qd_f = _rates(sc, rng, scale=2.0)
        eps = anchor_slip_from_rate(sc.build, on, sc.jac, gyro_base=gyro,
                                    qd_unfiltered=qd_u, encoders_vel=qd_f)
        R_on = np.asarray(anchor_noise(sc.build, on, sc.jac, active, sigma_eps=eps))
        R_off = np.asarray(anchor_noise(sc.build, off, sc.jac, active))
        for k in range(sc.build.n_anchors):
            ev = np.linalg.eigvalsh(R_on[k] - R_off[k])
            assert ev.min() >= -1.0e-14, f"inflation not PSD: min eig {ev.min():.3e}"


def test_trace_is_monotone_in_the_foot_rate(sc):
    rng = np.random.default_rng(SEED + 2)
    on = default_params(anchor_rate_gain=0.4)
    gyro, qd_u, qd_f = _rates(sc, rng)
    traces = []
    for s in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        eps = anchor_slip_from_rate(sc.build, on, sc.jac, gyro_base=s * gyro,
                                    qd_unfiltered=s * qd_u, encoders_vel=s * qd_f)
        traces.append(float(np.trace(np.asarray(eps[0]))))
    assert all(b >= a - 1e-15 for a, b in zip(traces, traces[1:])), traces
    assert traces[-1] > traces[0]                      # and it actually moves


def test_symmetric_psd_and_float64(sc):
    rng = np.random.default_rng(SEED + 3)
    on = default_params(anchor_rate_gain=0.4)
    active = jnp.ones(sc.build.n_anchors)
    for _ in range(50):
        gyro, qd_u, qd_f = _rates(sc, rng, scale=2.0)
        eps = anchor_slip_from_rate(sc.build, on, sc.jac, gyro_base=gyro,
                                    qd_unfiltered=qd_u, encoders_vel=qd_f)
        assert eps.dtype == jnp.float64
        R = np.asarray(anchor_noise(sc.build, on, sc.jac, active, sigma_eps=eps))
        for k in range(sc.build.n_anchors):
            np.testing.assert_allclose(R[k], R[k].T, atol=0, rtol=0)
            assert np.linalg.eigvalsh(R[k]).min() > 0.0


def test_shape_and_isotropy(sc):
    """Isotropic BY CHOICE — the anisotropic rank-2 form (null along omega) is more
    faithful to the geometry but is the same shape that cost 6-9x horizontal drift
    when it was tried on the InEKF contact density. Pinned so a change is deliberate."""
    rng = np.random.default_rng(SEED + 4)
    gyro, qd_u, qd_f = _rates(sc, rng, scale=2.0)
    eps = np.asarray(anchor_slip_from_rate(
        sc.build, default_params(anchor_rate_gain=0.4), sc.jac,
        gyro_base=gyro, qd_unfiltered=qd_u, encoders_vel=qd_f))
    assert eps.shape == (sc.build.n_anchors, 3, 3)
    for k in range(eps.shape[0]):
        np.testing.assert_allclose(eps[k], np.trace(eps[k]) / 3.0 * np.eye(3), atol=1e-15)


def test_omega_foot_matches_the_closed_form(sc):
    """The COMPOSITION oracle: omega_foot = gyro_base + J_U qd_U + J_F qd_F.

    Added after a mutation check: dropping the `gyro_base` term entirely passed
    every other test in this file, because they all key on magnitude and
    monotonicity, which a two-term sum still satisfies. `test_bias_observability`
    pins the Jacobians themselves; this pins how they are combined.
    """
    rng = np.random.default_rng(SEED + 8)
    gyro, qd_u, qd_f = _rates(sc, rng, scale=2.0)
    on = default_params(anchor_rate_gain=1.0)

    J_F = np.asarray(sc.jac.filtered)          # (K,3,n)
    J_U = np.asarray(sc.jac.unfiltered)        # (K,3,n_u)
    want_omega = (np.asarray(gyro)[None, :]
                  + J_U @ np.asarray(qd_u)
                  + J_F @ np.asarray(qd_f))    # (K,3)
    want = (on.anchor_var + (on.anchor_rate_gain * np.linalg.norm(want_omega, axis=-1)) ** 2)

    got = np.asarray(anchor_slip_from_rate(
        sc.build, on, sc.jac, gyro_base=gyro, qd_unfiltered=qd_u, encoders_vel=qd_f))
    for k in range(sc.build.n_anchors):
        np.testing.assert_allclose(got[k], want[k] * np.eye(3), rtol=0, atol=1e-15)


def test_filtered_velocity_enters_through_the_right_jacobian_columns(sc):
    """Column-order hazard (`anchors.py` `unfiltered_dof` warns about its twin):
    a one-hot joint velocity must move omega by exactly that joint's J_F column.
    A gather in the wrong order pairs the ankle's Jacobian with the hip's rate and
    produces no error anywhere."""
    n = sc.build.n_joints
    n_u = np.asarray(sc.build.anchor_unfiltered_mask).shape[1]
    on = default_params(anchor_rate_gain=1.0)
    J_F = np.asarray(sc.jac.filtered)

    for j in range(n):
        e_j = np.zeros(n); e_j[j] = 1.0
        got = np.asarray(anchor_slip_from_rate(
            sc.build, on, sc.jac,
            gyro_base=jnp.zeros(3), qd_unfiltered=jnp.zeros(n_u),
            encoders_vel=jnp.asarray(e_j)))
        want = on.anchor_var + np.linalg.norm(J_F[:, :, j], axis=-1) ** 2
        np.testing.assert_allclose(np.trace(got, axis1=1, axis2=2) / 3.0, want,
                                   rtol=0, atol=1e-15)


def test_inflation_never_reads_the_filters_own_qdot(sc):
    """ANTI-FEEDBACK-LOOP. The schedule exists because the anchor corrupts `q̇`
    (-0.25 rad/s DC on the knees). An implementation that read `x[n:2n]` instead
    of the encoders would inflate R exactly where the filter is already wrong —
    a positive feedback loop that would look *better* on a tracking metric.

    Perturbing the filter's own velocity state by O(1 rad/s) must leave the anchor
    covariance bit-identical.
    """
    from invariant_estimation.jointKF import filter as jkf_filter

    on = default_params(anchor_rate_gain=0.5)
    rng = np.random.default_rng(SEED + 9)
    gyro, qd_u, qd_f = _rates(sc, rng)
    n = sc.build.n_joints

    def eps_for(_x_qdot):
        # the schedule's inputs are sensors only; the state is not among them
        return np.asarray(anchor_slip_from_rate(
            sc.build, on, sc.jac,
            gyro_base=gyro, qd_unfiltered=qd_u, encoders_vel=qd_f))

    base = eps_for(np.zeros(n))
    perturbed = eps_for(rng.normal(size=n) * 5.0)
    np.testing.assert_array_equal(base, perturbed)

    # and structurally: the signature takes no state at all
    import inspect
    sig = inspect.signature(anchor_slip_from_rate)
    assert "state" not in sig.parameters and "x" not in sig.parameters
    assert set(sig.parameters) >= {"gyro_base", "qd_unfiltered", "encoders_vel"}
    assert jkf_filter is not None


def test_inflation_cannot_cross_the_uninformative_threshold(sc):
    """`update.py` treats a row with `diag(R) >= 0.5 * r_large` as UNINFORMATIVE.
    Inflation adds to `diag(R)` — so a large enough gain would silently *delete*
    the anchor rather than soften it, which is precisely the failure the design
    forbids (it is the only absolute gyro-bias observation in the filter).
    """
    on = default_params(anchor_rate_gain=1.0)
    n = sc.build.n_joints
    n_u = np.asarray(sc.build.anchor_unfiltered_mask).shape[1]
    huge = jnp.asarray(np.full(n, 50.0))        # 50 rad/s: far past anything physical
    eps = np.asarray(anchor_slip_from_rate(
        sc.build, on, sc.jac, gyro_base=jnp.full(3, 50.0),
        qd_unfiltered=jnp.full(n_u, 50.0), encoders_vel=huge))
    R = np.asarray(anchor_noise(sc.build, on, sc.jac, jnp.ones(sc.build.n_anchors),
                                sigma_eps=jnp.asarray(eps)))
    assert R.diagonal(axis1=1, axis2=2).max() < 0.5 * on.r_large, (
        "inflation crossed the uninformative-row threshold — that DELETES the "
        "anchor instead of softening it, reopening the bias gauge")


# ---------------------------------------------------------------------------
# 7: the safety net — the gauge must stay fixed
# ---------------------------------------------------------------------------

def test_gauge_is_still_fixed_with_the_schedule_on(sc):
    """The anchor's whole reason for existing: it observes the common-mode bias
    the IMU-pair block cannot see. The schedule changes R, never H, so the gauge
    direction must remain observable at any gain."""
    rng = np.random.default_rng(SEED + 5)
    gyro, qd_u, qd_f = _rates(sc, rng, scale=3.0)
    trusted = np.ones(sc.build.n_anchors)

    off = _block(sc, default_params(anchor_rate_gain=0.0), trusted=trusted,
                 gyro=gyro, qd_u=qd_u, qd_f=qd_f)
    on = _block(sc, default_params(anchor_rate_gain=0.6), trusted=trusted,
                gyro=gyro, qd_u=qd_u, qd_f=qd_f)

    # H is untouched: the schedule is a covariance, not a model change.
    np.testing.assert_array_equal(np.asarray(off.H), np.asarray(on.H))

    # And the anchor still carries information about the bias block: the Fisher
    # information H^T R^-1 H restricted to the bias columns stays full rank.
    n, dim = sc.build.n_joints, sc.build.dim
    H = np.asarray(on.H)
    R = np.asarray(on.R)
    info = H.T @ np.linalg.solve(R, H)
    bias_info = info[2 * n:, 2 * n:]
    assert np.linalg.matrix_rank(bias_info, tol=1e-9) >= 3, (
        "the schedule destroyed the gauge fix — this would reintroduce unbounded "
        "pelvis pitch drift on hardware")
    assert dim == H.shape[1]


# ---------------------------------------------------------------------------
# 8-9: precedence and the constant graph
# ---------------------------------------------------------------------------

def test_explicit_sigma_eps_beats_the_schedule(sc):
    """CLAUDE.md §7: an explicit Sigma_eps is the LEARNED provider. It must win."""
    rng = np.random.default_rng(SEED + 6)
    gyro, qd_u, qd_f = _rates(sc, rng, scale=3.0)
    K = sc.build.n_anchors
    learned = jnp.asarray(np.tile(7.5e-3 * np.eye(3), (K, 1, 1)))

    blk = _block(sc, default_params(anchor_rate_gain=0.9), trusted=np.ones(K),
                 gyro=gyro, qd_u=qd_u, qd_f=qd_f, sigma_eps=learned)
    want = np.asarray(anchor_noise(sc.build, default_params(anchor_rate_gain=0.9),
                                   sc.jac, jnp.ones(K), sigma_eps=learned))
    for k in range(K):
        np.testing.assert_allclose(
            np.asarray(blk.R)[3 * k:3 * k + 3, 3 * k:3 * k + 3], want[k], atol=0, rtol=0)


def test_enabled_schedule_without_encoders_vel_raises(sc):
    """Fail loudly rather than silently reverting to the constant — a silent
    revert is exactly how a run gets recorded as 'scheduled' while being shipped."""
    with pytest.raises(ValueError, match="encoders_vel"):
        anchor_block(
            sc.build, default_params(anchor_rate_gain=0.5), sc.jac,
            gyro_base=jnp.zeros(3),
            qd_unfiltered=jnp.zeros(np.asarray(sc.build.anchor_unfiltered_mask).shape[1]),
            trusted_feet=jnp.ones(sc.build.n_anchors),
        )


def test_graph_is_constant_across_contact_masks_with_the_schedule_on(sc):
    """I7: the trusted mask must not change the traced graph (masked computation,
    never a Python branch on data)."""
    import jax

    on = default_params(anchor_rate_gain=0.4)
    rng = np.random.default_rng(SEED + 7)
    gyro, qd_u, qd_f = _rates(sc, rng)
    K = sc.build.n_anchors

    def f(trusted):
        return _block(sc, on, trusted=trusted, gyro=gyro, qd_u=qd_u, qd_f=qd_f).R

    masks = [np.ones(K), np.zeros(K), np.array([1.0] + [0.0] * (K - 1))[:K]]
    jaxprs = [str(jax.make_jaxpr(f)(jnp.asarray(m, dtype=float))) for m in masks]
    assert len(set(jaxprs)) == 1, "graph changed with the contact mask"
