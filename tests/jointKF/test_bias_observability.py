r"""Port of `JointLevelKFBiasObservabilityTest` (4 tests) — plus the structural
observability tests `JOINTKF_PORT_PLAN.md` §5 Phase 2 requires on top of them.

What the Java class is really about
-----------------------------------
Not "does the anchor code run".  The claim is a *rank* statement about the
stacked measurement:

* the IMU-pair rows see gyro bias only as **rotated differences**, so the
  common-mode direction ``delta b_i = {}^{i}R_{W} beta`` is exactly in their
  nullspace — 3 dimensions of bias that no amount of relative-gyro data ever
  observes;
* the stance-anchor row's ``+I3`` on the base IMU's bias columns is the **only**
  absolute bias observation in the whole filter, and it fixes exactly those 3.

The map's four tests probe that with one ``beta``.  Per the plan, this file also
asserts it structurally (rank / nullspace dimension) and behaviourally (the
gauge-direction variance grows without bound with no anchor and converges with
one) — the latter is the property that would silently vanish if the ``+I3`` were
placed on the wrong IMU, since a misplaced identity still *looks* like an
absolute observation.

Independence of the oracles (`CONTRACT_CARD.md` §7)
---------------------------------------------------
The pair rows, the mixing operator ``L``, ``R_g = L Sigma Lᵀ`` and
``gauge_direction`` are all built here in NumPy from their closed forms.  Nothing
in this file calls `jointKF.measure` (another agent's module, and the point is
not to test it) — the pair rows are *context* that makes the anchor claim
meaningful, so they must come from an independent hand.

Kinematics come from the MJX seam (`tests/jointKF/_fixture.py`), which is not the
code under test.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.jointKF.anchors import (
    anchor_block,
    anchor_jacobians,
    unfiltered_dof,
)
from invariant_estimation.jointKF.build import KinematicTree, build_joint_kf
from invariant_estimation.jointKF.predict import build_transition, predict
from invariant_estimation.jointKF.process import build_process_noise
from invariant_estimation.jointKF.state import JointKFState, default_params, init_state
from invariant_estimation.jointKF.update import joseph_update

from . import _fixture as fx
from ._fixture import kinematic_tree
from ._oracles import SHAPES, assert_positive_semidefinite, reference_update

#: Java `JointLevelKFBiasObservabilityTest.SEED`.  Kept even though the port's
#: geometry stream is `_fixture.CHAIN_SEED`: it names the Java scenario a reader
#: is comparing against.
SEED = 20260712

#: The map's `beta` values, verbatim.
BETA = np.array([0.013, -0.007, 0.021])
BETA_ANKLE = np.array([0.011, 0.004, -0.017])

#: `singlePairFootBeyondIMUs(SEED, 10, 1, 5, 9)` — the Alex ankle case.  IMUs on
#: links 1 and 5 (so joints 2..5 are filter states) and the foot on link 9, so
#: joints 6..9 lie on the base->foot chain but are NOT states.  `_fixture` always
#: puts the foot site on the last link, which is link 9 for a 10-joint chain.
FOOT_BEYOND = {"name": "foot_beyond", "chain": 10, "imus": (1, 5),
               "pairs": ((0, 1),), "n": 4, "m": 2}

#: `singlePair(SEED, 10, 1, 9)` — the all-filtered comparison for the
#: anchor-covariance test.  Identical chain length, foot on the child IMU's own
#: link, so the base->foot chain has no unfiltered joints at all.
ALL_FILTERED = SHAPES[0]


# ---------------------------------------------------------------------------
# Fixture plumbing: MJX model -> the graph description `build.py` consumes
# ---------------------------------------------------------------------------

# `kinematic_tree` now lives in `_fixture.py` (shared with test_filter.py).
class Scenario:
    """One fixture, built, evaluated at a configuration, with its oracles ready.

    Bundled because every test needs the same five things (build, `q`, site
    rotations, world Jacobians, pair rows) and re-deriving them per test is how
    two tests end up silently disagreeing about the configuration.
    """

    def __init__(self, shape: dict, *, seed: int = SEED):
        self.fixture = (
            fx.fixture(shape["name"]) if shape["name"] in _NAMES else _extra_fixture(shape["name"])
        )
        self.shape = shape
        self.build = build_joint_kf(
            kinematic_tree(self.fixture),
            imu_sites=list(self.fixture.imu_names),
            pairs=[tuple(p) for p in self.fixture.pairs],
            foot_sites=["foot"],
            base_imu=0,
            use_mass_matrix=False,
        )
        self.params = default_params()

        rng = np.random.default_rng(seed + 17)
        self.q = self.fixture.random_q(rng)
        ev = self.fixture.model.evaluate(jnp.asarray(self.q))
        self.J_world = np.asarray(ev.J_ang)          # (n_sites, 3, nv)
        self.site_rot = np.asarray(ev.site_rot)      # (n_sites, 3, 3)
        self.foot_site = np.array([self.fixture.model.site_names.index("foot")])
        self.jac = anchor_jacobians(
            self.build, jnp.asarray(self.J_world), jnp.asarray(self.site_rot),
            base_site=0, foot_sites=jnp.asarray(self.foot_site),
        )
        self.H_pairs, self.R_pairs = pair_rows(self.build, self.J_world, self.site_rot)

    # -- convenience --------------------------------------------------------

    def block(self, trusted, *, gyro_base=None, qd_u=None):
        """The anchor block at a given trusted-feet mask."""
        n_u = np.asarray(self.build.anchor_unfiltered_mask).shape[1]
        return anchor_block(
            self.build, self.params, self.jac,
            gyro_base=jnp.zeros(3) if gyro_base is None else jnp.asarray(gyro_base),
            qd_unfiltered=jnp.zeros(n_u) if qd_u is None else jnp.asarray(qd_u),
            trusted_feet=jnp.asarray(trusted, dtype=float),
        )

    def stacked(self, trusted, **kw):
        """`H` of the full stacked measurement: pair rows then anchor rows."""
        blk = self.block(trusted, **kw)
        return np.vstack([self.H_pairs, np.asarray(blk.H)]), blk

    def gauge(self, beta) -> np.ndarray:
        return gauge_direction(self.build, self.site_rot, beta)


_NAMES = {s["name"] for s in SHAPES}
_EXTRA = {FOOT_BEYOND["name"]: FOOT_BEYOND}


@lru_cache(maxsize=None)
def _extra_fixture(name: str) -> fx.ChainFixture:
    """Cache for shapes outside `_fixture.SHAPES` — building an `mjx.Model` costs
    seconds and `FOOT_BEYOND` is used by most of this file."""
    return fx.build_fixture(_EXTRA[name])


# ---------------------------------------------------------------------------
# Independent NumPy oracles
# ---------------------------------------------------------------------------

def pair_rows(build, J_world, site_rot) -> tuple[np.ndarray, np.ndarray]:
    r"""The stacked IMU-pair block `(H_g, R_g)` — written from the closed form.

    For pair ``e = (p, c)`` the measurement is
    ``z = omega_c - {}^{c}R_{p} omega_p``, so::

        H[e] = [ 0_q | {}^{W}R_c^T (J_c - J_p) | -{}^{c}R_p at parent bias
                                               | +I3        at child bias ]

    The bias columns are the mixing operator ``L`` (invariant I6) and
    ``R_g = L Sigma Lᵀ`` with ``Sigma`` the block-diagonal per-IMU gyro
    covariance.  Deliberately **not** imported from `jointKF.measure`: this file
    must be able to make the gauge claim without borrowing the implementation
    whose structure the claim depends on.
    """
    n, m, E = build.n_joints, build.n_imus, build.n_pairs
    dim = build.dim
    H = np.zeros((3 * E, dim))
    L = np.zeros((3 * E, 3 * m))
    dof = np.asarray(build.dof_joint)

    for e in range(E):
        p, c = int(build.pair_parent[e]), int(build.pair_child[e])
        r = 3 * e
        R_c, R_p = site_rot[c], site_rot[p]
        J_rel = R_c.T @ (J_world[c] - J_world[p])[:, dof]
        H[r:r + 3, n:2 * n] = J_rel * np.asarray(build.pair_velocity_mask)[e][None, :]
        L[r:r + 3, 3 * c:3 * c + 3] = np.eye(3)
        L[r:r + 3, 3 * p:3 * p + 3] = -R_c.T @ R_p

    H[:, 2 * n:] = L
    Sigma = np.zeros((3 * m, 3 * m))
    for k in range(m):
        Sigma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = np.asarray(build.gyro_sigma)[k]
    return H, L @ Sigma @ L.T


def gauge_direction(build, site_rot, beta) -> np.ndarray:
    r"""Java `gaugeDirection(f, betaWorld)`: ``dx = (0, 0, db)``, ``db_i = {}^{i}R_W beta``.

    One common bias expressed in the world, written in each IMU's own frame —
    which is the frame `state.py` stores ``b_omega`` in.  This is the direction
    the relative-gyro rows are blind to.
    """
    dx = np.zeros(build.dim)
    for k in range(build.n_imus):
        col = build.bias_col(k)
        dx[col:col + 3] = site_rot[k].T @ np.asarray(beta, dtype=float)
    return dx


def gauge_basis(build, site_rot) -> np.ndarray:
    """The full 3-D gauge subspace as columns, `(dim, 3)` — for the rank tests."""
    return np.stack([gauge_direction(build, site_rot, e) for e in np.eye(3)], axis=1)


# ---------------------------------------------------------------------------
# The four ported tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [*SHAPES, FOOT_BEYOND], ids=lambda s: s["name"])
def test_common_mode_bias_is_unobservable_without_anchors(shape):
    """No trusted foot ⇒ the common-mode gauge is in the nullspace of `H`.

    Map: `testCommonModeBiasIsUnobservableWithoutAnchors`, tolerance 1e-9.

    Two things must hold for this to pass, and the test cannot tell them apart —
    which is fine, because both are required: the pair rows must cancel the
    rotated bias difference (oracle algebra), and the inactive anchor rows must
    contribute **nothing** to `H`.  The second is what `anchors.py` owns: a
    fixed-shape block that merely inflated `R` while leaving `+I3` standing in
    `H` would report the gauge as observed here.
    """
    sc = Scenario(shape)
    H, blk = sc.stacked(np.zeros(sc.build.n_anchors))
    assert float(blk.n_active) == 0.0
    assert np.linalg.norm(H @ sc.gauge(BETA)) < 1.0e-9


@pytest.mark.parametrize("shape", [*SHAPES, FOOT_BEYOND], ids=lambda s: s["name"])
def test_stance_anchor_fixes_the_gauge(shape):
    """All feet trusted ⇒ `||H gauge|| == ||beta||`.

    Map: `testStanceAnchorFixesTheGauge`, tolerance 1e-9.

    The equality (not merely "> 0") is the sharp part: it says the anchor reads
    the common-mode bias back through an **orthonormal** map, i.e. exactly the
    ``+I3`` on the base IMU's bias columns and nothing else.  Any extra bias
    coupling, or the identity landing on a different IMU with a different
    attitude, changes the norm.
    """
    sc = Scenario(shape)
    H, blk = sc.stacked(np.ones(sc.build.n_anchors))
    assert float(blk.n_active) == sc.build.n_anchors
    got = np.linalg.norm(H @ sc.gauge(BETA))
    assert abs(got - np.linalg.norm(BETA)) < 1.0e-9


def test_anchor_is_usable_when_chain_has_unfiltered_joints():
    """The Alex ankle regression — map `testAnchorIsUsableWhenChainHasUnfilteredJoints`.

    Joints 6..9 lie on the base->foot chain but are not filter states.  The
    anchor must still be *active* (an unfiltered joint is a known input, not a
    reason to drop the row) and must still fix the gauge.  Untrusting the foot
    must reopen it — proving the gauge is fixed by the **anchor**, not by some
    incidental property of this topology.
    """
    sc = Scenario(FOOT_BEYOND)
    assert np.asarray(sc.build.anchor_unfiltered_mask).sum() == 4, "joints 6..9 are the U split"

    H, blk = sc.stacked(np.ones(1))
    assert float(blk.n_active) == 1.0
    gauge = sc.gauge(BETA_ANKLE)
    assert abs(np.linalg.norm(H @ gauge) - np.linalg.norm(BETA_ANKLE)) < 1.0e-9

    H0, blk0 = sc.stacked(np.zeros(1))
    assert float(blk0.n_active) == 0.0
    assert np.linalg.norm(H0 @ gauge) < 1.0e-9


def test_anchor_covariance_includes_unfiltered_joint_noise():
    """`R_anchor = Sigma_eps + J_U diag(sigma_qd^2) J_U^T` — map
    `testAnchorCovarianceIncludesUnfilteredJointNoise`.

    ``Sigma_eps`` alone contributes ``3 * 4e-4 = 1.2e-3`` to the trace, so the
    ``> 1e-2`` threshold can only be met by the input-noise congruence over the
    four unfiltered ankle velocities at 0.1 rad/s.  The second assertion —
    all-filtered chain gives a *tighter* anchor — is what stops the first from
    being satisfiable by any old inflation: the congruence must be driven by the
    U split specifically.
    """
    sc = Scenario(FOOT_BEYOND)
    R = np.asarray(sc.block(np.ones(1)).R)
    anchor_trace = float(np.trace(R[-3:, -3:]))

    clean = Scenario(ALL_FILTERED)
    assert np.asarray(clean.build.anchor_unfiltered_mask).size == 0, \
        "the comparison fixture must have NO unfiltered chain joints"
    clean_trace = float(np.trace(np.asarray(clean.block(np.ones(1)).R)[-3:, -3:]))

    assert anchor_trace > 1.0e-2, (
        f"anchor trace {anchor_trace:.3e} — Sigma_eps alone is 1.2e-3, so the "
        "J_U diag(sigma_qd^2) J_U^T congruence is missing or too small"
    )
    assert clean_trace < anchor_trace
    assert abs(clean_trace - 3 * sc.params.anchor_var) < 1.0e-12, \
        "with an empty U split the anchor noise must be exactly Sigma_eps"


# ---------------------------------------------------------------------------
# Beyond the map — the observability claim, structurally (PLAN §5, Phase 2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [*SHAPES, FOOT_BEYOND], ids=lambda s: s["name"])
def test_gauge_nullspace_is_exactly_three_dimensional_and_the_anchor_closes_it(shape):
    """Rank statement, not a single probe vector.

    Without an anchor the whole 3-D gauge subspace is annihilated by `H`; with
    one anchor active the rank of `H` rises by exactly 3 and `H G` becomes
    orthonormal (its singular values are all 1, because each gauge column maps to
    ``{}^{b}R_W e_i``).  A one-vector probe would pass with an ``+I3`` degraded to
    a rank-1 or rank-2 block; this will not.
    """
    sc = Scenario(shape)
    G = gauge_basis(sc.build, sc.site_rot)
    assert np.linalg.matrix_rank(G) == 3

    H0, _ = sc.stacked(np.zeros(sc.build.n_anchors))
    H1, _ = sc.stacked(np.ones(sc.build.n_anchors))

    assert np.max(np.abs(H0 @ G)) < 1.0e-9, "the gauge subspace must be in ker(H) with no anchor"
    r0 = np.linalg.matrix_rank(H0, tol=1e-9)
    r1 = np.linalg.matrix_rank(H1, tol=1e-9)
    assert r1 == r0 + 3, f"one anchor must add exactly 3 to rank(H); got {r0} -> {r1}"

    sv = np.linalg.svd(H1 @ G, compute_uv=False)
    assert np.allclose(sv, 1.0, atol=1e-9), (
        f"H G must be orthonormal (the +I3 read back through ^bR_W); singular values {sv}"
    )


def test_identity_block_sits_on_the_base_imus_bias_columns_only():
    """Structural: the ``+I3`` is at `bias_col(base_imu)` and every other bias
    column of the anchor row is exactly zero.

    This is the placement whose failure the behavioural tests below detect only
    indirectly — worth asserting once, directly.
    """
    sc = Scenario(FOOT_BEYOND)
    H = np.asarray(sc.block(np.ones(1)).H)
    n, base = sc.build.n_joints, sc.build.base_imu
    col = sc.build.bias_col(base)
    bias = H[:, 2 * n:]
    assert np.allclose(H[:, col:col + 3], np.eye(3), atol=0.0)
    other = np.delete(bias, np.arange(col - 2 * n, col - 2 * n + 3), axis=1)
    assert np.all(other == 0.0), "no anchor row may touch a non-base IMU's bias"
    assert np.all(H[:, :n] == 0.0), "the anchor observes rates, never positions"


def test_gauge_variance_grows_without_bound_with_no_anchor_and_converges_with_one():
    """The practical consequence — and the reason the anchor exists at all.

    Under repeated predict/update the bias random walk injects
    ``dt * imu_bias_process_var`` into every bias direction each tick.  With no
    anchor the stacked update removes **none** of it along the gauge, so the
    gauge variance ramps linearly and forever; the joint KF then hands the InEKF
    an ``omega_bar`` wrong by a slowly wandering constant, which the InEKF
    integrates into attitude.  With one anchor active the same loop reaches a
    steady state.

    This is the test that would still catch a ``+I3`` placed on the *wrong* IMU
    (which keeps every structural property above intact for a single-pair chain
    only by accident) and a residual/`R` masking bug that leaves the anchor
    silently switched off.
    """
    sc = Scenario(FOOT_BEYOND)
    F = build_transition(sc.build, sc.params)
    Q = build_process_noise(sc.build, sc.params, None)
    g = jnp.asarray(sc.gauge(BETA / np.linalg.norm(BETA)))

    # Encoder rows are part of the loop on purpose.  Without them `P_qdqd`
    # diverges under the measurement-free double integrator, the anchor's own
    # `-J_F q_dot` columns swamp its `+I3`, and the anchored case fails to
    # converge for a reason that has nothing to do with the gauge.  That is the
    # filter the robot actually runs, and it is the only setting in which
    # "converges" is a statement about the anchor.
    n = sc.build.n_joints
    H_enc = np.zeros((n, sc.build.dim))
    H_enc[:, :n] = np.eye(n)
    R_enc = np.diag(np.asarray(sc.build.encoder_var))
    gate_off = default_params(cond_s_max=1.0e30)

    def run(trusted, ticks):
        blk = sc.block(trusted)
        H = jnp.vstack([jnp.asarray(H_enc), jnp.asarray(sc.H_pairs), blk.H])
        R = jax.scipy.linalg.block_diag(jnp.asarray(R_enc), jnp.asarray(sc.R_pairs), blk.R)
        z = jnp.zeros(H.shape[0])

        # `cond_s_max` is lifted for this test ONLY, so that the untrusted run
        # genuinely *applies* its encoder+gyro update. At the shipped 1e9 the
        # inactive anchor's `r_large` would gate the whole update out (the strict
        # xfail below), and the "free" ramp would then be pure prediction -- a
        # growth curve that proves nothing about observability.
        def step(state, _):
            state = predict(state, F, Q)
            state, _info = joseph_update(state, H, z, R, gate_off)
            return state, g @ state.P @ g

        state = init_state(sc.build, sc.params, jnp.asarray(sc.q))
        _, trace = jax.lax.scan(step, state, None, length=ticks)
        return np.asarray(trace)

    # 20 s of filter time. The anchored loop settles in ~2 s; the horizon is set
    # by wanting the *converged* plateau to be unmistakable rather than by the
    # free ramp, which is visible immediately.
    ticks = 20_000
    free = run(np.zeros(1), ticks)
    held = run(np.ones(1), ticks)

    # Free: strictly increasing, at the random-walk rate, with no sign of a knee.
    assert np.all(np.diff(free) > 0.0), "no anchor ⇒ the gauge variance never shrinks"
    rate = sc.params.dt * sc.params.imu_bias_process_var * sc.build.n_imus
    # `free[i]` is the variance AFTER tick `i`, so the span from index `a` to the
    # last index carries `ticks - 1 - a` increments, not `ticks - a`. Getting
    # this wrong shows up as a clean 1/2000 bias -- which is precisely how it was
    # found, rather than by loosening the bound.
    first = ticks // 2
    late = (free[-1] - free[first]) / (ticks - 1 - first)
    assert abs(late / rate - 1.0) < 1.0e-9, (
        f"late growth {late:.3e}/tick should be the pure random walk {rate:.3e}/tick"
    )

    # Held: a genuine fixed point, not merely a slower ramp. The free curve moves
    # 2e-3 over the same final decile.
    drift = held[-1] - held[-ticks // 10]
    assert abs(drift) < 1.0e-9, f"anchored gauge variance still drifting by {drift:.3e}"
    assert held[-1] < 0.05 * free[-1], \
        f"anchored gauge variance {held[-1]:.3e} not bounded below the free ramp {free[-1]:.3e}"
    assert held[-1] < held[0], "the anchor must reduce the gauge variance, not merely cap it"


# ---------------------------------------------------------------------------
# Beyond the map — the masking rules (CLAUDE.md §4), which live in THIS module
# ---------------------------------------------------------------------------

def test_masked_inactive_anchor_equals_dropping_its_rows_exactly():
    """A masked anchor must be a no-op, not an approximate one.

    Java's stacked measurement simply has no rows for an untrusted foot.  The
    port keeps the rows (fixed shape, invariant I2) and masks them, so the two
    are only the same filter if the masked posterior equals the row-excluded
    posterior **exactly**.  Zeroing `H` and the residual is what buys the
    exactness; `R = r_large * I3` is what keeps `S` invertible while it happens.
    """
    sc = Scenario(FOOT_BEYOND)
    blk = sc.block(np.zeros(1))
    H = np.vstack([sc.H_pairs, np.asarray(blk.H)])
    R = np.zeros((H.shape[0], H.shape[0]))
    R[:sc.H_pairs.shape[0], :sc.H_pairs.shape[0]] = sc.R_pairs
    R[sc.H_pairs.shape[0]:, sc.H_pairs.shape[0]:] = np.asarray(blk.R)
    z = np.concatenate([np.array([0.01, -0.02, 0.03]), np.asarray(blk.z)])

    state = init_state(sc.build, sc.params, jnp.asarray(sc.q))
    x0, P0 = np.asarray(state.x), np.asarray(state.P)

    # Both sides through the same independent reference KF, so this isolates the
    # masking algebra from `update.py`'s conditioning gate -- which, with
    # `r_large = 1e12`, currently fires here (see the xfail below).
    got_x, got_P = reference_update(x0, P0, H, z, R)
    keep = slice(0, sc.H_pairs.shape[0])
    want_x, want_P = reference_update(x0, P0, H[keep], z[keep], R[keep, keep])

    assert np.max(np.abs(got_x - want_x)) < 1.0e-12
    assert np.max(np.abs(got_P - want_P)) < 1.0e-12


def test_r_large_alone_converges_to_the_excluded_posterior():
    """The `R_LARGE -> infinity` oracle, on an **unmasked** `H`.

    Isolates the ``R`` half of the masking rule from the ``H`` half: with the
    anchor rows left in `H`, raising ``R_LARGE`` must drive the posterior to the
    row-excluded one at the expected ``O(1/R_LARGE)`` rate.  Without this, a
    reader could not tell whether the exactness above comes from `R_LARGE` doing
    its job or purely from the `H` zeroing.
    """
    sc = Scenario(FOOT_BEYOND)
    active = sc.block(np.ones(1))                       # unmasked H, real z
    H = np.vstack([sc.H_pairs, np.asarray(active.H)])
    z = np.concatenate([np.array([0.01, -0.02, 0.03]), np.asarray(active.z)])
    state = init_state(sc.build, sc.params, jnp.asarray(sc.q))

    keep = slice(0, sc.H_pairs.shape[0])
    want_x, _ = reference_update(
        np.asarray(state.x), np.asarray(state.P), H[keep], z[keep], sc.R_pairs,
    )

    errs = []
    for r_large in (1.0e4, 1.0e6, 1.0e8, 1.0e10):
        R = np.zeros((H.shape[0], H.shape[0]))
        R[keep, keep] = sc.R_pairs
        R[sc.H_pairs.shape[0]:, sc.H_pairs.shape[0]:] = r_large * np.eye(3)
        got, _ = reference_update(np.asarray(state.x), np.asarray(state.P), H, z, R)
        errs.append(np.max(np.abs(got - want_x)))

    # Asserted as a *rate*, not a final magnitude: each 100x in R_LARGE must buy
    # exactly 100x in accuracy. A final-magnitude bound would also be met by an
    # implementation that simply ignored the rows, which is the thing this test
    # exists to distinguish from.
    ratios = [b / a for a, b in zip(errs, errs[1:])]
    assert np.allclose(ratios, 1.0e-2, rtol=1.0e-2), f"not O(1/R_LARGE): {errs} -> {ratios}"


def test_inactive_anchor_keeps_the_innovation_covariance_invertible():
    """Zeroing the inactive anchor's `R` rows instead of using `R_LARGE` makes
    ``S = H P Hᵀ + R`` exactly singular (CLAUDE.md §6, a named trap).

    With `H` masked the anchor's diagonal block of `S` is *only* `R`, so a zeroed
    `R` is a zero block — the Cholesky in `joseph_update` goes non-finite and the
    conditioning gate drops the **entire** stacked update, pair rows included.
    Losing every gyro row because a foot is in swing is a silent, total failure,
    so it gets its own assertion rather than being left to the composition tests.
    """
    sc = Scenario(FOOT_BEYOND)
    blk = sc.block(np.zeros(1))
    state = init_state(sc.build, sc.params, jnp.asarray(sc.q))

    H = np.vstack([sc.H_pairs, np.asarray(blk.H)])
    P = np.asarray(state.P)
    R = np.zeros((H.shape[0], H.shape[0]))
    R[:sc.H_pairs.shape[0], :sc.H_pairs.shape[0]] = sc.R_pairs
    R[sc.H_pairs.shape[0]:, sc.H_pairs.shape[0]:] = np.asarray(blk.R)

    S = H @ P @ H.T + R
    assert_positive_semidefinite(S, "S with an inactive anchor")
    assert np.min(np.linalg.eigvalsh(0.5 * (S + S.T))) > 0.0, "S must be strictly PD"

    # And the counterfactual the rule exists to forbid.
    S_zeroed = S.copy()
    S_zeroed[sc.H_pairs.shape[0]:, sc.H_pairs.shape[0]:] = 0.0
    assert np.min(np.linalg.eigvalsh(0.5 * (S_zeroed + S_zeroed.T))) < 1.0e-30, \
        "sanity: a zeroed anchor R block is exactly what makes S singular"

    # Finite, and the Cholesky the filter actually runs succeeds.
    _, info = joseph_update(state, jnp.asarray(H), jnp.zeros(H.shape[0]),
                            jnp.asarray(R), sc.params)
    assert np.isfinite(float(info.condition_proxy))


def test_an_inactive_anchor_does_not_gate_out_the_whole_stacked_update():
    """A foot in swing must cost the filter its *anchor*, not its gyros.

    This was a genuine contract conflict, found by B2 and fixed in `update.py`.
    `CLAUDE.md` §4 sets `R_LARGE = 1e12` for an inactive anchor and
    `cond_s_max = 1e9` for the conditioning gate.  An inactive anchor's `R` block
    is structurally decoupled (its `H` rows are zero), so counting it gives
    `cond(S) >= 1e12 / lambda_min(pair block) ~ 4e11` — above the gate.  Every
    tick with any foot in swing therefore dropped the ENTIRE stacked update, gyro
    rows included: the filter would have stopped updating for the whole of
    walking, while reporting nothing worse than `was_applied = 0`.

    Java never meets this because its stacked measurement simply has no anchor
    rows when no foot is trusted.  The fixed-shape port has to say the same thing
    with a mask, and as configured the two constants are mutually destructive.

    Resolved by computing the condition proxy over **informative rows only**
    (`diag(R) < 0.5 * r_large`).  That is the gate's own semantics rather than a
    fudge: the gate exists to catch an `S` that inverts to a huge gain, and a row
    we have deliberately declared uninformative contributes gain ~1/R_LARGE ~ 0.
    """
    sc = Scenario(FOOT_BEYOND)
    blk = sc.block(np.zeros(1))
    H = np.vstack([sc.H_pairs, np.asarray(blk.H)])
    R = np.zeros((H.shape[0], H.shape[0]))
    R[:sc.H_pairs.shape[0], :sc.H_pairs.shape[0]] = sc.R_pairs
    R[sc.H_pairs.shape[0]:, sc.H_pairs.shape[0]:] = np.asarray(blk.R)

    state = init_state(sc.build, sc.params, jnp.asarray(sc.q))
    _, info = joseph_update(state, jnp.asarray(H), jnp.zeros(H.shape[0]),
                            jnp.asarray(R), sc.params)
    assert float(info.was_applied) == 1.0, (
        f"whole stacked update gated out by an INACTIVE anchor; "
        f"cond proxy {float(info.condition_proxy):.3e} vs cond_s_max "
        f"{sc.params.cond_s_max:.3e}"
    )


def test_anchor_jacobians_and_noise_match_the_closed_form():
    r"""Value oracle for ``J_F``, ``J_U`` and ``R_anchor``, from NumPy.

    Added after a mutation check: scaling the congruence by ``sqrt(sigma)``
    instead of ``sigma`` (a 10x inflation of ``R_anchor``) passed every other
    test in this file.  The trace threshold and the tighter-vs-looser comparison
    both constrain the congruence's *presence*, neither constrains its
    *magnitude* — `JOINTKF_PORT_PLAN.md` §4 lesson 1, found the hard way.

    Erring large is the safe direction, but a wrong exponent is still wrong: it
    would silently detune the one absolute bias observation in the filter.
    """
    sc = Scenario(FOOT_BEYOND)
    b, p = sc.build, sc.params
    dof_u = unfiltered_dof(b)

    R_base = sc.site_rot[0]
    foot = int(sc.foot_site[0])
    J = R_base.T @ (sc.J_world[foot] - sc.J_world[0])          # (3, nv), base frame
    J_F = J[:, np.asarray(b.dof_joint)] * np.asarray(b.anchor_filtered_mask)[0][None, :]
    J_U = J[:, dof_u] * np.asarray(b.anchor_unfiltered_mask)[0][None, :]

    assert np.max(np.abs(np.asarray(sc.jac.filtered)[0] - J_F)) < 1.0e-14
    assert np.max(np.abs(np.asarray(sc.jac.unfiltered)[0] - J_U)) < 1.0e-14

    want = p.anchor_var * np.eye(3) + J_U @ np.diag(
        np.full(len(dof_u), p.sigma_qd_unfiltered ** 2)
    ) @ J_U.T
    got = np.asarray(sc.block(np.ones(1)).R)[:3, :3]
    assert np.max(np.abs(got - want)) < 1.0e-15, f"R_anchor off by {np.max(np.abs(got - want)):.3e}"

    # And the velocity columns of H are exactly -J_F (the sign the Phase-3
    # marginalised oracle forces; see the module docstring).
    H = np.asarray(sc.block(np.ones(1)).H)
    n = b.n_joints
    assert np.max(np.abs(H[:, n:2 * n] + J_F)) < 1.0e-15


def test_anchor_noise_is_symmetric_psd_and_block_diagonal():
    """`R_anchor` is a sum of `Sigma_eps` and a Gram product, so PSD by
    construction — asserted so a future refactor to a non-Gram form is caught."""
    sc = Scenario(FOOT_BEYOND)
    for trusted in (np.zeros(1), np.ones(1)):
        R = np.asarray(sc.block(trusted).R)
        assert R.shape == (3 * sc.build.n_anchors,) * 2
        assert np.max(np.abs(R - R.T)) == 0.0
        assert_positive_semidefinite(R, "R_anchor")


def test_graph_is_constant_across_contact_masks():
    """Invariant I7: a foot landing changes a mask, never the graph.

    The jaxpr is compared across trusted-feet patterns.  As `PORT_NOTES.md`
    records, this is a weak proof on its own (traced arrays cannot change a
    jaxpr) — what it actually catches is a Python branch on the mask, which
    would raise instead.  Kept for exactly that.
    """
    sc = Scenario(FOOT_BEYOND)
    n_u = np.asarray(sc.build.anchor_unfiltered_mask).shape[1]

    def go(trusted, gyro, qd_u):
        return anchor_block(sc.build, sc.params, sc.jac, gyro_base=gyro,
                            qd_unfiltered=qd_u, trusted_feet=trusted).H

    jaxprs = {
        str(jax.make_jaxpr(go)(jnp.asarray(t, dtype=float), jnp.zeros(3), jnp.zeros(n_u)))
        for t in (np.zeros(1), np.ones(1))
    }
    assert len(jaxprs) == 1


def test_trusted_feet_is_an_argument_not_a_computation():
    """Phase ordering (CLAUDE.md §4): the previous tick's trust set drives this
    tick's anchors, so the mask must be injectable and nothing else may move it.

    Asserted behaviourally: with every input except the mask held fixed, the
    block is a pure function of the mask, and the two masks give genuinely
    different blocks.
    """
    sc = Scenario(FOOT_BEYOND)
    a = sc.block(np.ones(1))
    b = sc.block(np.ones(1))
    assert np.array_equal(np.asarray(a.H), np.asarray(b.H))
    assert np.array_equal(np.asarray(a.R), np.asarray(b.R))
    off = sc.block(np.zeros(1))
    assert not np.array_equal(np.asarray(a.H), np.asarray(off.H))


def test_unfiltered_dof_gather_matches_the_mask_columns():
    """`unfiltered_dof` depends on `build.py`'s concatenation order.

    A mis-ordered gather would pair the ankle's Jacobian column with the hip's
    velocity noise — same shape, wrong `R_anchor`, no error anywhere.  Pinned
    against the fixture's own DoF layout: hinge `j` of the chain owns DoF `j+6`,
    and the U split is joints 6..9.
    """
    sc = Scenario(FOOT_BEYOND)
    assert list(unfiltered_dof(sc.build)) == [12, 13, 14, 15]
    assert list(np.asarray(sc.build.dof_joint)) == [8, 9, 10, 11]


def test_dtypes_are_float64_at_the_boundary():
    """Invariant I8 — a float32 leak here silently halves the anchor's precision."""
    sc = Scenario(FOOT_BEYOND)
    blk = sc.block(np.ones(1))
    for name, a in zip(("H", "z", "R"), (blk.H, blk.z, blk.R)):
        assert a.dtype == jnp.float64, f"{name} is {a.dtype}"
    assert sc.jac.filtered.dtype == jnp.float64
    assert sc.jac.unfiltered.dtype == jnp.float64


def test_anchor_block_shapes_match_the_stacked_layout_contract():
    """The block must land at `build.anchor_row0` and fill `build.n_stacked_rows`
    together with the pair rows — the contract `measure.py` concatenates against."""
    for shape in (*SHAPES, FOOT_BEYOND):
        sc = Scenario(shape)
        blk = sc.block(np.ones(sc.build.n_anchors))
        assert blk.H.shape == (3 * sc.build.n_anchors, sc.build.dim)
        assert sc.H_pairs.shape[0] == sc.build.anchor_row0
        assert sc.H_pairs.shape[0] + blk.H.shape[0] == sc.build.n_stacked_rows


def test_measured_unfiltered_velocity_enters_the_residual():
    """`z = omega_tilde_base + J_U qdot_U`: the ankles' measured motion is a known
    input, so it must move the residual — otherwise the anchor asserts the *base*
    is still, not the *foot*."""
    sc = Scenario(FOOT_BEYOND)
    n_u = np.asarray(sc.build.anchor_unfiltered_mask).shape[1]
    gyro = np.array([0.05, -0.03, 0.02])
    qd_u = np.array([0.4, -0.3, 0.2, -0.1])[:n_u]

    z0 = np.asarray(sc.block(np.ones(1), gyro_base=gyro, qd_u=np.zeros(n_u)).z)
    z1 = np.asarray(sc.block(np.ones(1), gyro_base=gyro, qd_u=qd_u).z)
    assert np.allclose(z0, gyro, atol=0.0)
    want = gyro + np.asarray(sc.jac.unfiltered)[0] @ qd_u
    assert np.max(np.abs(z1 - want)) < 1.0e-15
    assert np.linalg.norm(z1 - z0) > 1.0e-3, "the U-split velocities must reach z"
