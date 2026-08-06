"""Port of `InvariantEKFReseedTest` + `TouchdownReseedLatchTest`.

Spec: `TEST_SUITE_MAP.md` §InvariantEKFReseedTest (G5). Trial counts preserved
(50 PSD trials, 200k latch chatter ticks); one fixed Python seed per test, per
CLAUDE.md §5 "RNG".

The three characteristic properties, in the order the Java suite asserts them:

1. the congruence keeps ``P`` symmetric PSD for any PSD input,
2. ``P_dd = P_pp + R N Rᵀ`` and ``P_θd = P_θp`` exactly,
3. **zero-release** — the same measurement re-applied straight after a re-seed
   produces zero innovation, zero NIS and zero correction.
"""
import numpy as np

import jax
import jax.numpy as jnp

from invariant_estimation.inEKF import ekf as ekf_mod
from invariant_estimation.inEKF.correct import innovation, linear_update
from invariant_estimation.inEKF.reseed import (
    ReseedParams,
    advance_latch,
    init_latch,
    pre_reseed_residual,
    reseed_contacts,
)
from invariant_estimation.inEKF.state import (
    BASE_POSITION_TANGENT_INDEX as P_IDX,
    contact_tangent_index,
)

from ._oracles import yaw_pitch_roll_to_matrix

N_CONTACTS = 2
M = 9 + 3 * N_CONTACTS
EPS = 1.0e-10


# ---------------------------------------------------------------------------
# Java helper ports
# ---------------------------------------------------------------------------

def random_psd(rng, scale, size=M):
    """Java `randomPsd`: ``A Aᵀ + 1e-9 I`` with ``A_ij = scale·(rand − 0.5)``."""
    A = scale * (rng.random((size, size)) - 0.5)
    return A @ A.T + 1.0e-9 * np.eye(size)


def randomly_initialized_state(rng, cov):
    """Java `randomlyInitializedEKF`: upright-ish base ~0.9 m up, 2 feet on the floor."""
    R = yaw_pitch_roll_to_matrix(
        rng.random() - 0.5, 0.4 * (rng.random() - 0.5), 0.4 * (rng.random() - 0.5)
    )
    p = np.array([rng.random(), rng.random(), 0.9 + 0.1 * rng.random()])
    d = np.array([
        [p[0] + 0.3 * (rng.random() - 0.5), p[1] + (0.1 if i == 0 else -0.1), 0.0]
        for i in range(N_CONTACTS)
    ])
    ekf = ekf_mod.create(N_CONTACTS, 1.0e-4, 1.0e-3, 1.0e-6)
    state = ekf_mod.initialize(
        ekf, rotation=jnp.asarray(R), velocity=jnp.zeros(3),
        position=jnp.asarray(p), contacts=jnp.asarray(d),
        covariance=jnp.asarray(cov),
    )
    return ekf, state


def consistent_measurement(state, world_foothold):
    """Java `consistentMeasurement`: ``y = Rᵀ(foothold − p)``, body frame."""
    return np.asarray(state.R).T @ (np.asarray(world_foothold) - np.asarray(state.p))


def fire_only(i, N=N_CONTACTS):
    """Fire mask selecting contact ``i`` alone."""
    return jnp.asarray(np.eye(N)[i], dtype=jnp.float64)


def body_cov(scale):
    """Java `bodyScaledIdentity` stacked over contacts: ``(N, 3, 3)``."""
    return jnp.tile(scale * jnp.eye(3), (N_CONTACTS, 1, 1))


# ---------------------------------------------------------------------------
# 1. PSD preservation — Java testReseedPreservesPositiveSemiDefiniteness
# ---------------------------------------------------------------------------

def test_reseed_preserves_positive_semidefiniteness():
    """50 trials: the congruence keeps ``P`` symmetric with min eigenvalue > −1e-8."""
    rng = np.random.default_rng(4242)

    for _ in range(50):
        _, state = randomly_initialized_state(rng, random_psd(rng, 0.5))
        fk_cov = body_cov(1.0e-6 * (1.0 + rng.random()))
        foothold = np.array([0.05 * rng.random(), 0.05 * rng.random(), 0.01 * rng.random()])
        y = np.tile(consistent_measurement(state, foothold), (N_CONTACTS, 1))

        out = reseed_contacts(state, jnp.asarray(y), fk_cov, fire_only(0))
        P = np.asarray(out.P)

        assert np.allclose(P, P.T, atol=EPS)
        assert np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -1.0e-8


# ---------------------------------------------------------------------------
# 2. Covariance consistency — Java testReseedCovarianceConsistency
# ---------------------------------------------------------------------------

def test_reseed_covariance_consistency():
    """``P_dd = P_pp + R N Rᵀ`` and ``P_θd = P_θp``, both exact (tol 1e-10).

    These two are the whole content of the re-seed. The second is the structural
    one: it forces the rotation rows of the gain to vanish for that contact, so a
    freshly re-anchored foot cannot rotate the estimate.
    """
    rng = np.random.default_rng(99)
    _, state = randomly_initialized_state(rng, random_psd(rng, 0.5))

    before = np.asarray(state.P).copy()
    R = np.asarray(state.R)
    d_idx = contact_tangent_index(0)

    fk_cov = body_cov(3.0e-6)
    y = np.tile(consistent_measurement(state, np.array([0.1, 0.05, 0.0])), (N_CONTACTS, 1))

    after = np.asarray(reseed_contacts(state, jnp.asarray(y), fk_cov, fire_only(0)).P)
    rotated_N = R @ (3.0e-6 * np.eye(3)) @ R.T

    # P_dd = P_pp + R N Rᵀ
    np.testing.assert_allclose(
        after[d_idx:d_idx + 3, d_idx:d_idx + 3],
        before[P_IDX:P_IDX + 3, P_IDX:P_IDX + 3] + rotated_N,
        atol=EPS,
    )
    # P_θd = P_θp  (rotation block is rows 0..2)
    np.testing.assert_allclose(
        after[0:3, d_idx:d_idx + 3], after[0:3, P_IDX:P_IDX + 3], atol=EPS
    )


def test_unfired_contact_is_bit_for_bit_untouched():
    """``fire_i = 0`` ⇒ identity congruence — the I7 masking contract.

    Not in the Java suite (which has no mask), and load-bearing here: the whole
    constant-graph story is that the congruence runs every tick and is exactly
    the identity when it must be. `atol=0`.
    """
    rng = np.random.default_rng(31337)
    _, state = randomly_initialized_state(rng, random_psd(rng, 0.5))
    y = jnp.asarray(rng.random((N_CONTACTS, 3)))

    out = reseed_contacts(state, y, body_cov(1.0e-4), jnp.zeros(N_CONTACTS))

    np.testing.assert_array_equal(np.asarray(out.d), np.asarray(state.d))
    np.testing.assert_array_equal(np.asarray(out.P), np.asarray(state.P))


def test_reseed_touches_only_the_fired_contact():
    """Re-seeding contact 0 leaves contact 1's mean and its own block alone."""
    rng = np.random.default_rng(1717)
    _, state = randomly_initialized_state(rng, random_psd(rng, 0.5))
    y = jnp.asarray(rng.random((N_CONTACTS, 3)))
    other = contact_tangent_index(1)

    out = reseed_contacts(state, y, body_cov(1.0e-4), fire_only(0))

    np.testing.assert_array_equal(np.asarray(out.d)[1], np.asarray(state.d)[1])
    np.testing.assert_allclose(
        np.asarray(out.P)[other:other + 3, other:other + 3],
        np.asarray(state.P)[other:other + 3, other:other + 3],
        atol=EPS,
    )


# ---------------------------------------------------------------------------
# 3. Zero release — Java testZeroReleaseAfterReseed
# ---------------------------------------------------------------------------

def test_zero_release_after_reseed():
    """The same measurement re-applied after a re-seed moves nothing.

    Java asserts: pre-reseed residual > 1e-3 (a real discrepancy), then update
    applied, NIS == 0, correction rotation norm == 0, and rotation/position
    unchanged — all at 1e-9.
    """
    rng = np.random.default_rng(7)
    ekf, state = randomly_initialized_state(rng, random_psd(rng, 0.5))

    fk_cov = body_cov(1.0e-4)
    y_one = consistent_measurement(state, np.array([0.08, -0.03, 0.005]))
    y = jnp.asarray(np.tile(y_one, (N_CONTACTS, 1)))

    pre = np.asarray(pre_reseed_residual(state, y))
    assert pre[0] > 1.0e-3, "the re-seed must be absorbing a genuine discrepancy"

    reseeded = reseed_contacts(state, y, fk_cov, fire_only(0))

    # Zero innovation on the re-seeded contact.
    nu = np.asarray(innovation(reseeded, y)).reshape(N_CONTACTS, 3)
    np.testing.assert_allclose(nu[0], np.zeros(3), atol=1.0e-9)

    # …and therefore zero correction when that measurement is applied.
    # Contact 0's rows ONLY: Java's `ekf.update(0, …)` is a single-contact
    # update, and contact 1 was not re-seeded, so its residual is legitimately
    # nonzero. Stacking both would test the wrong proposition.
    R_before, p_before = np.asarray(reseeded.R), np.asarray(reseeded.p)
    updated, diag = linear_update(
        reseeded, ekf.params.H[0:3], jnp.asarray(nu[0]),
        jnp.asarray(1.0e-4 * np.eye(3)),
    )

    assert float(diag.applied) == 1.0
    np.testing.assert_allclose(float(diag.nis), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(updated.R), R_before, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(updated.p), p_before, atol=1.0e-9)


def test_reseed_leaves_the_rotation_gain_block_at_zero():
    """The structural half of zero-release: ``K_θ = 0`` for the re-seeded contact.

    Stronger than "the residual happened to be zero" — it says a *slightly*
    inconsistent follow-up measurement still cannot rotate the estimate, which is
    what `P_θd = P_θp` buys and what makes the re-seed safe mid-stride.
    """
    rng = np.random.default_rng(2718)
    ekf, state = randomly_initialized_state(rng, random_psd(rng, 0.5))
    y = jnp.asarray(np.tile(consistent_measurement(state, np.array([0.1, 0.0, 0.0])),
                            (N_CONTACTS, 1)))

    reseeded = reseed_contacts(state, y, body_cov(1.0e-4), fire_only(0))

    H = np.asarray(ekf.params.H)
    P = np.asarray(reseeded.P)
    Rm = np.kron(np.eye(N_CONTACTS), 1.0e-4 * np.eye(3))
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rm)

    # Rotation rows (0..2), columns of the re-seeded contact's measurement block.
    np.testing.assert_allclose(K[0:3, 0:3], np.zeros((3, 3)), atol=1.0e-12)


# ---------------------------------------------------------------------------
# The latch — Java TouchdownReseedLatchTest
# ---------------------------------------------------------------------------

PARAMS = ReseedParams(enabled=True, trigger=0.5, rearm=0.1, dwell_ticks=100)


def drive(signal, params=PARAMS, n=1):
    """Run the latch over a 1-D signal; return the per-tick fire mask ``(T, n)``.

    `lax.scan`, not a Python loop — the latch is a scan carry in production and
    the 200k-tick property test below is unaffordable any other way.
    """
    def body(latch, p):
        latch, fire = advance_latch(latch, jnp.full(n, p), params)
        return latch, fire

    xs = jnp.asarray(np.asarray(signal, dtype=float))
    _, fires = jax.lax.scan(body, init_latch(n), xs)
    return np.asarray(fires)


def test_latch_fires_once_on_a_rising_crossing():
    fires = drive([0.0] * 150 + [1.0] * 500)
    assert fires.sum() == 1.0
    assert fires[150, 0] == 1.0


def test_latch_does_not_refire_without_a_full_dwell():
    """The mid-strike ``p: 1 → 0 → 1`` pulse must not produce a second re-seed.

    This is the named trap (CLAUDE.md §6, "reseed double-fire"): a 50-tick dip is
    half the dwell, so the latch stays disarmed.
    """
    fires = drive([0.0] * 150 + [1.0] * 200 + [0.0] * 50 + [1.0] * 200)
    assert fires.sum() == 1.0


def test_latch_rearms_after_a_full_dwell():
    fires = drive([0.0] * 150 + [1.0] * 200 + [0.0] * 100 + [1.0] * 200)
    assert fires.sum() == 2.0


def test_latch_dwell_counter_resets_on_any_high_tick():
    """99 low, one high, 99 low is *not* a dwell — the counter resets, not pauses."""
    fires = drive([0.0] * 150 + [1.0] * 200 + [0.0] * 99 + [0.3] * 1 + [0.0] * 99 + [1.0] * 50)
    assert fires.sum() == 1.0


def test_latch_is_armed_at_init():
    """The first touchdown after boot is exactly the one worth re-seeding."""
    fires = drive([1.0] * 10)
    assert fires[0, 0] == 1.0


def test_latch_ignores_the_band_between_rearm_and_trigger():
    """A signal parked at 0.3 neither fires nor re-arms."""
    fires = drive([0.0] * 150 + [1.0] * 10 + [0.3] * 5000 + [1.0] * 10)
    assert fires.sum() == 1.0


def test_latch_is_per_contact_independent():
    """Contacts advance their own latches; one firing does not disarm the other."""
    latch = init_latch(2)
    latch, fire = advance_latch(latch, jnp.array([1.0, 0.0]), PARAMS)
    np.testing.assert_array_equal(np.asarray(fire), np.array([1.0, 0.0]))
    latch, fire = advance_latch(latch, jnp.array([1.0, 1.0]), PARAMS)
    np.testing.assert_array_equal(np.asarray(fire), np.array([0.0, 1.0]))


def test_latch_chatter_property_over_200k_ticks():
    """200k ticks of chatter: never more than one fire per served re-arm dwell.

    The Java property test. `dwells` is counted by an independent NumPy pass over
    the same signal — a second implementation, not a re-read of the latch's own
    counter — and the ``+1`` is the armed-at-init fire.

    The signal is deliberately gait-like rather than uniform noise: uniform
    U(0,1) almost never produces 100 consecutive draws below 0.1, so it would
    never re-arm and the test would pass on a latch that fires exactly once.
    """
    rng = np.random.default_rng(20260806)
    phase = np.concatenate([np.zeros(150), np.ones(250)])          # 0.4 s gait
    signal = np.tile(phase, 500)[:200_000]
    # Chatter: flip ~2% of ticks to the opposite rail, which is what makes this a
    # test of the dwell rather than of a clean square wave.
    flip = rng.random(signal.size) < 0.02
    signal = np.where(flip, 1.0 - signal, signal)

    low_run = 0
    dwells = 0
    for p in signal:
        low_run = low_run + 1 if p < PARAMS.rearm else 0
        if low_run == PARAMS.dwell_ticks:
            dwells += 1

    fires = drive(signal).sum()

    assert fires <= dwells + 1
    assert fires > 1, "a 200k-tick gait signal must re-arm and re-fire many times"
