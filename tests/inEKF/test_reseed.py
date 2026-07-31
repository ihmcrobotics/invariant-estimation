"""Touchdown re-seed — the port of `InvariantEKFReseedTest` and `TouchdownReseedLatchTest`.

`TEST_SUITE_MAP.md` §InvariantEKFReseedTest / §TouchdownReseedLatchTest are the spec; scenarios,
trial counts and tolerances are taken from there verbatim (50 reseed-PSD trials, 200 000 latch
ticks, 1e-10 / 1e-9 tolerances). Per §5 the Java RNG is not reproduced — every one of these is a
property test that recomputes its oracle from the same draw — but the trial counts are.

Adaptations, all recorded in PORT_NOTES:

* Java's `reseedContact(i, y, N)` returns a scalar (the pre-re-seed residual norm) and mutates the
  filter. The port is pure and re-seeds every slot at once under a mask, so it returns the residual
  **vector per contact**; `norm(pre_residual[i])` is Java's return value.
* The introspection API (`wasLastUpdateApplied`, `getLastNormalizedInnovationSquared`,
  `getLastCorrectionRotationNorm`) is already the `UpdateDiagnostics` pytree, so the zero-release
  test reads fields instead of calling getters.
* `TouchdownReseedLatch` is a mutable object in Java; here it is a float carry advanced by a pure
  function, so `drive` scans rather than looping over a stateful instance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF.correct import contact_update
from invariant_estimation.inEKF.reseed import (
    advance_latch,
    default_reseed_params,
    expand_per_foot,
    init_latch,
    reseed_contacts,
    reseed_step,
)
from invariant_estimation.inEKF.state import (
    BASE_POSITION_TANGENT_INDEX as PI,
    InEKFState,
    contact_tangent_index,
)

N_CONTACTS = 2
M = 9 + 3 * N_CONTACTS          # 15
EPS = 1e-10

PARAMS = default_reseed_params()        # 0.5 / 0.1 / 100, the Java constants


# ---------------------------------------------------------------------------
# oracles — the Java helpers, ported
# ---------------------------------------------------------------------------

def random_psd(rng: np.random.Generator, scale: float, m: int = M) -> np.ndarray:
    """Java `randomPsd`: `A Aᵀ` with `A_ij = scale*(rand-0.5)`, plus 1e-9 on the diagonal."""
    A = scale * (rng.random((m, m)) - 0.5)
    return A @ A.T + 1e-9 * np.eye(m)


def randomly_initialized_state(rng: np.random.Generator, cov: np.ndarray) -> InEKFState:
    """Java `randomlyInitializedEKF`: a plausible standing pose, feet on the ground at +-0.1 y."""
    yaw = rng.random() - 0.5
    pitch = 0.4 * (rng.random() - 0.5)
    roll = 0.4 * (rng.random() - 0.5)
    cy, sy, cp, sp, cr, sr = (np.cos(yaw), np.sin(yaw), np.cos(pitch),
                              np.sin(pitch), np.cos(roll), np.sin(roll))
    R = np.array([                                     # Z-Y-X, Euclid's setYawPitchRoll
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])
    p = np.array([rng.random(), rng.random(), 0.9 + 0.1 * rng.random()])
    d = np.stack([np.array([p[0] + 0.3 * (rng.random() - 0.5),
                            p[1] + (0.1 if i == 0 else -0.1), 0.0])
                  for i in range(N_CONTACTS)])
    return InEKFState(R=jnp.asarray(R), v=jnp.zeros(3), p=jnp.asarray(p),
                      d=jnp.asarray(d), P=jnp.asarray(cov))


def consistent_measurement(state: InEKFState, world_foothold: np.ndarray) -> jnp.ndarray:
    """Java `consistentMeasurement`: `y = R̂ᵀ(foothold − p̂)`, the body-frame FK of a chosen point."""
    return state.R.T @ (jnp.asarray(world_foothold) - state.p)


def _fire_only(i: int, n: int = N_CONTACTS) -> jnp.ndarray:
    return jnp.asarray([1.0 if k == i else 0.0 for k in range(n)])


def _measurements(state: InEKFState, i: int, y_i) -> jnp.ndarray:
    """`y` for every contact, with slot `i` replaced — the others are masked off anyway."""
    y = jax.vmap(lambda k: state.R.T @ (state.d[k] - state.p))(jnp.arange(state.N))
    return y.at[i].set(y_i)


# ---------------------------------------------------------------------------
# InvariantEKFReseedTest
# ---------------------------------------------------------------------------

def test_reseed_preserves_positive_semidefiniteness():
    """50 trials, any PSD prior in, symmetric PSD out (min eigenvalue > −1e-8).

    The point of building the update as a congruence rather than a block write: PSD holds
    structurally, for every input, not just for the well-conditioned ones.
    """
    rng = np.random.default_rng(4242)
    worst = 0.0
    for _ in range(50):
        st = randomly_initialized_state(rng, random_psd(rng, 0.5))
        fk = np.eye(3) * (1e-6 * (1.0 + rng.random()))
        foothold = np.array([0.05 * rng.random(), 0.05 * rng.random(), 0.01 * rng.random()])
        y = _measurements(st, 0, consistent_measurement(st, foothold))
        out, _ = reseed_contacts(st, y, jnp.asarray(np.stack([fk] * N_CONTACTS)), _fire_only(0))

        P = np.asarray(out.P)
        assert np.allclose(P, P.T, atol=EPS), "covariance lost symmetry"
        lo = float(np.linalg.eigvalsh(P).min())
        worst = min(worst, lo)
        assert lo > -1e-8, f"min eigenvalue {lo:.3e}"
    assert worst > -1e-8


def test_reseed_covariance_consistency():
    r"""The two exact identities: `P_dd = P_pp + R N Rᵀ` and `P_θd = P_θp`.

    The second is the `K_θ = 0` condition — it is what stops a re-seed from injecting an attitude
    correction at every touchdown, and it is the reason `test_zero_release_after_reseed` can hold.
    """
    rng = np.random.default_rng(99)
    st = randomly_initialized_state(rng, random_psd(rng, 0.5))
    before = np.asarray(st.P).copy()
    R = np.asarray(st.R)
    fk = np.eye(3) * 3.0e-6
    y = _measurements(st, 0, consistent_measurement(st, np.array([0.1, 0.05, 0.0])))

    out, _ = reseed_contacts(st, y, jnp.asarray(np.stack([fk] * N_CONTACTS)), _fire_only(0))
    after = np.asarray(out.P)

    di = contact_tangent_index(0)
    rotated_N = R @ fk @ R.T
    np.testing.assert_allclose(after[di:di + 3, di:di + 3],
                               before[PI:PI + 3, PI:PI + 3] + rotated_N, atol=EPS)
    np.testing.assert_allclose(after[0:3, PI:PI + 3], after[0:3, di:di + 3], atol=EPS)


def test_zero_release_after_reseed():
    """Re-apply the same measurement straight after a re-seed: zero residual, NIS and rotation.

    The whole behavioural point. `pre_residual` must be a real geometric discrepancy first
    (> 1e-3), or the test would pass on a re-seed that did nothing.
    """
    rng = np.random.default_rng(7)
    st = randomly_initialized_state(rng, random_psd(rng, 0.5))
    fk = jnp.asarray(np.eye(3) * 1e-4)
    y_i = consistent_measurement(st, np.array([0.08, -0.03, 0.005]))
    y = _measurements(st, 0, y_i)

    out, pre = reseed_contacts(st, y, jnp.stack([fk] * N_CONTACTS), _fire_only(0))
    assert float(jnp.linalg.norm(pre[0])) > 1e-3, "no discrepancy to release"

    R_before, p_before = np.asarray(out.R), np.asarray(out.p)
    post, residual, diag = contact_update(out, 0, y_i, fk)

    assert bool(diag.applied), "the conditioning gate rejected the update"
    assert float(jnp.linalg.norm(residual)) == pytest.approx(0.0, abs=1e-9)
    assert float(diag.nis) == pytest.approx(0.0, abs=1e-9)
    assert float(diag.correction_rotation_norm) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(np.asarray(post.R), R_before, atol=1e-9)
    np.testing.assert_allclose(np.asarray(post.p), p_before, atol=1e-9)


def test_a_tick_that_does_not_fire_changes_nothing_at_all():
    """`fire = 0` must be the IDENTITY congruence, bit-for-bit — the I7 requirement.

    The congruence runs every tick to keep the graph constant, so a non-firing tick has to be
    provably free. Not `allclose`: exact equality, because `T` is exactly the identity and
    `0 * RNRᵀ` is exactly zero, so any drift here would be a real algebraic difference.
    """
    rng = np.random.default_rng(31337)
    st = randomly_initialized_state(rng, random_psd(rng, 0.5))
    y = _measurements(st, 0, consistent_measurement(st, np.array([0.2, 0.1, 0.03])))
    fk = jnp.stack([jnp.asarray(np.eye(3) * 1e-4)] * N_CONTACTS)

    out, _ = reseed_contacts(st, y, fk, jnp.zeros(N_CONTACTS))
    assert np.array_equal(np.asarray(out.d), np.asarray(st.d))
    assert np.array_equal(np.asarray(out.P), np.asarray(st.P))


def test_reseeding_two_contacts_at_once_equals_reseeding_them_in_either_order():
    """The commutation the single-`T` implementation relies on, asserted rather than argued.

    `T_i` writes only block `d_i` and reads only block `p`, which no `T_j` writes — so the batched
    congruence must equal both sequential orders. If that ever stopped holding, the batched form
    would silently produce a different covariance from the documented per-contact one.
    """
    rng = np.random.default_rng(5150)
    st = randomly_initialized_state(rng, random_psd(rng, 0.5))
    y = jnp.stack([consistent_measurement(st, np.array([0.1, 0.05, 0.0])),
                   consistent_measurement(st, np.array([-0.1, -0.05, 0.01]))])
    fk = jnp.stack([jnp.asarray(np.eye(3) * 2e-6), jnp.asarray(np.eye(3) * 5e-6)])

    both, _ = reseed_contacts(st, y, fk, jnp.ones(N_CONTACTS))
    a, _ = reseed_contacts(st, y, fk, _fire_only(0))
    a, _ = reseed_contacts(a, y, fk, _fire_only(1))
    b, _ = reseed_contacts(st, y, fk, _fire_only(1))
    b, _ = reseed_contacts(b, y, fk, _fire_only(0))

    np.testing.assert_allclose(np.asarray(a.P), np.asarray(both.P), atol=1e-12)
    np.testing.assert_allclose(np.asarray(b.P), np.asarray(both.P), atol=1e-12)
    np.testing.assert_allclose(np.asarray(a.d), np.asarray(both.d), atol=1e-12)


# ---------------------------------------------------------------------------
# TouchdownReseedLatchTest
# ---------------------------------------------------------------------------

def drive(latch, probability: float, ticks: int, params=PARAMS):
    """Java `drive`: advance `ticks` times at a constant probability, count the fires."""
    fires = 0
    for _ in range(ticks):
        latch, fire = advance_latch(params, latch, jnp.asarray([probability]))
        fires += int(float(fire[0]))
    return latch, fires


def _latch():
    return init_latch(1)


def armed(latch) -> bool:
    return bool(float(latch.armed[0]) > 0.5)


def test_fires_once_on_clean_touchdown_and_not_again_while_high():
    latch, f = drive(_latch(), 0.0, 100)
    assert f == 0 and armed(latch)
    latch, f = drive(latch, 1.0, 500)
    assert f == 1, f"fired {f} times on one sustained strike"
    assert not armed(latch)


def test_mid_strike_dropout_cannot_double_fire():
    """The hardware failure the latch exists for: `p: 1 -> 0 -> 1` inside one foot strike."""
    latch, _ = drive(_latch(), 0.0, 500)
    latch, f = drive(latch, 1.0, 90)
    assert f == 1
    latch, f = drive(latch, 0.0, 99)          # dwell − 1: too short to be a real swing
    assert f == 0 and not armed(latch)
    latch, f = drive(latch, 1.0, 500)
    assert f == 0, "re-seeded twice inside a single strike"


def test_sustained_swing_rearms_for_the_next_strike():
    latch, _ = drive(_latch(), 0.0, 500)
    latch, f = drive(latch, 1.0, 200)
    assert f == 1
    latch, f = drive(latch, 0.0, 300)
    assert f == 0 and armed(latch)
    latch, f = drive(latch, 1.0, 200)
    assert f == 1


def test_single_tick_dips_never_rearm():
    """1000 one-tick dips interleaved with high ticks must produce nothing.

    This is what pins the *reset* semantics: the dwell counter resets on any tick at or above
    `rearm`, so 1000 isolated low ticks never accumulate to 100.
    """
    latch, _ = drive(_latch(), 0.0, 500)
    latch, f = drive(latch, 1.0, 90)
    assert f == 1
    total = 0
    for _ in range(1000):
        latch, a = drive(latch, 0.05, 1)
        latch, b = drive(latch, 0.9, 5)
        total += a + b
    assert total == 0


def test_mid_band_probability_neither_arms_nor_fires():
    """`rearm < p < trigger` is the hysteresis band: it must do nothing in either direction."""
    latch, f = drive(_latch(), 0.3, 1000)
    assert f == 0 and not armed(latch)
    latch, f = drive(latch, 1.0, 100)
    assert f == 0, "an unarmed latch fired"


def _fires(ps: np.ndarray) -> np.ndarray:
    """The latch's fire train over a probability stream, via `lax.scan` (200k ticks is too many
    for a Python loop over traced scalars — it costs ~60 s)."""
    def body(latch, p):
        latch, fire = advance_latch(PARAMS, latch, p[None])
        return latch, fire[0]
    _, fire = jax.lax.scan(body, init_latch(1), jnp.asarray(ps, dtype=jnp.float64))
    return np.asarray(fire) > 0.5


def _assert_one_fire_per_episode(ps: np.ndarray, fire: np.ndarray) -> int:
    """The property, checked against a reference model recomputed from the same draws.

    Java's model: `consecutiveLow` counts sub-`rearm` ticks and resets otherwise; reaching
    `DWELL_TICKS` grants episode credit. A fire is legal only against unspent credit.
    """
    low = ps < PARAMS.rearm
    run = np.zeros(len(ps), dtype=int)                    # consecutive-low run length
    for t in range(len(ps)):                              # cheap: one integer op per tick
        run[t] = run[t - 1] + 1 if low[t] and t else int(low[t])
    credit_at = np.flatnonzero(run >= PARAMS.dwell_ticks)

    prev = -1
    for t in np.flatnonzero(fire):
        granted = credit_at[(credit_at > prev) & (credit_at <= t)]
        assert granted.size, (f"fired at tick {t} with no completed sub-rearm episode since the "
                              f"previous fire at {prev}")
        prev = t
    return int(fire.sum())


def test_at_most_one_fire_per_sustained_low_episode_under_random_chatter():
    """The ported scenario: 200 000 uniform draws, checked against the reference model.

    **This scenario cannot fire, and that is a property of the scenario, not a bug.** Arming needs
    100 consecutive draws below 0.1, which under i.i.d. uniform noise has probability 1e-100 per
    position — so the Java test is vacuous on the "at most one" half and only ever exercised "never
    fires without credit". Ported as specified (200k ticks, same reference model) because it is a
    real guard against a latch that fires on noise; the non-vacuous half is
    `test_one_fire_per_episode_on_a_gait_like_stream`, which supplies episodes that actually
    complete.
    """
    ps = np.random.default_rng(1868).random(200_000)
    fire = _fires(ps)
    assert _assert_one_fire_per_episode(ps, fire) == 0, \
        "uniform chatter armed the latch — the dwell is not being enforced"


def test_one_fire_per_episode_on_a_gait_like_stream():
    """The same property where episodes DO complete: a noisy 0.8 s gait cycle with dropouts.

    Swing (~0.35 s of near-zero) is long enough to arm; stance carries the mid-strike `1 -> 0 -> 1`
    dropout that the latch exists to survive. So the fire count is checkable against the cycle
    count, which is what makes this the non-vacuous half of the chatter property.
    """
    rng = np.random.default_rng(20260730)
    cycles, swing, stance = 250, 350, 450
    ps = []
    for _ in range(cycles):
        ps.append(np.abs(rng.normal(0.0, 0.02, swing)))                  # swing: near zero
        strike = np.clip(rng.normal(0.95, 0.03, stance), 0.0, 1.0)       # stance: near one
        lo = rng.integers(30, stance - 40)
        strike[lo:lo + rng.integers(5, 60)] = 0.0                        # mid-strike dropout
        ps.append(strike)
    ps = np.concatenate(ps)

    fire = _fires(ps)
    n = _assert_one_fire_per_episode(ps, fire)
    assert n == cycles, f"expected exactly one re-seed per gait cycle, got {n} over {cycles}"


def test_constructor_rejects_degenerate_configurations():
    with pytest.raises(ValueError, match="hysteresis"):
        default_reseed_params(0.5, 0.5, 100)
    with pytest.raises(ValueError, match="dwell"):
        default_reseed_params(0.5, 0.1, 0)


def test_the_latch_is_independent_per_contact():
    """Two feet, driven out of phase, must not share a counter — they swing alternately."""
    params, latch = PARAMS, init_latch(2)
    for _ in range(150):                       # left low (arming), right high
        latch, fire = advance_latch(params, latch, jnp.asarray([0.0, 1.0]))
        assert float(fire[1]) == 0.0
    assert float(latch.armed[0]) == 1.0 and float(latch.armed[1]) == 0.0
    latch, fire = advance_latch(params, latch, jnp.asarray([1.0, 1.0]))
    assert float(fire[0]) == 1.0 and float(fire[1]) == 0.0


# ---------------------------------------------------------------------------
# jit / scan shape — the I7 obligation
# ---------------------------------------------------------------------------

def test_the_whole_reseed_step_scans_under_jit_with_a_constant_graph():
    """`reseed_step` in a `lax.scan`, and the same jaxpr whatever the fire pattern.

    A re-seed that forced a retrace whenever a foot landed would defeat the point of the masked
    formulation, so this compares the lowered graph across two very different probability streams
    rather than trusting that `jnp.where` was used everywhere.
    """
    rng = np.random.default_rng(0)
    st = randomly_initialized_state(rng, random_psd(rng, 0.5))
    fk = jnp.stack([jnp.asarray(np.eye(3) * 1e-6)] * N_CONTACTS)
    y = jax.vmap(lambda k: st.R.T @ (st.d[k] - st.p))(jnp.arange(N_CONTACTS))

    def run(ps):
        def body(carry, p):
            latch, state = carry
            latch, state, fire, _ = reseed_step(PARAMS, latch, state, p, y, fk)
            return (latch, state), fire
        return jax.lax.scan(body, (init_latch(N_CONTACTS), st), ps)

    # Both built the same way and with the same dtype: a weak-typed literal against a strong-typed
    # array lowers to a different graph for reasons that have nothing to do with the fire pattern,
    # and would make this test pass or fail on how the INPUT was written.
    T = 400
    f64 = lambda a: jnp.asarray(a, dtype=jnp.float64)            # noqa: E731
    never = f64(np.full((T, N_CONTACTS), 0.3))                   # mid-band: never fires
    often = f64(np.tile(np.concatenate([np.zeros(120), np.ones(80)])[:, None],
                        (2, N_CONTACTS)))                        # two full swing/strike cycles

    (_, s_never), f_never = jax.jit(run)(never)
    (_, s_often), f_often = jax.jit(run)(often)
    assert float(f_never.sum()) == 0.0
    assert float(f_often.sum()) == 2 * N_CONTACTS, "expected one fire per swing per contact"
    assert np.array_equal(np.asarray(s_never.P), np.asarray(st.P)), \
        "a never-firing scan changed the covariance"

    h = lambda xs: jax.jit(run).lower(xs).compiler_ir().__str__()   # noqa: E731
    assert h(never) == h(often), "the fire pattern changed the compiled graph (I7)"


def test_expand_per_foot_maps_toe_and_heel_to_their_own_foot():
    """N=4 toe/heel: contacts are foot-major, so `[L, R] -> [L, L, R, R]`."""
    got = expand_per_foot(jnp.asarray([0.2, 0.9]), 4)
    np.testing.assert_array_equal(np.asarray(got), [0.2, 0.2, 0.9, 0.9])
    np.testing.assert_array_equal(np.asarray(expand_per_foot(jnp.asarray([0.2, 0.9]), 2)),
                                  [0.2, 0.9])
    with pytest.raises(ValueError):
        expand_per_foot(jnp.asarray([0.2, 0.9]), 3)
