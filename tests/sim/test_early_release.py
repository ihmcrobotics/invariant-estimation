r"""`sim.sensors.EarlyRelease` — the causal anticipatory anchor release.

The mechanism, its failure mode, and the reason it exists are in
`docs/theory/anchor_release_timing.md` and the class docstring.  These tests pin
the four things that a change could break silently:

1. **It reproduces the offline arm E'** (`experiments.process_socket_ablation
   .causal_early_release`) tick for tick.  That function is what produced every
   recorded number for this mechanism; a live port that merely "does the same
   sort of thing" would make those numbers uncomparable, and nothing would raise.
2. **The impact blanking is load-bearing.**  Without it the latch fires on the
   touchdown transient and releases most of the stance — the measured failure of
   the first version of arm E.
3. **It only ever loosens.**  It may move liftoff earlier; it must never move
   touchdown earlier.
4. **Off is bit-for-bit the old reader.**  Every number on record was produced
   with the flag off.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.process_socket_ablation import causal_early_release
from invariant_estimation.sim.sensors import EarlyRelease

DT = 1.0e-3


# ---------------------------------------------------------------------------
# A synthetic gait: double-humped stance with a touchdown impact spike
# ---------------------------------------------------------------------------

def gait(n_steps: int = 6, stance: int = 500, swing: int = 300,
         impact: float = 6.0, spike_len: int = 30, seed: int = 0) -> np.ndarray:
    """`(T, 1)` normal force with the shape that broke the first arm E.

    A ~500-tick stance carrying a double hump around 1.0, preceded by a `impact`x
    spike lasting `spike_len` ticks (measured on `data/dr5`: 2.0–10.8x the stance
    median, landing 1–16% into the stance), then a `swing` of exactly zero.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        t = np.linspace(0.0, 1.0, stance)
        # Double hump, decaying to zero at the end -- the unloading ramp the
        # release is supposed to catch.
        f = 1.0 + 0.25 * np.sin(2.0 * np.pi * t) * np.cos(np.pi * t)
        f *= np.clip(np.minimum(8.0 * t, 1.0) * (1.0 - t) ** 0.6, 0.0, None)
        f[:spike_len] += impact * np.hanning(2 * spike_len)[:spike_len]
        f = np.maximum(f + 0.002 * rng.standard_normal(stance), 1e-6)
        out.append(f)
        out.append(np.zeros(swing))
    return np.concatenate(out)[:, None]


def run_live(fn: np.ndarray, **kw) -> np.ndarray:
    """`(T, N)` release mask from the online state machine."""
    er = EarlyRelease(n_points=fn.shape[1], dt=DT, **kw)
    return np.stack([er.update(f) for f in fn])


# ---------------------------------------------------------------------------
# 1. equivalence with the offline arm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.3, 0.5, 0.7, 0.85])
@pytest.mark.parametrize("blank", [0, 50, 150])
def test_live_release_matches_the_offline_arm_e_tick_for_tick(frac, blank):
    """The live machine and `causal_early_release` must be the same function.

    `causal_early_release` returns chol scalars (it max-es with the heuristic), so
    the comparison is on the loose/tight classification, which is the only thing
    it encodes.  Bit-equality of the mask, not a tolerance: these are two
    implementations of one boolean recurrence and any difference is a bug.
    """
    fn = gait()
    # A stance-everywhere heuristic, so the offline function's `max` with the
    # heuristic cannot mask a disagreement: every loose tick in its output is one
    # this mechanism produced.
    s = np.full_like(fn, 1.0e-4)
    s[fn[:, 0] <= 0.0] = 1.0e1
    off = causal_early_release(s, fn, frac, blank=blank)
    live = run_live(fn, frac=frac, blank_ticks=blank)
    np.testing.assert_array_equal(live > 0.0, off >= 1.0)


def test_the_two_agree_on_a_multi_contact_signal():
    """Contacts are independent: a phase-shifted second point must not couple."""
    a = gait(seed=0)
    b = np.roll(gait(seed=1), 350, axis=0)
    fn = np.concatenate([a, b], axis=1)
    s = np.where(fn <= 0.0, 1.0e1, 1.0e-4)
    np.testing.assert_array_equal(
        run_live(fn, frac=0.5) > 0.0, causal_early_release(s, fn, 0.5) >= 1.0)


# ---------------------------------------------------------------------------
# 2. the impact blanking is what makes it a release and not a constant loosening
# ---------------------------------------------------------------------------

def test_without_blanking_the_impact_spike_releases_most_of_stance():
    """The measured failure of arm E v1, as a regression.

    A running peak taken from tick 0 locks onto the touchdown impact, so
    ``frac * peak`` sits above the whole rest of the stance and the latch fires
    almost immediately.  That is a constant-loose anchor, which scored
    monotonically *worse* than the baseline.
    """
    fn = gait()
    loaded = fn[:, 0] > 0.0
    frac_of_stance = lambda blank: float(
        (run_live(fn, frac=0.5, blank_ticks=blank)[loaded, 0] > 0.0).mean())
    assert frac_of_stance(0) > 0.6, "the spike must dominate an unblanked peak"
    assert frac_of_stance(150) < 0.35, "blanking must leave most of stance anchored"


def test_release_lands_in_the_unloading_ramp_not_at_zero_load():
    """The point of the mechanism: fire while the foot still carries load.

    A threshold on load LEVEL fires when the load has already gone.  This must
    fire strictly earlier than the last loaded tick, on every stance.
    """
    fn = gait()
    rel = run_live(fn, frac=0.5)[:, 0] > 0.0
    loaded = fn[:, 0] > 0.0
    # Stance episodes as maximal runs of `loaded`.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loaded.view(np.int8), [0]])))
    leads = []
    for lo, hi in zip(edges[::2], edges[1::2]):
        seg = rel[lo:hi]
        assert seg.any(), "every stance must release before it ends"
        leads.append(hi - (lo + int(np.argmax(seg))))
    assert min(leads) > 5, f"release is at the very end of stance: leads {leads}"


# ---------------------------------------------------------------------------
# 3. it only ever loosens, and it latches
# ---------------------------------------------------------------------------

def test_a_swing_point_is_always_released():
    fn = gait()
    rel = run_live(fn, frac=0.5)[:, 0]
    assert np.all(rel[fn[:, 0] <= 0.0] > 0.0)


def test_the_release_latches_through_the_mid_stance_dip():
    """Ground reaction force is double-humped; a dip must not re-tighten.

    Constructed so the load recovers above ``frac * peak`` after dipping under it.
    """
    f = np.concatenate([np.full(200, 1.0), np.full(50, 0.2), np.full(200, 1.0),
                        np.zeros(50)])[:, None]
    rel = run_live(f, frac=0.5, blank_ticks=10)[:, 0]
    fired = int(np.argmax(rel > 0.0))
    assert 200 <= fired < 250, f"expected the dip to fire it, got tick {fired}"
    assert np.all(rel[fired:250] > 0.0), "the latch must hold through the recovery"


def test_a_fresh_touchdown_clears_the_latch():
    fn = gait(n_steps=3)
    rel = run_live(fn, frac=0.5)[:, 0]
    loaded = fn[:, 0] > 0.0
    starts = np.flatnonzero(loaded[1:] & ~loaded[:-1]) + 1
    assert len(starts) >= 2
    for s in starts:
        assert rel[s] == 0.0, f"stance starting at {s} began already released"


def test_frac_zero_is_rejected_as_a_configuration_not_silently_off():
    """`frac` must be in [0, 1): 1.0 releases instantly, and the OFF switch is the
    reader's `early_release=0.0`, never a degenerate `frac`."""
    with pytest.raises(ValueError):
        EarlyRelease(n_points=2, dt=DT, frac=1.0)
    with pytest.raises(ValueError):
        EarlyRelease(n_points=2, dt=DT, frac=0.5, peak_mode="whatever")


# ---------------------------------------------------------------------------
# 4. the alternatives the theory note asks for
# ---------------------------------------------------------------------------

def test_prev_mode_can_act_inside_the_blanking_window():
    """`peak_mode='prev'` exists to remove the dependence on this stance's impact.

    From the second stance on, the reference is already available at tick 0, so
    the mechanism is armed everywhere -- which is what makes its lead insensitive
    to the impact transient.
    """
    fn = gait(n_steps=4)
    cur = run_live(fn, frac=0.5, blank_ticks=150, peak_mode="current")[:, 0]
    prv = run_live(fn, frac=0.5, blank_ticks=150, peak_mode="prev")[:, 0]
    loaded = fn[:, 0] > 0.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loaded.view(np.int8), [0]])))
    lead = lambda r: np.array([hi - (lo + int(np.argmax(r[lo:hi])))
                               for lo, hi in zip(edges[::2], edges[1::2])])
    lc, lp = lead(cur), lead(prv)
    # Same or earlier on every stance after the first, and never later.
    assert np.all(lp[1:] >= lc[1:] - 1), f"prev fired later: {lc} vs {lp}"


def test_the_rate_test_only_ever_adds_releases():
    """`rate_frac` ORs into the level test, so it can only move a release earlier."""
    fn = gait()
    lvl = run_live(fn, frac=0.5)[:, 0] > 0.0
    both = run_live(fn, frac=0.5, rate_frac=2.0)[:, 0] > 0.0
    assert np.all(both | ~lvl), "the rate test removed a release the level test made"


def test_the_stance_clock_gives_a_fixed_lead_on_a_regular_gait():
    """`clock_lead` is the only member of the family that can reproduce arm B.

    On a perfectly periodic gait the previous stance's length predicts this one's, so
    the lead is exactly `clock_lead` on every liftoff after the first -- the property
    every load-threshold predictor fails (31-50% of liftoffs get zero lead).
    """
    fn = gait(n_steps=5, stance=500, swing=300, seed=3)
    # `frac = 0` disables the level test, so this measures the clock alone.
    rel = run_live(fn, frac=0.0, clock_lead=100)[:, 0] > 0.0
    loaded = fn[:, 0] > 0.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loaded.view(np.int8), [0]])))
    # The TRAILING released run -- the anchor has to be loose AT liftoff.
    leads = []
    for lo, hi in zip(edges[::2], edges[1::2]):
        seg = rel[lo:hi]
        k = 0
        while k < len(seg) and seg[len(seg) - 1 - k]:
            k += 1
        leads.append(k)
    assert leads[0] == 0, "no previous stance exists yet, so the clock must not fire"
    assert all(l == 100 for l in leads[1:]), leads


def test_the_clock_only_arms_after_a_completed_stance():
    fn = gait(n_steps=2, seed=4)
    rel = run_live(fn, frac=0.0, clock_lead=100)[:, 0] > 0.0
    loaded = fn[:, 0] > 0.0
    first = np.flatnonzero(loaded)[0]
    end_first = np.flatnonzero(loaded[first:] == False)[0] + first
    assert not rel[first:end_first].any(), "the clock fired inside the first stance"


def test_every_test_disabled_is_rejected():
    with pytest.raises(ValueError):
        EarlyRelease(n_points=1, dt=DT, frac=0.0)


def test_lead_ticks_counts_loaded_released_ticks():
    """`lead_ticks` is the per-liftoff lead the theory note asks to be measured."""
    fn = gait(n_steps=2)
    er = EarlyRelease(n_points=1, dt=DT, frac=0.5)
    leads, prev_on = [], False
    for f in fn:
        er.update(f)
        on = f[0] > 0.0
        if prev_on and not on:
            leads.append(int(last))
        last = er.lead_ticks[0]
        prev_on = on
    assert len(leads) == 2 and all(l > 5 for l in leads), leads


def test_off_dwell_coalesces_a_fragmented_stance():
    """A one-tick dropout mid-stance must not reset the peak or clear the latch.

    The closed-loop measurement this exists for: at `off_dwell = 0` a 30 s walk reports
    306 "liftoffs" on two feet against a real cadence near 2 steps/s, because the
    per-foot normal load momentarily reads zero mid-stance.
    """
    f = np.concatenate([np.full(300, 1.0), np.zeros(2), np.full(300, 0.4),
                        np.zeros(200)])[:, None]
    # Without the dwell the gap starts a new stance, so the second half re-blanks and
    # the 0.4 plateau is measured against its own peak instead of the 1.0 one.
    a = run_live(f, frac=0.5, blank_ticks=50, off_dwell=0)[:, 0] > 0.0
    b = run_live(f, frac=0.5, blank_ticks=50, off_dwell=10)[:, 0] > 0.0
    assert not a[400], "without the dwell the fragment should re-arm and stay tight"
    assert b[400], "with the dwell, 0.4 < 0.5*1.0 should have released"


def test_off_dwell_zero_is_the_recorded_behaviour():
    """Default OFF, bit-for-bit — every number on record predates the field."""
    fn = gait()
    np.testing.assert_array_equal(run_live(fn, frac=0.5),
                                  run_live(fn, frac=0.5, off_dwell=0))


def test_off_dwell_does_not_extend_a_real_swing():
    """A swing longer than the dwell must still be a swing."""
    fn = gait(n_steps=3, swing=300)
    rel = run_live(fn, frac=0.5, off_dwell=10)[:, 0]
    loaded = fn[:, 0] > 0.0
    # Everything more than `off_dwell` ticks into an unloaded run must read released.
    gap = ~loaded
    run = np.zeros(len(gap), int)
    for i in range(1, len(gap)):
        run[i] = run[i - 1] + 1 if gap[i] else 0
    assert np.all(rel[run > 10] > 0.0)
