r"""`contactnet/dataset.ChainedBatcher` — the deployment error distribution.

Run 1 re-seeded every segment from ground truth, so every training sample began
at **zero** estimation error.  Over a 128 ms horizon the contact update then has
nothing to correct, the loss-minimising ``Sigma_C`` is infinite, and the network
duly learned to switch the contact update off (PORT_NOTES, 3835x velocity-gain
suppression).  `ChainedBatcher` carries ``(X_hat, P)`` between steps so a segment
starts wherever the filter actually got to.

These tests use a **fake** warm-in and a fake carry: the batcher's job is
bookkeeping — cursors, re-seed conditions, batching, detachment — and none of it
needs the real InEKF.  `experiments/alpha_sweep.py` is the end-to-end gate that
the resulting objective is non-degenerate; this file pins the mechanics.

Every test notes the mutant it kills, because a batcher test that merely calls
the methods and asserts shapes would survive most of the interesting bugs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect)
from invariant_estimation.contactnet import dataset
from invariant_estimation.contactnet.config import ContactNetConfig

from .test_dataset import (  # noqa: F401  — fixtures are reused wholesale
    N_C, dataset_dir, norm, preps,
)

P0 = np.eye(9 + 3 * N_C) * 1.0e-4


def _cfg(**kw) -> ContactNetConfig:
    """Chained config sized for the 1200-tick fabricated rollouts."""
    base = dict(F=24, sigma_0=1.0e-4, H=5, window_span_s=0.032, dt=1.0e-3,
                L=16, B=4, warm_in_s=0.05, episode_s=0.2)
    base.update(kw)
    return ContactNetConfig(**base)


class _FakeState(NamedTuple):
    v: jnp.ndarray
    P: jnp.ndarray


class _FakeCarry(NamedTuple):
    """Just enough of `InEKFCarry` for the divergence check to read.

    A NamedTuple and not a SimpleNamespace on purpose: the batcher stacks and
    unstacks carries with `jax.tree.map`, so the stand-in has to be a real
    pytree or the test would be exercising a different code path than training.
    """
    state: _FakeState


def _carry(tag: float, finite: bool = True) -> _FakeCarry:
    v = jnp.full((3,), tag if finite else jnp.nan)
    P = jnp.eye(9 + 3 * N_C) * (jnp.nan if not finite else 1.0)
    return _FakeCarry(state=_FakeState(v=v, P=P))


def _warm_in_fn(calls: list):
    """Records every warm-in call and returns a carry tagged with its index.

    Records the slice's LAST `omega` row as well as its length, so a test can
    locate where the warm-in ended in the source rollout without asking the
    batcher where it thinks the chain is.  Deriving the seed position from
    `chain.t` would make the contiguity check circular — the exact failure mode
    this suite's sibling docstring warns about.
    """
    def warm_in(state0, inputs):
        calls.append(SimpleNamespace(n=inputs.omega.shape[0],
                                     last=np.asarray(inputs.omega[-1])))
        return _carry(float(len(calls)))
    return warm_in


# ---------------------------------------------------------------------------
# Seeding and warm-in
# ---------------------------------------------------------------------------

def test_construction_seeds_B_chains_and_warms_each_in(preps):
    """One warm-in per chain, each over exactly `warm_in_ticks`.

    Kills: warming in once and sharing the carry across chains (the chains would
    be perfectly correlated), and a warm-in slice of the wrong length.
    """
    cfg = _cfg()
    calls = []
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn(calls), seed=0)

    assert len(b.chains) == cfg.B
    assert len(b.carries) == cfg.B
    assert [c.n for c in calls] == [cfg.warm_in_ticks] * cfg.B
    # Distinct carries, i.e. warm-in really ran per chain.
    tags = sorted(float(c.state.v[0]) for c in b.carries)
    assert tags == [1.0, 2.0, 3.0, 4.0]


def test_cursor_starts_where_the_warm_in_ended(preps):
    """The first *trained* tick is the one right after the last warmed-in tick.

    Located by matching the warm-in slice's final `omega` row back into the
    source rollout, NOT by subtracting `warm_in_ticks` from `chain.t` — that
    would be circular and would pass against `t = t_seed`.

    Kills: `t = t_seed` (the chain would train on the very ticks it warmed in
    on, reintroducing the zero-error start the design exists to remove), and an
    off-by-one in either direction.
    """
    cfg = _cfg()
    calls = []
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn(calls), seed=1)

    for c, call in zip(b.chains, calls):
        p = preps[c.rollout]
        # Warm-in ended at tick `c.t - 1`, so the segment starts at `c.t`.
        assert np.array_equal(np.asarray(p.inputs.omega[c.t - 1]), call.last), (
            "trained segment does not begin where the warm-in ended")
        assert p.t_lo <= c.t <= p.t_hi

    # Non-vacuity: `omega` is not constant, so the row match really pins the
    # index rather than passing everywhere.
    p0 = preps[b.chains[0].rollout]
    t = b.chains[0].t
    assert not np.array_equal(np.asarray(p0.inputs.omega[t - 1]),
                              np.asarray(p0.inputs.omega[t - 1 - cfg.warm_in_ticks]))


# ---------------------------------------------------------------------------
# Chaining — the property the whole class exists for
# ---------------------------------------------------------------------------

def test_update_feeds_the_previous_carry_back_and_advances_by_L(preps):
    """The carry out of step k is the carry into step k+1, and t advances by L.

    Kills: re-seeding every step (which is exactly run 1), dropping the carry,
    and advancing the cursor by anything but `L` (segments would overlap or gap).
    """
    cfg = _cfg()
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=2)
    t_before = [c.t for c in b.chains]

    _, carry0 = b.batch()
    out = jax.tree.map(lambda *xs: jnp.stack(xs),
                       *[_carry(100.0 + i) for i in range(cfg.B)])
    assert b.update(out) == 0                      # no chain hits a reseed yet

    assert [c.t for c in b.chains] == [t + cfg.L for t in t_before]
    got = sorted(float(c.state.v[0]) for c in b.carries)
    assert got == [100.0, 101.0, 102.0, 103.0], "carry was not fed back"

    # And the NEXT batch hands that carry to the loss, rather than a fresh seed.
    _, carry1 = b.batch()
    assert np.array_equal(np.asarray(carry1.state.v[:, 0]),
                          np.array([100.0, 101.0, 102.0, 103.0]))
    # Non-vacuity: the first batch's carry was different, so "fed back" is a
    # real constraint and not an artifact of a constant.
    assert not np.array_equal(np.asarray(carry0.state.v[:, 0]),
                              np.asarray(carry1.state.v[:, 0]))


def test_batch_carry_is_detached(preps):
    """The carry crossing a step boundary carries no gradient.

    Truncated BPTT is the only reason `L` bounds anything; a live carry would
    make the gradient run the whole episode.  Kills: dropping the
    `stop_gradient`.
    """
    cfg = _cfg()
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=3)
    stored = list(b.carries)

    def through_batch(x):
        b.carries = [jax.tree.map(lambda a: a * x, c) for c in stored]
        _, carry = b.batch()
        return jnp.sum(carry.state.v ** 2)

    assert float(jax.grad(through_batch)(jnp.array(1.0))) == 0.0

    # Non-vacuity: the same scalar function WITHOUT the batcher in the path has a
    # large nonzero gradient, so the zero above is `stop_gradient` and not an
    # accident of the carry values.
    v0 = jnp.stack([c.state.v for c in stored])
    direct = jax.grad(lambda x: jnp.sum((v0 * x) ** 2))(jnp.array(1.0))
    assert abs(float(direct)) > 1.0


# ---------------------------------------------------------------------------
# Re-seeding
# ---------------------------------------------------------------------------

def test_episode_length_forces_a_reseed(preps):
    """A chain re-seeds once it has run `episode_ticks`, and its counter resets.

    Kills: an episode bound that never fires (chains drift unboundedly) and one
    that fires every step (back to run-1 behaviour with extra cost).
    """
    # One chain, pinned near the START of its rollout, so the episode bound is
    # the only condition that can fire — the cursor cannot reach `t_hi` and the
    # carry stays finite.  With B > 1 chains legitimately reseed at different
    # steps (whichever seed landed nearest `t_hi` runs off first), which would
    # make "reseeded at step k" untestable.
    cfg = _cfg(B=1, warm_in_s=0.02, episode_s=0.1)   # 20 warm-in, 100 episode ticks
    calls = []
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn(calls), seed=4)
    n_seed = len(calls)
    b.chains[0].t = preps[b.chains[0].rollout].t_lo
    # Undo the constructor's phase stagger: this test is about the episode
    # bound, not about where in an episode a chain happens to start.
    b.chains[0].ticks = cfg.warm_in_ticks

    # ticks: 20 at seed, +L per step; reseed on the step that reaches 100.
    expect = int(np.ceil((cfg.episode_ticks - cfg.warm_in_ticks) / cfg.L))
    fired = None
    for k in range(1, expect + 2):
        b.batch()
        out = jax.tree.map(lambda *xs: jnp.stack(xs), _carry(1.0))
        if b.update(out):
            fired = k
            break

    assert fired == expect, f"reseed fired at step {fired}, expected {expect}"
    assert len(calls) == n_seed + 1, "reseed did not warm in"
    assert b.chains[0].ticks == cfg.warm_in_ticks, "episode counter did not reset"


def test_non_finite_carry_forces_a_reseed(preps):
    """A diverged chain is re-seeded, not carried.

    Without this one NaN poisons every later step of that chain.  Kills: a
    divergence check that only looks at `P` when `v` blew up, or none at all.
    """
    cfg = _cfg()
    calls = []
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn(calls), seed=5)
    n_seed = len(calls)
    b.batch()

    out = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[_carry(1.0, finite=(i != 2)) for i in range(cfg.B)])
    assert b.update(out) == 1, "exactly the diverged chain should reseed"
    assert len(calls) == n_seed + 1
    assert all(bool(jnp.all(jnp.isfinite(c.state.v))) for c in b.carries)


def test_reseed_when_the_cursor_would_run_off_the_rollout(preps):
    """A chain re-seeds rather than slicing past `t_hi`.

    Kills: an unguarded cursor, which `make_segment` would reject with a range
    error mid-run — after an hour of training, not at build.
    """
    cfg = _cfg(episode_s=100.0)                    # episode bound cannot be what fires
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=6)
    b.chains[0].t = preps[b.chains[0].rollout].t_hi - cfg.L + 1

    b.batch()
    out = jax.tree.map(lambda *xs: jnp.stack(xs),
                       *[_carry(1.0) for _ in range(cfg.B)])
    assert b.update(out) >= 1
    for c in b.chains:
        p = preps[c.rollout]
        assert p.t_lo <= c.t <= p.t_hi - cfg.L, "cursor left the legal range"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_batch_shapes_and_dtypes(preps):
    """`batch()` returns a device-side float64 `Segment` with a leading B axis."""
    cfg = _cfg()
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=7)
    seg, carry = b.batch()

    assert seg.windows.shape == (cfg.B, cfg.L, N_C, cfg.H, cfg.F)
    assert seg.v_true.shape == (cfg.B, cfg.L, 3)
    assert seg.inputs.contact_chol.shape == (cfg.B, cfg.L, N_C, 3, 3)
    assert carry.state.v.shape == (cfg.B, 3)
    for leaf in jax.tree.leaves(seg):
        assert leaf.dtype == jnp.float64, leaf.shape


def test_chains_are_seeded_across_distinct_rollouts(preps):
    """B chains should not all land on one rollout.

    `batch_stream` decorrelated a batch by re-drawing rollouts every step
    (dataset.py (d)); a chain is pinned for a whole episode, so the decorrelation
    has to come from the seeding instead.  Kills: seeding every chain from
    `preps[0]`.
    """
    cfg = _cfg(B=8)
    seen = set()
    for seed in range(6):
        b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=seed)
        seen.update(c.rollout for c in b.chains)
    assert len(seen) == len(preps), f"only {seen} of {len(preps)} rollouts seeded"


def test_rollouts_too_short_for_warm_in_are_rejected_loudly(preps):
    """A warm-in longer than any rollout must raise at build, not slice garbage."""
    cfg = _cfg(warm_in_s=5.0, episode_s=10.0)      # 5000 ticks vs a 1200-tick fixture
    with pytest.raises(ValueError, match="no rollout has room"):
        dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=8)


def test_initial_episode_phases_are_staggered(preps):
    """Chains must not all reach `episode_ticks` on the same step.

    Seeding every chain at `ticks = warm_in_ticks` makes them re-seed in a
    synchronised wave and then march in lockstep: the batch sits at one common
    time-since-seed forever, so any phase-dependent effect is perfectly
    correlated across it and most of the B independent samples are lost.
    Observed live in run 2's first launch as the reseed counter jumping 12 -> 37
    between steps 100 and 150.

    Kills: `ticks=cfg.warm_in_ticks` for every chain at construction.
    """
    cfg = _cfg(B=16)
    b = dataset.ChainedBatcher(preps, cfg, P0, _warm_in_fn([]), seed=11)

    phases = [c.ticks for c in b.chains]
    assert len(set(phases)) > cfg.B // 2, f"phases barely vary: {sorted(phases)}"
    assert all(cfg.warm_in_ticks <= t < cfg.episode_ticks for t in phases)

    # The property that matters: steps-until-reseed is spread, not a single value.
    remaining = sorted((cfg.episode_ticks - t + cfg.L - 1) // cfg.L for t in phases)
    assert remaining[-1] - remaining[0] > 1, (
        f"every chain reseeds within one step of the others: {remaining}")
