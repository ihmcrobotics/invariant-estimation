r"""
contactnet/dataset.py
=====================
Collected rollouts (`sim/collect.py`) → batches of `rollout.Segment`.

The pipeline is three passes, deliberately separated by what they cost and by
what they depend on:

1. **`build_channel_cache`** — the only pass that needs MJX.  Runs
   `features.make_contact_channels` and the contact FK over each saved rollout
   and writes ``(T, N_c, F)`` channels plus ``(T, N_c, 3)`` FK vectors to
   ``data/cache/``.  ~24 MB per rollout against the 130 MB rollout itself, so
   every later pass is a plain `np.load`.
2. **`fit_normalization`** — pools every cached rollout's *usable* region and
   freezes `normalize.NormConstants` to ``data/norm_constants.npz``.
3. **`prepare` / `batch_stream`** — pure NumPy.  No MJX, no estimator build.

Five things here are load-bearing.

**(a) The feature path does not survive a full rollout.**
`features.make_contact_channels` vmaps full-body MJX FK over the whole leading
axis; at ``T = 62 000`` that reached 38 GB of RSS.  Pass 1 therefore goes
through `collect.contact_channels_chunked` (and `chunked_fk` for the contact FK,
which has the same shape of problem).  This is why the cache exists at all: it
converts a pass that cannot be run casually into one `np.load`.

**(b) `inputs.contact_chol` is passed through, not frozen** (since 2026-07-28).
`sim.sensors.SimSensorReader` switches it ``1e-4 ↔ 1e1`` off `ContactTrust` —
the Schmitt-trigger port of `FootSwitchContactProbabilityProvider`, driven by
normal force ``f_n/(0.5 m g)``, i.e. a *sensor-derived* estimate and not the
sim's binary contact truth.

Run 1 froze it at the stance constant, on the argument that keeping it would
hand the network a free ground-truth contact flag.  That argument does not hold:
**the network never sees `contact_chol`.**  Its input is the 24 feature channels
(`features.channel_names`); `contact_chol` enters the *filter's process model*
only, so passing the real value changes the filter ContactNet is differentiated
through, not the information ContactNet receives.

The freeze was also measured to be the primary cause of run 1's collapse: at the
stance value a *swinging* foot is pinned world-static, and the contact update
then fights a process model wrong for half the gait — 10.2× worse in body-frame
velocity error than not using contacts at all
(`experiments/measure_tstar.py`).  `ContactNetConfig.freeze_contact_chol`
restores the old behaviour for reproducibility and defaults to `False`.

**(c) Warm-up and lead-in.**  `meta["warmup_ticks"]` (16 000, measured) is the
joint-KF gyro-bias plateau: before it, ``Σ_q`` — hence the InEKF's contact
``N = J Σ_q Jᵀ`` — is a transient that never occurs on hardware.  On top of it,
the first ``(H-1)·stride`` ticks cannot be segment *starts*.  Strictly, because
this module windows the **full** channel stream rather than the post-warm-up
slice, those windows are real rather than boxcar-clamped; the bound is kept
anyway because it costs 0.9% of the usable starts and removes the question.

**(d) Batches are composed across rollouts.**  Segments from one rollout share a
terrain, a seed, a spawn pose and one continuous filter trajectory, so ``B`` of
them is well under ``B`` independent samples.  `sample_starts` draws a fresh
permutation of the rollouts every batch and cycles it, so a batch touches
``min(B, n_rollouts)`` distinct rollouts.  Starts within a rollout are uniform
random, never tiled — tiled starts at stride ``L`` would make consecutive
batches replay one trajectory in order.

**(e) Ticks inside a segment are never shuffled.**  The segment is a trajectory:
`rollout.make_segment_loss` scans the filter along it.  `make_segment` only ever
slices contiguously.

**(f) Segments are chained, not independently seeded** (since 2026-07-28).
`ChainedBatcher` walks ``B`` filter chains through the rollouts in order,
carrying ``(X̂, P)`` and the gravity reference from one segment into the next.
`state0` below is therefore the *chain seed*, used at chain start and re-seed
only — not once per segment.

Run 1 re-seeded every segment from ground truth.  That is the force-teacher
scheme CoCo-InEKF (arXiv 2605.15122, §III-B) reports degrades learning, and the
reason is structural: a 128 ms segment starting at **zero** error gives the
contact update nothing to correct, so the loss-minimising ``Σ_C`` is infinite.
With the state carried, the error at a segment start is whatever the filter has
actually accumulated, which is the deployment distribution.

Seeding (`state0`)
------------------
``R, v, p`` come from ground truth at the segment start.  The contact anchors do
**not**: they are ``d_i = R_true·y_i(q̂) + p_true`` with ``y_i`` the FK evaluated
at the *recorded filter* joint estimate ``inputs.joint.q`` — the same vector the
filter will measure against on the first tick — so the first contact residual is
exactly zero.  Seeding them from truth ``q`` instead would inject the
filter-vs-truth joint offset as a step at tick 1, which is the failure
`pipeline.main_estimator.init_fused_carry` documents for the ankles.

The joint KF is **not** reseeded.  It ran continuously through the collection
and its outputs are frozen into `inputs`; there is nothing to reseed.

``P0`` is measured, not guessed — see `measure_p0`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..inEKF.filter import InEKFInputs, init_carry, make_step
from ..inEKF.state import InEKFState
from ..sim import collect
from . import features, normalize
from .config import ContactNetConfig
from .rollout import Segment

__all__ = [
    "DATA_DIR", "CACHE_DIR", "NORM_PATH",
    "rollout_paths", "chunked_fk", "build_channel_cache", "load_channel_cache",
    "fit_normalization", "measure_p0", "PreparedRollout", "prepare",
    "valid_start_range", "make_segment", "sample_starts", "make_batch", "batch_stream",
]

DATA_DIR = collect.DATA_DIR
CACHE_DIR = DATA_DIR / "cache"
NORM_PATH = DATA_DIR / "norm_constants.npz"


ROLLOUT_MARKER = "sensors.encoders"
"""A key every `collect.save_rollout` archive has and nothing else does."""


def is_rollout(path: Path | str) -> bool:
    """Does this ``.npz`` actually hold a `collect.Rollout`?

    Structural, not name-based.  ``data/`` is also where the normalization
    constants, the measured ``P0`` and training checkpoints land, and a
    name-based filter silently promotes each new artifact to a "rollout" that
    then fails deep inside `load_rollout` — which is exactly what happened the
    first time a checkpoint was written there.  Reading the archive's central
    directory is a few hundred bytes, not the 130 MB payload.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            return ROLLOUT_MARKER in z.files
    except Exception:
        return False


def rollout_paths(data_dir: Path | str = DATA_DIR) -> list[Path]:
    """Every saved rollout ``.npz`` in `data_dir`, sorted; non-rollouts skipped."""
    return sorted(p for p in Path(data_dir).glob("*.npz") if is_rollout(p))


# ---------------------------------------------------------------------------
# Pass 1 — the MJX-dependent cache
# ---------------------------------------------------------------------------

def chunked_fk(kinematics, q: np.ndarray, chunk: int = 2_000) -> np.ndarray:
    r"""``kinematics(q_t, 0).y`` for every tick, ``(T, n_j) -> (T, N_c, 3)``.

    Chunked for the same reason `collect.contact_channels_chunked` is: a single
    ``vmap`` of full-body MJX FK over ``T = 62 000`` ticks is tens of GB.  Unlike
    the channel path this one is *exactly* splittable — ``y`` has no cross-tick
    term at all — so no lead-in is needed and the result is bit-identical to a
    one-pass vmap up to XLA's batch-width-dependent FK rounding.
    """
    q = np.asarray(q, dtype=np.float64)
    fk = jax.jit(jax.vmap(lambda qq: kinematics(qq, jnp.zeros_like(qq)).y))
    out = [np.asarray(fk(jnp.asarray(q[lo:lo + chunk]))) for lo in range(0, q.shape[0], chunk)]
    return np.concatenate(out, axis=0)


def build_channel_cache(paths: Sequence[Path | str], collector: collect.Collector, *,
                        cache_dir: Path | str = CACHE_DIR, chunk: int = 2_000,
                        verbose: bool = True) -> list[Path]:
    r"""Write ``channels`` and ``y_fk`` for every rollout in `paths`.

    ``channels`` is `features.make_contact_channels` over `Rollout.sensors` —
    **raw**, not normalized: the normalization constants are fit *from* this
    cache (pass 2), so caching a normalized version would be circular and would
    force a full recompute on every refit.

    ``y_fk`` is the contact FK at the *recorded filter* joint estimate
    ``inputs.joint.q`` (13-wide: filtered ⧺ off-path ankles), used to seed
    `Segment.state0`'s contact anchors — see this module's docstring.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fused = collector.fused
    sub = features.build_subchain_indices(
        fused.build.joint_names, collect._unfiltered_names(collector))
    channels = features.make_contact_channels(
        sub, fused.base_imu, fused.kinematics, collector.dt)

    written = []
    for p in paths:
        p = Path(p)
        roll = collect.load_rollout(p)
        x = collect.contact_channels_chunked(channels, roll.sensors, chunk=chunk)
        y = chunked_fk(fused.kinematics, roll.inputs.joint.q, chunk=chunk)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise RuntimeError(f"{p.name}: non-finite features/FK in the cache pass")
        out = cache_dir / f"{p.stem}_feat.npz"
        np.savez_compressed(
            out, channels=x, y_fk=y,
            names=np.asarray(features.channel_names()),
            meta=np.array(json.dumps(roll.meta)))
        written.append(out)
        if verbose:
            print(f"  cached {p.name}: channels {x.shape}, y_fk {y.shape} "
                  f"-> {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    return written


def load_channel_cache(path: Path | str) -> dict:
    """Read one cache file back: ``channels``, ``y_fk``, ``names``, ``meta``."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {
            "channels": np.asarray(z["channels"], dtype=np.float64),
            "y_fk": np.asarray(z["y_fk"], dtype=np.float64),
            "names": tuple(str(s) for s in z["names"]),
            "meta": json.loads(str(z["meta"])),
        }


def cache_path(rollout_path: Path | str, cache_dir: Path | str = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{Path(rollout_path).stem}_feat.npz"


# ---------------------------------------------------------------------------
# Pass 2 — frozen normalization
# ---------------------------------------------------------------------------

def fit_normalization(paths: Sequence[Path | str], *, cache_dir: Path | str = CACHE_DIR,
                      source: str = "", usable_only: bool = True) -> normalize.NormConstants:
    r"""Pool every cached rollout and freeze `normalize.NormConstants`.

    Pools the **usable** region of each rollout (post-warm-up) by default, not
    the whole stream: the constants must describe what the network is actually
    fed, and the warm-up prefix includes the standing settle, which is precisely
    the "calibration set never exercised it" failure `NormConstants.floored`
    exists to expose.  Pass ``usable_only=False`` to fit over everything — useful
    only for comparing the two.

    The channels are pooled along time, so `normalize.fit`'s reduction over
    ``(time, contact)`` sees one flat calibration set across every terrain and
    seed.  One weight set ⇒ one normalization (§3.1).
    """
    parts, names, n = [], None, []
    for p in paths:
        c = load_channel_cache(cache_path(p, cache_dir))
        lo = int(c["meta"]["warmup_ticks"]) if usable_only else 0
        parts.append(c["channels"][lo:])
        n.append(f"{c['meta']['terrain']}/seed{c['meta']['seed']}")
        if names is None:
            names = c["names"]
        elif names != c["names"]:
            raise ValueError(f"channel names differ between rollouts: {p}")
    if not parts:
        raise ValueError("no rollouts to fit normalization on")
    pooled = jnp.asarray(np.concatenate(parts, axis=0))
    return normalize.fit(pooled, names, source=source or f"{len(parts)} rollouts: {', '.join(n)}")


# ---------------------------------------------------------------------------
# Pass 3 — prepared rollouts and segments
# ---------------------------------------------------------------------------

@dataclass
class PreparedRollout:
    """One rollout, normalized and boxcar-smoothed, ready to slice segments from.

    `smoothed` is ``boxcar(normalize.apply(channels), stride)`` over the **whole**
    stream.  Doing the boxcar once, globally, is what makes a segment's windows
    bit-identical to `features.window`'s over the full rollout: the boxcar is a
    cumulative sum, so running it on a slice would give a different (equally
    valid, but not equal) floating-point result, and `make_segment` then only has
    to gather.
    """

    name: str
    smoothed: np.ndarray          # (T, N_c, F)   normalized + boxcar(stride)
    inputs: InEKFInputs           # NumPy leaves, leading axis T
    y_fk: np.ndarray              # (T, N_c, 3)   FK contact vectors at inputs.joint.q
    R_true: np.ndarray            # (T, 3, 3)
    v_true: np.ndarray            # (T, 3)
    p_true: np.ndarray            # (T, 3)
    t_lo: int                     # first legal segment start
    t_hi: int                     # last  legal segment start (inclusive)
    meta: dict

    @property
    def n_starts(self) -> int:
        return max(0, self.t_hi - self.t_lo + 1)


def valid_start_range(T: int, warmup: int, cfg: ContactNetConfig) -> tuple[int, int]:
    r"""``(t_lo, t_hi)`` inclusive bounds on a segment start.

    ``t_lo = warmup + (H-1)·stride`` (docstring (c)); ``t_hi = T - L`` so the
    whole segment lies inside the rollout.  Raises rather than returning an empty
    range: a rollout too short to yield a single segment is a collection bug, not
    a sample to skip quietly.
    """
    lo = int(warmup) + (cfg.H - 1) * cfg.stride
    hi = int(T) - cfg.L
    if hi < lo:
        raise ValueError(
            f"no legal segment start: T={T}, warmup={warmup}, lead-in="
            f"{(cfg.H - 1) * cfg.stride}, L={cfg.L} leaves [{lo}, {hi}]")
    return lo, hi


def prepare(paths: Sequence[Path | str], norm: normalize.NormConstants,
            cfg: ContactNetConfig, *, cache_dir: Path | str = CACHE_DIR,
            verbose: bool = False) -> list[PreparedRollout]:
    """Load rollouts + caches, normalize, smooth, and compute the start bounds."""
    out = []
    for p in paths:
        p = Path(p)
        roll = collect.load_rollout(p)
        c = load_channel_cache(cache_path(p, cache_dir))
        if c["names"] != norm.names:
            raise ValueError(f"{p.name}: cached channel names disagree with the constants")
        x = np.asarray(normalize.apply(jnp.asarray(c["channels"]), norm))
        smoothed = np.asarray(features.boxcar(jnp.asarray(x), cfg.stride))
        T = smoothed.shape[0]
        t_lo, t_hi = valid_start_range(T, roll.meta["warmup_ticks"], cfg)
        prep = PreparedRollout(
            name=f"{roll.meta['terrain']}/seed{roll.meta['seed']}",
            smoothed=smoothed,
            inputs=jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), roll.inputs),
            y_fk=c["y_fk"],
            R_true=np.asarray(roll.truth["R"], dtype=np.float64),
            v_true=np.asarray(roll.truth["v"], dtype=np.float64),
            p_true=np.asarray(roll.truth["p"], dtype=np.float64),
            t_lo=t_lo, t_hi=t_hi, meta=roll.meta,
        )
        _assert_float64(prep)
        out.append(prep)
        if verbose:
            print(f"  prepared {prep.name}: T={T}, starts=[{t_lo}, {t_hi}] "
                  f"({prep.n_starts} legal)")
    return out


def _assert_float64(prep: PreparedRollout) -> None:
    """I8 at the dataset boundary: a float32 leaf here silently downcasts the filter."""
    leaves = [prep.smoothed, prep.y_fk, prep.R_true, prep.v_true, prep.p_true]
    leaves += list(jax.tree.leaves(prep.inputs))
    bad = [x.dtype for x in leaves if np.asarray(x).dtype != np.float64]
    if bad:
        raise TypeError(f"{prep.name}: float64 required at the dataset boundary, got {bad}")


def _constant_contact_chol(cfg: ContactNetConfig, L: int, N_c: int) -> np.ndarray:
    r"""``(L, N_c, 3, 3)`` of ``contact_chol_const · I₃`` — trap (b).

    The single place the sim's stance/swing ground truth is destroyed.  Matches
    `sim.sensors.SimSensorReader`'s layout (a scalar times the identity) so the
    only thing that changes is that the scalar no longer moves.
    """
    return np.broadcast_to(
        cfg.contact_chol_const * np.eye(3, dtype=np.float64), (L, N_c, 3, 3)).copy()


def _segment_window_indices(t0: int, cfg: ContactNetConfig) -> np.ndarray:
    r"""``(L, H)`` absolute tick indices, the ``[t0 : t0+L]`` rows of
    `features.window_indices`.

    Identical by construction, *without* the lower clamp: `valid_start_range`
    guarantees ``t0 ≥ (H-1)·stride``, so no row would clamp.  Asserted rather
    than assumed — a clamped row is a fabricated window and the whole point of
    the lead-in bound.
    """
    k = t0 + np.arange(cfg.L)[:, None]
    h = np.arange(cfg.H)[None, :]
    idx = k - (cfg.H - 1 - h) * cfg.stride
    if idx.min() < 0:
        raise ValueError(f"segment at t0={t0} would clamp its window (min index {idx.min()})")
    return idx


def make_segment(prep: PreparedRollout, t0: int, cfg: ContactNetConfig,
                 P0: np.ndarray) -> Segment:
    """One `rollout.Segment` starting at tick `t0`, NumPy leaves throughout."""
    if not (prep.t_lo <= t0 <= prep.t_hi):
        raise ValueError(
            f"{prep.name}: t0={t0} outside the legal range [{prep.t_lo}, {prep.t_hi}]")
    sl = slice(t0, t0 + cfg.L)

    windows = prep.smoothed[_segment_window_indices(t0, cfg)]     # (L, H, N_c, F)
    windows = np.swapaxes(windows, 1, 2)                          # (L, N_c, H, F)

    inputs = jax.tree.map(lambda a: np.asarray(a[sl]), prep.inputs)
    if cfg.freeze_contact_chol:
        # Run-1 behaviour, retained only for reproducibility -- measured at 10.2x
        # worse than not using contacts at all.  See `ContactNetConfig`.
        inputs = inputs._replace(
            contact_chol=_constant_contact_chol(cfg, cfg.L, prep.y_fk.shape[1]))

    R0, p0 = prep.R_true[t0], prep.p_true[t0]
    d0 = np.einsum("ij,kj->ki", R0, prep.y_fk[t0]) + p0[None, :]
    state0 = InEKFState(R=R0, v=prep.v_true[t0], p=p0, d=d0, P=np.asarray(P0))

    return Segment(inputs=inputs, windows=windows, state0=state0,
                   v_true=prep.v_true[sl], R_true=prep.R_true[sl])


def sample_starts(rng: np.random.Generator, preps: Sequence[PreparedRollout],
                  B: int) -> list[tuple[int, int]]:
    r"""``B`` ``(rollout_index, t0)`` pairs, spread across rollouts — docstring (d).

    A fresh permutation per batch, cycled if ``B > n_rollouts``, so the batch
    touches ``min(B, n_rollouts)`` distinct rollouts and no rollout is over-
    represented by more than one segment.  Starts are uniform over each
    rollout's legal range.
    """
    n = len(preps)
    if n == 0:
        raise ValueError("no prepared rollouts")
    picks, k = [], 0
    while len(picks) < B:
        order = rng.permutation(n)
        picks.extend(int(i) for i in order[:B - len(picks)])
        k += 1
    return [(i, int(rng.integers(preps[i].t_lo, preps[i].t_hi + 1))) for i in picks]


def make_batch(preps: Sequence[PreparedRollout], picks: Sequence[tuple[int, int]],
               cfg: ContactNetConfig, P0: np.ndarray) -> Segment:
    """Stack the given ``(rollout, t0)`` picks into one batched `Segment` on device."""
    segs = [make_segment(preps[i], t0, cfg, P0) for i, t0 in picks]
    stacked = jax.tree.map(lambda *xs: np.stack(xs), *segs)
    return jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), stacked)


def batch_stream(preps: Sequence[PreparedRollout], cfg: ContactNetConfig, P0: np.ndarray,
                 *, steps: int, seed: int = 0) -> Iterator[Segment]:
    """`steps` batches of `cfg.B` segments, sampled as `sample_starts` describes.

    Run-1 sampler: independent uniform starts, every segment re-seeded from
    ground truth.  Superseded by `ChainedBatcher` (docstring (f)) and kept for
    the ablation that shows the difference.
    """
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield make_batch(preps, sample_starts(rng, preps, cfg.B), cfg, P0)


# ---------------------------------------------------------------------------
# Chained segments — the deployment error distribution
# ---------------------------------------------------------------------------

@dataclass
class _Chain:
    """One filter chain: where it is, and how long it has been running."""
    rollout: int
    t: int                   # next segment start, in rollout ticks
    ticks: int               # ticks since this chain was seeded
    reseeds: int = 0


class ChainedBatcher:
    r"""``B`` filter chains walked through the rollouts in order, carrying ``(X̂, P)``.

    Replaces `batch_stream`'s independent uniform starts.  The point is the
    **error distribution at a segment start**: re-seeding from ground truth every
    segment means every training sample begins at exactly zero error, and over a
    128 ms horizon the loss-minimising ``Σ_C`` is then infinite (PORT_NOTES,
    "Run 1 learned to switch the contact update OFF").  Carrying the state means
    a segment starts wherever the filter actually got to.

    Lifecycle of one chain:

    1. **Seed** at a random legal start of a random rollout, from ground truth.
    2. **Warm in** ``cfg.warm_in_ticks`` with the filter run forward but *not*
       trained on, so the error grows to its natural level before the chain
       contributes a gradient.
    3. **Walk** forward ``cfg.L`` ticks per training step, carrying
       ``(X̂, P, gravity_ref)`` between steps.
    4. **Re-seed** on any of: the episode reaching ``cfg.episode_ticks``, the
       cursor running past the rollout's usable end, or the carry going
       non-finite (a diverged chain would otherwise poison every later step).

    Warm-in runs at ``Σ_C = σ₀² I`` rather than at the current network output.
    That is exact at step 0 and an approximation at later re-seeds; the
    discrepancy decays over an episode 10× longer than the warm-in, and the
    alternative (threading live params into the sampler) couples the data
    pipeline to the optimiser for a second-order effect.

    The carry is **detached** — chains provide an initial condition, not a
    gradient path.  Truncated BPTT is the whole reason ``L`` bounds anything;
    see `rollout.make_segment_loss`.

    Trade-off worth naming: `batch_stream` spread each batch across rollouts on
    purpose (docstring (d)).  A chain is pinned to one rollout for a whole
    episode, so that decorrelation now comes from having ``B`` chains seeded
    independently at different rollouts and phases, rather than from re-drawing
    every step.  With ``B = 32`` over 12 rollouts that is ~2.7 chains per
    rollout at unrelated gait phases.
    """

    def __init__(self, preps: Sequence[PreparedRollout], cfg: ContactNetConfig,
                 P0: np.ndarray, warm_in_fn, *, seed: int = 0):
        """
        Parameters
        ----------
        warm_in_fn : callable
            ``(state0, inputs) -> carry`` — runs the filter forward over the
            warm-in slice and returns the final carry.  Injected rather than
            built here so this module keeps its "no MJX, no estimator build"
            property (module docstring, pass 3); `rollout.make_warm_in` supplies it.
        """
        if not preps:
            raise ValueError("no prepared rollouts")
        self.preps, self.cfg, self.P0 = list(preps), cfg, np.asarray(P0)
        self._warm_in = warm_in_fn
        self.rng = np.random.default_rng(seed)
        self.chains: list[_Chain] = []
        self.carries: list = []
        for _ in range(cfg.B):
            c, carry = self._seed()
            self.chains.append(c)
            self.carries.append(carry)

    # -- internals ---------------------------------------------------------

    def _span(self) -> int:
        """Ticks a chain needs beyond its start: warm-in plus one segment."""
        return self.cfg.warm_in_ticks + self.cfg.L

    def _seed(self) -> tuple[_Chain, object]:
        """Pick a rollout and start, seed from truth, and warm in."""
        cfg = self.cfg
        # Only rollouts with room for warm-in plus at least one scored segment.
        room = [i for i, p in enumerate(self.preps)
                if p.t_hi - p.t_lo >= self._span()]
        if not room:
            raise ValueError(
                f"no rollout has room for warm_in_ticks + L = {self._span()} "
                f"ticks; shorten warm_in_s or collect longer rollouts")
        i = int(self.rng.choice(room))
        p = self.preps[i]
        t_seed = int(self.rng.integers(p.t_lo, p.t_hi - self._span() + 1))

        state0 = make_segment(p, t_seed, cfg, self.P0).state0
        warm = jax.tree.map(
            lambda a: jnp.asarray(a[t_seed:t_seed + cfg.warm_in_ticks]), p.inputs)
        if cfg.freeze_contact_chol:
            warm = warm._replace(contact_chol=jnp.asarray(
                _constant_contact_chol(cfg, cfg.warm_in_ticks, p.y_fk.shape[1])))
        carry = self._warm_in(jax.tree.map(jnp.asarray, state0), warm)
        return _Chain(rollout=i, t=t_seed + cfg.warm_in_ticks, ticks=cfg.warm_in_ticks), carry

    def _needs_reseed(self, c: _Chain, carry) -> bool:
        p = self.preps[c.rollout]
        if c.t + self.cfg.L > p.t_hi:
            return True
        if c.ticks >= self.cfg.episode_ticks:
            return True
        # A diverged chain never recovers and would poison every later step.
        return not bool(jnp.all(jnp.isfinite(carry.state.P))
                        and jnp.all(jnp.isfinite(carry.state.v)))

    # -- public ------------------------------------------------------------

    def batch(self) -> tuple[Segment, object]:
        """``(segment batched over B, carry batched over B)`` for one training step."""
        segs = [make_segment(self.preps[c.rollout], c.t, self.cfg, self.P0)
                for c in self.chains]
        stacked = jax.tree.map(lambda *xs: np.stack(xs), *segs)
        batch = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), stacked)
        carry = jax.tree.map(lambda *xs: jnp.stack(xs), *self.carries)
        return batch, jax.lax.stop_gradient(carry)

    def update(self, carry_out) -> int:
        """Store the step's final carries, advance cursors, re-seed as needed.

        Returns the number of chains re-seeded, which `train` logs: a rate that
        climbs mid-run means chains are diverging, not that episodes are ending.
        """
        n = len(self.chains)
        per_chain = [jax.tree.map(lambda a, i=i: a[i], carry_out) for i in range(n)]
        reseeded = 0
        for b in range(n):
            c = self.chains[b]
            c.t += self.cfg.L
            c.ticks += self.cfg.L
            self.carries[b] = per_chain[b]
            if self._needs_reseed(c, per_chain[b]):
                self.chains[b], self.carries[b] = self._seed()
                self.chains[b].reseeds = c.reseeds + 1
                reseeded += 1
        return reseeded

    def stream(self, steps: int) -> Iterator[tuple[Segment, object]]:
        """`steps` ``(batch, carry)`` pairs.  Caller must `update` between them."""
        for _ in range(steps):
            yield self.batch()


# ---------------------------------------------------------------------------
# P0 — measured, not guessed
# ---------------------------------------------------------------------------

def measure_p0(fused, prep: PreparedRollout, cfg: ContactNetConfig, *,
               t0: int | None = None, ticks: int = 3_000,
               verbose: bool = True) -> np.ndarray:
    r"""The InEKF's own converged covariance under **training** conventions.

    A segment is seeded from ground truth, so its ``P0`` should be what a
    continuously-running filter carries at that point — not the diffuse
    ``initial_covariance = 1.0`` prior, which would make the first ticks' NIS
    ≈ 0 and put `train.Metrics.nis_over_dof` on a ramp that has nothing to do
    with the network.

    Measuring it beats picking it, and the measurement is cheap: run the InEKF
    alone (no joint KF, no MJX beyond the FK the step already does) over
    ``ticks`` of recorded input from a diffuse prior, with ``contact_chol``
    frozen at the constant and ``contact_meas_chol`` at zero — i.e. exactly the
    conventions a training segment runs under, at the network's initialization
    where ``Σ_C = σ₀²I`` is negligible against ``J Σ_q Jᵀ``.  Take the final
    ``P``.

    Returned as one ``(3N+9, 3N+9)`` matrix shared by every segment.  A
    per-gait-phase ``P0`` would be more faithful still — it would mean storing
    ``P`` at every tick, 111 MB per rollout — and is the obvious refinement if
    the seeding transient ever shows up in the loss.
    """
    from ..inEKF import ekf as inekf_mod

    t0 = prep.t_lo if t0 is None else int(t0)
    if t0 + ticks > prep.smoothed.shape[0]:
        raise ValueError(f"burn-in [{t0}, {t0 + ticks}) runs past T={prep.smoothed.shape[0]}")

    xs = jax.tree.map(lambda a: jnp.asarray(a[t0:t0 + ticks]), prep.inputs)
    xs = xs._replace(
        contact_chol=jnp.asarray(_constant_contact_chol(cfg, ticks, prep.y_fk.shape[1])),
        contact_meas_chol=jnp.zeros_like(xs.contact_meas_chol),
    )
    d0 = jnp.asarray(np.einsum("ij,kj->ki", prep.R_true[t0], prep.y_fk[t0])
                     + prep.p_true[t0][None, :])
    state0 = inekf_mod.initialize(
        fused.ekf, rotation=jnp.asarray(prep.R_true[t0]), velocity=jnp.asarray(prep.v_true[t0]),
        position=jnp.asarray(prep.p_true[t0]), contacts=d0)

    step = make_step(fused.ekf, fused.kinematics)
    carry, _ = jax.lax.scan(step, init_carry(state0), xs)
    P = np.asarray(carry.state.P, dtype=np.float64)
    if not np.all(np.isfinite(P)):
        raise RuntimeError("burn-in produced a non-finite covariance")
    P = 0.5 * (P + P.T)
    w = np.linalg.eigvalsh(P)
    # A converged Joseph-form P lands a hair below zero in its smallest direction
    # (measured -1.4e-16 against a 1.0 spectral radius): float64 rounding, not a
    # broken filter.  Nudge that to strictly PD so downstream Choleskys cannot
    # trip on it, and raise only if the violation is real.
    if w.min() < -1.0e-8 * max(1.0, w.max()):
        raise RuntimeError(f"burn-in covariance is not PSD (min eig {w.min():.3e}, "
                           f"max {w.max():.3e}) — the burn-in diverged")
    if w.min() <= 0.0:
        P = P + (abs(w.min()) + 1.0e-15 * max(1.0, w.max())) * np.eye(P.shape[0])
        w = np.linalg.eigvalsh(P)
    if verbose:
        d = np.diag(P)
        print(f"  P0 from {prep.name} ticks [{t0}, {t0 + ticks}): "
              f"diag(R)={d[0:3]}, diag(v)={d[3:6]}, diag(p)={d[6:9]}, "
              f"diag(d)={d[9:]}, eig in [{w.min():.3e}, {w.max():.3e}]")
    return P
