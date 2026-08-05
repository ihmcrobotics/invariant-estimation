r"""Collected rollouts (`sim/collect.py`) -> batches of `rollout.Segment`.

Three passes, separated by cost and dependency:

1. `build_channel_cache` -- the only pass that needs MJX. Writes (T, N_c, F)
   channels plus (T, N_c, 3) FK vectors to data/cache/.
2. `fit_normalization` -- pools every cached rollout's usable region and freezes
   `normalize.NormConstants` to data/norm_constants.npz.
3. `prepare` / `ChainedBatcher` -- pure NumPy. No MJX, no estimator build.

Load-bearing points (see the reference dataset.py docstring for the full argument):
 (a) The full-rollout feature path OOMs (vmapped MJX FK over T=62k -> 38 GB), so
     pass 1 goes through collect.contact_channels_chunked + chunked_fk; the cache
     is what makes later passes a single np.load.
 (b) inputs.contact_chol is passed through, not frozen (the sim switches it
     1e-4 <-> 1e1 off ContactTrust). See ContactNetConfig.freeze_contact_chol.
 (c) Warm-up (meta["warmup_ticks"], the joint-KF bias plateau) plus the
     H-1 tick window lead-in bound the legal segment starts.
 (d) Batches are composed across rollouts; ChainedBatcher's B chains decorrelate.
 (e) Ticks inside a segment are never shuffled -- a segment is a trajectory.
 (f) Segments are chained, not independently re-seeded: force-teaching every
     segment from truth gives the contact update nothing to correct (CoCo
     arXiv 2605.15122 §III-B). state0 is the chain seed only. Contact anchors are
     d_i = R_true*y_i(q_hat) + p_true at inputs.joint.q (zero first residual),
     never from truth q. P0 is measured (measure_p0), not guessed.
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
    "valid_start_range", "make_segment", "sample_starts", "make_batch",
    "batch_stream", "ChainedBatcher",
]

DATA_DIR = collect.DATA_DIR
CACHE_DIR = DATA_DIR / "cache"
NORM_PATH = DATA_DIR / "norm_constants.npz"

ROLLOUT_MARKER = "sensors.encoders"
"""A key every collect.save_rollout archive has and nothing else does."""


def is_rollout(path: Path | str) -> bool:
    """Does this .npz actually hold a collect.Rollout? Structural, not name-based
    (data/ also holds norm constants, P0, checkpoints)."""
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            return ROLLOUT_MARKER in z.files
    except Exception:
        return False


def rollout_paths(data_dir: Path | str = DATA_DIR) -> list[Path]:
    """Every saved rollout .npz in data_dir, sorted; non-rollouts skipped."""
    return sorted(p for p in Path(data_dir).glob("*.npz") if is_rollout(p))


def chunked_fk(kinematics, q: np.ndarray, chunk: int = 2_000) -> np.ndarray:
    r"""kinematics(q_t, 0).y for every tick, (T, n_j) -> (T, N_c, 3). Chunked to
    bound RSS; exactly splittable (y has no cross-tick term)."""
    q = np.asarray(q, dtype=np.float64)
    fk = jax.jit(jax.vmap(lambda qq: kinematics(qq, jnp.zeros_like(qq)).y))
    out = [np.asarray(fk(jnp.asarray(q[lo:lo + chunk]))) for lo in range(0, q.shape[0], chunk)]
    return np.concatenate(out, axis=0)


def build_channel_cache(paths: Sequence[Path | str], collector: collect.Collector, *,
                        cache_dir: Path | str = CACHE_DIR, chunk: int = 2_000,
                        reuse: bool = True, verbose: bool = True) -> list[Path]:
    r"""Write raw (not normalized) channels + y_fk for every rollout in paths.

    channels is features.make_contact_channels over Rollout.sensors (raw: the
    constants are fit from this cache, so normalizing here would be circular).
    y_fk is the contact FK at the recorded filter joint estimate inputs.joint.q,
    used to seed Segment.state0's contact anchors.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fused = collector.fused
    sub = features.subchain_for(fused, collect._unfiltered_names(collector))
    channels = features.make_contact_channels(
        sub, fused.base_imu, fused.kinematics, collector.dt)

    expect_names = np.asarray(features.channel_names())
    n_c = int(fused.n_contacts)

    def _reusable(out: Path, src: Path) -> bool:
        """Is `out` a cache we can trust for `src` under the CURRENT feature code?

        Rebuilding every channel cache costs ~4 min/rollout, which dominates a
        training run whose inputs have not changed. But a stale cache is worse
        than a slow one -- it trains on features that no longer match the code
        and nothing downstream would notice -- so reuse is allowed only when all
        of these hold, and ANY mismatch falls through to a rebuild:
          * the cache is newer than the rollout it came from;
          * the channel NAMES are identical (catches a changed/reordered channel
            set, which is the realistic way this goes wrong);
          * the contact axis matches this collector's N (an N=2 cache must never
            be reused for an N=8 run).
        """
        if not (out.exists() and out.stat().st_mtime >= src.stat().st_mtime):
            return False
        try:
            with np.load(out, allow_pickle=False) as z:
                if not np.array_equal(z["names"], expect_names):
                    return False
                return z["channels"].shape[1] == n_c and z["y_fk"].shape[1] == n_c
        except Exception:
            return False

    written = []
    for p in paths:
        p = Path(p)
        out = cache_dir / f"{p.stem}_feat.npz"
        if reuse and _reusable(out, p):
            written.append(out)
            if verbose:
                print(f"  reused {out.name} (cache newer than rollout, channels match)")
            continue
        roll = collect.load_rollout(p)
        x = collect.contact_channels_chunked(channels, roll.sensors, chunk=chunk)
        y = chunked_fk(fused.kinematics, roll.inputs.joint.q, chunk=chunk)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise RuntimeError(f"{p.name}: non-finite features/FK in the cache pass")
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
    """Read one cache file back: channels, y_fk, names, meta."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {
            "channels": np.asarray(z["channels"], dtype=np.float64),
            "y_fk": np.asarray(z["y_fk"], dtype=np.float64),
            "names": tuple(str(s) for s in z["names"]),
            "meta": json.loads(str(z["meta"])),
        }


def cache_path(rollout_path: Path | str, cache_dir: Path | str = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{Path(rollout_path).stem}_feat.npz"


def fit_normalization(paths: Sequence[Path | str], *, cache_dir: Path | str = CACHE_DIR,
                      source: str = "", usable_only: bool = True) -> normalize.NormConstants:
    r"""Pool every cached rollout and freeze normalize.NormConstants.

    Pools the usable (post-warm-up) region by default: the constants must describe
    what the network is actually fed. usable_only=False only to compare. Channels
    pooled along time -> one flat calibration set across every terrain and seed.
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


@dataclass
class PreparedRollout:
    """One rollout, normalized, ready to slice segments from.

    channels = normalize.apply(cached channels) over the WHOLE stream. A
    segment's windows are then a pure gather out of it, so they are bit-identical
    to features.window over the full rollout -- normalization is per-tick, so
    unlike the boxcar this pass has no state to lose at a slice boundary.
    """
    name: str
    channels: np.ndarray          # (T, N_c, F)  normalized
    inputs: InEKFInputs           # NumPy leaves, leading axis T
    y_fk: np.ndarray              # (T, N_c, 3)  FK contact vectors at inputs.joint.q
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
    r"""(t_lo, t_hi) inclusive bounds on a segment start.
    t_lo = warmup + (H-1) (point (c)); t_hi = T - L. Raises rather than
    returning empty: a too-short rollout is a collection bug, not a skip."""
    lo = int(warmup) + (cfg.H - 1)
    hi = int(T) - cfg.L
    if hi < lo:
        raise ValueError(
            f"no legal segment start: T={T}, warmup={warmup}, lead-in="
            f"{cfg.H - 1}, L={cfg.L} leaves [{lo}, {hi}]")
    return lo, hi


def prepare(paths: Sequence[Path | str], norm: normalize.NormConstants,
            cfg: ContactNetConfig, *, cache_dir: Path | str = CACHE_DIR,
            verbose: bool = False) -> list[PreparedRollout]:
    """Load rollouts + caches, normalize, and compute the start bounds."""
    out = []
    for p in paths:
        p = Path(p)
        roll = collect.load_rollout(p)
        c = load_channel_cache(cache_path(p, cache_dir))
        if c["names"] != norm.names:
            raise ValueError(f"{p.name}: cached channel names disagree with the constants")
        x = np.asarray(normalize.apply(jnp.asarray(c["channels"]), norm))
        T = x.shape[0]
        t_lo, t_hi = valid_start_range(T, roll.meta["warmup_ticks"], cfg)
        prep = PreparedRollout(
            name=f"{roll.meta['terrain']}/seed{roll.meta['seed']}",
            channels=x,
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
    leaves = [prep.channels, prep.y_fk, prep.R_true, prep.v_true, prep.p_true]
    leaves += list(jax.tree.leaves(prep.inputs))
    bad = [x.dtype for x in leaves if np.asarray(x).dtype != np.float64]
    if bad:
        raise TypeError(f"{prep.name}: float64 required at the dataset boundary, got {bad}")


def _constant_contact_chol(cfg: ContactNetConfig, L: int, N_c: int) -> np.ndarray:
    r"""(L, N_c, 3, 3) of contact_chol_const * I3 -- the single place the sim's
    stance/swing ground truth is destroyed (point (b)). Only under freeze."""
    return np.broadcast_to(
        cfg.contact_chol_const * np.eye(3, dtype=np.float64), (L, N_c, 3, 3)).copy()


def _segment_window_indices(t0: int, cfg: ContactNetConfig) -> np.ndarray:
    r"""(L, H) absolute tick indices: the [t0:t0+L] rows of features.window_indices,
    without the lower clamp (valid_start_range guarantees t0 >= H-1)."""
    k = t0 + np.arange(cfg.L)[:, None]
    h = np.arange(cfg.H)[None, :]
    idx = k - (cfg.H - 1 - h)
    if idx.min() < 0:
        raise ValueError(f"segment at t0={t0} would clamp its window (min index {idx.min()})")
    return idx


def make_segment(prep: PreparedRollout, t0: int, cfg: ContactNetConfig,
                 P0: np.ndarray) -> Segment:
    """One rollout.Segment starting at tick t0, NumPy leaves throughout."""
    if not (prep.t_lo <= t0 <= prep.t_hi):
        raise ValueError(
            f"{prep.name}: t0={t0} outside the legal range [{prep.t_lo}, {prep.t_hi}]")
    sl = slice(t0, t0 + cfg.L)

    windows = prep.channels[_segment_window_indices(t0, cfg)]     # (L, H, N_c, F)
    windows = np.swapaxes(windows, 1, 2)                          # (L, N_c, H, F)

    inputs = jax.tree.map(lambda a: np.asarray(a[sl]), prep.inputs)
    if cfg.freeze_contact_chol:
        inputs = inputs._replace(
            contact_chol=_constant_contact_chol(cfg, cfg.L, prep.y_fk.shape[1]))

    R0, p0 = prep.R_true[t0], prep.p_true[t0]
    d0 = np.einsum("ij,kj->ki", R0, prep.y_fk[t0]) + p0[None, :]
    state0 = InEKFState(R=R0, v=prep.v_true[t0], p=p0, d=d0, P=np.asarray(P0))

    return Segment(inputs=inputs, windows=windows, state0=state0,
                   v_true=prep.v_true[sl], R_true=prep.R_true[sl])


def sample_starts(rng: np.random.Generator, preps: Sequence[PreparedRollout],
                  B: int) -> list[tuple[int, int]]:
    r"""B (rollout_index, t0) pairs, spread across rollouts (point (d)); cycled if
    B > n_rollouts, so no rollout is over-represented by more than one segment."""
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
    """Stack the given (rollout, t0) picks into one batched Segment on device."""
    segs = [make_segment(preps[i], t0, cfg, P0) for i, t0 in picks]
    stacked = jax.tree.map(lambda *xs: np.stack(xs), *segs)
    return jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), stacked)


def batch_stream(preps: Sequence[PreparedRollout], cfg: ContactNetConfig, P0: np.ndarray,
                 *, steps: int, seed: int = 0) -> Iterator[Segment]:
    """steps batches of cfg.B segments (run-1 sampler: independent uniform starts,
    every segment re-seeded from truth). Superseded by ChainedBatcher (point (f))."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield make_batch(preps, sample_starts(rng, preps, cfg.B), cfg, P0)


@dataclass
class _Chain:
    """One filter chain: where it is, and how long it has been running."""
    rollout: int
    t: int                   # next segment start, in rollout ticks
    ticks: int               # ticks since this chain was seeded
    reseeds: int = 0


class ChainedBatcher:
    r"""B filter chains walked through the rollouts in order, carrying (X_hat, P).

    Replaces batch_stream's independent uniform starts; the point is the error
    distribution at a segment start (point (f)). A chain is seeded from truth at a
    random legal start, warmed in for cfg.warm_in_ticks untrained, walked cfg.L
    ticks per step, re-seeded on episode end, rollout end, or a non-finite carry.
    Warm-in runs at Sigma_C = sigma0^2 I. The carry is detached -- chains provide
    an initial condition, not a gradient path (truncated BPTT).
    """

    def __init__(self, preps: Sequence[PreparedRollout], cfg: ContactNetConfig,
                 P0: np.ndarray, warm_in_fn, *, seed: int = 0):
        """warm_in_fn(state0, inputs) -> carry runs the filter over the warm-in
        slice; injected (rollout.make_warm_in) so this module needs no MJX."""
        if not preps:
            raise ValueError("no prepared rollouts")
        self.preps, self.cfg, self.P0 = list(preps), cfg, np.asarray(P0)
        self._warm_in = warm_in_fn
        self.rng = np.random.default_rng(seed)
        self.chains: list[_Chain] = []
        self.carries: list = []
        for b in range(cfg.B):
            # Stagger initial episode phase so the B chains do not re-seed in a
            # synchronised wave (which would correlate any phase-dependent effect
            # across the whole batch and cost the B independent samples).
            phase = int(self.rng.integers(cfg.warm_in_ticks, cfg.episode_ticks))
            c, carry = self._seed(ticks=phase)
            self.chains.append(c)
            self.carries.append(carry)

    def _span(self) -> int:
        """Ticks a chain needs beyond its start: warm-in plus one segment."""
        return self.cfg.warm_in_ticks + self.cfg.L

    def _seed(self, *, ticks: int | None = None) -> tuple[_Chain, object]:
        """Pick a rollout and start, seed from truth, and warm in."""
        cfg = self.cfg
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
        return _Chain(rollout=i, t=t_seed + cfg.warm_in_ticks,
                      ticks=cfg.warm_in_ticks if ticks is None else ticks), carry

    def _needs_reseed(self, c: _Chain, carry) -> bool:
        p = self.preps[c.rollout]
        if c.t + self.cfg.L > p.t_hi: # if rollout is over by high threshold
            return True
        if c.ticks >= self.cfg.episode_ticks: # same thing, just with ticks
            return True
        # A diverged chain never recovers and would poison every later step.
        return not bool(jnp.all(jnp.isfinite(carry.state.P))
                        and jnp.all(jnp.isfinite(carry.state.v)))

    def batch(self) -> tuple[Segment, object]:
        """(segment batched over B, carry batched over B) for one training step."""
        segs = [make_segment(self.preps[c.rollout], c.t, self.cfg, self.P0)
                for c in self.chains]
        stacked = jax.tree.map(lambda *xs: np.stack(xs), *segs)
        batch = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), stacked)
        carry = jax.tree.map(lambda *xs: jnp.stack(xs), *self.carries)
        return batch, jax.lax.stop_gradient(carry)

    def update(self, carry_out) -> int:
        """Store the step's final carries, advance cursors, re-seed as needed.
        Returns the number of chains re-seeded (train logs it: a climbing rate
        mid-run means chains are diverging, not that episodes are ending)."""
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
        """steps (batch, carry) pairs. Caller must update() between them."""
        for _ in range(steps):
            yield self.batch()


def measure_p0(fused, prep: PreparedRollout, cfg: ContactNetConfig, *,
               t0: int | None = None, ticks: int = 3_000,
               verbose: bool = True) -> np.ndarray:
    r"""The InEKF's own converged covariance under training conventions.

    A segment is seeded from truth, so P0 should be what a continuously-running
    filter carries there -- not the diffuse initial_covariance=1.0 prior (which
    would ramp the first ticks' NIS/dof). Run the InEKF alone over `ticks` of
    recorded input from a diffuse prior and take the final P. contact_chol burns
    in on the recorded heuristic; contact_meas_chol stays zero (the shipped
    N = J Sigma_q J^T, not a training convention).
    """
    from ..inEKF import ekf as inekf_mod

    t0 = prep.t_lo if t0 is None else int(t0)
    if t0 + ticks > prep.channels.shape[0]:
        raise ValueError(f"burn-in [{t0}, {t0 + ticks}) runs past T={prep.channels.shape[0]}")

    xs = jax.tree.map(lambda a: jnp.asarray(a[t0:t0 + ticks]), prep.inputs)
    if cfg.freeze_contact_chol:
        xs = xs._replace(contact_chol=jnp.asarray(
            _constant_contact_chol(cfg, ticks, prep.y_fk.shape[1])))
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
    # A converged Joseph-form P lands a hair below zero in float64 (measured
    # -1.4e-16); nudge to strictly PD, raise only if the violation is real.
    if w.min() < -1.0e-8 * max(1.0, w.max()):
        raise RuntimeError(f"burn-in covariance is not PSD (min eig {w.min():.3e}, "
                           f"max {w.max():.3e}) -- the burn-in diverged")
    if w.min() <= 0.0:
        P = P + (abs(w.min()) + 1.0e-15 * max(1.0, w.max())) * np.eye(P.shape[0])
        w = np.linalg.eigvalsh(P)
    if verbose:
        d = np.diag(P)
        print(f"  P0 from {prep.name} ticks [{t0}, {t0 + ticks}): "
              f"diag(R)={d[0:3]}, diag(v)={d[3:6]}, diag(p)={d[6:9]}, "
              f"diag(d)={d[9:]}, eig in [{w.min():.3e}, {w.max():.3e}]")
    return P
