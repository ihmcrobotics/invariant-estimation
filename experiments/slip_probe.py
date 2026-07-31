r"""Is slip PREDICTABLE from ContactNet's 24 feature channels at all?

This is the gate that decides whether a slip-heavy retrain is worth a GPU hour.

The hypothesis under test
-------------------------
Run 4's learned covariance has median ``std_x = 3.85e-4`` against
``std_z = 1.67e-1`` -- ~400x TIGHTER fore-aft than vertical.  If the network never
inflates the HORIZONTAL block, a sliding foot drags the base and produces
horizontal drift.  Two questions, in order:

**(a) Does the learned covariance track slip at all, or only phase?**
   ``R^2`` of ``log10 std_{x,y,z}`` on recorded ``truth.slip_sat`` versus on gait
   phase, loaded ticks only (``truth.contact_fn > 0``).  Both predictors are
   quantile-binned to the SAME bin count, so the two ``R^2`` are comparable --
   `phase_lock.r2_on_phase` uses 300 fixed-width phase bins against what would be
   ~20 slip bins, and that alone would hand phase a ~0.1 advantage for free.

**(b) Is slip predictable from the features at all?**
   A cheap probe -- ridge and gradient-boosted trees -- from the cached channels
   to ``slip_sat``, held out BY ROLLOUT.  If this is ~0, the network is
   structurally blind to slip, no amount of richer motion data helps, and the fix
   is new feature channels (a CLAUDE.md §7 change).

Why the held-out split must be by rollout
-----------------------------------------
``slip_sat = |f_t| / (mu f_n)``.  Leg torques and the contact Jacobian determine
the GRF, so ``|f_t|/f_n`` is in principle recoverable from the channels.  ``mu``
is not: it is a domain-randomisation draw, constant within a rollout, and
observable only through having already slipped.  A random within-rollout split
therefore lets the probe memorise that rollout's ``mu`` from any correlate of it
and reports an ``R^2`` that will not survive deployment.  Both splits are
reported; the by-rollout one is the answer.

The two controls
----------------
* **load** (``contact_fn``) -- predicted from the same features.  If load is
  predictable and slip is not, the features carry the *normal* channel and are
  missing the *tangential/friction* one, which names the fix.
* **cone ratio** (``|f_t|/f_n = mu * slip_sat``) -- the mu-free part of the
  target.  If the probe predicts this but not ``slip_sat``, the missing quantity
  is mu, not the force.

Usage
-----
    uv run --with scikit-learn python -m experiments.slip_probe \
        --data data/dr5 --checkpoint artifacts/contactnet_run7.npz
    uv run --with scikit-learn python -m experiments.slip_probe \
        --data data/dr4 --checkpoint artifacts/contactnet_run6_w128.npz
    uv run --with scikit-learn python -m experiments.slip_probe --data data/dr5 --no-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

WARMUP = 16_000
"""Ticks of lead-in the collector prepends; the same number `phase_lock` drops."""


# ---------------------------------------------------------------------------
# A comparable R^2: one nonparametric 1-D fit, same flexibility for every predictor
# ---------------------------------------------------------------------------

def r2_binned(y: np.ndarray, x: np.ndarray, n_bins: int = 24) -> float:
    r"""Variance of ``y`` explained by a binned conditional mean of ``x``.

    QUANTILE bins, so every predictor gets the same number of bins with roughly
    the same occupancy regardless of its distribution.  That is the only way the
    phase and slip numbers can be put in the same table: fixed-width binning gives
    a long-tailed predictor like phase far more effective degrees of freedom than
    a bounded one like ``slip_sat``.

    Returns NaN rather than a number when there is not enough data to be worth
    reading -- a silent 0.0 there would read as "no relationship".
    """
    y = np.asarray(y, float).ravel()
    x = np.asarray(x, float).ravel()
    good = np.isfinite(y) & np.isfinite(x)
    y, x = y[good], x[good]
    if y.size < 500 or np.var(y) == 0.0:
        return float("nan")
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return float("nan")
    b = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    pred = np.zeros_like(y)
    for k in range(edges.size - 1):
        sel = b == k
        if sel.sum() > 3:
            pred[sel] = y[sel].mean()
        else:
            pred[sel] = y.mean()
    return float(1.0 - np.var(y - pred) / np.var(y))


def gait_phase(trust: np.ndarray, hi: float = 0.5) -> np.ndarray:
    """Ticks since this foot's last touchdown; NaN before the first one.

    Identical to `phase_lock.gait_phase` -- duplicated rather than imported so the
    two scripts stay independently runnable, and asserted equal in
    `tests/experiments/test_slip_probe.py`.
    """
    on = trust > hi
    rise = np.zeros_like(on)
    rise[1:] = on[1:] & ~on[:-1]
    idx = np.arange(trust.shape[0])[:, None]
    last = np.maximum.accumulate(np.where(rise, idx, -1), axis=0)
    ph = (idx - last).astype(float)
    ph[last < 0] = np.nan
    return ph


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def rollout_paths(data: Path) -> list[Path]:
    return [p for p in sorted(data.glob("*.npz")) if p.name != "norm_constants.npz"]


def load_targets(path: Path, stride: int) -> dict:
    """Per-FOOT targets and the masks that say which samples are evidence."""
    with np.load(path, allow_pickle=True) as z:
        sl = np.asarray(z["truth.slip_sat"])[WARMUP::stride]
        fn = np.asarray(z["truth.contact_fn"])[WARMUP::stride]
        tr = np.asarray(z["sensors.contact"])[WARMUP:]
        meta = json.loads(str(z["meta"]))
    ph = gait_phase(tr)[::stride]
    mu = float(meta.get("friction_mu", np.nan))
    return {"slip": sl, "fn": fn, "phase": ph, "mu": mu,
            "terrain": meta.get("terrain", "?"), "loaded": fn > 0.0}


def load_channels(cache: Path, stem: str, stride: int) -> np.ndarray | None:
    """`(T', N_c, 24)` cached per-contact channels, subsampled like the targets."""
    p = cache / f"{stem}_feat.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return np.asarray(z["channels"])[WARMUP::stride]


CHANNEL_GROUPS = {
    "gyro": (0, 3), "accel": (3, 6), "q": (6, 12), "tau": (12, 18),
    "p_bc": (18, 21), "v_bc": (21, 24),
}
"""`features.channel_names` ordering, as slices. Used by `--groups` to ask WHICH
channel family carries the slip signal -- the answer names the cheapest fix."""


def select_channels(chan: np.ndarray, groups: list[str] | None) -> np.ndarray:
    """`(T, N_c, 24)` restricted to the named channel families (None = all)."""
    if not groups:
        return chan
    cols = np.concatenate([np.arange(*CHANNEL_GROUPS[g]) for g in groups])
    return chan[..., cols]


def foot_features(chan: np.ndarray, taps: np.ndarray, tap_stride: int) -> np.ndarray:
    r"""`(T', K, D)` per-FOOT feature matrix from `(T', N_c, 24)` channels.

    Two things are folded in:

    * **contact -> foot.**  At ``N = 4`` the contacts are foot-major (left heel,
      left toe, right heel, right toe -- `main_estimator.TOE_HEEL_SITES`), while
      ``slip_sat`` is per foot.  Both of a foot's contacts are concatenated, so the
      probe sees strictly more than the network does per output.
    * **history.**  ``taps`` selects lags (in units of ``tap_stride`` ticks) ending
      at the current sample, reproducing the network's causal window at coarser
      temporal resolution.  ``taps = [0]`` is the instantaneous set.

    Lags are clamped at 0, matching `features.window_indices`; only the lower
    bound, never the upper -- an upper clamp would hide a future-peeking bug.
    """
    T, n_c, F = chan.shape
    K = n_c if n_c <= 2 else n_c // 2
    per = n_c // K
    idx = np.arange(T)[:, None] - taps[None, :] * tap_stride
    np.clip(idx, 0, None, out=idx)
    # (T, n_taps, n_c, F) -> (T, K, per * n_taps * F)
    g = chan[idx]
    g = g.transpose(0, 2, 1, 3).reshape(T, K, per, -1)
    return g.reshape(T, K, per * len(taps) * F)


def assemble(paths: list[Path], cache: Path, stride: int, taps: np.ndarray,
             tap_stride: int, groups: list[str] | None = None) -> list[dict]:
    """One record per rollout: features, targets, masks, group id."""
    out = []
    for p in paths:
        chan = load_channels(cache, p.stem, stride)
        if chan is None:
            continue
        chan = select_channels(chan, groups)
        t = load_targets(p, stride)
        n = min(len(chan), len(t["slip"]))
        X = foot_features(chan[:n], taps, tap_stride)
        m = t["loaded"][:n]
        out.append({
            "name": p.stem, "terrain": t["terrain"], "mu": t["mu"],
            "X": X.reshape(-1, X.shape[-1])[m.ravel()],
            "slip": t["slip"][:n].ravel()[m.ravel()],
            "fn": t["fn"][:n].ravel()[m.ravel()],
            "phase": t["phase"][:n].ravel()[m.ravel()],
        })
    return out


# ---------------------------------------------------------------------------
# (b) the probe
# ---------------------------------------------------------------------------

def ridge_r2(Xtr, ytr, Xte, yte, lam: float = 1.0) -> float:
    """Held-out R^2 of a standardised ridge, closed form.

    Standardisation uses the TRAIN statistics only.  Zero-variance columns (a
    channel that never moves in the training rollouts) are passed through as
    zeros rather than dividing by ~0.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean()
    G = A.T @ A + lam * len(A) * np.eye(A.shape[1])
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    pred = B @ w + ym
    return float(1.0 - np.mean((yte - pred) ** 2) / np.var(yte))


def gbm_r2(Xtr, ytr, Xte, yte, seed: int = 0) -> float:
    from sklearn.ensemble import HistGradientBoostingRegressor
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                      max_depth=6, random_state=seed)
    g.fit(Xtr, ytr)
    pred = g.predict(Xte)
    return float(1.0 - np.mean((yte - pred) ** 2) / np.var(yte))


def split_by_rollout(recs: list[dict], n_test: int, seed: int = 0):
    """Held-out split at the ROLLOUT level -- the honest one (see the docstring)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    te, tr = order[:n_test], order[n_test:]
    return [recs[i] for i in tr], [recs[i] for i in te]


def split_random(recs: list[dict], frac: float = 0.25, seed: int = 0):
    """Within-rollout random split -- deliberately OPTIMISTIC, reported as a foil."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for r in recs:
        m = rng.random(len(r["slip"])) < frac
        te.append({k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in r.items()})
        tr.append({k: (v[~m] if isinstance(v, np.ndarray) else v) for k, v in r.items()})
    return tr, te


def stack(recs, key):
    return np.concatenate([r[key] for r in recs])


def run_probe(recs: list[dict], target: str, args) -> dict:
    """Held-out R^2 for one target under both splits and both model families."""
    out = {}
    for label, (tr, te) in (("by-rollout", split_by_rollout(recs, args.n_test, args.seed)),
                            ("within-rollout", split_random(recs, seed=args.seed))):
        Xtr, ytr = stack(tr, "X"), stack(tr, target)
        Xte, yte = stack(te, "X"), stack(te, target)
        if args.max_train and len(Xtr) > args.max_train:
            sel = np.random.default_rng(args.seed).choice(len(Xtr), args.max_train, False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        out[f"{label}/ridge"] = ridge_r2(Xtr, ytr, Xte, yte, args.lam)
        if not args.no_gbm:
            out[f"{label}/gbm"] = gbm_r2(Xtr, ytr, Xte, yte, args.seed)
        out[f"{label}/n_train"] = len(Xtr)
        out[f"{label}/n_test"] = len(Xte)
    return out


# ---------------------------------------------------------------------------
# (a) the model side
# ---------------------------------------------------------------------------

def model_stds(paths, cache, checkpoint, norm_path, stride):
    """`(rollout -> (T', N_c, 3))` per-axis learned std, on the same subsample."""
    import jax
    import jax.numpy as jnp
    from invariant_estimation.contactnet import features, network, normalize, train
    from invariant_estimation.contactnet.config import ContactNetConfig

    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4)
    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    params = train.load_params(checkpoint, like)
    nc = normalize.load(str(norm_path))
    fwd = jax.jit(jax.vmap(jax.vmap(lambda x: network.forward(params, x, cfg.eps))))

    out = {}
    for p in paths:
        c = cache / f"{p.stem}_feat.npz"
        if not c.exists():
            continue
        with np.load(c) as z:
            chan = np.asarray(z["channels"])
        starts = np.arange(WARMUP, chan.shape[0], stride)
        w = features.window(normalize.apply(jnp.asarray(chan), nc), cfg.H, cfg.stride)
        sd = []
        # Chunked: the (len(starts), N_c, H*F) gather is ~10 GB at stride 20 in one go.
        for k in range(0, len(starts), 4096):
            s = starts[k:k + 4096]
            L = fwd(w[s].reshape(len(s), chan.shape[1], -1))
            S = np.asarray(jnp.einsum("...ij,...kj->...ik", L, L))
            sd.append(np.sqrt(np.diagonal(S, axis1=-2, axis2=-1)))
        out[p.stem] = np.concatenate(sd)
    return out


def model_side(paths, cache, checkpoint, norm_path, stride, n_bins):
    """R^2 of log10 std_{x,y,z} on slip and on phase, pooled, matched bins."""
    stds = model_stds(paths, cache, checkpoint, norm_path, stride)
    ys, sl, ph = [], [], []
    for p in paths:
        if p.stem not in stds:
            continue
        t = load_targets(p, stride)
        sd = stds[p.stem]
        n = min(len(sd), len(t["slip"]))
        sd = sd[:n]
        K = t["slip"].shape[1]
        rep = sd.shape[1] // K
        s = np.repeat(t["slip"][:n], rep, axis=1)
        f = np.repeat(t["fn"][:n], rep, axis=1)
        q = np.repeat(t["phase"][:n], rep, axis=1)
        m = f > 0.0
        ys.append(np.log10(np.maximum(sd, 1e-30))[m])
        sl.append(s[m])
        ph.append(q[m])
    if not ys:
        return None
    Y, S, P = np.concatenate(ys), np.concatenate(sl), np.concatenate(ph)
    return {ax: {"slip": r2_binned(Y[:, i], S, n_bins),
                 "phase": r2_binned(Y[:, i], P, n_bins),
                 "median_std": float(np.median(10.0 ** Y[:, i]))}
            for i, ax in enumerate("xyz")}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(REPO / "data" / "dr5"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-model", action="store_true", help="skip part (a)")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--stride", type=int, default=20, help="tick subsample")
    ap.add_argument("--n-taps", type=int, default=8, help="history taps per feature")
    ap.add_argument("--tap-stride", type=int, default=56, help="ticks between taps")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=4, help="held-out rollouts")
    ap.add_argument("--max-train", type=int, default=120_000)
    ap.add_argument("--lam", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default=None,
                    help="comma-separated channel families to keep "
                         f"({'|'.join(CHANNEL_GROUPS)}); default all 24")
    ap.add_argument("--group-ablation", action="store_true",
                    help="run the slip probe once per channel family, and once with each "
                         "family REMOVED -- names which channels carry the signal")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    groups = args.groups.split(",") if args.groups else None

    data = Path(args.data)
    cache = Path(args.cache) if args.cache else data / "cache"
    norm = Path(args.norm) if args.norm else data / "norm_constants.npz"
    paths = rollout_paths(data)
    print(f"slip probe on {data} ({len(paths)} rollouts, stride {args.stride})")

    report: dict = {"data": str(data), "stride": args.stride}

    # -- (a) ---------------------------------------------------------------
    if args.checkpoint and not args.no_model:
        ms = model_side(paths, cache, args.checkpoint, norm, args.stride, args.n_bins)
        if ms:
            report["model_side"] = ms
            print(f"\n(a) learned covariance, {args.checkpoint}")
            print(f"    {'axis':>5} {'median std':>12} {'R2 on slip':>12} {'R2 on phase':>12}")
            for ax in "xyz":
                r = ms[ax]
                print(f"    {ax:>5} {r['median_std']:12.3e} {r['slip']:12.3f} "
                      f"{r['phase']:12.3f}")

    # -- (b) ---------------------------------------------------------------
    taps = np.arange(args.n_taps)
    # `tap_stride` is in raw ticks; the features are already subsampled by `stride`.
    ts = max(1, args.tap_stride // args.stride)
    recs = assemble(paths, cache, args.stride, taps, ts, groups)
    if not recs:
        print("\n  no cached features -- part (b) skipped")
        return
    d = recs[0]["X"].shape[1]
    n = sum(len(r["slip"]) for r in recs)
    mus = np.array([r["mu"] for r in recs])
    print(f"\n(b) probe: {n} loaded foot-samples, {d} features "
          f"({args.n_taps} taps x {ts * args.stride} ticks), "
          f"mu in [{np.nanmin(mus):.2f}, {np.nanmax(mus):.2f}] over "
          f"{len(np.unique(np.round(mus, 4)))} distinct values")

    # The mu-free part of the target, and the load control.
    for r in recs:
        r["cone"] = r["slip"] * r["mu"]
        r["load"] = r["fn"]

    rows = {}
    for tgt, label in (("slip", "slip_sat  (the question)"),
                       ("cone", "|f_t|/f_n  (mu removed)"),
                       ("load", "contact_fn (control)")):
        rows[tgt] = run_probe(recs, tgt, args)
        print(f"\n    target {label}")
        for k in sorted(rows[tgt]):
            if k.endswith("n_train") or k.endswith("n_test"):
                continue
            print(f"      {k:28s} R2 = {rows[tgt][k]:+.3f}")
    report["probe"] = rows

    if args.group_ablation:
        # ONE family at a time, and one family REMOVED. Both are needed: a family can
        # score well alone and be redundant, or score nothing alone and still be the
        # only thing carrying a term the rest cannot express.
        print(f"\n    channel ablation (target slip_sat, by-rollout GBM)")
        base = rows["slip"]["by-rollout/gbm"]
        print(f"      {'channels':>22}  {'R2':>7}  {'vs all 24':>9}")
        print(f"      {'ALL 24':>22}  {base:+7.3f}  {'--':>9}")
        abl = {"all": base}
        for g in CHANNEL_GROUPS:
            for keep, tag in ((([g]), f"only {g}"), ([k for k in CHANNEL_GROUPS if k != g],
                                                    f"drop {g}")):
                r = assemble(paths, cache, args.stride, taps, ts, keep)
                v = run_probe(r, "slip", args)["by-rollout/gbm"]
                abl[tag] = v
                print(f"      {tag:>22}  {v:+7.3f}  {v - base:+9.3f}")
        report["group_ablation"] = abl

    print("\n  read: by-rollout is the deployable number; within-rollout can memorise "
          "\n  the rollout's mu and is reported only as an upper bound.")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
        print(f"  -> {args.json}")


if __name__ == "__main__":
    main()
