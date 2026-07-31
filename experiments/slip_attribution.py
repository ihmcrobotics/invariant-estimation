r"""How much of the horizontal drift is SLIP, and how much is everything else?

The hypothesis: a sliding foot drags the base, because the InEKF's contact anchor is
still asserted world-static while the real contact point translates, so the FK residual
is absorbed partly by the base.  If that is the dominant term, horizontal drift should
scale with how much the feet actually slipped.

Two independent tests, because either alone is weak:

**Between rollouts** (n = 12).  Rollout-level horizontal drift rate against that
rollout's slip fraction and its `mu`.  A real effect shows up as a monotone relation
across a friction sweep.  Small n, so the correlation is reported with its scatter and
not dressed up as a fit.

**Within a rollout** (n = thousands).  The estimator's horizontal error is a random walk
plus drift, so the quantity to regress is its per-window INCREMENT, not its level:

.. math::
    \Delta_k = \| (\hat p - p)_{xy}(t_{k+1}) \| - \| (\hat p - p)_{xy}(t_k) \|

against the mean friction-cone saturation of the loaded samples in that window.  Levels
would correlate with anything that grows monotonically, including time itself.

Every number is read from the collected rollouts -- the estimator ran during collection
and its output is in `aux.est_p` -- so nothing here re-runs a filter.

Usage
-----
    uv run python -m experiments.slip_attribution --data data/dr5
    uv run python -m experiments.slip_attribution --data data/dr6 --window 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WARMUP = 16_000
DT = 1.0e-3


def load(path: Path) -> dict | None:
    with np.load(path, allow_pickle=True) as z:
        if "aux.est_p" not in z.files or "truth.slip_sat" not in z.files:
            return None
        est = np.asarray(z["aux.est_p"])[WARMUP:]
        tru = np.asarray(z["truth.p"])[WARMUP:]
        sl = np.asarray(z["truth.slip_sat"])[WARMUP:]
        fn = np.asarray(z["truth.contact_fn"])[WARMUP:]
        meta = json.loads(str(z["meta"]))
    err = np.linalg.norm((est - tru)[:, :2], axis=1)
    # Re-zero: the warm-up leaves a constant offset that is not this window's drift.
    return {"name": path.stem, "err": err - err[0], "slip": sl, "fn": fn,
            "true": tru, "mu": float(meta.get("friction_mu", np.nan)),
            "terrain": meta.get("terrain", "?")}


def per_rollout(r: dict) -> dict:
    loaded = r["fn"] > 0.0
    n = int(loaded.sum())
    travelled = float(np.abs(np.diff(r["true"][:, :2], axis=0)).sum())
    secs = len(r["err"]) * DT
    return {
        "name": r["name"], "terrain": r["terrain"], "mu": r["mu"],
        "slip_frac": float((r["slip"][loaded] >= 0.99).sum() / n) if n else 0.0,
        "slip_mean": float(r["slip"][loaded].mean()) if n else 0.0,
        "drift_m": float(r["err"][-1]),
        "drift_m_per_s": float(r["err"][-1] / secs),
        "drift_pct_travel": 100.0 * float(r["err"][-1]) / max(travelled, 1e-9),
    }


def within(r: dict, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (drift increment, mean loaded cone saturation)."""
    T = (len(r["err"]) // w) * w
    e = r["err"][:T].reshape(-1, w)
    d = e[:, -1] - e[:, 0]
    loaded = (r["fn"] > 0.0)[:T].reshape(-1, w, r["fn"].shape[1])
    sl = r["slip"][:T].reshape(-1, w, r["slip"].shape[1])
    s = np.where(loaded, sl, np.nan)
    with np.errstate(invalid="ignore"):
        s = np.nanmean(s.reshape(len(d), -1), axis=1)
    good = np.isfinite(d) & np.isfinite(s)
    return d[good], s[good]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(REPO / "data" / "dr5"))
    ap.add_argument("--window", type=int, default=2000, help="ticks per drift window")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = [p for p in sorted(Path(args.data).glob("*.npz")) if p.name != "norm_constants.npz"]
    recs = [x for x in (load(p) for p in paths) if x is not None]
    if not recs:
        raise SystemExit(f"no rollouts in {args.data} carry both aux.est_p and truth.slip_sat")

    rows = [per_rollout(r) for r in recs]
    print(f"between rollouts ({len(rows)}), {args.data}")
    print(f"  {'rollout':30} {'terrain':>16} {'mu':>5} {'slip%':>6} {'slipbar':>8} "
          f"{'drift[m]':>9} {'m/s':>8} {'%travel':>8}")
    for x in rows:
        print(f"  {x['name']:30} {x['terrain']:>16} {x['mu']:5.2f} "
              f"{100 * x['slip_frac']:6.1f} {x['slip_mean']:8.3f} {x['drift_m']:9.3f} "
              f"{x['drift_m_per_s']:8.4f} {x['drift_pct_travel']:8.2f}")

    mu = np.array([x["mu"] for x in rows])
    sf = np.array([x["slip_frac"] for x in rows])
    dr = np.array([x["drift_m_per_s"] for x in rows])
    out = {"between": {
        "r_slipfrac_vs_drift": float(np.corrcoef(sf, dr)[0, 1]),
        "r_mu_vs_drift": float(np.corrcoef(mu, dr)[0, 1]),
        "n": len(rows)}}
    print(f"\n  corr(slip fraction, drift rate) = {out['between']['r_slipfrac_vs_drift']:+.3f}  "
          f"(n={len(rows)}, so read the scatter above, not the number)")
    print(f"  corr(mu,             drift rate) = {out['between']['r_mu_vs_drift']:+.3f}")

    D, S = [], []
    for r in recs:
        d, s = within(r, args.window)
        D.append(d)
        S.append(s)
    D, S = np.concatenate(D), np.concatenate(S)
    r_all = float(np.corrcoef(S, D)[0, 1])
    # R^2 of the drift increment on binned saturation -- the same nonparametric estimator
    # `slip_probe` uses, so the two are comparable.
    from experiments.slip_probe import r2_binned
    r2 = r2_binned(D, S, 16)
    out["within"] = {"n": int(D.size), "window_ticks": args.window,
                     "pearson": r_all, "r2_binned": r2,
                     "drift_incr_std_m": float(D.std())}
    print(f"\nwithin rollouts: {D.size} windows of {args.window} ticks")
    print(f"  corr(mean cone saturation, drift increment) = {r_all:+.3f}")
    print(f"  R^2 of drift increment on binned saturation = {r2:.3f}")
    print(f"  drift increment std = {D.std():.4f} m per window")
    lo, hi = S < np.percentile(S, 25), S > np.percentile(S, 75)
    print(f"  mean increment, low-slip quartile  = {D[lo].mean():+.4f} m")
    print(f"  mean increment, high-slip quartile = {D[hi].mean():+.4f} m")
    out["within"]["mean_incr_low_q"] = float(D[lo].mean())
    out["within"]["mean_incr_high_q"] = float(D[hi].mean())

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
