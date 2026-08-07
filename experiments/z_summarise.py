r"""experiments/z_summarise.py — pool `z_budget.py` records into per-condition means.

Every previous drift report rested on a single rollout and had to say so. This
groups the `.npz` records written by `z_budget.py --out` by everything except the
noise seed, and reports the mean and spread across seeds, so a verdict is applied
to a distribution rather than to a lucky run.

    uv run python experiments/z_summarise.py results/zmatrix
    uv run python experiments/z_summarise.py results/adetect --detection

Tag convention: `<arm>_<condition>[_<variant>]_s<seed>.npz`; everything before
`_s<seed>` is the group key.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

_SEED = re.compile(r"_s(\d+)$")


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    dt = float(z["dt"])
    e_z = z["p_stage"][:, 3, 2] - z["p_true"][:, 2]
    ev_z = z["v_stage"][:, 3, 2] - z["v_true"][:, 2]
    T = len(e_z)
    dur = T * dt
    out = {
        "drift_rate": (e_z[-1] - e_z[0]) / dur,
        "final_ez": e_z[-1],
        "mean_evz": ev_z.mean(),
        "duration": dur,
    }
    # Horizontal error, normalised by ground-track path length rather than net
    # displacement: in the turning condition the robot walks a circle, and
    # displacement inflates the ratio several-fold.
    p, pt = z["p_stage"][:, 3, :2], z["p_true"][:, :2]
    path = float(np.abs(np.diff(pt, axis=0)).sum())
    out["horiz_over_path"] = float(np.linalg.norm((p - pt)[-1]) / max(path, 1e-9))
    # Fall detector. A terrain run that walks off the heightfield edge produces a
    # perfectly well-formed record with a meaningless drift (measured once:
    # true z = -378 m, tilt 96 deg). Judge it on the TRUE state only: a large
    # *estimate* error is exactly what some conditions legitimately produce -- the
    # x0=50 world-origin lever run drifts -2.9 m and is perfectly valid -- so
    # keying on `e_z` gives a false positive on the very runs worth keeping.
    fell_z = abs(z["p_true"][-1, 2] - z["p_true"][0, 2]) > 0.5
    up_true = z["R_true"][:, 2, 2]                     # cos(tilt) of the true base
    out["fell"] = bool(fell_z or up_true.min() < 0.5)  # >60 deg true tilt
    for key in z.files:
        if key.startswith(("pos_", "vel_", "cm_")):
            out[key] = float(np.asarray(z[key]).sum())
    if "frame_gap" in z.files:
        out["frame_gap"] = float(np.asarray(z["frame_gap"]).mean())
    if "contact_true" in z.files:
        truth, trust = z["contact_true"], z["contact"]
        n = min(len(truth), len(trust))
        truth, trust = truth[:n], trust[:n]
        out["fp"] = float(((trust > 0) & (truth == 0)).mean())
        out["fn"] = float(((trust == 0) & (truth > 0)).mean())
        out["duty_true"] = float(truth.mean())
        out["duty_trust"] = float(trust.mean())
    return out


def group_key(name: str) -> str:
    return _SEED.sub("", name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--detection", action="store_true",
                    help="also print the FP/FN columns")
    ap.add_argument("--settle", type=float, default=20.0, metavar="S",
                    help="skip .npz files modified within this many seconds — they "
                         "may still be being written by a running sweep")
    ap.add_argument("--ledger", action="store_true",
                    help="print the full per-term ledger instead of the summary")
    args = ap.parse_args()

    rows: dict[str, list] = {}
    now = time.time()
    for d in args.dirs:
        for p in sorted(Path(d).glob("*.npz")):
            # Skip files still being written. A sweep running alongside this script
            # will have a partially-flushed .npz on disk, and numpy will happily
            # read it and return numbers that change between invocations — which is
            # exactly how a torn read gets quoted in a report.
            if now - p.stat().st_mtime < args.settle:
                print(f"  .. {p.name}: written {now - p.stat().st_mtime:.0f}s ago, "
                      f"still settling — skipped")
                continue
            try:
                rows.setdefault(group_key(p.stem), []).append(load(p))
            except Exception as exc:                       # noqa: BLE001
                print(f"  !! {p.name}: {exc}")

    if not rows:
        print("no records found")
        return 1

    def col(g, k):
        return np.array([r[k] for r in rows[g] if k in r])

    hdr = f"{'group':34s} {'n':>2s} {'drift [m/s]':>18s} {'final e_z [m]':>14s} " \
          f"{'horiz/path':>11s}"
    if args.detection:
        hdr += f" {'FP%':>6s} {'FN%':>6s} {'duty T/E':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for g in sorted(rows):
        keep = [r for r in rows[g] if not r.get("fell")]
        n_fell = len(rows[g]) - len(keep)
        if not keep:
            print(f"{g:34s} {'--':>2s}  ALL {n_fell} RUN(S) FELL — no usable drift")
            continue
        rows[g] = keep
        dr, fz = col(g, "drift_rate"), col(g, "final_ez")
        line = (f"{g:34s} {len(dr):2d} {dr.mean():+10.5f}±{dr.std():<7.5f} "
                f"{fz.mean():+14.4f} {col(g, 'horiz_over_path').mean() * 100:10.2f}%"
                + (f"  [{n_fell} FELL, excluded]" if n_fell else ""))
        if args.detection and len(col(g, "fp")):
            line += (f" {col(g, 'fp').mean() * 100:5.2f} {col(g, 'fn').mean() * 100:6.2f}"
                     f" {col(g, 'duty_true').mean():5.3f}/{col(g, 'duty_trust').mean():.3f}")
        print(line)

    if args.ledger:
        terms = sorted({k for g in rows for r in rows[g] for k in r
                        if k.startswith(("pos_", "vel_", "cm_"))})
        print(f"\n{'group':34s} " + " ".join(f"{t.replace('pos_', 'p.').replace('vel_', 'v.').replace('cm_', 'c.'):>14s}" for t in terms))
        for g in sorted(rows):
            vals = " ".join(f"{col(g, t).mean():14.5f}" if len(col(g, t)) else f"{'-':>14s}"
                            for t in terms)
            print(f"{g:34s} {vals}")

    # Pairwise deltas for any group that differs only by its trailing variant.
    print("\nPAIRED DELTAS (same arm+condition, differing variant)")
    for g in sorted(rows):
        for other in sorted(rows):
            if other == g or not other.startswith(g.rsplit("_", 1)[0]):
                continue
            a, b = col(g, "drift_rate").mean(), col(other, "drift_rate").mean()
            if g < other and abs(a) > 1e-12:
                print(f"  {g:32s} -> {other:32s}  "
                      f"{a:+.5f} -> {b:+.5f} m/s  ({100 * (b - a) / abs(a):+7.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
