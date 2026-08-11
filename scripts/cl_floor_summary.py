#!/usr/bin/env python
"""Summarise a `cl_floor_sweep.sh` run: drift per floor, per arm, WITH the sign check.

Invariant N3: a mean drift number is not evidence on its own. The 2026-08-07 sweep
found drift crossing zero inside `contact_meas_var`, so a floor can look good purely
because forward cancels lateral. Four of the five floors swept were mixed-sign. This
prints the per-motion sign spread beside the total and flags the cancellations, so a
floor is never selected on a small mean alone.

    uv run python scripts/cl_floor_summary.py --dir results/zdrift_bexp/closed_loop
"""
import argparse
import json
import re
from pathlib import Path


def _walking_nis(pm):
    """Median-of-motions contact NIS/dof over the walking segments, or NaN.

    Reported beside drift, never instead of it (N2): NIS says whether ``S`` is the
    right size, drift says whether the filter is any good, and the project has
    already been burned once by ranking on a proxy.
    """
    v = [q["nis_per_dof"] for q in pm
         if q["motion"] != "stand" and q.get("nis_per_dof", float("nan"))
         == q.get("nis_per_dof", float("nan"))]
    return float(sorted(v)[len(v) // 2]) if v else float("nan")


def load(d: Path):
    rows = []
    for p in sorted(d.glob("cmv_*_*.json")):
        # arm is free-form ("analytic", "learned", "bexp", ...); floors never contain "_"
        m = re.fullmatch(r"cmv_([^_]+)_(.+)", p.stem)
        if not m:
            continue
        j = json.loads(p.read_text())
        pm = j["per_motion"]
        # e_z is CUMULATIVE across the clip, so the run's total error is the last
        # motion's endpoint, while the per-motion deltas are what carry the sign.
        rows.append(dict(
            floor=m.group(1), arm=m.group(2), n=len(pm),
            total_ez=pm[-1]["ez_end"] if pm else float("nan"),
            horiz=pm[-1]["horiz_end"] if pm else float("nan"),
            deltas={q["motion"]: q["ez_delta"] for q in pm},
            rates={q["motion"]: q["rate_mps"] for q in pm},
            # Walking-only NIS/dof (the `stand` segment is a different regime and
            # would dilute it). Missing on runs recorded before it was instrumented.
            nis=_walking_nis(pm),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/zdrift_bexp/closed_loop")
    ap.add_argument("--out", default=None, help="also write the table as JSON")
    args = ap.parse_args()

    rows = load(Path(args.dir))
    if not rows:
        raise SystemExit(f"no cmv_*_*.json under {args.dir}")

    # `stand` is 1 s of standing still and drifts ~0 by construction; it would dilute
    # the sign check, which is about the walking motions.
    def signs(r):
        return {k: v for k, v in r["deltas"].items() if k != "stand"}

    print(f"{'floor':>8} {'arm':>16} {'total e_z':>10} {'|e_z|':>8} {'horiz':>7} "
          f"{'NIS/dof':>8} {'signs':>7}  per-motion d(e_z)")
    for r in sorted(rows, key=lambda r: (float(r["floor"]), r["arm"])):
        s = signs(r)
        pos = sum(1 for v in s.values() if v > 0)
        neg = sum(1 for v in s.values() if v < 0)
        mixed = "MIXED" if pos and neg else f"{'+' if pos else '-'}only"
        per = " ".join(f"{k[:4]}{v:+.2f}" for k, v in s.items())
        nis = f"{r['nis']:8.4f}" if r["nis"] == r["nis"] else f"{'-':>8}"
        print(f"{r['floor']:>8} {r['arm']:>16} {r['total_ez']:>+10.4f} "
              f"{abs(r['total_ez']):>8.4f} {r['horiz']:>7.3f} {nis} {mixed:>7}  {per}")

    an = {r["floor"]: r for r in rows if r["arm"] == "analytic"}
    if an:
        clean = {f: r for f, r in an.items()
                 if not (any(v > 0 for v in signs(r).values())
                         and any(v < 0 for v in signs(r).values()))}
        pool = clean or an
        best = min(pool, key=lambda f: abs(an[f]["total_ez"]))
        print(f"\nanalytic-arm selection: floor {best} "
              f"(|e_z| {abs(an[best]['total_ez']):.4f} m)"
              + ("" if clean else "  -- WARNING: every floor is mixed-sign, so this "
                                  "is a cancellation ranking, not a fix (N3)"))
        print("learned arm is reported, never selected on (N2).")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
