#!/usr/bin/env python
"""Summarise the contact-R-floor sweep: drift and consistency against the floor.

Two goals are being traded here, and the point of the table is to show whether they
trade at all:

  drift_z / final_ez   height must stop sinking
  nis_over_dof -> 1    the contact confidence must be CONSISTENT; 0.18 means the
                       innovation is 5x over-covered, i.e. the filter claims far
                       more uncertainty than its residuals justify

The floor sets innovation covariance directly (it enters as sigma_q + cmv*I, so
S grows with it), so raising it pushes NIS DOWN and lowering it pushes NIS UP toward
target. If drift and NIS prefer opposite directions, that tension is the result and
it needs saying out loud rather than optimising one and quietly losing the other.

`--best` prints one floor for the shell to consume. It ranks on a combined score
because picking on drift alone is how you end up with a filter that does not sink
and still cannot be trusted.
"""
import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "zdrift"


def load():
    rows = []
    for p in sorted(glob.glob(str(OUT / "sweep_cmv_*.json"))):
        floor = Path(p).stem.replace("sweep_cmv_", "")
        for r in json.loads(Path(p).read_text()):
            rows.append(dict(r, floor=floor, floor_f=float(floor)))
    return rows


def sign_consistent(r):
    """Does every held-out terrain agree on the SIGN of the accumulated error?

    A floor sitting on the zero crossing of drift produces a small MEAN while
    individual terrains sit either side of zero. That is a cancellation, not a fix,
    and the 2026-08-07 study measured this parameter doing exactly that: its
    flat-ground optimum drifts upward on terrain. Selecting on |mean| alone walks
    straight into it, so a mixed-sign floor is disqualified rather than ranked.
    """
    per = r.get("per_rollout")
    if not per:
        return None                      # evaluated before per-rollout was recorded
    signs = {e["final_ez"] > 0 for e in per}
    return len(signs) == 1


def score(r):
    """Distance from the two targets, in log space so both are scale-free.

    |drift_z| wants 0 -- use accumulated error, which is the quantity a walking
    robot actually experiences, and which ranked the cells differently from slope.
    nis_over_dof wants 1, and log makes 0.2 and 5.0 equally wrong, which they are.
    """
    import math
    nis = max(r["nis_over_dof"], 1e-6)
    return abs(r["final_ez"]) * (1.0 + abs(math.log(nis)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true")
    args = ap.parse_args()
    rows = load()
    if not rows:
        return

    if args.best:
        by_floor = {}
        for r in rows:
            by_floor.setdefault(r["floor"], []).append(score(r))
        # disqualify any floor whose terrains disagree on sign -- see sign_consistent
        ok = {}
        for r in rows:
            c = sign_consistent(r)
            ok.setdefault(r["floor"], []).append(c is not False)
        eligible = {f: v for f, v in by_floor.items() if all(ok.get(f, [True]))}
        pool = eligible or by_floor
        best = min(pool, key=lambda f: sum(pool[f]) / len(pool[f]))
        print(best)
        return

    print("\n| floor | cell | |drift_z| | final e_z | NIS/dof (→1) | analytic NIS | sign agrees |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["floor_f"], r["cell"])):
        agree = {True: "yes", False: "**MIXED**", None: "?"}[sign_consistent(r)]
        print(f"| {r['floor']} | {r['cell']} | {abs(r['drift_z']):.5f} | "
              f"{r['final_ez']:+.3f} | {r['nis_over_dof']:.3f} | "
              f"{r['base_nis_over_dof']:.3f} | {agree} |")
    by_floor = {}
    for r in rows:
        by_floor.setdefault(r["floor"], []).append(r)
    print("\n| floor | mean |drift_z| | mean final e_z | mean NIS/dof | combined |")
    print("|---|---|---|---|---|")
    for f in sorted(by_floor, key=float):
        g = by_floor[f]
        n = len(g)
        print(f"| {f} | {sum(abs(r['drift_z']) for r in g)/n:.5f} | "
              f"{sum(r['final_ez'] for r in g)/n:+.3f} | "
              f"{sum(r['nis_over_dof'] for r in g)/n:.3f} | "
              f"{sum(score(r) for r in g)/n:.4f} |")
    print("\ncombined = |final e_z| * (1 + |log NIS/dof|): lower is better on BOTH "
          "axes at once. Read the columns too -- if drift and NIS pull in opposite "
          "directions, the combined number hides the tension that IS the result.")


if __name__ == "__main__":
    main()
