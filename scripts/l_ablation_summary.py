#!/usr/bin/env python
"""Tabulate the L-ablation grid from its 12 `summary.json` files.

Emits four markdown tables, in the order they have to be READ:

  1. **Audit** first, deliberately. The whole point of this ladder is that the
     2026-08-06 arms could not be compared because they got 7639..9839 steps. A
     grid whose cells did not all run the same number of steps has the same defect,
     so `steps_run` is reported before any metric -- and any cell that is missing,
     short, or configured differently from its siblings is called out by name
     rather than averaged into a number that looks fine.
  2. Held-out metrics per cell (learned), which is the result.
  3. The analytic baseline per cell, which should be near-CONSTANT down each
     column: it does not depend on the network, so drift there means the cells are
     not evaluating the same thing and the learned comparison is not clean.
  4. The frozen pose weights, which legitimately differ across L (each term is
     auto-sized to 0.5 * L_vel at init, and L_pos grows with segment length).

Validation metrics are objective-independent by construction, so every cell in the
grid is directly comparable on them.

Usage:
    uv run python scripts/l_ablation_summary.py
    uv run python scripts/l_ablation_summary.py --root results/l_ablation --steps 6000
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TAGS = [("A_l2vel", "l2_velocity"), ("B_l2velpos", "l2_vel_pos"),
        ("C_l2velori", "l2_vel_ori"), ("D_l2velposori", "l2_vel_pos_ori")]

# (summary.json key, column header, format). `vel_nees` targets 3 and `nis_over_dof`
# targets 1; both are calibration, not accuracy, and they routinely move OPPOSITE to
# vel_rmse -- the 2026-08-06 arms improved the mean while NIS/dof fell 0.35 -> 0.08.
METRICS = [("vel_rmse", "vel RMSE [m/s]", "{:.4f}"),
           ("vel_nees", "vel NEES (→3)", "{:.2f}"),
           ("nis_over_dof", "NIS/dof (→1)", "{:.3f}")]


def load(root, L, tag):
    p = Path(root) / f"L{L}_{tag}" / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None      # written but truncated -- a partial run, not a result


def mean(s, arm, key):
    """Mean over the held-out rollouts (one per terrain under --pool)."""
    vals = [m[arm][key] for m in s["val"]]
    return sum(vals) / len(vals) if vals else float("nan")


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/l_ablation")
    ap.add_argument("--L", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--steps", type=int, default=6000,
                    help="the matched step budget every cell was supposed to run")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / root
    grid = {(L, tag): load(root, L, tag) for L in args.L for tag, _ in TAGS}

    # ---- 1. audit -------------------------------------------------------------
    problems = []
    rows = []
    for L in args.L:
        cells = []
        for tag, _obj in TAGS:
            s = grid[(L, tag)]
            if s is None:
                cells.append("**missing**")
                problems.append(f"L={L} {tag}: no summary.json")
                continue
            ran = s["steps_run"]
            cells.append(str(ran) if ran == args.steps else f"**{ran}**")
            if ran != args.steps:
                problems.append(
                    f"L={L} {tag}: steps_run={ran}, wanted {args.steps} -- NOT "
                    f"comparable; this is the exact confound the ladder removes")
        rows.append([f"L={L}"] + cells)
    print("### Audit — steps run per cell\n")
    print(table(rows, ["L"] + [t for t, _ in TAGS]))

    # Configuration drift: every cell must share the filter and optimizer setup, or
    # the grid is measuring more than one thing at a time.
    fixed = {}
    for (L, tag), s in grid.items():
        if s is None:
            continue
        key = tuple(s["cfg"][k] for k in
                    ("B", "H", "F", "remat", "contact_meas_var", "peak_lr",
                     "init_seed", "batcher_seed", "episode_s", "warm_in_s"))
        fixed.setdefault(key, []).append(f"L{L}_{tag}")
    if len(fixed) > 1:
        problems.append(
            "cells do not share one configuration (B/H/F/remat/contact_meas_var/"
            "peak_lr/seeds/episode): " + "; ".join(
                f"[{', '.join(v)}]" for v in fixed.values()))

    if problems:
        print("\n> **The grid is not clean.** Fix or re-run before quoting it:")
        for p in problems:
            print(f">   * {p}")
    else:
        print(f"\nAll {len(grid)} cells ran {args.steps} steps under one "
              f"configuration — the grid is matched.")

    # ---- 2/3. metrics ---------------------------------------------------------
    for arm, title, note in [
        ("learned", "Held-out, LEARNED Sigma_C", "the result"),
        ("baseline", "Held-out, analytic baseline", "network-independent: should be "
         "near-constant DOWN each column, since only L changed. Drift here means the "
         "cells are not evaluating the same thing"),
    ]:
        print(f"\n### {title}\n\n*({note}; mean over held-out rollouts, one per "
              f"terrain)*\n")
        for key, label, fmt in METRICS:
            rows = []
            for L in args.L:
                cells = []
                for tag, _obj in TAGS:
                    s = grid[(L, tag)]
                    cells.append("—" if s is None else fmt.format(mean(s, arm, key)))
                rows.append([f"L={L}"] + cells)
            print(f"\n**{label}**\n")
            print(table(rows, ["L"] + [t for t, _ in TAGS]))

    # ---- 4. pose weights ------------------------------------------------------
    print("\n### Frozen pose weights (auto-sized per cell)\n")
    print("*Each active term is sized on the first warm batch to `0.5 x L_vel`, then "
          "frozen. `L_pos` is a segment-relative DISPLACEMENT, so it grows with `L` "
          "and the absolute weights differ down a column by design — what is held "
          "fixed across `L` is the relative pull between terms, which is what makes "
          "\"the same objective at a different L\" mean anything.*\n")
    rows = []
    for L in args.L:
        for tag, _obj in TAGS:
            s = grid[(L, tag)]
            if s is None or not (s["cfg"]["w_pos"] or s["cfg"]["w_ori"]):
                continue
            rows.append([f"L={L}", tag, f"{s['cfg']['w_pos']:.4g}",
                         f"{s['cfg']['w_ori']:.4g}"])
    print(table(rows, ["L", "arm", "w_pos", "w_ori"]) if rows else "*(none yet)*")

    print("\n> Caveat that belongs on every table above: at matched STEPS, L=512 "
          "consumes 4x the trajectory per step of L=128 (`ChainedBatcher` advances "
          "`t += L`), so *data seen* is not matched and chains reseed ~4x more often. "
          "That is inherent to a matched-step L ablation, not an oversight.")


if __name__ == "__main__":
    main()
