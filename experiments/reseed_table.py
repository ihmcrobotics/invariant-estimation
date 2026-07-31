r"""Closed-loop ablation table — `artifacts/run7_cl/*.npz` -> one comparison.

Reads the per-tick histories `run_estimator.py --out` writes and reports the drift metrics the
re-seed and the objective are being judged on.

**How yaw error is obtained.** `run_estimator` logs total attitude error (`att_deg`) and tilt error
(`tilt_deg`, the roll/pitch part) but no yaw channel. For small errors the rotation-vector norms
compose orthogonally, so

.. math::  \psi_{err} \approx \sqrt{\max(0,\; \mathrm{att}^2 - \mathrm{tilt}^2)}

which is exact when the error rotation's tilt and yaw parts are orthogonal and degrades gracefully
otherwise. Errors here are a few degrees, so the small-angle reading is safe — but it is a derived
quantity, not a logged one, and is labelled as such.

**Tail, not mean.** Drift is what is being measured, so every number is over the last third of the
run: a mean over the whole trace is dominated by the transient while the filter is still settling
and would understate a drift that is still growing at the end.

    uv run python -m experiments.reseed_table
    uv run python -m experiments.reseed_table --dir artifacts/run7_cl --tail 0.33
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Presentation order: each arm sits next to the one it differs from by a single flag.
ORDER = ["analytic_noreseed", "analytic_reseed", "run6w128_noreseed",
         "run7_noreseed", "run7_reseed",
         "turn_analytic", "turn_run6w128", "turn_run7"]

LABEL = {
    "analytic_noreseed": "N=4 analytic",
    "analytic_reseed": "N=4 analytic + reseed",
    "run6w128_noreseed": "run 6 w128 (dr4 commands)",
    "run7_noreseed": "run 7 w128 (dr5 commands)",
    "run7_reseed": "run 7 w128 + reseed",
    # A TURNING command (vx 0.4, yaw 0.8). The forward-walk arms above are in run 6's training
    # distribution and out of run 7's, which flatters run 6; these close that confound.
    "turn_analytic": "[turn] N=4 analytic",
    "turn_run6w128": "[turn] run 6 w128",
    "turn_run7": "[turn] run 7 w128",
}


def yaw_err_deg(att_deg: np.ndarray, tilt_deg: np.ndarray) -> np.ndarray:
    """The yaw part of the attitude error, by orthogonal decomposition (see the module docstring)."""
    return np.sqrt(np.maximum(0.0, att_deg**2 - tilt_deg**2))


def summarise(path: Path, tail: float) -> dict:
    z = np.load(path, allow_pickle=True)
    n = len(z["t"])
    k = slice(int(n * (1.0 - tail)), n)
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a)[k] ** 2)))     # noqa: E731

    dz = np.asarray(z["dz"])
    t = np.asarray(z["t"])
    # Sink RATE over the tail, by least squares — the single number that says whether height is
    # still running away or has settled, which a final-value reading cannot distinguish.
    slope = float(np.polyfit(t[k], dz[k], 1)[0])
    return {
        "yaw_tail_deg": rms(yaw_err_deg(np.asarray(z["att_deg"]), np.asarray(z["tilt_deg"]))),
        "tilt_tail_deg": rms(z["tilt_deg"]),
        "att_tail_deg": rms(z["att_deg"]),
        "v_err_rms": rms(z["v_err"]),
        "p_err_rms": rms(z["p_err"]),
        "dz_final": float(dz[-1]),
        "dz_rate": slope,
        "nis_tail": float(np.mean(np.asarray(z["nis"])[k])),
        "ticks": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(REPO / "artifacts/run7_cl"))
    ap.add_argument("--tail", type=float, default=1 / 3)
    args = ap.parse_args()

    d = Path(args.dir)
    rows = [(a, summarise(d / f"{a}.npz", args.tail))
            for a in ORDER if (d / f"{a}.npz").exists()]
    missing = [a for a in ORDER if not (d / f"{a}.npz").exists()]
    if not rows:
        raise SystemExit(f"no histories in {d}; run artifacts/reseed_ablation.sh first")

    cols = [("YAW tail [deg]", "yaw_tail_deg", "{:.3f}"),
            ("tilt tail [deg]", "tilt_tail_deg", "{:.3f}"),
            ("v err rms", "v_err_rms", "{:.4f}"),
            ("p err rms", "p_err_rms", "{:.4f}"),
            ("dz final [m]", "dz_final", "{:+.4f}"),
            ("dz rate [m/s]", "dz_rate", "{:+.5f}"),
            ("NIS tail", "nis_tail", "{:.3f}")]

    w = max(len(LABEL.get(a, a)) for a, _ in rows)
    print(f"\nclosed loop, last {100 * args.tail:.0f}% of {rows[0][1]['ticks']} control ticks\n")
    print(f"{'arm':{w}s} " + " ".join(f"{c[0]:>15s}" for c in cols))
    print("-" * (w + 16 * len(cols)))
    for a, s in rows:
        print(f"{LABEL.get(a, a):{w}s} " + " ".join(f"{f.format(s[k]):>15s}" for _, k, f in cols))

    # The two questions this table exists to answer, stated as deltas rather than left to the eye.
    by = dict(rows)
    def delta(new, old, key="yaw_tail_deg"):
        if new in by and old in by:
            a, b = by[old][key], by[new][key]
            return f"{a:.3f} -> {b:.3f}  ({b / a:.2f}x)" if a else "n/a"
        return None

    print()
    for label, new, old in (("re-seed, no network      ", "analytic_reseed", "analytic_noreseed"),
                            ("diversified commands     ", "run7_noreseed", "run6w128_noreseed"),
                            ("re-seed on top of run 7  ", "run7_reseed", "run7_noreseed"),
                            ("run 7 vs analytic        ", "run7_noreseed", "analytic_noreseed"),
                            ("[turn] diversified cmds  ", "turn_run7", "turn_run6w128"),
                            ("[turn] run 7 vs analytic ", "turn_run7", "turn_analytic")):
        got = delta(new, old)
        if got:
            print(f"  yaw {label}: {got}")
    if missing:
        print(f"\n  (not run: {', '.join(missing)})")
    print("\n  Yaw is DERIVED from att/tilt (see module docstring), and remains an unobservable")
    print("  direction in every arm — these are drift rates, not observability.")


if __name__ == "__main__":
    main()
