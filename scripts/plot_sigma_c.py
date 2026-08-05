#!/usr/bin/env python
"""Gate G figures: Sigma_C over a stride, and per-terrain conditioning/applied rate.

Two figures, both of which have to be able to show a NEGATIVE result clearly --
the PARTIAL verdict in the plan is "per-corner Sigma_C is degenerate/identical
across a foot's corners", and a plot that cannot make that visible is useless
here. So:

  * fig 1 draws all corners of one foot on the SAME axes. Identical curves
    overlapping is immediately legible as "the net did not distinguish them",
    and the per-corner spread is printed as a number besides.
  * fig 2 plots the conditioning proxy and the applied/gated rate per terrain,
    because the pooled average is exactly what hides a gate that collapses on
    flat ground (Gate C's known trap).

Usage:
    uv run python scripts/plot_sigma_c.py --run results/latest --out results/latest
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# colourblind-safe, distinguishable in greyscale by dash pattern too
CORNER_STYLE = [("#4C72B0", "-"), ("#DD8452", "--"), ("#55A868", "-."), ("#C44E52", ":")]
CORNER_NAME = ["heel-R", "heel-L", "toe-R", "toe-L"]


def plot_sigma_over_stride(sigma, contact_force, out_path, foot=0, cpf=4, dt=1e-3):
    """sigma: (T, N, 3, 3) learned contact covariances; contact_force: (T, N)."""
    T = sigma.shape[0]
    t = np.arange(T) * dt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    for j in range(cpf):
        i = foot * cpf + j
        # isotropic scale of the block: cube root of det, in metres
        det = np.linalg.det(sigma[:, i])
        scale = np.cbrt(np.maximum(det, 1e-300))
        c, ls = CORNER_STYLE[j]
        ax.plot(t, scale, color=c, ls=ls, lw=1.8, label=CORNER_NAME[j])
    ax.set_yscale("log")
    ax.set_ylabel(r"$\det(\Sigma_C)^{1/3}$  [m]")
    ax.set_title(f"Learned contact covariance over a stride — foot {foot}, {cpf} corners")
    ax.legend(ncol=4, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for j in range(cpf):
        i = foot * cpf + j
        c, ls = CORNER_STYLE[j]
        # offset each corner slightly so overlapping 0/1 traces stay readable
        ax.plot(t, contact_force[:, i] * 0.9 + 0.04 * j, color=c, ls=ls, lw=1.2)
    # This is the per-corner CONTACT TRUST mask the sim publishes (0/1), not a
    # force in newtons -- labelling it "normal force" would be a plain untruth
    # about what the reader is looking at.
    ax.set_ylabel("per-corner contact trust")
    ax.set_yticks([0, 1])
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # The number that decides PASS vs PARTIAL: how much do corners actually differ?
    per_corner = np.cbrt(np.maximum(
        np.linalg.det(sigma[:, foot * cpf:(foot + 1) * cpf]), 1e-300))
    spread = per_corner.std(axis=1) / np.maximum(per_corner.mean(axis=1), 1e-30)
    return {
        "median_relative_spread_across_corners": float(np.median(spread)),
        "max_relative_spread_across_corners": float(np.max(spread)),
        "corner_scale_median": [float(v) for v in np.median(per_corner, axis=0)],
    }


def plot_conditioning(per_terrain, out_path):
    """per_terrain: {terrain: {"cond": [...], "applied": [...]}}."""
    terrains = list(per_terrain)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    data = [np.asarray(per_terrain[t]["cond"], float) for t in terrains]
    ax.boxplot(data, tick_labels=terrains, showfliers=False)
    ax.axhline(1e9, color="#C44E52", ls="--", lw=1.5, label="cond_s_max = 1e9")
    ax.set_yscale("log")
    ax.set_ylabel("condition proxy of S")
    ax.set_title("Innovation conditioning by terrain")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rates = [100.0 * np.mean(np.asarray(per_terrain[t]["applied"], float)) for t in terrains]
    ax.bar(terrains, rates, color="#4C72B0")
    ax.set_ylabel("contact updates applied [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Update applied rate by terrain")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", type=str, required=True,
                    help="npz with sigma (T,N,3,3), force (T,N), and optional per-terrain stats")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--contacts-per-foot", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.diag, allow_pickle=True)

    stats = plot_sigma_over_stride(
        z["sigma"], z["force"], out / "sigma_c_over_stride.png",
        cpf=args.contacts_per_foot)
    print(json.dumps(stats, indent=2))

    if "per_terrain" in z:
        plot_conditioning(z["per_terrain"].item(), out / "conditioning_by_terrain.png")
    (out / "sigma_c_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"-> {out}/sigma_c_over_stride.png")


if __name__ == "__main__":
    main()
