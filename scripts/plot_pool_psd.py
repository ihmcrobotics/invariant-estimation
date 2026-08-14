#!/usr/bin/env python
"""Before/after PSD of vertical base velocity: periodic walking vs randomized motion.

The premise of the randomized-motion experiment is that periodic walking puts gait
phase into the sensor stream as a near-deterministic signal, letting ContactNet
regress the phase-conditional mean of Sigma_C instead of learning contact condition
(measured on the periodic pool: phase R^2 of log tr Sigma_C = 0.942). Breaking the
periodicity is supposed to remove that shortcut. This plot is the check that the
collected data actually differs in that way, rather than merely carrying a new tag.

Two panels, because "concentrated" needs both a shape and a scalar:
  left   normalized PSD -- power FRACTION per Hz, so the two pools are compared on
         spectral shape rather than on total power, which differs between them.
  right  cumulative fraction of in-band power vs frequency. A sharp gait line shows
         as a step; a spread spectrum as a diagonal. This is the panel that makes
         the claim unambiguous.

Usage:  uv run python scripts/plot_pool_psd.py [--pools n8fix n8rand]
"""
import argparse
import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

DT = 1.0e-3
WARMUP = 8000          # skip the joint-KF bias plateau
BAND = (0.3, 8.0)      # plausible gait band [Hz]

# Categorical slots 1 and 2 of the validated reference palette; the pair passes the
# six checks on the light surface (worst CVD dE 24.7, normal-vision 33.6, both >= 3:1).
SERIES = {"n8fix": "#2a78d6", "n8rand": "#eb6834"}
LABEL = {"n8fix": "periodic walking (n8fix)", "n8rand": "randomized motion (n8rand)"}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def welch(x, dt, nperseg=8192):
    """Welch PSD, 50% overlap, Hann window. Plain NumPy so this has no new dependency."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    step = nperseg // 2
    win = np.hanning(nperseg)
    scale = 1.0 / (win.sum() ** 2 / dt)      # -> power spectral DENSITY
    segs = [x[i:i + nperseg] for i in range(0, len(x) - nperseg + 1, step)]
    if not segs:
        raise ValueError("series shorter than one segment")
    P = np.mean([np.abs(np.fft.rfft(s * win)) ** 2 for s in segs], axis=0) * scale
    P[1:-1] *= 2.0                            # one-sided
    return np.fft.rfftfreq(nperseg, dt), P


def pool_psd(tag, n_max=6):
    paths = sorted(glob.glob(str(REPO / "data" / f"*_{tag}_seed*.npz")))[:n_max]
    if not paths:
        raise SystemExit(f"no rollouts for pool tag {tag!r}")
    acc, f = None, None
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            vz = z["truth.v"][WARMUP:, 2]
        f, P = welch(vz, DT)
        acc = P if acc is None else acc + P
    return f, acc / len(paths), len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs=2, default=["n8fix", "n8rand"])
    ap.add_argument("--out", default="results/rand_motion/pool_psd.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for tag in args.pools:
        f, P, n = pool_psd(tag)
        band = (f >= BAND[0]) & (f <= BAND[1])
        data[tag] = dict(f=f[band], P=P[band], n=n)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1), facecolor=SURFACE)
    for a in ax:
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color("#d8d7d2")
        a.tick_params(colors=INK2, labelsize=9, length=3)
        a.grid(True, color="#ebeae5", lw=0.8, zorder=0)      # recessive grid
        a.set_axisbelow(True)

    # -- left: normalized PSD (shape, not level) ------------------------------
    for tag in args.pools:
        d = data[tag]
        frac = d["P"] / np.trapezoid(d["P"], d["f"])
        ax[0].semilogy(d["f"], frac, lw=2.0, color=SERIES[tag],
                       label=f"{LABEL[tag]}  (n={d['n']})", zorder=3)
    ax[0].set_xlabel("frequency [Hz]", color=INK2, fontsize=10)
    ax[0].set_ylabel("normalized PSD  [fraction of power / Hz]", color=INK2, fontsize=10)
    ax[0].set_title("Vertical base velocity spectrum (Welch, 6 rollouts/pool)", color=INK, fontsize=11.5,
                    loc="left", pad=10)
    ax[0].set_xlim(*BAND)
    leg = ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)          # text wears text tokens, not the series color

    # -- right: where the power actually sits ---------------------------------
    # A cumulative curve was tried here first and undersold the result: the two
    # integrals look similar because the randomized pool GAINS low-frequency power
    # while LOSING the gait line, and a cumulative plot partly cancels the two. Bands
    # separate them, which is the honest summary -- and the sub-gait band is the one
    # that matters, because drift is a DC/low-frequency property.
    bands = [("0.3-1.5 Hz\nsub-gait", 0.3, 1.5),
             ("1.5-2.5 Hz\ngait line", 1.5, 2.5),
             ("2.5-8 Hz\nharmonics", 2.5, 8.0)]
    x = np.arange(len(bands))
    w = 0.36
    for k, tag in enumerate(args.pools):
        d = data[tag]
        tot = d["P"].sum()
        vals = [100 * d["P"][(d["f"] >= lo) & (d["f"] < hi)].sum() / tot
                for _, lo, hi in bands]
        # edgecolor == surface gives the 2px surface gap between adjacent bars
        ax[1].bar(x + (k - 0.5) * w, vals, w, color=SERIES[tag], zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax[1].annotate(f"{v:.0f}%", (xi, v), xytext=(0, 3),
                           textcoords="offset points", ha="center",
                           fontsize=9, color=INK2)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([b[0] for b in bands], fontsize=9, color=INK2)
    ax[1].set_ylabel("share of in-band power [%]", color=INK2, fontsize=10)
    ax[1].set_title("Power leaves the gait line for the sub-gait band",
                    color=INK, fontsize=11.5, loc="left", pad=10)
    ax[1].set_ylim(0, 72)
    ax[1].grid(axis="x", visible=False)

    fig.tight_layout()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")

    # the scalar the plot is evidence for
    for tag in args.pools:
        d = data[tag]
        pk = d["f"][np.argmax(d["P"])]
        near = np.abs(d["f"] - pk) < 0.25
        print(f"  {tag:8s} n={d['n']}  peak {pk:.2f} Hz  "
              f"power within +-0.25 Hz of peak: {100 * d['P'][near].sum() / d['P'].sum():.1f}%")


if __name__ == "__main__":
    main()
