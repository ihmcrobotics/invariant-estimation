#!/usr/bin/env python
"""Event-triggered Sigma_C against contact ground truth, at toe-off and heel-strike.

Asks whether the learned contact covariance is aligned with the events it should care
about. Heel-strike is an impact transient and toe-off is contact breaking; both are
moments when the "anchor is world-static" assumption is most wrong, so Sigma_C should
be largest there. The analytic heuristic instead switches on a binary stance/swing
trust decision, so it is a plateau, not a pair of peaks -- and the question is which
of those two shapes the network learned.

Two measures on different scales (log10 tr Sigma_C, and a 0-1 contact fraction), so
this is small multiples rather than a dual axis: the rows share a time base and never
share a y-scale.

N=8 (four corners per foot). Events are detected PER CORNER from the recorded
analytic `inputs.contact_chol`, which switches stance 1e-4 / swing 1e1 and is
therefore the trust decision the collection run actually used; `sensors.contact` is
the independent binary ground truth from the simulator and is what the bottom row
shows.

Usage:  JAX_PLATFORMS=cpu uv run python scripts/plot_contact_phase.py \
            --ckpt results/zdrift/L256_A_l2vel_cmv1e-3 --diag-param softplus
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import invariant_estimation  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
from invariant_estimation.contactnet import (dataset as DS, features as F,
                                             network as N, normalize as NM,
                                             train as TR)
from invariant_estimation.contactnet.checkpoint import config_for_checkpoint

# Validated categorical slots 1-3 (light surface): worst-pair CVD dE 9.2,
# normal-vision 27.6. Slot 3 (aqua) is 2.74:1 against the surface, below the 3:1
# floor, so it carries a visible direct label rather than a legend chip.
LEARNED, ANALYTIC, TRUTH = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ebeae5"
DT, W = 1.0e-3, 400          # +-0.4 s: a swing is ~0.4-0.5 s at this gait


def load(ckpt, diag_param, t_max):
    # `diag_param=None` (the default) restores what the run recorded; an explicit value
    # overrides it, which is the only legitimate use -- deliberately reading a
    # checkpoint under the wrong parameterisation to see the size of the error.
    over = {} if diag_param is None else {"diag_param": diag_param}
    cfg = config_for_checkpoint(str(ckpt), **over)
    spec = cfg.diag_spec
    z = np.load(Path(ckpt) / "norm_constants.npz", allow_pickle=False)
    consts = NM.NormConstants(
        mean=jnp.asarray(z["mean"]), std=jnp.asarray(z["std"]),
        names=tuple(str(s) for s in z["names"]),
        floored=tuple(str(s) for s in z["floored"]), n_ticks=0, source=str(ckpt))
    like = N.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths, cfg.sigma_0,
                  cfg.eps, spec)
    params = TR.load_params(str(Path(ckpt) / "params.npz"), like)

    path = sorted(glob.glob(str(REPO / "data" / "*_n8fix_seed*.npz")))[0]
    cache = DS.load_channel_cache(DS.cache_path(path))
    x = NM.apply(jnp.asarray(cache["channels"][:t_max]), consts)
    win = np.asarray(F.window(x, cfg.H))
    T, n_c = win.shape[0], win.shape[1]
    L = np.asarray(jax.vmap(lambda p, xi: N.forward(p, xi, cfg.eps, spec),
                            in_axes=(None, 0))(
        params, jnp.asarray(win.reshape(T * n_c, -1))))
    learned = (L ** 2).sum(axis=(-2, -1)).reshape(T, n_c)

    with np.load(path, allow_pickle=True) as zz:
        an = np.asarray(zz["inputs.contact_chol"])[:T]
        gt = np.asarray(zz["sensors.contact"])[:T].astype(float)
    analytic = (an ** 2).sum(axis=(-2, -1))
    swing = analytic > 3e-2          # between the 1e-4 and 1e1 factor levels
    return learned, analytic, gt, swing, cfg.H, n_c, Path(path).name, cfg.diag_param


def triggered(sig, swing, H, kind):
    """Mean of `sig` in a +-W window about every per-corner event of `kind`."""
    T, n_c = sig.shape
    out = []
    for c in range(n_c):
        d = np.diff(swing[:, c].astype(int))
        idx = np.where(d == (1 if kind == "toe-off" else -1))[0]
        out += [sig[k - W:k + W, c] for k in idx if H + W < k < T - W]
    return (np.mean(out, axis=0), len(out)) if out else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/zdrift/L256_A_l2vel_cmv1e-3")
    ap.add_argument("--diag-param", default=None, choices=list(N.DIAG_PARAMS),
                    help="override the parameterisation recorded in the run's "
                         "summary.json; default is to use what was recorded")
    ap.add_argument("--ticks", type=int, default=30000)
    ap.add_argument("--out", default="results/zdrift/contact_phase.png")
    args = ap.parse_args()

    learned, analytic, gt, swing, H, n_c, roll, dparam = load(
        args.ckpt, args.diag_param, args.ticks)
    lg = np.log10(np.clip(learned, 1e-30, None))
    la = np.log10(np.clip(analytic, 1e-30, None))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 6.6), facecolor=SURFACE,
                           sharex=True)
    t = (np.arange(-W, W) * DT) * 1e3

    for j, kind in enumerate(("toe-off", "heel-strike")):
        m_l, n_ev = triggered(lg, swing, H, kind)
        m_a, _ = triggered(la, swing, H, kind)
        m_g, _ = triggered(gt, swing, H, kind)

        a0, a1 = ax[0, j], ax[1, j]
        for a in (a0, a1):
            a.set_facecolor(SURFACE)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color("#d8d7d2")
            a.tick_params(colors=INK2, labelsize=9, length=3)
            a.grid(True, color=GRID, lw=0.8, zorder=0)
            a.set_axisbelow(True)
            a.axvline(0.0, color="#b9b8b2", lw=1.2, ls=(0, (4, 3)), zorder=1)

        a0.plot(t, m_a, lw=2.0, color=ANALYTIC, zorder=3, label="analytic heuristic")
        a0.plot(t, m_l, lw=2.0, color=LEARNED, zorder=4, label="learned")
        a0.set_title(f"{kind}   (n={n_ev} events, N=8 corners)", color=INK,
                     fontsize=11.5, loc="left", pad=9)
        a0.set_ylim(-4.2, 3.2)

        a1.plot(t, m_g, lw=2.0, color=TRUTH, zorder=3)
        # slot 3 is below 3:1 on this surface -> direct label, not a legend chip
        a1.annotate("contact ground truth", xy=(t[int(W * 0.12)], m_g[int(W * 0.12)]),
                    xytext=(6, 8), textcoords="offset points",
                    fontsize=9, color=INK2, zorder=5)
        a1.set_ylim(-0.05, 1.05)
        a1.set_xlabel(f"time relative to {kind} [ms]", color=INK2, fontsize=10)
        a1.set_xlim(t[0], t[-1])

    ax[0, 0].set_ylabel("log$_{10}$ tr $\\Sigma_C$", color=INK2, fontsize=10)
    ax[1, 0].set_ylabel("fraction of corners in contact", color=INK2, fontsize=10)
    leg = ax[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(INK2)

    fig.suptitle("Learned $\\Sigma_C$ is a swing plateau, not a pair of event peaks",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}   (rollout {roll}, N_c={n_c}, diag_param={dparam})")


if __name__ == "__main__":
    main()
