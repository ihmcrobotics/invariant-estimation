#!/usr/bin/env python
"""Vertical drift for every trained checkpoint, and whether it ranks like RMSE.

Motivation. Every ContactNet arm is SELECTED on held-out body-frame velocity RMSE,
but the 2026-08-07 study found the drift ranking of arms A/B/C/D is nearly the
INVERSE of the velocity-RMSE ranking. If that holds, the grid is optimising and
ranking on a proxy anti-correlated with the goal, and picking a winner from the RMSE
table would pick the wrong arm. This script measures drift directly, from checkpoints
that already exist, and prints the two rankings side by side so the question
"are we ranking the thing we care about?" is answered rather than assumed.

Method. Replay the filter over each held-out rollout's usable region -- the same
region and the same seeding `run_contactnet.validate` uses (state seeded from truth
at t0, so vertical error starts at exactly zero) -- and report:

    drift_z    slope of (p_hat_z - p_true_z) in m/s, least-squares over the run.
               This is the quantity the z-drift reports quote.
    final_ez   accumulated vertical error at the end of the region [m].
    horiz_pct  final horizontal error as a percentage of path length, the
               companion number those reports track so a vertical "win" bought by
               wrecking horizontal accuracy is visible rather than hidden.

Each is reported for the LEARNED Sigma_C and for the recorded analytic
`contact_chol` baseline, on identical inputs, so the ratio is the meaningful figure.

**Every cell is evaluated over an identical region.** `prepare` sets
`t_hi = T - L`, so using each cell's own L would give cells with different L slightly
different evaluation spans and make the drift numbers incomparable -- which is
exactly the sort of silent mismatch this whole exercise keeps turning up. EVAL_L
pins one region for all of them.

This is a REPLAY drift measurement over recorded rollouts, not a fresh closed-loop
sim. That is a fair comparison between checkpoints (identical inputs, identical
region) and it is cheap, but it does NOT replace `experiments/z_budget.py`, which
attributes the sink to the specific filter write it flows through and can sweep
terrain/command conditions to catch cancellations. Use this to RANK; use z_budget to
explain and to check the sign transfers across conditions.

Usage:
    uv run --extra gpu python scripts/drift_backfill.py
    uv run --extra gpu python scripts/drift_backfill.py --root results/l_ablation
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

import run_contactnet as rc            # sets rp.DT=1e-3, rp.DECIMATION=20 at import
import jax
import jax.numpy as jnp

import invariant_estimation  # noqa: F401  (x64)
from invariant_estimation.contactnet import dataset, network, normalize
from invariant_estimation.contactnet import rollout as cn_rollout
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.inEKF.filter import init_carry, make_step
from invariant_estimation.inEKF.state import InEKFState
from invariant_estimation.sim import collect

# One evaluation region for every cell, independent of the L each was trained at.
EVAL_L = 1024


def drift_of(prep, cache, norm, cfg, params, fused, P0, eps, dt):
    """(baseline, learned) drift dicts for one held-out rollout."""
    wins_full = rc._windows_full(cache, norm, cfg)
    t0, t1 = prep.t_lo, prep.t_hi + cfg.L
    sl = slice(t0, t1)
    wins = jnp.asarray(wins_full[sl])
    inputs = jax.tree.map(lambda a: jnp.asarray(a[sl]), prep.inputs)

    R0, p0 = prep.R_true[t0], prep.p_true[t0]
    d0 = np.einsum("ij,kj->ki", R0, prep.y_fk[t0]) + p0[None, :]
    state0 = InEKFState(R=jnp.asarray(R0), v=jnp.asarray(prep.v_true[t0]),
                        p=jnp.asarray(p0), d=jnp.asarray(d0), P=jnp.asarray(P0))
    step = make_step(fused.ekf, fused.kinematics)

    p_true = np.asarray(prep.p_true[sl])
    t = np.arange(p_true.shape[0]) * dt
    # Path length from TRUTH, so the horizontal-error denominator does not itself
    # depend on the estimate being scored.
    path = float(np.sum(np.linalg.norm(np.diff(p_true[:, :2], axis=0), axis=1)))

    def run(contact_chol):
        xs = inputs._replace(contact_chol=contact_chol)
        _, out = jax.lax.scan(step, init_carry(state0), xs)
        p_est = np.asarray(out.state.p)
        e = p_est - p_true
        # Seeded from truth at t0, so e[0] == 0 and the slope is drift from a
        # zero-error start -- the same convention the z-drift reports use.
        slope = float(np.polyfit(t, e[:, 2], 1)[0])
        horiz = float(np.linalg.norm(e[-1, :2]))
        dof = 3 * out.state.d.shape[-2]
        return dict(drift_z=slope, final_ez=float(e[-1, 2]),
                    horiz_pct=100.0 * horiz / max(path, 1e-9),
                    nis_over_dof=float(jnp.mean(out.contact_diagnostics.nis)) / dof,
                    seconds=float(t[-1]), path_m=path)

    baseline = run(inputs.contact_chol)
    learned = run(cn_rollout.contact_factors(params, wins, eps))
    return baseline, learned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/l_ablation")
    ap.add_argument("--pool", default="n8fix")
    ap.add_argument("--contacts-per-foot", type=int, default=4)
    ap.add_argument("--contact-meas-var", type=float, default=1.0e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma-q-scale", nargs="+", type=float, default=None,
                    help="rescale recorded Sigma_q per joint (1 value = uniform, "
                         "9 = per-joint). Joint NEES is 48.5 vs a target of 9, so the "
                         "joint KF is overconfident ~5.4x, structured 0.56-6.41 across "
                         "joints. Tests whether an ISOTROPIC correction suffices -- "
                         "invariant I9 says it should not.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to cells whose name matches one of these exactly")
    args = ap.parse_args()

    root = Path(args.root) if Path(args.root).is_absolute() else REPO / Path(args.root)
    cells = sorted(p.parent for p in root.glob("L*_*/params.npz"))
    if args.only:
        cells = [c for c in cells if c.name in set(args.only)]
    if not cells:
        raise SystemExit(f"no cells with params.npz under {root}")
    print(f"cells: {[c.name for c in cells]}")

    cfg = ContactNetConfig(L=EVAL_L)
    c = collect.build_collector(contacts_per_foot=args.contacts_per_foot,
                                contact_meas_var=args.contact_meas_var)

    pool = sorted(collect.DATA_DIR.glob(f"*_{args.pool}_seed*.npz"))
    by_terrain = {}
    for p in pool:
        by_terrain.setdefault(p.name.split(f"_{args.pool}_")[0], []).append(p)
    val_paths = [v[-1] for v in by_terrain.values() if v]
    train_paths = [p for p in pool if p not in set(val_paths)]
    print(f"held-out: {[p.name for p in val_paths]}")

    dataset.build_channel_cache(train_paths + val_paths, c)
    pool_cmv = rc.pool_contact_meas_var(train_paths + val_paths)
    caches = {p: dataset.load_channel_cache(dataset.cache_path(p)) for p in val_paths}

    # Each cell is evaluated under ITS OWN frozen normalization, loaded from the
    # checkpoint directory -- never refit here. `params.npz` and `norm_constants.npz`
    # are a matched pair (RUNNING.md: "load them together -- a mismatch silently
    # shifts the input distribution"). Refitting from this pool would be invisible
    # and harmless for a cell trained on this pool, and silently wrong for one
    # trained on another, which is exactly the comparison we want to make: train on
    # randomized motion, evaluate on the walking distribution we deploy into.
    #
    # Distinct norms are grouped so `prepare` runs once per norm, not once per cell.
    def norm_key(cell):
        z = np.load(cell / "norm_constants.npz")
        return (z["mean"].tobytes(), z["std"].tobytes())

    def load_norm(cell):
        z = np.load(cell / "norm_constants.npz")
        return normalize.NormConstants(
            mean=jnp.asarray(z["mean"], dtype=jnp.float64),
            std=jnp.asarray(z["std"], dtype=jnp.float64),
            names=tuple(str(s) for s in z["names"]),
            n_ticks=0, source=f"loaded from {cell.name}",
            floored=tuple(str(s) for s in z["floored"]))

    groups = {}
    for cell in cells:
        groups.setdefault(norm_key(cell), []).append(cell)
    print(f"{len(groups)} distinct normalization(s) across {len(cells)} cells")

    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                        cfg.sigma_0, cfg.eps)
    from invariant_estimation.contactnet import train as cn_train

    rows = []
    for key, group in groups.items():
        norm = load_norm(group[0])
        train_preps = dataset.apply_contact_meas_floor(
            dataset.prepare(train_paths, norm, cfg), args.contact_meas_var, pool_cmv)
        val_preps = dataset.apply_contact_meas_floor(
            dataset.prepare(val_paths, norm, cfg), args.contact_meas_var, pool_cmv)
        if args.sigma_q_scale:
            sc = (args.sigma_q_scale[0] if len(args.sigma_q_scale) == 1
                  else args.sigma_q_scale)
            train_preps = dataset.scale_sigma_q(train_preps, sc)
            val_preps = dataset.scale_sigma_q(val_preps, sc)
        P0 = dataset.measure_p0(c.fused, train_preps[0], cfg, ticks=3000)
        vcache = {vp.name: caches[p] for vp, p in zip(val_preps, val_paths)}

        for cell in group:
            params = cn_train.load_params(str(cell / "params.npz"), like)
            per = [drift_of(vp, vcache[vp.name], norm, cfg, params, c.fused, P0,
                            cfg.eps, cfg.dt) for vp in val_preps]
            mean = lambda arm, k: float(np.mean([r[arm][k] for r in per]))
            summ = json.loads((cell / "summary.json").read_text())
            rmse = float(np.mean([v["learned"]["vel_rmse"] for v in summ["val"]]))
            rows.append(dict(
                cell=cell.name, L=summ["cfg"]["L"], objective=summ["cfg"]["objective"],
                vel_rmse=rmse,
                drift_z=mean(1, "drift_z"), base_drift_z=mean(0, "drift_z"),
                final_ez=mean(1, "final_ez"), base_final_ez=mean(0, "final_ez"),
                horiz_pct=mean(1, "horiz_pct"), base_horiz_pct=mean(0, "horiz_pct"),
                nis_over_dof=mean(1, "nis_over_dof"),
                base_nis_over_dof=mean(0, "nis_over_dof"),
                contact_meas_var=float(args.contact_meas_var),
                seconds=per[0][1]["seconds"],
                per_rollout=[{"name": vp.name,
                              "drift_z": r[1]["drift_z"],
                              "final_ez": r[1]["final_ez"],
                              "base_drift_z": r[0]["drift_z"],
                              "nis_over_dof": r[1]["nis_over_dof"]}
                             for vp, r in zip(val_preps, per)]))
            print(f"  {cell.name:22s} drift_z {rows[-1]['drift_z']:+.5f} m/s "
              f"(analytic {rows[-1]['base_drift_z']:+.5f})  rmse {rmse:.4f}", flush=True)

    out = Path(args.out) if args.out else root / "drift_backfill.json"
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n### Drift vs the metric the arms were selected on "
          f"({rows[0]['seconds']:.0f} s of held-out walking per rollout)\n")
    print("| cell | L | vel RMSE | |drift_z| [m/s] | analytic | ratio | final e_z | horiz % |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: abs(r["drift_z"])):
        ratio = abs(r["drift_z"]) / max(abs(r["base_drift_z"]), 1e-12)
        print(f"| {r['cell']} | {r['L']} | {r['vel_rmse']:.4f} | "
              f"{abs(r['drift_z']):.5f} | {abs(r['base_drift_z']):.5f} | "
              f"{ratio:.2f}x | {r['final_ez']:+.3f} | {r['horiz_pct']:.2f} |")

    # The decisive comparison: does ranking by drift agree with ranking by RMSE?
    by_drift = [r["cell"] for r in sorted(rows, key=lambda r: abs(r["drift_z"]))]
    by_rmse = [r["cell"] for r in sorted(rows, key=lambda r: r["vel_rmse"])]
    print(f"\nrank by |drift_z| : {by_drift}")
    print(f"rank by vel_rmse  : {by_rmse}")
    n = len(rows)
    if n > 2:
        pos = {c: i for i, c in enumerate(by_rmse)}
        d = np.array([pos[c] for c in by_drift], dtype=float)
        rho = 1.0 - 6.0 * np.sum((d - np.arange(n)) ** 2) / (n * (n * n - 1))
        print(f"Spearman(drift, rmse) = {rho:+.2f}")
        if rho < 0.3:
            print("\n>>> The rankings DISAGREE. Selecting an arm on velocity RMSE does "
                  "not select it for drift, so the RMSE table must not be used to "
                  "choose a direction. This reproduces the 2026-08-07 finding.")
        else:
            print("\n>>> The rankings AGREE, so RMSE is a usable proxy for drift under "
                  "this configuration -- which would RETIRE the 2026-08-07 inversion, "
                  "most likely because the contact R floor changed the regime.")


if __name__ == "__main__":
    main()
