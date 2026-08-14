#!/usr/bin/env python
"""Gate E: collect a stratified env-DR rollout pool at a given contact count.

Stratification is EXPLICIT rather than sampled: seeds are dealt round-robin over
the terrain list so a short night still yields balanced coverage. Sampling the
terrain per seed would, on a run that gets cut off early, silently leave one
terrain with zero rollouts and the pooled metrics would hide it.

Each DR axis draws from its OWN `default_rng` (plan Gate E: terrain _|_ friction
_|_ pushes _|_ IMU _|_ command), which `collect_rollout` already honours -- the
friction/push streams are seeded there off the rollout seed, independent of the
IMU noise and command schedule streams.

Usage:
    uv run python scripts/collect_dr_pool.py --contacts-per-foot 4 --seeds 24 --seconds 45
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_policy as rp
rp.DT = 0.001
rp.DECIMATION = 20            # same coherent 1 kHz regime as run_contactnet.py

import numpy as np

import invariant_estimation  # noqa: F401  (x64)
from invariant_estimation.sim import collect
from invariant_estimation.contactnet.config import ContactNetConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts-per-foot", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=24, help="number of rollouts")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--terrains", type=str, nargs="+",
                    default=["flat", "waves", "stepping_stones", "hard_stepping"])
    ap.add_argument("--out-dir", type=str, default=str(REPO / "data"))
    ap.add_argument("--time-budget-s", type=float, default=7200.0)
    ap.add_argument("--name-tag", type=str, default="",
                    help="filename tag, e.g. n8fix. Use to keep a re-collected pool "
                         "(e.g. after the waves-seed fix) from writing over an "
                         "existing same-N pool.")
    # --- motion-randomization knobs (see RUNNING.md, "breaking the stride clock") ---
    # Default cmd_resample_s=3.0 is SLOWER than the ~1.0 s stride, so the gait settles
    # into a clean limit cycle and gait phase becomes a near-deterministic function of
    # the sensor window -- which lets ContactNet regress the phase-conditional mean of
    # Sigma_C instead of learning contact condition (measured: phase R^2 = 0.942).
    # Resampling below the stride, and allowing the ranges to reach zero (stop-start,
    # turn-in-place), destroys that shortcut.
    ap.add_argument("--cmd-resample-s", type=float, default=None,
                    help="command resample period [s]. Below the ~1.0 s stride to "
                         "break gait periodicity; 3.0 (default cfg) preserves it.")
    ap.add_argument("--cmd-vx", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--cmd-vy", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--cmd-yaw", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--disturb-rate-hz", type=float, default=None,
                    help="push rate; raising it also breaks periodicity")
    args = ap.parse_args()

    over = {}
    if args.cmd_resample_s is not None: over["cmd_resample_s"] = args.cmd_resample_s
    if args.cmd_vx  is not None: over["cmd_vx_range"]  = tuple(args.cmd_vx)
    if args.cmd_vy  is not None: over["cmd_vy_range"]  = tuple(args.cmd_vy)
    if args.cmd_yaw is not None: over["cmd_yaw_range"] = tuple(args.cmd_yaw)
    if args.disturb_rate_hz is not None: over["disturb_rate_hz"] = args.disturb_rate_hz
    cfg = ContactNetConfig(env_dr=True, **over)
    if over:
        print(f"motion randomization overrides: {over}", flush=True)
    c = collect.build_collector(contacts_per_foot=args.contacts_per_foot, verbose=True)
    print(f"collecting N={c.fused.n_contacts} pool, terrains={args.terrains}, "
          f"{args.seeds} rollouts of {args.seconds}s -> {args.out_dir}", flush=True)

    t_start = time.time()
    metas, failures = [], []
    for k in range(args.seeds):
        if time.time() - t_start > args.time_budget_s:
            print(f"!! time budget hit after {k} rollouts -- stopping", flush=True)
            break
        seed = args.seed0 + k
        terrain = args.terrains[k % len(args.terrains)]      # round-robin = stratified
        t0 = time.time()
        try:
            roll = collect.collect_rollout(
                seed, args.seconds, terrain=terrain, collector=c, cfg=cfg,
                out_dir=args.out_dir, name_tag=args.name_tag, verbose=True)
            metas.append(roll.meta)
            print(f"  [{k + 1}/{args.seeds}] {terrain}/seed{seed} OK "
                  f"({time.time() - t0:.0f}s, mu={roll.meta['friction_mu']:.2f})", flush=True)
        except RuntimeError as e:
            failures.append({"seed": seed, "terrain": terrain, "error": str(e)})
            print(f"  [{k + 1}/{args.seeds}] {terrain}/seed{seed} FELL: {e}", flush=True)

    # Per-terrain survival is the Gate E STOP signal: if a tier mostly fails, the
    # flat-trained policy cannot walk it and the plan says drop it and SAY SO.
    by_terrain = {}
    for t in args.terrains:
        ok = sum(1 for m in metas if m["terrain"] == t)
        bad = sum(1 for f in failures if f["terrain"] == t)
        by_terrain[t] = {"ok": ok, "failed": bad}
    summary = {
        "n_contacts": int(c.fused.n_contacts),
        "requested": args.seeds,
        "collected": len(metas),
        "failed": len(failures),
        "by_terrain": by_terrain,
        "wall_s": time.time() - t_start,
        "failures": failures,
        "friction_mu": [m["friction_mu"] for m in metas],
    }
    out = Path(args.out_dir) / f"dr_pool_n{c.fused.n_contacts}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(by_terrain, indent=2), flush=True)
    print(f"-> {out}  ({len(metas)} ok, {len(failures)} fell, "
          f"{summary['wall_s'] / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
