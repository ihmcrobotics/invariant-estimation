#!/usr/bin/env python
"""Evaluate a finished training run from its checkpoint, and dump Sigma_C diagnostics.

Split out from `run_contactnet.py` so a validation failure cannot cost the
training that preceded it -- which is exactly what happened: validation OOM'd the
GPU after 300 steps had already completed, and the run died with a checkpoint on
disk but no metrics. This reads `params.npz` + `norm_constants.npz` + `P0.npy`
back and redoes only the cheap part.

Also writes `sigma_diag.npz` (Sigma_C, per-corner contact force, conditioning and
applied mask over a window) which `plot_sigma_c.py` turns into the Gate G figures.

Usage:
    uv run python scripts/evaluate_run.py --run results/<dir> [--pool n8] [--contacts-per-foot 4]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_policy as rp
rp.DT = 0.001
rp.DECIMATION = 20

import numpy as np
import jax
import jax.numpy as jnp

import invariant_estimation  # noqa: F401
from invariant_estimation.contactnet import dataset, network, normalize, rollout as cn_rollout
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.sim import collect

import importlib
_run = importlib.import_module("run_contactnet") if False else None
sys.path.insert(0, str(REPO / "scripts"))
import run_contactnet as R                                   # reuse validate()


def load_params(path):
    z = np.load(path, allow_pickle=True)
    if "tree" in z:                       # pytree pickled whole
        return jax.tree.map(jnp.asarray, z["tree"].item())
    # flat arrays keyed by path
    from jax.tree_util import tree_unflatten
    keys = sorted(z.files)
    return [jnp.asarray(z[k]) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--contacts-per-foot", type=int, default=1)
    ap.add_argument("--objective", type=str, default="l2_velocity")
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[24, 25, 26, 27])
    ap.add_argument("--diag-ticks", type=int, default=4000)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = ContactNetConfig(objective=args.objective)
    c = collect.build_collector(contacts_per_foot=args.contacts_per_foot, verbose=True)

    if args.pool:
        pool = sorted(collect.DATA_DIR.glob(f"*_{args.pool}_seed*.npz"))
        by = {}
        for p in pool:
            by.setdefault(p.name.split(f"_{args.pool}_")[0], []).append(p)
        val_paths = [v[-1] for v in by.values() if v]
    else:
        val_paths = [collect.DATA_DIR / f"flat_seed{s:03d}.npz" for s in args.val_seeds]
    val_paths = [p for p in val_paths if p.exists()]
    if not val_paths:
        raise SystemExit("no validation rollouts found")
    print(f"val: {[p.name for p in val_paths]}")

    nz = np.load(run / "norm_constants.npz", allow_pickle=True)
    norm = normalize.NormConstants(
        mean=jnp.asarray(nz["mean"]), std=jnp.asarray(nz["std"]),
        names=tuple(str(s) for s in nz["names"]),
        floored=tuple(str(s) for s in nz["floored"]),
        n_ticks=int(nz["n_ticks"]) if "n_ticks" in nz else 0,
        source=str(nz["source"]) if "source" in nz else f"reloaded from {run.name}")
    P0 = np.load(run / "P0.npy")
    # The checkpoint is bare leaves; the tree STRUCTURE has to come from a
    # freshly-initialised net of the same shape (`train.load_params`'s contract).
    from invariant_estimation.contactnet import train as cn_train
    like = network.init(jax.random.PRNGKey(cfg.init_seed), cfg.d_in, cfg.widths,
                        cfg.sigma_0, cfg.eps)
    params = cn_train.load_params(str(run / "params.npz"), like)

    dataset.build_channel_cache(val_paths, c, verbose=True)
    preps = dataset.prepare(val_paths, norm, cfg, verbose=False)

    metrics = []
    for prep, vpath in zip(preps, val_paths):
        cache = dataset.load_channel_cache(dataset.cache_path(vpath))
        base, learned = R.validate(prep, cache, norm, cfg, params, c.fused, P0, cfg.eps)
        label = (vpath.name.split(f"_{args.pool}_")[0] if args.pool
                 else vpath.name.replace(".npz", ""))
        print(f"  {label}: baseline={base}\n           learned={learned}", flush=True)
        metrics.append({"label": label, "rollout": vpath.name,
                        "baseline": base, "learned": learned})

    (run / "eval.json").write_text(json.dumps(metrics, indent=2))
    print(f"-> {run/'eval.json'}")

    # ---- Sigma_C over a stride, from the first val rollout -----------------
    prep, vpath = preps[0], val_paths[0]
    cache = dataset.load_channel_cache(dataset.cache_path(vpath))
    wins = R._windows_full(cache, norm, cfg)
    t0 = prep.t_lo
    n = min(args.diag_ticks, wins.shape[0] - t0)
    Lc = np.asarray(cn_rollout.contact_factors(
        params, jnp.asarray(wins[t0:t0 + n]), cfg.eps))
    sigma = np.einsum("tnij,tnkj->tnik", Lc, Lc)          # L L^T
    roll = collect.load_rollout(vpath)
    force = np.asarray(roll.sensors.contact)[t0:t0 + n]   # per-slot trust/contact
    np.savez(run / "sigma_diag.npz", sigma=sigma, force=force)
    print(f"-> {run/'sigma_diag.npz'}  sigma {sigma.shape}")


if __name__ == "__main__":
    main()
