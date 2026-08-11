#!/usr/bin/env python
"""Does the DEPLOYED ContactNet produce the same Sigma_C as the trained one?

Motivation, 2026-08-10. The recommended checkpoint measures -0.085 m of vertical
error in offline replay and +2.49 m in closed loop -- 21x WORSE than the analytic
baseline it beats by 2.3x offline. The analytic baseline behaves correctly in the same
closed-loop harness and the contact R floor is exonerated (floor 0 is equally bad), so
the suspect is the ONLINE path: same weights, different Sigma_C.

RESULTS.md records this oracle passing at ~1e-15 for the F=30 / N=2 configuration.
Either it regressed at N=8, or the online feature path has a mismatch that only
appears there. This script re-runs it against a real rollout.

Three comparisons, cheapest first, because each localises the fault differently:

  1. SUBCHAIN. The online provider builds its subchain from `reader.unfiltered_names`
     (run_estimator.py:368); the offline cache builds it from
     `collect._unfiltered_names(collector)` (dataset.build_channel_cache). If those
     disagree, the network is fed different joints online and nothing downstream
     would notice.
  2. FEATURES. Online per-tick windows vs the offline `features.window` over the
     cached channels, on identical sensors.
  3. Sigma_C. The end-to-end quantity the filter actually consumes.

A mismatch at (1) or (2) means the deployment path is broken and the offline numbers
describe a network that has never actually run. Agreement at all three would mean the
online path is faithful and the closed-loop failure is a genuine property of the
network under that trajectory distribution -- a very different conclusion.

Usage:
    uv run --extra gpu python scripts/online_offline_oracle.py \
        --ckpt results/zdrift/L256_A_l2vel_cmv1e-3
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

import run_policy as rp
rp.DT = 0.001
rp.DECIMATION = 20

import jax
import jax.numpy as jnp

import invariant_estimation  # noqa: F401  (x64)
from invariant_estimation.contactnet import (
    dataset, features as cn_features, network as cn_network,
    normalize as cn_normalize, online as cn_online, rollout as cn_rollout,
    train as cn_train)
from invariant_estimation.contactnet.checkpoint import config_for_checkpoint
from invariant_estimation.sim import collect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run directory with params.npz")
    ap.add_argument("--pool", default="n8fix")
    ap.add_argument("--contacts-per-foot", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=3000)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = config_for_checkpoint(str(ckpt))
    c = collect.build_collector(contacts_per_foot=args.contacts_per_foot, verbose=False)
    path = sorted(collect.DATA_DIR.glob(f"*_{args.pool}_seed*.npz"))[0]
    print(f"rollout: {path.name}   ckpt: {ckpt.name}")

    # ---- 1. subchain -------------------------------------------------------
    off_names = collect._unfiltered_names(c)
    reader = getattr(c, "reader", None)
    on_names = getattr(reader, "unfiltered_names", None)
    print("\n[1] subchain source")
    print(f"    offline (collect._unfiltered_names) : {list(off_names)}")
    if on_names is None:
        print("    online  (reader.unfiltered_names)   : <collector exposes no reader; "
              "compared via the estimator loop below>")
    else:
        print(f"    online  (reader.unfiltered_names)   : {list(on_names)}")
        print(f"    IDENTICAL: {tuple(off_names) == tuple(on_names)}")
    sub_off = cn_features.subchain_for(c.fused, off_names)
    print(f"    offline subchain: {len(sub_off)} contacts, "
          f"{len(sub_off[0].joint_idx) if hasattr(sub_off[0], 'joint_idx') else '?'} joints/contact")

    # ---- load the checkpoint's frozen constants ---------------------------
    z = np.load(ckpt / "norm_constants.npz", allow_pickle=False)
    consts = cn_normalize.NormConstants(
        mean=jnp.asarray(z["mean"], dtype=jnp.float64),
        std=jnp.asarray(z["std"], dtype=jnp.float64),
        names=tuple(str(s) for s in z["names"]),
        floored=tuple(str(s) for s in z["floored"]),
        n_ticks=0, source=str(ckpt))
    like = cn_network.init(jax.random.PRNGKey(cfg.init_seed), cfg.d_in, cfg.widths,
                           cfg.sigma_0, cfg.eps, cfg.diag_spec)
    params = cn_train.load_params(str(ckpt / "params.npz"), like)

    roll = collect.load_rollout(path)
    n = min(args.ticks, len(roll.sensors.encoders))
    sensors = jax.tree.map(lambda a: jnp.asarray(a[:n], dtype=jnp.float64), roll.sensors)

    # ---- 2. features: online ring buffer vs offline window ----------------
    feat_step = cn_online.make_online_features(
        sub_off, int(c.fused.base_imu), c.fused.kinematics, cfg, consts)
    ostate = cn_online.init_state(cfg, len(sub_off))

    def fstep(st, s):
        st, win, ready = feat_step(st, s)
        return st, (win, ready)

    _, (win_online, ready) = jax.lax.scan(fstep, ostate, sensors)
    win_online = np.asarray(win_online)           # (T, N_c, H, F)
    ready = np.asarray(ready)

    channels = cn_features.make_contact_channels(
        sub_off, int(c.fused.base_imu), c.fused.kinematics, c.dt)
    raw = np.asarray(collect.contact_channels_chunked(channels, roll.sensors, chunk=2000))[:n]
    x = np.asarray(cn_normalize.apply(jnp.asarray(raw), consts))
    win_offline = np.asarray(cn_features.window(jnp.asarray(x), cfg.H))   # (T, N_c, H, F)

    k = np.where(ready)[0]
    k = k[k < win_offline.shape[0]]
    print(f"\n[2] features, over {len(k)} ready ticks")
    if len(k):
        d = np.abs(win_online[k] - win_offline[k])
        scale = np.abs(win_offline[k]).max() + 1e-30
        print(f"    max abs diff {d.max():.3e}   max rel {d.max()/scale:.3e}")
        print(f"    VERDICT: {'MATCH' if d.max()/scale < 1e-9 else '** MISMATCH **'}")

    # ---- 3. Sigma_C end to end --------------------------------------------
    prov = cn_online.make_provider(sub_off, int(c.fused.base_imu), c.fused.kinematics,
                                   cfg, consts, params)
    _, chol_online = jax.lax.scan(prov, cn_online.init_state(cfg, len(sub_off)), sensors)
    chol_online = np.asarray(chol_online)
    chol_offline = np.asarray(cn_rollout.contact_factors(
        params, jnp.asarray(win_offline), cfg.eps, cfg.diag_spec))
    print(f"\n[3] Sigma_C (contact_chol) over {len(k)} ready ticks")
    if len(k):
        d = np.abs(chol_online[k] - chol_offline[k])
        scale = np.abs(chol_offline[k]).max() + 1e-30
        print(f"    max abs diff {d.max():.3e}   max rel {d.max()/scale:.3e}")
        print(f"    VERDICT: {'MATCH' if d.max()/scale < 1e-9 else '** MISMATCH **'}")
        tr_on = np.trace(chol_online[k] @ np.swapaxes(chol_online[k], -1, -2), axis1=-2, axis2=-1)
        tr_off = np.trace(chol_offline[k] @ np.swapaxes(chol_offline[k], -1, -2), axis1=-2, axis2=-1)
        print(f"    median tr(Sigma_C)  online {np.median(tr_on):.3e}   "
              f"offline {np.median(tr_off):.3e}")


if __name__ == "__main__":
    main()
