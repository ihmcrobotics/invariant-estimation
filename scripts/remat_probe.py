#!/usr/bin/env python
"""Measure what rematerialization actually costs and buys, per BPTT length L.

Answers the one question the L-ablation cannot start without: does L=512 fit on this
card, and what does remat charge for it? Reports, for each L x remat cell:

    temp_MB     compiled scratch memory for the backward pass, read statically off
                the executable (`memory_analysis().temp_size_in_bytes`). Deterministic,
                available on CPU, and unaffected by allocator history.
    peak_MB     runtime device high-water mark (`peak_bytes_in_use`). The number that
                decides OOM. Only meaningful on an accelerator, and only across
                processes -- the high-water mark never resets within one, which is why
                every cell runs in its own subprocess.
    s_step      wall seconds per training step, post-compile.
    OOM         the cell could not run. This is a RESULT, not a failure: it is the
                measurement that says L=512 requires remat.

Cells are independent, so a crash in one does not cost the others.

Usage:
    uv run --extra gpu python scripts/remat_probe.py
    uv run --extra gpu python scripts/remat_probe.py --L 128 256 512 --steps 5
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "l_ablation" / "remat_probe.json"

# Two rollouts is enough: activation memory and step time are set by (L, B, N, H),
# not by how many rollouts the batcher draws from. Keeps each subprocess short.
PROBE_ROLLOUTS = 2


def run_cell(L, remat, steps, contacts_per_foot, pool, contact_meas_var):
    """One cell, in-process. Prints a JSON line; the parent collects it."""
    import numpy as np
    import jax

    import run_policy as rp
    rp.DT = 0.001
    rp.DECIMATION = 20

    import invariant_estimation  # noqa: F401  (x64)
    from invariant_estimation.sim import collect
    from invariant_estimation.contactnet import dataset, network, train as cn_train
    from invariant_estimation.contactnet import rollout as cn_rollout
    from invariant_estimation.contactnet.config import ContactNetConfig

    cfg = ContactNetConfig(L=L, remat=remat)
    c = collect.build_collector(contacts_per_foot=contacts_per_foot, verbose=False,
                                contact_meas_var=contact_meas_var)
    paths = sorted(collect.DATA_DIR.glob(f"*_{pool}_seed*.npz"))[:PROBE_ROLLOUTS]
    if not paths:
        raise SystemExit(f"no rollouts matching *_{pool}_seed*.npz")

    dataset.build_channel_cache(paths, c, verbose=False)
    norm = dataset.fit_normalization(paths)
    preps = dataset.prepare(paths, norm, cfg)
    P0 = dataset.measure_p0(c.fused, preps[0], cfg)
    params = network.init(jax.random.PRNGKey(cfg.init_seed), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    warm_in = cn_rollout.make_warm_in(c.fused.ekf, c.fused.kinematics)
    batcher = dataset.ChainedBatcher(preps, cfg, P0, warm_in, seed=cfg.batcher_seed)

    batch_loss = cn_rollout.make_batch_loss(
        c.fused.ekf, c.fused.kinematics, cfg.eps, beta=cfg.beta,
        objective=cfg.objective, remat=cfg.remat)
    tx = cn_train.make_optimizer(cfg.peak_lr, steps, 1, cfg.max_norm, cfg.weight_decay)
    opt_state = tx.init(params)
    step_fn = cn_train.make_train_step(batch_loss, tx, cfg.dof)

    # Static scratch estimate, before anything runs: the number that survives on a
    # CPU-only machine. Lowered from `step_fn` -- the real value_and_grad train step
    # -- and NOT from `batch_loss`: remat only trades memory on the BACKWARD pass, so
    # a forward-only lowering reports identical scratch for on and off and reads as
    # "remat buys nothing", which is how this column was wrong the first time.
    batch, carry0 = batcher.batch()
    temp_bytes = (step_fn.lower(params, opt_state, batch, carry0)
                  .compile().memory_analysis().temp_size_in_bytes)

    # One warm step to compile, excluded from the timing.
    params, opt_state, metrics, carry = step_fn(params, opt_state, batch, carry0)
    jax.block_until_ready(metrics.loss)
    batcher.update(carry)

    t0 = time.time()
    for _ in range(steps):
        batch, carry0 = batcher.batch()
        params, opt_state, metrics, carry = step_fn(params, opt_state, batch, carry0)
        jax.block_until_ready(metrics.loss)
        batcher.update(carry)
    s_step = (time.time() - t0) / steps

    dev = jax.local_devices()[0]
    try:
        peak = float(dev.memory_stats()["peak_bytes_in_use"])
    except (AttributeError, KeyError, TypeError):
        peak = None   # CPU backend exposes no allocator stats

    return dict(L=L, remat=remat, ok=True, temp_MB=temp_bytes / 1e6,
                peak_MB=(peak / 1e6 if peak is not None else None),
                s_step=s_step, loss=float(metrics.loss),
                grad_norm=float(metrics.grad_norm), device=str(dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--contacts-per-foot", type=int, default=4)
    ap.add_argument("--pool", type=str, default="n8fix")
    ap.add_argument("--contact-meas-var", type=float, default=1.0e-4)
    ap.add_argument("--cell", type=int, nargs=2, default=None,
                    help=argparse.SUPPRESS)   # internal: --cell L REMAT(0|1)
    args = ap.parse_args()

    if args.cell is not None:
        L, remat = args.cell
        rec = run_cell(L, bool(remat), args.steps, args.contacts_per_foot,
                       args.pool, args.contact_meas_var)
        print("PROBE_JSON " + json.dumps(rec))
        return

    rows = []
    for L in args.L:
        for remat in (False, True):
            print(f"== cell L={L} remat={remat} ==", flush=True)
            cmd = [sys.executable, __file__, "--cell", str(L), str(int(remat)),
                   "--steps", str(args.steps), "--pool", args.pool,
                   "--contacts-per-foot", str(args.contacts_per_foot),
                   "--contact-meas-var", str(args.contact_meas_var)]
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines()
                         if l.startswith("PROBE_JSON ")), None)
            if line is None:
                tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
                oom = any("RESOURCE_EXHAUSTED" in t or "out of memory" in t.lower()
                          for t in (p.stderr or "").splitlines())
                rows.append(dict(L=L, remat=remat, ok=False,
                                 reason="OOM" if oom else "FAILED",
                                 tail=" | ".join(tail)))
                print(f"   -> {'OOM' if oom else 'FAILED'}: {' | '.join(tail)}",
                      flush=True)
                continue
            rec = json.loads(line[len("PROBE_JSON "):])
            rows.append(rec)
            peak = f"{rec['peak_MB']:.0f}" if rec["peak_MB"] is not None else "n/a"
            print(f"   -> temp {rec['temp_MB']:.0f} MB  peak {peak} MB  "
                  f"{rec['s_step']:.3f} s/step", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))

    print("\n| L | remat | temp MB | peak MB | s/step | vs remat=off |")
    print("|---|---|---|---|---|---|")
    by = {(r["L"], r["remat"]): r for r in rows if r.get("ok")}
    for r in rows:
        if not r.get("ok"):
            print(f"| {r['L']} | {r['remat']} | — | — | — | **{r['reason']}** |")
            continue
        peak = f"{r['peak_MB']:.0f}" if r["peak_MB"] is not None else "n/a"
        ref = by.get((r["L"], False))
        rel = ("—" if (not r["remat"] or ref is None) else
               f"{r['temp_MB']/ref['temp_MB']:.2f}x mem, "
               f"{r['s_step']/ref['s_step']:.2f}x time")
        print(f"| {r['L']} | {r['remat']} | {r['temp_MB']:.0f} | {peak} | "
              f"{r['s_step']:.3f} | {rel} |")
    print(f"\nwritten -> {OUT}")
    print("Gradient identity (remat on vs off) is asserted separately and "
          "continuously by tests/contactnet/test_remat.py.")


if __name__ == "__main__":
    main()
