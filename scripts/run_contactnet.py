#!/usr/bin/env python
"""ContactNet overnight orchestrator: collect -> cache -> normalize -> train -> validate.

Coherent 1 kHz regime (rp.DT=0.001, rp.DECIMATION=20; CONTROL_DT=0.02 unchanged),
process socket only, F=30 (q-dot channel), stride=1 window. Held-out validation:
body-frame velocity RMSE + velocity NEES + contact NIS/dof, learned Sigma_C vs the
analytic-heuristic contact_chol baseline (the recorded stance/swing factors).

Usage:
    uv run python scripts/run_contactnet.py --collect --steps 300 --seconds 45
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
rp.DECIMATION = 20 # for the policy, NOT for the estimator framework. (Policies trained at 50Hz)

import numpy as np
import jax
import jax.numpy as jnp

import invariant_estimation  # noqa: F401  (x64)
from invariant_estimation.sim import collect
from invariant_estimation.contactnet import (
    dataset, features, network, normalize, train as cn_train, rollout as cn_rollout)
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.inEKF.filter import init_carry, make_step

RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)


def collect_rollouts(c, seeds, seconds):
    paths = []
    for s in seeds:
        p = collect.DATA_DIR / f"flat_seed{s:03d}.npz"
        if p.exists():
            print(f"  seed{s}: exists, skipping collection")
            paths.append(p)
            continue
        try:
            collect.collect_rollout(seed=s, seconds=seconds, collector=c, out_dir=collect.DATA_DIR)
            paths.append(p)
        except RuntimeError as e:
            print(f"  seed{s}: FELL/skipped: {e}")
    return paths


def save_norm(norm, path):
    np.savez(path, mean=np.asarray(norm.mean), std=np.asarray(norm.std),
             names=np.asarray(norm.names), floored=np.asarray(norm.floored))


def _windows_full(cache, norm, cfg):
    """(T, N_c, H, F) normalized feature windows over the whole stream (single boxcar)."""
    x = normalize.apply(jnp.asarray(cache["channels"]), norm)
    return np.asarray(features.window(x, cfg.H, cfg.stride))


def validate(prep, cache, norm, cfg, params, fused, P0, eps):
    """Run the filter over the held-out usable region with (a) recorded analytic
    contact_chol and (b) learned Sigma_C. Return body-frame velocity RMSE, velocity
    NEES (3-DoF), and contact NIS/dof for each."""
    from invariant_estimation.inEKF.state import InEKFState

    wins_full = _windows_full(cache, norm, cfg)          # (T, N_c, H, F)
    t0, t1 = prep.t_lo, prep.t_hi + cfg.L                 # usable region
    sl = slice(t0, t1)
    wins = jnp.asarray(wins_full[sl])
    inputs = jax.tree.map(lambda a: jnp.asarray(a[sl]), prep.inputs)

    R0, p0 = prep.R_true[t0], prep.p_true[t0]
    d0 = np.einsum("ij,kj->ki", R0, prep.y_fk[t0]) + p0[None, :]
    state0 = InEKFState(R=jnp.asarray(R0), v=jnp.asarray(prep.v_true[t0]),
                        p=jnp.asarray(p0), d=jnp.asarray(d0), P=jnp.asarray(P0))
    step = make_step(fused.ekf, fused.kinematics)

    v_true = jnp.asarray(prep.v_true[sl])
    R_true = jnp.asarray(prep.R_true[sl])

    def run(contact_chol):
        xs = inputs._replace(contact_chol=contact_chol)
        _, out = jax.lax.scan(step, init_carry(state0), xs)
        # body-frame velocity error
        bv_est = jnp.einsum("tji,tj->ti", out.state.R, out.state.v)
        bv_true = jnp.einsum("tji,tj->ti", R_true, v_true)
        rmse = float(jnp.sqrt(jnp.mean(jnp.sum((bv_est - bv_true) ** 2, axis=-1))))
        # 3-DoF world velocity NEES against the velocity block of P (tangent 3:6)
        ev = out.state.v - v_true
        Pvv = out.state.P[:, 3:6, 3:6]
        nees = jax.vmap(lambda e, P: e @ jnp.linalg.solve(P, e))(ev, Pvv)
        nees = float(jnp.mean(nees))
        nis = float(jnp.mean(out.contact_diagnostics.nis)) / cfg.dof
        applied = float(jnp.mean(out.contact_diagnostics.applied))
        return dict(vel_rmse=rmse, vel_nees=nees, nis_over_dof=nis, applied=applied)

    baseline = run(inputs.contact_chol)
    L_c = cn_rollout.contact_factors(params, wins, eps)
    learned = run(L_c)
    return baseline, learned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--time-budget-s", type=float, default=3600.0)
    args = ap.parse_args()

    t_start = time.time()
    c = collect.build_collector(policy_name="baseline", chunk_ticks=10_000)

    if args.collect:
        print("== collecting ==")
        collect_rollouts(c, args.train_seeds + args.val_seeds, args.seconds)

    train_paths = [collect.DATA_DIR / f"flat_seed{s:03d}.npz" for s in args.train_seeds]
    val_paths = [collect.DATA_DIR / f"flat_seed{s:03d}.npz" for s in args.val_seeds]
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [p for p in val_paths if p.exists()]
    print(f"train rollouts: {[p.name for p in train_paths]}")
    print(f"val rollouts:   {[p.name for p in val_paths]}")

    print("== building channel caches ==")
    dataset.build_channel_cache(train_paths + val_paths, c)

    print("== fitting normalization (train, walking set) ==")
    norm = dataset.fit_normalization(train_paths, source="overnight train set")
    print(f"  floored channels: {list(norm.floored)}")
    save_norm(norm, RESULTS / "norm_constants.npz")

    cfg = ContactNetConfig()
    print(f"  cfg: F={cfg.F} d_in={cfg.d_in} H={cfg.H} stride={cfg.stride} "
          f"L={cfg.L} B={cfg.B} objective={cfg.objective} episode_s={cfg.episode_s}")

    print("== preparing rollouts ==")
    train_preps = dataset.prepare(train_paths, norm, cfg, verbose=True)
    val_preps = dataset.prepare(val_paths, norm, cfg, verbose=True)

    print("== measuring P0 ==")
    P0 = dataset.measure_p0(c.fused, train_preps[0], cfg, ticks=3000)
    np.save(RESULTS / "P0.npy", P0)

    print("== building network + batcher ==")
    params = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    batch_loss = cn_rollout.make_batch_loss(
        c.fused.ekf, c.fused.kinematics, cfg.eps, beta=cfg.beta,
        objective=cfg.objective, remat=cfg.remat)
    warm_in = cn_rollout.make_warm_in(c.fused.ekf, c.fused.kinematics)
    batcher = dataset.ChainedBatcher(train_preps, cfg, P0, warm_in, seed=0)

    # Fast-fail: one train step before committing to the full run.
    print("== training ==")
    tx = cn_train.make_optimizer(cfg.peak_lr, args.steps, args.warmup_steps,
                                 cfg.max_norm, cfg.weight_decay)
    opt_state = tx.init(params)
    step_fn = cn_train.make_train_step(batch_loss, tx, cfg.dof)

    history, reseeds = [], 0
    for i in range(args.steps):
        batch, carry0 = batcher.batch()
        params, opt_state, metrics, carry = step_fn(params, opt_state, batch, carry0)
        reseeds += batcher.update(carry)
        history.append((float(metrics.loss), float(metrics.grad_norm),
                        float(metrics.nis_over_dof), float(metrics.applied_frac), reseeds))
        if i % 10 == 0 or i == args.steps - 1:
            print(f"step {i:4d} loss {history[-1][0]:.5e} |g| {history[-1][1]:.2e} "
                  f"NIS/dof {history[-1][2]:.3e} applied {history[-1][3]:.2f} reseeds {reseeds}")
        if time.time() - t_start > args.time_budget_s:
            print(f"  time budget hit at step {i}; stopping.")
            break

    cn_train.save_params(str(RESULTS / "params.npz"), params)
    hist = np.array(history)
    np.save(RESULTS / "history.npy", hist)

    print("== validating on held-out ==")
    val_metrics = []
    for vp, vpath in zip(val_preps, val_paths):
        cache = dataset.load_channel_cache(dataset.cache_path(vpath))
        base, learned = validate(vp, cache, norm, cfg, params, c.fused, P0, cfg.eps)
        print(f"  {vp.name}: baseline={base}  learned={learned}")
        val_metrics.append({"rollout": vp.name, "baseline": base, "learned": learned})

    summary = {
        "n_train": len(train_preps), "n_val": len(val_preps),
        "steps_run": len(history), "final_reseeds": reseeds,
        "floored": list(norm.floored),
        "cfg": {"F": cfg.F, "d_in": cfg.d_in, "H": cfg.H, "stride": cfg.stride,
                "L": cfg.L, "B": cfg.B, "objective": cfg.objective,
                "episode_s": cfg.episode_s, "warm_in_s": cfg.warm_in_s,
                "peak_lr": cfg.peak_lr},
        "wall_s": time.time() - t_start,
        "val": val_metrics,
        "loss_first": history[0][0] if history else None,
        "loss_last": history[-1][0] if history else None,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("== summary ==")
    print(json.dumps(summary, indent=2))
    make_plots(hist, val_metrics)


def make_plots(hist, val_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hist.size:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(hist[:, 0]); ax[0].set_title("training loss (l2_velocity)")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].set_yscale("log")
        ax[1].plot(hist[:, 2]); ax[1].axhline(1.0, ls="--", c="k", lw=0.8)
        ax[1].set_title("contact NIS / dof"); ax[1].set_xlabel("step")
        ax[2].plot(hist[:, 4]); ax[2].set_title("cumulative reseeds"); ax[2].set_xlabel("step")
        fig.tight_layout(); fig.savefig(RESULTS / "training.png", dpi=110); plt.close(fig)

    if val_metrics:
        names = [m["rollout"] for m in val_metrics]
        xb = np.arange(len(names))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for j, (key, title, ref) in enumerate([
                ("vel_rmse", "held-out body-frame velocity RMSE [m/s]", None),
                ("vel_nees", "held-out velocity NEES (target 3)", 3.0),
                ("nis_over_dof", "held-out contact NIS/dof (target 1)", 1.0)]):
            b = [m["baseline"][key] for m in val_metrics]
            l = [m["learned"][key] for m in val_metrics]
            ax[j].bar(xb - 0.2, b, 0.4, label="analytic baseline")
            ax[j].bar(xb + 0.2, l, 0.4, label="learned")
            if ref is not None:
                ax[j].axhline(ref, ls="--", c="k", lw=0.8)
            ax[j].set_title(title); ax[j].set_xticks(xb); ax[j].set_xticklabels(names)
            ax[j].legend()
        fig.tight_layout(); fig.savefig(RESULTS / "validation.png", dpi=110); plt.close(fig)
    print(f"  plots -> {RESULTS}/training.png, {RESULTS}/validation.png")


if __name__ == "__main__":
    main()
