#!/usr/bin/env python
"""ContactNet overnight orchestrator: collect -> cache -> normalize -> train -> validate.

Coherent 1 kHz regime (rp.DT=0.001, rp.DECIMATION=20; CONTROL_DT=0.02 unchanged),
process socket only, F=30 (q-dot channel), H consecutive ticks of history (no
boxcar, no decimation). Held-out validation:
body-frame velocity RMSE + world-frame velocity NEES (3-DoF and per world axis) +
contact NIS/dof, learned Sigma_C vs the
analytic-heuristic contact_chol baseline (the recorded stance/swing factors).

Usage:
    uv run python scripts/run_contactnet.py --collect --steps 300 --seconds 45

Artifacts go to results/<YYYY-MM-DD_HH-MM-SS>[_tag]/ (one directory per run, with
results/latest symlinked at the newest); --tag names a run, --out-dir overrides.
"""
import argparse
import json
import sys
import time
from datetime import datetime
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
from invariant_estimation.contactnet.losses import (
    l2_velocity, l2_position, so3_log_orientation)
from invariant_estimation.inEKF.filter import init_carry, make_step

RESULTS_ROOT = REPO / "results"

# validation modes for later
VAL_MODES = { # val seed -> (label, constant (vx, vy, yaw))
    900: ("forward", (0.45, 0.00, 0.00)),
    901: ("backward", (-0.45, 0.00, 0.00)),
    902: ("lateral_L", (0.00, 0.40, 0.00)),
    903: ("lateral_R", (0.00, -0.40, 0.00)),
    904: ("turn_L", (0.00, 0.00, 0.75)),
    905: ("turn_R", (0.00, 0.00, -0.75)),
}

def make_run_dir(root, tag=None, explicit=None):
    """results/<YYYY-MM-DD_HH-MM-SS>[_tag]/ — one directory per run, sorted by date.

    `explicit` (--out-dir) overrides the naming entirely. A `results/latest`
    symlink is repointed at the new directory so downstream tooling has a stable
    path to the most recent run.
    """
    if explicit is not None:
        run_dir = Path(explicit)
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = root / (f"{stamp}_{tag}" if tag else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    link = root / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.resolve().relative_to(root.resolve()), target_is_directory=True)
    except (OSError, ValueError):
        pass  # non-POSIX fs, or --out-dir outside results/: the run dir still stands
    return run_dir


def collect_rollouts(c, seeds, seconds, cfg, cmd_override_map=None):
    paths = []
    for s in seeds:
        p = collect.DATA_DIR / f"flat_seed{s:03d}.npz"
        if p.exists():
            print(f"  seed{s}: exists, skipping collection")
            paths.append(p)
            continue
        try:
            collect.collect_rollout(seed=s, seconds=seconds, collector=c, out_dir=collect.DATA_DIR, cfg=cfg, cmd_override=(cmd_override_map or {}).get(s))
            paths.append(p)
        except RuntimeError as e:
            print(f"  seed{s}: FELL/skipped: {e}")
    return paths


def save_norm(norm, path):
    np.savez(path, mean=np.asarray(norm.mean), std=np.asarray(norm.std),
             names=np.asarray(norm.names), floored=np.asarray(norm.floored))


def _windows_full(cache, norm, cfg):
    """(T, N_c, H, F) normalized feature windows over the whole stream.

    Forced onto the CPU device. This materialises T*N_c*H*F float64 -- 0.6 GB at
    T=62k, N=2, and 2.4 GB at N=8 -- plus a transpose copy, and doing that on the
    accelerator OOM'd a 12 GB card mid-validation after training had already
    finished, losing the run. It is a one-shot gather with no math in it, so the
    GPU buys nothing here; the result is handed back as NumPy and only the sliced
    usable region is put back on device by the caller.
    """
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        x = normalize.apply(jnp.asarray(cache["channels"], dtype=jnp.float64), norm)
        return np.asarray(features.window(x, cfg.H))


def validate(prep, cache, norm, cfg, params, fused, P0, eps):
    """Run the filter over the held-out usable region with (a) recorded analytic
    contact_chol and (b) learned Sigma_C. Return body-frame velocity RMSE, world-frame
    velocity NEES (3-DoF and per world axis x/y/z), and contact NIS/dof for each."""
    from invariant_estimation.inEKF.group import skew
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

        # World-frame velocity error and its covariance.
        #
        # P is the right-invariant TANGENT covariance (I4: rotation 0:3, velocity
        # 3:6). With X_hat = exp(xi) X (I5) the velocity column of the group element
        # gives  v_hat = (I + (xi_phi)_x) v + xi_v, i.e. the world-frame error is
        #     dv = v_hat - v = xi_v - (v)_x xi_phi,
        # so  Sigma_dv = J P[0:6,0:6] J^T   with   J = [ -(v)_x   I_3 ].
        # Taking P[3:6,3:6] alone (the previous 3-DoF metric) drops the attitude
        # coupling and under-reports the covariance whenever the robot is moving,
        # which inflates NEES. Both are reported: `vel_nees_tangent` is the old
        # number, `vel_nees_world*` are the world-frame ones the axes decompose.
        ev = out.state.v - v_true
        Pvv = out.state.P[:, 3:6, 3:6]
        nees_tangent = float(jnp.mean(
            jax.vmap(lambda e, P: e @ jnp.linalg.solve(P, e))(ev, Pvv)))

        eye = jnp.broadcast_to(jnp.eye(3), (out.state.v.shape[0], 3, 3))
        J = jnp.concatenate([-jax.vmap(skew)(out.state.v), eye], axis=-1)   # (T,3,6)
        Sv = J @ out.state.P[:, 0:6, 0:6] @ jnp.swapaxes(J, -1, -2)         # (T,3,3)
        nees_world = float(jnp.mean(
            jax.vmap(lambda e, S: e @ jnp.linalg.solve(S, e))(ev, Sv)))
        # per world axis: 1-DoF marginal NEES, target 1 each
        nees_axis = jnp.mean(ev ** 2 / jnp.diagonal(Sv, axis1=-2, axis2=-1), axis=0)

        nis = float(jnp.mean(out.contact_diagnostics.nis)) / cfg.dof
        applied = float(jnp.mean(out.contact_diagnostics.applied))
        return dict(vel_rmse=rmse, vel_nees=nees_world, vel_nees_tangent=nees_tangent,
                    vel_nees_x=float(nees_axis[0]), vel_nees_y=float(nees_axis[1]),
                    vel_nees_z=float(nees_axis[2]),
                    nis_over_dof=nis, applied=applied)

    baseline = run(inputs.contact_chol)
    L_c = cn_rollout.contact_factors(params, wins, eps)
    learned = run(L_c)
    return baseline, learned


def check_pool_contact_meas_var(paths, train_value):
    """Warn loudly when the pool was COLLECTED at a different R floor than we train at.

    `contact_meas_var` is recorded per rollout (`collect.collect_rollout`'s meta). It
    is a filter parameter, not a data parameter, so a mismatch is legitimate: the
    fused estimator is rebuilt at train time and `validate()` runs the analytic
    baseline and the learned Sigma_C through that SAME rebuilt filter, so the two stay
    matched and the comparison stands. The existing `n8fix` pool was collected at 0.0
    and is trained at 1e-4 for exactly this reason.

    What it is NOT safe to leave silent: the recorded `inputs.contact_chol` heuristic
    -- the analytic baseline -- was produced under the pool's value, so quoting a
    number from this run against a number from a differently-floored run is only valid
    for the LEARNED arm. Print it rather than discover it in a results table.
    """
    seen = {}
    for p in paths:
        try:
            with np.load(p) as z:
                seen[float(json.loads(str(z["meta"]))["contact_meas_var"])] = p.name
        except (KeyError, ValueError, OSError):
            continue          # pre-dates the meta field; nothing to check against
    stale = {v: n for v, n in seen.items() if v != float(train_value)}
    if stale:
        print(f"  NOTE: pool collected at contact_meas_var={sorted(stale)} "
              f"(e.g. {list(stale.values())[0]}), training/validating at "
              f"{train_value:g}. Expected -- the filter is rebuilt here and the "
              f"analytic baseline is re-run through it, so baseline and learned stay "
              f"matched. Do not compare the BASELINE column across differing floors.")


_POSE_USE = {"l2_vel_pos": (True, False),
             "l2_vel_ori": (False, True),
             "l2_vel_pos_ori": (True, True)}


def resolve_pose_weights(cfg, params, fused, train_preps, P0, warm_in):
    """Return the frozen ``(w_pos, w_ori)`` for the composite pose objectives.

    For an active term whose config weight is unset (``None``), measure
    ``L_vel / L_pos / L_ori`` on ONE warm batch with the init network and size the
    weight so that term starts at ``pose_weight_ratio x L_vel``; then it is frozen
    for the whole run (measure-once, not per-step adaptive). Explicit
    ``--w-pos/--w-ori`` pass straight through. Non-pose objectives return ``(0, 0)``.
    """
    use_pos, use_ori = _POSE_USE.get(cfg.objective, (False, False))
    if not (use_pos or use_ori):
        return 0.0, 0.0

    need_measure = (use_pos and cfg.w_pos is None) or (use_ori and cfg.w_ori is None)
    L_vel = L_pos = L_ori = None
    if need_measure:
        # A throwaway batcher on the same seed: its first (warm-in) batch is the
        # same one training will see, and the real batcher stays untouched.
        meas = dataset.ChainedBatcher(train_preps, cfg, P0, warm_in, seed=cfg.batcher_seed)
        mb, mc = meas.batch()
        measure_loss = cn_rollout.make_batch_loss(
            fused.ekf, fused.kinematics, cfg.eps, beta=cfg.beta,
            objective="l2_velocity", remat=cfg.remat)
        _, (mout, _c) = measure_loss(params, mb, mc)
        L_vel = float(l2_velocity(mout.state.v, mout.state.R, mb.v_true, mb.R_true))
        L_pos = float(l2_position(mout.state.p, mb.p_true))
        L_ori = float(so3_log_orientation(mout.state.R, mb.R_true))
        print(f"  pose-weight measurement (warm batch, init net): "
              f"L_vel={L_vel:.3e} L_pos={L_pos:.3e} L_ori={L_ori:.3e}")

    r = cfg.pose_weight_ratio
    w_pos = (cfg.w_pos if cfg.w_pos is not None
             else r * L_vel / max(L_pos, 1e-12)) if use_pos else 0.0
    w_ori = (cfg.w_ori if cfg.w_ori is not None
             else r * L_vel / max(L_ori, 1e-12)) if use_ori else 0.0
    print(f"  pose weights (frozen): w_pos={w_pos:.4e} w_ori={w_ori:.4e} "
          f"(ratio={r}, objective={cfg.objective})")
    return float(w_pos), float(w_ori)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--val-seeds", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--time-budget-s", type=float, default=3600.0)
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix appended to the timestamped run directory name")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="write artifacts here instead of results/<timestamp>/")
    ap.add_argument("--contacts-per-foot", type=int, default=1,
                    help="1 = the shipped N=2 soles, 4 = the N=8 corner set")
    ap.add_argument("--objective", type=str, default=None,
                    choices=["l2_velocity", "beta_nll",
                             "l2_vel_pos", "l2_vel_ori", "l2_vel_pos_ori"],
                    help="override ContactNetConfig.objective (the run-ladder knob)")
    ap.add_argument("--w-pos", type=float, default=None,
                    help="explicit position-term weight (overrides the auto-measure)")
    ap.add_argument("--w-ori", type=float, default=None,
                    help="explicit orientation-term weight (overrides the auto-measure)")
    ap.add_argument("--pose-weight-ratio", type=float, default=None,
                    help="target: each active pose term starts at ratio x L_vel "
                         "(default 0.5); used only when --w-pos/--w-ori are unset")
    ap.add_argument("--L", type=int, default=None,
                    help="BPTT segment length in ticks (ContactNetConfig.L, default "
                         "128). The L-ablation knob; note L is also a MEMORY axis -- "
                         "activation memory over the filter scan is O(L), so raising "
                         "it without --remat is what OOMs a 12 GB card.")
    ap.add_argument("--remat", action=argparse.BooleanOptionalAction, default=None,
                    help="rematerialize the BPTT scan body (jax.checkpoint). Trades "
                         "~one extra forward pass for O(L)-fold activation memory; "
                         "mathematically identity (tests/contactnet/test_remat.py).")
    ap.add_argument("--contact-meas-var", type=float, default=1.0e-4,
                    help="InEKF contact-measurement noise floor [m^2] "
                         "(main_estimator 'landmine #2', port default 0.0). 1e-4 is "
                         "the tuned value from the 2026-08-07 z-drift study -- the "
                         "one config lever ContactNet responds to (+31.1%%). Do NOT "
                         "raise to 1e-2: that is a flat-ground cancellation that "
                         "drifts UPWARD on both terrains.")
    ap.add_argument("--pool", type=str, default=None,
                    help="rollout pool tag, e.g. 'n8'. Selects data/*_<tag>_seed*.npz "
                         "and holds out one rollout PER TERRAIN for validation. "
                         "Omit for the flat N=2 pool addressed by --train-seeds.")
    args = ap.parse_args()

    overrides = {}
    if args.objective is not None:
        overrides["objective"] = args.objective
    if args.w_pos is not None:
        overrides["w_pos"] = args.w_pos
    if args.w_ori is not None:
        overrides["w_ori"] = args.w_ori
    if args.pose_weight_ratio is not None:
        overrides["pose_weight_ratio"] = args.pose_weight_ratio
    if args.L is not None:
        overrides["L"] = args.L
    if args.remat is not None:
        overrides["remat"] = args.remat
    cfg = ContactNetConfig(**overrides)

    t_start = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    out = make_run_dir(RESULTS_ROOT, tag=args.tag, explicit=args.out_dir)
    print(f"== run directory: {out} ==")
    c = collect.build_collector(policy_name="baseline", chunk_ticks=10_000,
                                contacts_per_foot=args.contacts_per_foot,
                                contact_meas_var=args.contact_meas_var)

    if args.collect:
        print("== collecting ==")
        val_cmd = {s : cmd for s, (_label, cmd) in VAL_MODES.items()}
        collect_rollouts(c, args.train_seeds, args.seconds, cfg)
        collect_rollouts(c, args.val_seeds, args.seconds, cfg, val_cmd)

    if args.pool:
        # Hold out one rollout PER TERRAIN, not a random slice: Gate G evaluates on
        # terrain specifically, and a pooled split can leave a terrain unrepresented
        # in val, which is exactly the averaging that hides the terrain effect.
        pool = sorted(collect.DATA_DIR.glob(f"*_{args.pool}_seed*.npz"))
        if not pool:
            raise SystemExit(f"no rollouts matching *_{args.pool}_seed*.npz in {collect.DATA_DIR}")
        by_terrain = {}
        for p in pool:
            by_terrain.setdefault(p.name.split(f"_{args.pool}_")[0], []).append(p)
        val_paths = [v[-1] for v in by_terrain.values() if v]
        train_paths = [p for p in pool if p not in set(val_paths)]
        print(f"pool '{args.pool}': {len(pool)} rollouts over "
              f"{ {k: len(v) for k, v in by_terrain.items()} }")
    else:
        train_paths = [collect.DATA_DIR / f"flat_seed{s:03d}.npz" for s in args.train_seeds]
        val_paths = [collect.DATA_DIR / f"flat_seed{s:03d}.npz" for s in args.val_seeds]
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [p for p in val_paths if p.exists()]
    print(f"train rollouts: {[p.name for p in train_paths]}")
    print(f"val rollouts:   {[p.name for p in val_paths]}")
    check_pool_contact_meas_var(train_paths + val_paths, args.contact_meas_var)

    print("== building channel caches ==")
    dataset.build_channel_cache(train_paths + val_paths, c)

    print("== fitting normalization (train, walking set) ==")
    norm = dataset.fit_normalization(train_paths, source="overnight train set")
    print(f"  floored channels: {list(norm.floored)}")
    save_norm(norm, out / "norm_constants.npz")

    print(f"  cfg: F={cfg.F} d_in={cfg.d_in} H={cfg.H} "
          f"(window {cfg.window_span_seconds * 1e3:.0f} ms, consecutive ticks) "
          f"L={cfg.L} B={cfg.B} objective={cfg.objective} episode_s={cfg.episode_s}")

    print("== preparing rollouts ==")
    train_preps = dataset.prepare(train_paths, norm, cfg, verbose=True)
    val_preps = dataset.prepare(val_paths, norm, cfg, verbose=True)

    print("== measuring P0 ==")
    P0 = dataset.measure_p0(c.fused, train_preps[0], cfg, ticks=3000)
    np.save(out / "P0.npy", P0)

    print("== building network + batcher ==")
    params = network.init(jax.random.PRNGKey(cfg.init_seed), cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    warm_in = cn_rollout.make_warm_in(c.fused.ekf, c.fused.kinematics)
    batcher = dataset.ChainedBatcher(train_preps, cfg, P0, warm_in, seed=cfg.batcher_seed)

    # Size + freeze the pose-loss weights for the composite objectives (no-op for
    # l2_velocity / beta_nll). Measured once here so the graph carries constants.
    w_pos, w_ori = resolve_pose_weights(cfg, params, c.fused, train_preps, P0, warm_in)

    batch_loss = cn_rollout.make_batch_loss(
        c.fused.ekf, c.fused.kinematics, cfg.eps, beta=cfg.beta,
        objective=cfg.objective, remat=cfg.remat, w_pos=w_pos, w_ori=w_ori)

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

    cn_train.save_params(str(out / "params.npz"), params)
    hist = np.array(history)
    np.save(out / "history.npy", hist)

    print("== validating on held-out ==")
    val_metrics = []
    for vp, vpath in zip(val_preps, val_paths):
        cache = dataset.load_channel_cache(dataset.cache_path(vpath))
        base, learned = validate(vp, cache, norm, cfg, params, c.fused, P0, cfg.eps)
        seed = int(vp.name.split("seed")[1].split(".")[0])
        if args.pool:
            # Under --pool the held-out set is one rollout PER TERRAIN, so the
            # terrain IS the label -- that is what makes the per-terrain table in
            # results.md possible instead of a single pooled average.
            label = vp.name.split(f"_{args.pool}_")[0]
        else:
            label = VAL_MODES.get(seed, ("mixed", None))[0]
        print(f"  {vp.name}: baseline={base}  learned={learned}")
        val_metrics.append({"mode": label, "rollout": vp.name, "baseline": base, "learned": learned})

    summary = {
        "run": {"dir": out.name, "started_at": started_at, "tag": args.tag,
                "args": vars(args)},
        "n_train": len(train_preps), "n_val": len(val_preps),
        "steps_run": len(history), "final_reseeds": reseeds,
        "floored": list(norm.floored),
        "cfg": {"F": cfg.F, "d_in": cfg.d_in, "H": cfg.H,
                "window_span_s": cfg.window_span_seconds,
                "L": cfg.L, "B": cfg.B, "objective": cfg.objective,
                # remat and contact_meas_var are recorded because BOTH were
                # silently wrong before: remat was accepted and dropped on the
                # floor (dead flag, four arms), and contact_meas_var defaults to
                # 0.0 in build_collector with no record in the run summary. A run
                # that cannot state its own filter configuration is not evidence.
                "remat": cfg.remat,
                "contact_meas_var": float(args.contact_meas_var),
                "w_pos": w_pos, "w_ori": w_ori,
                "pose_weight_ratio": cfg.pose_weight_ratio,
                "episode_s": cfg.episode_s, "warm_in_s": cfg.warm_in_s,
                "peak_lr": cfg.peak_lr,
                "init_seed": cfg.init_seed, "batcher_seed": cfg.batcher_seed,
                "cmd_ranges": {
                    "vx": cfg.cmd_vx_range,
                    "vy": cfg.cmd_vy_range,
                    "yaw": cfg.cmd_yaw_range
                },
        },
        "wall_s": time.time() - t_start,
        "val": val_metrics,
        "loss_first": history[0][0] if history else None,
        "loss_last": history[-1][0] if history else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("== summary ==")
    print(json.dumps(summary, indent=2))
    make_plots(hist, val_metrics, out, cfg.objective)


def make_plots(hist, val_metrics, out, objective="l2_velocity"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hist.size:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(hist[:, 0])
        # The objective is NOT always l2_velocity, and beta_nll's loss is NEGATIVE
        # (0.5*(NIS + logdet S) is unbounded below in logdet). A hardcoded log scale
        # drops every point and renders an EMPTY panel. Log only when the curve is
        # all-positive (the composite pose objectives are sums of squares -> positive).
        ax[0].set_title(f"training loss ({objective})")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
        if np.all(hist[:, 0] > 0):
            ax[0].set_yscale("log")
        ax[1].plot(hist[:, 2]); ax[1].axhline(1.0, ls="--", c="k", lw=0.8)
        ax[1].set_title("contact NIS / dof"); ax[1].set_xlabel("step")
        ax[2].plot(hist[:, 4]); ax[2].set_title("cumulative reseeds"); ax[2].set_xlabel("step")
        fig.tight_layout(); fig.savefig(out / "training.png", dpi=110); plt.close(fig)

    if val_metrics:
        names = [m["rollout"] for m in val_metrics]
        xb = np.arange(len(names))
        panels = [
            ("vel_rmse", "held-out body-frame velocity RMSE [m/s]", None),
            ("vel_nees", "held-out world velocity NEES, 3-DoF (target 3)", 3.0),
            ("nis_over_dof", "held-out contact NIS/dof (target 1)", 1.0),
            ("vel_nees_x", "world velocity NEES, x (target 1)", 1.0),
            ("vel_nees_y", "world velocity NEES, y (target 1)", 1.0),
            ("vel_nees_z", "world velocity NEES, z (target 1)", 1.0),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        ax = axes.ravel()
        for j, (key, title, ref) in enumerate(panels):
            base = [m["baseline"][key] for m in val_metrics]
            learned = [m["learned"][key] for m in val_metrics]
            ax[j].bar(xb - 0.2, base, 0.4, label="analytic baseline")
            ax[j].bar(xb + 0.2, learned, 0.4, label="learned")
            if ref is not None:
                ax[j].axhline(ref, ls="--", c="k", lw=0.8)
            ax[j].set_title(title); ax[j].set_xticks(xb); ax[j].set_xticklabels(names)
            ax[j].legend()
        fig.tight_layout(); fig.savefig(out / "validation.png", dpi=110); plt.close(fig)
    print(f"  plots -> {out}/training.png, {out}/validation.png")


if __name__ == "__main__":
    main()
