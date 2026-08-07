r"""experiments/z_authority.py — does Sigma_C have authority over the z drift, and
does any training loss point at it?

Two questions, both about gradients, both asked at arm D's checkpoint on arm D's
own training distribution.

**Alignment (rule c).**  Define the drift objective

    D(theta) = mean_b [ mean_k ( v_est,z - v_true,z ) ]^2          (world frame)

— the squared DC vertical velocity error over a segment, i.e. the sink rate,
squared.  Both `D` and the training loss `L` are things we want small, so
`cos(grad D, grad L) > 0` means descending `L` also descends the drift.  Report
`||grad D||`, `||grad L||` and the cosine for every objective, plus the beta-NLL
cosine as a single alignment datum (no training, no formulation work).

**Authority ceiling (rule d).**  Optimise the network **directly against D**,
ignoring every training loss.  If drift barely moves when drift *is* the
objective, then no objective over Sigma_C — beta-NLL included — can do better,
and the lever is somewhere else entirely.  The optimised params are written out
so the claim can be checked end-to-end in the closed loop by `z_budget.py`
rather than resting on the segment-local proxy.

Usage
-----
    JAX_PLATFORMS=cuda uv run python experiments/z_authority.py --mode align
    JAX_PLATFORMS=cuda uv run python experiments/z_authority.py --mode ceiling \
        --steps 400 --out results/zauth_ceiling.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import run_policy as rp                                              # noqa: E402
rp.DT = 0.001
rp.DECIMATION = 20      # matches scripts/run_contactnet.py — the 1 kHz training regime

import numpy as np                                                   # noqa: E402
import jax                                                           # noqa: E402
import jax.numpy as jnp                                              # noqa: E402
import optax                                                         # noqa: E402

import invariant_estimation                     # noqa: F401,E402  (enables x64)
from invariant_estimation.sim import collect                         # noqa: E402
from invariant_estimation.contactnet import (                        # noqa: E402
    dataset, network, normalize, rollout as cn_rollout, train as cn_train)
from invariant_estimation.contactnet.config import ContactNetConfig  # noqa: E402
from invariant_estimation.inEKF.filter import init_carry, make_step  # noqa: E402

ARM_D = REPO / "results/2026-08-06_06-42-29_D_l2velposori"


# ---------------------------------------------------------------------------
# The drift objective, and a z-only velocity loss for the axis-share question
# ---------------------------------------------------------------------------

def make_combined_loss(ekf, kinematics, eps, w_pos, w_ori, lam,
                       use_pos=True, use_ori=False):
    r"""``l2_vel_pos[_ori] + lam * D`` — one scan, both terms.

    The remedy implied by the diagnosis: the existing objective is not blind to the
    drift *direction* (its gradient is aligned) but to the drift *time scale*,
    because ``L = 128`` ticks and the position term is segment-relative. Adding the
    segment DC of the vertical velocity error puts the accumulated mode back in
    view without lengthening the horizon.  Built here rather than in
    ``contactnet/losses.py`` so the running sweeps keep importing an unmodified
    ``src/``.
    """
    from invariant_estimation.contactnet.losses import pose_l2

    step = make_step(ekf, kinematics)

    def segment_loss(params, segment, carry0=None):
        L_c = cn_rollout.contact_factors(params, segment.windows, eps)
        inputs = segment.inputs._replace(contact_chol=L_c)
        c0 = init_carry(segment.state0) if carry0 is None else carry0
        carry, out = jax.lax.scan(step, c0, inputs)
        base = pose_l2(out.state.v, out.state.R, out.state.p,
                       segment.v_true, segment.R_true, segment.p_true,
                       w_pos=w_pos, w_ori=w_ori, use_pos=use_pos, use_ori=use_ori)
        drift = jnp.mean(out.state.v[..., 2] - segment.v_true[..., 2]) ** 2
        return base + lam * drift, (out, carry)

    def batch_loss(params, batch, carry0=None):
        axes = (None, 0) if carry0 is None else (None, 0, 0)
        args = (params, batch) if carry0 is None else (params, batch, carry0)
        losses, aux = jax.vmap(segment_loss, in_axes=axes)(*args)
        return jnp.mean(losses), aux

    return batch_loss


def make_drift_loss(ekf, kinematics, eps, kind="dc"):
    r"""``(params, segment, carry0) -> (D, (outputs, carry))``.

    ``kind="dc"``   — squared DC vertical velocity error, the sink rate squared.
                      This is the quantity the closed-loop drift integrates.
    ``kind="zmse"`` — per-tick world-z velocity MSE.  Used only to measure what
                      share of a training loss's gradient the z axis commands;
                      it is NOT the drift (a zero-mean z error integrates to
                      nothing, which is exactly the distinction that matters).
    """
    step = make_step(ekf, kinematics)

    def segment_loss(params, segment, carry0=None):
        L_c = cn_rollout.contact_factors(params, segment.windows, eps)
        inputs = segment.inputs._replace(contact_chol=L_c)
        c0 = init_carry(segment.state0) if carry0 is None else carry0
        carry, outputs = jax.lax.scan(step, c0, inputs)
        err_z = outputs.state.v[..., 2] - segment.v_true[..., 2]     # (L,) world
        loss = jnp.mean(err_z) ** 2 if kind == "dc" else jnp.mean(err_z ** 2)
        return loss, (outputs, carry)

    def batch_loss(params, batch, carry0=None):
        axes = (None, 0) if carry0 is None else (None, 0, 0)
        args = (params, batch) if carry0 is None else (params, batch, carry0)
        losses, aux = jax.vmap(segment_loss, in_axes=axes)(*args)
        return jnp.mean(losses), aux

    return batch_loss


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def _flat(tree) -> np.ndarray:
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])


def cosine(g1, g2) -> float:
    a, b = _flat(g1), _flat(g2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return float("nan")
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# Setup — rebuild arm D's exact training distribution
# ---------------------------------------------------------------------------

def build(pool="n8fix", contacts_per_foot=4, arm=ARM_D, batches=8, verbose=True):
    """Rebuild the collector, pool split, normalisation and batcher of a finished run.

    Deliberately reuses `dataset.prepare` / `ChainedBatcher` / arm D's saved
    `norm_constants.npz` and `P0.npy` rather than re-deriving them, so the
    gradients are measured on the distribution the checkpoint was actually
    trained on.
    """
    cfg = ContactNetConfig()
    c = collect.build_collector(policy_name="baseline", chunk_ticks=10_000,
                                contacts_per_foot=contacts_per_foot)

    paths = sorted(collect.DATA_DIR.glob(f"*_{pool}_seed*.npz"))
    if not paths:
        raise SystemExit(f"no rollouts matching *_{pool}_seed*.npz")
    by_terrain: dict[str, list] = {}
    for p in paths:
        by_terrain.setdefault(p.name.split(f"_{pool}_")[0], []).append(p)
    val_paths = [v[-1] for v in by_terrain.values() if v]
    train_paths = [p for p in paths if p not in set(val_paths)]
    if verbose:
        print(f"pool '{pool}': {len(paths)} rollouts, "
              f"{len(train_paths)} train / {len(val_paths)} val")

    dataset.build_channel_cache(train_paths, c, verbose=verbose)
    # Arm D's OWN normalisation, read the same way `run_estimator` reads it: the
    # saved artifact carries only {mean, std, names, floored}, so build the
    # NormConstants directly rather than through `normalize.load`, which also wants
    # provenance fields nothing downstream reads.
    z = np.load(arm / "norm_constants.npz", allow_pickle=False)
    norm = normalize.NormConstants(
        mean=jnp.asarray(z["mean"], dtype=jnp.float64),
        std=jnp.asarray(z["std"], dtype=jnp.float64),
        names=tuple(str(s) for s in z["names"]),
        floored=tuple(str(s) for s in z["floored"]),
        n_ticks=int(z["n_ticks"]) if "n_ticks" in z.files else 0,
        source=str(arm / "norm_constants.npz"))
    preps = dataset.prepare(train_paths, norm, cfg, verbose=False)
    P0 = np.load(arm / "P0.npy")

    warm_in = cn_rollout.make_warm_in(c.fused.ekf, c.fused.kinematics)
    batcher = dataset.ChainedBatcher(preps, cfg, P0, warm_in, seed=cfg.batcher_seed)

    like = network.init(jax.random.PRNGKey(cfg.init_seed), cfg.d_in, cfg.widths,
                        cfg.sigma_0, cfg.eps)
    params = cn_train.load_params(str(arm / "params.npz"), like)

    summary = json.loads((arm / "summary.json").read_text())
    return c, cfg, batcher, params, like, summary


def _pose_weights(summary, cfg):
    """Arm D's frozen pose weights, measured once at its own run start."""
    for key in ("w_pos", "w_ori"):
        if key not in summary:
            break
    else:
        return float(summary["w_pos"]), float(summary["w_ori"])
    return 4.7033, 9.8465      # recorded in results.md for arms B/D


# ---------------------------------------------------------------------------
# Mode: alignment
# ---------------------------------------------------------------------------

def run_align(args):
    c, cfg, batcher, params, _like, summary = build(
        pool=args.pool, contacts_per_foot=args.contacts_per_foot)
    w_pos, w_ori = _pose_weights(summary, cfg)
    print(f"pose weights: w_pos={w_pos:.4f} w_ori={w_ori:.4f}")

    ekf, kin = c.fused.ekf, c.fused.kinematics
    drift = jax.jit(jax.value_and_grad(
        make_drift_loss(ekf, kin, cfg.eps, "dc"), has_aux=True))
    zmse = jax.jit(jax.value_and_grad(
        make_drift_loss(ekf, kin, cfg.eps, "zmse"), has_aux=True))

    objectives = ["l2_velocity", "l2_vel_pos", "l2_vel_pos_ori", "beta_nll"]
    graders = {}
    for obj in objectives:
        bl = cn_rollout.make_batch_loss(ekf, kin, cfg.eps, beta=cfg.beta,
                                        objective=obj, remat=cfg.remat,
                                        w_pos=w_pos, w_ori=w_ori)
        graders[obj] = jax.jit(jax.value_and_grad(bl, has_aux=True))

    acc: dict[str, list] = {o: [] for o in objectives}
    dn, zshare, dval = [], [], []
    for i in range(args.batches):
        batch, carry0 = batcher.batch()
        (dv, (_o, carry)), gd = drift(params, batch, carry0)
        (_zv, _), gz = zmse(params, batch, carry0)
        dn.append(float(np.linalg.norm(_flat(gd))))
        dval.append(float(dv))
        for obj in objectives:
            (lv, _), gl = graders[obj](params, batch, carry0)
            acc[obj].append((float(lv), float(np.linalg.norm(_flat(gl))),
                             cosine(gd, gl)))
        zshare.append(float(np.linalg.norm(_flat(gz))))
        batcher.update(carry)
        print(f"  batch {i}: D={dv:.3e} |gD|={dn[-1]:.3e}")

    print("\n=== GRADIENT AUTHORITY (arm D checkpoint, arm D distribution) ===")
    print("  drift objective D = (DC world-z velocity error)^2")
    print(f"    mean D      {np.mean(dval):.4e}  (rms DC error "
          f"{np.sqrt(np.mean(dval)):.4e} m/s)")
    print(f"    ||grad D||  mean {np.mean(dn):.4e}   -- is the drift REACHABLE at all?")
    print(f"    ||grad L_zmse|| mean {np.mean(zshare):.4e}")
    print(f"\n  {'objective':16s} {'loss':>12s} {'||grad L||':>12s} "
          f"{'cos(gD,gL)':>12s} {'verdict':>10s}")
    out = {}
    for obj in objectives:
        lv = np.mean([a[0] for a in acc[obj]])
        gn = np.mean([a[1] for a in acc[obj]])
        cs = np.array([a[2] for a in acc[obj]])
        verdict = ("ALIGNED" if abs(np.mean(cs)) > 0.3 else
                   "BLIND" if abs(np.mean(cs)) < 0.1 else "weak")
        print(f"  {obj:16s} {lv:12.4e} {gn:12.4e} "
              f"{np.mean(cs):>+12.4f} {verdict:>10s}   (per-batch "
              f"{np.array2string(cs, precision=2, floatmode='fixed')})")
        out[obj] = dict(loss=lv, grad_norm=gn, cos_mean=float(np.mean(cs)),
                        cos=cs.tolist())
    out["drift"] = dict(D=float(np.mean(dval)), grad_norm=float(np.mean(dn)),
                        zmse_grad_norm=float(np.mean(zshare)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args.out}")
    return out


# ---------------------------------------------------------------------------
# Mode: authority ceiling
# ---------------------------------------------------------------------------

def run_ceiling(args):
    """Optimise the network against the drift (alone, or added to the pose loss).

    ``--mode ceiling`` is the pure upper bound; ``--mode combined`` is the candidate
    remedy, trained from the SAME start so the two are comparable.
    """
    c, cfg, batcher, params, like, summary = build(
        pool=args.pool, contacts_per_foot=args.contacts_per_foot)
    ekf, kin = c.fused.ekf, c.fused.kinematics
    if args.mode == "combined":
        w_pos, w_ori = _pose_weights(summary, cfg)
        print(f"combined objective: l2_vel_pos + {args.lam} * D "
              f"(w_pos={w_pos:.4f})")
        batch_loss = make_combined_loss(ekf, kin, cfg.eps, w_pos, w_ori, args.lam)
    else:
        batch_loss = make_drift_loss(ekf, kin, cfg.eps, "dc")

    tx = cn_train.make_optimizer(cfg.peak_lr, args.steps, cfg.warmup_steps,
                                 cfg.max_norm, cfg.weight_decay)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, batch, carry0):
        (loss, (_out, carry)), grads = jax.value_and_grad(
            batch_loss, has_aux=True)(params, batch, carry0)
        gnorm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, gnorm, carry

    print(f"\n=== AUTHORITY CEILING: {args.steps} steps of AdamW on the DRIFT itself ===")
    t0, hist = time.time(), []
    for i in range(args.steps):
        batch, carry0 = batcher.batch()
        params, opt_state, loss, gnorm, carry = step(params, opt_state, batch, carry0)
        batcher.update(carry)
        hist.append((float(loss), float(gnorm)))
        if i % 25 == 0:
            print(f"  step {i:4d}  D {float(loss):.4e}  "
                  f"(DC {np.sqrt(max(float(loss), 0)):.4e} m/s)  "
                  f"|g| {float(gnorm):.3e}  {time.time() - t0:6.0f}s")

    h = np.array(hist)
    n = max(1, len(h) // 10)
    label = "D" if args.mode == "ceiling" else "loss (base + lam*D)"
    print(f"\n  {label}: first-decile mean {h[:n, 0].mean():.4e} "
          f"-> last-decile mean {h[-n:, 0].mean():.4e} "
          f"({h[:n, 0].mean() / max(h[-n:, 0].mean(), 1e-300):.2f}x)")
    if args.mode == "ceiling":
        # sqrt(loss) is the DC velocity error only when the loss IS D; under
        # --mode combined it is base + lam*D and the square root means nothing.
        print(f"  DC velocity error: {np.sqrt(h[:n, 0].mean()):.4e} "
              f"-> {np.sqrt(h[-n:, 0].mean()):.4e} m/s")
    print("  NOTE: this is the segment-local proxy. The end-to-end check is a "
          "closed-loop z_budget run with the params written below.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cn_train.save_params(args.out, params)
        np.save(str(Path(args.out).with_suffix(".hist.npy")), h)
        print(f"  wrote {args.out}")
    return params


def run_sigma(args):
    r"""What does the network output ON ITS TRAINING DISTRIBUTION?

    The closed loop shows ContactNet running $\sim$20x looser in stance than the
    analytic heuristic. That is only a *learned preference* if it also holds on the
    pool the network was fitted to; if it appears only at deployment, it is
    distribution shift and belongs to a different diagnosis. Compares the network's
    `Sigma_C` against the recorded analytic `contact_chol` tick for tick, split by
    the recorded stance/swing decision — no filter scan needed.
    """
    c, cfg, batcher, params, _like, _summary = build(
        pool=args.pool, contacts_per_foot=args.contacts_per_foot)
    tr_net, tr_ana, stance = [], [], []
    for _ in range(args.batches):
        batch, carry0 = batcher.batch()
        # `contact_factors` expects one segment `(L, N_c, H, F)`; a batch carries a
        # leading axis, so vmap over it rather than letting the reshape silently
        # fold `N_c` into the feature dimension.
        L_c = jax.vmap(cn_rollout.contact_factors, in_axes=(None, 0, None))(
            params, batch.windows, cfg.eps)
        sig = jnp.einsum("...ij,...kj->...ik", L_c, L_c)
        rec = batch.inputs.contact_chol
        rec_sig = jnp.einsum("...ij,...kj->...ik", rec, rec)
        tr_net.append(np.asarray(jnp.trace(sig, axis1=-2, axis2=-1)).ravel())
        tr_ana.append(np.asarray(jnp.trace(rec_sig, axis1=-2, axis2=-1)).ravel())
        # the recorded heuristic is the stance/swing label: stance == the small one
        stance.append(tr_ana[-1] < 1.0)
        batcher.update(carry0)

    net = np.concatenate(tr_net); ana = np.concatenate(tr_ana)
    st = np.concatenate(stance)
    print("\n=== Sigma_C ON THE TRAINING DISTRIBUTION (no floor applied) ===")
    for lbl, m in (("stance", st), ("swing", ~st)):
        if m.sum() == 0:
            continue
        print(f"  {lbl:7s} n={m.sum():7d}  analytic median {np.median(ana[m]):.3e}"
              f"   network median {np.median(net[m]):.3e}"
              f"   ratio {np.median(net[m]) / max(np.median(ana[m]), 1e-300):8.1f}x")
    print("  (closed-loop measurement for comparison: stance 6.02e-3 vs analytic "
          "3.00e-4 after the 1e-4 additive floor, i.e. ~20x)")
    return {"net": net, "ana": ana, "stance": st}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("align", "ceiling", "combined", "sigma"),
                    default="align")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on the drift term in --mode combined")
    ap.add_argument("--pool", default="n8fix")
    ap.add_argument("--contacts-per-foot", type=int, default=4)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    {"align": run_align, "sigma": run_sigma}.get(args.mode, run_ceiling)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
