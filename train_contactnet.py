r"""`train_contactnet.py` — collected rollouts → a trained ContactNet, in one command.

    uv run python train_contactnet.py cache          # pass 1: MJX features -> data/cache/
    uv run python train_contactnet.py norm           # pass 2: freeze data/norm_constants.npz
    uv run python train_contactnet.py p0 --p0 artifacts/p0_process.npz   # measure P0
    uv run python train_contactnet.py train --steps 200 --objective l2_velocity
    uv run python train_contactnet.py measure-b      # peak RSS vs B, remat on/off
    uv run python train_contactnet.py check-init     # the §4 init-parity properties

`cache` and `norm` are idempotent and are run automatically by `train` if their
artifacts are missing, so the single command is just the third line.  They are
separate subcommands because pass 1 is the only one that needs MJX and it costs
minutes; nothing downstream should ever pay it twice.

What this file owns is *wiring*, deliberately: the config, the estimator build,
`dataset` → `rollout.make_batch_loss` → `train.train`.  Every number it prints
comes from a module that is tested on its own.

Two things worth knowing before reading a run:

* Under ``beta_nll`` the loss is **not** a progress metric — watch
  ``nis_over_dof → 1`` (PORT_NOTES.md, "ContactNet training loop").  Under
  ``l2_velocity`` the loss *is* monotone and readable, which is why run 1 uses it.
* Steps 1 and 2 barely move: the warmup schedule starts at ``lr = 0`` and the
  head is zero-initialised, so the trunk gradient is exactly zero until the head
  is nonzero.  That is correct, and it reads as a broken loop if unexpected.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

# `jax_enable_x64` is flipped at package import and MUST precede any array construction (I8).
import invariant_estimation  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from invariant_estimation.contactnet import (
    dataset, features, network, normalize, train as train_mod)
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet.rollout import (contact_factors, make_batch_loss,
                                                     make_warm_in)
from invariant_estimation.sim import collect

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config / build
# ---------------------------------------------------------------------------

def make_config(args) -> ContactNetConfig:
    """`ContactNetConfig` from the CLI. ``F = 24`` is fixed by `features.channel_names`."""
    return ContactNetConfig(
        F=len(features.channel_names()),
        sigma_0=args.sigma_0,
        H=args.H,
        dt=args.dt,
        L=args.L,
        B=args.B,
        objective=args.objective,
        peak_lr=args.lr,
        warmup_steps=min(args.warmup_steps, max(1, args.steps - 1)),
        total_steps=args.steps,
        remat=not args.no_remat,
        freeze_contact_chol=getattr(args, "freeze_contact_chol", False),
        warm_in_s=getattr(args, "warm_in_s", 1.0),
        episode_s=getattr(args, "episode_s", 43.0),
    )


def build_estimator(args, verbose: bool = True):
    """The fused estimator, with the same options `sim.collect` collected under.

    `contact_fk_unfiltered=True` is a hard requirement (the ankles are two of the
    six subchain joints) and ``contact_meas_var = 0.0`` is deliberate: it is the
    hand-tuned isotropic stand-in ContactNet replaces.
    """
    return collect.build_collector(chunk_ticks=args.chunk, verbose=verbose)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_cache(args) -> list[Path]:
    paths = dataset.rollout_paths(args.data)
    if not paths:
        raise SystemExit(f"no rollouts in {args.data}; run "
                         f"`uv run python -m invariant_estimation.sim.collect` first")
    todo = [p for p in paths if args.force or not dataset.cache_path(p, args.cache).exists()]
    if not todo:
        print(f"cache: up to date ({len(paths)} rollouts)")
        return paths
    print(f"cache: {len(todo)}/{len(paths)} rollouts to build")
    c = build_estimator(args)
    dataset.build_channel_cache(todo, c, cache_dir=args.cache, chunk=args.fk_chunk)
    return paths


def stage_norm(args) -> normalize.NormConstants:
    paths = stage_cache(args)
    out = Path(args.norm)
    if out.exists() and not args.force:
        c = normalize.load(str(out))
        print(f"norm: loaded {out} ({c.n_ticks} samples, {len(c.names)} channels)")
    else:
        c = dataset.fit_normalization(paths, cache_dir=args.cache)
        out.parent.mkdir(parents=True, exist_ok=True)
        normalize.save(str(out), c)
        print(f"norm: fit over {c.n_ticks} samples -> {out}\n  source: {c.source}")
    print(f"  floored ({len(c.floored)}): {list(c.floored) or 'none'}")
    for n, m, s in zip(c.names, np.asarray(c.mean), np.asarray(c.std)):
        print(f"    {n:14s} mean {m: .6e}  std {s: .6e}"
              + ("   <- FLOORED" if n in c.floored else ""))
    return c


def stage_prepare(args, cfg):
    """Cache + norm + prepared rollouts + the measured `P0`. The whole data side."""
    norm = stage_norm(args)
    paths = dataset.rollout_paths(args.data)
    if args.n_rollouts:
        # `measure-b` only: each prepared rollout is ~210 MB resident, and the
        # point of that mode is to see the GRADIENT's footprint, not the loader's.
        paths = paths[:args.n_rollouts]
    preps = dataset.prepare(paths, norm, cfg, cache_dir=args.cache, verbose=True)
    total = sum(p.n_starts for p in preps)
    print(f"prepare: {len(preps)} rollouts, {total} legal segment starts "
          f"(L={cfg.L}, H={cfg.H}, stride={cfg.stride}, span={cfg.window_span_seconds:.3f}s, "
          f"Nyquist={cfg.nyquist_hz:.1f}Hz)")
    return preps


def make_loss(fused, cfg):
    return make_batch_loss(fused.ekf, fused.kinematics, cfg.eps,
                           beta=cfg.beta, objective=cfg.objective, remat=cfg.remat)


# ---------------------------------------------------------------------------
# check-init — the §4 properties, on real feature windows
# ---------------------------------------------------------------------------

def check_init(params, batch, cfg, batch_loss) -> dict:
    r"""At initialisation: ``Σ_C = σ₀²I`` for every contact, trunk grad exactly 0.

    Both are properties of `network.init`'s zero head, but they are checked here,
    end to end on **real** feature windows, because that is where they can break:
    a normalization bug that made a window non-finite, or a head that was not
    actually zero, shows up as a non-identity ``Σ_C`` or a nonzero trunk
    gradient long before it shows up in a loss curve.
    """
    L = contact_factors(params, batch.windows[0], cfg.eps)         # (L, N_c, 3, 3)
    Sigma = jnp.einsum("...ij,...kj->...ik", L, L)
    target = cfg.sigma_0 ** 2 * jnp.eye(3)
    sigma_err = float(jnp.max(jnp.abs(Sigma - target)))

    # Jitted: an interpreted `scan` of L ticks x B segments through MJX FK is
    # minutes, and this is a pre-flight check, not the run.
    (loss, _), grads = jax.jit(jax.value_and_grad(batch_loss, has_aux=True))(params, batch)
    trunk = [np.asarray(x) for x in jax.tree.leaves(grads.trunk)]
    head = [np.asarray(x) for x in jax.tree.leaves(grads.head)]
    trunk_max = max(float(np.abs(g).max()) for g in trunk)
    head_max = max(float(np.abs(g).max()) for g in head)
    return {
        "sigma_abs_err": sigma_err,
        "sigma_rel_err": sigma_err / cfg.sigma_0 ** 2,
        "trunk_grad_max": trunk_max,
        "trunk_grad_exactly_zero": all(bool(np.all(g == 0.0)) for g in trunk),
        "head_grad_max": head_max,
        "loss_at_init": float(loss),
        "windows_finite": bool(np.all(np.isfinite(np.asarray(batch.windows)))),
    }


# ---------------------------------------------------------------------------
# measure-b — peak RSS for forward+backward, per (B, remat)
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGESIZE") / 1e6


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def measure_one(args) -> dict:
    """One ``(B, remat)`` point, in its **own process** — `ru_maxrss` is a high-water
    mark, so two points in one process cannot be told apart."""
    cfg = make_config(args)
    c = build_estimator(args, verbose=False)
    preps = stage_prepare(args, cfg)
    P0 = np.eye(9 + 3 * cfg.n_contacts) * 1e-4
    batch = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=0)))
    key = jax.random.PRNGKey(0)
    params = network.init(key, cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    batch_loss = make_loss(c.fused, cfg)
    grad_fn = jax.jit(jax.grad(batch_loss, has_aux=True))

    # `ru_maxrss` is a high-water mark that cannot be reset, so compilation and
    # execution have to be separated in TIME: compile ahead of time, read the
    # peak, then execute and read it again.  Reporting one number for "peak of
    # compile+run" would let a 25 s XLA compile hide the number that actually
    # bounds the training loop.
    base_setup = _rss_mb()
    t0 = time.perf_counter()
    compiled = grad_fn.lower(params, batch).compile()
    compile_s = time.perf_counter() - t0
    peak_compile = _peak_mb()

    base_exec = _rss_mb()
    t1 = time.perf_counter()
    g, _ = compiled(params, batch)
    jax.block_until_ready(g)
    first_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    g, _ = compiled(params, batch)
    jax.block_until_ready(g)
    step_s = time.perf_counter() - t2

    return {"B": cfg.B, "L": cfg.L, "remat": cfg.remat,
            "setup_mb": base_setup, "peak_compile_mb": peak_compile,
            "exec_base_mb": base_exec, "peak_mb": _peak_mb(),
            "exec_delta_mb": _peak_mb() - base_exec,
            "compile_delta_mb": peak_compile - base_setup,
            "compile_s": compile_s, "first_s": first_s, "step_s": step_s,
            "grad_norm": float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g))))}


def measure_b(args):
    rows = []
    for B in args.batch_sizes:
        for remat in (True, False):
            cmd = [sys.executable, str(REPO_ROOT / "train_contactnet.py"), "measure-one",
                   "--B", str(B), "--L", str(args.L), "--H", str(args.H),
                   "--data", str(args.data), "--cache", str(args.cache),
                   "--norm", str(args.norm), "--n-rollouts", str(args.n_rollouts or 4)]
            if not remat:
                cmd.append("--no-remat")
            print(f"\n=== B={B} remat={remat} ===", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("MEASURE ")]
            if not line:
                print(r.stdout[-3000:], r.stderr[-3000:])
                rows.append({"B": B, "remat": remat, "error": r.stderr.strip().splitlines()[-1:]})
                continue
            rows.append(json.loads(line[-1][len("MEASURE "):]))
            print(f"  peak {rows[-1]['peak_mb']:.0f} MB "
                  f"(exec delta {rows[-1]['exec_delta_mb']:.0f} MB), "
                  f"step {rows[-1]['step_s']:.2f}s")
    print("\n" + "=" * 96)
    print("peak[MB] is the whole process; exec-delta is the fwd+bwd footprint on top of the\n"
          "already-loaded dataset, measured AFTER an ahead-of-time compile so the XLA\n"
          "compiler's own peak (compile-delta) is a separate column.\n")
    print(f"{'B':>4} {'remat':>6} {'setup[MB]':>10} {'compileD[MB]':>13} {'execD[MB]':>10} "
          f"{'peak[MB]':>10} {'compile[s]':>11} {'step[s]':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['B']:>4} {str(r['remat']):>6}   FAILED: {r['error']}")
            continue
        print(f"{r['B']:>4} {str(r['remat']):>6} {r['setup_mb']:10.0f} "
              f"{r['compile_delta_mb']:13.0f} {r['exec_delta_mb']:10.0f} {r['peak_mb']:10.0f} "
              f"{r['compile_s']:11.1f} {r['step_s']:9.2f}")
    return rows


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(args):
    cfg = make_config(args)
    print(f"config: {cfg}")
    c = build_estimator(args)
    preps = stage_prepare(args, cfg)

    P0 = (np.load(args.p0)["P0"] if args.p0 and Path(args.p0).exists()
          else dataset.measure_p0(c.fused, preps[0], cfg, ticks=args.p0_ticks))
    if args.p0:
        Path(args.p0).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.p0, P0=P0)

    key = jax.random.PRNGKey(args.seed)
    params = network.init(key, cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    n_params = sum(int(np.size(x)) for x in jax.tree.leaves(params))
    print(f"network: d_in={cfg.d_in}, widths={cfg.widths}, {n_params} params")

    batch_loss = make_loss(c.fused, cfg)

    probe = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=args.seed)))
    print("\ncheck-init (§4, on real feature windows):")
    for k, v in check_init(params, probe, cfg, batch_loss).items():
        print(f"  {k:26s} {v}")

    if args.chained:
        # No `sigma_0`: on the process socket the warm-in runs on the recorded
        # heuristic, i.e. on the shipped filter. See `make_warm_in`.
        warm_in = make_warm_in(c.fused.ekf, c.fused.kinematics)
        t_build = time.perf_counter()
        batcher = dataset.ChainedBatcher(preps, cfg, P0, warm_in, seed=args.seed)
        print(f"chains: {cfg.B} seeded + warmed in "
              f"({cfg.warm_in_s:.1f}s each, episode {cfg.episode_s:.0f}s) "
              f"in {time.perf_counter() - t_build:.0f}s")
        src = dict(batcher=batcher)
    else:
        print("chains: DISABLED -- every segment re-seeded from truth (run-1 behaviour)")
        src = dict(batches=dataset.batch_stream(preps, cfg, P0, steps=args.steps,
                                                seed=args.seed))

    print(f"\ntraining {args.steps} steps, objective={cfg.objective}\n")
    t0 = time.perf_counter()
    params, _, history = train_mod.train(
        params, batch_loss, **src,
        peak_lr=cfg.peak_lr, total_steps=cfg.total_steps, warmup_steps=cfg.warmup_steps,
        max_norm=cfg.max_norm, weight_decay=cfg.weight_decay, dof=cfg.dof,
        log_every=args.log_every)
    wall = time.perf_counter() - t0

    hist = {k: [float(getattr(m, k)) for m in history] for k in history[0]._fields}
    print(f"\n{args.steps} steps in {wall:.0f}s ({wall / max(1, args.steps):.2f} s/step)")
    print(f"{'step':>6} {'loss':>13} {'|g|':>12} {'NIS/dof':>12} {'applied':>9} {'cond':>11}")
    idx = sorted(set(list(range(min(6, args.steps))) + list(range(0, args.steps, max(1, args.steps // 12))) + [args.steps - 1]))
    for i in idx:
        print(f"{i:6d} {hist['loss'][i]:13.6e} {hist['grad_norm'][i]:12.4e} "
              f"{hist['nis_over_dof'][i]:12.4e} {hist['applied_frac'][i]:9.3f} "
              f"{hist['cond_proxy_max'][i]:11.3e}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        train_mod.save_params(args.out, params)
        with open(str(Path(args.out).with_suffix(".history.json")), "w") as f:
            json.dump({"config": cfg.__dict__, "history": hist, "wall_s": wall}, f, indent=1)
        print(f"\n-> {args.out}")
    return params, hist


def run_p0(args):
    """Measure `P0` under the CURRENT training conventions and write it.

    Its own subcommand because a stale `P0` is silent: it seeds every segment
    from the wrong prior and moves `nis_over_dof` for reasons unrelated to the
    network. `measure_p0`'s conventions changed with the socket move on
    2026-07-29 (it burns in on the recorded heuristic now, not on the frozen
    constant), so **every `p0_*.npz` written before that date is stale** —
    including `artifacts/p0_dr.npz`.
    """
    if not args.p0:
        raise SystemExit("pass --p0 <path.npz>: this subcommand exists to write one")
    cfg = make_config(args)
    c = build_estimator(args)
    preps = stage_prepare(args, cfg)
    P0 = dataset.measure_p0(c.fused, preps[0], cfg, ticks=args.p0_ticks)
    Path(args.p0).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.p0, P0=P0)
    w = np.linalg.eigvalsh(P0)
    print(f"\n-> {args.p0}  ({P0.shape[0]}x{P0.shape[0]}, "
          f"eig [{w.min():.3e}, {w.max():.3e}], "
          f"freeze_contact_chol={cfg.freeze_contact_chol})")
    return P0


def run_check_init(args):
    cfg = make_config(args)
    c = build_estimator(args)
    preps = stage_prepare(args, cfg)
    P0 = dataset.measure_p0(c.fused, preps[0], cfg, ticks=args.p0_ticks)
    batch = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=args.seed)))
    params = network.init(jax.random.PRNGKey(args.seed), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    for k, v in check_init(params, batch, cfg, make_loss(c.fused, cfg)).items():
        print(f"{k:26s} {v}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["cache", "norm", "prepare", "train", "p0",
                                        "measure-b", "measure-one", "check-init"])
    ap.add_argument("--data", default=str(dataset.DATA_DIR))
    ap.add_argument("--cache", default=str(dataset.CACHE_DIR))
    ap.add_argument("--norm", default=str(dataset.NORM_PATH))
    ap.add_argument("--out", default="")
    ap.add_argument("--p0", default="", help="npz to cache the measured P0 in")
    ap.add_argument("--p0-ticks", type=int, default=3000)
    ap.add_argument("--force", action="store_true", help="rebuild cache / refit norm")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="l2_velocity", choices=["l2_velocity", "beta_nll"])
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--sigma-0", dest="sigma_0", type=float, default=1.0e-4)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--no-remat", action="store_true")
    # Chained segments are the default (PORT_NOTES, run-1 root cause).  --no-chained
    # restores run-1's per-segment truth re-seed for the ablation.
    ap.add_argument("--no-chained", dest="chained", action="store_false", default=True)
    ap.add_argument("--freeze-contact-chol", action="store_true",
                    help="run-1 behaviour: pin the process socket at stance (10.2x worse)")
    ap.add_argument("--warm-in-s", type=float, default=1.0)
    ap.add_argument("--episode-s", type=float, default=43.0)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--fk-chunk", type=int, default=2_000)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--n-rollouts", type=int, default=0,
                    help="limit prepared rollouts (measure-b only; keeps setup RSS low)")
    args = ap.parse_args()

    if args.command == "cache":
        stage_cache(args)
    elif args.command == "norm":
        stage_norm(args)
    elif args.command == "prepare":
        stage_prepare(args, make_config(args))
    elif args.command == "train":
        run_train(args)
    elif args.command == "p0":
        run_p0(args)
    elif args.command == "check-init":
        run_check_init(args)
    elif args.command == "measure-b":
        measure_b(args)
    elif args.command == "measure-one":
        print("MEASURE " + json.dumps(measure_one(args)))


if __name__ == "__main__":
    main()
