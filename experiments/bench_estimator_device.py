"""CPU vs GPU for the estimator loop — the honest A/B.

    uv run python experiments/bench_estimator_device.py                 # both, if a GPU is present
    uv run python experiments/bench_estimator_device.py --device cpu    # one device (child mode)

Why a subprocess per device
---------------------------
The JAX backend has to be chosen BEFORE `jax` is first imported, and importing anything from
`invariant_estimation` runs `jax.config.update("jax_enable_x64", True)` at
`src/invariant_estimation/__init__.py:15`. So a `--device` flag inside one process would need an
`sys.argv` scan above the imports -- an import-order landmine. Instead the parent re-execs itself
once per device with `JAX_PLATFORMS` set, which is also exactly how a user switches backends:

    JAX_PLATFORMS=cpu uv run python run_estimator.py ...

What it reports, and why all three columns
------------------------------------------
Build+compile time, steady-state ms per control tick, AND the resulting tilt/drift error. The
error columns are not decoration: a backend that is faster but less accurate is not a win, and on
this filter the two are genuinely coupled (float64 on a consumer GPU runs at 1/64 of float32).

The result, and why it was a surprise
-------------------------------------
The prediction going in was that **CPU would win, and by a lot**: per estimator step MJX FK/CRB is
only ~0.4 ms of ~2.5 ms, the rest being the two filters' own small dense linear algebra at batch
size 1 with no `vmap` anywhere; that is hundreds of tiny kernels (the worst case for launch
overhead), float64 on a consumer card runs at 1/64 of float32, and every tick does a blocking host
round-trip. Every one of those statements is still true.

**The GPU won anyway.** Measured on an RTX 4070 SUPER (driver 580.173.02), baseline policy,
250 ticks at vx=0.6, three interleaved repeats:

    cpu   p50 13.59-13.84 ms   xRT 1.45-1.47   tilt 0.856 deg   drift 0.353 m
    gpu   p50  8.93- 9.05 ms   xRT 2.21-2.24   tilt 0.856 deg   drift 0.353 m

1.50-1.55x on the median tick, with the error columns IDENTICAL to three decimals -- so it is a
real win, not a speed/accuracy trade. Build+compile is ~17 s slower on GPU (71 s vs 54 s), paid
once. Keep re-running this rather than trusting the paragraph above: the reasoning was sound and
the conclusion was still wrong, which is the whole reason this script exists.

The GPU extra also unblocks TERRAIN.md Stage 0 (thousands of vmapped envs), where the win is not
in question.
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bench(policy, ticks, vx, warmup):
    """Child mode: measure on whatever backend `JAX_PLATFORMS` selected."""
    import time

    import numpy as np

    sys.path.insert(0, REPO)
    import run_estimator as re_
    import run_policy as rp
    import jax

    t0 = time.time()
    loop = re_.make_estimated_loop(policy, with_visuals=False, verbose=False)
    build_s = time.time() - t0

    loop.cmd[0:3] = (vx, 0.0, 0.0)
    loop.cmd[3] = 0.0 if abs(vx) > 1e-9 else 1.0

    per_tick = []
    for k in range(ticks):
        t = time.perf_counter()
        loop.control_tick()
        per_tick.append((time.perf_counter() - t) * 1e3)
    # Drop the warm-up ticks: the first few carry cache effects, not steady state.
    steady = np.array(per_tick[warmup:])
    s = re_.summarise(loop.history)
    return {
        "device": jax.devices()[0].platform,
        "device_str": str(jax.devices()[0]),
        "build_s": build_s,
        "ms_p50": float(np.percentile(steady, 50)),
        "ms_p90": float(np.percentile(steady, 90)),
        "ms_mean": float(steady.mean()),
        "x_realtime": float(rp.DECIMATION * rp.DT * 1e3 / np.median(steady)),
        "tilt_tail_rms_deg": s["tilt_deg_tail_rms"],
        "att_tail_rms_deg": s["att_deg_tail_rms"],
        "p_err_tail_rms_m": s["p_err_tail_rms"],
        "final_p_err_m": s["final_p_err"],
    }


def _run_child(device, args):
    env = dict(os.environ, JAX_PLATFORMS=device)
    cmd = [sys.executable, os.path.abspath(__file__), "--device", device,
           "--policy", args.policy, "--ticks", str(args.ticks), "--vx", str(args.vx),
           "--warmup", str(args.warmup), "--_child"]
    p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print(f"  {device}: FAILED\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return None


def _has_gpu():
    try:
        import jax_cuda13_plugin  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="baseline")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--device", default=None, help="cpu|cuda; default runs every available one")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:
        print("RESULT " + json.dumps(_bench(args.policy, args.ticks, args.vx, args.warmup)))
        return

    devices = [args.device] if args.device else (["cpu", "cuda"] if _has_gpu() else ["cpu"])
    if not args.device and "cuda" not in devices:
        print("no CUDA jaxlib installed -- CPU only. Install the opt-in extra to A/B it:\n"
              "    uv sync --extra gpu\n")

    rows = [r for r in (_run_child(d, args) for d in devices) if r]
    if not rows:
        raise SystemExit("every device failed")

    print(f"\n{args.policy}, {args.ticks} ticks at vx={args.vx} "
          f"(steady state = after {args.warmup} warm-up ticks)\n")
    hdr = (f"{'device':10s} {'build':>7s} {'p50':>8s} {'p90':>8s} {'mean':>8s} {'xRT':>6s} "
           f"{'tilt':>7s} {'att':>7s} {'drift':>7s}")
    print(hdr)
    print(f"{'':10s} {'s':>7s} {'ms':>8s} {'ms':>8s} {'ms':>8s} {'':>6s} "
          f"{'deg':>7s} {'deg':>7s} {'m':>7s}")
    for r in rows:
        print(f"{r['device']:10s} {r['build_s']:7.1f} {r['ms_p50']:8.2f} {r['ms_p90']:8.2f} "
              f"{r['ms_mean']:8.2f} {r['x_realtime']:6.2f} {r['tilt_tail_rms_deg']:7.3f} "
              f"{r['att_tail_rms_deg']:7.3f} {r['p_err_tail_rms_m']:7.3f}")

    print("\n  xRT > 1 means the loop keeps real time (20 ms control period).")
    if len(rows) > 1:
        a, b = rows[0], rows[1]
        print(f"  {a['device']} vs {b['device']}: {a['ms_p50'] / b['ms_p50']:.2f}x on median tick, "
              f"tilt {b['tilt_tail_rms_deg'] - a['tilt_tail_rms_deg']:+.3f} deg.")
        print("  A faster device that costs accuracy is NOT a win -- read both halves.")
    else:
        print("  Only one backend measured; record the verdict in RUNNING.md either way.")


if __name__ == "__main__":
    main()
