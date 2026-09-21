"""Produce the InEKF comparison arms from THIS implementation, in the scorer's schema.

Arms 4, 5 and 6 differ only in where their measurement noise comes from, and that difference is the
paper's result. Producing them from one codebase is therefore not a convenience: if arm 4 came from
the Java replay and arm 6 from here, a difference between them would be unattributable between the
intended variable and the two implementations.

    arm 4  HAND_TUNED           noise as shipped on the robot
    arm 5  NIS_REFITTED         contact_floor scaled, fitted against NIS on a fit window
    arm 6  REDUNDANCY_OBSERVED  per-tick IMU-pair R from the array's own off-axis residual

Arm 3 (InEKF + encoder-only) is NOT produced here: this pipeline has no path that bypasses the
joint-level filter, so producing it would mean building that path, not configuring one. It stays on
the Java side for now, which leaves the "does the distributed IMU help" comparison spanning two
implementations -- recorded in the manifest rather than left for a reader to discover.

Two conventions this file exists to get right, both silent when wrong:

* **Timestamps are `tick_index * dt * 1e9`**, matching `AlexEstimatorLogReplay`'s
  `(long)(tickIndex * logDt * 1.0e9)` exactly -- NOT the log's own timestamps and NOT
  `LogWindow.time`, which is window-relative and derived from those timestamps. On the 2026-07-17
  log the two differ by ~14% in rate and by the whole window offset in base, so a CSV written on the
  wrong clock would never align with a Java-produced arm against the same mocap capture.
* **`wx..wz` is the bias-corrected base gyro in WORLD**, obtained as `R @ (imu_to_body @ (raw -
  bias))`. That inner quantity is what `two_stage.make_step` feeds the filter and what Java's
  `getMeasuredAngularVelocityInBody` returns; the outer rotation is the writer's job.

Usage:
    ALEX_AB_LOG=<log dir> python scripts/run_comparison_arms.py <out dir> <start> <end> [--fit a b]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_contact_anchor_offset import FEET, LOG, TRUST_FORMAT, build_stack  # noqa: E402
from offaxis_adaptive_r import (causal_extra, fit_constants, inflation, make_session,  # noqa: E402
                                offaxis, rollout, split)

from invariant_estimation.eval.comparison_csv import write  # noqa: E402

# Fitted on the fit window only; see retune_constant_noise.py and offaxis_adaptive_r.py.
ARM5_CONTACT_FLOOR_SCALE = 0.005
ARM6_ALPHA = 0.366
# The strictly causal variant is the default: it beats the same-tick form on every held-out window
# (cross-regime spread 10.7x against 14.7x) AND never reads the current tick, which both answers the
# circularity objection and makes the law implementable online.
ARM6_CAUSAL_WINDOW = 100


def timestamps_ns(window):
    """Java's clock: the GLOBAL tick index times the handshake dt, in nanoseconds.

    `LogWindow.tick` is the global index, so this is independent of where the window starts and of
    the log's own (irregular) timestamps -- which is the point, since the Java arms use the same
    construction and the two must be interchangeable.
    """
    return np.rint(np.asarray(window.tick, dtype=np.float64) * window.dt * 1.0e9).astype(np.int64)


def base_omega_body(ctx, out):
    """The bias-corrected base gyro in body, per tick -- the filter's own omega.

    Mirrors `two_stage.make_step`: `imu_to_body @ (raw - bias)`, with the bias read out of the joint
    filter's state at the same tick. Recomputed rather than plumbed through the rollout so this
    script needs no change to the filter's return type.
    """
    build = ctx["build"]
    session = ctx["session"]
    n = build.n_joints
    base = int(build.base_imu)

    x = np.asarray(out.joint_state.x)                       # (T, 2n + 3m)
    bias = x[:, 2 * n + 3 * base: 2 * n + 3 * base + 3]     # (T, 3)
    raw = np.asarray(session.sensors.gyros)[:, base, :]     # (T, 3)
    imu_to_body = np.asarray(session.imu_to_body)
    return (raw - bias) @ imu_to_body.T


def arm_extra(name, fit_ctx, ctx):
    """The per-tick pair-R inflation for one arm, or None for a frozen-R arm."""
    if name != "arm6":
        return None
    scale, s0, _ = fit_constants(fit_ctx, ARM6_ALPHA)
    _, _, off, _, _ = offaxis(ctx)
    _, fast = split(off)
    return causal_extra(inflation(fast, scale) * s0[None, :], ARM6_CAUSAL_WINDOW)


def arm_ekf(name, ctx):
    """The EKF for one arm; only arm 5 moves a constant."""
    ekf = ctx["ekf"]
    if name != "arm5":
        return ekf
    return ekf._replace(params=ekf.params._replace(
        contact_floor=ekf.params.contact_floor * ARM5_CONTACT_FLOOR_SCALE))


def main():
    out_dir = sys.argv[1]
    window_s = (float(sys.argv[2]), float(sys.argv[3]))
    fit_w = (110.0, 113.6)
    if "--fit" in sys.argv:
        i = sys.argv.index("--fit")
        fit_w = (float(sys.argv[i + 1]), float(sys.argv[i + 2]))

    os.makedirs(out_dir, exist_ok=True)
    print(f"log    : {LOG}")
    print(f"window : ticks {int(window_s[0] * 1000)}..{int(window_s[1] * 1000)}")
    print(f"fit    : {fit_w} (arms 5 and 6 only; nothing in the produced window is fitted)\n")

    ctx = make_session(window_s)
    fit_ctx = make_session(fit_w)
    stamps = timestamps_ns(ctx["window"])

    produced = []
    for name in ("arm4", "arm5", "arm6"):
        extra = arm_extra(name, fit_ctx, ctx)
        out = rollout({**ctx, "ekf": arm_ekf(name, ctx)}, extra)

        state = out.base.state
        path = os.path.join(out_dir, f"{name}-estimate.csv")
        write(path,
              timestamps_ns=stamps,
              rotations=np.asarray(state.R),
              positions=np.asarray(state.p),
              velocities_world=np.asarray(state.v),
              angular_velocities_body=base_omega_body(ctx, out),
              covariances=np.asarray(state.P))
        speed = np.linalg.norm(np.asarray(state.v), axis=-1).mean()
        print(f"{name}: {len(stamps)} ticks  |v| mean={speed:.4f} m/s -> {path}", flush=True)
        produced.append((name, path, speed))

    manifest = os.path.join(out_dir, "arms_python.csv")
    with open(manifest, "w", encoding="utf-8") as stream:
        stream.write(f"log,{LOG}\n")
        stream.write(f"implementation,invariant-estimation (python/jax)\n")
        stream.write(f"window_start_tick,{int(window_s[0] * 1000)}\n")
        stream.write(f"window_end_tick,{int(window_s[1] * 1000)}\n")
        stream.write(f"fit_window,{fit_w[0]}-{fit_w[1]}\n")
        stream.write("timestamp_convention,tick_index * dt * 1e9 (matches AlexEstimatorLogReplay)\n")
        stream.write("arm3_note,not produced here: no encoder-only path in this pipeline; "
                     "if scored against these arms it spans two implementations\n")
        stream.write("arm,noise_source,estimate_csv,mean_speed_mps\n")
        for name, path, speed in produced:
            source = {"arm4": "HAND_TUNED", "arm5": "NIS_REFITTED",
                      "arm6": "REDUNDANCY_OBSERVED"}[name]
            stream.write(f"{name},{source},{os.path.basename(path)},{speed}\n")
    print(f"\nwrote {manifest}")


if __name__ == "__main__":
    main()
