#!/usr/bin/env python
"""Record a closed-loop ContactNet demo video WITH the ghost (estimated state).

`run_estimator.py` cannot do this from the CLI: `--ghost` is viewer-only and
`--video` implies headless, and the two are mutually exclusive there (the ghost is
a viewer `user_scn` overlay, the offscreen recorder never sees it). This script
renders offscreen AND injects the ghost geoms into each frame, while driving a
command schedule through the six validation motions so the clip shows the robot
doing forward / backward / strafe L,R / turn L,R with the learned contact-noise
model in the loop.

    uv run --extra gpu python scripts/record_contactnet_demo.py \
        --contactnet results/latest/params.npz --contacts-per-foot 4 \
        --out results/latest/closed_loop/contactnet_demo_ghost.mp4

Drop --contactnet for the analytic-baseline version. `--ghost attitude` pins the
ghost at the true position (isolates orientation error) if the full-pose ghost
drifts too far off-screen.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")   # offscreen render; must precede the mujoco import

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

import run_policy as rp
import run_estimator as re_mod
from invariant_estimation.sim.ghost import Ghost

# (label, (vx, vy, yaw), seconds) -- the six validation motions, magnitudes per VAL_MODES.
SCHEDULE = [
    ("stand",     (0.00,  0.00,  0.00), 1.0),
    ("forward",   (0.45,  0.00,  0.00), 4.0),
    ("backward",  (-0.45, 0.00,  0.00), 4.0),
    ("lateral_L", (0.00,  0.40,  0.00), 4.0),
    ("lateral_R", (0.00, -0.40,  0.00), 4.0),
    ("turn_L",    (0.00,  0.00,  0.75), 4.0),
    ("turn_R",    (0.00,  0.00, -0.75), 4.0),
]


def capture_with_ghost(rec, d, ghost, est):
    """One recorder frame with the ghost drawn on top -- `VideoRecorder.capture`
    split so the ghost geoms can be appended between `update_scene` and `render`."""
    rec.renderer.update_scene(d, camera=rec.cam, scene_option=rec.opt)
    if est is not None:
        ghost.update(est, d)
        ghost.draw(rec.renderer.scene)
    rec.proc.stdin.write(rec.renderer.render().tobytes())
    rec.n += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contactnet", default=None, metavar="PARAMS.npz",
                    help="run the trained ContactNet in the loop (omit for the analytic baseline)")
    ap.add_argument("--contactnet-norm", default=None)
    ap.add_argument("--contacts-per-foot", type=int, choices=(1, 4), default=1,
                    help="contact slots per foot; must match the checkpoint's training "
                         "geometry (the N=8 ladder arms need 4). run_estimator.py "
                         "enforces this against the checkpoint's summary.json")
    ap.add_argument("--out", required=True, metavar="PATH.mp4")
    ap.add_argument("--ghost", choices=("full", "attitude"), default="full")
    ap.add_argument("--seconds", type=float, default=None,
                    help="override every motion's duration (default: the schedule's own)")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--size", default="1280x720", metavar="WxH")
    ap.add_argument("--contact-meas-var", type=float, default=1.0e-3,
                    help="InEKF contact measurement-noise floor. MUST match what the "
                         "checkpoint was TRAINED at -- the 2026-08-10 sweep found this "
                         "parameter dominates drift (it is ~30,000x the measured "
                         "Sigma_q, so it effectively sets how much the contact FK "
                         "measurement is listened to at all). The recommended "
                         "checkpoint was trained at 1e-3; running it at 0 evaluates a "
                         "configuration it never saw.")
    ap.add_argument("--imu-noise", action="store_true",
                    help="corrupt the IMU as the training pool was collected "
                         "(n8fix has imu_noise=True and a true gyro bias). Running a "
                         "net trained on noisy IMU against clean sensors is a "
                         "distribution shift, and the estimator's bias state has "
                         "nothing to estimate.")
    ap.add_argument("--zero-velocity", action="store_true",
                    help="add the contact zero-velocity measurement block (Z6 gate): "
                         "H gains velocity columns instead of Sigma_C reweighting a "
                         "correction whose direction H fixes")
    ap.add_argument("--nv-scale", type=float, default=1.0, metavar="KAPPA",
                    help="diagnostic multiplier on the zero-velocity block's noise "
                         "N^v (only with --zero-velocity). The derivation fixes this "
                         "at 1 with no free parameter; sweeping it is the "
                         "falsification of that claim. KAPPA -> inf must reproduce "
                         "the no-ZV baseline exactly.")
    ap.add_argument("--gyro-var", type=float, default=None,
                    help="InEKF process noise on the gyro channel [(rad/s)^2/Hz]. "
                         "Default (None) takes config/filter_cfg.yaml's 1e-4, which "
                         "is inherited from the Java config and was never fitted to "
                         "this sim (sim/sensors.IMUNoise: gyro white 1e-3 rad/s).")
    ap.add_argument("--accel-var", type=float, default=None,
                    help="InEKF process noise on the accelerometer channel "
                         "[(m/s^2)^2/Hz]. Default (None) takes the config's 1e-3; "
                         "the sim's accel white noise is 3e-2 m/s^2.")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--metrics", default=None, metavar="PATH.json",
                    help="also record per-motion vertical drift (estimate vs truth)")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    loop = re_mod.make_estimated_loop(
        "baseline", with_visuals=True, contactnet=args.contactnet,
        contactnet_norm=args.contactnet_norm, verbose=True,
        contacts_per_foot=args.contacts_per_foot,
        contact_meas_var=args.contact_meas_var,
        zero_velocity=args.zero_velocity, nv_scale=args.nv_scale,
        gyro_var=args.gyro_var, accel_var=args.accel_var,
        noise=(re_mod.IMUNoise(seed=args.noise_seed) if args.imu_noise else None))
    print(f"  contact_meas_var={args.contact_meas_var:g} "
          f"zero_velocity={args.zero_velocity} nv_scale={args.nv_scale:g} "
          f"gyro_var={loop.rt.fused.ekf.params.gyro_var:g} "
          f"accel_var={loop.rt.fused.ekf.params.accel_var:g}")
    # dof of the contact-update NIS: the block is 3 rows per contact slot.
    nis_dof = 3 * loop.rt.fused.n_contacts
    ghost = Ghost(loop.m, loop.maps, loop.filtered_slots, mode=args.ghost)

    rec = rp.VideoRecorder(loop.m, args.out, body=loop.maps["BASE_BID"],
                           width=w, height=h, fps=args.fps)
    control_dt = rp.DECIMATION * rp.DT
    stride = max(1, int(round((1.0 / control_dt) / args.fps)))   # control ticks per recorded frame

    print(f"  recording {w}x{h} @ {args.fps:.0f}fps, ghost={args.ghost}, "
          f"contactnet={'yes' if args.contactnet else 'no (baseline)'}")
    tick = 0
    stop = False
    trace = []                       # (label, t, e_z, horiz_err, nis/dof) per control tick
    for label, cmd, secs in SCHEDULE:
        if stop:
            break
        loop.cmd[0:3] = cmd
        loop.cmd[3] = 0.0 if np.any(np.abs(np.asarray(cmd)) > 1e-9) else 1.0
        n = int(round((args.seconds if args.seconds else secs) / control_dt))
        print(f"    {label:10s} cmd={cmd} for {n} ticks")
        for _ in range(n):
            loop.control_tick()
            est = loop.current_estimate()
            if est is not None:
                p_true = np.asarray(loop.d.xpos[loop.maps["BASE_BID"]], dtype=float)
                e = np.asarray(est.p, dtype=float) - p_true
                # Contact-update NIS on the PRIOR (CLAUDE.md §6), normalised by its
                # 3N dof so the target is 1.0 regardless of `--contacts-per-foot`.
                # Reported BESIDE drift, never instead of it (N2): a filter can be
                # consistent and still sink.
                trace.append((label, tick * control_dt, float(e[2]),
                              float(np.linalg.norm(e[:2])),
                              float(loop.history[-1]["nis"]) / nis_dof))
            if tick % stride == 0:
                capture_with_ghost(rec, loop.d, ghost, est)
            tick += 1
            if not np.all(np.isfinite(loop.d.qpos)):
                print(f"    !! non-finite state during {label}; stopping")
                stop = True
                break
    rec.close()

    if trace:
        import json
        from collections import OrderedDict
        by = OrderedDict()
        for label, t, ez, eh, nis in trace:
            by.setdefault(label, []).append((t, ez, eh, nis))
        print("\n  closed-loop vertical error, per motion "
              "(e_z = p_hat_z - p_true_z; CUMULATIVE across the clip)")
        print(f"    {'motion':10s} {'secs':>5s} {'e_z start':>10s} {'e_z end':>9s} "
              f"{'d(e_z)':>8s} {'rate m/s':>9s} {'horiz':>7s} {'NIS/dof':>8s}")
        rows = []
        for label, seg in by.items():
            t = np.array([r[0] for r in seg]); ez = np.array([r[1] for r in seg])
            eh = np.array([r[2] for r in seg])
            nis = np.array([r[3] for r in seg])
            # rate WITHIN the motion: the clip is one continuous run, so absolute e_z
            # carries in from earlier motions and only the slope is attributable here.
            rate = float(np.polyfit(t, ez, 1)[0]) if len(t) > 2 else float("nan")
            finite = nis[np.isfinite(nis)]
            rows.append(dict(motion=label, seconds=float(t[-1] - t[0]),
                             ez_start=float(ez[0]), ez_end=float(ez[-1]),
                             ez_delta=float(ez[-1] - ez[0]), rate_mps=rate,
                             horiz_end=float(eh[-1]),
                             # median, not mean: the NIS distribution has a heavy
                             # touchdown tail and a mean is set by a handful of ticks.
                             nis_per_dof=float(np.median(finite)) if finite.size else
                             float("nan"),
                             nis_per_dof_mean=float(finite.mean()) if finite.size else
                             float("nan")))
            print(f"    {label:10s} {rows[-1]['seconds']:5.1f} {ez[0]:+10.3f} "
                  f"{ez[-1]:+9.3f} {rows[-1]['ez_delta']:+8.3f} {rate:+9.5f} "
                  f"{eh[-1]:7.3f} {rows[-1]['nis_per_dof']:8.4f}")
        if args.metrics:
            pathlib_out = args.metrics
            with open(pathlib_out, "w") as f:
                json.dump(dict(contact_meas_var=args.contact_meas_var,
                               contactnet=args.contactnet,
                               zero_velocity=args.zero_velocity,
                               nv_scale=args.nv_scale,
                               gyro_var=args.gyro_var, accel_var=args.accel_var,
                               nis_dof=nis_dof, per_motion=rows), f, indent=2)
            print(f"  metrics -> {pathlib_out}")


if __name__ == "__main__":
    main()
