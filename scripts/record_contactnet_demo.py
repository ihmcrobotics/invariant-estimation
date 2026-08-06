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
    ap.add_argument("--rolling", action="store_true",
                    help="enable the rolling-anchor contact density; must match the "
                         "checkpoint's training filter (run_estimator._check_contact_"
                         "geometry enforces this against summary.json)")
    ap.add_argument("--rolling-tau", type=float, default=None, metavar="SECONDS")
    ap.add_argument("--rolling-sigma-r", type=float, default=None, metavar="METRES")
    ap.add_argument("--out", required=True, metavar="PATH.mp4")
    ap.add_argument("--ghost", choices=("full", "attitude"), default="full")
    ap.add_argument("--seconds", type=float, default=None,
                    help="override every motion's duration (default: the schedule's own)")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--size", default="1280x720", metavar="WxH")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    loop = re_mod.make_estimated_loop(
        "baseline", with_visuals=True, contactnet=args.contactnet,
        contactnet_norm=args.contactnet_norm, verbose=True,
        contacts_per_foot=args.contacts_per_foot,
        rolling=(re_mod.me.inekf_mod.default_rolling_anchor_params(
            enabled=True, tau=args.rolling_tau, sigma_r=args.rolling_sigma_r)
            if args.rolling else None))
    ghost = Ghost(loop.m, loop.maps, loop.filtered_slots, mode=args.ghost)

    rec = rp.VideoRecorder(loop.m, args.out, body=loop.maps["BASE_BID"],
                           width=w, height=h, fps=args.fps)
    control_dt = rp.DECIMATION * rp.DT
    stride = max(1, int(round((1.0 / control_dt) / args.fps)))   # control ticks per recorded frame

    print(f"  recording {w}x{h} @ {args.fps:.0f}fps, ghost={args.ghost}, "
          f"contactnet={'yes' if args.contactnet else 'no (baseline)'}")
    tick = 0
    stop = False
    for label, cmd, secs in SCHEDULE:
        if stop:
            break
        loop.cmd[0:3] = cmd
        loop.cmd[3] = 0.0 if np.any(np.abs(np.asarray(cmd)) > 1e-9) else 1.0
        n = int(round((args.seconds if args.seconds else secs) / control_dt))
        print(f"    {label:10s} cmd={cmd} for {n} ticks")
        for _ in range(n):
            loop.control_tick()
            if tick % stride == 0:
                capture_with_ghost(rec, loop.d, ghost, loop.current_estimate())
            tick += 1
            if not np.all(np.isfinite(loop.d.qpos)):
                print(f"    !! non-finite state during {label}; stopping")
                stop = True
                break
    rec.close()


if __name__ == "__main__":
    main()
