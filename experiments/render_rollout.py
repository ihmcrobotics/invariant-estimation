r"""Render a collected training rollout to video — see what ContactNet trained on.

`sim/collect.py` records sensors, ground truth and the estimator's inputs, but
nothing you can look at.  This re-simulates a saved rollout with a camera on it.

**It re-simulates rather than replays**, because the saved `.npz` holds
`truth.R/v/p` and the sensor streams but not `qpos` — there is no pose sequence
to play back.  That is fine here for a reason worth stating: `collect._RecordingLoop`
builds its observation from `MjData`, i.e. ground truth, so the *estimator is not
in the control loop* during collection.  Physics + policy + the same spawn
therefore reproduce the same trajectory, and the run is much cheaper without the
estimator (collection measured 3.4 wall-s per sim-s, almost all of it the
estimator pass; physics alone is 0.14).

Whether that reproduction actually held is **checked, not assumed**: the script
compares its final base position and yaw against the rollout's recorded
`truth.p[-1]` / `truth.R[-1]` and says so.  A video of a *different* trajectory
than the one trained on would be worse than no video.

`with_visuals=True` is safe: per `run_policy._add_scene_look` it adds textures,
materials and a light — no geom, mass or collision — so the dynamics are the ones
that were collected.

Usage
-----
    uv run python -m experiments.render_rollout data/flat_seed000.npz
    uv run python -m experiments.render_rollout data/stepping_stones_seed000.npz \
        --seconds 20 --out artifacts/video/stones.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# MUJOCO_GL must be set before `import mujoco` — same argv dance as run_estimator.py.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import run_policy as rp  # noqa: E402
from invariant_estimation.sim import terrain as tr  # noqa: E402
from invariant_estimation.sim.collect import spawn_pose, terrain_field  # noqa: E402


def _yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rollout", help="a .npz written by sim/collect.py")
    ap.add_argument("--out", default=None, help="mp4 path (default artifacts/video/<name>.mp4)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="render only the first N walking seconds (default: the whole rollout)")
    ap.add_argument("--fps", type=int, default=50, help="capped by the 50 Hz control loop")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--distance", type=float, default=3.2)
    ap.add_argument("--azimuth", type=float, default=120.0)
    ap.add_argument("--elevation", type=float, default=-15.0)
    args = ap.parse_args()

    src = Path(args.rollout)
    with np.load(src, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        p_true = z["truth.p"]
        R_true = z["truth.R"]

    terrain_name, seed = meta["terrain"], int(meta["seed"])
    vx, settle_s = meta["vx"], meta["settle_s"]
    walk_s = meta["seconds"] if args.seconds is None else float(args.seconds)
    full = args.seconds is None

    out = Path(args.out) if args.out else REPO / "artifacts/video" / f"{src.stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = (int(x) for x in args.size.lower().split("x"))

    print(f"{src.name}: terrain={terrain_name} seed={seed} vx={vx} "
          f"{settle_s:.0f}s settle + {walk_s:.0f}s walk")
    print(f"  recorded: travelled {meta['travelled_m']:.2f} m, "
          f"tilt_max {meta['tilt_max_deg']:.2f} deg")

    # -- the same model, plus the scene look ---------------------------------
    field = terrain_field(terrain_name, seed)
    floor = tr.HeightfieldFloor(field)
    policy = rp.load_policy(meta.get("policy", "baseline"))
    m = rp.build_sim_model(policy, with_visuals=True, with_imu_sensors=True, floor=floor)
    loop = rp.Loop(m, policy, rp.make_maps(m, policy))

    # -- the same spawn ------------------------------------------------------
    x0, y0, yaw = spawn_pose(seed)
    assert np.allclose([x0, y0, yaw], meta["spawn_xy_yaw"], atol=1e-9), (
        "spawn_pose no longer reproduces the recorded spawn — the rollout and "
        "this build disagree about the spawn box")
    loop.d.qpos[0:2] = (x0, y0)
    loop.d.qpos[2] += tr.spawn_lift(field)
    loop.d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)

    control_dt = rp.DT * rp.DECIMATION
    settle_ticks = int(round(settle_s / control_dt))
    walk_ticks = int(round(walk_s / control_dt))

    base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if base_bid < 0:
        base_bid = 1                       # root body: the free joint's child
    rec = rp.VideoRecorder(m, str(out), body=base_bid,
                           width=w, height=h, fps=args.fps, distance=args.distance,
                           azimuth=args.azimuth, elevation=args.elevation)
    try:
        for k in range(settle_ticks + walk_ticks):
            if k >= settle_ticks:
                loop.cmd[0:3] = (vx, 0.0, 0.0)
                loop.cmd[3] = 0.0
            loop.control_tick()
            if not np.all(np.isfinite(loop.d.qpos)):
                raise RuntimeError(f"non-finite qpos at control tick {k}")
            rec.capture(loop.d)
    finally:
        rec.close()

    # -- did we reproduce the trajectory, or merely something like it? -------
    got_p = np.asarray(loop.d.qpos[0:3])
    k_end = min(len(p_true) - 1, (settle_ticks + walk_ticks) * rp.DECIMATION - 1)
    want_p = np.asarray(p_true[k_end])
    dp = float(np.linalg.norm(got_p[:2] - want_p[:2]))
    dyaw = np.degrees(abs(_yaw(np.asarray(R_true[k_end])) -
                          _yaw(loop.d.xmat[base_bid].reshape(3, 3))))

    print(f"\nreproduction check at tick {k_end}:")
    print(f"  recorded base xy {want_p[:2]}   this run {got_p[:2]}")
    print(f"  horizontal error {dp:.4f} m, yaw error {dyaw:.3f} deg")
    if dp < 0.05:
        print("  MATCH — this is the trajectory ContactNet trained on.")
    elif full:
        print("  DIVERGED — the video shows a different trajectory than the "
              "recorded rollout. Do not draw conclusions about the training data "
              "from it until this is explained.")
    else:
        print("  (partial render; the check only means anything for a full one)")


if __name__ == "__main__":
    main()
