"""TERRAIN.md Stage 1 — does the walking policy survive IsaacLab's terrain, in OUR sim?

Plain MuJoCo, no MJX: swap `run_policy`'s floor plane for a heightfield rasterised from the
IsaacLab sub-terrain parameters (TERRAIN.md §1) and walk the baseline policy across each one.

This is the cheapest gate in the plan and it PASSES — all four terrains, ~0.38 m/s against a
commanded 0.4, upright for 20 s.

The rasterisers and the floor spec now live in `invariant_estimation.sim.terrain`; this script is
just the driver. Nothing here rebuilds the sim model — `terrain.build_terrain_model` calls
`run_policy.build_sim_model(floor=...)`, so the collision set, contact parameters and actuators
have exactly one definition.

    uv run python experiments/terrain_stage1_walk.py
    uv run python experiments/terrain_stage1_walk.py --secs 30 --terrain waves
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # `run_policy` lives at the repo root
import run_policy as rp                                        # noqa: E402
from invariant_estimation.sim import terrain as tr             # noqa: E402

CONTROL_DT = rp.DT * rp.DECIMATION      # 0.02 s; travel is compared per SECOND, never per tick
SETTLE_TICKS = 100                      # 2 s of standing before the walk command


def run(label, field, policy, vx=0.4, secs=20.0):
    """Settle, then walk +x for `secs`. Returns the measurements the gate is judged on."""
    floor = tr.HeightfieldFloor(field)
    m = rp.build_sim_model(policy, with_visuals=False, floor=floor)

    # The compiled model must actually carry this terrain. A flat hfield behind a "terrain" label
    # is the failure mode TERRAIN.md §7's last bullet warns about, and it passes every other check.
    got = m.hfield_data.reshape(field.shape) * tr.EZ
    assert np.allclose(got, field, atol=1e-6), "hfield_data does not match the rasterised field"

    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    loop.d.qpos[2] += tr.spawn_lift(field)          # start clear of the terrain
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)      # re-seed the height ramp from the new pose
    for _ in range(SETTLE_TICKS):
        loop.control_tick()

    rest_z = float(loop.d.qpos[2])
    x0 = loop.d.qpos[0]
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("LEFT_FOOT", "RIGHT_FOOT")]
    tilts, soles, under = [], [], []
    for _ in range(int(secs / CONTROL_DT)):
        loop.cmd[0:3] = (vx, 0.0, 0.0)
        loop.cmd[3] = 0.0
        loop.control_tick()
        tilts.append(loop.tilt_deg())
        # The sole plane of the LOWER foot: the height of the ground the robot is actually on.
        soles.append(min(loop.d.xpos[b][2] - rp.ANKLE_HEIGHT for b in feet))
        under.append(float(tr.sample(field, loop.d.qpos[0], loop.d.qpos[1])))
    t, sole, ter = np.array(tilts), np.array(soles), np.array(under)
    dx = float(loop.d.qpos[0] - x0)
    fell = bool((t > 45).any()) or not np.all(np.isfinite(loop.d.qpos))
    print(f"  {label:34s} relief={floor.relief * 100:5.1f}cm  travelled={dx:+6.2f}m "
          f"({dx / secs:+.2f} m/s)  tilt_max={t.max():5.1f}  "
          f"sole_z={sole.mean() * 100:+5.1f}+-{sole.std() * 100:4.1f}cm  "
          f"terrain_under={ter.mean() * 100:5.1f}cm  "
          + ("FELL" if fell else f"UPRIGHT {secs:.0f}s"))
    return dict(label=label, relief=floor.relief, dx=dx, speed=dx / secs,
                tilt_max=float(t.max()), rest_z=rest_z, fell=fell,
                sole_mean=float(sole.mean()), sole_std=float(sole.std()),
                terrain_mean=float(ter.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="baseline")
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--terrain", action="append", choices=list(tr.TERRAINS),
                    help="run only these (default: all four)")
    args = ap.parse_args()

    policy = rp.load_policy(args.policy)
    names = args.terrain or list(tr.TERRAINS)
    fields = {n: tr.TERRAINS[n]() for n in names}

    print(f"hfield {tr.N}x{tr.N} px over {tr.EXTENT} m at {tr.HSCALE} m/px, "
          f"elevation scale {tr.EZ} m, {tr.N * tr.N * 4 / 1e6:.1f} MB of hfield_data")
    print(f"physics dt={rp.DT} decimation={rp.DECIMATION} -> control {1 / CONTROL_DT:.0f} Hz\n")
    print(f"IsaacLab's sub-terrains, walking {args.policy} at vx={args.vx} for {args.secs:.0f} s:")
    out = {n: run(n, fields[n], policy, vx=args.vx, secs=args.secs) for n in names}

    # A run that "completed" over silently flat ground proves nothing (TERRAIN.md §7, last bullet),
    # and neither does the RESTING pose: `hard_stepping` deliberately spawns on a flat 1 m platform,
    # so its resting height matches flat by design. The load-bearing statement is that the sole
    # plane RISES ONTO the terrain over the 7.6 m walk, and tracks the field we rasterised.
    print("\n  terrain-is-real check (mean over the walk, cm):")
    for n, r in out.items():
        gap = (r["sole_mean"] - r["terrain_mean"]) * 100
        print(f"    {n:18s} sole {r['sole_mean'] * 100:+5.1f}   terrain under pelvis "
              f"{r['terrain_mean'] * 100:5.1f}   sole-terrain {gap:+5.1f}")
        # The sole rides on the sampled terrain to within the size of one foot box (0.26 x 0.14 m),
        # which can bridge a stone edge or a wave crest -- hence cm-scale, not mm-scale, agreement.
        assert abs(gap) < 2.5, f"{n}: sole plane is {gap:.1f} cm off its own terrain"
    for n in ("waves", "hard_stepping"):
        if n in out and "flat" in out:
            lift = out[n]["sole_mean"] - out["flat"]["sole_mean"]
            assert lift > 0.01, (f"{n}: sole averaged only {lift * 100:.1f} cm above flat -- the "
                                 "robot is walking on flat ground, whatever the label says")

    if any(r["fell"] for r in out.values()):
        raise SystemExit("FAILED: fell on " + ", ".join(r["label"] for r in out.values() if r["fell"]))


if __name__ == "__main__":
    main()
