r"""Stability envelope of the Alex walking policy under domain randomisation.

The question this answers
------------------------
`experiments/friction_feasibility.py` established that floor friction is safe to
randomise all the way down to mu = 0.15 (the policy walks) but never saturates the
friction cone on the one gait we collect (vx = 0.4, straight, flat).  Every dataset
so far is that single gait, which is why the learned ``Sigma_C`` is 79% explained by
stride phase alone.

To break that degeneracy the collector needs *more axes* of randomisation, and each
one needs a measured safe range before it goes into an overnight run: a range that is
too wide throws away rollouts to falls, a range that is too narrow buys no diversity.
This script measures three axes against the live sim:

1. ``push``     -- random pelvis disturbance forces (``d.xfrc_applied``) while walking.
2. ``cmd``      -- the policy command vector ``[vx, vy, yaw, standing, height]``,
                   one axis at a time, resampled mid-rollout.
3. ``combined`` -- everything at once at the ranges the first two found safe, plus
                   floor friction randomised per rollout.

Method
------
Same skeleton as `friction_feasibility.py` (build via ``run_policy.build_sim_model``,
spawn via ``sim.collect.spawn_pose``, walk, measure), same fall definition (non-finite
qpos, tilt > 45 deg, or pelvis below 0.3 m) and the same friction-cone slip measure
(``mj_contactForce``; sliding iff ``|f_t| >= 0.99 mu f_n``, which is Coulomb's law, not
a proxy).  Nothing outside ``experiments/`` is touched -- this is measurement only.

MEASURED, 2026-07-28 (baseline policy, 90.5 kg robot, 20 s walks, 5-8 seeds each)
--------------------------------------------------------------------------------
Pushes (flat, mu = 1.0, vx = 0.4), fraction of seeds still up after 20 s::

    600 N 5/5 (tilt 10.0 deg)   700 N 5/5 (12.9)   750 N 4/5   900 N 2/5   1200 N 0/5

Command axes: **nothing fell, on any axis, anywhere in these ranges** --
vx in [-0.8, +1.6], vy in [-1.0, +1.0], yaw in [-1.8, +1.8], the whole height band
[0.83, 0.93], and walk/stand toggling.  The command envelope is not the binding
constraint; the only unusable region is the policy's own deadband, ``|vx| < 0.2``
with ``|yaw|`` small, where it stands still (0.03 m travelled in 20 s, 0.00% slip).

Combined (mu ~ U[0.2, 1.2], vx ~ U[0.25, 1.0], vy ~ U[-0.5, 0.5], yaw ~ U[-0.8, 0.8],
height over the full band, pushes ~ U[0, F]):

    F = 400 N   flat 8/8, waves 8/8, hard_stepping 8/8   slip 3.06 / 3.17 / 3.45 %
    F = 600 N   flat 8/8, waves 4/5 (fell at 4.6 s)      slip 1.43 %

so 400 N is the range to collect over and 600 N is already marginal on terrain.
A no-push control on waves gives 8/8 and 2.99% slip: the pushes buy +0.2 pp of slip
and ~2 deg of extra tilt, while the bulk of the slip comes from mu and the commands.

Usage
-----
    uv run python -m experiments.dr_envelope all          # everything (~5 min)
    uv run python -m experiments.dr_envelope push
    uv run python -m experiments.dr_envelope cmd --axis vx
    uv run python -m experiments.dr_envelope combined --seeds 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import run_policy as rp  # noqa: E402
from invariant_estimation.sim import terrain as tr  # noqa: E402
from invariant_estimation.sim.collect import spawn_pose, terrain_field  # noqa: E402

CTRL_DT = rp.DT * rp.DECIMATION          # 0.02 s -- one control tick
BASE_VX = 0.4                            # the vx every dataset so far used
DEFAULT_MU = 1.0                         # `run_policy.CONTACT`'s floor friction
SLIP_FRAC = 0.99                         # cone saturation threshold

_POLICY = None


def _policy():
    global _POLICY
    if _POLICY is None:
        _POLICY = rp.load_policy("baseline")
    return _POLICY


def _foot_geom_ids(m) -> set[int]:
    ids = set()
    for name in rp.FOOT_GEOMS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g >= 0:
            ids.add(g)
    return ids


# ---------------------------------------------------------------------------
# One rollout
# ---------------------------------------------------------------------------
def rollout(*, seed: int, seconds: float = 20.0, settle_s: float = 2.0,
            mu: float | tuple[float, float] = DEFAULT_MU,
            push_n: float | tuple[float, float] = 0.0,
            push_every: tuple[float, float] = (1.0, 3.0),
            push_dur: tuple[float, float] = (0.10, 0.20),
            vx: float | tuple[float, float] = BASE_VX,
            vy: float | tuple[float, float] = 0.0,
            yaw: float | tuple[float, float] = 0.0,
            height: tuple[float, float] | None = None,
            stand_toggle: bool = False,
            resample_every: tuple[float, float] = (2.0, 4.0),
            terrain_name: str = "flat") -> dict:
    """Walk one 20 s rollout under the given randomisation; return fall/slip/travel stats.

    Scalars are held fixed; ``(lo, hi)`` tuples are resampled -- friction and push
    magnitude once per rollout / per push, command axes every ``resample_every``.
    """
    rng = np.random.default_rng(0xDEC0DE + 7919 * int(seed))
    mu_v = float(rng.uniform(*mu)) if isinstance(mu, tuple) else float(mu)

    field = terrain_field(terrain_name, seed)
    floor = tr.HeightfieldFloor(field)
    policy = _policy()
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=False, floor=floor)
    m.geom_friction[:, 0] = mu_v            # MuJoCo takes the max of the two geoms

    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    feet = _foot_geom_ids(m)
    pelvis = loop.maps["BASE_BID"]
    h_lo, h_hi = policy["height_range"]

    x0, y0, yaw0 = spawn_pose(seed)
    loop.d.qpos[0:2] = (x0, y0)
    loop.d.qpos[2] += tr.spawn_lift(field)
    loop.d.qpos[3:7] = (np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2))
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)

    settle_ticks = int(round(settle_s / CTRL_DT))
    walk_ticks = int(round(seconds / CTRL_DT))

    def draw(spec, default):
        return float(rng.uniform(*spec)) if isinstance(spec, tuple) else float(
            default if spec is None else spec)

    # --- schedules ---------------------------------------------------------
    next_resample = 0
    next_push = settle_ticks + int(rng.uniform(*push_every) / CTRL_DT)
    push_off = -1
    standing = False
    n_push = 0
    push_mags: list[float] = []
    cmd_v = np.array([draw(vx, BASE_VX), draw(vy, 0.0), draw(yaw, 0.0)])

    frc = np.zeros(6)
    n_contact = n_slip = 0
    sat: list[float] = []
    tilt_max = 0.0
    fell = False
    t_fell = float("nan")
    p0 = loop.d.qpos[:2].copy()

    for k in range(settle_ticks + walk_ticks):
        if k >= settle_ticks:
            # -- command --------------------------------------------------
            if k >= next_resample:
                cmd_v = np.array([draw(vx, BASE_VX), draw(vy, 0.0), draw(yaw, 0.0)])
                if stand_toggle:
                    standing = not standing
                if height is not None:
                    loop.set_height_target(rng.uniform(max(height[0], h_lo),
                                                       min(height[1], h_hi)))
                next_resample = k + int(rng.uniform(*resample_every) / CTRL_DT)
            if standing:
                loop.cmd[0:3] = 0.0
                loop.cmd[3] = 1.0
            else:
                loop.cmd[0:3] = cmd_v
                loop.cmd[3] = 0.0
            # -- disturbance ----------------------------------------------
            hard = isinstance(push_n, tuple) or push_n > 0.0
            if hard and k >= next_push:
                mag = float(rng.uniform(*push_n)) if isinstance(push_n, tuple) else float(push_n)
                th = rng.uniform(-np.pi, np.pi)
                loop.d.xfrc_applied[pelvis, :3] = (mag * np.cos(th), mag * np.sin(th), 0.0)
                push_off = k + max(1, int(round(rng.uniform(*push_dur) / CTRL_DT)))
                next_push = push_off + int(rng.uniform(*push_every) / CTRL_DT)
                n_push += 1
                push_mags.append(mag)
            if push_off >= 0 and k >= push_off:
                loop.d.xfrc_applied[pelvis, :3] = 0.0
                push_off = -1

        loop.control_tick()

        # -- fall check (identical to friction_feasibility) -----------------
        if not np.all(np.isfinite(loop.d.qpos)):
            fell, t_fell = True, (k - settle_ticks) * CTRL_DT
            break
        R = loop.d.xmat[pelvis].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        if tilt > 45.0 or loop.d.qpos[2] < 0.3:
            fell, t_fell = True, (k - settle_ticks) * CTRL_DT
            break
        if k < settle_ticks:
            continue

        # -- friction-cone saturation ---------------------------------------
        for i in range(loop.d.ncon):
            c = loop.d.contact[i]
            if c.geom1 not in feet and c.geom2 not in feet:
                continue
            mujoco.mj_contactForce(m, loop.d, i, frc)
            fn = abs(frc[0])
            if fn < 5.0:                    # ignore grazing contacts
                continue
            s = float(np.hypot(frc[1], frc[2])) / max(1e-9, mu_v * fn)
            sat.append(s)
            n_contact += 1
            n_slip += int(s >= SLIP_FRAC)

    travelled = float(np.linalg.norm(loop.d.qpos[:2] - p0))
    a = np.asarray(sat) if sat else np.zeros(1)
    return dict(seed=seed, mu=mu_v, fell=fell, t_fell=t_fell, travelled=travelled,
                tilt_max=tilt_max, n_contact=n_contact, n_push=n_push,
                push_mean=float(np.mean(push_mags)) if push_mags else 0.0,
                slip_frac=n_slip / max(1, n_contact),
                sat_p50=float(np.percentile(a, 50)), sat_p99=float(np.percentile(a, 99)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def sweep(label: str, seeds, **kw) -> dict:
    rs = [rollout(seed=s, **kw) for s in seeds]
    up = [r for r in rs if not r["fell"]]
    return dict(label=label, n=len(rs), n_up=len(up), rows=rs,
                travel=float(np.mean([r["travelled"] for r in up])) if up else 0.0,
                tilt=float(np.max([r["tilt_max"] for r in rs])),
                slip=float(np.mean([r["slip_frac"] for r in rs])),
                sat99=float(np.mean([r["sat_p99"] for r in rs])),
                t_fell=float(np.nanmin([r["t_fell"] for r in rs])) if len(up) < len(rs)
                else float("nan"))


HDR = (f"{'case':>22} {'up/n':>7} {'travel':>8} {'tiltmax':>8} {'slip%':>7} "
       f"{'cone p99':>9} {'1st fall':>9}")


def show(s: dict) -> None:
    tf = "-" if not np.isfinite(s["t_fell"]) else f"{s['t_fell']:.1f}s"
    print(f"{s['label']:>22} {s['n_up']:>3}/{s['n']:<3} {s['travel']:8.2f} "
          f"{s['tilt']:8.1f} {100 * s['slip']:7.2f} {s['sat99']:9.3f} {tf:>9}")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_push(args) -> None:
    seeds = range(args.seeds)
    print(f"\n== 1. pelvis disturbance forces ==  {args.seconds:.0f}s walk, vx={BASE_VX}, "
          f"mu={args.mu}, pushes every 1-3 s for 0.10-0.20 s, random horizontal direction")
    print(HDR)
    out = []
    for n in args.newtons:
        s = sweep(f"push {n:.0f} N", seeds, seconds=args.seconds, mu=args.mu,
                  push_n=n, terrain_name=args.terrain)
        out.append((n, s))
        show(s)
    ok = [n for n, s in out if s["n_up"] == s["n"]]
    print(f"\n  all-seeds-survive up to: {max(ok):.0f} N" if ok else "\n  nothing survives")


CMD_BINS = {
    # The trained command band is vx +-0.9, vy +-0.5, yaw +-1.5 (RUNNING.md); the outer
    # bins deliberately leave it, since "safe to randomise" is an empirical question.
    "vx": [(-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2), (-0.2, 0.0), (0.0, 0.2), (0.2, 0.4),
           (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)],
    "vy": [(-1.0, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2), (-0.2, 0.0),
           (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)],
    "yaw": [(-1.8, -1.5), (-1.5, -1.2), (-1.2, -0.9), (-0.9, -0.6), (-0.6, -0.3), (-0.3, 0.0),
            (0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 1.8)],
}


def cmd_cmd(args) -> None:
    seeds = range(args.seeds)
    axes = [args.axis] if args.axis else ["vx", "vy", "yaw", "height", "stand"]
    for axis in axes:
        print(f"\n== 2. command axis: {axis} ==  {args.seconds:.0f}s, resampled every 2-4 s, "
              f"other axes at baseline (vx={BASE_VX})")
        print(HDR)
        if axis in CMD_BINS:
            for lo, hi in CMD_BINS[axis]:
                kw = dict(seconds=args.seconds, mu=args.mu, terrain_name=args.terrain)
                kw[axis] = (lo, hi)          # `vx` / `vy` / `yaw` are rollout kwargs
                show(sweep(f"{axis} [{lo:+.2f},{hi:+.2f}]", seeds, **kw))
        elif axis == "height":
            h_lo, h_hi = _policy()["height_range"]
            bins = [(h_lo, h_lo + (h_hi - h_lo) / 3), (h_lo + (h_hi - h_lo) / 3,
                    h_lo + 2 * (h_hi - h_lo) / 3), (h_lo + 2 * (h_hi - h_lo) / 3, h_hi),
                    (h_lo, h_hi)]
            for lo, hi in bins:
                show(sweep(f"h [{lo:.3f},{hi:.3f}]", seeds, seconds=args.seconds,
                           mu=args.mu, height=(lo, hi), terrain_name=args.terrain))
        elif axis == "stand":
            show(sweep("walk/stand toggle", seeds, seconds=args.seconds, mu=args.mu,
                       stand_toggle=True, terrain_name=args.terrain))
            show(sweep("toggle + vx rand", seeds, seconds=args.seconds, mu=args.mu,
                       stand_toggle=True, vx=(0.2, 0.8), terrain_name=args.terrain))


def cmd_combined(args) -> None:
    seeds = range(args.seeds)
    print(f"\n== 3. combined ==  {args.seconds:.0f}s, {args.seeds} seeds; "
          f"mu~U[{args.mu_lo},{args.mu_hi}], push~U[{args.push_lo},{args.push_hi}] N, "
          f"vx~U{tuple(args.vx)}, vy~U{tuple(args.vy)}, yaw~U{tuple(args.yaw)}, height~full band")
    print(HDR)
    h_lo, h_hi = _policy()["height_range"]
    kw = dict(seconds=args.seconds, mu=(args.mu_lo, args.mu_hi),
              push_n=(args.push_lo, args.push_hi), vx=tuple(args.vx), vy=tuple(args.vy),
              yaw=tuple(args.yaw), height=(h_lo, h_hi), stand_toggle=args.stand,
              terrain_name=args.terrain)
    s = sweep("combined", seeds, **kw)
    show(s)
    print(f"\n  survival {s['n_up']}/{s['n']}  slip {100 * s['slip']:.2f}%  "
          f"cone p99 {s['sat99']:.3f}")
    print(f"  {'seed':>5} {'mu':>6} {'up':>4} {'travel':>8} {'tilt':>7} {'slip%':>7} "
          f"{'pushes':>7} {'<F>N':>7}")
    for r in s["rows"]:
        print(f"  {r['seed']:5d} {r['mu']:6.2f} {'no' if r['fell'] else 'yes':>4} "
              f"{r['travelled']:8.2f} {r['tilt_max']:7.1f} {100 * r['slip_frac']:7.2f} "
              f"{r['n_push']:7d} {r['push_mean']:7.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["push", "cmd", "combined", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--mu", type=float, default=DEFAULT_MU)
    ap.add_argument("--newtons", type=float, nargs="+",
                    default=[0, 25, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 2000])
    ap.add_argument("--axis", default=None, choices=["vx", "vy", "yaw", "height", "stand"])
    ap.add_argument("--mu-lo", type=float, default=0.2)
    ap.add_argument("--mu-hi", type=float, default=1.2)
    ap.add_argument("--push-lo", type=float, default=0.0)
    ap.add_argument("--push-hi", type=float, default=150.0)
    ap.add_argument("--vx", type=float, nargs=2, default=[0.0, 0.8])
    ap.add_argument("--vy", type=float, nargs=2, default=[-0.3, 0.3])
    ap.add_argument("--yaw", type=float, nargs=2, default=[-0.5, 0.5])
    ap.add_argument("--stand", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    p = _policy()
    print(f"baseline policy: base_height={p['base_height']} height_range={p['height_range']}; "
          f"fall = |qpos| non-finite, tilt>45deg, or z<0.3 m")
    if args.mode in ("push", "all"):
        cmd_push(args)
    if args.mode in ("cmd", "all"):
        cmd_cmd(args)
    if args.mode in ("combined", "all"):
        cmd_combined(args)
    print(f"\n[{time.time() - t0:.0f}s wall]")


if __name__ == "__main__":
    main()
