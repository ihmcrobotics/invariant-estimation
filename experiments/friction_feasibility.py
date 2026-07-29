r"""Is friction randomisation a viable source of contact diversity? Sweep and measure.

The question this answers
------------------------
The learned ``Sigma_C`` from runs 2 and 3 is **79% explained by gait phase alone**
(``R^2 = 0.790`` of ``log10 std_z`` against time-since-touchdown; contact trust
explains 0.02).  With one gait, "contact quality" and "stride phase" are the same
variable, so a stride-phase clock is the most the network can learn — and more of
the same gait teaches it nothing new.

Friction randomisation is one way out: if slip appears at times the phase clock
cannot predict, contact quality decorrelates from phase and the network is forced
to key on something else.  That only works if there is a friction window where
**slip actually occurs and the policy still walks**.  Both halves are measured
here.

Two facts frame it:

* Our sim floor is ``friction="1 0.05 0.01"`` (`run_policy.CONTACT`), i.e. mu = 1.0.
* The RL policy's own domain randomisation is floor friction **[0.8, 1.4]**
  (`~/alex/alexander-mujoco/training/envs/alexander/randomize.py`).  Below 0.8 we
  are extrapolating outside the distribution the policy was trained on, so
  "walks" cannot be assumed and has to be checked per point.

How slip is measured
--------------------
Not by differentiating FK — `experiments/dataset_stats.py` records why that
cannot separate sliding from heel-to-toe roll.  Here the sim still knows the
contact geometry, so slip is read from the **friction cone**: for every foot
contact, `mj_contactForce` gives the normal and tangential components in the
contact frame, and the contact is sliding when

    |f_tangential| >= slip_frac * mu * f_normal

Coulomb's law makes that the definition rather than a proxy: a contact strictly
inside its cone cannot slide.  This is the instrumentation PORT_NOTES asked for
and it belongs at collection time, which is why it lives against the live sim.

Usage
-----
    uv run python -m experiments.friction_feasibility
    uv run python -m experiments.friction_feasibility --mu 1.0 0.6 0.3 --seconds 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import run_policy as rp  # noqa: E402
from invariant_estimation.sim import terrain as tr  # noqa: E402
from invariant_estimation.sim.collect import spawn_pose, terrain_field  # noqa: E402

POLICY_DR_RANGE = (0.8, 1.4)
"""Floor friction the RL policy was trained over. Outside it, walking is not given."""


def _foot_geom_ids(m) -> set[int]:
    ids = set()
    for name in rp.FOOT_GEOMS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g >= 0:
            ids.add(g)
    return ids


def probe(mu: float, *, terrain_name: str, seed: int, seconds: float,
          settle_s: float = 2.0, vx: float = 0.4) -> dict:
    """Walk at floor friction `mu`; return fall/slip/travel statistics."""
    field = terrain_field(terrain_name, seed)
    floor = tr.HeightfieldFloor(field)
    policy = rp.load_policy("baseline")
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True, floor=floor)

    # Override sliding friction on every geom; the feet and the floor both matter,
    # and MuJoCo takes the elementwise max of the two geoms' friction by default.
    m.geom_friction[:, 0] = mu

    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    feet = _foot_geom_ids(m)
    x0, y0, yaw = spawn_pose(seed)
    loop.d.qpos[0:2] = (x0, y0)
    loop.d.qpos[2] += tr.spawn_lift(field)
    loop.d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)
    p0 = loop.d.qpos[:2].copy()

    settle_ticks = int(round(settle_s / (rp.DT * rp.DECIMATION)))
    walk_ticks = int(round(seconds / (rp.DT * rp.DECIMATION)))
    frc = np.zeros(6)
    n_contact = n_slip = 0
    sat = []
    tilt_max = 0.0
    fell = False

    for k in range(settle_ticks + walk_ticks):
        if k >= settle_ticks:
            loop.cmd[0:3] = (vx, 0.0, 0.0)
            loop.cmd[3] = 0.0
        loop.control_tick()
        if not np.all(np.isfinite(loop.d.qpos)):
            fell = True
            break
        R = loop.d.xmat[1].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        if tilt > 45.0 or loop.d.qpos[2] < 0.3:
            fell = True
            break
        if k < settle_ticks:
            continue
        for i in range(loop.d.ncon):
            c = loop.d.contact[i]
            if c.geom1 not in feet and c.geom2 not in feet:
                continue
            mujoco.mj_contactForce(m, loop.d, i, frc)
            fn = abs(frc[0])
            if fn < 5.0:                      # ignore grazing contacts
                continue
            ft = float(np.hypot(frc[1], frc[2]))
            s = ft / max(1e-9, mu * fn)
            sat.append(s)
            n_contact += 1
            n_slip += int(s >= 0.99)

    travelled = float(np.linalg.norm(loop.d.qpos[:2] - p0))
    sat = np.asarray(sat) if sat else np.zeros(1)
    return dict(mu=mu, fell=fell, travelled=travelled, tilt_max=tilt_max,
                n_contact=n_contact, slip_frac=n_slip / max(1, n_contact),
                sat_p50=float(np.percentile(sat, 50)),
                sat_p99=float(np.percentile(sat, 99)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mu", type=float, nargs="+",
                    default=[1.4, 1.0, 0.8, 0.6, 0.45, 0.35, 0.25, 0.15])
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    print(f"friction sweep: {args.terrain}/seed{args.seed}, {args.seconds:.0f}s walk")
    print(f"  sim default mu = 1.0; policy DR range = {POLICY_DR_RANGE}")
    print(f"  slip = friction-cone saturation |f_t| >= 0.99 mu f_n\n")
    print(f"{'mu':>6} {'walks':>6} {'travel':>8} {'tilt':>7} {'contacts':>9} "
          f"{'slip%':>7} {'cone p50':>9} {'cone p99':>9}")
    rows = []
    for mu in args.mu:
        r = probe(mu, terrain_name=args.terrain, seed=args.seed, seconds=args.seconds)
        rows.append(r)
        flag = "" if POLICY_DR_RANGE[0] <= mu <= POLICY_DR_RANGE[1] else "  (outside policy DR)"
        print(f"{r['mu']:6.2f} {'no' if r['fell'] else 'yes':>6} "
              f"{r['travelled']:8.2f} {r['tilt_max']:7.2f} {r['n_contact']:9d} "
              f"{100 * r['slip_frac']:7.2f} {r['sat_p50']:9.3f} {r['sat_p99']:9.3f}{flag}")

    ok = [r for r in rows if not r["fell"]]
    slipping = [r for r in ok if r["slip_frac"] > 0.01]
    print()
    if not ok:
        print("No friction in this sweep keeps the robot walking.")
    elif not slipping:
        lo = min(r["mu"] for r in ok)
        print(f"VERDICT: the robot walks down to mu = {lo:.2f} but never saturates its "
              f"friction cone. Lowering mu alone does not create slip on this gait -- "
              f"the tangential demand is simply too low.")
    else:
        lo = min(r["mu"] for r in slipping)
        hi = max(r["mu"] for r in slipping)
        print(f"VERDICT: usable window mu in [{lo:.2f}, {hi:.2f}] -- the robot walks "
              f"AND the contacts slide. That is the range to randomise over.")


if __name__ == "__main__":
    main()
