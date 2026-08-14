#!/usr/bin/env python
"""Instrument the quasi-static gate over a real closed-loop walking clip.

Why. The gate exists to keep roll/pitch observable, and it passes **0 of 20 000**
walking ticks — so attitude runs open-loop on the gyro through the entire gait,
and the z-budget attributes 14.7% of the vertical sink to attitude x specific
force. Nobody has looked at the actual per-tick values, so nobody knows *which*
of the three conditions is doing the rejecting or by how much.

    uv run python scripts/gravity_gate_trace.py --out /tmp/gate.npz

What it does. Drives the same six-motion validation clip `record_contactnet_demo`
uses, in closed loop, and records the InEKF's own boundary inputs per estimator
tick by wrapping `EstimatorRuntime._view`. Nothing is replayed through the filter:
the numbers are the ones the running filter saw.

The one reconstruction. `GravityRef` lives inside the scan carry and is not
emitted, so the complementary reference is recomputed in NumPy from the recorded
`(accel, raw_omega)` by re-running `update_gravity_reference` -- a pure function
of exactly those inputs. It is **verified, not assumed**: the recomputed gate is
compared tick-by-tick against the `quasi_static` mask the filter published, and
the script fails loudly if they ever disagree. See F.3 in CLAUDE.md §6 -- the gate
must read the sensor-driven reference and never `R^T e_z`, which is precisely why
the reference is reconstructible from sensors alone.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")   # no rendering here, but the mujoco
# import wires a GL backend regardless and osmesa is not installed in this venv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np

import run_policy as rp
import run_estimator as re_mod
from invariant_estimation.inEKF.gravity_update import (   # noqa: E402
    default_gravity_params, is_quasi_static,
    update_gravity_reference,
)
import jax.numpy as jnp

from record_contactnet_demo import SCHEDULE  # noqa: E402  (same clip, one source)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contacts-per-foot", type=int, choices=(1, 4), default=4)
    ap.add_argument("--contact-meas-var", type=float, default=1.0e-3)
    ap.add_argument("--seconds", type=float, default=None,
                    help="override every motion's duration (default: the clip's own)")
    ap.add_argument("--out", default=None, metavar="PATH.npz")
    args = ap.parse_args()

    loop = re_mod.make_estimated_loop(
        "baseline", with_visuals=False, verbose=True,
        contacts_per_foot=args.contacts_per_foot,
        contact_meas_var=args.contact_meas_var)
    dt = float(loop.rt.fused.ekf.params.dt)

    # The reference the filter starts from, taken off the seeded carry rather than
    # assumed (`init_fused_carry` seeds it from truth attitude, not `init_gravity_ref`).
    ref0 = loop.rt.carry[1].gravity_ref

    rec = {"accel": [], "raw_omega": [], "gate": [], "tilt": [], "label": []}
    label = [""]
    orig_view = loop.rt._view

    def view(out, last_sensors):
        rec["accel"].append(np.asarray(out.inekf_inputs.accel))
        rec["raw_omega"].append(np.asarray(out.inekf_inputs.raw_omega))
        rec["gate"].append(np.asarray(out.inekf.quasi_static))
        rec["tilt"].append(np.asarray(out.inekf.tilt_angle))
        rec["label"].extend([label[0]] * len(np.asarray(out.inekf.quasi_static)))
        return orig_view(out, last_sensors)

    loop.rt._view = view

    control_dt = rp.DECIMATION * rp.DT
    for name, cmd, secs in SCHEDULE:
        label[0] = name
        loop.cmd[0:3] = cmd
        loop.cmd[3] = 0.0 if np.any(np.abs(np.asarray(cmd)) > 1e-9) else 1.0
        n = int(round((args.seconds if args.seconds else secs) / control_dt))
        print(f"  {name:10s} cmd={cmd} for {n} control ticks")
        for _ in range(n):
            loop.control_tick()
            if not np.all(np.isfinite(loop.d.qpos)):
                raise SystemExit(f"non-finite state during {name}")

    accel = np.concatenate(rec["accel"])
    omega = np.concatenate(rec["raw_omega"])
    gate = np.concatenate(rec["gate"])
    tilt = np.concatenate(rec["tilt"])
    labels = np.array(rec["label"])
    n = len(gate)
    print(f"\n  {n} estimator ticks at dt={dt:g}s ({n * dt:.1f}s)")

    # -- reconstruct the complementary reference, then VERIFY it ------------
    p = default_gravity_params()
    # NOT `init_gravity_ref()`: `init_fused_carry(seed_gravity=True)` seeds the
    # reference from the truth attitude (`R0ᵀ e_z`) rather than leaving it unseeded,
    # so starting the reconstruction unseeded desynchronises it for the first few
    # hundred ticks. Take the filter's own initial reference.
    ref = ref0
    dirs = np.empty((n, 3))
    mine = np.empty(n)
    for k in range(n):
        a, w = jnp.asarray(accel[k]), jnp.asarray(omega[k])
        # `is_quasi_static` seeds internally, exactly as the filter's step does,
        # and the seed happens BEFORE the reference is advanced.
        mine[k] = float(is_quasi_static(ref, a, w, p))
        dirs[k] = np.asarray(
            (ref.direction if ref.initialized > 0.5
             else a / jnp.linalg.norm(a)))
        ref = update_gravity_reference(ref, a, w, dt, p)

    bad = int((mine != gate).sum())
    if bad:
        raise SystemExit(
            f"reconstruction disagrees with the filter on {bad}/{n} ticks — the "
            "gate terms below would not be the ones the filter used")
    print(f"  reference reconstruction verified against the filter's own mask "
          f"on all {n} ticks")

    # -- the three terms ---------------------------------------------------
    g = p.gravity
    norm_term = np.abs(np.linalg.norm(accel, axis=1) - g) / g      # vs norm_tol
    rot_term = np.linalg.norm(omega, axis=1)                       # vs rot_tol
    along = (accel * dirs).sum(axis=1)[:, None] * dirs
    horiz_term = np.linalg.norm(accel - along, axis=1)             # vs horiz_tol

    terms = [("norm  |f|-g|/g", norm_term, p.norm_tol),
             ("rot   |w_raw|", rot_term, p.rot_tol),
             ("horiz |f_perp|", horiz_term, p.horiz_tol)]

    walk = labels != "stand"
    print(f"\n  gate passes {gate.mean() * 100:.3f}% of all ticks, "
          f"{gate[walk].mean() * 100:.3f}% of WALKING ticks "
          f"({int(gate[walk].sum())}/{int(walk.sum())})")
    print(f"  tilt angle: median {np.median(tilt):.4f} rad, "
          f"p99 {np.quantile(tilt, 0.99):.4f} rad\n")

    print(f"  {'term':16s} {'tol':>8s} {'pass%':>7s} {'walk%':>7s} "
          f"{'med':>9s} {'p90':>9s} {'p99':>9s} {'max':>9s} {'med/tol':>8s}")
    for name, v, tol in terms:
        print(f"  {name:16s} {tol:8.3g} {(v <= tol).mean() * 100:7.2f} "
              f"{(v[walk] <= tol).mean() * 100:7.2f} {np.median(v):9.4f} "
              f"{np.quantile(v, 0.90):9.4f} {np.quantile(v, 0.99):9.4f} "
              f"{v.max():9.4f} {np.median(v) / tol:8.2f}")

    # Which single condition is binding? Drop one at a time.
    print("\n  leave-one-out pass rate on WALKING ticks (which condition binds):")
    ok = {name: (v <= tol) for name, v, tol in terms}
    allok = ok["norm  |f|-g|/g"] & ok["rot   |w_raw|"] & ok["horiz |f_perp|"]
    for name in ok:
        rest = np.ones(n, bool)
        for other, m in ok.items():
            if other != name:
                rest &= m
        print(f"    drop {name:16s} -> {rest[walk].mean() * 100:6.2f}% "
              f"(all three: {allok[walk].mean() * 100:.3f}%)")

    # What tolerance would each term need for a target duty cycle, holding the
    # other two at their configured values?
    print("\n  tolerance each term would need for a given WALKING duty cycle,")
    print("  with the OTHER TWO held at their configured values:")
    print(f"    {'term':16s} {'1%':>10s} {'5%':>10s} {'10%':>10s} {'25%':>10s}")
    for name, v, tol in terms:
        rest = np.ones(n, bool)
        for other, m in ok.items():
            if other != name:
                rest &= m
        pool = v[walk & rest]
        cells = []
        for duty in (0.01, 0.05, 0.10, 0.25):
            need = duty * walk.sum() / max(len(pool), 1)
            cells.append(f"{np.quantile(pool, min(need, 1.0)):10.4f}"
                         if len(pool) and need <= 1.0 else f"{'unreachable':>10s}")
        print(f"    {name:16s} " + " ".join(cells))

    if args.out:
        np.savez(args.out, accel=accel, raw_omega=omega, gate=gate, tilt=tilt,
                 ref_dir=dirs, labels=labels, norm_term=norm_term,
                 rot_term=rot_term, horiz_term=horiz_term)
        print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
