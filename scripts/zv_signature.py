#!/usr/bin/env python
"""Split closed-loop vertical error into its INTEGRATED and DEPOSITED halves.

    uv run python scripts/zv_signature.py results/.../hist_base.npz hist_zvK1.npz

Two questions, both written down in `zero-velocity-implementation-plan.md` before
any number was measured, and neither answerable from the metrics JSON:

**P1 — is the drift linear or sqrt(t)?** Linear means bias-dominated. The plan
predicts that if the zero-velocity constraint handles the foot-roll bias the
signature turns sqrt(t), and that if it stays linear the bias was not handled.
Measured as the ratio of R^2 for `e_z ~ a t` against `e_z ~ a sqrt(t)`, both fit
through the origin of the walking segment.

**P2 — how much of the error was DEPOSITED by the update?** Vertical error has two
routes: velocity error integrated over time, and position written directly by the
contact update into the base. The second is the fingerprint of the common-mode
null mode, and it sat at -0.18..-0.21 m across four earlier arms regardless of
what was changed:

    deposited = e_z(T) - sum_k (v_hat - v_true)_z dt

The subtracted term is the part any velocity-error reduction can explain; what is
left is what the update put there directly, which is the part `ker H` owns.
"""
import argparse
import sys

import numpy as np


def fit_through_origin(x, y):
    """Least-squares `y = a f(x)` and its R^2 (no intercept: t=0 has zero error)."""
    a = float(x @ y / (x @ x))
    ss_res = float(((y - a * x) ** 2).sum())
    ss_tot = float((y ** 2).sum())
    return a, 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def report(path):
    z = np.load(path)
    t = z["t"]
    dt = float(np.median(np.diff(t)))
    ez = z["est_p"][:, 2] - z["true_p"][:, 2]
    dv = z["est_v"][:, 2] - z["true_v"][:, 2]

    # Skip the 1 s standing prologue: the question is about gait.
    k0 = int(round(1.0 / dt))
    tt = t[k0:] - t[k0]
    yy = ez[k0:] - ez[k0]

    integrated = float(np.cumsum(dv[k0:])[-1] * dt)
    total = float(yy[-1])
    deposited = total - integrated

    a_lin, r2_lin = fit_through_origin(tt, yy)
    a_sqrt, r2_sqrt = fit_through_origin(np.sqrt(tt), yy)

    print(f"\n{path}")
    print(f"  {len(t)} ticks, dt={dt:g}s, walking window {tt[-1]:.1f}s")
    print(f"  vertical error over the walking window   {total:+.4f} m")
    print(f"    integrated from velocity error         {integrated:+.4f} m "
          f"({100 * integrated / total if total else float('nan'):5.1f}%)")
    print(f"    DEPOSITED directly by the update       {deposited:+.4f} m "
          f"({100 * deposited / total if total else float('nan'):5.1f}%)")
    print(f"  signature: linear R2={r2_lin:.4f} (a={a_lin:+.5f} m/s)   "
          f"sqrt(t) R2={r2_sqrt:.4f} (a={a_sqrt:+.5f} m/sqrt(s))")
    print(f"    -> {'LINEAR' if r2_lin >= r2_sqrt else 'SQRT(t)'} fits better "
          f"(dR2 = {abs(r2_lin - r2_sqrt):.4f})")
    print(f"  |dv_z| rms {np.sqrt((dv[k0:] ** 2).mean()):.4f} m/s, "
          f"mean {dv[k0:].mean():+.5f} m/s   "
          f"tilt rms {np.sqrt((z['tilt_deg'][k0:] ** 2).mean()):.3f} deg")
    return dict(total=total, integrated=integrated, deposited=deposited,
                r2_lin=r2_lin, r2_sqrt=r2_sqrt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", nargs="+")
    args = ap.parse_args()
    for p in args.npz:
        report(p)


if __name__ == "__main__":
    sys.exit(main())
