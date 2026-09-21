"""Does the contact floor become tunable once the IMU-pair noise model is right?

The contact-floor retune (scripts/retune_constant_noise.py) improved held-out consistency on every
window and degraded the estimate: mean |v| +79%, std(v_y) 0.149 -> 0.368, |dv/dt| 2.7 -> 3.4 m/s^2.
The increase was variance, not drift, and the diagnosis was that tightening the floor makes the
filter trust contact forward kinematics more -- while contact FK during walking is exactly what the
`stacked` finding shows to be corrupted, so the corruption reaches the state.

That diagnosis makes a falsifiable prediction, and this script tests it.

The mechanism it depends on is real and worth stating, because the two knobs live in different
stages: the joint filter's posterior `Sigma_q = P[:n, :n]` becomes the InEKF's contact measurement
noise through `N = J Sigma_q J^T` (two_stage line 62, correct.map_encoder_noise). So when arm 6
widens the joint filter's R during an impact, `Sigma_q` widens, `N` widens, and the InEKF
correctly distrusts contact FK at exactly the ticks where it is corrupted.

    PREDICTION: with arm 6's adaptive R active, tightening the contact floor should cost much less
    velocity variance than it does with the frozen R -- because the corruption it was letting in is
    now modelled.

If the velocity penalty is unchanged, the diagnosis was wrong and the floor's effect comes from
somewhere else. Either answer is worth having; the script reports the number, not a verdict.

RESULT (2026-09-21): the prediction FAILED, and it replicates on both held-out windows.

    window        cost of tightening the floor to 0.005, std(v_y)
    102.4-106.0   arm4 frozen 0.149 -> 0.368 (2.48x)   arm6 adaptive 0.144 -> 0.366 (2.53x)
    118.0-121.6   arm4 frozen 0.100 -> 0.220 (2.21x)   arm6 adaptive 0.096 -> 0.220 (2.29x)

Arm 6 repairs `stacked` dramatically on both windows (266 -> 1.25 and 17.4 -> 0.58) and buys the
velocity estimate nothing at all -- if anything it is marginally worse. The two channels are
decoupled, so the corruption the floor lets in does not arrive through Sigma_q.

The likelier reading, which this does not itself establish: the contact floor is not a mis-set
noise constant but structural slack absorbing anchor motion the process model has no way to
represent. A single point anchor per foot is an approximation during roll-over -- the instantaneous
axis is the toe or heel edge, not any fixed point on the sole -- so the foot genuinely moves while
the model says it does not, and the floor is where that goes. Tightening it forbids real motion and
the filter attributes the difference to the base. On that reading the floor becomes tunable only
after the anchor model represents roll-over (multiple contact points, or an anchor free to rotate),
which is a different piece of work from noise calibration.

Usage:
    ALEX_AB_LOG=<log dir> python scripts/floor_after_redundancy.py <start> <end> [more windows...]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offaxis_adaptive_r import (causal_extra, fit_constants, inflation, make_session,  # noqa: E402
                                offaxis, rollout, split)
from run_comparison_arms import ARM6_ALPHA, ARM6_CAUSAL_WINDOW  # noqa: E402

from invariant_estimation.eval.consistency import two_stage_consistency  # noqa: E402

FIT_WINDOW = (110.0, 113.6)
FLOOR_SCALES = (1.0, 0.05, 0.005)


def measure(ctx, extra, floor_scale):
    """One rollout: consistency plus the estimate-quality numbers the floor retune damaged."""
    base_ekf = ctx["ekf"]
    ekf = base_ekf if floor_scale == 1.0 else base_ekf._replace(
        params=base_ekf.params._replace(contact_floor=base_ekf.params.contact_floor * floor_scale))

    out = rollout({**ctx, "ekf": ekf}, extra)
    reports = {r.channel: (float(r.anis), float(r.dof))
               for r in two_stage_consistency(out.joint_diagnostics, out.base,
                                              n_contacts=2, n_joints=9)}
    R = np.asarray(out.base.state.R)
    v = np.asarray(out.base.state.v)
    vb = np.einsum("tji,tj->ti", R, v)
    return dict(contact=reports["contact"][0] / reports["contact"][1],
                stacked=reports["stacked"][0] / reports["stacked"][1],
                speed=float(np.linalg.norm(v, axis=-1).mean()),
                std_vy=float(vb[:, 1].std()),
                jerk=float(np.linalg.norm(np.diff(v, axis=0), axis=1).mean() / 1.0e-3))


def main():
    windows = [(float(sys.argv[i]), float(sys.argv[i + 1])) for i in range(1, len(sys.argv) - 1, 2)]
    fit_ctx = make_session(FIT_WINDOW)
    scale, s0, _ = fit_constants(fit_ctx, ARM6_ALPHA)
    print(f"arm 6 constants fitted on {FIT_WINDOW} (alpha={ARM6_ALPHA}, "
          f"causal window {ARM6_CAUSAL_WINDOW}); every window below is held out.\n")

    for window in windows:
        ctx = make_session(window)
        _, _, off, _, _ = offaxis(ctx)
        _, fast = split(off)
        adaptive = causal_extra(inflation(fast, scale) * s0[None, :], ARM6_CAUSAL_WINDOW)

        print(f"########## window {window[0]}-{window[1]}")
        print(f"{'noise model':16s} {'floor':>7s} {'contact/6':>10s} {'stacked/27':>11s} "
              f"{'|v| mean':>9s} {'std v_y':>8s} {'|dv/dt|':>8s}")
        summary = {}
        for label, extra in (("arm4 frozen R", None), ("arm6 adaptive R", adaptive)):
            for floor in FLOOR_SCALES:
                m = measure(ctx, extra, floor)
                print(f"{label:16s} {floor:>7.3f} {m['contact']:>10.3f} {m['stacked']:>11.3f} "
                      f"{m['speed']:>9.4f} {m['std_vy']:>8.3f} {m['jerk']:>8.2f}", flush=True)
                summary[(label, floor)] = m

        print()
        for label in ("arm4 frozen R", "arm6 adaptive R"):
            loose, tight = summary[(label, 1.0)], summary[(label, min(FLOOR_SCALES))]
            print(f"  {label:16s} cost of tightening the floor to {min(FLOOR_SCALES)}: "
                  f"std v_y {loose['std_vy']:.3f} -> {tight['std_vy']:.3f} "
                  f"({tight['std_vy'] / max(loose['std_vy'], 1e-9):.2f}x), "
                  f"|v| {loose['speed']:.4f} -> {tight['speed']:.4f} "
                  f"({tight['speed'] / max(loose['speed'], 1e-9):.2f}x)")
        print()


if __name__ == "__main__":
    main()
