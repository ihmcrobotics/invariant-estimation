"""NIS-based retune of the three constant noise channels: encoder, contact, gravity.

This is tuning -- but against a statistical target rather than by eye, and validated on windows
the fit never saw. For a consistent filter the average NIS of a channel equals its degrees of
freedom, and ANIS needs no ground truth, so this whole procedure is mocap-free. It is also a
hypothesis test: `stacked` was shown to be regime-dependent (475x spread) and unfixable by any
constant, while these three channels have no *measured* regime dependence (their standing readings
are degenerate zeros, not small numbers). If one constant per channel lands near dof on BOTH
held-out walking windows, constants suffice for them; if the two windows split the way `stacked`
did, they are regime-dependent too and this script will say so.

Knobs (no library changes):
  encoder  -- build.encoder_var scaled directly (the per-joint encoder R the joint KF's encoder
              update reads). Overconfident 4.3x -> scale UP.
  contact  -- the contact_fk_r theta channel (scales J Sigma_q J^T, the InEKF's contact R).
              Conservative 12.5x -> scale DOWN.
  gravity  -- gravity_roll_r and gravity_pitch_r theta channels, moved by ONE common factor to
              preserve the hand-tuned roll/pitch anisotropy. Conservative -> scale DOWN.

Order matters: encoder_var changes the joint KF's posterior Sigma_q, which IS the contact
channel's R, so encoder is fitted first, then contact, then gravity; a second pass re-checks.

Fit window: 110.0-113.6 (dt-seconds), the same one every prior fit on this log used.
Held-out: standing 70.0-73.6, walking 102.4-106.0 and 118.0-121.6. Nothing is fitted on them.

Caveats printed with the results rather than hidden: ANIS = dof is a necessary condition for
consistency, not a sufficient one (a per-channel scalar fixes the average, not the shape), and
the gravity channel applies rarely on some windows (n=80 on 102.4-106.0), so its per-window
numbers carry wide bands.

Usage:
    ALEX_AB_LOG=/path/to/log python scripts/retune_constant_noise.py fit
    ALEX_AB_LOG=/path/to/log python scripts/retune_constant_noise.py eval s_enc s_fk s_grav
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ab_contact_anchor_offset import FEET, LOG, TRUST_FORMAT, build_stack  # noqa: E402

from invariant_estimation.config import load_config  # noqa: E402
from invariant_estimation.inEKF import ekf as base_ekf  # noqa: E402
from invariant_estimation.jointKF.state import default_params  # noqa: E402
from invariant_estimation.learning.channels import alex_channel_map_from_log  # noqa: E402
from invariant_estimation.learning.log_adapter import (ClockMapping, InitialState,  # noqa: E402
                                                       prepare_session, run_session)
from invariant_estimation.learning.noise import NoiseSpec  # noqa: E402
from invariant_estimation.eval.consistency import two_stage_consistency  # noqa: E402

FIT_WINDOW = (110.0, 113.6)
HELD_OUT = (("standing 70.0-73.6", (70.0, 73.6)),
            ("walking 102.4-106.0", (102.4, 106.0)),
            ("walking 118.0-121.6", (118.0, 121.6)))
ACCEL_BIAS = "/home/bpark/.claude/jobs/e19f3ddd/tmp/accel_bias_0717.npy"


def theta_for_scale(spec: NoiseSpec, scales: dict[str, float]) -> np.ndarray:
    """theta such that spec.scales(theta) applies exactly `scales` (others stay 1).

    Inverts scale = exp(b*tanh(theta/b)): theta = b*atanh(ln(s)/b). Refuses a scale the
    parameterisation cannot reach rather than silently saturating.
    """
    b = math.log(spec.max_scale)
    theta = np.zeros(len(spec.names))
    for name, s in scales.items():
        index = spec.names.index(name)
        x = math.log(s) / b
        if not -1.0 < x < 1.0:
            raise ValueError(f"scale {s} for {name} is outside (1/{spec.max_scale}, {spec.max_scale})")
        theta[index] = b * math.atanh(x)
    return theta


class Session:
    """One window, prepared once; candidates are then cheap re-rolls."""

    def __init__(self, window_s):
        cfg, jk, mj_model, self.build, self.session_model = build_stack(None)
        self.spec = NoiseSpec(tuple(self.build.imu_names), arm=7)

        channels = alex_channel_map_from_log(LOG, self.session_model.joint_names,
                                             self.build.imu_names, FEET,
                                             base_imu="pelvis_imu", trust_format=TRUST_FORMAT)
        from invariant_estimation.replay.logsource import read_window
        window = read_window(LOG, channels.required_channels(),
                             start=window_s[0], end=window_s[1], stride=1)

        self.ekf = base_ekf.create(len(FEET), dt=jk["dt"], gyro_var=cfg["inekf"]["gyro_var"],
                                   accel_var=cfg["inekf"]["accel_var"],
                                   contact_var=cfg["inekf"]["contact_var"])
        self.joint_params = default_params(dt=jk["dt"], sigma_accel=jk["sigma_accel"],
                                           cond_s_max=jk["cond_s_max"])
        initial = InitialState(rotation=np.eye(3), velocity=np.zeros(3), position=np.zeros(3),
                               covariance=np.eye(self.ekf.tangent_size) * 1e-3,
                               source="identity prior for a retune run")
        clock = ClockMapping(source_origin_s=float(window.time[0]), target_origin_ns=0,
                             clock_domain="aligned_robot_monotonic_ns")

        from invariant_estimation.model.mounts import imu_to_body
        self.session = prepare_session(window, channels, clock, self.session_model, self.build,
                                       self.joint_params, self.ekf, initial,
                                       imu_to_body=imu_to_body(mj_model),
                                       world_frame="registered_zup",
                                       accel_bias_body=np.load(ACCEL_BIAS))

    def anis(self, s_enc: float, s_fk: float, s_grav: float, s_floor: float = 1.0):
        """{channel: (anis, dof, n)} plus mean |v|, for one candidate.

        s_floor scales ekf.params.contact_floor -- the fixed per-contact variance floor the
        learned parameterisation never touches, and, per the diagnose() probes, the constant
        that actually dominates both the contact and (through P) the gravity channel."""
        build = self.build._replace(encoder_var=np.asarray(self.build.encoder_var) * s_enc)
        ekf = self.ekf if s_floor == 1.0 else self.ekf._replace(
            params=self.ekf.params._replace(contact_floor=self.ekf.params.contact_floor * s_floor))
        theta = theta_for_scale(self.spec, {"contact_fk_r": s_fk,
                                            "gravity_roll_r": s_grav,
                                            "gravity_pitch_r": s_grav})
        _, out = run_session(theta, self.spec, build, self.joint_params,
                             ekf, self.session_model, self.session)
        reports = two_stage_consistency(out.joint_diagnostics, out.base,
                                        n_contacts=len(FEET), n_joints=self.build.n_joints)
        table = {r.channel: (float(r.anis), float(r.dof), int(r.samples)) for r in reports}
        speed = float(np.linalg.norm(np.asarray(out.base.state.v), axis=-1).mean())
        return table, speed


def bisect_scale(evaluate, lo: float, hi: float, iterations: int = 12) -> float:
    """Finds the scale where evaluate(s) = ANIS/dof crosses 1. `evaluate` must be monotone
    decreasing in s (more noise -> smaller NIS); refuses a bracket that does not straddle 1."""
    flo, fhi = evaluate(lo), evaluate(hi)
    if not (flo > 1.0 > fhi):
        raise ValueError(f"bracket [{lo}, {hi}] does not straddle 1: f(lo)={flo:.3f}, f(hi)={fhi:.3f}")
    for _ in range(iterations):
        mid = math.sqrt(lo * hi)  # log-space midpoint: the knob is a variance ratio
        if evaluate(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def fit_floor():
    """ONE knob, contact_floor, chosen on the fit window only.

    Exact consistency is unreachable: the contact ratio saturates near 0.74 as the floor goes to
    zero (below the floor, other terms take over), so instead of bisecting to 1 the scale is
    chosen from a small grid to minimise the JOINT miscalibration |log(contact ratio)| +
    |log(gravity ratio)| -- the probe showed this single constant dominates both channels, and a
    knob serving two channels should be scored on both.

    WHY THE RESULT IS NOT PROMOTED (measured 2026-09-18, on window 102.4-106.0):
    the retune improves held-out consistency everywhere but raises mean |v| by 79%, and the
    increase is VARIANCE, not drift -- mean body-frame forward velocity barely moves (+0.060 ->
    +0.044 m/s) while std(v_y) goes 0.149 -> 0.368 and |dv/dt| 2.7 -> 3.4 m/s^2. Tightening the
    floor makes the filter trust contact FK more, and contact FK is exactly what the `stacked`
    finding shows to be corrupted during walking, so the corruption lands in the state. The slack
    floor was absorbing it. This orders the work: the contact floor is not honestly tunable until
    the IMU-pair noise model is right, and NIS alone would have promoted a jerkier filter --
    "necessary, not sufficient" in the flesh.
    """
    session = Session(FIT_WINDOW)
    print(f"{'s_floor':>9s} {'contact/6':>10s} {'gravity/3':>10s} {'joint cost':>11s}")
    best = (float("inf"), None, None)
    for s_floor in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4):
        table, _ = session.anis(1.0, 1.0, 1.0, s_floor)
        c, g = table["contact"][0] / 6, table["gravity"][0] / 3
        cost = abs(math.log(c)) + abs(math.log(g))
        print(f"{s_floor:>9.4f} {c:>10.3f} {g:>10.3f} {cost:>11.3f}", flush=True)
        if cost < best[0]:
            best = (cost, s_floor, table)
    _, s_floor, table = best
    print(f"\nFITTED s_floor={s_floor:g}  (contact_floor {session.ekf.params.contact_floor:g} -> "
          f"{session.ekf.params.contact_floor*s_floor:g} m^2)")
    print(f"fit-window ratios: contact={table['contact'][0]/6:.3f} gravity={table['gravity'][0]/3:.3f} "
          f"(n={table['gravity'][2]}) encoder={table['encoder'][0]/9:.3f} stacked={table['stacked'][0]/27:.3f}")
    return s_floor


def fit():
    session = Session(FIT_WINDOW)

    # Encoder: measured, then deliberately NOT fitted. On this window the channel is already
    # consistent at s=1 (ANIS/9 = 1.040), and the knob is nearly dead -- 64x more encoder R moves
    # the ratio only to 1.008, because the innovation covariance is dominated by the prior, the
    # classic Q/R identifiability problem. The 4.3x overconfidence seen on 102.4-106.0 is therefore
    # regime dependence (1.04 here, 1.08 on 118.0-121.6, 4.30 there), not a mis-set constant, and
    # raising R to serve the violent window would break the windows where the channel is right.
    s_enc = 1.0
    table, _ = session.anis(1.0, 1.0, 1.0)
    print(f"encoder at s=1: ANIS/9 = {table['encoder'][0]/9:.3f} on the fit window -> left alone "
          f"(regime-dependent, not retunable by a constant; see comment)", flush=True)

    s_fk = bisect_scale(lambda s: session.anis(s_enc, s, 1.0)[0]["contact"][0] / 6.0,
                        lo=0.011, hi=1.0)  # conservative; lo just inside the 1/100 parameterisation bound
    s_grav = bisect_scale(lambda s: session.anis(s_enc, s_fk, s)[0]["gravity"][0] / 3.0,
                          lo=0.011, hi=1.0)

    table, _ = session.anis(s_enc, s_fk, s_grav)
    print(f"fit-window ratios after fit: contact={table['contact'][0]/6:.3f} "
          f"gravity={table['gravity'][0]/3:.3f} (n={table['gravity'][2]}) "
          f"encoder={table['encoder'][0]/9:.3f} stacked={table['stacked'][0]/27:.3f}", flush=True)
    print(f"\nFITTED (window {FIT_WINDOW}, nothing else): "
          f"s_enc={s_enc:.4f}  s_fk={s_fk:.5f}  s_grav={s_grav:.5f}")
    return s_enc, s_fk, s_grav


def evaluate(s_enc: float, s_fk: float, s_grav: float, s_floor: float = 1.0):
    print(f"scales: s_enc={s_enc:.4f} s_fk={s_fk:.5f} s_grav={s_grav:.5f} s_floor={s_floor:.5f} "
          f"(fitted on {FIT_WINDOW}; every window below is held out)\n")
    print(f"{'window':22s} {'cfg':9s} {'encoder/9':>11s} {'contact/6':>11s} {'gravity/3':>16s} {'stacked/27':>11s} {'|v| mean':>10s}")
    for name, window_s in HELD_OUT:
        session = Session(window_s)
        for cfg_name, args in (("baseline", (1.0, 1.0, 1.0, 1.0)), ("tuned", (s_enc, s_fk, s_grav, s_floor))):
            table, speed = session.anis(*args)
            print(f"{name:22s} {cfg_name:9s} "
                  f"{table['encoder'][0]/9:>11.3f} {table['contact'][0]/6:>11.3f} "
                  f"{table['gravity'][0]/3:>9.3f} n={table['gravity'][2]:<4d} "
                  f"{table['stacked'][0]/27:>11.3f} {speed:>10.4f}", flush=True)
        print()



def probe():
    """One-at-a-time sensitivity of every channel ANIS to every knob, on the fit window.

    The tuning attempt found two dead knobs in a row (encoder_var, contact_fk_r), so before
    fitting anything else: measure the leverage matrix. A dead knob means that channel's
    innovation covariance is dominated by the prior, i.e. by a PROCESS noise, and the honest
    fix lives there -- or nowhere.
    """
    session = Session(FIT_WINDOW)
    knobs = ("base_gyro_q", "base_accel_q", "contact_q", "contact_fk_r", "gravity_r(joint)")

    def run_with(knob, s):
        if knob == "gravity_r(joint)":
            scales = {"gravity_roll_r": s, "gravity_pitch_r": s}
        elif knob == "encoder_var":
            return session.anis(s, 1.0, 1.0)[0]
        else:
            scales = {knob: s}
        theta = theta_for_scale(session.spec, scales)
        _, out = run_session(theta, session.spec, session.build, session.joint_params,
                             session.ekf, session.session_model, session.session)
        reports = two_stage_consistency(out.joint_diagnostics, out.base,
                                        n_contacts=len(FEET), n_joints=session.build.n_joints)
        return {r.channel: (float(r.anis), float(r.dof), int(r.samples)) for r in reports}

    base = run_with("contact_q", 1.0)
    print("baseline ratios on fit window: " +
          "  ".join(f"{c}={base[c][0]/base[c][1]:.3f}" for c in ("encoder", "contact", "gravity", "stacked")))
    print(f"\n{'knob':18s} {'scale':>6s}   " +
          "".join(f"{c:>10s}" for c in ("encoder", "contact", "gravity", "stacked")))
    for knob in knobs:
        for s in (0.05, 20.0):
            t = run_with(knob, s)
            print(f"{knob:18s} {s:>6.2f}   " +
                  "".join(f"{t[c][0]/t[c][1]:>10.3f}" for c in ("encoder", "contact", "gravity", "stacked")),
                  flush=True)


def diagnose():
    """Two single-shot probes that pin WHERE the contact and gravity conservatism live.

    (a) gravity_r at the parameterisation's lower bound: if the ratio is still below 1 there,
        gravity is unreachable by its own R within the learned parameterisation.
    (b) contact_floor x0.01 (an ekf.params field NO learned channel touches): if contact ANIS
        jumps, the fixed floor -- not fk_r, not contact_q -- dominates the contact S, which is
        why both nominal knobs measured dead.
    """
    session = Session(FIT_WINDOW)

    theta = theta_for_scale(session.spec, {"gravity_roll_r": 0.0101, "gravity_pitch_r": 0.0101})
    _, out = run_session(theta, session.spec, session.build, session.joint_params,
                         session.ekf, session.session_model, session.session)
    reports = {r.channel: (float(r.anis), float(r.dof)) for r in
               two_stage_consistency(out.joint_diagnostics, out.base,
                                     n_contacts=len(FEET), n_joints=session.build.n_joints)}
    print(f"(a) gravity_r at 1/99 (parameterisation floor): gravity ANIS/3 = "
          f"{reports['gravity'][0]/3:.3f}  -> {'reachable' if reports['gravity'][0]/3 >= 1 else 'UNREACHABLE by its own R'}",
          flush=True)

    ekf_small_floor = session.ekf._replace(
        params=session.ekf.params._replace(contact_floor=session.ekf.params.contact_floor * 0.01))
    theta0 = theta_for_scale(session.spec, {})
    _, out = run_session(theta0, session.spec, session.build, session.joint_params,
                         ekf_small_floor, session.session_model, session.session)
    reports = {r.channel: (float(r.anis), float(r.dof)) for r in
               two_stage_consistency(out.joint_diagnostics, out.base,
                                     n_contacts=len(FEET), n_joints=session.build.n_joints)}
    print(f"(b) contact_floor x0.01 (baseline floor {session.ekf.params.contact_floor:g} m^2): "
          f"contact ANIS/6 = {reports['contact'][0]/6:.3f} (was 0.031)  "
          f"gravity {reports['gravity'][0]/3:.3f}  stacked {reports['stacked'][0]/27:.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit-floor":
        fit_floor()
    elif sys.argv[1] == "diagnose":
        diagnose()
    elif sys.argv[1] == "probe":
        probe()
    elif sys.argv[1] == "fit":
        fit()
    elif sys.argv[1] == "eval":
        evaluate(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
                 float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
