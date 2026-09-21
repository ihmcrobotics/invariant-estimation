"""A/B: does moving the contact anchor from the foot link origin to the sole change anything?

RESULT (2026-09-17, on the recovered 2026-07-17 Alex001 overground-walking log):

              |dv| A vs B, mean      contact NIS  A / B
    standing   0.000148 m/s           0.000 / 0.000
    walking    0.009475 m/s           0.479 / 0.495

The difference is **64x larger walking than standing**, which is exactly the prediction: the
anchor d_i is a state variable, so a constant offset is absorbed and only its time derivative --
foot rotation in stance -- reaches the estimate.

It does NOT show that the sole is the better anchor. Both arms sit equally far below the contact
update's degrees of freedom, so NIS does not pick a winner, and that is expected: during roll-over
the instantaneous axis is the toe or heel edge, so neither candidate is the right point. Deciding
between them needs mocap.

Separately worth noting: contact NIS ~0.5 against a dof of 3-6 means the filter is markedly
CONSERVATIVE about its contact updates -- its assumed measurement noise is far larger than the
innovations it actually sees (0.24 mm). An earlier version of this note pre-registered the learned
contact_fk_r channel as the fix; the leverage probe in scripts/retune_constant_noise.py REFUTED
that: contact ANIS is insensitive to contact_fk_r (and to contact_q) across their whole range.
The dominating constant is ekf.params.contact_floor, which no learned channel touches.


The prediction, from the contact residual r = R.y - (d_i - p) with the anchor d_i a STATE
variable: a constant offset is absorbed, so on a standing window the two runs should be
indistinguishable, and any difference at all must come from foot ROTATION during stance --
which is why this is run on a walking window.

Reports contact-update NIS (self-contained, needs no mocap) and the base velocity, for the
anchor at the foot link origin (A) versus at the sole (B).
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

from invariant_estimation.config import load_config
from invariant_estimation.inEKF import ekf as base_ekf
from invariant_estimation.jointKF.build import KinematicTree, build_joint_kf
from invariant_estimation.jointKF.state import default_params
from invariant_estimation.learning.channels import alex_channel_map_from_log
from invariant_estimation.learning.log_adapter import (ClockMapping, InitialState,
                                                       prepare_session, run_session)
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.learning.session_model import MjxSessionModel
from invariant_estimation.model.mjx_model import MjxModel
from invariant_estimation.model.mounts import imu_to_body
from invariant_estimation.model.urdf2mjcf import convert_log_model
from invariant_estimation.replay.logsource import read_window

# A log directory. The 2026-07-17 walking log needs its index rebuilt first (see
# scripts/rebuild_log_index.py) because @Recycle kept the .bsz but not the .dat.
LOG = os.environ.get("ALEX_AB_LOG", "/opt/ihmc/LogData/incoming/20260717_160126_Alex001UnifiedControlProcess")

IMUS = ("pelvis_imu", "torso_imu",
        "left_hip_x_imu", "left_thigh_imu", "left_shin_imu",
        "right_hip_x_imu", "right_thigh_imu", "right_shin_imu")
PAIRS = ((0, 1), (0, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7))
FEET = ("LEFT_FOOT", "RIGHT_FOOT")

# AlexV2PhysicalProperties: (footLength/2 - footBack, 0, -ankleHeight); the bottom face of
# model.sdf's foot collision box agrees to the millimetre.
SOLE = (0.053, 0.0, -0.055)

# This July build names the per-foot contact flag differently from the September one
# (isLEFT_FOOTFootTrusted). WrenchMatrixHelperInContact exists in BOTH builds, which is why it
# is the one used here. It is the controller's contact state rather than the estimator's trust
# decision -- a different signal, but the same one in both arms of this A/B, so the comparison
# is unaffected by the choice.
TRUST_FORMAT = "{contact}WrenchMatrixHelperInContact"


def kinematic_tree(mj_model, spec, site_names):
    """Copied from scripts/run_real_session.py -- the adapter does not build one yet."""
    hinges = [i for i in range(mj_model.njnt)
              if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinges]
    free = [i for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    base_dofs = (np.arange(mj_model.jnt_dofadr[free[0]], mj_model.jnt_dofadr[free[0]] + 6)
                 if free else np.empty(0, int))
    site_body = {}
    for name in site_names:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"model has no site {name!r}")
        site_body[name] = int(mj_model.site_bodyid[sid])
    return KinematicTree(
        joint_names=tuple(names),
        joint_body=np.array([mj_model.jnt_bodyid[i] for i in hinges], dtype=int),
        body_parent=np.array(mj_model.body_parentid, dtype=int),
        joint_dof=np.array([mj_model.jnt_dofadr[i] for i in hinges], dtype=int),
        base_dofs=base_dofs,
        site_body=site_body,
        tau_max=np.array([spec.effort_limits.get(n, 0.0) for n in names], dtype=float),
    )


def build_stack(sole_offset):
    cfg = load_config()
    jk = cfg["joint_kf"]

    extra = {"pelvis": "PELVIS_LINK"}
    for foot in FEET:
        extra[foot] = foot if sole_offset is None else (foot, sole_offset)

    spec = convert_log_model(LOG, rotor_inertia=jk["rotor_inertia"],
                             rotor_inertia_default=jk["rotor_inertia_default"],
                             extra_sites=extra)
    mj_model = mujoco.MjModel.from_xml_string(spec.mjcf)

    sites = tuple(IMUS) + FEET + ("pelvis",)
    tree = kinematic_tree(mj_model, spec, sites)
    build = build_joint_kf(tree, imu_sites=list(IMUS), pairs=list(PAIRS),
                           foot_sites=list(FEET), cfg=jk)
    model = MjxModel.from_xml_string(spec.mjcf, site_names=sites, pairs=list(PAIRS),
                                     joint_names=build.joint_names)
    session_model = MjxSessionModel(model, build, body_site="pelvis", contact_sites=FEET)
    return cfg, jk, mj_model, build, session_model


def run(label, sole_offset, window_s, accel_bias):
    cfg, jk, mj_model, build, session_model = build_stack(sole_offset)

    channels = alex_channel_map_from_log(LOG, session_model.joint_names, build.imu_names,
                                         FEET, base_imu="pelvis_imu",
                                         trust_format=TRUST_FORMAT)
    window = read_window(LOG, channels.required_channels(),
                         start=window_s[0], end=window_s[1], stride=1)

    ekf = base_ekf.create(len(FEET), dt=jk["dt"], gyro_var=cfg["inekf"]["gyro_var"],
                          accel_var=cfg["inekf"]["accel_var"],
                          contact_var=cfg["inekf"]["contact_var"])
    joint_params = default_params(dt=jk["dt"], sigma_accel=jk["sigma_accel"],
                                  cond_s_max=jk["cond_s_max"])
    initial = InitialState(rotation=np.eye(3), velocity=np.zeros(3), position=np.zeros(3),
                           covariance=np.eye(ekf.tangent_size) * 1e-3,
                           source="identity prior for an A/B run")
    clock = ClockMapping(source_origin_s=float(window.time[0]), target_origin_ns=0,
                         clock_domain="aligned_robot_monotonic_ns")

    # The accel bias is SUPPLIED rather than calibrated here. No clean run in this log's walking
    # region has a quiet prefix (see the timestamp-regression note), and loosening the stationary
    # check until walking data passed it would fabricate a calibration. This bias was measured
    # separately, from a genuinely quiet 25000-tick stretch of the SAME log (|gyro| mean 0.005
    # rad/s), and the identical value goes into both arms -- so whatever error it carries is a
    # common term that cancels in the A/B difference, which is the quantity this run reports.
    session = prepare_session(
        window, channels, clock, session_model, build, joint_params, ekf, initial,
        imu_to_body=imu_to_body(mj_model), world_frame="registered_zup",
        accel_bias_body=accel_bias,
    )

    spec_noise = NoiseSpec(tuple(build.imu_names), arm=7)
    _, out = run_session(spec_noise.initial_theta(), spec_noise, build, joint_params,
                         ekf, session_model, session)
    return out


def summarise(label, out):
    v = np.asarray(out.base.state.v)
    speed = np.linalg.norm(v, axis=-1)

    cd = out.base.contact_diagnostics
    applied = np.asarray(cd.applied).astype(bool)
    nis = np.asarray(cd.nis)
    inno = np.linalg.norm(np.asarray(out.base.contact_innovation), axis=-1)

    ok = applied & np.isfinite(nis)
    nis_mean = nis[ok].mean() if ok.any() else float("nan")
    inno_mean = inno[ok].mean() if ok.any() else float("nan")

    print(f"  {label:26s} |v| mean={speed.mean():.4f}  "
          f"contact NIS mean={nis_mean:7.3f}  |innovation| mean={inno_mean:.5f} m  "
          f"applied={int(ok.sum())}/{len(nis)}")
    return v, nis, ok, inno


if __name__ == "__main__":
    window_s = (float(sys.argv[1]), float(sys.argv[2]))
    bias = np.load(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else ""

    print(f"{tag}  ticks {int(window_s[0]*1000)}..{int(window_s[1]*1000)}")

    res = {}
    for label, offset in (("A anchor at link origin", None), ("B anchor at the sole", SOLE)):
        res[label] = summarise(label, run(label, offset, window_s, bias))

    (va, na, oa, ia), (vb, nb, ob, ib) = res.values()
    n = min(len(va), len(vb))
    d = np.linalg.norm(va[:n] - vb[:n], axis=-1)
    both = oa[:n] & ob[:n]
    print(f"  {'-> difference':26s} |dv| mean={d.mean():.6f} max={d.max():.6f} m/s "
          f"({d.mean()/np.linalg.norm(va[:n],axis=-1).mean()*100:.2f}% of |v|)")
    if both.any():
        print(f"  {'':26s} NIS  A={na[:n][both].mean():.3f}  B={nb[:n][both].mean():.3f}  "
              f"(closer to dof is the more honest covariance)")
        print(f"  {'':26s} |innovation|  A={ia[:n][both].mean():.5f}  B={ib[:n][both].mean():.5f} m")
    print()
