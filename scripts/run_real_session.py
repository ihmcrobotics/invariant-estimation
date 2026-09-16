"""Assemble a real Alex session from a hardware log and run it through both filters.

Exercises the whole log boundary on real data -- decoder, ChannelMap, LogWindow,
prepare_session, two-stage rollout -- which is the one thing synthetic fixtures
cannot do. There is no mocap here, so no loss: that still needs a capture.

Run it::

    python scripts/run_real_session.py [/opt/ihmc/LogData/incoming/<log>]

Every constant below that is not read from the log or from filter_cfg.yaml is a
choice this script had to make, and each one is commented with why. They are the
list of things a real study has to settle properly:

* the IMU-pair topology (mirrored from AlexStateEstimatorParameters, not read
  from anywhere machine-readable);
* the sole sites, which the URDF does not carry -- placed at the foot link
  origin here, which is not the real sole offset;
* the sole sites, whose true offset the URDF does not carry (above);
* the window, chosen to begin on a genuinely stationary stretch and to stop
  before a logged controller stall;
* the nominal specific force at rest, which assumes a level pelvis -- see the
  caveat at that argument.
"""
import sys

import jax.numpy as jnp
import mujoco
import numpy as np

from invariant_estimation.config import load_config
from invariant_estimation.inEKF import ekf as base_ekf
from invariant_estimation.jointKF.build import KinematicTree, build_joint_kf
from invariant_estimation.jointKF.state import default_params
from invariant_estimation.learning.channels import alex_channel_map_from_log
from invariant_estimation.learning.log_adapter import ClockMapping, InitialState, prepare_session, run_session
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.learning.session_model import MjxSessionModel
from invariant_estimation.model.mjx_model import MjxModel
from invariant_estimation.model.mounts import describe as describe_mount, imu_to_body
from invariant_estimation.model.urdf2mjcf import convert_log_model
from invariant_estimation.replay.logsource import read_window

LOG = sys.argv[1] if len(sys.argv) > 1 else \
    "/opt/ihmc/LogData/incoming/20260916_101745_Alex001UnifiedControlProcess"

FILTERED = ("SPINE_Z", "LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
            "RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y")
UNFILTERED = ("LEFT_ANKLE_Y", "LEFT_ANKLE_X", "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X")
# Pelvis-centred star, as AlexStateEstimatorParameters configures it.
IMUS = ("pelvis_imu", "torso_imu",
        "left_hip_x_imu", "left_thigh_imu", "left_shin_imu",
        "right_hip_x_imu", "right_thigh_imu", "right_shin_imu")
# spine, then per side: hip-X, hip-ZY, knee -- the chain that makes the knees observable.
PAIRS = ((0, 1), (0, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7))
# Named as the ROBOT names its feet, not as a convenience: these flow into the artifact's
# contact_names and into the log's own trust channels (isLEFT_FOOTFootTrusted).
FEET = ("LEFT_FOOT", "RIGHT_FOOT")


def kinematic_tree(mj_model, spec, site_names):
    """A `KinematicTree` from a MuJoCo model — the adapter does not build one yet."""
    hinges = [i for i in range(mj_model.njnt)
              if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinges]
    free = [i for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    base_dofs = np.arange(mj_model.jnt_dofadr[free[0]], mj_model.jnt_dofadr[free[0]] + 6) if free else np.empty(0, int)
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


def main():
    cfg = load_config()
    jk = cfg["joint_kf"]

    print(f"log: {LOG}")
    # The converter emits IMU sites automatically but not contact frames, so the sole sites
    # the InEKF anchors on have to be requested explicitly. Placed at the foot link origin
    # here -- a real study needs the actual sole offset, which the URDF does not carry.
    spec = convert_log_model(LOG, rotor_inertia=jk["rotor_inertia"],
                             rotor_inertia_default=jk["rotor_inertia_default"],
                             extra_sites={"LEFT_FOOT": "LEFT_FOOT", "RIGHT_FOOT": "RIGHT_FOOT",
                                          # The pelvis LINK origin -- the frame mocap registers and
                                          # the Java estimator reports. Without it the only pelvis-ish
                                          # site is the IMU's own, 90 deg and ~12 cm away.
                                          "pelvis": "PELVIS_LINK"})
    mj_model = mujoco.MjModel.from_xml_string(spec.mjcf)
    print(f"model: {mj_model.njnt} joints, {mj_model.nsite} sites")
    print(f"mount: {describe_mount(mj_model)}")

    sites = tuple(IMUS) + FEET + ("pelvis",)
    tree = kinematic_tree(mj_model, spec, sites)
    build = build_joint_kf(tree, imu_sites=list(IMUS), pairs=list(PAIRS),
                           foot_sites=list(FEET), cfg=jk)
    print(f"build: {build.n_joints} filtered joints {build.joint_names}")
    print(f"       {len(build.imu_names)} IMUs, {build.n_anchors} anchors")

    # joint_names must be the FILTERED set: MjxSessionModel requires the model's joint order
    # to equal the build's, and the default takes every hinge in the robot.
    # pairs are SITE indices; the IMU sites lead `sites`, so they coincide with PAIRS here.
    model = MjxModel.from_xml_string(spec.mjcf, site_names=sites, pairs=list(PAIRS),
                                     joint_names=build.joint_names)
    # body_site is the pelvis LINK, so the filter estimates the frame the study compares against.
    session_model = MjxSessionModel(model, build, body_site="pelvis", contact_sites=FEET)

    channels = alex_channel_map_from_log(
        LOG, session_model.joint_names, build.imu_names, FEET, base_imu="pelvis_imu")
    print(f"channels: {len(channels.required_channels())} required")

    # A window that BEGINS stationary: the accel-bias calibration needs a quiet prefix, and
    # the adapter checks the declared window really is quiet rather than taking our word.
    # t=73.5 s is the quietest stretch in this log (max |gyro| 0.0055 rad/s).
    window = read_window(LOG, channels.required_channels(), start=73.5, end=74.25, stride=1)
    print(f"window: {len(window.time)} ticks, dt={window.dt}")

    ekf = base_ekf.create(len(FEET), dt=jk["dt"], gyro_var=cfg["inekf"]["gyro_var"],
                          accel_var=cfg["inekf"]["accel_var"],
                          contact_var=cfg["inekf"]["contact_var"])
    joint_params = default_params(dt=jk["dt"], sigma_accel=jk["sigma_accel"],
                                  cond_s_max=jk["cond_s_max"])
    initial = InitialState(rotation=np.eye(3), velocity=np.zeros(3), position=np.zeros(3),
                           covariance=np.eye(ekf.tangent_size) * 1e-3,
                           source="identity prior for a smoke run")
    clock = ClockMapping(source_origin_s=float(window.time[0]), target_origin_ns=0,
                         clock_domain="aligned_robot_monotonic_ns")

    session = prepare_session(
        window, channels, clock, session_model, build, joint_params, ekf, initial,
        # Identity is correct ONLY because body_site is the pelvis IMU site itself, so the
        # filter's "body" frame IS the IMU frame. That is self-consistent but it is not the
        # frame the study wants: mocap registers the pelvis, and the Java estimator reports
        # the pelvis, while this estimates a frame yawed 90 degrees and offset ~12 cm from it
        # (mounts.imu_to_body reads that transform out of the robot's own description).
        #
        # Estimating the pelvis instead needs body_site on PELVIS_LINK with imu_to_body =
        # mounts.imu_to_body(mj_model). MjxSessionModel rejects that today: it requires the
        # base IMU site and the body site to share a body id, and they are separate bodies
        # joined by a fixed joint. Loosening that check to "rigidly attached" is the fix, and
        # it is a deliberate decision rather than something to slip in here.
        # Read from the robot's own description rather than assumed. prepare_session cross-checks
        # it against the model's measurement frames, so a wrong value fails loudly here.
        imu_to_body=imu_to_body(mj_model), world_frame="registered_zup",
        stationary_window=(0, 200),
        # The joint-velocity limit is raised from 0.05 because it is checked over ALL 49 model
        # joints, and NECK_Y is panning at 0.070 rad/s while the legs and pelvis are still.
        # What the accel-bias calibration actually needs is a stationary PELVIS, and the two
        # checks that measure that directly pass with room to spare: base gyro 0.011 (limit
        # 0.15) and accel std 0.029 (limit 0.20). Raised deliberately and narrowly, not
        # switched off.
        stationary_limits=(0.15, 0.10, 0.2),
        # What a bias-free IMU should read at rest in the body frame. Measured mean on this
        # window is [-0.057, 0.381, 9.589]; the difference from this nominal IS the bias the
        # calibration is being asked to find, so it must not be fitted from the same data.
        # CAVEAT: asserting the pelvis is exactly level at rest. It is not, and this
        # calibration cannot tell a real accelerometer bias from a small tilt -- the y term
        # that comes out (0.38 m/s^2) is about a 2 degree lean, not a sensor offset. A real
        # study needs the attitude at rest, which is what the gravity-leveling update
        # estimates; feeding a nominal here conflates the two.
        stationary_specific_force_body=np.array([0.0, 0.0, 9.81]),
    )
    print(f"session: accel bias {np.asarray(session.accel_bias_body)}")

    spec_noise = NoiseSpec(tuple(build.imu_names), arm=7)
    _, out = run_session(spec_noise.initial_theta(), spec_noise, build, joint_params,
                         ekf, session_model, session)
    v = np.asarray(out.base.state.v)
    print(f"ROLLOUT OK: {v.shape[0]} ticks, |v| mean {np.linalg.norm(v, axis=-1).mean():.4f} m/s")
    print(f"           finite: {np.isfinite(v).all()}")


if __name__ == "__main__":
    main()
