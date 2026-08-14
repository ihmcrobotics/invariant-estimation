"""run_policy.py — IHMC RL policies in a standalone MuJoCo sim, driven live.

Runs an exported ONNX policy directly via `onnxruntime` (no JAX, no port) in a plain MuJoCo loop
with ground-truth observations. Independent of the estimator.

    uv run python run_policy.py                       # viewer, standing demo
    uv run python run_policy.py --policy baseline     # the 29-joint walking policy
    uv run python run_policy.py --policy baseline --wasd   # own-window WASD/Space/Shift, no controller
    uv run python run_policy.py --headless --ticks 400

Live command, in order of preference: an Xbox-style gamepad on /dev/input/js0 (`Gamepad`), the
viewer's NUMERIC KEYPAD, or letters typed at the terminal (`stdin_commands`). Letters cannot be keys
of the default (passive) viewer -- see the BINDINGS section. The `--wasd` mode (`run_free_viewer`)
sidesteps that by owning its own GLFW window, giving true key press/release for hold-to-move control.

Every magic number here is matched to SCS2's MuJoCo backend, the reference implementation that
works. **`EXPERIMENTS.md` is the full log** of which source each one comes from and of every
hypothesis already tested; `RUNNING.md` documents the controls. Read both before changing any of it.

The one trap worth repeating here: `base_ang_vel` must come from `mjOBJ_XBODY`, not `mjOBJ_BODY`.
See `build_obs`.
"""
import argparse
import glob
import os
import struct
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import onnxruntime as ort
import yaml

from invariant_estimation.pipeline import main_estimator as me
from invariant_estimation.model.urdf2mjcf import _rpy_to_quat

# Everything this script reads is VENDORED under `assets/` (~18 MB), so a fresh clone runs with no
# sibling repos and no files outside the checkout. The copies are byte-identical to their sources:
#
#   assets/alex_with_imus.urdf          <- ~/Documents/alex_with_imus.urdf
#   assets/rl_models/                   <- alex/src/main/resources/rl_models/
#   assets/alex_virtual_description/    <- ihmc-alex-sdk/alex-models/alex_virtual_description/
#                                          (only the 29 visual meshes this URDF references, 16.5 MB
#                                           of its 43 MB; the 7 ability-hand meshes are omitted --
#                                           `cycloid_forearm_urdf` deletes those links anyway)
#
# Each has an env override, for running against a live working copy (a retrained policy, an edited
# URDF) without touching the source: ALEX_URDF, ALEX_RL_MODELS, ALEX_MESHDIR. `ALEX_URDF` is the
# same variable the test suite uses.
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

URDF = os.environ.get("ALEX_URDF", os.path.join(ASSETS, "alex_with_imus.urdf"))
RL_MODELS = os.environ.get("ALEX_RL_MODELS", os.path.join(ASSETS, "rl_models"))
MESHDIR = os.environ.get("ALEX_MESHDIR", os.path.join(ASSETS, "alex_virtual_description"))

# 200 Hz physics / 50 Hz control, matching IsaacLab's SIM_DT = 0.005 and CONTROL_DT = 0.02. The
# policy is queried at exactly the rate it was trained at; changing DT without changing DECIMATION
# changes the CONTROL rate, which silently rescales how far the robot travels per tick. (This was
# briefly left at 0.001 during a rate sweep -- 250 Hz control -- which reads as a walking
# regression if you measure travel per control tick rather than per second.)
DT = 0.005             # physics timestep
DECIMATION = 4         # physics steps per control tick -> 50 Hz

# `alex_with_imus.urdf` is the FULL_ROBOT_ABILITY_HANDS assembly; the policies were trained on
# AlexV2Version.CYCLOID_FOREARMS. Dropping the two hand-adapter subtrees reproduces the Java cycloid
# model exactly (49 links, 90.539488 kg, identical inertia tensor on every link).
HAND_ADAPTER_JOINTS = ("LEFT_ABILITY_HAND_ADAPTER", "RIGHT_ABILITY_HAND_ADAPTER")

# MujocoSimulationParameters defaults, as AlexRLSimulation instantiates them.
CONTACT = dict(condim="4", friction="1 0.05 0.01",
               solref="0.02 1", solimp="0.9 0.99 0.0007 0.5 2")
# Robot geoms test only against terrain, so the coarse foot boxes cannot catch each other in swing.
ROBOT_GROUP = dict(contype="1", conaffinity="2")
TERRAIN_GROUP = dict(contype="2", conaffinity="1")

# The collision set SCS2 actually emits, from `AlexSimulationCollisionModel` — NOT the URDF
# `<collision>` tags, whose foot box is 29% narrower. MuJoCo sizes: box = half-extents,
# capsule = (radius, half-length). Poses are in the ankle-roll frame = the `*_FOOT` body frame.
_FOOT_BOX_SIZE = " ".join(repr(v) for v in me.ALEX_FOOT_BOX_HALF)
_FOOT_BOX_POS = " ".join(repr(v) for v in me.ALEX_FOOT_BOX_CENTER)

SCS2_COLLISION_GEOMS = (
    ("PELVIS_LINK",          "capsule", "0.135 0.025",      "-0.06 0 -0.02",
     "0.70710678118654746 -0.70710678118654768 0 0"),
    # Half-extents/center come from `main_estimator`, which is also where the N=8
    # corner FK offsets are derived from -- one constant, so a geom change moves
    # the estimator's corners with it instead of silently desyncing them.
    ("LEFT_FOOT",            "box",     _FOOT_BOX_SIZE, _FOOT_BOX_POS, "1 0 0 0"),
    ("RIGHT_FOOT",           "box",     _FOOT_BOX_SIZE, _FOOT_BOX_POS, "1 0 0 0"),
    ("TORSO_LINK",           "capsule", "0.1 0.05",         "-0.01 0 0.22",  "1 0 0 0"),
    ("LEFT_GRIPPER_Z_LINK",  "capsule", "0.06 0.03",        "0 0 0",         "1 0 0 0"),
    ("RIGHT_GRIPPER_Z_LINK", "capsule", "0.06 0.03",        "0 0 0",         "1 0 0 0"),
    ("HEAD_LINK",            "capsule", "0.15 0.04", "0.05203239 0.01559991 0.06487138", "1 0 0 0"),
)
FOOT_GEOMS = ("LEFT_FOOT_collision_0", "RIGHT_FOOT_collision_0")

ANKLE_HEIGHT = 0.072   # AlexV1PhysicalProperties: sole plane below the `*_FOOT` (ankle-roll) frame
GO_HOME_DURATION = 0.5  # RLHeightManager: seconds for the height command to reach its target

# `base_height` is the TARGET of the height ramp (RLHeightManager's `homeHeight`): 0.89 is Java's
# default for the walking policies; the standing demo wants 0.75.
# `height_range` is the band the policy saw in training and is what the triggers clamp to. For the
# walking policies that is `AlexCommandsCfg.base_height.ranges.lin_pos_z = (0.83, 0.93)`; for the
# standing demo it is empirical (holds at 0.75, falls at 0.70 and 0.79).
POLICIES = {
    "standing": {"cfg": "20251219_standing18_def.yaml", "onnx": "20251219_standing18.onnx",
                 "base_height": 0.75, "height_range": (0.72, 0.78)},
    "baseline": {"cfg": "2026-07-10_baseline/policy_cfg.yaml", "onnx": "2026-07-10_baseline/policy.onnx",
                 "base_height": 0.89, "height_range": (0.83, 0.93)},
    "forearms": {"cfg": "2026-04-13_002_forearms/policy_cfg.yaml",
                 "onnx": "2026-04-13_002_forearms/policy.onnx",
                 "base_height": 0.89, "height_range": (0.83, 0.93)},
}


def _read_cfg(rel):
    with open(f"{RL_MODELS}/{rel}") as f:
        return yaml.safe_load(f)


# Gains/home for joints a policy does not drive (wrists/grippers for the 23-joint standing policy),
# so they hold home instead of flopping.
_FALLBACK = {p["name"]: p for p in _read_cfg(POLICIES["baseline"]["cfg"])["jointParameters"]}


def _ort_session(path):
    """Build the policy's ONNX session PINNED TO ONE NON-SPINNING THREAD.

    ORT's defaults are tuned for serving a big model on an idle box: one intra-op thread per
    physical core, and after each `Run` those threads *spin* rather than sleep so they are ready
    for the next call.  Both defaults are actively harmful here.  The policy is a tiny MLP -- one
    thread computes it faster than N can be synchronised -- and this is not an idle box: XLA runs
    the two filters on its own pool between control ticks, so the spinning ORT threads sit on
    exactly the cores the estimator needs.  Measured, this 20-thread box: a default session adds
    **9 threads**, a pinned one adds **0**.

    THIS IS THE WHOLE REASON THE LOOP NOW KEEPS REAL TIME.  Interleaved A/B, walking, with an
    offscreen render per tick, one session per process:

        default                     p50 24.0 / 29.7 ms   p90 34.7 / 41.5   -> 0.83x / 0.67x real time
        pinned, 1 thread, no spin   p50 17.8 / 16.1 ms   p90 31.5 / 24.6   -> 1.12x / 1.24x real time

    ~1.6x on the median, and it carries the viewer across the real-time line on its own -- the
    "viewer runs at 0.6x" problem was this, not the estimator.  (An earlier planning estimate had
    predicted a tail-only win with a noisy median; on the full loop the median moves robustly too.)

    One session per process is the only valid way to measure this: swapping sessions mid-run
    leaves the first pool alive and measures *more* threads, not fewer, which inverts the result.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policy(name):
    """Read a policy's `policy_cfg.yaml` (obs list, joint order, per-joint gains) + its ONNX."""
    spec = POLICIES[name]
    cfg = _read_cfg(spec["cfg"])
    jp = cfg["jointParameters"]
    return {
        "name": name,
        "order": [p["name"] for p in jp],          # policy joint order (IsaacLab breadth-first)
        "home": {p["name"]: float(p["homePosition"]) for p in jp},
        "kp": {p["name"]: float(p["kp"]) for p in jp},
        "kd": {p["name"]: float(p["kd"]) for p in jp},
        "tau": {p["name"]: float(p["maxEffort"]) for p in jp},
        "obs_terms": cfg["observations"],
        "action_scale": float(cfg.get("actionScale", 0.3)),
        "input_size": int(cfg["inputSize"]),
        "sess": _ort_session(f"{RL_MODELS}/{spec['onnx']}"),
        "base_height": float(spec["base_height"]),
        "height_range": tuple(spec["height_range"]),
    }


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
def cycloid_forearm_urdf(urdf_path):
    """Write a hands-free copy of `urdf_path` and return its path (no-op if already hands-free)."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    if not any(a in joints for a in HAND_ADAPTER_JOINTS):
        return urdf_path
    links = {lk.get("name"): lk for lk in root.findall("link")}
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(
            (j.get("name"), j.find("child").get("link")))

    dead_j, dead_l = set(), set()
    stack = [(a, joints[a].find("child").get("link")) for a in HAND_ADAPTER_JOINTS if a in joints]
    while stack:
        jn, ln = stack.pop()
        dead_j.add(jn); dead_l.add(ln)
        stack += children.get(ln, [])
    for n in dead_j:
        root.remove(joints[n])
    for n in dead_l:
        root.remove(links[n])

    out = os.path.join(tempfile.gettempdir(),
                       os.path.basename(urdf_path).replace(".urdf", "_cycloid_forearms.urdf"))
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


def _sub(parent, tag, **attrs):
    """`ET.SubElement` plus attributes, stringified."""
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        e.set(k, str(v))
    return e


def _geom(parent, name, contact_group, **attrs):
    """A geom carrying SCS2's contact parameters and one of the ROBOT/TERRAIN collision groups."""
    return _sub(parent, "geom", name=name, **{**attrs, **contact_group, **CONTACT})


def _asset(root):
    """The single `<asset>` element, created on first use (MuJoCo merges duplicates, we don't)."""
    a = root.find("asset")
    return a if a is not None else ET.SubElement(root, "asset")


# Cosmetic only -- no geom here carries collision, mass or inertia, so the dynamics are identical
# with or without `_add_scene_look`.
SKY_TOP, SKY_BOTTOM = "0.3 0.5 0.7", "0 0 0"          # MuJoCo's default gradient skybox
TILE_LIGHT, TILE_DARK = "0.2 0.3 0.4", "0.1 0.2 0.3"  # ... and its blue/dark-blue checker floor
ROBOT_RGBA = "0.09 0.09 0.10 1"                       # Alex in black


def _add_scene_look(root):
    """MuJoCo's stock studio look: gradient skybox, blue checkered floor, black robot.

    Purely for legibility on video -- against a flat grey floor the feet have nothing to swing
    past, so a walk reads as a hover. The tiles give the stride a scale (each is 1 m).
    """
    asset = _asset(root)
    _sub(asset, "texture", type="skybox", builtin="gradient", rgb1=SKY_TOP, rgb2=SKY_BOTTOM,
         width="512", height="3072")
    _sub(asset, "texture", type="2d", name="groundplane", builtin="checker", mark="edge",
         rgb1=TILE_LIGHT, rgb2=TILE_DARK, markrgb="0.8 0.8 0.8", width="300", height="300")
    # texrepeat is in metres with texuniform, and the checker is 2x2 -- so 1 1 makes 0.5 m tiles.
    _sub(asset, "material", name="groundplane", texture="groundplane", texuniform="true",
         texrepeat="1 1", reflectance="0.2")
    # A black robot needs a specular highlight to keep any shape at all against a dark floor.
    _sub(asset, "material", name="robot", rgba=ROBOT_RGBA, specular="0.6", shininess="0.6")

    # The URDF ships no lights, so without these the scene is lit by the headlight alone and the
    # robot is a silhouette. The directional light also gives it a shadow to stand on.
    _sub(root.find("worldbody"), "light", pos="0 0 4", dir="0 0 -1", directional="true",
         diffuse="0.55 0.55 0.55", specular="0.25 0.25 0.25", castshadow="true")

    vis = _sub(root, "visual")
    _sub(vis, "headlight", diffuse="0.75 0.75 0.75", ambient="0.45 0.45 0.45",
         specular="0.2 0.2 0.2")
    _sub(vis, "rgba", haze="0.15 0.25 0.35 1")
    # `offwidth/offheight` size MuJoCo's offscreen buffer: without them `--video` is capped at
    # 640x480 no matter what resolution is asked for.
    _sub(vis, "global", azimuth="140", elevation="-20", offwidth="1920", offheight="1080")
    _sub(vis, "quality", shadowsize="4096")


def _add_visual_meshes(root, urdf_path):
    """Cosmetic: the URDF's visual meshes as non-colliding geoms."""
    urdf = ET.parse(urdf_path).getroot()
    root.find("compiler").set("meshdir", MESHDIR)
    asset = _asset(root)
    bodies = {b.get("name"): b for b in root.iter("body")}
    seen = {}
    for link in urdf.findall("link"):
        body, vis = bodies.get(link.get("name")), link.find("visual")
        mesh = vis.find("geometry/mesh") if vis is not None else None
        if body is None or mesh is None:
            continue
        rel = mesh.get("filename").replace("package://alex_virtual_description/", "")
        if not os.path.exists(os.path.join(MESHDIR, rel)):
            continue
        if rel not in seen:
            seen[rel] = f"mesh{len(seen)}"
            _sub(asset, "mesh", name=seen[rel], file=rel)
        o = vis.find("origin")
        xyz = (o is not None and o.get("xyz")) or "0 0 0"
        w, x, y, z = _rpy_to_quat([float(v) for v in
                                  ((o is not None and o.get("rpy")) or "0 0 0").split()])
        _sub(body, "geom", type="mesh", mesh=seen[rel], pos=xyz, quat=f"{w} {x} {y} {z}",
             contype="0", conaffinity="0", group="1", material="robot")


def build_sim_model(policy, with_visuals=True, with_imu_sensors=False, terrain=None):
    """Free-base Alex: estimator MJCF + floor + SCS2's collision set + per-joint position servos.

    kd is applied as MuJoCo joint damping, which with a kp-only `position` actuator reproduces
    Java's `tau = kp*(q_d - q) + kd*(0 - qd)` closely (verified in EXPERIMENTS.md §4).

    `with_imu_sensors` adds a gyro + accelerometer on each of the estimator's 8 IMU sites, for
    `run_estimator.py`. Sensors are massless and stateless, so the dynamics are bit-identical
    either way (`tests/sim/test_sensors.py::test_sensors_do_not_change_the_dynamics`).
    """
    urdf = cycloid_forearm_urdf(URDF)
    root = ET.fromstring(me.alex_spec_from_urdf(urdf).mjcf)

    _sub(root, "option", timestep=DT, gravity="0 0 -9.81", integrator="implicitfast",
         solver="Newton", iterations="25", noslip_iterations="5", impratio="1", cone="pyramidal")

    # The look (textures, materials, lighting) rides along with the visual meshes: headless runs
    # compile the same dynamics with neither.
    if with_visuals:
        _add_scene_look(root)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for body, typ, size, pos, quat in SCS2_COLLISION_GEOMS:
        _geom(bodies[body], f"{body}_collision_0", ROBOT_GROUP,
              type=typ, size=size, pos=pos, quat=quat, group="3")   # group 3 = hidden in viewer

    act = ET.SubElement(root, "actuator")
    for j in root.iter("joint"):
        n = j.get("name")
        if n not in _FALLBACK:
            continue
        fb = _FALLBACK[n]
        kp = policy["kp"].get(n, float(fb["kp"]))
        kd = policy["kd"].get(n, float(fb["kd"]))
        tau = policy["tau"].get(n, float(fb["maxEffort"]))
        j.set("damping", repr(kd))
        _sub(act, "position", name=n, joint=n, kp=repr(kp),
             forcelimited="true", forcerange=f"{-tau} {tau}")

    if with_visuals:
        _add_visual_meshes(root, urdf)
    if terrain is None:
        _geom(root.find("worldbody"), "floor", TERRAIN_GROUP, type="plane", size="20 20 0.1",
              **({"material": "groundplane"} if with_visuals else {}))
    else:
        from invariant_estimation.sim import terrain as terr
        hf = ET.SubElement(_asset(root), "hfield")
        hf.set("name","terrain")
        hf.set("nrow", str(terr.N))
        hf.set("ncol",str(terr.N))
        hf.set("size", f"{terr.EXTENT/2} {terr.EXTENT/2} {terr.EZ} 0.1")
        _geom(root.find("worldbody"), "floor", TERRAIN_GROUP, type="hfield", hfield="terrain",
              **({"material": "groundplane"} if with_visuals else {}))
    if with_imu_sensors:
        from invariant_estimation.sim.sensors import add_imu_sensors
        add_imu_sensors(root, me.ALEX_IMU_SITES)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if terrain is not None:
        from invariant_estimation.sim import terrain as terr
        m.hfield_data[:] = (np.asarray(terrain, np.float32) /  terr.EZ).clip(0,1).ravel()
    return m


# ---------------------------------------------------------------------------
# Index maps, observations, geometry helpers
# ---------------------------------------------------------------------------
_DOWN = np.array([0.0, 0.0, -1.0])   # world gravity direction, for `projected_gravity`


def make_maps(m, policy):
    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def aid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    order = policy["order"]
    return {
        "QADR": np.array([m.jnt_qposadr[jid(n)] for n in order]),
        "DOFADR": np.array([m.jnt_dofadr[jid(n)] for n in order]),
        "AID": np.array([aid(n) for n in order]),
        "HOME": np.array([policy["home"][n] for n in order]),
        "ALL_AID": np.array([aid(n) for n in _FALLBACK]),
        "ALL_HOME": np.array([policy["home"].get(n, float(_FALLBACK[n]["homePosition"]))
                              for n in _FALLBACK]),
        "ALL_QADR": np.array([m.jnt_qposadr[jid(n)] for n in _FALLBACK]),
        "BASE_BID": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "PELVIS_LINK"),
    }


def build_obs(m, d, policy, maps, cmd, last_action, est=None):
    """The observation vector, term-by-term in the order the policy's `observations` declares.

    cmd = [vx, vy, yaw_rate, standing, base_height].

    `est` optionally supplies ESTIMATED values for individual terms, keyed by term name
    (`run_estimator.py` passes `base_ang_vel` and `projected_gravity`). A term present in `est`
    replaces the ground-truth one; everything else is unchanged, so `est=None` is exactly the
    old behaviour and the two modes are directly A/B-comparable.
    """
    bid = maps["BASE_BID"]
    est = est or {}
    out = []
    for term in policy["obs_terms"]:
        if term in est:
            out.append(np.asarray(est[term], dtype=np.float64).ravel())
        elif term == "base_ang_vel":
            # mjOBJ_XBODY, NOT mjOBJ_BODY. With flg_local=1 MuJoCo resolves mjOBJ_BODY in the body's
            # INERTIAL frame, and Alex's pelvis inertia frame is ~180 deg about (1,0,1)/sqrt(2) — so
            # mjOBJ_BODY hands back a gyro with x/z swapped and y negated. This was THE bug that
            # made every policy fall; see EXPERIMENTS.md. mjOBJ_XBODY uses the body frame, matching
            # Java's RLEstimates.root_AngularVelocity to 1e-4.
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, v6, 1)
            out.append(v6[:3])
        elif term == "projected_gravity":
            out.append(d.xmat[bid].reshape(3, 3).T @ _DOWN)
        elif term == "base_velocity_plus_standing":
            out.append(cmd[:4])
        elif term == "base_height":
            out.append(cmd[4:5])
        elif term == "joint_pos_rel":
            out.append(d.qpos[maps["QADR"]] - maps["HOME"])
        elif term == "joint_vel_rel":
            out.append(d.qvel[maps["DOFADR"]])
        elif term == "last_action":
            out.append(last_action)
        else:
            raise ValueError(f"unknown obs term {term}")
    return np.concatenate(out).astype(np.float32)


def lowest_foot_to_root_height(m, d):
    """Java's `RLEstimates.lowestFootToRootHeight`: root height above the LOWER sole plane."""
    return d.qpos[2] - min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)][2] - ANKLE_HEIGHT
                           for b in ("LEFT_FOOT", "RIGHT_FOOT"))


def foot_rest_height(m, maps):
    """Pelvis z putting the lowest foot-box corner ~1 mm off the floor at the home pose."""
    d = mujoco.MjData(m)
    d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
    d.qpos[0:3] = [0.0, 0.0, 1.0]; d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS]
    return 1.0 - min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in gids) + 0.001


# ---------------------------------------------------------------------------
# Live command bindings
#
# MuJoCo's viewer reserves EVERY letter A-Z (plus , / ; ' \ `) for render-flag toggles -- the
# shortcut column of mjVISSTRING/mjRNDSTRING -- and it applies its own toggle *in addition* to
# calling our key_callback. So a WASD mapping silently toggles wireframe / auto-connect / shadows
# / static-body while also steering. The numeric keypad is untouched by the viewer, so the live
# command lives there; `stdin_commands()` gives the same control from the terminal for keyboards
# without a numpad. RESERVED_KEYS + the check below turn a future letter binding into a loud failure.
# ---------------------------------------------------------------------------
RESERVED_KEYS = frozenset(
    row[2].strip().upper()
    for table in (mujoco.mjVISSTRING, mujoco.mjRNDSTRING)
    for row in table
    if len(row) > 2 and row[2].strip()
)

# Keypad label -> GLFW keycode (GLFW_KEY_KP_0..KP_9 are contiguous from 320).
# 330 KP_DECIMAL and 331 KP_DIVIDE are free too; 332 KP_MULTIPLY is the ghost toggle.
KEYPAD = {str(n): 320 + n for n in range(10)} | {"-": 333, "+": 334, "*": 332}

VX_STEP, VY_STEP, YAW_STEP, HEIGHT_STEP = 0.1, 0.1, 0.1, 0.02
# Command ranges the policy was TRAINED on: alex_ihmc_walk_env_cfg.AlexCommandsCfg
# ranges=(lin_vel_x=(-0.9,0.9), lin_vel_y=(-0.5,0.5), ang_vel_z=(-1.5,1.5)).
VX_MAX, VY_MAX, YAW_MAX = 0.9, 0.5, 1.5

# The policy stands still for small commands and only tracks above a deadband (measured: vx ratio
# 0.01 at 0.20, 0.74 at 0.30, 0.95 at 0.45; yaw ratio 0.03 at 0.25, 0.69 at 0.50, 1.05 at 0.75).
# So a linear stick would need ~40% deflection to do anything. Map the travel just past the stick
# deadzone straight onto the smallest command that actually moves.
WALK_MIN_VX, WALK_MIN_VY, WALK_MIN_YAW = 0.30, 0.28, 0.60

# command name -> (keypad label, terminal letter, help text). The letters are terminal-only; typing
# them into the viewer window does nothing (by design -- see above).
BINDINGS = (
    ("forward",    "8", "w", "+vx"),
    ("back",       "2", "s", "-vx"),
    ("left",       "4", "a", "+vy (y is LEFT)"),
    ("right",      "6", "d", "-vy"),
    ("turn_left",  "7", "q", "+yaw rate"),
    ("turn_right", "9", "e", "-yaw rate"),
    ("stop",       "5", "x", "zero vx/vy/yaw, restore standing"),
    ("higher",     "+", "+", "raise base height"),
    ("lower",      "-", "-", "lower base height"),
    ("stand",      "0", "t", "toggle standing flag"),
    # Viewer-only, and a no-op unless a `Ghost` is attached (`run_estimator.py --ghost`).
    ("ghost",      "*", "g", "cycle the estimator ghost: off -> full -> attitude"),
)
_BY_KEY = {KEYPAD[label]: name for name, label, _, _ in BINDINGS}
_BY_CHAR = {ch: name for name, _, ch, _ in BINDINGS}

# The velocity commands, as (index into `Loop.cmd`, step). Clamped to +-_VEL_MAX[index].
_VEL_MAX = (VX_MAX, VY_MAX, YAW_MAX)
_VEL_STEP = {"forward": (0, VX_STEP), "back": (0, -VX_STEP),
             "left": (1, VY_STEP), "right": (1, -VY_STEP),
             "turn_left": (2, YAW_STEP), "turn_right": (2, -YAW_STEP)}

if any(k < 128 and chr(k).upper() in RESERVED_KEYS for k in _BY_KEY):
    raise RuntimeError(f"viewer keys {sorted(chr(k) for k in _BY_KEY if k < 128)} collide with "
                       "MuJoCo's render toggles; bind a keypad code instead")

GAMEPAD_HELP = ("       left stick   forward/back + strafe (vx, vy)\n"
                "      right stick   yaw rate (X axis)\n"
                "          RT / LT   raise / lower base height\n"
                "                A   toggle standing flag\n"
                "                B   stop\n"
                "            START   height back to the policy default")

KEYMAP_HELP = "\n".join(
    [f"  {'keypad':>8}  {'terminal':>8}   effect"]
    + [f"  {label:>8}  {ch:>8}   {name} ({doc})" for name, label, ch, doc in BINDINGS])


# ---------------------------------------------------------------------------
# Xbox-style gamepad, straight off the legacy joystick device
#
# `/dev/input/js0` speaks a fixed 8-byte struct, so this needs no dependency and no thread: we drain
# pending events once per control tick from the loop's own thread. (GLFW has a gamepad API and is
# already a dependency, but in `launch_passive` the GLFW context lives on the viewer's thread and
# joystick calls must come from the thread that owns it -- reading the device sidesteps that.)
#
# Axis/button numbering below is Linux `xpad`, confirmed against the attached pad: 8 axes,
# 11 buttons, triggers resting at -32767.
# ---------------------------------------------------------------------------
JS_DEVICE = "/dev/input/js0"
JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80

AX_LEFT_X, AX_LEFT_Y, AX_LT, AX_RIGHT_X, AX_RIGHT_Y, AX_RT = 0, 1, 2, 3, 4, 5
BTN_A, BTN_B, BTN_START = 0, 1, 7
JS_DEADZONE = 0.15          # the attached pad rests up to 0.072 off centre
HEIGHT_RATE = 0.15          # m/s while a trigger is held


def _deadzone(v):
    """Normalised axis with the deadzone removed and the remainder rescaled to full travel."""
    if abs(v) < JS_DEADZONE:
        return 0.0
    return (abs(v) - JS_DEADZONE) / (1.0 - JS_DEADZONE) * (1.0 if v > 0 else -1.0)


def _stick_to_command(v, lo, hi):
    """Stick axis -> command, skipping the policy's own deadband.

    Any deflection past the hardware deadzone lands at `lo` (the smallest command that actually
    produces motion) and full deflection reaches `hi`. Without this the first ~40% of stick travel
    commands a velocity the policy ignores, which reads as "the pad works but the robot won't move".
    """
    n = _deadzone(v)
    if n == 0.0:
        return 0.0
    return (1.0 if n > 0 else -1.0) * (lo + (hi - lo) * abs(n))


class Gamepad:
    """Absolute stick -> velocity command; triggers -> base height rate. See GAMEPAD_HELP."""

    def __init__(self, path=JS_DEVICE):
        self.axes, self.buttons, self.fd, self.name = {}, {}, None, None
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return
        self.name = next((os.path.basename(q) for q in glob.glob("/dev/input/by-id/*-joystick")
                          if os.path.realpath(q) == os.path.realpath(path)), path)

    @property
    def present(self):
        return self.fd is not None

    def _drain(self):
        """Read every pending event. Returns the buttons that went down this tick."""
        pressed = []
        while True:
            try:
                buf = os.read(self.fd, 8)
            except BlockingIOError:
                return pressed
            except OSError:                            # unplugged
                os.close(self.fd); self.fd = None
                print("\n  gamepad disconnected")
                return pressed
            if not buf or len(buf) < 8:
                return pressed
            _t, val, typ, num = struct.unpack("IhBB", buf)
            # JS_EVENT_INIT marks the synthetic startup value; store it like any other event.
            if typ & ~JS_EVENT_INIT == JS_EVENT_AXIS:
                self.axes[num] = val / 32767.0
            elif typ & ~JS_EVENT_INIT == JS_EVENT_BUTTON:
                was = self.buttons.get(num, 0)
                self.buttons[num] = val
                if val and not was:
                    pressed.append(num)

    def read(self):
        """Drain the device and decode it: (buttons pressed this tick, (vx, vy, yaw), (up, down)).

        `up`/`down` are the trigger travel in 0..1; the caller turns them into a height rate.
        """
        pressed = self._drain()
        ax = self.axes
        # Stick up and stick left both read negative on xpad, hence the sign flips.
        vel = (-_stick_to_command(ax.get(AX_LEFT_Y, 0.0), WALK_MIN_VX, VX_MAX),
               -_stick_to_command(ax.get(AX_LEFT_X, 0.0), WALK_MIN_VY, VY_MAX),
               -_stick_to_command(ax.get(AX_RIGHT_X, 0.0), WALK_MIN_YAW, YAW_MAX))
        # Triggers rest at -1 and travel to +1; take the half above rest as 0..1.
        trig = (max(0.0, (ax.get(AX_RT, -1.0) + 1.0) * 0.5),
                max(0.0, (ax.get(AX_LT, -1.0) + 1.0) * 0.5))
        return pressed, vel, trig

    def apply(self, loop, dt):
        """Drain the device and write the command into `loop`."""
        if not self.present:
            return
        pressed, vel, (up, down) = self.read()
        for btn in pressed:
            if btn == BTN_A:
                loop.command("stand")
            elif btn == BTN_B:
                loop.command("stop")
            elif btn == BTN_START:
                loop.set_height_target(loop.policy["base_height"])

        loop.cmd[0:3] = vel
        loop.cmd[3] = 0.0 if sum(abs(v) for v in vel) > 1e-9 else 1.0
        if up or down:
            loop.nudge_height((up - down) * HEIGHT_RATE * dt)


class Loop:
    """Sim state, the live command, and one control tick (DECIMATION physics steps)."""

    def __init__(self, m, policy, maps):
        self.m, self.policy, self.maps = m, policy, maps
        self.sess, self.scale = policy["sess"], policy["action_scale"]
        self.n = len(policy["order"])

        self.d = mujoco.MjData(m)
        self.d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
        self.d.qpos[0:3] = [0.0, 0.0, foot_rest_height(m, maps)]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        mujoco.mj_forward(m, self.d)

        self.last_action = np.zeros(self.n, np.float32)
        # RLHeightManager.goHome(): ease from wherever the root is to `base_height`.
        self.height_target = policy["base_height"]
        self._ramp_from = lowest_foot_to_root_height(m, self.d)
        self._ramp_t = 0.0
        # [vx, vy, yaw_rate, standing, base_height]
        self.cmd = np.array([0.0, 0.0, 0.0, 1.0, self._ramp_from])
        # Optional `sim.ghost.Ghost`, attached by the viewer entry points. Kept on the Loop so the
        # existing keypad/terminal command plumbing reaches it without a second dispatch path.
        self.ghost = None

    # -- height command (RLHeightManager) ------------------------------------
    def _height(self):
        """Zero-end-velocity cubic from `_ramp_from` to `height_target` over GO_HOME_DURATION."""
        tau = min(self._ramp_t / GO_HOME_DURATION, 1.0)
        blend = 1.0 - 3.0 * tau ** 2 + 2.0 * tau ** 3
        return self.height_target + (self._ramp_from - self.height_target) * blend

    def set_height_target(self, target):
        """Discrete jump: re-seed the ramp from the current command, as `goHome()` does."""
        self._ramp_from = self._height()
        self._ramp_t = 0.0
        self.height_target = float(np.clip(target, *self.policy["height_range"]))

    def nudge_height(self, delta):
        """Continuous (analog) height change: move the command itself, no ramp.

        Re-seeding the ramp every tick would pin it at tau=0 and freeze the command, so collapse
        the ramp instead -- with `_ramp_from == height_target`, `_height()` just returns the target.
        """
        h = float(np.clip(self._height() + delta, *self.policy["height_range"]))
        self.height_target = self._ramp_from = h
        self._ramp_t = GO_HOME_DURATION

    # -- one control tick ----------------------------------------------------
    def control_tick(self):
        self.cmd[4] = self._height()
        obs = build_obs(self.m, self.d, self.policy, self.maps, self.cmd, self.last_action)
        self.last_action = self.sess.run(
            None, {self.sess.get_inputs()[0].name: obs[None]})[0][0]
        self.d.ctrl[self.maps["ALL_AID"]] = self.maps["ALL_HOME"]     # undriven joints hold home
        self.d.ctrl[self.maps["AID"]] = self.maps["HOME"] + self.scale * self.last_action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)
        self._ramp_t += DECIMATION * DT

    def tilt_deg(self):
        c = self.d.xmat[self.maps["BASE_BID"]].reshape(3, 3)[2, 2]
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    # -- live command --------------------------------------------------------
    def command(self, name):
        """Apply a named command (see BINDINGS). Unknown names are ignored."""
        c = self.cmd
        if name in _VEL_STEP:
            i, step = _VEL_STEP[name]
            c[i] = np.clip(c[i] + step, -_VEL_MAX[i], _VEL_MAX[i])
        elif name == "stop":
            c[0:3] = 0.0
        elif name == "higher":
            return self.set_height_target(self.height_target + HEIGHT_STEP)
        elif name == "lower":
            return self.set_height_target(self.height_target - HEIGHT_STEP)
        elif name == "stand":
            c[3] = 1.0 - c[3]
            return
        elif name == "ghost":
            if self.ghost is not None:
                print(f"\n  ghost: {self.ghost.cycle()}")
            return
        else:
            return
        # Any nonzero velocity clears the standing flag: base_velocity_plus_standing[3] gates
        # walking, so commanding vx while it is set does nothing.
        c[3] = 0.0 if np.any(np.abs(c[0:3]) > 1e-9) else 1.0

    def key(self, keycode):
        """Viewer key_callback: keypad only (every letter is a MuJoCo render toggle)."""
        name = _BY_KEY.get(keycode)
        if name:
            self.command(name)

    def status(self):
        c = self.cmd
        return (f"cmd vx={c[0]:+.1f} vy={c[1]:+.1f} yaw={c[2]:+.1f} stand={c[3]:.0f} "
                f"h={c[4]:.3f}->{self.height_target:.3f} | z={self.d.qpos[2]:+.3f} "
                f"tilt={self.tilt_deg():5.1f}deg |a|={np.abs(self.last_action).max():5.2f} "
                f"ncon={self.d.ncon}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def make_loop(policy_name, with_visuals):
    policy = load_policy(policy_name)
    m = build_sim_model(policy, with_visuals=with_visuals)
    return Loop(m, policy, make_maps(m, policy))


def run_headless(policy_name, ticks, use_gamepad=False):
    """No window. With --gamepad, also prints the raw axes next to the command they produced and
    the distance travelled -- the diagnostic for "the pad reads but the robot won't move"."""
    loop = make_loop(policy_name, with_visuals=False)
    m, policy = loop.m, loop.policy
    pad = Gamepad() if use_gamepad else None
    print(f"policy={policy_name} obs={policy['input_size']} njoints={loop.n} "
          f"scale={loop.scale} height_target={loop.height_target}")
    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} mass={m.body_mass.sum():.6f} "
          f"start pelvis_z={loop.d.qpos[2]:.3f}")
    if pad is not None:
        print(f"gamepad: present={pad.present} name={pad.name}\n{GAMEPAD_HELP}\n"
              f"  --> move the sticks now; 'axes' below is what this process reads from the device.")
    x0, y0 = loop.d.qpos[0], loop.d.qpos[1]
    for k in range(ticks):
        if pad is not None:
            pad.apply(loop, DECIMATION * DT)
        loop.control_tick()
        if k % 25 == 0:
            extra = ""
            if pad is not None:
                ax = {a: round(pad.axes.get(a, 0.0), 3)
                      for a in (AX_LEFT_X, AX_LEFT_Y, AX_RIGHT_X, AX_LT, AX_RT)}
                extra = (f"  axes LX/LY/RX/LT/RT={list(ax.values())}"
                         f" moved=({loop.d.qpos[0] - x0:+.2f},{loop.d.qpos[1] - y0:+.2f})m")
            print(f"  t={k * DECIMATION * DT:5.2f}s  {loop.status()}{extra}")
    print(f"final {loop.status()}  travelled=({loop.d.qpos[0] - x0:+.2f},{loop.d.qpos[1] - y0:+.2f})m"
          f"  finite={np.all(np.isfinite(loop.d.qpos))}")


# ---------------------------------------------------------------------------
# Offscreen video
#
# `mujoco.Renderer` + a raw-RGB pipe into ffmpeg: no imageio/mediapy dependency, and nothing
# has to be held in memory (a 30 s 720p run is ~4 GB of frames if you buffer them).
#
# Needs an offscreen GL context. `MUJOCO_GL=egl` is the headless-safe pick and is what the
# entry points default to, but it MUST be set before `import mujoco` -- see the top of
# `run_estimator.py`.
# ---------------------------------------------------------------------------
class VideoRecorder:
    """Renders a tracking view of `body` and streams it to `path` as H.264."""

    def __init__(self, m, path, *, body=0, width=1280, height=720, fps=50,
                 distance=3.2, azimuth=120.0, elevation=-15.0):
        import subprocess
        self.m, self.path, self.fps = m, path, fps
        self.renderer = mujoco.Renderer(m, height, width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = body
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.n = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    def capture(self, d, overlay=None):
        """Render one frame. `overlay(scn)` may append geoms to the offscreen scene.

        The hook is what lets the estimator ghost -- previously viewer-only, because
        headless "has nothing to draw into" -- be recorded: `mujoco.Renderer` owns a
        real `mjvScene`, so `Ghost.draw` works against it unchanged.
        """
        self.renderer.update_scene(d, camera=self.cam, scene_option=self.opt)
        if overlay is not None:
            overlay(self.renderer.scene)
        self.proc.stdin.write(self.renderer.render().tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.renderer.close()
        print(f"  video -> {self.path}  ({self.n} frames, {self.n / self.fps:.1f}s @ {self.fps}fps)")


def stdin_commands(loop):
    """Daemon thread: terminal letters drive the same commands as the keypad (see BINDINGS), plus
    `h <metres>` for the height target and `v <vx> <vy> <yaw>` to set the velocity outright."""
    def pump():
        for line in sys.stdin:
            parts = line.strip().lower().split()
            if not parts:
                continue
            head = parts[0]
            try:
                if head == "h" and len(parts) == 2:
                    loop.set_height_target(float(parts[1]))
                elif head == "v" and len(parts) == 4:
                    loop.cmd[0:3] = [float(x) for x in parts[1:]]
                    loop.cmd[3] = 0.0 if np.any(np.abs(loop.cmd[0:3]) > 1e-9) else 1.0
                else:
                    for ch in head:                     # "www" == three presses
                        loop.command(_BY_CHAR.get(ch, ""))
            except ValueError:
                print("  ? expected e.g. 'w', 'www', 'h 0.85', 'v 0.4 0 0'")
    threading.Thread(target=pump, daemon=True).start()


def probe_gamepad(seconds=30.0, policy_name="baseline"):
    """Print raw axes next to the command they produce, so the sign conventions can be eyeballed.

    The xpad axis numbering is standard but the sign of each stick is worth confirming by hand:
    push the stick the way you want the robot to go and check the command agrees.
    """
    pad = Gamepad()
    if not pad.present:
        raise SystemExit(f"no gamepad at {JS_DEVICE}")
    print(f"gamepad: {pad.name}\n{GAMEPAD_HELP}")
    print("\nExpect: stick UP -> vx > 0 (forward), stick LEFT -> vy > 0 (left),")
    print("        right stick LEFT -> yaw > 0 (turn left), RT -> height up.\n")
    print(f"{'LX':>7}{'LY':>7}{'RX':>7}{'LT':>7}{'RT':>7}  |{'vx':>7}{'vy':>7}{'yaw':>7}{'height':>8}  buttons")
    band = POLICIES[policy_name]["height_range"]
    height = POLICIES[policy_name]["base_height"]
    t0 = last = time.time()
    while time.time() - t0 < seconds:
        pressed, (vx, vy, yaw), (up, down) = pad.read()
        for btn in pressed:
            print(f"  button {btn} down"
                  f"{'  (A: stand)' if btn == BTN_A else '  (B: stop)' if btn == BTN_B else ''}")
        a = pad.axes
        height = float(np.clip(height + (up - down) * HEIGHT_RATE * 0.05, *band))
        if time.time() - last > 0.15:
            held = [b for b, v in pad.buttons.items() if v]
            print(f"{a.get(AX_LEFT_X, 0):+7.3f}{a.get(AX_LEFT_Y, 0):+7.3f}{a.get(AX_RIGHT_X, 0):+7.3f}"
                  f"{a.get(AX_LT, -1):+7.3f}{a.get(AX_RT, -1):+7.3f}  |"
                  f"{vx:+7.2f}{vy:+7.2f}{yaw:+7.2f}{height:8.3f}  {held}", end="\r", flush=True)
            last = time.time()
        time.sleep(0.02)
    print()


# ---------------------------------------------------------------------------
# Free (WASD) viewer -- our own GLFW window, so we get real key press/release
#
# `launch_passive` is MuJoCo's Simulate GUI: it reserves every letter for render
# toggles AND only forwards key PRESS (never RELEASE) to `key_callback`, so it can
# neither host a WASD binding without also flipping wireframe/shadows nor tell when a
# key is let go. Owning a plain GLFW window and rendering into it ourselves sidesteps
# both: raw GLFW hands us PRESS *and* RELEASE with no built-in bindings at all. Keys
# accumulate into `held`; each control tick maps the current set to the command, so
# holding W walks and releasing it stops -- Minecraft-style, no controller needed.
# ---------------------------------------------------------------------------
# Held-key command magnitudes. Chosen above the policy's walk deadband (see the
# WALK_MIN_* note) so a tap produces motion instead of the ignored first ~40%.
WASD_VX, WASD_VY, WASD_YAW = 0.6, 0.4, 0.9

WASD_HELP = (
    "  move:  W/S forward/back   A/D strafe left/right   Q/E turn left/right\n"
    "  height: SPACE up   SHIFT down   R reset to policy default\n"
    "  X stop        ESC quit\n"
    "  mouse:  left-drag orbit   right-drag pan   scroll zoom (camera tracks the pelvis)")


def _apply_held(loop, held, dt):
    """Map the set of currently-held GLFW keycodes onto the live command (continuous)."""
    import glfw
    vx = (glfw.KEY_W in held) - (glfw.KEY_S in held)
    vy = (glfw.KEY_A in held) - (glfw.KEY_D in held)          # A is +vy (y is LEFT)
    yaw = (glfw.KEY_Q in held) - (glfw.KEY_E in held)
    loop.cmd[0] = vx * WASD_VX
    loop.cmd[1] = vy * WASD_VY
    loop.cmd[2] = yaw * WASD_YAW
    # Any nonzero velocity clears the standing flag (base_velocity_plus_standing[3]).
    loop.cmd[3] = 0.0 if (vx or vy or yaw) else 1.0
    up = glfw.KEY_SPACE in held
    down = (glfw.KEY_LEFT_SHIFT in held) or (glfw.KEY_RIGHT_SHIFT in held)
    if up or down:
        loop.nudge_height((up - down) * HEIGHT_RATE * dt)


def run_free_viewer(policy_name, loop=None):
    """Standalone GLFW window with WASD+Space+Shift live control. See WASD_HELP.

    `loop` optionally supplies a PRE-BUILT loop -- `run_estimator.py --wasd` passes its
    `EstimatedLoop` so this window can drive the estimator (and its ghost) instead of ground truth.
    Default `None` builds the plain truth-driven loop, i.e. the original behaviour.
    """
    import glfw
    # This viewer owns its own GLFW window, so it must run on the main thread -- and it must NOT be
    # launched under `mjpython`, which runs the script on a *secondary* thread (mjpython keeps the
    # main thread for its own Cocoa loop). GLFW window creation off the main thread throws an opaque
    # `libc++abi ... NSException` on macOS; catch that here with a message that says what to do.
    if threading.current_thread() is not threading.main_thread():
        raise SystemExit(
            "--wasd must run on the main thread. Launch it with plain python, NOT mjpython:\n"
            "    uv run python run_policy.py --wasd\n"
            "(mjpython runs this script off the main thread, so GLFW raises an NSException. mjpython "
            "is only needed for the DEFAULT viewer, which uses MuJoCo's own Simulate GUI.)")
    loop = loop if loop is not None else make_loop(policy_name, with_visuals=True)
    m, d = loop.m, loop.d

    if not glfw.init():
        raise SystemExit("glfw.init() failed -- no display?")
    window = glfw.create_window(1280, 960, f"Alex -- {policy_name} (WASD)", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("glfw.create_window() failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(m, maxgeom=10000)
    context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    # Third-person chase cam on the pelvis, so the view follows the robot as it walks.
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = loop.maps["BASE_BID"]
    cam.distance, cam.azimuth, cam.elevation = 3.5, 90.0, -20.0

    held = set()
    mouse = {"L": False, "R": False, "x": 0.0, "y": 0.0}

    def on_key(_w, key, _sc, action, _mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_R:
                loop.set_height_target(loop.policy["base_height"])
            elif key == glfw.KEY_X:
                loop.command("stop")
            elif key == glfw.KEY_G:
                # This window owns its `on_key`, so unlike the passive viewer a plain letter is
                # available here -- no keypad needed.
                loop.command("ghost")
            else:
                held.add(key)
        elif action == glfw.RELEASE:
            held.discard(key)

    def on_mouse_button(_w, button, action, _mods):
        down = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["L"] = down
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse["R"] = down
        mouse["x"], mouse["y"] = glfw.get_cursor_pos(window)

    def on_cursor(_w, xpos, ypos):
        dx, dy = xpos - mouse["x"], ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos
        if not (mouse["L"] or mouse["R"]):
            return
        h = max(1, glfw.get_window_size(window)[1])
        act = (mujoco.mjtMouse.mjMOUSE_MOVE_V if mouse["R"]
               else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        mujoco.mjv_moveCamera(m, act, dx / h, dy / h, scene, cam)

    def on_scroll(_w, _xoff, yoff):
        mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoff, scene, cam)

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)

    print(f"Viewer (WASD): policy={policy_name}\n{WASD_HELP}")
    print("  Terminal also accepts: 'w'/'www', 'h 0.85', 'v 0.4 0 0'.")
    stdin_commands(loop)

    dt = DECIMATION * DT
    last_print = 0.0
    while not glfw.window_should_close(window):
        t0 = time.time()
        _apply_held(loop, held, dt)
        loop.control_tick()

        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        # After `mjv_updateScene` (which resets `ngeom` itself, so no manual clear here) and
        # before rendering. No-op unless an estimated loop attached a ghost.
        if loop.ghost is not None:
            # Only an `EstimatedLoop` ever attaches a ghost, so `current_estimate` is present.
            est = loop.current_estimate()
            if est is not None:
                loop.ghost.update(est, d)
                loop.ghost.draw(scene)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, fb_w, fb_h), scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

        if t0 - last_print > 0.5:
            # print(f"  {loop.status()}   ", end="\r", flush=True)
            last_print = t0
        sleep = dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)
    glfw.terminate()


def run_viewer(policy_name):
    import mujoco.viewer
    loop = make_loop(policy_name, with_visuals=True)
    pad = Gamepad()
    print(f"Viewer: policy={policy_name}")
    print(f"  gamepad: {pad.name}\n{GAMEPAD_HELP}" if pad.present
          else f"  no gamepad at {JS_DEVICE}; use the keypad or the terminal\n{KEYMAP_HELP}")
    print("  Terminal also accepts: 'w'/'www', 'h 0.85', 'v 0.4 0 0'.\n"
          "  Letters do nothing in the viewer window -- MuJoCo owns all 26 for render toggles.\n"
          "  Press 3 in the viewer to show the collision geoms.")
    stdin_commands(loop)
    with mujoco.viewer.launch_passive(loop.m, loop.d, key_callback=loop.key) as v:
        last_print = 0.0
        while v.is_running():
            t0 = time.time()
            pad.apply(loop, DECIMATION * DT)
            loop.control_tick()
            v.sync()
            if t0 - last_print > 0.5:
                print(f"  {loop.status()}   ", end="\r", flush=True)
                last_print = t0
            sleep = DECIMATION * DT - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="standing", choices=list(POLICIES))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--probe-gamepad", action="store_true",
                    help="print raw axes vs the command they produce, to check the sign conventions")
    ap.add_argument("--gamepad", action="store_true",
                    help="with --headless: drive from the gamepad and print axes + command + travel")
    ap.add_argument("--wasd", action="store_true",
                    help="own-window GLFW viewer with WASD+Space+Shift hold-to-move (no controller)")
    args = ap.parse_args()
    if args.probe_gamepad:
        probe_gamepad(policy_name=args.policy)
    elif args.headless:
        run_headless(args.policy, args.ticks, use_gamepad=args.gamepad)
    elif args.wasd:
        run_free_viewer(args.policy)
    else:
        run_viewer(args.policy)
