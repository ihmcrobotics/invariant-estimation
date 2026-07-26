"""run_policy.py — IHMC RL policies in a standalone MuJoCo sim, driven live.

Runs an exported ONNX policy directly via `onnxruntime` (no JAX, no port) in a plain MuJoCo loop
with ground-truth observations. Independent of the estimator.

    uv run python run_policy.py                       # viewer, standing demo
    uv run python run_policy.py --policy baseline     # the 29-joint walking policy
    uv run python run_policy.py --headless --ticks 400

Viewer keys: WASD walk (W/S = ±vx, A/D = ±vy), Q/E turn, X stop, SPACE/SHIFT raise/lower the
commanded base height, T toggle the standing flag.

Everything here that looks like a magic number is matched to SCS2's MuJoCo backend, which is the
reference implementation that works: `MujocoMultiBodyRobotFactory` (physics options, contact
params, collision grouping), `AlexSimulationCollisionModel` (the collision shapes),
`RLHeightManager` (the height command), `ObservationDefinitions` (the observation vector).
**`EXPERIMENTS.md` is the full log** — read it before changing any of it, and before re-testing a
hypothesis about why a policy might fall.

The one trap worth repeating here: `base_ang_vel` must come from `mjOBJ_XBODY`, not `mjOBJ_BODY`.
See `build_obs`.
"""
import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import onnxruntime as ort
import yaml

from invariant_estimation.pipeline import main_estimator as me
from invariant_estimation.model.urdf2mjcf import _rpy_to_quat

URDF = "/home/llibshutz/Documents/alex_with_imus.urdf"
RL_MODELS = "/home/llibshutz/workspaces/robot-stuff/alex/src/main/resources/rl_models"
MESHDIR = "/home/llibshutz/workspaces/robot-stuff/ihmc-alex-sdk/alex-models/alex_virtual_description"

DT = 0.005             # physics timestep (200 Hz), == IsaacLab's SIM_DT
DECIMATION = 4         # -> 50 Hz control, == IsaacLab's CONTROL_DT

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
SCS2_COLLISION_GEOMS = (
    ("PELVIS_LINK",          "capsule", "0.135 0.025",      "-0.06 0 -0.02",
     "0.70710678118654746 -0.70710678118654768 0 0"),
    ("LEFT_FOOT",            "box",     "0.13 0.07 0.0275", "0.045 0 -0.05", "1 0 0 0"),
    ("RIGHT_FOOT",           "box",     "0.13 0.07 0.0275", "0.045 0 -0.05", "1 0 0 0"),
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
POLICIES = {
    "standing": {"yaml": "20251219_standing18_def.yaml", "onnx": "20251219_standing18.onnx",
                 "base_height": 0.75},
    "baseline": {"dir": "2026-07-10_baseline", "base_height": 0.89},
    "forearms": {"dir": "2026-04-13_002_forearms", "base_height": 0.89},
}
# Gains/home for joints a policy does not drive (wrists/grippers for the 23-joint standing policy),
# so they hold home instead of flopping.
_FALLBACK = {p["name"]: p for p in
             yaml.safe_load(open(f"{RL_MODELS}/2026-07-10_baseline/policy_cfg.yaml"))["jointParameters"]}


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policy(name):
    """Read a policy's `policy_cfg.yaml` (obs list, joint order, per-joint gains) + its ONNX."""
    spec = POLICIES[name]
    if "dir" in spec:
        ybase, obase = f"{spec['dir']}/policy_cfg.yaml", f"{spec['dir']}/policy.onnx"
    else:
        ybase, obase = spec["yaml"], spec["onnx"]
    cfg = yaml.safe_load(open(f"{RL_MODELS}/{ybase}"))
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
        "sess": ort.InferenceSession(f"{RL_MODELS}/{obase}", providers=["CPUExecutionProvider"]),
        "base_height": float(spec["base_height"]),
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


def _geom(parent, name, contact_group, **attrs):
    """A geom carrying SCS2's contact parameters and one of the ROBOT/TERRAIN collision groups."""
    g = ET.SubElement(parent, "geom")
    g.set("name", name)
    for k, v in {**attrs, **contact_group, **CONTACT}.items():
        g.set(k, v)
    return g


def _add_visual_meshes(root, urdf_path):
    """Cosmetic: the URDF's visual meshes as non-colliding geoms."""
    urdf = ET.parse(urdf_path).getroot()
    root.find("compiler").set("meshdir", MESHDIR)
    asset = ET.SubElement(root, "asset")
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
            e = ET.SubElement(asset, "mesh"); e.set("name", seen[rel]); e.set("file", rel)
        o = vis.find("origin")
        xyz = o.get("xyz") if (o is not None and o.get("xyz")) else "0 0 0"
        rpy = [float(v) for v in (o.get("rpy").split() if (o is not None and o.get("rpy")) else "0 0 0".split())]
        w, x, y, z = _rpy_to_quat(rpy)
        g = ET.SubElement(body, "geom")
        g.set("type", "mesh"); g.set("mesh", seen[rel]); g.set("pos", xyz)
        g.set("quat", f"{w} {x} {y} {z}")
        g.set("contype", "0"); g.set("conaffinity", "0")
        g.set("group", "1"); g.set("rgba", "0.72 0.74 0.80 1")


def build_sim_model(policy, with_visuals=True):
    """Free-base Alex: estimator MJCF + floor + SCS2's collision set + per-joint position servos.

    kd is applied as MuJoCo joint damping, which with a kp-only `position` actuator reproduces
    Java's `tau = kp*(q_d - q) + kd*(0 - qd)` closely (verified in EXPERIMENTS.md §4).
    """
    urdf = cycloid_forearm_urdf(URDF)
    root = ET.fromstring(me.alex_spec_from_urdf(urdf).mjcf)

    opt = ET.SubElement(root, "option")
    for k, v in dict(timestep=str(DT), gravity="0 0 -9.81", integrator="implicitfast",
                     solver="Newton", iterations="25", noslip_iterations="5",
                     impratio="1", cone="pyramidal").items():
        opt.set(k, v)

    _geom(root.find("worldbody"), "floor", TERRAIN_GROUP, type="plane", size="20 20 0.1")
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
        a = ET.SubElement(act, "position")
        a.set("name", n); a.set("joint", n); a.set("kp", repr(kp))
        a.set("forcelimited", "true"); a.set("forcerange", f"{-tau} {tau}")

    if with_visuals:
        _add_visual_meshes(root, urdf)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


# ---------------------------------------------------------------------------
# Index maps, observations, geometry helpers
# ---------------------------------------------------------------------------
_UP = np.array([0.0, 0.0, -1.0])


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


def build_obs(m, d, policy, maps, cmd, last_action):
    """The observation vector, term-by-term in the order the policy's `observations` declares.

    cmd = [vx, vy, yaw_rate, standing, base_height].
    """
    bid = maps["BASE_BID"]
    out = []
    for term in policy["obs_terms"]:
        if term == "base_ang_vel":
            # mjOBJ_XBODY, NOT mjOBJ_BODY. With flg_local=1 MuJoCo resolves mjOBJ_BODY in the body's
            # INERTIAL frame, and Alex's pelvis inertia frame is ~180 deg about (1,0,1)/sqrt(2) — so
            # mjOBJ_BODY hands back a gyro with x/z swapped and y negated. This was THE bug that
            # made every policy fall; see EXPERIMENTS.md. mjOBJ_XBODY uses the body frame, matching
            # Java's RLEstimates.root_AngularVelocity to 1e-4.
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, v6, 1)
            out.append(v6[:3])
        elif term == "projected_gravity":
            out.append(d.xmat[bid].reshape(3, 3).T @ _UP)
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


def lowest_foot_to_root_height(m, d, maps):
    """Java's `RLEstimates.lowestFootToRootHeight`: root height above the LOWER sole plane."""
    return d.qpos[2] - min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)][2] - ANKLE_HEIGHT
                           for b in ("LEFT_FOOT", "RIGHT_FOOT"))


def foot_rest_height(m, maps, home_full):
    """Pelvis z putting the lowest foot-box corner ~1 mm off the floor at the home pose."""
    d = mujoco.MjData(m)
    d.qpos[maps["ALL_QADR"]] = home_full
    d.qpos[0:3] = [0.0, 0.0, 1.0]; d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS]
    return 1.0 - min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in gids) + 0.001


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
# GLFW key codes the viewer passes to key_callback.
KEY_SPACE, KEY_LEFT_SHIFT, KEY_RIGHT_SHIFT = 32, 340, 344
VX_STEP, VY_STEP, YAW_STEP, HEIGHT_STEP = 0.1, 0.1, 0.1, 0.02
VX_MAX, VY_MAX, YAW_MAX = 0.9, 0.5, 1.5
HEIGHT_RANGE = (0.55, 1.00)


class Loop:
    """Sim state, the live command, and one 50 Hz control tick."""

    def __init__(self, m, policy, maps):
        self.m, self.policy, self.maps = m, policy, maps
        self.sess, self.scale = policy["sess"], policy["action_scale"]
        self.n = len(policy["order"])

        self.d = mujoco.MjData(m)
        self.d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
        self.d.qpos[0:3] = [0.0, 0.0, foot_rest_height(m, maps, maps["ALL_HOME"])]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        mujoco.mj_forward(m, self.d)

        self.last_action = np.zeros(self.n, np.float32)
        # RLHeightManager.goHome(): ease from wherever the root is to `base_height`.
        self.height_target = policy["base_height"]
        self._ramp_from = lowest_foot_to_root_height(m, self.d, maps)
        self._ramp_t = 0.0
        # [vx, vy, yaw_rate, standing, base_height]
        self.cmd = np.array([0.0, 0.0, 0.0, 1.0, self._ramp_from])

    # -- height command (RLHeightManager) ------------------------------------
    def _height(self):
        """Zero-end-velocity cubic from `_ramp_from` to `height_target` over GO_HOME_DURATION."""
        tau = min(self._ramp_t / GO_HOME_DURATION, 1.0)
        blend = 1.0 - 3.0 * tau ** 2 + 2.0 * tau ** 3
        return self.height_target + (self._ramp_from - self.height_target) * blend

    def set_height_target(self, target):
        """Re-seed the ramp from the current command, as `RLHeightManager.goHome()` does."""
        self._ramp_from = self._height()
        self._ramp_t = 0.0
        self.height_target = float(np.clip(target, *HEIGHT_RANGE))

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
    def key(self, keycode):
        """W/S = +/-vx, A/D = +/-vy (y is LEFT), Q/E = +/-yaw rate, X = stop,
        SPACE/SHIFT = raise/lower the commanded base height, T = toggle the standing flag.

        Any nonzero velocity command clears the standing flag: the policy's
        `base_velocity_plus_standing[3]` gates walking, so holding it while commanding vx does
        nothing. T is there to force it back by hand.
        """
        c = self.cmd
        if keycode == ord("W"):
            c[0] = np.clip(c[0] + VX_STEP, -VX_MAX, VX_MAX)
        elif keycode == ord("S"):
            c[0] = np.clip(c[0] - VX_STEP, -VX_MAX, VX_MAX)
        elif keycode == ord("A"):
            c[1] = np.clip(c[1] + VY_STEP, -VY_MAX, VY_MAX)
        elif keycode == ord("D"):
            c[1] = np.clip(c[1] - VY_STEP, -VY_MAX, VY_MAX)
        elif keycode == ord("Q"):
            c[2] = np.clip(c[2] + YAW_STEP, -YAW_MAX, YAW_MAX)
        elif keycode == ord("E"):
            c[2] = np.clip(c[2] - YAW_STEP, -YAW_MAX, YAW_MAX)
        elif keycode == ord("X"):
            c[0:3] = 0.0
        elif keycode == ord("T"):
            c[3] = 1.0 - c[3]
            return
        elif keycode == KEY_SPACE:
            self.set_height_target(self.height_target + HEIGHT_STEP)
            return
        elif keycode in (KEY_LEFT_SHIFT, KEY_RIGHT_SHIFT):
            self.set_height_target(self.height_target - HEIGHT_STEP)
            return
        else:
            return
        c[3] = 0.0 if np.any(np.abs(c[0:3]) > 1e-9) else 1.0

    def status(self):
        c = self.cmd
        return (f"cmd vx={c[0]:+.1f} vy={c[1]:+.1f} yaw={c[2]:+.1f} stand={c[3]:.0f} "
                f"h={c[4]:.3f}->{self.height_target:.3f} | z={self.d.qpos[2]:+.3f} "
                f"tilt={self.tilt_deg():5.1f}deg |a|={np.abs(self.last_action).max():5.2f} "
                f"ncon={self.d.ncon}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_headless(policy_name, ticks):
    policy = load_policy(policy_name)
    m = build_sim_model(policy, with_visuals=False)
    loop = Loop(m, policy, make_maps(m, policy))
    print(f"policy={policy_name} obs={policy['input_size']} njoints={loop.n} "
          f"scale={loop.scale} height_target={loop.height_target}")
    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} mass={m.body_mass.sum():.6f} "
          f"start pelvis_z={loop.d.qpos[2]:.3f}")
    for k in range(ticks):
        loop.control_tick()
        if k % 25 == 0:
            print(f"  t={k * DECIMATION * DT:5.2f}s  {loop.status()}")
    print(f"final {loop.status()}  finite={np.all(np.isfinite(loop.d.qpos))}")


def run_viewer(policy_name):
    import mujoco.viewer
    policy = load_policy(policy_name)
    m = build_sim_model(policy)
    loop = Loop(m, policy, make_maps(m, policy))
    print(f"Viewer: policy={policy_name}.  WASD = walk (W/S vx, A/D vy), Q/E = turn, X = stop,\n"
          f"        SPACE / SHIFT = raise / lower base height, T = toggle standing flag.")
    with mujoco.viewer.launch_passive(m, loop.d, key_callback=loop.key) as v:
        last_print = 0.0
        while v.is_running():
            t0 = time.time()
            loop.control_tick()
            v.sync()
            if t0 - last_print > 0.5:
                print(f"  {loop.status()}", end="\r", flush=True)
                last_print = t0
            sleep = DECIMATION * DT - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="standing", choices=list(POLICIES))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=400)
    args = ap.parse_args()
    (run_headless(args.policy, args.ticks) if args.headless else run_viewer(args.policy))
