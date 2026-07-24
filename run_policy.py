"""
run_policy.py — IHMC RL policies in a standalone MuJoCo sim, driven live.

Runs an exported ONNX policy DIRECTLY via onnxruntime (no JAX, no port) inside a plain
MuJoCo loop, with ground-truth observations. Live command via the keyboard in the
viewer (WASD/QE); an Xbox gamepad can drop into `read_gamepad()` later.

    uv run python run_policy.py                      # viewer, default = standing demo
    uv run python run_policy.py --policy baseline    # the walking_baseline policy
    uv run python run_policy.py --headless           # no window, prints diagnostics

The harness is policy-agnostic: it reads each policy's `policy_cfg.yaml` (observation
list, 29/23-joint order, per-joint kp/kd/home/effort, action scale) and assembles the
observation vector term-by-term in the order the config declares. Verified against the
IsaacLab training env (`Isaac-*-Alex-*`), the exported yaml, and IHMC's Java oracle:
obs are UNSCALED; `target = home + action_scale * action`; joint order is the yaml order
(IsaacLab breadth-first). The `20251219_standing18` policy BALANCES in this harness (the
proof the sim/obs/action/order are correct); WalkingUneven policies are more marginal —
see POLICY_DEBUG.md.

Obs terms (built only if present in a policy's `observations` list):
  base_ang_vel(3, pelvis body frame), projected_gravity(3, unit, body),
  base_velocity_plus_standing(4)=[vx,vy,yaw,stand], base_height(1, commanded),
  joint_pos_rel(n)=q-home, joint_vel_rel(n)=qd, last_action(n).
"""
import argparse
import os
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
# The v1 visual meshes the v2 URDF references (package://alex_virtual_description/...).
MESHDIR = "/home/llibshutz/workspaces/robot-stuff/ihmc-alex-sdk/alex-models/alex_virtual_description"

DT = 0.005            # physics timestep (200 Hz)
DECIMATION = 4        # -> 50 Hz control

# Policy registry. `base_height` is the commanded-height obs (a genuinely sensitive input:
# the standing policy holds at 0.75 but falls at 0.70/0.79). A dir means <dir>/policy_cfg.yaml
# + <dir>/policy.onnx; otherwise explicit yaml+onnx filenames in RL_MODELS.
POLICIES = {
    "standing": {"yaml": "20251219_standing18_def.yaml", "onnx": "20251219_standing18.onnx",
                 "base_height": 0.75},
    "baseline": {"dir": "2026-07-10_baseline", "base_height": 0.90},
    "forearms": {"dir": "2026-04-13_002_forearms", "base_height": 0.90},
}

# Fallback gains/home for joints a given policy does NOT drive (e.g. wrists/grippers for the
# 23-joint standing policy): held at home by the baseline's PD so they don't flop.
_FALLBACK = {p["name"]: p for p in
             yaml.safe_load(open(f"{RL_MODELS}/2026-07-10_baseline/policy_cfg.yaml"))["jointParameters"]}


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policy(name):
    spec = POLICIES[name]
    if "dir" in spec:
        ybase, obase = f"{spec['dir']}/policy_cfg.yaml", f"{spec['dir']}/policy.onnx"
    else:
        ybase, obase = spec["yaml"], spec["onnx"]
    cfg = yaml.safe_load(open(f"{RL_MODELS}/{ybase}"))
    jp = cfg["jointParameters"]
    return {
        "name": name,
        "order": [p["name"] for p in jp],                            # policy's joint order
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
# Model building (policy-aware: policy joints use the policy's gains, others the fallback)
# ---------------------------------------------------------------------------
def add_visual_meshes(root):
    """Wire the URDF's visual meshes into the MJCF as NON-colliding geoms (cosmetic)."""
    urdf = ET.parse(URDF).getroot()
    comp = root.find("compiler")
    (comp if comp is not None else ET.SubElement(root, "compiler")).set("meshdir", MESHDIR)
    asset = root.find("asset")
    if asset is None or asset.tag != "asset":
        asset = ET.SubElement(root, "asset")
    bodies = {b.get("name"): b for b in root.iter("body")}
    seen = {}
    for link in urdf.findall("link"):
        body = bodies.get(link.get("name"))
        vis = link.find("visual")
        mesh = vis.find("geometry/mesh") if vis is not None else None
        if body is None or mesh is None:
            continue
        rel = mesh.get("filename").replace("package://alex_virtual_description/", "")
        if not os.path.exists(os.path.join(MESHDIR, rel)):
            continue
        name = seen.get(rel)
        if name is None:
            name = f"mesh{len(seen)}"
            seen[rel] = name
            e = ET.SubElement(asset, "mesh"); e.set("name", name); e.set("file", rel)
        o = vis.find("origin")
        xyz = o.get("xyz") if (o is not None and o.get("xyz")) else "0 0 0"
        rpy = [float(v) for v in (o.get("rpy").split() if (o is not None and o.get("rpy")) else (0, 0, 0))]
        w, x, y, z = _rpy_to_quat(rpy)
        g = ET.SubElement(body, "geom")
        g.set("type", "mesh"); g.set("mesh", name)
        g.set("pos", " ".join(map(str, [float(v) for v in xyz.split()])))
        g.set("quat", f"{w} {x} {y} {z}")
        g.set("contype", "0"); g.set("conaffinity", "0")
        g.set("group", "1"); g.set("rgba", "0.72 0.74 0.80 1")


def build_sim_model(policy, with_visuals=True):
    """Free-base Alex sim: estimator MJCF + floor + foot boxes + per-joint position servos.

    Every controllable joint gets a position servo. Joints the policy drives use the
    policy's kp/kd; joints it does not (wrists/grippers for the 23-DoF standing policy) use
    the baseline fallback so they hold home instead of flopping. Physics config and the foot
    collision box are copied from IHMC's SCS2 MuJoCo sim (verified bit-for-bit).
    """
    root = ET.fromstring(me.alex_spec_from_urdf(URDF).mjcf)
    opt = ET.SubElement(root, "option")
    opt.set("timestep", str(DT)); opt.set("gravity", "0 0 -9.81")
    opt.set("integrator", "implicitfast"); opt.set("solver", "Newton")
    opt.set("iterations", "25"); opt.set("noslip_iterations", "5")
    opt.set("impratio", "1"); opt.set("cone", "pyramidal")

    CONTACT = dict(condim="4", friction="1 0.05 0.01",
                   solref="0.02 1", solimp="0.9 0.99 0.0007 0.5 2")

    wb = root.find("worldbody")
    fl = ET.SubElement(wb, "geom")
    fl.set("name", "floor"); fl.set("type", "plane"); fl.set("size", "5 5 0.1")
    fl.set("contype", "1"); fl.set("conaffinity", "1")
    for k, v in CONTACT.items():
        fl.set(k, v)

    def foot(body_name):
        for b in root.iter("body"):
            if b.get("name") == body_name:
                g = ET.SubElement(b, "geom")
                g.set("type", "box"); g.set("size", "0.11 0.05 0.01"); g.set("pos", "0.05 0 -0.06")
                g.set("contype", "1"); g.set("conaffinity", "1")
                for k, v in CONTACT.items():
                    g.set(k, v)
                return
        raise SystemExit(f"body {body_name} not found")
    foot("LEFT_FOOT"); foot("RIGHT_FOOT")

    def gains(n):
        """(kp, kd, tau) for joint n: policy's if it drives it, else baseline fallback."""
        if n in policy["kp"]:
            return policy["kp"][n], policy["kd"][n], policy["tau"][n]
        p = _FALLBACK[n]
        return float(p["kp"]), float(p["kd"]), float(p["maxEffort"])

    for j in root.iter("joint"):
        n = j.get("name")
        if n in _FALLBACK:
            j.set("damping", repr(gains(n)[1]))
    act = ET.SubElement(root, "actuator")
    for j in root.iter("joint"):
        n = j.get("name")
        if n in _FALLBACK:
            kp, _, tau = gains(n)
            a = ET.SubElement(act, "position")
            a.set("name", n); a.set("joint", n); a.set("kp", repr(kp))
            a.set("forcelimited", "true"); a.set("forcerange", f"{-tau} {tau}")

    if with_visuals:
        add_visual_meshes(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


# ---------------------------------------------------------------------------
# Index maps + observation assembly
# ---------------------------------------------------------------------------
_UP = np.array([0.0, 0.0, -1.0])


def make_maps(m, policy):
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    order = policy["order"]
    return {
        "QADR": np.array([m.jnt_qposadr[jid(n)] for n in order]),       # policy-order q
        "DOFADR": np.array([m.jnt_dofadr[jid(n)] for n in order]),      # policy-order qd
        "AID": np.array([aid(n) for n in order]),                       # policy-order actuators
        "HOME": np.array([policy["home"][n] for n in order]),
        "ALL_AID": np.array([aid(n) for n in _FALLBACK]),               # every actuator
        "ALL_HOME": np.array([policy["home"].get(n, float(_FALLBACK[n]["homePosition"]))
                              for n in _FALLBACK]),
        "ALL_QADR": np.array([m.jnt_qposadr[jid(n)] for n in _FALLBACK]),
        "BASE_BID": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "PELVIS_LINK"),
    }


def build_obs(m, d, policy, maps, cmd, last_action):
    """Assemble the observation vector term-by-term per the policy's `observations` list.

    cmd = [vx, vy, yaw, stand, base_height]; the standing policy ignores the first four.
    """
    bid = maps["BASE_BID"]
    out = []
    for term in policy["obs_terms"]:
        if term == "base_ang_vel":
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 1)
            out.append(v6[:3])                                          # pelvis body-frame omega
        elif term == "projected_gravity":
            out.append(d.xmat[bid].reshape(3, 3).T @ _UP)               # unit gravity in body frame
        elif term == "base_velocity_plus_standing":
            out.append(cmd[:4])                                         # [vx, vy, yaw, stand]
        elif term == "base_height":
            out.append(cmd[4:5])                                        # commanded height
        elif term == "joint_pos_rel":
            out.append(d.qpos[maps["QADR"]] - maps["HOME"])
        elif term == "joint_vel_rel":
            out.append(d.qvel[maps["DOFADR"]])
        elif term == "last_action":
            out.append(last_action)
        else:
            raise ValueError(f"unknown obs term {term}")
    return np.concatenate(out).astype(np.float32)


def foot_rest_height(m, maps, home_full):
    """Pelvis z that puts the lowest foot-box corner ~1 mm above the floor at the home pose."""
    d = mujoco.MjData(m)
    d.qpos[maps["ALL_QADR"]] = home_full
    d.qpos[0:3] = [0.0, 0.0, 1.0]; d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    bottoms = [d.geom_xpos[g][2] - m.geom_size[g][2]
               for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    return 1.0 - min(bottoms) + 0.001


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
class Loop:
    """Holds the sim state, the live command, and one control tick for a given policy."""

    def __init__(self, m, policy, maps):
        self.m, self.policy, self.maps = m, policy, maps
        self.sess = policy["sess"]
        self.n = len(policy["order"])
        self.scale = policy["action_scale"]
        self.d = mujoco.MjData(m)
        self.d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
        self.d.qpos[0:3] = [0.0, 0.0, foot_rest_height(m, maps, maps["ALL_HOME"])]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        mujoco.mj_forward(m, self.d)
        self.last_action = np.zeros(self.n, np.float32)
        # [vx, vy, yaw, stand, base_height]
        self.cmd = np.array([0.0, 0.0, 0.0, 1.0, policy["base_height"]])

    def policy_action(self, obs):
        name = self.sess.get_inputs()[0].name
        return self.sess.run(None, {name: obs[None].astype(np.float32)})[0][0]

    def control_tick(self):
        obs = build_obs(self.m, self.d, self.policy, self.maps, self.cmd, self.last_action)
        action = self.policy_action(obs)
        self.last_action = action
        # non-policy joints stay at home; policy joints track home + scale*action
        self.d.ctrl[self.maps["ALL_AID"]] = self.maps["ALL_HOME"]
        self.d.ctrl[self.maps["AID"]] = self.maps["HOME"] + self.scale * action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)

    def tilt_deg(self):
        c = self.d.xmat[self.maps["BASE_BID"]].reshape(3, 3)[2, 2]
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    # -- live command --------------------------------------------------------
    def key(self, keycode):
        """Keyboard command (viewer). WASD = vx/vy, QE = yaw, X = zero, SPACE = stand toggle."""
        c = self.cmd
        if keycode == ord("W"):
            c[0] = np.clip(c[0] + 0.1, -0.9, 0.9)
        elif keycode == ord("S"):
            c[0] = np.clip(c[0] - 0.1, -0.9, 0.9)
        elif keycode == ord("A"):
            c[1] = np.clip(c[1] + 0.1, -0.5, 0.5)
        elif keycode == ord("D"):
            c[1] = np.clip(c[1] - 0.1, -0.5, 0.5)
        elif keycode == ord("Q"):
            c[2] = np.clip(c[2] + 0.1, -1.5, 1.5)
        elif keycode == ord("E"):
            c[2] = np.clip(c[2] - 0.1, -1.5, 1.5)
        elif keycode == ord("X"):
            c[0:3] = 0.0
        elif keycode == ord(" "):
            c[3] = 1.0 - c[3]

    def apply_gamepad(self):
        gp = read_gamepad()
        if gp is not None:
            self.cmd[0], self.cmd[1], self.cmd[2] = gp


def read_gamepad():
    """Return (vx, vy, yaw) from an Xbox pad, or None if unavailable (stub)."""
    return None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_headless(policy_name, ticks):
    policy = load_policy(policy_name)
    m = build_sim_model(policy)
    maps = make_maps(m, policy)
    loop = Loop(m, policy, maps)
    print(f"policy={policy_name} obs={policy['input_size']} njoints={loop.n} "
          f"scale={loop.scale} base_height={loop.cmd[4]:.3f}")
    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu}  start pelvis_z={loop.d.qpos[2]:.3f}")
    for k in range(ticks):
        loop.control_tick()
        if k % 25 == 0:
            d = loop.d
            print(f"  t={k * DECIMATION * DT:4.2f}s  pelvis_z={d.qpos[2]:+.3f}  "
                  f"tilt={loop.tilt_deg():5.1f}deg  |action|={np.abs(loop.last_action).max():.2f}  "
                  f"ncon={d.ncon}  finite={np.all(np.isfinite(d.qpos))}")
    print(f"final pelvis_z={loop.d.qpos[2]:.3f}  tilt={loop.tilt_deg():.1f}  "
          f"finite={np.all(np.isfinite(loop.d.qpos))}")


def run_viewer(policy_name):
    import mujoco.viewer
    policy = load_policy(policy_name)
    m = build_sim_model(policy)
    maps = make_maps(m, policy)
    loop = Loop(m, policy, maps)
    print(f"Viewer: policy={policy_name}. WASD = walk (vx/vy), Q/E = turn, X = stop, "
          f"SPACE = stand toggle.")
    with mujoco.viewer.launch_passive(m, loop.d, key_callback=loop.key) as v:
        while v.is_running():
            t0 = time.time()
            loop.apply_gamepad()
            loop.control_tick()
            v.sync()
            sleep = DECIMATION * DT - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="standing", choices=list(POLICIES),
                    help="which policy to run (default: standing = the working demo)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=200)
    args = ap.parse_args()
    (run_headless(args.policy, args.ticks) if args.headless else run_viewer(args.policy))
