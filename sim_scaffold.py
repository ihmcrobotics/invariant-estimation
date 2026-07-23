import numpy as np, mujoco, yaml
import xml.etree.ElementTree as ET
from invariant_estimation.pipeline import main_estimator as me

URDF = '/home/llibshutz/Documents/alex_with_imus.urdf'
PCFG = '/home/llibshutz/alex/persona_rl/projects/ihmc_data_processing/Alex001/20260611_001_ForwardWalk/policy_cfg.yaml'

xml = me.alex_spec_from_urdf(URDF).mjcf
root = ET.fromstring(xml)

opt = ET.SubElement(root, "option")
opt.set("timestep", "0.005"); opt.set("gravity", "0 0 -9.81")

# --- Step 1: floor + foot boxes (kept; inert in this fixed-base servo test) ---
wb = root.find("worldbody")
floor = ET.SubElement(wb, "geom")
floor.set("name", "floor"); floor.set("type", "plane"); floor.set("size", "5 5 0.1")
floor.set("contype", "1"); floor.set("conaffinity", "1")

def add_foot(body_name):
    for b in root.iter("body"):
        if b.get("name") == body_name:
            g = ET.SubElement(b, "geom")
            g.set("type", "box"); g.set("size", "0.11 0.05 0.01"); g.set("pos", "0.05 0 -0.06")
            g.set("contype", "1"); g.set("conaffinity", "1"); g.set("friction", "1 0.01 0.001")
            return
    raise SystemExit(f"body {body_name} not found")

add_foot("LEFT_FOOT"); add_foot("RIGHT_FOOT")

# --- Step 2: position servos + damping from the deployed policy's gains ---
params = {p["name"]: p for p in yaml.safe_load(open(PCFG))["jointParameters"]}
for j in root.iter("joint"):
    p = params.get(j.get("name"))
    if p is not None:
        j.set("damping", repr(float(p["kd"])))
act = ET.SubElement(root, "actuator")
for name, p in params.items():
    a = ET.SubElement(act, "position")
    a.set("name", name); a.set("joint", name); a.set("kp", repr(float(p["kp"])))
    a.set("forcelimited", "true")
    a.set("forcerange", f"{-float(p['maxEffort'])} {float(p['maxEffort'])}")

# --- WELD the base to the world: remove the free joint and lift the body up so the
# legs dangle in the air (no floor contact). Now gravity genuinely loads every
# joint during integration, so the servos have something to hold against -- unlike
# a post-hoc qpos pin, which lets the whole robot free-fall each step (no joint
# load, hence the bogus 0.000 tracking error). ---
for b in root.iter("body"):
    fj = b.find("freejoint")
    if fj is not None:
        b.remove(fj)
        b.set("pos", "0 0 1.2")           # dangle the (now fixed) base above the floor
        break

m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} actuators  (base welded -> no free DOFs)")
d = mujoco.MjData(m)

home = {name: float(p["homePosition"]) for name, p in params.items()}
for name, h in home.items():
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = h
for i in range(m.nu):
    d.ctrl[i] = home[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]

qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in home}
def err_deg():
    return {n: np.degrees(d.qpos[qadr[n]] - home[n]) for n in home}

print("\n-- fixed base, legs dangling; do the 29 position servos hold home? --")
for k in range(600):                       # 3 s to reach steady state
    mujoco.mj_step(m, d)
    if k % 100 == 0:
        e = {n: abs(v) for n, v in err_deg().items()}
        print(f"t={k*0.005:4.2f}s  max|q-home|={max(e.values()):6.3f} deg  finite={np.all(np.isfinite(d.qpos))}")

e = err_deg()
worst = sorted(e.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
print(f"\nsteady-state max |tracking error| = {max(abs(v) for v in e.values()):.2f} deg")
print("worst joints (signed droop, deg):", [(n, round(v, 2)) for n, v in worst])

import jax, jax.numpy as jnp
from mujoco import mjx

d0 = mujoco.MjData(m)
for name, h in home.items():
    d0.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = h
d0.qpos[2] = 1.2
ctrl0 = np.array([
    home[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]
    for i in range(m.nu)
])

d0.ctrl[:] = ctrl0
mujoco.mj_forward(m, d0) # derived quantities before transferring

mx = mjx.put_model(m)
# dx = mjx.put_data(m, d0)
dx = mjx.make_data(mx).replace(qpos=jnp.asarray(d0.qpos),ctrl=jnp.asarray(ctrl0))

# 3b - the checkpoint: one mjx.step and one mj_step

step = jax.jit(mjx.step)
dx1 = step(mx, dx)
mujoco.mj_step(m, d0)

dq = np.abs(np.asarray(dx1.qpos) - d0.qpos).max()
print(f"One step MJX vs. MuJoCo max|dqpos| = {dq:.2e}")

#3c: a jitted, scanned rolout
@jax.jit
def rollout(dx, ctrl_seq):
    def body(dx, u):
        dx = mjx.step(mx, dx.replace(ctrl=u))
        return dx, dx.qpos[2]
    return jax.lax.scan(body, dx, ctrl_seq)

T = 200
ctrl_seq = jnp.broadcast_to(jnp.asarray(ctrl0), (T, m.nu))
dxT, zs = rollout(dx, ctrl_seq)
print(f"base-z first/last: {float(zs[0]), float(zs[-1])}, finite: {bool(jnp.all(jnp.isfinite(zs)))}")

# 3d - N parallel environments = vmapoping rollout


N = 8
batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (N,) + x.shape), dx)
batch = batch.replace(qpos=batch.qpos.at[:, 2].set(jnp.linspace(1.0, 1.4, N)))
seqs = jnp.broadcast_to(ctrl_seq, (N,) + ctrl_seq.shape)
dxN, zsN = jax.vmap(rollout)(batch, seqs)
print(f"per env final base-z: {np.round(np.asarray(zsN[:, -1]),)}")
