"""Diff our MuJoCo harness against the Java (SCS2-MuJoCo) reference. Three independent checks.

These are the tools that found the `mjOBJ_BODY` gyro bug. Keep them runnable: each one isolates a
different layer, so when a policy misbehaves you can tell *which* layer disagrees instead of
guessing. See ../EXPERIMENTS.md for what each has already ruled out.

    uv run python experiments/java_parity.py dynamics   # rigid-body model vs Java's compiled MJCF
    uv run python experiments/java_parity.py obs        # observation vector + ONNX action vs Java
    uv run python experiments/java_parity.py openloop   # replay Java's setpoints, compare state

`dynamics` needs only `/tmp/scs2-mujoco-*/world.xml`, which SCS2 writes whenever its MuJoCo engine
compiles a world. `obs` and `openloop` additionally need `/tmp/alex_java_obs_dump.csv` from
`AlexMujocoObsDumpTest` in the alex repo (recipe in ../EXPERIMENTS.md §6).
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_policy as rp                                                   # noqa: E402

DUMP = "/tmp/alex_java_obs_dump.csv"
WORLD_GLOB = "/tmp/scs2-mujoco-*/world.xml"
STEADY_TICK = 250                                                         # t = 5 s, fully settled


def _java_world():
    paths = sorted(glob.glob(WORLD_GLOB))
    if not paths:
        raise SystemExit(f"no {WORLD_GLOB} — run AlexMujocoObsDumpTest first (EXPERIMENTS.md §6)")
    return paths[-1]


def _dump():
    if not Path(DUMP).exists():
        raise SystemExit(f"no {DUMP} — run AlexMujocoObsDumpTest first (EXPERIMENTS.md §6)")
    rows = list(csv.DictReader(open(DUMP)))
    joints = [k[len("residual_"):] for k in rows[0] if k.startswith("residual_")]
    return rows, joints


def _ours(policy_name="baseline"):
    pol = rp.load_policy(policy_name)
    m = rp.build_sim_model(pol, with_visuals=False)
    return pol, m, rp.make_maps(m, pol)


def _seed(m, row, hinge_names, strip=""):
    """Put a model at the dump's configuration (positions and velocities)."""
    d = mujoco.MjData(m)
    for n in hinge_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, (strip + n) if strip else n)
        if jid < 0 or ("q_" + n) not in row:
            continue
        d.qpos[m.jnt_qposadr[jid]] = float(row["q_" + n])
        d.qvel[m.jnt_dofadr[jid]] = float(row["qd_" + n])
    d.qpos[0:3] = [float(row["rootX"]), float(row["rootY"]), float(row["rootZ"])]
    d.qpos[3:7] = [float(row[c]) for c in ("rootQs", "rootQx", "rootQy", "rootQz")]
    mujoco.mj_forward(m, d)
    return d


# ---------------------------------------------------------------------------
def check_dynamics():
    """Mass matrix, bias force and CoM against Java's own compiled MJCF. No sim, no contact.

    This is the strongest check available: it removes actuation, contact and integration entirely,
    so any disagreement is kinematics or inertia. Expect equality to ~1e-11 once our rotor armature
    is subtracted — Java's MJCF sets `<joint armature="0.0"/>` globally.
    """
    world = _java_world()
    mj = mujoco.MjModel.from_xml_path(world)
    _pol, mo, _maps = _ours()
    print(f"java {world}\n  nbody={mj.nbody} njnt={mj.njnt} nv={mj.nv} mass={mj.body_mass.sum():.6f}")
    print(f"ours\n  nbody={mo.nbody} njnt={mo.njnt} nv={mo.nv} mass={mo.body_mass.sum():.6f}")
    print("  (Java collapses fixed joints, hence fewer bodies at the same nv)")

    def dofmap(m, strip=""):
        out = {}
        for i in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n is None:
                continue
            key = n[len(strip):] if strip and n.startswith(strip) else n
            out[key] = (m.jnt_dofadr[i], m.jnt_type[i])
        return out

    jdof, odof = dofmap(mj, "Alex_"), dofmap(mo)
    hinge = sorted(n for n in odof if n in jdof and odof[n][1] == mujoco.mjtJoint.mjJNT_HINGE)

    def free_adr(m):
        return next(m.jnt_dofadr[i] for i in range(m.njnt)
                    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)

    fj, fo = free_adr(mj), free_adr(mo)
    perm_j = list(range(fj, fj + 6)) + [jdof[n][0] for n in hinge]
    perm_o = list(range(fo, fo + 6)) + [odof[n][0] for n in hinge]
    names = ["base_wx", "base_wy", "base_wz", "base_vx", "base_vy", "base_vz"] + hinge
    print(f"  matched {len(hinge)} hinges by name\n")

    rows, _ = _dump()
    dj = _seed(mj, rows[STEADY_TICK], hinge, strip="Alex_")
    do = _seed(mo, rows[STEADY_TICK], hinge)

    def full_M(m, d):
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        return M

    Mj = full_M(mj, dj)[np.ix_(perm_j, perm_j)]
    Mo = full_M(mo, do)[np.ix_(perm_o, perm_o)]
    arm = np.array([mo.dof_armature[i] for i in perm_o])

    for tag, A in (("as built", Mo), ("our armature subtracted", Mo - np.diag(arm))):
        D = np.abs(A - Mj)
        i, j = np.unravel_index(np.argmax(D), D.shape)
        print(f"  mass matrix, {tag:24s}: max|dM| = {D.max():.3e}  "
              f"(rel {D.max() / np.abs(Mj).max():.2e})")
        if D.max() > 1e-9:
            print(f"      worst M[{names[i]}, {names[j]}]  java={Mj[i, j]:+.6f} ours={A[i, j]:+.6f}")
    B = np.abs(do.qfrc_bias[perm_o] - dj.qfrc_bias[perm_j])
    print(f"  bias force (gravity+Coriolis)        : max|dB| = {B.max():.3e}")
    print(f"  subtree CoM                          : max|d|  = "
          f"{np.abs(dj.subtree_com[0] - do.subtree_com[0]).max():.3e}")
    print(f"\n  our armature on hinges: max {arm[6:].max():.3f} "
          f"(a deliberate deviation; the estimator needs the rotor table for Qa)")


# ---------------------------------------------------------------------------
def check_obs():
    """Two things: does our ONNX reproduce Java's action, and does our obs match Java's?

    The second half is what missed the gyro bug for a whole session: seed EVERY channel, including
    the free-joint twist, or a term that is zero on both sides looks like agreement.
    """
    rows, JJ = _dump()
    pol, m, maps = _ours()
    sess = pol["sess"]
    iname = sess.get_inputs()[0].name
    print(f"joint order matches the yaml: {JJ == pol['order']}")

    def java_obs(r):
        f = r.__getitem__
        return np.concatenate([
            [float(f("angVelX")), float(f("angVelY")), float(f("angVelZ"))],
            [float(f("projGX")), float(f("projGY")), float(f("projGZ"))],
            [float(f("cmdVx")), float(f("cmdVy")), float(f("cmdYaw")), float(f("cmdStand"))],
            [float(f("cmdHeight"))],
            [float(f("q_" + j)) - float(f("qhome_" + j)) for j in JJ],
            [float(f("qd_" + j)) for j in JJ],
            [float(f("lastAction_" + j)) for j in JJ],
        ]).astype(np.float32)

    err = np.array([np.abs(sess.run(None, {iname: java_obs(r)[None]})[0][0]
                           - np.array([float(r["residual_" + j]) for j in JJ])).max() for r in rows])
    print(f"\n1. our ONNX on Java's observation vs Java's action, {len(rows)} ticks:")
    print(f"   max={err.max():.3e} mean={err.mean():.3e} median={np.median(err):.3e}"
          "   (a few e-3 is the dump's one-tick sampling skew, not an error)")

    r = rows[STEADY_TICK]
    d = _seed(m, r, list(rp._FALLBACK))
    # seed the base twist too -- MuJoCo freejoint qvel is [linear WORLD; angular WORLD]
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, np.array([float(r[c]) for c in ("rootQs", "rootQx", "rootQy", "rootQz")]))
    Rm = Rm.reshape(3, 3)
    d.qvel[3:6] = Rm @ np.array([float(r["angVelX"]), float(r["angVelY"]), float(r["angVelZ"])])
    d.qvel[0:3] = Rm @ np.array([float(r["rootVelX"]), float(r["rootVelY"]), float(r["rootVelZ"])])
    mujoco.mj_forward(m, d)

    last = np.array([float(r["lastAction_" + j]) for j in JJ], np.float32)
    cmd = np.array([float(r["cmdVx"]), float(r["cmdVy"]), float(r["cmdYaw"]),
                    float(r["cmdStand"]), float(r["cmdHeight"])])
    ours, theirs = rp.build_obs(m, d, pol, maps, cmd, last), java_obs(r)
    print(f"\n2. our observation vs Java's at the same state (t={r['t']}s), term by term:")
    for name, a, b in (("base_ang_vel", 0, 3), ("projected_gravity", 3, 6), ("base_vel+stand", 6, 10),
                       ("base_height", 10, 11), ("joint_pos_rel", 11, 40),
                       ("joint_vel_rel", 40, 69), ("last_action", 69, 98)):
        dif = np.abs(ours[a:b] - theirs[a:b]).max()
        print(f"   {name:20s} max|diff| = {dif:.3e}{'   <-- MISMATCH' if dif > 1e-3 else ''}")


# ---------------------------------------------------------------------------
def check_openloop(ticks=25):
    """Replay Java's own qdes setpoints and compare our state to Java's, per joint.

    Read ONLY the first ~0.1 s: a standing biped is stabilised closed-loop through the floating
    base, so an open-loop replay cannot stay upright however faithful the plant is.
    """
    rows, JJ = _dump()
    pol, m, maps = _ours()
    d = _seed(m, rows[0], list(rp._FALLBACK))
    errs = []
    for k in range(1, ticks + 1):
        d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        d.ctrl[maps["AID"]] = np.array([float(rows[k - 1]["qdes_" + j]) for j in JJ])
        for _ in range(rp.DECIMATION):
            mujoco.mj_step(m, d)
        errs.append(d.qpos[maps["QADR"]] - np.array([float(rows[k]["q_" + j]) for j in JJ]))
    E = np.array(errs)
    print("replaying Java's setpoints, max|dq| vs Java's state:")
    for t in (0, 4, 9, ticks - 1):
        print(f"   t={(t + 1) * 0.02:5.2f}s   max|dq| = {np.abs(E[t]).max():.4f} rad")
    print("\n   worst joints at t=0.02s (sign kept — it localises the disagreement):")
    for i in np.argsort(-np.abs(E[0]))[:6]:
        print(f"     {JJ[i]:<20} {E[0, i]:+.4f}  ->  t={ticks * 0.02:.2f}s {E[ticks - 1, i]:+.4f}")


CHECKS = {"dynamics": check_dynamics, "obs": check_obs, "openloop": check_openloop}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in CHECKS:
        raise SystemExit(f"usage: java_parity.py {{{'|'.join(CHECKS)}}}")
    CHECKS[which]()
