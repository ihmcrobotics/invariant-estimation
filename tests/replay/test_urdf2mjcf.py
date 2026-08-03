r"""
Gate G1 for the parity harness: the converted model must *be* the logged robot.

The parity comparison is only meaningful if the Python side's kinematics and
inertia match the description the Java estimator ran on.  These tests check the
converter against the URDF directly -- never against MuJoCo's own reading of it,
which would be circular.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from invariant_estimation.model.urdf2mjcf import _rpy_to_quat, urdf_to_mjcf


# --------------------------------------------------------------------------
# an independent URDF forward-kinematics implementation (the oracle)
# --------------------------------------------------------------------------


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    )
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def _axis_rot(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def urdf_fk(urdf_xml: str, q: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Link name -> (position, rotation) in the root frame, straight from the XML.

    Deliberately naive: walk the joint tree composing 4x4s. Its only job is to be
    obviously correct, so a disagreement with MuJoCo indicts the converter.
    """
    root = ET.fromstring(urdf_xml)
    joints = root.findall("joint")
    children: dict[str, list] = {}
    child_links = set()
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        children.setdefault(p, []).append(j)
        child_links.add(c)
    base = [lk.get("name") for lk in root.findall("link")
            if lk.get("name") not in child_links][0]

    out = {base: (np.zeros(3), np.eye(3))}
    stack = [base]
    while stack:
        link = stack.pop()
        pos, rot = out[link]
        for j in children.get(link, []):
            o = j.find("origin")
            xyz = np.array(
                [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
            )
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
            j_rot = rot @ _rot(rpy)
            j_pos = pos + rot @ xyz
            if j.get("type") in ("revolute", "continuous"):
                ax = [float(v) for v in j.find("axis").get("xyz").split()]
                j_rot = j_rot @ _axis_rot(ax, q.get(j.get("name"), 0.0))
            c = j.find("child").get("link")
            out[c] = (j_pos, j_rot)
            stack.append(c)
    return out


# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def urdf_text(log_dir):
    return (log_dir / "model.sdf").read_text()


def test_forward_kinematics_matches_the_urdf(urdf_text, model_spec):
    """MuJoCo FK on the converted model == FK read straight off the URDF.

    Checked at 20 random configurations rather than the zero pose, because the
    zero pose hides exactly the two mistakes most likely to be made: a wrong
    joint axis and a rotation-order error in the ``rpy`` conversion (both vanish
    when every angle is 0).
    """
    m = mujoco.MjModel.from_xml_string(model_spec.mjcf)
    d = mujoco.MjData(m)
    hinges = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(m.njnt)
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    rng = np.random.default_rng(20260717)

    worst_pos = worst_rot = 0.0
    for _ in range(20):
        q = {j: float(rng.uniform(-1.0, 1.0)) for j in hinges}
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        for j, v in q.items():
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
        mujoco.mj_kinematics(m, d)

        truth = urdf_fk(urdf_text, q)
        for link, (pos, rot) in truth.items():
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            if bid < 0:
                continue
            worst_pos = max(worst_pos, np.abs(d.xpos[bid] - pos).max())
            worst_rot = max(worst_rot, np.abs(d.xmat[bid].reshape(3, 3) - rot).max())

    assert worst_pos < 1e-10, f"position mismatch {worst_pos:.2e}"
    assert worst_rot < 1e-10, f"rotation mismatch {worst_rot:.2e}"


def test_total_mass_matches_the_urdf(urdf_text, model_spec):
    """Mass is conserved through the conversion, including the massless frames."""
    root = ET.fromstring(urdf_text)
    expected = sum(
        float(lk.find("inertial").find("mass").get("value"))
        for lk in root.findall("link")
        if lk.find("inertial") is not None
    )
    m = mujoco.MjModel.from_xml_string(model_spec.mjcf)
    assert m.body_mass.sum() == pytest.approx(expected, abs=1e-9)


def test_massless_sensor_frames_do_not_perturb_inertia(urdf_text):
    """The placeholder mass on the 19 zero-mass sensor frames is inert.

    They are pure coordinate frames in the URDF (mass 0, zero tensor). MuJoCo
    cannot represent that, so they carry ~1e-12 kg. This asserts the substitution
    is below float64 noise in M(q) -- i.e. that the frames really are frames,
    not a silent inertia injection.
    """
    heavy = urdf_to_mjcf(urdf_text).mjcf
    # Shrink the placeholder by 1000x. M(q) must not move: that is what "the
    # value is arbitrary because the frames are inertia-less" *means*. (Comparing
    # against literal zero is impossible -- MuJoCo rejects it, which is the whole
    # reason a placeholder exists.)
    lighter = heavy.replace('mass="1e-12"', 'mass="1e-15"')
    assert lighter != heavy, "expected placeholder masses in the emitted MJCF"

    def mass_matrix(xml):
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        return M

    a, b = mass_matrix(heavy), mass_matrix(lighter)
    # Bound it *relative to the robot's own mass*: the absolute change is the
    # total placeholder mass (19 frames x 1e-12 kg), which is only meaningful
    # next to the ~90 kg it sits in.
    assert np.abs(a - b).max() / a[0, 0] < 1e-12


def test_armature_carries_the_rotor_table(urdf_text):
    """Rotor inertia reaches ``qM`` as armature -- and only as armature."""
    table = {"HIP_X": 0.062, "KNEE": 0.167, "SPINE": 0.062}
    spec = urdf_to_mjcf(urdf_text, rotor_inertia=table, rotor_inertia_default=0.005)
    m = mujoco.MjModel.from_xml_string(spec.mjcf)

    def armature(joint):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        return m.dof_armature[m.jnt_dofadr[jid]]

    assert armature("LEFT_HIP_X") == pytest.approx(0.062)
    assert armature("LEFT_KNEE_Y") == pytest.approx(0.167)
    assert armature("SPINE_Z") == pytest.approx(0.062)
    assert armature("LEFT_ANKLE_Y") == pytest.approx(0.005)  # unmatched -> default

    # The floating base never carries reflected rotor inertia.
    assert np.all(m.dof_armature[:6] == 0.0)


def test_effort_limits_are_carried_out_of_the_urdf(model_spec):
    """``sigma_tau,i = alpha_i * tau_max,i`` needs these, and MJCF has nowhere
    to put them -- so the converter returns them as data."""
    assert model_spec.effort_limits["LEFT_KNEE_Y"] == pytest.approx(217.2)
    assert len(model_spec.effort_limits) == 29


def test_rpy_conversion_is_fixed_axis():
    """URDF ``rpy`` is extrinsic XYZ. The Alex IMUs are mounted at yaw=+90 deg,
    so an intrinsic/extrinsic mix-up here is a 90 deg frame error, not a rounding
    difference -- this pins the convention against a hand-computed case."""
    w, x, y, z = _rpy_to_quat((0.0, 0.0, np.pi / 2))
    assert (w, x, y) == pytest.approx((np.cos(np.pi / 4), 0.0, 0.0), abs=1e-15)
    assert z == pytest.approx(np.sin(np.pi / 4), abs=1e-15)

    # A case where the two conventions genuinely differ.
    rpy = (0.3, -0.2, 1.1)
    q = _rpy_to_quat(rpy)
    R_from_quat = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(R_from_quat)
    mujoco.mj_kinematics(R_from_quat, d)
    assert np.abs(d.xmat[1].reshape(3, 3) - _rot(rpy)).max() < 1e-12
