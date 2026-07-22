r"""
model/urdf2mjcf.py
==================
Convert the IHMC robot description that ships **inside every SCS2 log directory**
(`model.sdf` -- despite the extension it is plain URDF: ``<robot>`` with
``<link>``/``<joint>`` and a few Gazebo sensor blocks) into an MJCF that MuJoCo
can compile.

Why convert the log's own model rather than use a vendored Alex MJCF
-------------------------------------------------------------------
The parity harness (`replay/`) compares this port against the Java estimator's
*logged* output from a specific hardware run.  Any difference in link inertia,
joint origin or IMU mount between the model the Java filter used and the model
the Python filter uses shows up as a parity failure that has nothing to do with
the port.  `model.sdf` is, by construction, the exact description that ran --
the logger writes it alongside the data.  Using it removes an entire class of
false negatives, at the cost of this file.

What is deliberately dropped
----------------------------
Only what cannot affect ``mj_kinematics`` / ``mj_crb``:

* ``<visual>`` and ``<collision>`` -- meshes live in ``resources.zip`` and would
  make the converter depend on unpacking it.  The estimator never queries
  geometry: it needs FK, site Jacobians and ``M(q)``, none of which read geoms.
* ``<gazebo>`` sensor noise blocks -- the filter's noise model comes from
  `config/filter_cfg.yaml`, not the description.
* Joint ``<limit>`` **ranges** -- ``mj_kinematics`` does not clamp ``qpos``, so a
  range can only change behaviour by accident.  The ``effort`` limit *is* kept,
  but returned as data (`AlexModelSpec.effort_limits`) rather than written into
  the MJCF, because that is how the filter consumes it: ``sigma_tau,i =
  alpha_i * tau_max,i`` (invariant I9).

Two conventions that are easy to get wrong
------------------------------------------
1. **URDF ``rpy`` is fixed-axis (extrinsic) XYZ**, i.e.
   ``R = Rz(yaw) Ry(pitch) Rx(roll)``.  MJCF's ``euler`` attribute defaults to
   ``eulerseq="xyz"``, which MuJoCo applies as *intrinsic* rotations, giving
   ``R = Rx Ry Rz`` -- the transpose-ish of what URDF means, and silently wrong
   for the IMU mounts (the Alex pelvis IMU is yawed +90 deg, so this is not a
   small-angle detail).  This module therefore never emits ``euler``: it
   converts ``rpy`` to a quaternion itself and emits ``quat="w x y z"``.

2. **A URDF joint's origin belongs on the MJCF child body, not the joint.**  In
   URDF the ``<origin>`` transforms parent-link frame to the joint frame, and the
   child-link frame coincides with the joint frame.  In MJCF the child ``<body>``
   carries ``pos``/``quat`` and the ``<joint>`` sits at the body origin.  So the
   mapping is body-pos := joint-origin, joint-pos := 0, axis unchanged.

Armature
--------
Reflected rotor inertia is written as the MJCF ``armature`` attribute, resolved
from `config/filter_cfg.yaml`'s substring table.  This is the *only* place rotor
inertia enters: MuJoCo folds ``armature`` into ``qM`` before the Schur complement
sees it, so ``Lambda`` computed downstream is already ``Lambda_eff``.  Adding it
again post-Schur is the double-add trap (`CLAUDE.md` §6) -- which is why
`jointKF.process` defaults ``rotor=ROTOR_IN_MASS_MATRIX``.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = ["AlexModelSpec", "urdf_to_mjcf", "convert_log_model"]

# A welded, inertia-less link (several IMU mounts are pure frames) still has to
# be a body so its site lands in the right place. MuJoCo rejects a body with
# neither inertial nor geom, so such bodies get this placeholder mass. It is
# ~18 orders below the smallest real Alex link mass, so it cannot perturb M(q)
# above float64 round-off -- asserted in tests/model/test_urdf2mjcf.py.
_MASSLESS = 1e-12


@dataclass(frozen=True)
class AlexModelSpec:
    """The MJCF plus everything the URDF knew that MJCF has no place for."""

    mjcf: str
    """MJCF XML source, ready for `mujoco.MjModel.from_xml_string`."""

    effort_limits: dict[str, float] = field(default_factory=dict)
    """``joint name -> <limit effort>`` [N.m]. Feeds ``sigma_tau,i``."""

    imu_sites: dict[str, str] = field(default_factory=dict)
    """``log sensor name -> MJCF site name``, e.g. ``pelvis_imu -> pelvis_imu``.

    Derived from the ``*_IMU_JOINT`` fixed joints, lowercased to match the
    ``gyroscope_<name>{X,Y,Z}`` variables in the log.  This is the join key
    between the description and the log, so it is data, not a naming convention
    buried in a helper.
    """

    root_body: str = ""
    """Name of the floating-base link (the one no joint has as a child)."""


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------


def _rpy_to_quat(rpy: Iterable[float]) -> tuple[float, float, float, float]:
    """URDF fixed-axis ``rpy`` -> MJCF ``(w, x, y, z)``.

    ``R = Rz(y) Ry(p) Rx(r)``. Written out rather than delegated to a rotation
    library so the convention is auditable at the point of use -- this is the
    conversion that silently breaks IMU mounts if taken from the wrong library
    default.
    """
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _vec(text: str | None, default: tuple[float, float, float]) -> tuple[float, ...]:
    if text is None:
        return default
    return tuple(float(v) for v in text.split())


def _origin(elem: ET.Element | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``<origin xyz rpy>`` -> ``(xyz, quat_wxyz)``, both defaulted to identity."""
    if elem is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    o = elem.find("origin")
    if o is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    return _vec(o.get("xyz"), (0.0, 0.0, 0.0)), _rpy_to_quat(_vec(o.get("rpy"), (0.0, 0.0, 0.0)))


def _quat_to_mat(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotate_inertia(inertia: ET.Element, quat: Iterable[float]) -> np.ndarray:
    r"""URDF ``<inertia>`` expressed in the inertial-origin frame -> body axes.

    URDF states the tensor in the frame defined by the inertial ``<origin>``'s
    *rotation*; MJCF's ``fullinertia`` is stated in the **body** frame (it refuses
    to accept an orientation alongside, which is what forces this rotation to be
    explicit).  So emit ``I_body = R I_urdf R^T``.

    Doing the rotation here rather than handing MuJoCo a ``quat`` +
    ``diaginertia`` avoids an eigendecomposition whose eigenvector sign/order
    conventions are unspecified -- a congruence is exact and has no branch.
    """
    ixx = float(inertia.get("ixx"))
    iyy = float(inertia.get("iyy"))
    izz = float(inertia.get("izz"))
    ixy = float(inertia.get("ixy", 0.0))
    ixz = float(inertia.get("ixz", 0.0))
    iyz = float(inertia.get("iyz", 0.0))
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    rot = _quat_to_mat(quat)
    return rot @ tensor @ rot.T


def _fmt(values: Iterable[float]) -> str:
    """Full float64 round-trip precision -- a truncated joint origin is a
    millimetre-level FK error, which is the same order as the contact residuals
    the InEKF is trying to resolve."""
    return " ".join(repr(float(v)) for v in values)


# ---------------------------------------------------------------------------
# rotor inertia
# ---------------------------------------------------------------------------


def _armature_for(joint_name: str, table: dict[str, float], default: float) -> float:
    """Case-insensitive **substring** match, **first key in table order wins**.

    This mirrors `JointLevelKFPreFilter.lookupRotorInertia`, which scans
    ``ROTOR_INERTIA_JOINT_KEYS`` in declaration order and returns on the first
    hit (`JointLevelKFRotorAndGramTest.testRotorInertiaTableLookup` locks the
    semantics: substring, case-insensitive, default 0.005).

    Order therefore *matters* and is part of the config, not an implementation
    detail: `filter_cfg.yaml` lists the keys in the same order as the Java array,
    and PyYAML preserves it.  A "more specific key wins" rule would be the
    obvious alternative and would agree on today's Alex table -- no key there is
    a substring of another -- but it would diverge the moment someone adds a
    generic ``ANKLE`` alongside ``ANKLE_X``, and the divergence would show up as
    a silently wrong process noise on one joint.
    """
    upper = joint_name.upper()
    for key, value in table.items():
        if key.upper() in upper:
            return float(value)
    return float(default)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


def urdf_to_mjcf(
    urdf_xml: str,
    *,
    model_name: str = "alex",
    rotor_inertia: dict[str, float] | None = None,
    rotor_inertia_default: float = 0.005,
    imu_link_suffix: str = "_IMU_LINK",
    extra_sites: dict[str, str] | None = None,
) -> AlexModelSpec:
    """Convert URDF source to an MJCF string plus the side-band data.

    Parameters
    ----------
    rotor_inertia, rotor_inertia_default
        The `filter_cfg.yaml` ``joint_kf.rotor_inertia`` table, written into the
        MJCF as ``armature``.  Pass ``{}`` to emit a model with **no** armature --
        that is the second half of the G3 armature-equivalence oracle, which
        needs an armature-free twin of the same robot.
    imu_link_suffix
        Links whose name ends with this get a site named after the link, in
        lowercase, with the suffix stripped and ``_imu`` appended -- the log's
        sensor naming.
    extra_sites
        ``site name -> link name`` for anything else the estimator needs a frame
        on (foot soles for the InEKF contact update).
    """
    rotor_inertia = {} if rotor_inertia is None else dict(rotor_inertia)
    extra_sites = {} if extra_sites is None else dict(extra_sites)

    root = ET.fromstring(urdf_xml)
    if root.tag != "robot":
        raise ValueError(f"expected a URDF <robot> root, got <{root.tag}>")

    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")

    children: dict[str, list[ET.Element]] = {name: [] for name in links}
    child_links: set[str] = set()
    effort_limits: dict[str, float] = {}

    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint '{joint.get('name')}' lacks <parent>/<child>")
        p_name, c_name = parent.get("link"), child.get("link")
        if p_name not in links or c_name not in links:
            raise ValueError(f"joint '{joint.get('name')}' references an unknown link")
        children[p_name].append(joint)
        child_links.add(c_name)

        limit = joint.find("limit")
        if limit is not None and limit.get("effort") is not None:
            effort_limits[joint.get("name")] = float(limit.get("effort"))

    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {sorted(roots)}")
    root_link = roots[0]

    # Sites: IMU frames from the description's own naming, plus caller extras.
    site_of_link: dict[str, list[str]] = {}
    imu_sites: dict[str, str] = {}
    for link_name in links:
        if link_name.endswith(imu_link_suffix):
            sensor = link_name[: -len(imu_link_suffix)].lower() + "_imu"
            site_of_link.setdefault(link_name, []).append(sensor)
            imu_sites[sensor] = sensor
    for site_name, link_name in extra_sites.items():
        if link_name not in links:
            raise ValueError(f"extra site '{site_name}' names unknown link '{link_name}'")
        site_of_link.setdefault(link_name, []).append(site_name)

    out: list[str] = [
        f'<mujoco model="{model_name}">',
        '  <compiler angle="radian" autolimits="false"/>',
        "  <worldbody>",
    ]

    def emit_link(link_name: str, joint: ET.Element | None, depth: int) -> None:
        """Depth-first emit of one body and its subtree.

        Recursive rather than iterative: Alex's tree is 49 links and ~8 deep, so
        the recursion cost is nil and the nesting reads like the MJCF it writes.
        """
        pad = "  " * (depth + 2)
        link = links[link_name]

        if joint is None:
            pos, quat = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        else:
            pos, quat = _origin(joint)

        out.append(f'{pad}<body name="{link_name}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">')

        if joint is None:
            out.append(f'{pad}  <freejoint name="root"/>')
        elif joint.get("type") == "revolute" or joint.get("type") == "continuous":
            axis_elem = joint.find("axis")
            axis = _vec(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))
            arm = _armature_for(joint.get("name"), rotor_inertia, rotor_inertia_default)
            arm_attr = f' armature="{arm!r}"' if arm else ""
            out.append(
                f'{pad}  <joint name="{joint.get("name")}" type="hinge" '
                f'pos="0 0 0" axis="{_fmt(axis)}"{arm_attr}/>'
            )
        elif joint.get("type") != "fixed":
            raise ValueError(
                f"joint '{joint.get('name')}' has unsupported type '{joint.get('type')}'"
            )

        inertial = link.find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) <= 0.0:
            # Every Alex sensor mount (19 of them: the *_IMU_LINKs, the ZED
            # brackets) is declared with mass 0 and a zero inertia tensor -- they
            # are pure coordinate frames, not bodies. MuJoCo rejects a zero
            # tensor ("inertia must have positive eigenvalues"), so they take the
            # massless-placeholder path below. This is exact, not a fudge: their
            # true contribution to M(q) is zero and the placeholder's is ~1e-12.
            inertial = None
        if inertial is not None:
            i_pos, i_quat = _origin(inertial)
            mass = float(inertial.find("mass").get("value"))
            inertia = inertial.find("inertia")
            i_body = _rotate_inertia(inertia, i_quat)
            # MJCF fullinertia order is (ixx, iyy, izz, ixy, ixz, iyz).
            full = [
                i_body[0, 0], i_body[1, 1], i_body[2, 2],
                i_body[0, 1], i_body[0, 2], i_body[1, 2],
            ]
            out.append(
                f'{pad}  <inertial pos="{_fmt(i_pos)}" '
                f'mass="{mass!r}" fullinertia="{_fmt(full)}"/>'
            )
        else:
            out.append(f'{pad}  <inertial pos="0 0 0" mass="{_MASSLESS!r}" diaginertia="0 0 0"/>')

        for site_name in site_of_link.get(link_name, []):
            out.append(f'{pad}  <site name="{site_name}" pos="0 0 0"/>')

        for child_joint in children[link_name]:
            emit_link(child_joint.find("child").get("link"), child_joint, depth + 1)

        out.append(f"{pad}</body>")

    emit_link(root_link, None, 0)
    out += ["  </worldbody>", "</mujoco>"]

    return AlexModelSpec(
        mjcf="\n".join(out),
        effort_limits=effort_limits,
        imu_sites=imu_sites,
        root_body=root_link,
    )


def convert_log_model(log_dir: str | Path, **kwargs) -> AlexModelSpec:
    """Convert the ``model.sdf`` that ships inside an SCS2 log directory."""
    path = Path(log_dir) / "model.sdf"
    if not path.exists():
        raise FileNotFoundError(f"no model.sdf in {log_dir}")
    return urdf_to_mjcf(path.read_text(), **kwargs)
