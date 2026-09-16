r"""Sensor mount rotations, derived from the robot model rather than assumed.

`two_stage.make_step` rotates the base gyro into the body frame with
``imu_to_body``, and nothing downstream can tell a wrong one from a right one:
the filter simply estimates a differently-oriented frame and every number it
produces stays finite and plausible. On Alex the true value is **not** identity
-- the pelvis IMU is mounted yawed 90 degrees:

    <joint name="PELVIS_IMU_JOINT" type="fixed">
        <origin xyz="-0.08687724 0.01225028 -0.08051472"
                rpy="0.005114 -0.0001262 1.570796"/>
        <parent link="PELVIS_LINK"/>
        <child  link="PELVIS_IMU_LINK"/>

so passing identity feeds the filter a gyro rotated a quarter turn about z.

These helpers read that transform out of whatever model the session was built
from -- in practice the ``model.sdf`` that ships inside the log being replayed,
which is the description the robot actually ran with.

Nominal, not measured
---------------------
This is the CAD mount. A real mount deviates from it, and this cannot tell the
difference. It is the right *prior*, and it is enormously better than identity,
but a study that needs the true rotation has to measure or estimate it.
"""
import mujoco
import numpy as np


def _rotation(quaternion):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def body_from_body(mj_model, child_body: str, parent_body: str):
    """The transform of ``child_body``'s frame in ``parent_body``'s frame.

    Returns ``(rotation, translation)`` with ``rotation`` mapping a vector in the
    child frame to the parent frame, i.e. ``v_parent = rotation @ v_child``.

    Walks the body tree upward and composes, so a sensor reached through one or
    more fixed joints -- which is how every IMU on Alex is attached -- resolves
    correctly. A site's own ``site_quat`` is NOT this transform: the converter
    places each IMU site at the origin of its own link, so that value is
    identity and says nothing about the mount.
    """
    child = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, child_body)
    parent = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_body)
    if child < 0:
        raise KeyError(f"model has no body {child_body!r}")
    if parent < 0:
        raise KeyError(f"model has no body {parent_body!r}")

    rotation = np.eye(3)
    translation = np.zeros(3)
    body = child
    while body != parent:
        step = _rotation(mj_model.body_quat[body])
        translation = step @ translation + mj_model.body_pos[body]
        rotation = step @ rotation
        nxt = int(mj_model.body_parentid[body])
        if nxt == body:  # reached the world root without meeting `parent`
            raise ValueError(f"{parent_body!r} is not an ancestor of {child_body!r}")
        body = nxt
    return rotation, translation


def imu_to_body(mj_model, imu_link: str = "PELVIS_IMU_LINK", body_link: str = "PELVIS_LINK"):
    """``imu_to_body`` for `two_stage`: maps the IMU's measured vectors into the body frame.

    Defaults are Alex's pelvis IMU and pelvis link. Returns a plain 3x3; the caller passes it
    straight to `prepare_session`.
    """
    rotation, _ = body_from_body(mj_model, imu_link, body_link)
    return rotation


def describe(mj_model, imu_link: str = "PELVIS_IMU_LINK", body_link: str = "PELVIS_LINK") -> str:
    """One line naming the mount's rotation angle and offset, for a run's log header."""
    rotation, translation = body_from_body(mj_model, imu_link, body_link)
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    return (f"{imu_link} -> {body_link}: {angle:.3f} deg, offset "
            f"[{translation[0]:+.4f} {translation[1]:+.4f} {translation[2]:+.4f}] m")
