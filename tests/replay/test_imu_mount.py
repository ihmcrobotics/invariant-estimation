r"""
Java parity, tier 1: the **IMU mount rotation**, model-free.

The single most error-prone constant in the whole pipeline is the pelvis IMU
mount.  On Alex it is yawed +90 deg (``PELVIS_IMU_JOINT`` rpy yaw = 1.570796), so
IMU axes are *not* pelvis axes: an IMU-frame X bias drives pelvis PITCH, a
Y bias drives pelvis ROLL.  ``urdf2mjcf`` reconstructs this rotation from the
URDF ``rpy`` via a hand-written fixed-axis -> quaternion conversion; a
convention slip there (intrinsic vs extrinsic, or a library default) is a
90-degree frame error that would quietly corrupt every downstream update.

The log pins it exactly.  The Java estimator publishes the applied gyro bias in
**both** frames -- ``invariantAppliedGyroBiasInIMUFrame{X,Y,Z}`` and
``...InPelvisFrame{X,Y,Z}`` -- and the pelvis-frame vector is, by construction,
``R_mount @ imu_frame``.  So the rotation the converter builds from the URDF
must map one logged vector onto the other, tick by tick, to sensor precision.

This needs no mass matrix and no filter state: just the fixed mount rpy and two
logged 3-vectors.
"""
from __future__ import annotations

import numpy as np
import pytest

from invariant_estimation.model.urdf2mjcf import _rpy_to_quat, _quat_to_mat

_WINDOW = (200.0, 260.0)


@pytest.fixture(scope="module")
def mount_window(log_dir):
    from invariant_estimation.replay.logsource import read_window

    names = [f"invariantAppliedGyroBiasInIMUFrame{a}" for a in "XYZ"] + [
        f"invariantAppliedGyroBiasInPelvisFrame{a}" for a in "XYZ"
    ]
    return read_window(log_dir, names, start=_WINDOW[0], end=_WINDOW[1], stride=25)


def _pelvis_imu_rpy(log_dir):
    import xml.etree.ElementTree as ET

    root = ET.fromstring((log_dir / "model.sdf").read_text())
    for joint in root.findall("joint"):
        if joint.get("name") == "PELVIS_IMU_JOINT":
            return [float(v) for v in joint.find("origin").get("rpy").split()]
    raise AssertionError("PELVIS_IMU_JOINT not found in model.sdf")


def test_converter_mount_matches_the_logged_bias_frames(log_dir, mount_window):
    """``R_mount`` from the converter maps IMU-frame bias onto pelvis-frame bias.

    The bias is small (~5e-3 rad/s), so this is a stringent test of the rotation's
    *direction*, not its magnitude: a transposed or mis-ordered R would leave a
    residual of the same order as the signal.
    """
    R = _quat_to_mat(_rpy_to_quat(_pelvis_imu_rpy(log_dir)))

    imu = mount_window.stack([f"invariantAppliedGyroBiasInIMUFrame{a}" for a in "XYZ"])
    pelvis = mount_window.stack([f"invariantAppliedGyroBiasInPelvisFrame{a}" for a in "XYZ"])

    predicted = imu @ R.T
    residual = np.abs(predicted - pelvis).max()
    signal = np.abs(pelvis).max()
    assert residual < 1e-9, (
        f"R_mount does not map the IMU-frame bias to the pelvis-frame bias: "
        f"residual {residual:.2e} vs signal {signal:.2e} -- the rpy->quat "
        f"convention in urdf2mjcf is wrong (intrinsic/extrinsic mix-up?)"
    )


def test_pelvis_imu_is_actually_yawed_ninety_degrees(log_dir):
    """Guard the premise: if a future model un-yaws the IMU, the trap is gone and
    the frame-swap gotcha in the log skill should be revisited. Pinning it here
    means that change announces itself instead of silently altering axis mapping."""
    rpy = _pelvis_imu_rpy(log_dir)
    assert rpy[2] == pytest.approx(np.pi / 2, abs=1e-3), (
        f"PELVIS_IMU_JOINT yaw is {rpy[2]:.4f}, expected ~pi/2; the IMU-vs-pelvis "
        f"axis mapping the estimator assumes has changed"
    )
