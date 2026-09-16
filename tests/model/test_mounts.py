"""Sensor mount rotations.

The synthetic cases pin the composition rule against transforms written here by hand. The real-log
case pins it against Alex's own URDF, which declares the pelvis IMU mount as a literal rpy -- so the
number this module produces can be checked against the number a person would read off the robot
description, independently of how it was computed.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from invariant_estimation.model.mounts import body_from_body, describe, imu_to_body

REAL_LOG = Path("/opt/ihmc/LogData/incoming/20260916_101745_Alex001UnifiedControlProcess")


def chain_model(rpy_a=(0, 0, 0), xyz_a=(0, 0, 0), rpy_b=(0, 0, 0), xyz_b=(0, 0, 0)):
    """root -> mid -> tip, each attached by a fixed transform."""
    # angle="radian" explicitly: MuJoCo's `euler` defaults to DEGREES, so a fixture written in
    # radians silently becomes a ~1.57 degree rotation and every expectation below fails by ~90.
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="root">
          <geom type="sphere" size="0.1"/>
          <body name="mid" pos="{' '.join(map(str, xyz_a))}" euler="{' '.join(map(str, rpy_a))}">
            <geom type="sphere" size="0.1"/>
            <body name="tip" pos="{' '.join(map(str, xyz_b))}" euler="{' '.join(map(str, rpy_b))}">
              <geom type="sphere" size="0.1"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>""")


class TestComposition:
    def test_a_single_hop_returns_that_hops_own_transform(self):
        rotation, translation = body_from_body(chain_model(rpy_a=(0, 0, np.pi / 2), xyz_a=(1, 2, 3)),
                                               "mid", "root")
        np.testing.assert_allclose(rotation @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)
        np.testing.assert_allclose(translation, [1, 2, 3], atol=1e-9)

    def test_two_hops_compose_in_the_right_order(self):
        """Two 90-degree yaws make 180, and the second hop's offset arrives rotated by the first.
        Composing the other way round gives a different answer, which is the mistake this pins."""
        model = chain_model(rpy_a=(0, 0, np.pi / 2), xyz_a=(1, 0, 0),
                            rpy_b=(0, 0, np.pi / 2), xyz_b=(1, 0, 0))
        rotation, translation = body_from_body(model, "tip", "root")

        np.testing.assert_allclose(rotation @ np.array([1.0, 0, 0]), [-1, 0, 0], atol=1e-9)
        # the tip's own +x offset is yawed 90 by the first hop, then added to that hop's offset
        np.testing.assert_allclose(translation, [1, 1, 0], atol=1e-9)

    def test_the_rotation_maps_child_vectors_into_the_parent_frame(self):
        """Direction matters: this is v_parent = R @ v_child, which is what two_stage's
        `imu_to_body @ raw_gyro` needs. The transpose is just as orthonormal and just as wrong."""
        rotation, _ = body_from_body(chain_model(rpy_a=(0, 0, np.pi / 2)), "mid", "root")
        assert rotation[1, 0] == pytest.approx(1.0, abs=1e-9)

    def test_asking_across_an_unrelated_branch_fails_loudly(self):
        with pytest.raises(ValueError, match="not an ancestor"):
            body_from_body(chain_model(), "root", "tip")

    def test_an_unknown_body_is_named_in_the_error(self):
        with pytest.raises(KeyError, match="nope"):
            body_from_body(chain_model(), "nope", "root")

    def test_a_body_to_itself_is_the_identity(self):
        rotation, translation = body_from_body(chain_model(rpy_a=(0, 0, 1.0)), "mid", "mid")
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-12)


@pytest.mark.skipif(not REAL_LOG.exists(), reason=f"no hardware log at {REAL_LOG}")
class TestAlexPelvisMount:
    """Against the real robot description, where the answer is independently readable."""

    def _model(self):
        from invariant_estimation.config import load_config
        from invariant_estimation.model.urdf2mjcf import convert_log_model

        jk = load_config()["joint_kf"]
        spec = convert_log_model(REAL_LOG, rotor_inertia=jk["rotor_inertia"],
                                 rotor_inertia_default=jk["rotor_inertia_default"])
        return mujoco.MjModel.from_xml_string(spec.mjcf)

    def test_the_pelvis_imu_is_mounted_a_quarter_turn_off_and_not_identity(self):
        """The headline: passing identity for imu_to_body -- the obvious default -- feeds the filter
        a base gyro rotated 90 degrees about z, and nothing downstream can detect it."""
        rotation = imu_to_body(self._model())
        assert not np.allclose(rotation, np.eye(3), atol=1e-3), "identity is the wrong default here"

        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        assert angle == pytest.approx(90.0, abs=0.05)

    def test_it_matches_the_rpy_the_urdf_declares(self):
        """PELVIS_IMU_JOINT's origin is rpy="0.005114 -0.0001262 1.570796". Rebuilding the rotation
        from those three numbers and comparing is a check against the robot description itself, not
        against this module's own arithmetic."""
        roll, pitch, yaw = 0.005114, -0.0001262, 1.570796
        expected = (np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                    @ np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
                    @ np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]))

        np.testing.assert_allclose(imu_to_body(self._model()), expected, atol=1e-6)

    def test_the_offset_matches_the_urdf_too(self):
        _, translation = body_from_body(self._model(), "PELVIS_IMU_LINK", "PELVIS_LINK")
        np.testing.assert_allclose(translation, [-0.08687724, 0.01225028, -0.08051472], atol=1e-7)

    def test_the_description_names_the_angle_and_the_offset(self):
        text = describe(self._model())
        assert "PELVIS_IMU_LINK" in text and "90." in text
