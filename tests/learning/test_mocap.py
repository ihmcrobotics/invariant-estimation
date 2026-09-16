"""Mocap CSV -> truth arrays.

The schemas are fixed on the Java side (`ReplayRunner`), so these fixtures are written by hand in
exactly that format rather than by a helper that could drift along with the reader.
"""
import numpy as np
import pytest

from invariant_estimation.learning.mocap import (
    MocapTruth, load_truth, nearest_index, quaternion_to_rotation, read_pose_csv,
    read_velocity_csv, select_valid,
)

def _cell(value):
    """Ints exactly (timestamps are nanoseconds), floats at full precision. repr() is wrong here:
    under numpy 2 a numpy scalar reprs as "np.float64(0.7)", which is not a number in a CSV."""
    return str(int(value)) if isinstance(value, (int, np.integer)) else format(float(value), ".17g")


POSE_HEADER = "timestamp_ns,x,y,z,qx,qy,qz,qs"
VELOCITY_HEADER = "timestamp_ns,vx,vy,vz,wx,wy,wz"


def write_pose(path, rows):
    path.write_text(POSE_HEADER + "\n" + "\n".join(",".join(_cell(v) for v in r) for r in rows) + "\n")
    return path


def write_velocity(path, rows):
    path.write_text(VELOCITY_HEADER + "\n" + "\n".join(",".join(_cell(v) for v in r) for r in rows) + "\n")
    return path


def identity_pose(t):
    return [t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class TestQuaternionConvention:
    """The one convention error that no shape or finiteness check can catch."""

    def test_ninety_degrees_about_z_maps_body_x_to_world_y(self):
        half = np.sqrt(0.5)
        rotation = quaternion_to_rotation(np.array([[0.0, 0.0, half, half]]))[0]
        np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)

    def test_ninety_degrees_about_x_maps_body_y_to_world_z(self):
        half = np.sqrt(0.5)
        rotation = quaternion_to_rotation(np.array([[half, 0.0, 0.0, half]]))[0]
        np.testing.assert_allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)

    def test_the_scalar_is_read_last_not_first(self):
        """Same four numbers read as wxyz instead of xyzs give a different rotation; if this ever
        passes with both orders the test has stopped constraining anything."""
        components = np.array([[0.1, 0.2, 0.3, np.sqrt(1 - 0.14)]])
        as_xyzs = quaternion_to_rotation(components)[0]
        as_wxyz = quaternion_to_rotation(components[:, [1, 2, 3, 0]])[0]
        assert not np.allclose(as_xyzs, as_wxyz)

    def test_a_general_rotation_is_orthonormal_and_right_handed(self):
        q = np.array([0.2, -0.4, 0.5, 0.0])
        q[3] = np.sqrt(1.0 - q[:3] @ q[:3])
        rotation = quaternion_to_rotation(q[None])[0]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)

    def test_it_is_not_accidentally_the_transpose(self):
        """A transposed rotation is still orthonormal with det 1, so orthonormality cannot catch it;
        only a known vector mapping can, and body->world is the direction the filter's R uses."""
        half = np.sqrt(0.5)
        rotation = quaternion_to_rotation(np.array([[0.0, 0.0, half, half]]))[0]
        assert rotation[1, 0] == pytest.approx(1.0, abs=1e-12), "R maps body x to world y; this is R^T"


class TestReading:
    def test_both_schemas_round_trip(self, tmp_path):
        pose = write_pose(tmp_path / "pelvis.csv", [[100, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
        timestamps, position, quaternion = read_pose_csv(pose)
        assert timestamps.tolist() == [100]
        np.testing.assert_allclose(position[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(quaternion[0], [0.0, 0.0, 0.0, 1.0])

        velocity = write_velocity(tmp_path / "v.csv", [[100, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        timestamps, linear, angular = read_velocity_csv(velocity)
        np.testing.assert_allclose(linear[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(angular[0], [0.4, 0.5, 0.6])

    def test_a_wrong_header_is_rejected_rather_than_read_positionally(self, tmp_path):
        """Reading by position would silently swap columns if the exporter's schema ever changed."""
        path = tmp_path / "pelvis.csv"
        path.write_text("timestamp_ns,x,y,z,qs,qx,qy,qz\n100,0,0,0,1,0,0,0\n")
        with pytest.raises(ValueError, match="expected header"):
            read_pose_csv(path)

    def test_nanosecond_timestamps_survive_a_realistic_magnitude(self, tmp_path):
        """A ns count near 1e18 is not representable in float64; reading it as a float quantises to
        ~100 ns, which would corrupt every alignment offset at the edge of tolerance."""
        base = 1_789_000_000_000_000_001
        pose = write_pose(tmp_path / "pelvis.csv", [identity_pose(base), identity_pose(base + 1)])
        timestamps, _, _ = read_pose_csv(pose)
        assert timestamps.tolist() == [base, base + 1]

    def test_unsorted_timestamps_are_rejected(self, tmp_path):
        pose = write_pose(tmp_path / "pelvis.csv", [identity_pose(200), identity_pose(100)])
        with pytest.raises(ValueError, match="not sorted"):
            read_pose_csv(pose)


class TestAlignment:
    def test_the_nearest_sample_is_chosen_on_both_sides_of_a_tie_boundary(self):
        source = np.array([0, 100, 200], dtype=np.int64)
        index, offset = nearest_index(source, np.array([-10, 40, 60, 210], dtype=np.int64))
        assert index.tolist() == [0, 0, 1, 2]
        assert offset.tolist() == [-10, 40, -40, 10]

    def test_targets_beyond_the_source_range_clamp_rather_than_wrap(self):
        source = np.array([1000, 1100], dtype=np.int64)
        index, offset = nearest_index(source, np.array([0, 5000], dtype=np.int64))
        assert index.tolist() == [0, 1]
        assert offset.tolist() == [-1000, 3900]


class TestLoadTruth:
    def _session(self, tmp_path, pose_rows, velocity_rows):
        return (write_pose(tmp_path / "pelvis.csv", pose_rows),
                write_velocity(tmp_path / "pelvisVelocity.csv", velocity_rows))

    def test_truth_lands_on_the_target_clock_with_the_right_values(self, tmp_path):
        half = np.sqrt(0.5)
        pose, velocity = self._session(
            tmp_path,
            [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1000, 1.0, 2.0, 3.0, 0.0, 0.0, half, half]],
            [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1000, 0.5, -0.5, 0.25, 0.1, 0.2, 0.3]])

        truth = load_truth(pose, velocity, np.array([0, 1000]), max_offset_ns=100)

        assert truth.coverage() == 1.0
        np.testing.assert_allclose(np.asarray(truth.rotation[0]), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.asarray(truth.rotation[1]) @ np.array([1.0, 0.0, 0.0]),
                                   [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.asarray(truth.velocity[1]), [0.5, -0.5, 0.25], atol=1e-12)
        np.testing.assert_allclose(np.asarray(truth.angular_velocity[1]), [0.1, 0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(np.asarray(truth.position[1]), [1.0, 2.0, 3.0], atol=1e-12)

    def test_the_two_files_are_aligned_independently(self, tmp_path):
        """They are written from separate passes and their timestamps need not coincide; assuming a
        shared clock would pair a pose with a velocity from a different instant."""
        pose, velocity = self._session(
            tmp_path,
            [identity_pose(0), identity_pose(1000)],
            [[500, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1500, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        truth = load_truth(pose, velocity, np.array([1000]), max_offset_ns=600)

        np.testing.assert_allclose(np.asarray(truth.rotation[0]), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.asarray(truth.velocity[0]), [1.0, 0.0, 0.0], atol=1e-12)
        assert truth.worst_offset_ns() == 500

    def test_a_tick_with_no_nearby_sample_is_invalid_not_filled_with_a_stale_neighbour(self, tmp_path):
        pose, velocity = self._session(tmp_path, [identity_pose(0)],
                                       [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        truth = load_truth(pose, velocity, np.array([0, 10_000_000]), max_offset_ns=1000)

        assert np.asarray(truth.valid).tolist() == [True, False]
        assert truth.coverage() == 0.5

    def test_a_dropout_is_marked_invalid_and_cannot_poison_the_arrays(self, tmp_path):
        """One NaN anywhere makes body_velocity_l2 NaN for every sample, so an excluded row must not
        merely be masked — it must not contain a NaN at all."""
        pose, velocity = self._session(
            tmp_path,
            [identity_pose(0), [1000, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        truth = load_truth(pose, velocity, np.array([0, 1000]), max_offset_ns=100)

        assert np.asarray(truth.valid).tolist() == [True, False]
        assert np.isfinite(np.asarray(truth.rotation)).all(), "an invalid row still carries a NaN"
        assert np.isfinite(np.asarray(truth.position)).all()

    def test_a_non_unit_quaternion_is_rejected_rather_than_renormalised(self, tmp_path):
        """Renormalising would turn a corrupt reading into a plausible-looking rotation."""
        pose, velocity = self._session(
            tmp_path, [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]],
            [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        truth = load_truth(pose, velocity, np.array([0]), max_offset_ns=100)
        assert np.asarray(truth.valid).tolist() == [False]

    def test_it_rejects_a_nonpositive_tolerance(self, tmp_path):
        pose, velocity = self._session(tmp_path, [identity_pose(0)],
                                       [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="max_offset_ns"):
            load_truth(pose, velocity, np.array([0]), max_offset_ns=0)


class TestSelectValid:
    def _truth(self, valid):
        n = len(valid)
        return MocapTruth(np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.zeros((n, 3)),
                          np.zeros((n, 3)), np.array(valid), np.zeros(n, dtype=np.int64))

    def test_truth_and_estimate_are_filtered_together(self):
        """Masking one side but not the other pairs tick i with tick j and reads as estimator error."""
        estimate = np.arange(4.0)[:, None] * np.ones((1, 3))
        _, _, filtered = select_valid(self._truth([True, False, True, False]), estimate)
        np.testing.assert_allclose(np.asarray(filtered)[:, 0], [0.0, 2.0])

    def test_a_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="samples"):
            select_valid(self._truth([True, True]), np.zeros((3, 3)))

    def test_no_valid_samples_raises_rather_than_returning_empty_arrays(self):
        """An empty loss is a silent zero; this has to be loud."""
        with pytest.raises(ValueError, match="no valid mocap samples"):
            select_valid(self._truth([False, False]), np.zeros((2, 3)))
