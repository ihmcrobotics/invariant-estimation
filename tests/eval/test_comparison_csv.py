"""The comparison-CSV schema, as a cross-language contract.

This writer exists so the InEKF arms can all be produced by one implementation. That only works if
its output is interchangeable with Java's `InvariantEstimatorComparisonCsvWriter` -- a column in the
wrong place, a quaternion in the wrong order or a twist in the wrong frame does not raise anything.
It produces a results table that is wrong by exactly the pelvis orientation, which this project has
already caught three times.
"""
import numpy as np
import pytest

from invariant_estimation.eval.comparison_csv import (BASE_TANGENT_AXIS_LABELS, header,
                                                      quaternion_from_rotation, rows, write)


# Copied verbatim from the Java writer's own unit test expectation: the 14 state columns then the
# 45 upper-triangle covariance columns, row-major over i <= j.
EXPECTED_PREFIX = "timestamp_ns,x,y,z,qx,qy,qz,qs,vx,vy,vz,wx,wy,wz"


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sample(ticks=3, n=15):
    rng = np.random.default_rng(0)
    P = np.zeros((ticks, n, n))
    for t in range(ticks):
        A = rng.normal(size=(n, n))
        P[t] = A @ A.T                      # symmetric, as a covariance is
    return dict(timestamps_ns=np.arange(ticks) * 1_000_000 + 1_000_000_000_000,
                rotations=np.stack([rotation_z(0.1 * t) for t in range(ticks)]),
                positions=rng.normal(size=(ticks, 3)),
                velocities_world=rng.normal(size=(ticks, 3)),
                angular_velocities_body=rng.normal(size=(ticks, 3)),
                covariances=P)


class TestHeader:
    def test_the_state_prefix_is_the_fourteen_columns_java_parses_by_index(self):
        assert header().startswith(EXPECTED_PREFIX + ",")

    def test_there_are_exactly_45_covariance_columns_in_upper_triangle_order(self):
        names = header().split(",")
        assert len(names) == 14 + 45
        covariance = names[14:]
        expected = [f"P_{BASE_TANGENT_AXIS_LABELS[i]}_{BASE_TANGENT_AXIS_LABELS[j]}"
                    for i in range(9) for j in range(i, 9)]
        assert covariance == expected

    def test_the_axis_labels_are_the_tangent_order_the_filter_uses(self):
        # [delta_phi; delta_v; delta_p] -- if this ever reads p before v, every NEES block is wrong.
        assert BASE_TANGENT_AXIS_LABELS == ("phi_x", "phi_y", "phi_z",
                                            "v_x", "v_y", "v_z",
                                            "p_x", "p_y", "p_z")


# The Java writer's own constants, read out of its source rather than restated, so this test
# compares against what Java actually emits. Kept as a path check too: if the class moves, the test
# skips loudly rather than passing on a stale literal.
JAVA_SOURCE = ("/home/bpark/workspace-state-estimation/repository-group/alex/src/main/java/"
               "us/ihmc/alex/logAnalysis/InvariantEstimatorComparisonCsvWriter.java")


def java_header():
    import re
    from pathlib import Path
    text = Path(JAVA_SOURCE).read_text()
    required = re.findall(r'"([^"]+)"', re.search(r"REQUIRED_COLUMNS = \{([^}]*)\}", text).group(1))
    axes = re.findall(r'"([^"]+)"', re.search(r"BASE_TANGENT_AXIS_LABELS = \{([^}]*)\}", text).group(1))
    return ",".join(list(required)
                    + [f"P_{axes[i]}_{axes[j]}" for i in range(len(axes)) for j in range(i, len(axes))])


@pytest.mark.skipif(not __import__("pathlib").Path(JAVA_SOURCE).exists(),
                    reason="alex checkout not present")
def test_the_header_is_byte_identical_to_the_java_writers():
    """The whole point of this module is that its files are interchangeable with Java's. Comparing
    against Java's own constants, rather than a literal copied once, means a rename on either side
    fails here instead of surfacing as a misparsed column in a results table."""
    assert header() == java_header()


class TestQuaternion:
    def test_the_scalar_is_LAST(self):
        # Identity -> (0, 0, 0, 1). Scalar-first would put the 1 in slot 0 and every downstream
        # orientation error would be a 180-degree rotation reported as a small one.
        np.testing.assert_allclose(quaternion_from_rotation(np.eye(3)), [0, 0, 0, 1], atol=1e-12)

    def test_a_quarter_turn_about_z_is_the_hand_computed_value(self):
        q = quaternion_from_rotation(rotation_z(np.pi / 2))
        np.testing.assert_allclose(q, [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-12)

    def test_it_stays_accurate_at_a_half_turn_where_the_naive_formula_degrades(self):
        # The pelvis IMU is already mounted at 90 degrees, so large rotations are routine here.
        for axis, expected in ((rotation_z(np.pi), [0, 0, 1, 0]),
                               (np.diag([1.0, -1.0, -1.0]), [1, 0, 0, 0]),
                               (np.diag([-1.0, 1.0, -1.0]), [0, 1, 0, 0])):
            q = quaternion_from_rotation(axis)
            assert np.allclose(q, expected, atol=1e-9) or np.allclose(q, -np.asarray(expected), atol=1e-9)

    def test_round_trips_through_a_rotation_matrix(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            A = rng.normal(size=(3, 3))
            R, _ = np.linalg.qr(A)
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
            qx, qy, qz, qs = quaternion_from_rotation(R)
            # rebuild and compare
            back = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qs), 2 * (qx * qz + qy * qs)],
                [2 * (qx * qy + qz * qs), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qs)],
                [2 * (qx * qz - qy * qs), 2 * (qy * qz + qx * qs), 1 - 2 * (qx * qx + qy * qy)]])
            np.testing.assert_allclose(back, R, atol=1e-9)


class TestRows:
    def test_every_row_has_as_many_fields_as_the_header(self):
        data = sample()
        for line in rows(**data):
            assert len(line.split(",")) == len(header().split(","))

    def test_the_angular_velocity_is_rotated_into_world_and_the_linear_one_is_not(self):
        """The recurring silent failure in this project. The filter carries velocity in world
        already, but the gyro rate it integrates is in body; writing that straight through is wrong
        by exactly the pelvis orientation and raises nothing."""
        R = rotation_z(np.pi / 2)
        omega_body = np.array([[1.0, 0.0, 0.0]])
        v_world = np.array([[3.0, 0.0, 0.0]])
        line = rows(timestamps_ns=[0], rotations=R[None], positions=np.zeros((1, 3)),
                    velocities_world=v_world, angular_velocities_body=omega_body,
                    covariances=np.eye(15)[None])[0].split(",")

        np.testing.assert_allclose([float(x) for x in line[8:11]], [3.0, 0.0, 0.0], atol=1e-12)
        # a +90 deg yaw sends body +x to world +y
        np.testing.assert_allclose([float(x) for x in line[11:14]], [0.0, 1.0, 0.0], atol=1e-12)

    def test_the_covariance_columns_reconstruct_the_base_block(self):
        data = sample(ticks=1)
        line = rows(**data)[0].split(",")[14:]
        P = data["covariances"][0]
        k = 0
        for i in range(9):
            for j in range(i, 9):
                assert float(line[k]) == pytest.approx(P[i, j], rel=1e-12)
                k += 1
        assert k == 45

    def test_only_the_base_block_is_exported_not_the_contact_anchors(self):
        """A 15x15 covariance (2 contacts) must still yield 45 numbers, all from P[0:9, 0:9]."""
        data = sample(ticks=1, n=15)
        # Poison the contact blocks: if any of them leaked into the output this would show.
        data["covariances"][0][9:, :] = 1e9
        data["covariances"][0][:, 9:] = 1e9
        values = [float(v) for v in rows(**data)[0].split(",")[14:]]
        assert len(values) == 45
        assert max(abs(v) for v in values) < 1e9

    def test_the_timestamp_is_written_as_an_integer_not_in_scientific_notation(self):
        """Nanosecond timestamps are ~1e18; a float rendering silently quantises them to ~256 ns
        steps, which is the same bug the mocap loader had."""
        data = sample(ticks=1)
        data["timestamps_ns"] = [1_726_480_000_123_456_789]
        assert rows(**data)[0].split(",")[0] == "1726480000123456789"

    def test_a_mis_shaped_input_is_rejected_rather_than_broadcast(self):
        data = sample(ticks=3)
        data["positions"] = np.zeros((2, 3))
        with pytest.raises(ValueError, match="positions"):
            rows(**data)

    def test_a_covariance_too_small_for_the_base_block_is_rejected(self):
        data = sample(ticks=1, n=6)
        with pytest.raises(ValueError, match="too small"):
            rows(**data)


def test_the_file_round_trips_through_the_scorers_own_parsing_rules(tmp_path):
    """The scorer skips blanks, '#' comments and the header, then parses by index. This checks the
    file it will actually receive, not just the strings we generated."""
    data = sample(ticks=4)
    path = write(tmp_path / "arm6-estimate.csv", **data)

    lines = [l for l in path.read_text().splitlines()
             if l and not l.startswith("#") and not l.startswith("timestamp")]
    assert len(lines) == 4
    for t, line in enumerate(lines):
        columns = line.split(",")
        assert len(columns) == 59
        np.testing.assert_allclose([float(c) for c in columns[1:4]], data["positions"][t], rtol=1e-12)
