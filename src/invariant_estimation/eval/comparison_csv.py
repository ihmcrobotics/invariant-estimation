"""Export a rollout in the schema `ComparisonArmScorer` reads.

The paper's InEKF arms differ only in where their measurement noise comes from, so they must come
from ONE implementation: if arm 4 were produced by the Java replay and arm 6 by this pipeline, a
difference between them would be unattributable between the intended variable and the two
codebases. This writer lets arms 3-6 all be produced here. Arms 1-2 are the DRC estimator and have
no counterpart here, so those stay on the Java side -- but they are a different estimator anyway,
so that comparison spans implementations by necessity rather than by accident.

The schema is Java's `InvariantEstimatorComparisonCsvWriter`, reproduced exactly:

    timestamp_ns, x, y, z, qx, qy, qz, qs, vx, vy, vz, wx, wy, wz,
    P_phi_x_phi_x, P_phi_x_phi_y, ... P_p_z_p_z          (45 columns, upper triangle incl. diagonal)

Three conventions that are easy to get wrong and silent when wrong. Each produces plausible numbers:

* The quaternion is **scalar LAST** (qx, qy, qz, qs).
* `vx..vz` and `wx..wz` are in the **WORLD** frame. The filter carries velocity in world already,
  but the gyro rate it integrates is in body, so the angular part is rotated by R. Writing the body
  rate straight through is wrong by exactly the pelvis orientation and has been caught three times
  in this project.
* `timestamp_ns` must share an epoch with the mocap capture this will be scored against, or the
  matcher pairs samples that were never simultaneous and the result reads as estimator error.

The covariance block is `P[0:9, 0:9]` in the tangent order `[delta_phi; delta_v; delta_p]` --
the base state only, which is what a pelvis NEES check needs.
"""
from __future__ import annotations

import numpy as np

REQUIRED_COLUMNS = ("timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qs",
                    "vx", "vy", "vz", "wx", "wy", "wz")

BASE_TANGENT_AXIS_LABELS = ("phi_x", "phi_y", "phi_z",
                            "v_x", "v_y", "v_z",
                            "p_x", "p_y", "p_z")
BASE_TANGENT_SIZE = len(BASE_TANGENT_AXIS_LABELS)


def header() -> str:
    """The 14 state columns then the 45 covariance columns, exactly as Java emits them."""
    names = list(REQUIRED_COLUMNS)
    for i in range(BASE_TANGENT_SIZE):
        for j in range(i, BASE_TANGENT_SIZE):
            names.append(f"P_{BASE_TANGENT_AXIS_LABELS[i]}_{BASE_TANGENT_AXIS_LABELS[j]}")
    return ",".join(names)


def quaternion_from_rotation(R):
    """Rotation matrix -> (qx, qy, qz, qs), scalar LAST.

    Shepperd's method: pick the branch whose divisor is largest, so no branch divides by something
    near zero. The naive single-branch formula loses precision near a 180-degree rotation, which on
    a pelvis is not an exotic case -- the mount is already yawed 90 degrees.
    """
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qs, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qs, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qs, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qs, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s

    return np.array([qx, qy, qz, qs], dtype=float)


def rows(timestamps_ns, rotations, positions, velocities_world, angular_velocities_body, covariances):
    """One CSV row per tick, as a list of strings.

    Parameters
    ----------
    timestamps_ns : (T,) int
        Must share an epoch with the mocap capture. Not checked here -- nothing at this layer can.
    rotations : (T, 3, 3)
    positions, velocities_world : (T, 3)
        Both already in world; the filter carries them that way.
    angular_velocities_body : (T, 3)
        The bias-corrected gyro rate the filter integrates, in BODY. Rotated to world here, which is
        the one transformation this writer performs and the one the Java writer performs too.
    covariances : (T, n, n)
        The full tangent covariance; only the leading 9x9 base block is exported.
    """
    timestamps_ns = np.asarray(timestamps_ns)
    rotations = np.asarray(rotations, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities_world = np.asarray(velocities_world, dtype=float)
    angular_velocities_body = np.asarray(angular_velocities_body, dtype=float)
    covariances = np.asarray(covariances, dtype=float)

    ticks = len(timestamps_ns)
    for name, array, shape in (("rotations", rotations, (ticks, 3, 3)),
                               ("positions", positions, (ticks, 3)),
                               ("velocities_world", velocities_world, (ticks, 3)),
                               ("angular_velocities_body", angular_velocities_body, (ticks, 3))):
        if array.shape != shape:
            raise ValueError(f"{name} must be {shape}, got {array.shape}")
    if covariances.shape[0] != ticks or covariances.ndim != 3:
        raise ValueError(f"covariances must be (T, n, n) with T={ticks}, got {covariances.shape}")
    if covariances.shape[1] < BASE_TANGENT_SIZE:
        raise ValueError(f"covariance is {covariances.shape[1]}x{covariances.shape[2]}, "
                         f"too small for the {BASE_TANGENT_SIZE}x{BASE_TANGENT_SIZE} base block")

    upper = [(i, j) for i in range(BASE_TANGENT_SIZE) for j in range(i, BASE_TANGENT_SIZE)]

    out = []
    for t in range(ticks):
        quaternion = quaternion_from_rotation(rotations[t])
        omega_world = rotations[t] @ angular_velocities_body[t]
        P = covariances[t]
        values = [str(int(timestamps_ns[t]))]
        values += [repr(float(v)) for v in positions[t]]
        values += [repr(float(v)) for v in quaternion]
        values += [repr(float(v)) for v in velocities_world[t]]
        values += [repr(float(v)) for v in omega_world]
        values += [repr(float(P[i, j])) for i, j in upper]
        out.append(",".join(values))
    return out


def write(path, timestamps_ns, rotations, positions, velocities_world,
          angular_velocities_body, covariances):
    """Writes the header and one row per tick to `path`."""
    lines = [header()]
    lines += rows(timestamps_ns, rotations, positions, velocities_world,
                  angular_velocities_body, covariances)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return path
