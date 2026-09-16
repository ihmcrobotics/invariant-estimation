r"""Mocap ground truth, from the exported CSVs onto the robot's clock.

Turns `ReplayRunner`'s two exports into the `truth_rotation` / `truth_velocity`
arrays `optimize.body_velocity_l2` consumes, sampled at whatever timestamps the
robot log's ticks carry.

The two files and their conventions
-----------------------------------
Both schemas are fixed in `ReplayRunner` and are read here exactly as written:

* ``pelvis.csv`` -- ``timestamp_ns,x,y,z,qx,qy,qz,qs``. The quaternion is
  **scalar-LAST** (Euclid's ``getX/getY/getZ/getS`` order). Reading it as
  ``wxyz`` produces a rotation that is wrong in a way no shape check catches.
* ``pelvisVelocity.csv`` -- ``timestamp_ns,vx,vy,vz,wx,wy,wz``. Both triples are
  in the **world** frame: `PelvisTwistEstimator` differentiates the pose origin
  in the single fixed frame ``Wg`` for the linear part, and takes
  ``vee(skew(Ṙ Rᵀ))`` -- the spatial, not body, form -- for the angular part.
  That is what makes `body_velocity_l2`'s ``Rᵀv`` the right thing to write.

The rotation returned here is world-from-body, matching the filter's own ``R``,
so the two sides of the loss are the same kind of object.

Alignment
---------
The two files carry **independent** timestamps (they are written from separate
passes), so each is aligned to the target clock on its own. Alignment is
nearest-neighbour with a rejection threshold, not interpolation, so this agrees
sample-for-sample with `EstimatorComparisonRunner`'s Java-side matching; a
Python evaluation that silently interpolated would disagree with the Java
numbers for reasons no one would think to look for. The price is up to half a
mocap period of offset, which `MocapTruth.offset_ns` reports rather than hides.

Validity
--------
Mocap drops out, and `body_velocity_l2` has no NaN handling -- one bad sample
makes the whole loss NaN. Every sample is therefore marked valid or not (a
non-finite reading, a non-unit quaternion, or no source sample within
``max_offset_ns``), and `select_valid` is the intended way to drop them before
building a loss.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

POSE_COLUMNS = ("timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qs")
VELOCITY_COLUMNS = ("timestamp_ns", "vx", "vy", "vz", "wx", "wy", "wz")

#: A quaternion further than this from unit length is treated as a bad sample rather than
#: renormalised. Renormalising hides a real problem: at this magnitude the source is not a rotation.
QUATERNION_UNIT_TOLERANCE = 1.0e-3


class MocapTruth(NamedTuple):
    """Ground truth resampled onto the target clock, with the honesty fields attached.

    Attributes
    ----------
    rotation : (T, 3, 3)
        World-from-body, same convention as the filter's ``R``.
    velocity : (T, 3)
        World-frame linear velocity.
    angular_velocity : (T, 3)
        World-frame (spatial) angular velocity.
    position : (T, 3)
        World-frame pelvis origin.
    valid : (T,) bool
        False where a sample is unusable -- see the module docstring.
    offset_ns : (T,) int
        Signed alignment error actually incurred per sample (target minus source),
        worst over the two files. Zero where invalid.
    """

    rotation: jnp.ndarray
    velocity: jnp.ndarray
    angular_velocity: jnp.ndarray
    position: jnp.ndarray
    valid: jnp.ndarray
    offset_ns: jnp.ndarray

    def coverage(self) -> float:
        """Fraction of target ticks with usable truth. A low number invalidates a training run."""
        return float(np.mean(np.asarray(self.valid)))

    def worst_offset_ns(self) -> int:
        """Largest alignment error among the valid samples; 0 when nothing is valid."""
        valid = np.asarray(self.valid)
        if not valid.any():
            return 0
        return int(np.max(np.abs(np.asarray(self.offset_ns)[valid])))


def _read_csv(path, columns):
    """Read one export, checking the header rather than trusting column order."""
    with open(path, "r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    found = tuple(name.strip() for name in header.split(","))
    if found != tuple(columns):
        raise ValueError(f"{path}: expected header {','.join(columns)}, found {header}")
    # The timestamp column is parsed SEPARATELY, straight to int64. Reading the whole table as
    # float64 and casting afterwards loses the low digits of a nanosecond count near 1e18 -- float64
    # has 53 bits of mantissa, so ~1e18 quantises to steps of 256 ns and two adjacent 1 kHz ticks can
    # land on the same value. The cast has already happened by then; nothing downstream can tell.
    timestamps = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,), ndmin=1, dtype=np.int64)
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64,
                       usecols=tuple(range(1, len(columns))))
    if timestamps.size == 0:
        raise ValueError(f"{path}: no rows")
    if table.shape[1] != len(columns) - 1:
        raise ValueError(f"{path}: expected {len(columns)} columns, found {table.shape[1] + 1}")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{path}: timestamps are not sorted")
    return timestamps, table


def read_pose_csv(path):
    """``pelvis.csv`` -> (timestamps, position (N,3), quaternion (N,4) as x,y,z,s)."""
    timestamps, rest = _read_csv(path, POSE_COLUMNS)
    return timestamps, rest[:, 0:3], rest[:, 3:7]


def read_velocity_csv(path):
    """``pelvisVelocity.csv`` -> (timestamps, linear (N,3), angular (N,3)), both world-frame."""
    timestamps, rest = _read_csv(path, VELOCITY_COLUMNS)
    return timestamps, rest[:, 0:3], rest[:, 3:6]


def quaternion_to_rotation(quaternion):
    """Scalar-LAST unit quaternions ``(x, y, z, s)`` -> world-from-body rotation matrices.

    No renormalisation: a quaternion that is not already unit length is a bad sample, and
    `load_truth` marks it invalid rather than quietly rescaling a corrupt reading into a
    plausible-looking rotation.
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"quaternion must have shape (N, 4), got {quaternion.shape}")
    x, y, z, s = (quaternion[:, i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - s * z), 2 * (x * z + s * y)], axis=-1),
        np.stack([2 * (x * y + s * z), 1 - 2 * (x * x + z * z), 2 * (y * z - s * x)], axis=-1),
        np.stack([2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def nearest_index(source_ns, target_ns):
    """For each target timestamp, the index of the closest source timestamp, and the signed offset.

    ``source_ns`` must be sorted (checked on read). Binary search rather than a full pairwise
    distance matrix: a 10-minute 1 kHz log against 250 Hz mocap is ~600k x ~150k, which is not a
    matrix anyone should allocate to answer this.
    """
    source_ns = np.asarray(source_ns, dtype=np.int64)
    target_ns = np.asarray(target_ns, dtype=np.int64)
    right = np.searchsorted(source_ns, target_ns)
    left = np.clip(right - 1, 0, source_ns.size - 1)
    right = np.clip(right, 0, source_ns.size - 1)
    take_left = np.abs(target_ns - source_ns[left]) <= np.abs(source_ns[right] - target_ns)
    index = np.where(take_left, left, right)
    return index, target_ns - source_ns[index]


def load_truth(pose_csv, velocity_csv, target_timestamps_ns, *, max_offset_ns) -> MocapTruth:
    """Read both exports and resample them onto ``target_timestamps_ns``.

    Parameters
    ----------
    target_timestamps_ns : (T,) integer nanoseconds
        The robot ticks the training inputs were built from, in the synchronized clock domain the
        capture session declares. Aligning against a different clock is the failure this cannot
        detect and the session manifest's ``clock_domain`` exists to prevent.
    max_offset_ns : int
        A target tick whose nearest mocap sample is further away than this is marked invalid rather
        than filled with the stale neighbour.
    """
    if max_offset_ns <= 0:
        raise ValueError(f"max_offset_ns must be positive, got {max_offset_ns}")
    target = np.asarray(target_timestamps_ns, dtype=np.int64)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("target_timestamps_ns must be a nonempty 1-D array")

    pose_ns, position, quaternion = read_pose_csv(pose_csv)
    velocity_ns, linear, angular = read_velocity_csv(velocity_csv)

    pose_index, pose_offset = nearest_index(pose_ns, target)
    velocity_index, velocity_offset = nearest_index(velocity_ns, target)

    position = position[pose_index]
    quaternion = quaternion[pose_index]
    linear = linear[velocity_index]
    angular = angular[velocity_index]

    norm = np.linalg.norm(quaternion, axis=-1)
    valid = (
        (np.abs(pose_offset) <= max_offset_ns)
        & (np.abs(velocity_offset) <= max_offset_ns)
        & np.isfinite(position).all(axis=-1)
        & np.isfinite(quaternion).all(axis=-1)
        & np.isfinite(linear).all(axis=-1)
        & np.isfinite(angular).all(axis=-1)
        & (np.abs(norm - 1.0) <= QUATERNION_UNIT_TOLERANCE)
    )

    # Neutralise invalid rows BEFORE building rotations: a NaN quaternion would otherwise produce a
    # NaN rotation, and a single NaN anywhere in the array poisons every gradient computed from it,
    # even for samples the mask would have excluded.
    safe = np.where(valid[:, None], quaternion, np.array([0.0, 0.0, 0.0, 1.0]))
    rotation = quaternion_to_rotation(safe)
    position = np.where(valid[:, None], position, 0.0)
    linear = np.where(valid[:, None], linear, 0.0)
    angular = np.where(valid[:, None], angular, 0.0)
    offset = np.where(valid, np.maximum(np.abs(pose_offset), np.abs(velocity_offset)), 0)

    return MocapTruth(jnp.asarray(rotation), jnp.asarray(linear), jnp.asarray(angular),
                      jnp.asarray(position), jnp.asarray(valid), jnp.asarray(offset))


def select_valid(truth: MocapTruth, *estimates):
    """Drop invalid samples from the truth and from any same-length estimate arrays, together.

    The alignment is the point: masking the truth but not the estimate silently pairs tick *i* of
    one with tick *j* of the other, which looks like an estimator error rather than a bookkeeping
    one. Returns ``(truth_rotation, truth_velocity, *estimates)``, all filtered identically.
    """
    valid = np.asarray(truth.valid)
    if not valid.any():
        raise ValueError("no valid mocap samples; check the clock domain and max_offset_ns")
    index = jnp.asarray(np.flatnonzero(valid))
    filtered = []
    for array in estimates:
        array = jnp.asarray(array)
        if array.shape[0] != valid.size:
            raise ValueError(f"estimate has {array.shape[0]} samples, truth has {valid.size}")
        filtered.append(array[index])
    return (truth.rotation[index], truth.velocity[index], *filtered)
