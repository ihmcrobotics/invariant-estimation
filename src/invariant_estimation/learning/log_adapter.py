"""Validated log boundary, independent of the optional IHMC binary decoder.

See LOG_INPUT_CONTRACT.md. No channel guessing, mocap-based initialization, or
time-axis compression happens here. Model quantities are evaluated in the scan.
"""

from dataclasses import dataclass
from typing import Mapping, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..inEKF import ekf as base_ekf, filter as base_filter
from ..jointKF import filter as joint_filter
from .mocap import MocapTruth, load_truth, select_valid
from .optimize import body_velocity_l2
from .realdata import (
    accel_body_from_raw,
    contact_chol_heuristic,
    estimate_static_accel_bias,
)
from .two_stage import TwoStageCarry, TwoStageInputs, make_step


def _array(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label}: expected finite shape {shape}, got {value.shape}")
    return value


def _rotation(value, label):
    value = _array(value, (3, 3), label)
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-9, rtol=0) or not np.isclose(
        np.linalg.det(value), 1
    ):
        raise ValueError(f"{label}: expected a proper rotation")
    return value


@dataclass(frozen=True)
class ChannelMap:
    """Exact processed-sensor channel names; vectors are ordered x,y,z.

    Positions/velocities: rad, rad/s. Gyros: rad/s in each IMU measurement
    frame. Accel: specific force in m/s², in the base IMU frame (gravity included).
    Trust is a binary anchor decision; probability is the separate [0,1] signal.
    """

    positions: Mapping[str, str]
    velocities: Mapping[str, str]
    gyros: Mapping[str, tuple[str, str, str]]
    accel: tuple[str, str, str]
    anchor_trust: Mapping[str, str]
    contact_probability: Mapping[str, str]
    sensor_processing: str
    schema_version: int = 1

    def required_channels(self):
        if self.schema_version != 1 or not self.sensor_processing.strip():
            raise ValueError(
                "unsupported schema or missing sensor_processing provenance"
            )
        vectors = [self.accel, *self.gyros.values()]
        if any(len(v) != 3 for v in vectors):
            raise ValueError("IMU vectors must have three channels in x,y,z order")
        names = [
            *self.positions.values(),
            *self.velocities.values(),
            *(n for v in vectors for n in v),
            *self.anchor_trust.values(),
            *self.contact_probability.values(),
        ]
        if any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("channel names must be nonempty strings")
        # Aliases are rejected within each physical group; trust/probability may
        # deliberately share a binary channel but never silently threshold an EMA.
        for group in (
            list(self.positions.values()),
            list(self.velocities.values()),
            [n for v in self.gyros.values() for n in v],
            list(self.accel),
            list(self.anchor_trust.values()),
            list(self.contact_probability.values()),
        ):
            if len(group) != len(set(group)):
                raise ValueError("duplicate physical channel mapping")
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class ClockMapping:
    """Explicit origin mapping, NOT an offset/drift estimator.

    source_origin_s belongs to LogWindow.time. target_origin_ns belongs to
    clock_domain. Integer addition preserves large absolute ns timestamps.
    """

    source_origin_s: float
    target_origin_ns: int
    clock_domain: str

    def timestamps(self, time):
        if (
            not np.isfinite(self.source_origin_s)
            or type(self.target_origin_ns) is not int
            or not self.clock_domain.strip()
        ):
            raise ValueError("invalid clock mapping")
        delta = np.rint((np.asarray(time) - self.source_origin_s) * 1e9)
        # Python integers avoid overflow before the range check.
        result = [self.target_origin_ns + int(d) for d in delta]
        bounds = np.iinfo(np.int64)
        if any(t < bounds.min or t > bounds.max for t in result):
            raise ValueError("timestamp outside int64 range")
        return np.asarray(result, dtype=np.int64)


class SessionModel(Protocol):
    joint_names: tuple[str, ...]  # all model hinges, including unfiltered ankles
    unfiltered_names: tuple[str, ...]  # anchor-unfiltered order from build
    contact_names: tuple[str, ...]  # one-to-one anchor/contact slots in v1

    def model_inputs(self, filtered_q, all_q) -> joint_filter.ModelInputs: ...
    def contact_frames(
        self, filtered_q, filtered_qd, all_q, all_qd
    ) -> base_filter.ContactFrames: ...


@dataclass(frozen=True)
class InitialState:
    """State at t_first-dt, in the declared world frame; no implicit truth seed."""

    rotation: object
    velocity: object
    position: object
    covariance: object
    source: str


@dataclass(frozen=True)
class PreparedSession:
    timestamps_ns: np.ndarray
    sensors: joint_filter.SensorInputs
    positions: object
    velocities: object
    accel_body: object
    contact_chol: object
    carry: TwoStageCarry
    imu_to_body: object
    clock_domain: str
    world_frame: str
    contact_process_model: str
    sensor_processing: str
    initialization_source: str
    accel_bias_body: object
    truth: MocapTruth | None = None

    def with_truth(
        self, pose_csv, velocity_csv, *, clock_domain, world_frame, max_offset_ns
    ):
        from dataclasses import replace

        if clock_domain != self.clock_domain or world_frame != self.world_frame:
            raise ValueError(
                "truth clock/world frame differs; register and synchronize before loading"
            )
        truth = load_truth(
            pose_csv, velocity_csv, self.timestamps_ns, max_offset_ns=max_offset_ns
        )
        if not np.asarray(truth.valid).any():
            raise ValueError("no valid truth samples")
        return replace(self, truth=truth)


def prepare_session(
    window,
    channels,
    clock,
    model,
    build,
    joint_params,
    ekf,
    initial,
    *,
    imu_to_body,
    world_frame,
    accel_bias_body=None,
    stationary_window=None,
    stationary_specific_force_body=None,
    firm_variance=None,
    swing_variance=None,
    stationary_limits=(0.15, 0.05, 0.2),
):
    """Validate a contiguous LogWindow and build a fresh independent session.

    Provide a fixed body-frame bias OR a declared stationary prefix [0,end).
    Specific force at rest must be supplied independently of held-out mocap.
    Default contact process noise is the existing constant ekf.sigma_c. A
    probability heuristic is opt-in and cannot use the v1 constant artifact.
    """
    names = channels.required_channels()
    missing = [n for n in names if n not in window.channels]
    if missing:
        raise ValueError(f"missing required channels: {missing}")
    for actual, expected, label in (
        (channels.positions, model.joint_names, "positions"),
        (channels.velocities, model.joint_names, "velocities"),
        (channels.gyros, build.imu_names, "gyros"),
        (channels.anchor_trust, model.contact_names, "anchor_trust"),
        (channels.contact_probability, model.contact_names, "contact_probability"),
    ):
        if set(actual) != set(expected):
            raise ValueError(f"{label}: expected exactly {tuple(expected)}")
    if (
        len(model.contact_names) != build.n_anchors
        or ekf.N != build.n_anchors
        or len(model.unfiltered_names) != build.anchor_unfiltered_mask.shape[1]
    ):
        raise ValueError("model/build contact or unfiltered-joint count mismatch")
    if len(set(model.joint_names)) != len(model.joint_names) or not set(
        build.joint_names
    ).issubset(model.joint_names):
        raise ValueError("invalid model joint names")
    T = len(window.time)
    time = _array(window.time, (T,), "time")
    if T < 2:
        raise ValueError("at least two ticks required")
    tick = np.asarray(window.tick)
    if (
        tick.shape != (T,)
        or tick.dtype.kind not in "iu"
        or not np.all(np.diff(tick) == 1)
    ):
        raise ValueError(
            "ticks must be contiguous: split gaps into independent sessions; do not stride"
        )
    dt = float(joint_params.dt)
    if (
        not np.isfinite(dt)
        or dt <= 0
        or not np.isclose(dt, float(ekf.params.dt), rtol=0, atol=1e-12)
    ):
        raise ValueError("JointKF/InEKF dt mismatch")
    if not np.isclose(window.dt, dt, rtol=0, atol=1e-12) or not np.allclose(
        np.diff(time), dt, rtol=0, atol=1e-9
    ):
        raise ValueError("log cadence differs from filter dt; no implicit resampling")
    timestamps = clock.timestamps(time)
    if not np.all(np.diff(timestamps) > 0) or not world_frame.strip():
        raise ValueError("invalid timestamp/world frame")
    for name in names:
        _array(window.channels[name], (T,), name)

    def stack(mapping, order):
        return (
            np.column_stack([window.channels[mapping[n]] for n in order])
            if order
            else np.empty((T, 0))
        )

    q = stack(channels.positions, model.joint_names)
    qd = stack(channels.velocities, model.joint_names)
    encoders = stack(channels.positions, build.joint_names)
    gyros = np.stack([window.stack(channels.gyros[n]) for n in build.imu_names], axis=1)
    trust = stack(channels.anchor_trust, model.contact_names)
    probability = stack(channels.contact_probability, model.contact_names)
    if not np.isin(trust, [0, 1]).all():
        raise ValueError(
            "anchor_trust must be binary, not smoothed contact probability"
        )
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("contact_probability must lie in [0,1]")
    rotation = _rotation(imu_to_body, "imu_to_body")
    if hasattr(model, "imu_to_body") and not np.allclose(
        rotation, model.imu_to_body, atol=1e-9, rtol=0
    ):
        raise ValueError("imu_to_body disagrees with model measurement frames")
    accel = window.stack(channels.accel) @ rotation.T
    if accel_bias_body is None:
        if stationary_window is None or stationary_specific_force_body is None:
            raise ValueError("explicit accel bias or stationary calibration required")
        start, end = stationary_window
        if (
            type(start) is not int
            or type(end) is not int
            or start != 0
            or not 2 <= end <= T
        ):
            raise ValueError(
                "stationary_window must be a prefix [0,end) with at least two samples"
            )
        force = _array(
            stationary_specific_force_body, (3,), "stationary specific force"
        )
        limits = _array(stationary_limits, (3,), "stationary_limits")
        if np.any(limits <= 0):
            raise ValueError("stationary limits must be positive")
        if (
            np.max(np.linalg.norm(gyros[:end, build.base_imu], axis=-1)) > limits[0]
            or np.max(np.abs(qd[:end])) > limits[1]
            or np.max(np.std(accel[:end], axis=0)) > limits[2]
        ):
            raise ValueError("declared stationary calibration window contains motion")
        accel_bias_body = estimate_static_accel_bias(
            jnp.asarray(accel[:end]), jnp.asarray(force)
        )
    elif stationary_window is not None or stationary_specific_force_body is not None:
        raise ValueError("choose fixed bias OR stationary calibration")
    bias = _array(accel_bias_body, (3,), "accel_bias_body")
    corrected = accel_body_from_raw(jnp.asarray(accel), jnp.asarray(bias))
    if firm_variance is None and swing_variance is None:
        sigma = _array(ekf.sigma_c, (ekf.N, 3, 3), "contact covariance")
        chol = np.broadcast_to(np.linalg.cholesky(sigma), (T, ekf.N, 3, 3)).copy()
        process_model = "constant_body_isotropic"
        if not np.allclose(sigma, np.eye(3)[None] * sigma[0, 0, 0], rtol=0, atol=1e-15):
            raise ValueError("v1 constant contact covariance must be common isotropic")
    else:
        if (
            firm_variance is None
            or swing_variance is None
            or not np.isfinite([firm_variance, swing_variance]).all()
        ):
            raise ValueError("both finite contact variances required")
        chol = jax.vmap(
            lambda p: contact_chol_heuristic(p, firm_variance, swing_variance)
        )(jnp.asarray(probability))
        process_model = "probability_body_isotropic"

    R = _rotation(initial.rotation, "initial rotation")
    v = _array(initial.velocity, (3,), "initial velocity")
    p = _array(initial.position, (3,), "initial position")
    P = _array(
        initial.covariance, (ekf.tangent_size, ekf.tangent_size), "initial covariance"
    )
    if (
        not np.allclose(P, P.T, atol=1e-12, rtol=0)
        or np.linalg.eigvalsh(P).min() < -1e-12
    ):
        raise ValueError("initial covariance must be symmetric PSD")
    if not initial.source.strip():
        raise ValueError("initialization source must be recorded")
    joint = joint_filter.init_carry(build, joint_params, jnp.asarray(encoders[0]))
    # Feet start untrusted, matching init_carry's causal previous-tick rule.
    frames = model.contact_frames(
        joint.state.x[: build.n_joints],
        jnp.zeros(build.n_joints),
        jnp.asarray(q[0]),
        jnp.asarray(qd[0]),
    )
    contacts = _array(frames.y, (ekf.N, 3), "initial contact FK") @ R.T + p
    base = base_ekf.initialize(
        ekf,
        jnp.asarray(R),
        jnp.asarray(v),
        jnp.asarray(p),
        jnp.asarray(contacts),
        jnp.asarray(P),
    )
    sensors = joint_filter.SensorInputs(
        jnp.asarray(encoders),
        jnp.asarray(gyros),
        jnp.asarray(stack(channels.velocities, model.unfiltered_names)),
        jnp.asarray(trust),
    )
    return PreparedSession(
        timestamps,
        sensors,
        jnp.asarray(q),
        jnp.asarray(qd),
        corrected,
        jnp.asarray(chol),
        TwoStageCarry(joint, base_filter.init_carry(base)),
        jnp.asarray(rotation),
        clock.clock_domain,
        world_frame,
        process_model,
        channels.sensor_processing,
        initial.source,
        jnp.asarray(bias),
    )


def run_session(theta, spec, build, joint_params, ekf, model, session):
    """All ticks run, including ticks without mocap. Re-evaluate model at carry q.

    JointKF linearization is at the entering carry (not its post-encoder state);
    this is the existing JointKF API boundary, not a Java-policy parity claim.
    InEKF FK is evaluated at the updated joint estimate and live unfiltered q.
    """

    def step(carry, values):
        sensors, q, qd, accel, chol = values
        mdl = model.model_inputs(carry.joint.state.x[: build.n_joints], q)

        def kin(filtered_q, filtered_qd):
            return model.contact_frames(filtered_q, filtered_qd, q, qd)

        tick = make_step(
            theta, spec, build, joint_params, ekf, kin, session.imu_to_body
        )
        return tick(carry, TwoStageInputs(sensors, mdl, accel, chol))

    return jax.lax.scan(
        step,
        session.carry,
        (
            session.sensors,
            session.positions,
            session.velocities,
            session.accel_body,
            session.contact_chol,
        ),
    )


def session_loss(theta, spec, build, joint_params, ekf, model, session):
    if session.truth is None:
        raise ValueError("mocap truth required for supervised loss")
    _, out = run_session(theta, spec, build, joint_params, ekf, model, session)
    R, v, estimated_R, estimated_v = select_valid(
        session.truth, out.base.state.R, out.base.state.v
    )
    return body_velocity_l2(estimated_R, estimated_v, R, v)


def read_session(log_directory, channels, *args, start=0.0, end=None, **kwargs):
    """Binary-reader entry point; same adapter as synthetic LogWindow tests.

    args begins with clock, model, build, joint_params, ekf, initial. The
    optional external decoder is needed here, not for prepare_session.
    """
    from ..replay.logsource import read_window

    window = read_window(
        log_directory, channels.required_channels(), start=start, end=end, stride=1
    )
    return prepare_session(window, channels, *args, **kwargs)
