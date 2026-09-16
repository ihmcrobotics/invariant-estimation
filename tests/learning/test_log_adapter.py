"""Exercise named LogWindow inputs, not preassembled filter inputs."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.learning.log_adapter import (
    ChannelMap,
    ClockMapping,
    InitialState,
    prepare_session,
    run_session,
    session_loss,
)
from invariant_estimation.learning.synthetic import generate_session
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.learning.optimize import fit_scalars
from invariant_estimation.learning.artifact import (
    from_fit,
    save_artifact,
    load_artifact,
)
from invariant_estimation.learning.mocap import MocapTruth
from invariant_estimation.replay.logsource import LogWindow


class AnalyticModel:
    """Same 3-coordinate abstraction as synthetic.py, not an Alex model."""

    joint_names = ("c0", "c1", "c2")
    unfiltered_names = ()
    contact_names = ("foot",)

    def __init__(self, data):
        self.data = data

    def model_inputs(self, q, all_q):
        return jax.tree.map(lambda x: x[0], self.data.inputs.model)

    def contact_frames(self, q, qd, all_q, all_qd):
        return self.data.kinematics(q, qd)


def fixture(seed=1, ticks=16):
    data = generate_session(seed, ticks=ticks)
    model = AnalyticModel(data)
    mapping = ChannelMap(
        {n: f"processed_q_{n}" for n in reversed(model.joint_names)},
        {n: f"processed_qd_{n}" for n in reversed(model.joint_names)},
        {
            n: tuple(f"gyro_{n}_{axis}" for axis in "xyz")
            for n in reversed(data.build.imu_names)
        },
        tuple(f"accel_{axis}" for axis in "xyz"),
        {"foot": "foot_trusted"},
        {"foot": "foot_probability"},
        "synthetic processed SI channels",
    )
    channels = {}
    for i, n in enumerate(model.joint_names):
        channels[mapping.positions[n]] = np.asarray(
            data.inputs.sensors.encoders[:, i]
        ).copy()
        channels[mapping.velocities[n]] = np.zeros(ticks)
    for i, n in enumerate(data.build.imu_names):
        for k, name in enumerate(mapping.gyros[n]):
            channels[name] = np.asarray(data.inputs.sensors.gyros[:, i, k]).copy()
    for k, name in enumerate(mapping.accel):
        channels[name] = np.asarray(data.inputs.accel_body[:, k]).copy()
    channels["foot_trusted"] = np.ones(ticks)
    channels["foot_probability"] = np.ones(ticks)
    dt = float(data.joint_params.dt)
    window = LogWindow(
        Path("synthetic"),
        12.0 + np.arange(ticks) * dt,
        np.arange(300, 300 + ticks),
        channels,
        dt,
    )
    clock = ClockMapping(12.0, 1_800_000_000_000_000_017, "robot_monotonic_ns")
    initial = InitialState(
        np.eye(3),
        np.zeros(3),
        np.array([1.0, 2.0, 3.0]),
        np.eye(12) * 1e-3,
        "fixed test prior",
    )
    return data, model, mapping, window, clock, initial


def prepared(f, **kwargs):
    data, model, mapping, window, clock, initial = f
    return prepare_session(
        window,
        mapping,
        clock,
        model,
        data.build,
        data.joint_params,
        data.ekf,
        initial,
        imu_to_body=kwargs.pop("imu_to_body", np.eye(3)),
        world_frame="registered_world",
        **({"accel_bias_body": np.zeros(3)} | kwargs),
    )


def truth_for(data, ticks, invalid=()):
    valid = np.ones(ticks, dtype=bool)
    valid[list(invalid)] = False
    return MocapTruth(
        data.truth_rotation,
        data.truth_velocity,
        jnp.zeros((ticks, 3)),
        jnp.zeros((ticks, 3)),
        jnp.asarray(valid),
        jnp.zeros(ticks, dtype=jnp.int64),
    )


def test_names_not_dictionary_order_and_exact_nanosecond_origin():
    f = fixture()
    data, model, mapping, window, clock, initial = f
    session = prepared(f)
    np.testing.assert_array_equal(
        session.sensors.encoders, data.inputs.sensors.encoders
    )
    np.testing.assert_array_equal(session.sensors.gyros, data.inputs.sensors.gyros)
    np.testing.assert_array_equal(
        session.timestamps_ns, clock.target_origin_ns + np.arange(16) * 4_000_000
    )
    # Initial contact state is WORLD position, not the body-frame FK vector.
    np.testing.assert_allclose(
        session.carry.base.state.d[0], session.sensors.encoders[0] + initial.position
    )
    np.testing.assert_array_equal(session.carry.joint.trusted_feet, [0.0])


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "missing required"),
        ("nan", "finite shape"),
        ("gap", "contiguous"),
        ("time", "cadence"),
        ("trust", "binary"),
        ("probability", "\\[0,1\\]"),
        ("rotation", "proper rotation"),
        ("mapping", "duplicate physical"),
    ],
)
def test_rejects_bad_log_inputs(problem, match):
    f = list(fixture())
    window = f[3]
    kwargs = {}
    if problem == "missing":
        del window.channels["accel_x"]
    if problem == "nan":
        window.channels["accel_x"][3] = np.nan
    if problem == "gap":
        f[3] = replace(window, tick=window.tick * 2)
    if problem == "time":
        f[3] = replace(window, time=window.time * 2)
    if problem == "trust":
        window.channels["foot_trusted"][2] = 0.01
    if problem == "probability":
        window.channels["foot_probability"][2] = 1.1
    if problem == "rotation":
        kwargs["imu_to_body"] = np.eye(3) * 2
    if problem == "mapping":
        f[2] = replace(f[2], accel=("accel_x",) * 3)
    with pytest.raises(ValueError, match=match):
        prepared(f, **kwargs)


def test_rotation_bias_and_declared_stationary_calibration():
    f = fixture()
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    bias = np.array([0.2, -0.3, 0.1])
    force = np.array([0.0, 0.0, 9.81])
    raw_imu = R.T @ (force + bias)
    for k, name in enumerate(f[2].accel):
        f[3].channels[name][:] = raw_imu[k]
    for name in f[2].gyros[f[0].build.imu_names[f[0].build.base_imu]]:
        f[3].channels[name][:] = 0
    session = prepared(
        f,
        imu_to_body=R,
        accel_bias_body=None,
        stationary_window=(0, 4),
        stationary_specific_force_body=force,
    )
    np.testing.assert_allclose(session.accel_bias_body, bias, atol=1e-14)
    np.testing.assert_allclose(session.accel_body, np.tile(force, (16, 1)), atol=1e-14)
    f[3].channels["processed_qd_c0"][0] = 2.0
    with pytest.raises(ValueError, match="contains motion"):
        prepared(
            f,
            imu_to_body=R,
            accel_bias_body=None,
            stationary_window=(0, 4),
            stationary_specific_force_body=force,
        )


def test_contact_schedule_is_explicit_and_not_v1_constant_artifact():
    f = fixture()
    f[3].channels["foot_probability"][1] = 0.0
    session = prepared(f, firm_variance=1e-6, swing_variance=1.0)
    np.testing.assert_allclose(session.contact_chol[0, 0], np.eye(3) * 0.001)
    np.testing.assert_allclose(session.contact_chol[1, 0], np.eye(3))
    assert session.contact_process_model == "probability_body_isotropic"


def test_csv_time_alignment_drops_only_loss_samples(tmp_path):
    session = prepared(fixture())
    pose = tmp_path / "pelvis.csv"
    velocity = tmp_path / "pelvisVelocity.csv"
    # Half-period offset, one missing mocap sample. No sensor tick is removed.
    pose.write_text(
        "timestamp_ns,x,y,z,qx,qy,qz,qs\n"
        + "".join(
            f"{int(t) + 100},0,0,0,0,0,0,1\n"
            for i, t in enumerate(session.timestamps_ns)
            if i != 7
        )
    )
    velocity.write_text(
        "timestamp_ns,vx,vy,vz,wx,wy,wz\n"
        + "".join(
            f"{int(t) + 100},0,0,0,0,0,0\n"
            for i, t in enumerate(session.timestamps_ns)
            if i != 7
        )
    )
    aligned = session.with_truth(
        pose,
        velocity,
        clock_domain=session.clock_domain,
        world_frame=session.world_frame,
        max_offset_ns=200,
    )
    assert not bool(aligned.truth.valid[7])
    assert int(aligned.truth.valid.sum()) == 15
    assert aligned.sensors.encoders.shape[0] == 16
    assert aligned.truth.worst_offset_ns() == 100
    with pytest.raises(ValueError, match="clock/world"):
        session.with_truth(
            pose,
            velocity,
            clock_domain="wall_clock",
            world_frame=session.world_frame,
            max_offset_ns=200,
        )


def test_log_to_train_holdout_and_artifact_with_dropout(tmp_path):
    train_f, test_f = fixture(1), fixture(8)
    data, model = train_f[:2]
    train = replace(prepared(train_f), truth=truth_for(data, 16, (5, 6)))
    test = replace(prepared(test_f), truth=truth_for(test_f[0], 16, (3,)))
    spec = NoiseSpec(tuple(data.build.imu_names), arm=7)
    args = (spec, data.build, data.joint_params, data.ekf, model)

    def loss(theta):
        return session_loss(theta, *args, train)

    fit = fit_scalars(loss, spec.initial_theta(), steps=4)
    assert fit.losses[-1] < fit.losses[0]
    assert np.isfinite(session_loss(fit.theta, *args, test))
    _, out = run_session(fit.theta, *args, train)
    assert out.base.state.v.shape == (16, 3)
    # Corrupting truth at an excluded tick cannot poison BPTT gradients.
    poisoned = replace(
        train,
        truth=train.truth._replace(velocity=train.truth.velocity.at[5:7].set(jnp.nan)),
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: session_loss(x, *args, poisoned))(fit.theta),
        jax.grad(loss)(fit.theta),
        rtol=1e-10,
        atol=1e-12,
    )
    artifact = from_fit(
        fit.theta,
        spec,
        context={
            "robot_id": "synthetic",
            "model_sha256": "a" * 64,
            "dt": float(data.joint_params.dt),
            "imu_names": list(spec.imu_names),
            "joint_names": list(data.build.joint_names),
            "contact_names": list(model.contact_names),
            "base_frame": "pelvis",
            "contact_process_model": train.contact_process_model,
            "fk_noise_model": "joint_covariance",
        },
        provenance={
            "created_at": "2026-09-16T12:00:00-05:00",
            "python_revision": "test",
            "java_revision": "test",
            "capture_manifest_sha256": "b" * 64,
            "baseline_config_sha256": "c" * 64,
            "train_sessions": ["train"],
            "validation_sessions": [],
            "test_sessions": ["holdout"],
            "objective": "body_velocity_l2",
        },
        baseline={
            "base_gyro_q": float(data.ekf.params.gyro_var),
            "base_accel_q": float(data.ekf.params.accel_var),
            "contact_q": float(data.ekf.sigma_c[0, 0, 0]),
            "gravity_roll_r": float(data.ekf.gravity_params.roll_var),
            "gravity_pitch_r": float(data.ekf.gravity_params.pitch_var),
            "encoder_fallback_var": 5e-5,
            "imu_gyro_covariances": dict(
                zip(spec.imu_names, np.asarray(data.build.gyro_sigma))
            ),
        },
    )
    save_artifact(tmp_path / "noise.json", artifact)
    assert load_artifact(tmp_path / "noise.json") == artifact
    # Sessions always start fresh, even after another scan has finished.
    _, repeat = run_session(fit.theta, *args, train)
    np.testing.assert_array_equal(repeat.base.state.P, out.base.state.P)
