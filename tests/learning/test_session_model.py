"""MJX glue checked against MuJoCo FK finite differences, including ankle motion."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from invariant_estimation.jointKF.build import KinematicTree, build_joint_kf
from invariant_estimation.model.mjx_model import MjxModel
from invariant_estimation.learning.session_model import MjxSessionModel


@pytest.fixture(scope="module")
def binding():
    xml = """<mujoco><worldbody><body name="pelvis">
    <freejoint/><inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
    <site name="body" pos=".1 0 0"/><site name="base" euler="0 0 30"/>
    <body name="shin" pos="0 0 -.3"><joint name="knee" axis="0 1 0" armature=".1"/>
    <inertial pos="0 0 -.2" mass="2" diaginertia=".1 .1 .1"/><site name="shin_imu"/>
    <body name="foot_body" pos="0 0 -.4"><joint name="ankle" axis="1 0 0" armature=".05"/>
    <inertial pos=".1 0 -.1" mass="1" diaginertia=".05 .05 .05"/>
    <site name="foot" pos=".1 .02 -.15"/>
    </body></body></body></worldbody></mujoco>"""
    model = MjxModel.from_xml_string(
        xml, site_names=("body", "base", "shin_imu", "foot"), pairs=((1, 2),)
    )
    mj = model.mj_model
    tree = KinematicTree(
        ("knee", "ankle"),
        np.array([2, 3]),
        np.asarray(mj.body_parentid),
        np.array([6, 7]),
        np.arange(6),
        {"base": 1, "shin_imu": 2, "foot": 3},
        np.ones(2) * 10,
    )
    build = build_joint_kf(
        tree, imu_sites=("base", "shin_imu"), pairs=((0, 1),), foot_sites=("foot",)
    )
    return MjxSessionModel(model, build, body_site="body", contact_sites=("foot",))


def independent_position(binding, q):
    mj = binding.model.mj_model
    d = mujoco.MjData(mj)
    d.qpos[:] = mj.qpos0
    d.qpos[binding.qpos_indices] = q
    mujoco.mj_forward(mj, d)
    ids = binding.model.site_ids
    R = d.site_xmat[ids[binding.body]].reshape(3, 3)
    return (R.T @ (d.site_xpos[ids[binding.feet[0]]] - d.site_xpos[ids[binding.body]]))[
        None
    ]


def test_contact_fk_and_jacobian_include_live_unfiltered_ankle(binding):
    q = np.array([0.31, -0.24])
    frames = binding.contact_frames(
        jnp.array(q[:1]), jnp.zeros(1), jnp.array(q), jnp.zeros(2)
    )
    expected = independent_position(binding, q)
    np.testing.assert_allclose(frames.y, expected, atol=1e-12)
    eps = 1e-6
    fd = (
        independent_position(binding, q + [eps, 0])
        - independent_position(binding, q - [eps, 0])
    ) / (2 * eps)
    np.testing.assert_allclose(frames.J[:, :, 0], fd, atol=1e-9)
    ankle_changed = binding.contact_frames(
        jnp.array(q[:1]), jnp.zeros(1), jnp.array(q + [0, 0.2]), jnp.zeros(2)
    )
    assert np.linalg.norm(np.asarray(ankle_changed.y) - expected) > 0.01
    assert binding.unfiltered_names == ("ankle",)


def test_relative_rotation_anchor_frame_and_mass_offpath_convention(binding):
    q = jnp.array([0.31, -0.24])
    mdl = binding.model_inputs(q[:1], q)
    mj = binding.model.mj_model
    d = mujoco.MjData(mj)
    d.qpos[:] = mj.qpos0
    d.qpos[binding.qpos_indices] = q
    mujoco.mj_forward(mj, d)
    ids = binding.model.site_ids
    base_R = d.site_xmat[ids[binding.base]].reshape(3, 3)
    child_R = d.site_xmat[ids[binding.imu_sites[1]]].reshape(3, 3)
    np.testing.assert_allclose(mdl.R_rel[0], child_R.T @ base_R, atol=1e-12)
    # C API site Jacobian is an independent path from MJX model.evaluate.
    jp, jr = np.zeros((3, mj.nv)), np.zeros((3, mj.nv))
    mujoco.mj_jacSite(mj, d, jp, jr, ids[binding.feet[0]])
    np.testing.assert_allclose(
        mdl.anchor_jac.unfiltered[0, :, 0], base_R.T @ jr[:, 7], atol=1e-12
    )
    np.testing.assert_allclose(
        mdl.anchor_jac.filtered[0, :, 0], base_R.T @ jr[:, 6], atol=1e-12
    )
    changed = binding.model_inputs(q[:1], q.at[1].set(0.8))
    np.testing.assert_array_equal(changed.M, mdl.M)  # welded ankle mass, live ankle FK
    assert not np.allclose(binding.model_inputs(jnp.array([0.7]), q).R_rel, mdl.R_rel)


def test_model_glue_traces_and_differentiates(binding):
    q = jnp.array([0.31, -0.24])
    fn = jax.jit(
        lambda x: binding.contact_frames(x[:1], jnp.zeros(1), x, jnp.zeros(2)).y
    )
    np.testing.assert_allclose(
        fn(q), independent_position(binding, np.asarray(q)), atol=1e-12
    )
    assert np.isfinite(jax.jacfwd(fn)(q)).all()


def test_named_log_runs_both_filters_with_mjx_inside_gradient(binding):
    from pathlib import Path
    from invariant_estimation.inEKF import ekf
    from invariant_estimation.jointKF.state import default_params
    from invariant_estimation.learning.log_adapter import (
        ChannelMap,
        ClockMapping,
        InitialState,
        prepare_session,
        run_session,
    )
    from invariant_estimation.learning.noise import NoiseSpec
    from invariant_estimation.replay.logsource import LogWindow

    T, dt = 4, 0.001
    mapping = ChannelMap(
        {n: "q_" + n for n in binding.joint_names},
        {n: "qd_" + n for n in binding.joint_names},
        {n: tuple(n + "_" + a for a in "xyz") for n in binding.build.imu_names},
        ("ax", "ay", "az"),
        {"foot": "trust"},
        {"foot": "prob"},
        "test-SI",
    )
    channels = {n: np.zeros(T) for n in mapping.required_channels()}
    channels["q_knee"][:] = 0.1
    channels["q_ankle"][:] = -0.1
    channels["ax"][:] = 0.1
    channels["az"][:] = 9.81
    channels["trust"][:] = 1
    channels["prob"][:] = 1
    window = LogWindow(
        Path("synthetic-mjx"), np.arange(T) * dt, np.arange(T), channels, dt
    )
    params = default_params(dt=dt)
    base = ekf.create(1, dt=dt)
    session = prepare_session(
        window,
        mapping,
        ClockMapping(0.0, 0, "sim_ns"),
        binding,
        binding.build,
        params,
        base,
        InitialState(np.eye(3), np.zeros(3), np.zeros(3), np.eye(12) * 0.01, "fixed"),
        imu_to_body=binding.imu_to_body,
        world_frame="sim_world",
        accel_bias_body=np.zeros(3),
    )
    spec = NoiseSpec(tuple(binding.build.imu_names))

    def objective(theta):
        _, out = run_session(theta, spec, binding.build, params, base, binding, session)
        return jnp.sum(out.base.state.v**2)

    value, grad = jax.jit(jax.value_and_grad(objective))(spec.initial_theta())
    assert np.isfinite(value) and np.isfinite(grad).all()
    assert np.linalg.norm(grad) > 1e-16
