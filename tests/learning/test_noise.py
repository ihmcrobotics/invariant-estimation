import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import ekf as base_ekf
from invariant_estimation.inEKF import filter as base_filter
from invariant_estimation.inEKF.group import Gamma0
from invariant_estimation.jointKF import filter as joint_filter, measure
from invariant_estimation.jointKF.anchors import AnchorJacobians
from invariant_estimation.jointKF.build import KinematicTree, build_joint_kf
from invariant_estimation.jointKF.state import default_params
from invariant_estimation.learning.noise import (
    NoiseSpec, apply_joint_noise, apply_inekf_noise, apply_inekf_inputs, run_inekf,
)
from invariant_estimation.learning.two_stage import TwoStageCarry, TwoStageInputs, run
from invariant_estimation.learning.optimize import body_velocity_l2, fit_scalars


def scene():
    tree = KinematicTree(
        joint_names=("j0", "j1", "j2"), joint_body=np.array([1, 2, 3]),
        body_parent=np.array([-1, 0, 1, 2]), joint_dof=np.array([6, 7, 8]),
        base_dofs=np.arange(6), site_body={"base": 0, "mid": 2, "tip": 3, "foot": 3},
        tau_max=np.ones(3)*10,
    )
    build = build_joint_kf(tree, imu_sites=["base", "mid", "tip"],
                         pairs=[(0, 1), (0, 2)], foot_sites=["foot"])
    build = build._replace(use_mass_matrix=False, gyro_sigma=jnp.stack([
        jnp.diag(jnp.array([0.003, 0.004, 0.005])*(i+1)) for i in range(3)]))
    params = default_params(dt=0.02, sigma_accel=0.3, cond_s_max=1e14)
    ekf = base_ekf.create(1, dt=0.02, gyro_var=0.01, accel_var=0.1)
    q = jnp.array([0.1, -0.2, 0.05])

    def kin(q, qd):
        return base_filter.ContactFrames(
            jnp.array([[0., 0., -0.9]]) + 0.1*q[None],
            0.1*jnp.eye(3)[None], jnp.zeros((1, 3, 3)))

    state = base_ekf.initialize(ekf, Gamma0(jnp.array([0.02, -0.03, 0.01])),
                                jnp.array([0.04, -0.03, 0.02]), jnp.zeros(3),
                                kin(q, q).y, jnp.eye(12)*0.01)
    joint = joint_filter.init_carry(build, params, q)._replace(trusted_feet=jnp.ones(1))
    carry = TwoStageCarry(joint, base_filter.init_carry(state))
    inputs = TwoStageInputs(
        joint_filter.SensorInputs(q + 0.01, jnp.array([[.01, .02, .01], [.03, -.01, .01], [.02, .04, -.01]]),
                                  jnp.zeros(0), jnp.ones(1)),
        joint_filter.ModelInputs(jnp.stack([jnp.diag(jnp.array([1., 1., 0.])), jnp.eye(3)]),
                                 jnp.tile(jnp.eye(3), (2, 1, 1)),
                                 AnchorJacobians(jnp.eye(3)[None], jnp.zeros((1, 3, 0)))),
        jnp.array([0.02, -0.03, 9.81]), jnp.eye(3)[None]*0.03,
    )
    return build, params, ekf, kin, carry, inputs


def stack(tick, count=8):
    return jax.tree.map(lambda x: jnp.repeat(x[None], count, axis=0), tick)


@pytest.mark.parametrize("arm", [4, 5, 6, 7])
def test_scales_identity_bounds_and_arm_gradient(arm):
    spec = NoiseSpec(("base", "leg"), arm=arm)
    np.testing.assert_array_equal(jnp.concatenate((spec.scales(spec.initial_theta()).imu_gyro,
                                                 jnp.array(spec.scales(spec.initial_theta())[1:]))), 1.)
    def values(t):
        s = spec.scales(t)
        return jnp.concatenate((s.imu_gyro, jnp.array(s[1:])))
    jac = jax.jit(jax.jacrev(values))(spec.initial_theta())
    np.testing.assert_allclose(jnp.diag(jac), [arm in (5, 7)]*2 + [arm in (6, 7)]*6, atol=1e-15)
    extremes = values(jnp.linspace(-1e6, 1e6, 8))
    assert np.all(np.asarray(extremes) >= 0.01-1e-14)
    assert np.all(np.asarray(extremes) <= 100+1e-12)


def test_scaled_raw_imus_preserve_shared_pair_covariance():
    build, params, _, _, _, tick = scene()
    spec = NoiseSpec(build.imu_names)
    scales = spec.scales(spec.initial_theta().at[0].set(0.7))
    learned = apply_joint_noise(build, scales)
    result = measure.build_stacked(learned, params, gyros=tick.sensors.gyros,
                                    J_rel=tick.model.J_rel, R_rel=tick.model.R_rel)
    # Shared BASE IMU induces this exact off-diagonal pair block.
    np.testing.assert_allclose(result.R[:3, 3:6], learned.gyro_sigma[0], atol=1e-14)
    assert np.linalg.eigvalsh(np.asarray(result.R[:6, :6])).min() > 0
    np.testing.assert_allclose(build.gyro_sigma[0], np.diag([.003, .004, .005]))
    with pytest.raises(ValueError, match="order"):
        NoiseSpec(tuple(reversed(build.imu_names))).validate_build(build)


def test_inekf_baseline_identity_and_input_mean_unchanged():
    build, _, ekf, kin, carry, tick = scene()
    spec = NoiseSpec(build.imu_names)
    joint = base_filter.JointFilterOutput(tick.sensors.encoders, jnp.ones(3)*.01,
                                          jnp.eye(3)*.003, jnp.eye(3)*.01)
    inputs = stack(base_filter.InEKFInputs(tick.sensors.gyros[0], tick.accel_body,
                                         tick.sensors.gyros[0], joint, tick.contact_chol))
    actual = run_inekf(spec.initial_theta(), spec, ekf, kin, carry.base.state, inputs)
    expected = base_filter.run(ekf, kin, carry.base.state, inputs)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=1e-13, equal_nan=True)
    changed = apply_inekf_inputs(inputs, spec.scales(jnp.ones(len(spec.names))))
    np.testing.assert_array_equal(changed.joint.q, inputs.joint.q)
    np.testing.assert_array_equal(changed.omega, inputs.omega)
    np.testing.assert_array_equal(changed.joint.sigma_q_dot, inputs.joint.sigma_q_dot)


def test_fused_bptt_all_noise_channels_and_finite_difference():
    build, params, ekf, kin, carry, tick = scene()
    spec = NoiseSpec(build.imu_names)
    inputs = stack(tick)
    def loss(theta):
        _, out = run(theta, spec, build, params, ekf, kin, jnp.eye(3), carry, inputs)
        # State L2 only: every parameter must actually reach the trajectory.
        return body_velocity_l2(out.base.state.R, out.base.state.v,
                                jnp.tile(jnp.eye(3), (8, 1, 1)), jnp.zeros((8, 3)))
    theta = jnp.ones(len(spec.names))*.1
    value, grad = jax.jit(jax.value_and_grad(loss))(theta)
    assert np.isfinite(value) and np.isfinite(np.asarray(grad)).all()
    assert np.all(np.abs(np.asarray(grad)) > 1e-14), grad
    evaluate = jax.jit(loss)
    for i in range(len(spec.names)):
        h = 1e-4
        fd = (evaluate(theta.at[i].add(h)) - evaluate(theta.at[i].add(-h))) / (2*h)
        np.testing.assert_allclose(grad[i], fd, rtol=2e-3, atol=2e-9)


def test_synthetic_state_loss_decreases_without_claiming_noise_identifiability():
    build, params, ekf, kin, carry, tick = scene()
    spec = NoiseSpec(build.imu_names, arm=7)
    inputs = stack(tick)
    rng = np.random.default_rng(42)
    inputs = inputs._replace(accel_body=inputs.accel_body + jnp.array(rng.normal(0, .03, (8, 3))))
    def loss(theta):
        _, out = run(theta, spec, build, params, ekf, kin, jnp.eye(3), carry, inputs)
        return body_velocity_l2(out.base.state.R, out.base.state.v,
                                jnp.tile(jnp.eye(3), (8, 1, 1)), jnp.zeros((8, 3)))
    result = fit_scalars(loss, spec.initial_theta(), steps=15, learning_rate=.1)
    assert min(result.losses) < result.losses[0]*.95


def test_identifiable_gaussian_noise_recovery_separate_from_filter_l2():
    # Independently observed zero-mean residuals: optimum is empirical variance,
    # NOT an assertion that arbitrary filter Q/R are recoverable from pose L2.
    samples = np.random.default_rng(15).normal(0, 2, 4096)
    target = float(np.mean(samples**2))
    spec = NoiseSpec(("base",), arm=5)
    def nll(theta):
        variance = spec.scales(theta).imu_gyro[0]
        return .5*(jnp.log(variance) + target/variance)
    result = fit_scalars(nll, spec.initial_theta(), steps=250, learning_rate=.04)
    np.testing.assert_allclose(spec.scales(result.theta).imu_gyro[0], target, rtol=.01)


def test_optimizer_rejects_nan():
    with pytest.raises(FloatingPointError):
        fit_scalars(lambda theta: jnp.sum(theta)*jnp.nan, jnp.zeros(2), steps=1)


@pytest.mark.parametrize("kwargs", [{"arm": 3}, {"max_scale": 1}, {"max_scale": float("nan")}])
def test_bad_schema(kwargs):
    with pytest.raises(ValueError):
        NoiseSpec(("base",), **kwargs)
