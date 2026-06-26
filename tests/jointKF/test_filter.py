"""Tests for joint_kf/filter.py — the predict→update orchestrator + scan.

Also verifies the package-wide float64 setting from invariant_estimation/__init__.
"""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from invariant_estimation import robot as robot_mod
from invariant_estimation.jointKF import measurement as meas
from invariant_estimation.jointKF import noise
from invariant_estimation.jointKF.filter import SensorInputs, run, step
from invariant_estimation.jointKF.predict import predict
from invariant_estimation.jointKF.state import default_params, init_state
from invariant_estimation.jointKF.update import update


# (n_joints, n_pairs), including the encoder-only edge case m = 0.
SHAPES = [(6, 2), (1, 1), (4, 0), (12, 3)]


class DummyRobot:
    """A minimal RobotModel: SPD diagonal mass matrix, q-dependent gyro Jacobian."""

    def __init__(self, n, m):
        self.n = n
        self.m = m

    def mass_matrix(self, q):
        return jnp.diag(1.0 + 0.5 * jnp.sin(q))          # SPD (entries in [0.5, 1.5])

    def relative_gyro_jacobian(self, q):
        # (m, 3, n), smoothly q-dependent and deterministic.
        base = jnp.cos(jnp.outer(jnp.arange(3 * self.m), q) + q.sum())
        return base.reshape(self.m, 3, self.n)


def _sensors(n, m, t=0):
    return SensorInputs(
        q_tilde=0.1 * jnp.arange(1.0, n + 1.0) + t,
        omega_a=jnp.ones((m, 3)) * (1 + t),
        omega_b=1.5 * jnp.ones((m, 3)) * (1 + t),
        R_ba=jnp.broadcast_to(jnp.eye(3), (m, 3, 3)),
    )


def _sensor_seq(n, m, T):
    """Stack T per-step SensorInputs into a leading-time pytree."""
    steps = [_sensors(n, m, t) for t in range(T)]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *steps)


# ---------------------------------------------------------------------------
# float64
# ---------------------------------------------------------------------------

def test_float64_is_active():
    assert jnp.zeros(1).dtype == jnp.float64
    st = init_state(4, 2)
    assert st.P.dtype == jnp.float64


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_step_matches_manual_compose(n, m):
    p = default_params()
    robot = DummyRobot(n, m)
    st = init_state(n, m, q0=0.2 * jnp.arange(1.0, n + 1.0))
    sensors = _sensors(n, m)

    out, info = step(st, p, sensors, robot)

    # Manual predict → update with the same resolved arrays.
    M = robot.mass_matrix(st.q_hat)
    pred = predict(st, p, M)
    J = robot.relative_gyro_jacobian(pred.q_hat)
    z, H = meas.build_measurement(sensors.q_tilde, sensors.omega_a,
                                  sensors.omega_b, sensors.R_ba, J, n)
    R = noise.build_R(p, n, m)
    out_ref, info_ref = update(pred, z, H, R)

    assert jnp.allclose(out.x, out_ref.x)
    assert jnp.allclose(out.P, out_ref.P)
    assert jnp.allclose(info.nu, info_ref.nu)
    assert jnp.allclose(info.S, info_ref.S)


@pytest.mark.parametrize("n, m", SHAPES)
def test_step_no_robot_uses_diagonal_and_zero_jacobian(n, m):
    p = default_params()
    st = init_state(n, m)
    sensors = _sensors(n, m)

    out, _ = step(st, p, sensors, robot=None)

    pred = predict(st, p, None)                          # diagonal Q_a
    z, H = meas.build_measurement(sensors.q_tilde, sensors.omega_a,
                                  sensors.omega_b, sensors.R_ba,
                                  jnp.zeros((m, 3, n)), n)
    R = noise.build_R(p, n, m)
    out_ref, _ = update(pred, z, H, R)
    assert jnp.allclose(out.x, out_ref.x)
    assert jnp.allclose(out.P, out_ref.P)


# ---------------------------------------------------------------------------
# run / scan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", SHAPES)
def test_run_matches_python_loop(n, m):
    p = default_params()
    robot = DummyRobot(n, m)
    T = 5
    seq = _sensor_seq(n, m, T)
    st0 = init_state(n, m)

    final, states, infos = run(st0, p, seq, robot)

    # Python-loop reference.
    carry = st0
    xs, Ps, nus = [], [], []
    for t in range(T):
        sensors = jax.tree_util.tree_map(lambda a: a[t], seq)
        carry, info = step(carry, p, sensors, robot)
        xs.append(carry.x)
        Ps.append(carry.P)
        nus.append(info.nu)

    assert jnp.allclose(final.x, carry.x)
    assert jnp.allclose(final.P, carry.P)
    assert jnp.allclose(states.x, jnp.stack(xs))
    assert jnp.allclose(states.P, jnp.stack(Ps))
    assert jnp.allclose(infos.nu, jnp.stack(nus))


@pytest.mark.parametrize("n, m", SHAPES)
def test_run_trajectory_shapes(n, m):
    p = default_params()
    robot = DummyRobot(n, m)
    T = 4
    D = 2 * n + 3 * m
    dz = n + 3 * m
    final, states, infos = run(init_state(n, m), p, _sensor_seq(n, m, T), robot)
    assert states.q_hat.shape == (T, n)
    assert states.q_dot_hat.shape == (T, n)
    assert states.b_omega.shape == (T, 3 * m)
    assert states.P.shape == (T, D, D)
    assert infos.nu.shape == (T, dz)
    assert infos.S.shape == (T, dz, dz)
    assert final.P.shape == (D, D)


@pytest.mark.parametrize("n, m", SHAPES)
def test_run_covariance_psd_along_trajectory(n, m):
    p = default_params()
    robot = DummyRobot(n, m)
    _, states, _ = run(init_state(n, m), p, _sensor_seq(n, m, 6), robot)
    for P in states.P:
        assert jnp.allclose(P, P.T, atol=1e-9)
        eig = jnp.linalg.eigvalsh(P)
        assert eig.min() >= -1e-9 * jnp.maximum(eig.max(), 1.0)


def test_run_jit_matches_eager():
    n, m = 6, 2
    p = default_params()
    robot = DummyRobot(n, m)
    seq = _sensor_seq(n, m, 5)
    st0 = init_state(n, m)

    final_e, states_e, infos_e = run(st0, p, seq, robot)
    run_jit = jax.jit(partial(run, robot=robot))
    final_j, states_j, infos_j = run_jit(st0, p, seq)

    assert jnp.allclose(final_e.P, final_j.P)
    assert jnp.allclose(states_e.x, states_j.x)
    assert jnp.allclose(infos_e.S, infos_j.S)


# ---------------------------------------------------------------------------
# robot seam
# ---------------------------------------------------------------------------

def test_dummy_robot_satisfies_protocol():
    assert isinstance(DummyRobot(4, 2), robot_mod.RobotModel)
