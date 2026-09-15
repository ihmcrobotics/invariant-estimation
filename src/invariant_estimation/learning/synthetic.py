r"""Self-consistent synthetic capture sessions, for rehearsing the training pipeline
before real data exists.

Why this is not the fixture in `tests/learning/test_noise.py`
------------------------------------------------------------
Those fixtures feed the filters *arbitrary* arrays. That is fine for checking
gradients flow and shapes line up, but it cannot answer the question this module
exists for: **does training actually make the estimate better?** For that, the
sensor stream and the ground truth have to be two views of one trajectory, with
a known noise level between them.

So the trajectory is chosen first, and every sensor is derived from it
analytically, matching the filter's own conventions:

* ``Ṙ = R [ω]ₓ``           -- so the gyro is the body-frame angular rate.
* ``v̇ = R a_body + g``     -- so the accelerometer reads
  ``a_body = Rᵀ(v̇ - g)``, which is ``[0, 0, +9.81]`` for a stationary upright
  base, matching `GravityLevelingUpdater`'s own static test.
* A contact point fixed in the world at ``d`` sits at ``Rᵀ(d - p)`` in the body
  frame, which is what the encoders measure and what the contact update checks
  against.

Then Gaussian noise of a **known** covariance is added. Because the truth is
known exactly and the noise level is known exactly, a filter whose Q/R match
that noise should beat one whose Q/R do not -- which is the property the
training loop has to demonstrate before anyone trusts it on a real capture.

What this is not
----------------
Not a robot, and not a substitute for one. The joint stage here is a
3-coordinate abstraction whose "encoders" measure the contact offset directly,
not a kinematic chain with real Jacobians; the contact is always planted; there
is no ground reaction, no gait, no model error, and the noise is exactly the
i.i.d. Gaussian the filters assume. Real data violates all of that. A pipeline
that fails here is definitely broken; one that passes here is merely *not yet
known* to be broken.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..inEKF import ekf as base_ekf
from ..inEKF import filter as base_filter
from ..inEKF.group import Gamma0
from ..jointKF import filter as joint_filter
from ..jointKF.anchors import AnchorJacobians
from ..jointKF.build import KinematicTree, build_joint_kf
from ..jointKF.state import default_params
from .two_stage import TwoStageCarry, TwoStageInputs

GRAVITY = jnp.array([0.0, 0.0, -9.81])


class SyntheticSession(NamedTuple):
    """One generated session: everything `two_stage.run` needs, plus the truth it was built from."""

    build: object
    joint_params: object
    ekf: object
    kinematics: object
    carry: TwoStageCarry
    inputs: TwoStageInputs
    truth_rotation: jnp.ndarray   # (T, 3, 3)
    truth_velocity: jnp.ndarray   # (T, 3)
    gyro_sigma: float
    accel_sigma: float
    encoder_sigma: float


def _skew(w):
    return jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def true_trajectory(ticks: int, dt: float, seed: int):
    """A smooth rotating-and-translating base trajectory, integrated the way the filter does.

    Integrating ``R`` with the same ``Gamma0`` exponential the propagation uses (rather than a
    generic ODE step) matters: any mismatch would show up as a constant bias the noise parameters
    cannot fix, and would be misread as "training failed".
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    rate = jnp.array(rng.uniform(0.15, 0.35, size=3))
    amplitude = jnp.array(rng.uniform(0.05, 0.15, size=3))

    def scan(carry, t):
        R, v, p = carry
        # Velocity is a chosen analytic signal; its derivative is therefore exact, not differenced.
        omega = 0.2 * jnp.sin(rate * t + jnp.array(phase))
        v_next = amplitude * jnp.sin(rate * t + jnp.array(phase))
        v_dot = amplitude * rate * jnp.cos(rate * t + jnp.array(phase))
        accel_body = R.T @ (v_dot - GRAVITY)
        R_next = R @ Gamma0(omega * dt)
        p_next = p + v_next * dt
        return (R_next, v_next, p_next), (R, v_next, p, omega, accel_body)

    times = jnp.arange(ticks, dtype=jnp.float64) * dt
    init = (jnp.eye(3), jnp.zeros(3), jnp.zeros(3))
    _, (R, v, p, omega, accel_body) = jax.lax.scan(scan, init, times)
    return R, v, p, omega, accel_body


def generate_session(seed: int, *, ticks: int = 120, dt: float = 0.004,
                     gyro_sigma: float = 0.02, accel_sigma: float = 0.25,
                     encoder_sigma: float = 0.004) -> SyntheticSession:
    """Build one self-consistent session with the given sensor noise levels.

    The noise levels are the *ground truth* the training loop is being asked to
    discover. The filters are constructed with deliberately mismatched
    variances (see `baseline_ekf_variances`), so theta=0 is a genuinely
    suboptimal starting point rather than an already-correct one.
    """
    R_true, v_true, p_true, omega_true, accel_true = true_trajectory(ticks, dt, seed)
    rng = np.random.default_rng(seed + 9_000)

    # A single contact fixed in the world; its body-frame position is what the encoders see.
    contact_world = jnp.array([0.0, 0.0, -0.9])
    contact_body = jnp.einsum("tji,tj->ti", R_true, contact_world[None] - p_true)

    tree = KinematicTree(
        joint_names=("c0", "c1", "c2"), joint_body=np.array([1, 2, 3]),
        body_parent=np.array([-1, 0, 1, 2]), joint_dof=np.array([6, 7, 8]),
        base_dofs=np.arange(6), site_body={"base": 0, "mid": 2, "tip": 3, "foot": 3},
        tau_max=np.ones(3) * 10.0,
    )
    build = build_joint_kf(tree, imu_sites=["base", "mid", "tip"], pairs=[(0, 1), (0, 2)],
                           foot_sites=["foot"])
    build = build._replace(use_mass_matrix=False)
    joint_params = default_params(dt=dt, sigma_accel=0.3, cond_s_max=1e14)

    gyro_var, accel_var, contact_var = baseline_ekf_variances()
    ekf = base_ekf.create(1, dt=dt, gyro_var=gyro_var, accel_var=accel_var, contact_var=contact_var)

    # The joint state IS the body-frame contact offset here (see the module docstring), so the
    # kinematics map is the identity -- deliberately trivial, keeping the base filter's velocity
    # error dominated by IMU noise, which is what the InEKF channels actually control.
    def kinematics(q, q_dot):
        return base_filter.ContactFrames(q[None], jnp.eye(3)[None], jnp.zeros((1, 3, 3)))

    state = base_ekf.initialize(ekf, R_true[0], v_true[0], p_true[0],
                                contact_body[0][None], jnp.eye(12) * 1e-3)
    joint = joint_filter.init_carry(build, joint_params, contact_body[0])._replace(
        trusted_feet=jnp.ones(1))
    carry = TwoStageCarry(joint, base_filter.init_carry(state))

    encoders = contact_body + jnp.array(rng.normal(0.0, encoder_sigma, size=(ticks, 3)))
    gyros = jnp.tile(omega_true[:, None, :], (1, 3, 1)) + jnp.array(
        rng.normal(0.0, gyro_sigma, size=(ticks, 3, 3)))
    accel = accel_true + jnp.array(rng.normal(0.0, accel_sigma, size=(ticks, 3)))

    inputs = TwoStageInputs(
        sensors=joint_filter.SensorInputs(
            encoders, gyros, jnp.zeros((ticks, 0)), jnp.ones((ticks, 1))),
        model=joint_filter.ModelInputs(
            jnp.tile(jnp.stack([jnp.diag(jnp.array([1.0, 1.0, 0.0])), jnp.eye(3)])[None], (ticks, 1, 1, 1)),
            jnp.tile(jnp.tile(jnp.eye(3), (2, 1, 1))[None], (ticks, 1, 1, 1)),
            AnchorJacobians(jnp.tile(jnp.eye(3)[None][None], (ticks, 1, 1, 1)),
                            jnp.zeros((ticks, 1, 3, 0)))),
        accel_body=accel,
        contact_chol=jnp.tile((jnp.eye(3) * 1e-3)[None][None], (ticks, 1, 1, 1)),
    )

    return SyntheticSession(build, joint_params, ekf, kinematics, carry, inputs,
                            R_true, v_true, gyro_sigma, accel_sigma, encoder_sigma)


def baseline_ekf_variances():
    """The deliberately-mismatched starting variances the filters are built with.

    These are far from the noise `generate_session` actually injects, on purpose: if the baseline
    were already optimal, a training run that changed nothing would look like a success. Returned
    from one place so the dry run and the artifact's recorded baseline cannot drift apart.
    """
    return 1.0e-4, 1.0e-3, 1.0e-6
