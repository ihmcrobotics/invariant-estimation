"""
joint_kf/state.py
=================
State and parameter types for the linear joint-chain Kalman filter.
 
The filter operates on the stacked joint state
 
    x = [q ; q_dot]  ∈ R^{2n}
 
where n is the number of 1-DoF joints.  The covariance P ∈ R^{2n × 2n}
tracks uncertainty over x in the usual KF sense.
 
The output of one filter step — (q_hat, q_dot_hat, Sigma_q) — feeds
directly into the InEKF measurement model:
 
    N = J_C(q) @ Sigma_q @ J_C(q).T          (FK noise propagation)
 
where J_C is the contact-point Jacobian and Sigma_q is the marginal
position covariance extracted from P.
 
Design notes
------------
* JointKFState is a NamedTuple so it is a valid JAX pytree with no
  extra registration.  This lets jax.lax.scan carry it as loop state
  without any static-field issues.
 
* n_joints is NOT stored in the state.  Array shapes encode it
  implicitly; storing it would make the struct non-pytree-safe with
  jit unless marked static everywhere.
 
* JointKFParams holds the scalar/matrix constants that are fixed for the
  lifetime of a filter run.  These live here for now; noise.py will
  reference this type when building Q and R.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

class JointKFState(NamedTuple):
    """Sufficient statistic for the linear joint-chain KF. 

    Attributes
    ----------
    q_hat : Array, shape (n,)
        Filtered joint position estimate [rad].
    q_dot_hat : Array, shape (n,)
        Filtered joint velocity estimate [rad/s].
    P : Array, shape (2n, 2n)
        Full joint error covariance.
        Block structure:
            P = [[P_qq,     P_q_qdot ],
                 [P_qdot_q, P_qdot   ]]
        where P_qq = Sigma_q is the marginal used downstream in N.
    """
    q_hat: Array      # (n,)

    q_dot_hat: Array  # (n,)
    P: Array          # (2n, 2n)

    @property
    def n_joints(self) -> int:
        """Number of joints, inferred from the `q_hat` shape."""
        return self.q_hat.shape[0]

    @property
    def sigma_q(self) -> Array:
        r"""Marginal position covariance $\Sigma_q$, shape (n, n).

        This is the top-left block of P and is what the InEKF measurement
        model needs for FK noise propoagation:

            N = J_C @ sigma_q @ J_C.T
        """
        n = self.n_joints
        return self.P[:n, :n]

    @property
    def sigma_q_dot(self) -> Array:
        """Marginal velocity covariance, shape (n, n)."""
        n = self.n_joints
        return self.P[n:, n:]

    
    @property
    def x(self) -> Array:
        """Stacked state vector [q; q_dot], shape (2n, )."""
        return jnp.concatenate([self.q_hat, self.q_dot_hat])


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class JointKFParams(NamedTuple):
    """Fixed parameters for the joint KF - constant across a filter run.

    These are passed into `predict()` and `update()` rather than stored in 
    the mutable state. Keeping them separate makes it straightforward
    to JIT-compile the flter step with params as a static argument or 
    a traced pytree depending on the use case.

    Attributes
    ----------
    sigma_enc : float
        Encoder position measurement noise std dev [rad].
        Diagonal entry of R  = sigma_enc^2 * I_n.

    sigma_vel : float
        Velocity measurement noise std dev [rad/s], used when a
        velocity signal (e.g. from different encoders or tachometers)
        is available as a direct measurement. Set to `jnp.inf` to
        disable velocity measurement updates.

    sigma_acc : float
        Joint acceleration process noise std dev [rad/s^2].
        Drives the random walk on q_dot in the process model.
        The process noise covariance Q is built from this in `noise.py`;
        when the tree-structured version is active, `noise.py` overrides
        this with the full M(q)-weighted block structure.

    dt : float
        Filter timestep [s]. 
        Used in `predict.py` for F and Q_d.
    """

    sigma_enc: float    # [rad]
    sigma_vel: float    # [rad/s]  — set to jnp.inf to disable
    sigma_acc: float    # [rad/s^2]
    dt: float           # [s]

# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_state(n_joints: int, q0: Array | None = None) -> JointKFState:
    """Construct a zeroed-out initial JointKFState.

    Parameters
    ----------
    n_joints : int
        Number of 1-DoF joints.
    q0 : Array of shape (n,), optional
        Initial joint position seed (e.g. from the first encoder reading).
        Defaults to zeros.

    Returns
    -------
    JointKFState
        q_hat  = q0 (or zeros)
        q_dot_hat = zeros
        P      = large diagonal (diffuse prior)
    """
    q_hat = q0 if q0 is not None else jnp.zeros(n_joints)
    q_dot_hat = jnp.zeros(n_joints)

    # Diffuse prior: large variance on position, very large on velocity.
    # These will shrink quickly once encoder measurements arrive.
    p_q = 1.0      # [rad^2]   — roughly ±1 rad uncertainty at init
    p_qd = 10.0    # [rad/s]^2 — roughly ±3 rad/s uncertainty at init
    P = jnp.diag(
        jnp.concatenate([
            jnp.full(n_joints, p_q),
            jnp.full(n_joints, p_qd),
        ])
    )

    return JointKFState(q_hat=q_hat, q_dot_hat=q_dot_hat, P=P)


def default_params(dt: float = 1e-3) -> JointKFParams:
    """Sensible default parameters for a 1 kHz humanoid joint KF.

    These are starting-point values, not tuned constants.  Adjust
    sigma_enc and sigma_acc to match your encoder spec and the dynamics
    roughness you expect.

    Parameters
    ----------
    dt : float
        Filter timestep [s].  Default matches IHMC 1 kHz control loop.
    """
    return JointKFParams(
        sigma_enc=jnp.deg2rad(0.05).item(),   # 0.05 deg encoder noise
        sigma_vel=jnp.inf,             # no direct velocity measurement
        sigma_acc=5.0,                 # [rad/s^2] — fairly permissive
        dt=dt,
    )
