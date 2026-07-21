"""
joint_kf/state.py
=================
State and parameter types for the linear joint-chain Kalman filter.

The filter operates on the bias-augmented joint state

    x = [q ; q_dot ; b_omega]  ∈ R^{2n + 3m}

where

    n  = number of 1-DoF joints,
    m  = number of IMU pairs being fused,
    b_omega = residual *relative* gyro bias, one 3-vector per IMU pair.

`b_omega` is the *fine residual* bias left over after the per-IMU Mahony
filter (the upstream coarse absolute-bias attenuator).  Because Mahony has
already removed the bulk of the bias, `b_omega` is modeled with a TIGHT
random-walk noise and a TIGHT initial covariance — otherwise the two
estimators fight over the same error and produce a slow oscillation.

The covariance P ∈ R^{(2n+3m) × (2n+3m)} tracks uncertainty over x.  Its
marginal blocks feed the downstream InEKF measurement model and ContactNet:

    Sigma_q   := P[0:n,     0:n  ]   → InEKF position FK noise  N = J_C Σ_q J_C.T
    Sigma_qd  := P[n:2n,   n:2n ]   → kinematic part of contact-velocity noise
    Sigma_b   := P[2n:,    2n:  ]   → residual-bias marginal (ContactNet feature)

Design notes
------------
* JointKFState is a NamedTuple so it is a valid JAX pytree with no extra
  registration.  This lets jax.lax.scan carry it as loop state without any
  static-field issues.

* n_joints and n_pairs are NOT stored in the state.  Array shapes encode
  them implicitly; storing them would make the struct non-pytree-safe with
  jit unless marked static everywhere.

* b_omega is stored FLAT, shape (3m,), so that the stacked state vector `x`
  is a plain concatenation and the P block layout is contiguous.  Use the
  `b_omega_pairs` property for a per-pair (m, 3) view.

* JointKFParams holds the scalar constants that are fixed for the lifetime of
  a filter run.  noise.py / update.py reference this type when building Q and
  R.  IMU pairing topology and selection matrices S_ab are NOT here — they are
  deferred to measurement.py; JointKFParams carries only scalars for now.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ..config import section


class JointKFState(NamedTuple):
    """Sufficient statistic for the bias-augmented joint-chain KF.

    Attributes
    ----------
    q_hat : Array, shape (n,)
        Filtered joint position estimate [rad].
    q_dot_hat : Array, shape (n,)
        Filtered joint velocity estimate [rad/s].
    b_omega : Array, shape (3m,)
        Residual relative gyro bias, one 3-vector per IMU pair, stored flat.
    P : Array, shape (2n+3m, 2n+3m)
        Full joint error covariance.
        Block structure (n joints, m IMU pairs):
            P = [[ P_qq    P_q_qd   P_q_b  ],
                 [ P_qd_q  P_qdqd   P_qd_b ],
                 [ P_b_q   P_b_qd   P_bb   ]]
        where P_qq = Sigma_q is the marginal used downstream in N = J_C Σ_q J_C.T.
    """
    q_hat: Array      # (n,)
    q_dot_hat: Array  # (n,)
    b_omega: Array    # (3m,)
    P: Array          # (2n+3m, 2n+3m)

    @property
    def n_joints(self) -> int:
        """Number of joints, inferred from the `q_hat` shape."""
        return self.q_hat.shape[0]

    @property
    def n_pairs(self) -> int:
        """Number of IMU pairs m, inferred from the `b_omega` shape (3m,)."""
        return self.b_omega.shape[0] // 3

    @property
    def sigma_q(self) -> Array:
        r"""Marginal position covariance $\Sigma_q$, shape (n, n).

        Top-left block of P; what the InEKF measurement model needs for FK
        noise propagation:

            N = J_C @ sigma_q @ J_C.T
        """
        n = self.n_joints
        return self.P[:n, :n]

    @property
    def sigma_q_dot(self) -> Array:
        """Marginal velocity covariance, shape (n, n).

        Middle block P[n:2n, n:2n].  Note the explicit 2n upper bound: P now
        carries the bias block after the velocity block, so an open-ended
        slice would incorrectly fold the bias rows/cols into this marginal.
        """
        n = self.n_joints
        return self.P[n:2 * n, n:2 * n]

    @property
    def sigma_b(self) -> Array:
        """Marginal residual-bias covariance, shape (3m, 3m).

        Bottom-right block P[2n:, 2n:].  Its diagonal is a ContactNet trust
        feature; the whole block stays tight by construction (see module docstring).
        """
        n = self.n_joints
        return self.P[2 * n:, 2 * n:]

    @property
    def b_omega_pairs(self) -> Array:
        """Per-pair view of the residual bias, shape (m, 3)."""
        return self.b_omega.reshape(self.n_pairs, 3)

    @property
    def x(self) -> Array:
        """Stacked state vector [q ; q_dot ; b_omega], shape (..., 2n+3m).

        Concatenates on the last axis, so this also works on a batched/stacked
        state (e.g. a `jax.lax.scan` trajectory with a leading time axis), not
        only a single 1-D state.
        """
        return jnp.concatenate([self.q_hat, self.q_dot_hat, self.b_omega], axis=-1)


def split_x(x: Array, n_joints: int) -> tuple[Array, Array, Array]:
    """Inverse of `JointKFState.x`: split [q ; q_dot ; b_omega] into its parts.

    Reconstructs the three state segments from a stacked vector, so the filter
    steps (predict / update) that operate on `x` can rebuild a JointKFState
    without duplicating the slice arithmetic.  `b_omega` is the remainder, so
    `n_pairs` is not needed.

    Parameters
    ----------
    x : Array, shape (2n+3m,)
        Stacked state vector.
    n_joints : int
        Number of joints n.

    Returns
    -------
    (q_hat, q_dot_hat, b_omega) : tuple of Array, shapes (n,), (n,), (3m,)
    """
    n = n_joints
    return x[:n], x[n:2 * n], x[2 * n:]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class JointKFParams(NamedTuple):
    """Fixed parameters for the joint KF — constant across a filter run.

    These are passed into `predict()` and `update()` rather than stored in the
    mutable state.  Keeping them separate makes it straightforward to
    JIT-compile the filter step with params as a static argument or a traced
    pytree depending on the use case.

    Attributes
    ----------
    sigma_enc : float
        Encoder position measurement noise std dev [rad].
        Builds the encoder block of R:  R_enc = sigma_enc^2 * I_n.

    sigma_omega : float
        Relative-gyro measurement noise std dev [rad/s] for the IMU block of R
        (R_omega).  This is the *Mahony-cleaned* relative-gyro covariance —
        smaller than a raw gyro because the per-IMU pre-filter already
        attenuated bias and noise.  One scalar shared across pairs for now.

    sigma_tau : float
        Torque-space process-noise std dev [N·m].  The physically-correct
        acceleration process noise is the mass-matrix sandwich
        Q_a = sigma_tau^2 * M(q)^{-2}, built in noise.py.  This is the eventual
        replacement for the `sigma_acc` diagonal stand-in once M(q) shaping is
        switched on.

    sigma_acc : float
        Joint acceleration process noise std dev [rad/s^2], used as the
        EARLY-DEV diagonal substitute for Q_a (Q_a = sigma_acc^2 * I_n).  Lets
        the InEKF / ContactNet be validated before mass-matrix coupling is
        wired up.  Ignored once noise.py uses the M(q)-weighted Q_a.

    sigma_b : float
        Residual relative-bias random-walk std dev [rad/s].  Builds the bias
        block of Q_d:  Q_d^{bb} = sigma_b^2 * dt * I_3m.  Keep TIGHT — b_omega
        is only the leftover after Mahony, so a loose value lets the two
        estimators fight over the same error.

    dt : float
        Filter timestep [s].  Used in predict.py for F and Q_d.
    """

    sigma_enc: float    # [rad]
    sigma_omega: float  # [rad/s]   — Mahony-cleaned relative-gyro std (R_omega)
    sigma_tau: float    # [N·m]     — torque-space process noise (Q_a = σ_τ² M⁻²)
    sigma_acc: float    # [rad/s^2] — early-dev diagonal Q_a substitute
    sigma_b: float      # [rad/s]   — residual-bias random walk (keep TIGHT)
    dt: float           # [s]


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_state(
    n_joints: int,
    n_pairs: int,
    q0: Array | None = None,
) -> JointKFState:
    """Construct a zeroed-out initial JointKFState.

    Parameters
    ----------
    n_joints : int
        Number of 1-DoF joints (n).
    n_pairs : int
        Number of IMU pairs being fused (m).  May be 0 (encoder-only fallback,
        no bias block).
    q0 : Array of shape (n,), optional
        Initial joint position seed (e.g. from the first encoder reading).
        Defaults to zeros.

    Returns
    -------
    JointKFState
        q_hat     = q0 (or zeros)
        q_dot_hat = zeros
        b_omega   = zeros, shape (3m,)
        P         = diagonal diffuse prior on [q, q_dot], TIGHT on b_omega
    """
    q_hat = q0 if q0 is not None else jnp.zeros(n_joints)
    q_dot_hat = jnp.zeros(n_joints)
    b_omega = jnp.zeros(3 * n_pairs)

    # Diffuse prior on the joint state: large variance on position, very large
    # on velocity — both shrink quickly once encoder / IMU measurements arrive.
    p_q = 1.0      # [rad^2]   — roughly ±1 rad uncertainty at init
    p_qd = 10.0    # [rad/s]^2 — roughly ±3 rad/s uncertainty at init
    # TIGHT prior on the residual bias: Mahony has already removed the bulk, so
    # b_omega starts near zero with little uncertainty (see module docstring).
    p_b = 1e-6     # [rad/s]^2 — ±1e-3 rad/s, deliberately tight

    P = jnp.diag(
        jnp.concatenate([
            jnp.full(n_joints, p_q),
            jnp.full(n_joints, p_qd),
            jnp.full(3 * n_pairs, p_b),
        ])
    )

    return JointKFState(q_hat=q_hat, q_dot_hat=q_dot_hat, b_omega=b_omega, P=P)


def default_params(dt: float | None = None) -> JointKFParams:
    """Default parameters for a 1 kHz humanoid joint KF.

    All values come from the ``joint_kf`` section of ``config/filter_cfg.yaml``
    — the single place tuning numbers live.  These are starting-point values,
    NOT tuned constants: match them to the encoder spec, the Mahony-cleaned gyro
    noise, and the dynamics roughness you expect.

    Parameters
    ----------
    dt : float, optional
        Filter timestep [s].  ``None`` takes the configured value (1 kHz).
    """
    cfg = section("joint_kf")
    return JointKFParams(
        sigma_enc=jnp.deg2rad(cfg["sigma_enc_deg"]).item(),
        sigma_omega=cfg["sigma_omega"],         # [rad/s] Mahony-cleaned rel-gyro
        sigma_tau=cfg["sigma_tau"],             # [N·m] torque-space process noise
        sigma_acc=cfg["sigma_acc"],             # [rad/s^2] early-dev diagonal Q_a
        sigma_b=cfg["sigma_b"],                 # [rad/s] residual bias — TIGHT
        dt=cfg["dt"] if dt is None else dt,
    )
