"""
joint_kf/measurement.py
=======================
Stacked measurement model for the joint-chain KF (CLAUDE.md §3) — the centerpiece
that fuses encoder positions with distributed-IMU relative angular velocity:

    z = [ q̃        ]        H = [ I_n   0          0    ]
        [ z_{ω,1}  ]            [ 0     J¹_q̇(q̂)    I_3  ]
        [ ⋮        ]            [       ⋮                ]
        [ z_{ω,m}  ]            [ 0     Jᵐ_q̇(q̂)    I_3  ]

with state `x = [q ; q̇ ; b_ω] ∈ R^{2n+3m}` (state.py) and measurement dim `n+3m`.

This module holds the **only EKF nonlinearity** in the filter: `H` depends on `q̂`
through the relative-FK Jacobian `J^k_q̇(q̂)`.  The position dynamics stay exactly
linear; keep the nonlinearity contained here.

The predicted measurement is `H x = [q ; J q̇ + b_ω]`.  The innovation
`ν = z − H x⁻` is formed in `update.py`, not here.

Conventions
-----------
* Featherstone / Traversaro notation (CLAUDE.md §9): left superscript = expressed-in
  frame, index pair = reference/target.  `R^b_a` rotates a quantity from IMU-a's
  frame into IMU-b's frame.
* Pure JAX, vectorized (`vmap`/broadcasting), jit-safe.  `m` is inferred from array
  leading dims; `n_joints` is passed explicitly.  The relative Jacobian stack
  `J_omega` (m, 3, n) and the raw IMU readings are injected as plain arrays — the
  caller (`filter.py`) sources `J_omega` from a `robot.RobotModel`; this module
  never imports a simulator.  `R = blkdiag(R_enc, R_ω)` is built by `noise.build_R`.
"""
import jax
import jax.numpy as jnp
from jax import Array


def relative_gyro_measurement(
    omega_a: Array,
    omega_b: Array,
    R_ba: Array,
) -> Array:
    r"""Differenced relative-gyro measurement, shape (m, 3).

    For each IMU pair, rotate IMU-a's gravity-corrected gyro reading into IMU-b's
    frame and subtract, cancelling the common base motion (CLAUDE.md §3b):

        z_{ω,k} = ω_b^{b,w} − R^b_a · ω_a^{a,w}

    Parameters
    ----------
    omega_a, omega_b : Array, shape (m, 3)
        Per-pair Mahony-cleaned gyro readings, each in its own IMU frame.
    R_ba : Array, shape (m, 3, 3)
        Per-pair inter-IMU rotation `R^b_a` (a-frame → b-frame), from Mahony.

    Returns
    -------
    Array, shape (m, 3)
        The relative angular-velocity measurement per pair.

    Notes
    -----
    Frame-sensitive: `R_ba` must rotate a's reading into b's frame.  A wrong
    convention silently biases the fused velocity.
    """
    rotated_a = jax.vmap(lambda R, w: R @ w)(R_ba, omega_a)   # (m, 3)
    return omega_b - rotated_a


def build_z(q_tilde: Array, z_omega: Array) -> Array:
    """Stack the measurement vector `z = [q̃ ; z_ω]`, shape (n + 3m,).

    Parameters
    ----------
    q_tilde : Array, shape (n,)
        Encoder position reading.
    z_omega : Array, shape (m, 3)
        Per-pair differenced gyro measurement (`relative_gyro_measurement`).

    Returns
    -------
    Array, shape (n + 3m,)
        Encoder block first, then the per-pair gyro blocks flattened in pair
        order — matching the row order of `build_H` and the `b_ω` state layout.
    """
    return jnp.concatenate([q_tilde, z_omega.reshape(-1)])


def build_H(J_omega: Array, n_joints: int) -> Array:
    r"""Measurement Jacobian `H`, shape (n + 3m, 2n + 3m).

        H = [ I_n   0          0    ]      (encoder rows)
            [ 0     J_stack    I_3m ]      (IMU rows)

    where `J_stack = J_omega.reshape(3m, n)` are the stacked angular relative-FK
    Jacobians.  The `I_3m` bias block maps pair `k`'s 3 measurement components to
    pair `k`'s 3 residual-bias components — the only place `b_ω` enters `H`.

    Parameters
    ----------
    J_omega : Array, shape (m, 3, n)
        Stacked per-pair Jacobians `J^k_q̇(q̂)` in full joint space
        (`robot.RobotModel.relative_gyro_jacobian`).
    n_joints : int
        Number of joints n.

    Returns
    -------
    Array, shape (n + 3m, 2n + 3m)
    """
    m = J_omega.shape[0]
    n = n_joints
    rows = n + 3 * m
    cols = 2 * n + 3 * m

    H = jnp.zeros((rows, cols)).at[:n, :n].set(jnp.eye(n))   # encoder block
    if m > 0:                                                 # static: m is a Python int
        H = H.at[n:, n:2 * n].set(J_omega.reshape(3 * m, n))  # q̇ columns
        H = H.at[n:, 2 * n:].set(jnp.eye(3 * m))              # bias columns
    return H


def build_measurement(
    q_tilde: Array,
    omega_a: Array,
    omega_b: Array,
    R_ba: Array,
    J_omega: Array,
    n_joints: int,
) -> tuple[Array, Array]:
    """Build `(z, H)` for one update — thin orchestrator for `filter.py`.

    Differences the gyros (`relative_gyro_measurement`), stacks `z` (`build_z`),
    and assembles `H` (`build_H`).  The caller supplies the raw IMU arrays and
    `J_omega = robot.relative_gyro_jacobian(q̂)`.

    Returns
    -------
    (z, H) : tuple of Array, shapes (n + 3m,) and (n + 3m, 2n + 3m)
    """
    z_omega = relative_gyro_measurement(omega_a, omega_b, R_ba)
    z = build_z(q_tilde, z_omega)
    H = build_H(J_omega, n_joints)
    return z, H
