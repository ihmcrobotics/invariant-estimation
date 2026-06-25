"""
joint_kf/noise.py
=================
Builders for the constant / semi-constant matrices of the joint-chain KF:

    F     — exact transition matrix            (CLAUDE.md §2)
    Q_d   — discrete process noise             (CLAUDE.md §2)
    R     — stacked measurement noise          (CLAUDE.md §3)

State layout (see state.py):  x = [q ; q̇ ; b_ω] ∈ R^{2n+3m},  D = 2n + 3m.

Mass-matrix injection
---------------------
The process noise `Q_a = σ_τ² M(q̂)⁻²` needs the composite-rigid-body inertia
`M(q̂)`.  This module does NOT compute `M` — it takes it as a plain (n, n) array
so the builders stay pure and unit-testable.  The caller (e.g. predict.py) sources
`M` from a `robot.RobotModel` (the IsaacLab seam), or passes `M=None` to use the
hand-tuned diagonal early-dev stand-in `Q_a = σ_acc² I_n`.

JAX conventions
---------------
Everything here is vectorized (`jnp.eye`, broadcasting) and jit-safe; integer dims
`n_joints` / `n_pairs` are static.  No Python loops over data/pair dimensions — the
per-pair bias and gyro blocks are isotropic `σ² I_3m`, pure broadcasting.
"""
import jax.numpy as jnp
from jax import Array

from .state import JointKFParams


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def build_F(n_joints: int, n_pairs: int, dt: float) -> Array:
    r"""Exact discrete transition matrix `F = exp(A·Δt)`, shape (D, D).

    `A` is nilpotent in the `[q;q̇]` block and zero elsewhere, so
    `F = I + A·Δt` is exact for all Δt (no truncation, CLAUDE.md §2):

        F = [ I_n   Δt·I_n   0    ]
            [ 0     I_n      0    ]
            [ 0     0        I_3m ]
    """
    n = n_joints
    dim = 2 * n + 3 * n_pairs
    return jnp.eye(dim).at[:n, n:2 * n].set(dt * jnp.eye(n))


# ---------------------------------------------------------------------------
# Acceleration process covariance  Q_a  (n, n)
# ---------------------------------------------------------------------------

def acceleration_cov_diag(n_joints: int, sigma_acc: float) -> Array:
    """Early-dev diagonal stand-in `Q_a = σ_acc² I_n`, shape (n, n).

    Lets the InEKF / ContactNet be validated before mass-matrix coupling is
    switched on (CLAUDE.md §2, "Early-development substitution").
    """
    return sigma_acc ** 2 * jnp.eye(n_joints)


def acceleration_cov_mass(M: Array, sigma_tau: float) -> Array:
    r"""Mass-matrix-shaped acceleration covariance, shape (n, n).

    A torque disturbance `w_τ ~ N(0, σ_τ² I)` maps to an acceleration
    disturbance through Newton's law `M(q) q̈ = τ + …`, giving (CLAUDE.md §2):

        Q_a = M⁻¹ Σ_τ M⁻ᵀ = σ_τ² · M⁻¹ M⁻ᵀ      (Σ_τ = σ_τ² I)

    Because `M` is symmetric PD, `M⁻ᵀ = M⁻¹`, so analytically `Q_a = σ_τ² M⁻²`.
    We form it as `σ_τ² (M⁻¹ M⁻ᵀ)` so the result is symmetric / PSD by
    construction.  `M⁻¹` is dense: this spreads diagonal torque uncertainty into
    **correlated** acceleration uncertainty across the kinematic tree — the entire
    reason for using `M` instead of a per-joint scalar.
    """
    Minv = jnp.linalg.inv(M)
    return sigma_tau ** 2 * (Minv @ Minv.T)


# ---------------------------------------------------------------------------
# Discrete process noise  Q_d  (D, D)
# ---------------------------------------------------------------------------

def build_Q_d(Q_a: Array, params: JointKFParams, n_pairs: int) -> Array:
    r"""Assemble discrete process noise `Q_d`, shape (D, D), from `Q_a` (n, n).

    Exact Van Loan closed form for the nilpotent `[q;q̇]` double integrator
    (CLAUDE.md §2):

        Q_d^{[q,q̇]} = [ (Δt³/3)·Q_a   (Δt²/2)·Q_a ]
                      [ (Δt²/2)·Q_a   Δt·Q_a      ]

    Bias block is a tight random walk `Q_d^{bb} = σ_b² · Δt · I_3m`, and
    `Q_d = blkdiag(Q_d^{[q,q̇]}, Q_d^{bb})`.
    """
    n = Q_a.shape[0]
    dt = params.dt
    dim = 2 * n + 3 * n_pairs

    qq = (dt ** 3 / 3.0) * Q_a
    qv = (dt ** 2 / 2.0) * Q_a
    vv = dt * Q_a
    top = jnp.block([[qq, qv], [qv, vv]])          # (2n, 2n)

    Q_d = jnp.zeros((dim, dim)).at[:2 * n, :2 * n].set(top)
    if n_pairs > 0:                                 # static: n_pairs is a Python int
        bb = params.sigma_b ** 2 * dt * jnp.eye(3 * n_pairs)
        Q_d = Q_d.at[2 * n:, 2 * n:].set(bb)
    return Q_d


def build_process_noise(
    params: JointKFParams,
    n_joints: int,
    n_pairs: int,
    M: Array | None = None,
) -> Array:
    """Single entry point for `Q_d` — selects the diagonal or mass-matrix `Q_a`.

    `M is None` → early-dev diagonal `Q_a = σ_acc² I_n`.
    `M` given   → `Q_a = σ_τ² M⁻²` (sourced from a `robot.RobotModel`).
    """
    if M is None:
        Q_a = acceleration_cov_diag(n_joints, params.sigma_acc)
    else:
        Q_a = acceleration_cov_mass(M, params.sigma_tau)
    return build_Q_d(Q_a, params, n_pairs)


# ---------------------------------------------------------------------------
# Measurement noise  R  (n + 3m, n + 3m)
# ---------------------------------------------------------------------------

def build_R(params: JointKFParams, n_joints: int, n_pairs: int) -> Array:
    r"""Stacked measurement noise `R = blkdiag(R_enc, R_ω)`, shape (n+3m, n+3m).

        R_enc = σ_enc²   I_n      (encoder block, CLAUDE.md §3a)
        R_ω   = σ_omega² I_3m     (Mahony-cleaned relative gyro, §3b)

    `R_ω` is small because the per-IMU Mahony pre-filter already attenuated bias
    and noise — the shrinking of `R_ω` is exactly what the Mahony layer buys.
    """
    n = n_joints
    dim = n + 3 * n_pairs
    R = jnp.zeros((dim, dim)).at[:n, :n].set(params.sigma_enc ** 2 * jnp.eye(n))
    if n_pairs > 0:                                 # static: n_pairs is a Python int
        R = R.at[n:, n:].set(params.sigma_omega ** 2 * jnp.eye(3 * n_pairs))
    return R
