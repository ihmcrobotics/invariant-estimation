"""The **external seam** where rigid-body dynamics/kinematics enter the pipeline.

No estimator computes robot dynamics itself; each depends only on the
`RobotModel` Protocol below.  `model.MjxModel` is the shipped implementation
(MuJoCo/MJX), deliberately NOT imported here so this module stays importable
without a simulator dependency.

Featherstone / Traversaro notation (left superscript = expressed-in frame, index
pair = reference/target) matches `jointKF/CLAUDE.md` §9; §2 covers the mass
matrix in the process noise, §3b the relative FK Jacobian + selection matrix.
"""
from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class RobotModel(Protocol):
    """Rigid-body quantities at a joint configuration `q`, as JAX arrays so that
    callers stay jit/vmap-friendly."""

    def mass_matrix(self, q: Array) -> Array:
        """Composite-rigid-body inertia `M(q)`, shape (n, n), symmetric PD.

        `jointKF.process` Schur-complements it against the nuisance DoFs and
        shapes the process noise as `Qa = Λ_eff⁻¹ Σ_τ Λ_eff⁻ᵀ` (CLAUDE.md I1/§2).
        """
        raise NotImplementedError

    def relative_gyro_jacobian(self, q: Array) -> Array:
        r"""Stacked angular relative-FK Jacobians, shape (m, 3, n), `m` = IMU pairs.

        Row block `k` is `J^k_q̇(q) = J^{b_k,a_k}_{b_k}(q) · S_{ab,k}`, expressed
        in full joint space (the path selection `S_ab` is already applied, so the
        non-path joints contribute zero columns).  Maps `q̇` to each IMU pair's
        relative angular velocity; consumed by `jointKF.measure`.
        """

        raise NotImplementedError
