"""
robot.py
========
Shared robot-data interface for the estimation pipeline — the **external seam**
where rigid-body dynamics/kinematics are injected.

The estimators (joint KF now; InEKF / ContactNet later) never compute robot
dynamics themselves.  They depend only on the `RobotModel` Protocol below, and a
concrete adapter supplies the actual numbers.  The intended adapter queries
**IsaacLab** articulation APIs (mass matrix, Jacobians) — that adapter is TODO and
is deliberately NOT imported here, so this module stays importable without any
simulator dependency.

Featherstone / Traversaro notation (left superscript = expressed-in frame, index
pair = reference/target) matches `jointKF/CLAUDE.md` §9.

Design references: `jointKF/CLAUDE.md` §2 (mass matrix in process noise), §3b/§7
(relative FK Jacobian + selection matrix for the IMU measurement block).
"""
from typing import Protocol, runtime_checkable

from jax import Array
##NOTE: this should be calling IsaacSim/Mujoco primitives for this Python version,
        ## while for the main Java estimator this would call Euclid via SCS2.

@runtime_checkable
class RobotModel(Protocol):
    """Contract for the external robot-data provider (the IsaacLab seam).

    A concrete implementation evaluates rigid-body quantities at a joint
    configuration `q`.  Implementations must return JAX arrays so callers stay
    jit/vmap-friendly.

    Required
    --------
    mass_matrix(q) -> (n, n)
        Composite-rigid-body inertia `M(q)`, symmetric positive-definite.
        Consumed by `jointKF.noise.acceleration_cov_mass` to shape the process
        noise `Q_a = σ_τ² M(q)⁻²` (CLAUDE.md §2).

    relative_gyro_jacobian(q) -> (m, 3, n)
        Stacked angular relative-FK Jacobians for the IMU pairs, already in full
        joint space (selection `S_ab` applied).  Consumed by
        `jointKF.measurement.build_H` (CLAUDE.md §3b).
    """

    def mass_matrix(self, q: Array) -> Array:
        """Composite-rigid-body inertia `M(q)`, shape (n, n), symmetric PD."""
        raise NotImplementedError

    def relative_gyro_jacobian(self, q: Array) -> Array:
        r"""Stacked angular relative-FK Jacobians, shape (m, 3, n).

        Row block `k` is `J^k_q̇(q) = J^{b_k,a_k}_{b_k}(q) · S_{ab,k}`, expressed
        in full joint space (the path selection `S_ab` is already applied, so the
        non-path joints contribute zero columns).  Maps `q̇` to each IMU pair's
        relative angular velocity (CLAUDE.md §3b).  `m` = number of IMU pairs.
        """

        raise NotImplementedError
