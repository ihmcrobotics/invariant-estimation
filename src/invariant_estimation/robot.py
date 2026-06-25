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


@runtime_checkable
class RobotModel(Protocol):
    """Contract for the external robot-data provider (the IsaacLab seam).

    A concrete implementation evaluates rigid-body quantities at a joint
    configuration `q`.  Implementations must return JAX arrays so callers stay
    jit/vmap-friendly.

    Currently required
    ------------------
    mass_matrix(q) -> (n, n)
        Composite-rigid-body inertia `M(q)`, symmetric positive-definite.
        Consumed by `jointKF.noise.acceleration_cov_mass` to shape the process
        noise `Q_a = σ_τ² M(q)⁻²` (CLAUDE.md §2).

    Planned (added when `jointKF/measurement.py` lands)
    ---------------------------------------------------
    relative_jacobian(q, pair) -> angular relative-FK Jacobian `J^{b,a}_b(q)` for
        an IMU pair, and the path selection matrix `S_ab` (CLAUDE.md §3b).  These
        will extend this Protocol; keep additions here so `noise.py` and
        `measurement.py` share one seam.
    """

    def mass_matrix(self, q: Array) -> Array:
        """Composite-rigid-body inertia `M(q)`, shape (n, n), symmetric PD."""
        ...
