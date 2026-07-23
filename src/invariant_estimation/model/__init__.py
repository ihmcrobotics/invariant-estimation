"""`model` -- the rigid-body seam.

`MjxModel` is the concrete `robot.RobotModel`: MuJoCo/MJX supplies forward
kinematics, site angular Jacobians, and the composite-rigid-body inertia `M(q)`,
so no estimator module ever computes robot dynamics itself.

Kept as a thin re-export: importing this package pulls in `mujoco`, which the
pure-filter modules deliberately do not depend on.
"""
from .mjx_model import MassMatrixBlocks, MjxModel

__all__ = ["MjxModel", "MassMatrixBlocks"]
