"""The rigid-body seam: `MjxModel` is the concrete `robot.RobotModel`.

A thin re-export -- importing this package pulls in `mujoco`, which the
pure-filter modules deliberately do not depend on.
"""
from .mjx_model import MassMatrixBlocks, MjxModel

__all__ = ["MjxModel", "MassMatrixBlocks"]
