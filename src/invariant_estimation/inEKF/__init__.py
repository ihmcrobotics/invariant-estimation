"""
inEKF — world-centric, right-invariant contact-aided InEKF on SE_{N+2}(3).

Fuses IMU + forward-kinematics into a pose/velocity/contact estimate; consumes
the joint-KF pre-filter outputs and external ContactNet covariances on the
correction side only (see `CLAUDE.md` for the full design record).  The filter
holds no trainable parameters — BPTT during training flows *through* it.

Build order (one module per prompt): group -> state -> propagate -> correct ->
contact -> filter.  Only `group` is implemented so far.
"""
from .group import (
    Adjoint,
    Gamma0,
    Gamma1,
    Gamma2,
    exp_SEn3,
    log_SEn3,
    skew,
)

__all__ = [
    # Lie-group ops (SE_{N+2}(3))
    "skew",
    "Gamma0",
    "Gamma1",
    "Gamma2",
    "exp_SEn3",
    "log_SEn3",
    "Adjoint",
]
