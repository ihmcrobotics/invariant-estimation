"""
inEKF — world-centric, right-invariant contact-aided InEKF on SE_{N+2}(3).

Fuses IMU + forward-kinematics into a pose/velocity/contact estimate; consumes
the joint-KF pre-filter outputs and external ContactNet covariances on the
correction side only (see `CLAUDE.md` for the full design record).  The filter
holds no trainable parameters — BPTT during training flows *through* it.

Build order (one module per prompt): group -> state -> propagate -> correct ->
contact -> filter.  Implemented so far: group, state, propagate, correct.
"""
from .correct import (
    apply_correction,
    correct,
    innovation,
    joseph_update,
    kalman_gain,
    measurement_noise,
    predicted_contact,
)
from .group import (
    Adjoint,
    Gamma0,
    Gamma1,
    Gamma2,
    exp_SEn3,
    log_SEn3,
    skew,
)
from .propagate import (
    build_Qd,
    inertial_Qd,
    propagate,
    propagate_cov,
    propagate_mean,
)
from .state import (
    InEKFParams,
    InEKFState,
    build_H,
    build_Phi,
    default_params,
    init_state,
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
    # state + params
    "InEKFState",
    "InEKFParams",
    "init_state",
    "default_params",
    "build_Phi",
    "build_H",
    # propagation (§3)
    "propagate",
    "propagate_mean",
    "propagate_cov",
    "build_Qd",
    "inertial_Qd",
    # correction (§4)
    "correct",
    "innovation",
    "predicted_contact",
    "measurement_noise",
    "kalman_gain",
    "joseph_update",
    "apply_correction",
]
