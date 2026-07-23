"""
inEKF — world-centric, right-invariant contact-aided InEKF on SE_{N+2}(3).

Fuses IMU + forward-kinematics into a pose/velocity/contact estimate; consumes
the joint-KF pre-filter outputs and external ContactNet covariances on the
correction side only (see `CLAUDE.md` for the full design record).  The filter
holds no trainable parameters — BPTT during training flows *through* it.

Build order: group -> state -> propagate -> correct -> contact -> gravity_update
-> ekf -> filter.  All implemented; `filter.step`/`filter.run` are the scan body
and trajectory driver.  Deferred by decision: reseed, contact trust (see
PORT_NOTES.md).
"""
from .contact import (
    apply_floor,
    digest,
    reconstruct_cov,
    rotate_to_world,
)
from .correct import (
    UpdateDiagnostics,
    apply_correction,
    contact_jacobian,
    contact_residual,
    contact_update,
    correct,
    innovation,
    joseph_update,
    kalman_gain,
    linear_update,
    map_encoder_noise,
    measurement_noise,
    no_update_diagnostics,
    predicted_contact,
    rotate_measurement_covariance,
)
from .filter import (
    ContactFrames,
    ContactKinematics,
    InEKFCarry,
    InEKFInputs,
    InEKFOutputs,
    JointFilterOutput,
    contact_position_noise,
    contact_velocity_noise,
    init_carry,
    make_step,
    run,
)
from .ekf import (
    InvariantEKF,
    create,
    gravity_leveling_update,
    initial_diagnostics,
    initialize,
    initialize_from_state,
    predict,
    update,
)
from .gravity_update import (
    GravityMeasurement,
    GravityParams,
    GravityRef,
    apply_gravity_leveling,
    assemble_gravity_leveling,
    default_gravity_params,
    gravity_jacobian,
    gravity_measurement_covariance,
    init_gravity_ref,
    is_quasi_static,
    isotropic_gravity_params,
    tilt_angle,
    update_gravity_reference,
)
from .group import (
    Adjoint,
    Gamma0,
    Gamma1,
    Gamma2,
    exp_SEk3,
    exp_SEn3,
    log_SEn3,
    skew,
)
from .propagate import (
    build_Qd,
    continuous_Qc,
    propagate,
    propagate_cov,
    propagate_mean,
)
from .state import (
    BASE_POSITION_TANGENT_INDEX,
    BASE_VELOCITY_TANGENT_INDEX,
    ROTATION_TANGENT_INDEX,
    InEKFParams,
    InEKFState,
    build_H,
    build_Phi,
    contact_tangent_index,
    default_params,
    init_state,
)

__all__ = [
    # Lie-group ops (SE_{N+2}(3))
    "skew",
    "Gamma0",
    "Gamma1",
    "Gamma2",
    "exp_SEk3",
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
    # tangent layout (I4)
    "ROTATION_TANGENT_INDEX",
    "BASE_VELOCITY_TANGENT_INDEX",
    "BASE_POSITION_TANGENT_INDEX",
    "contact_tangent_index",
    # propagation (§3)
    "propagate",
    "propagate_mean",
    "propagate_cov",
    "build_Qd",
    "continuous_Qc",
    # correction (§4)
    "correct",
    "innovation",
    "predicted_contact",
    "measurement_noise",
    "kalman_gain",
    "joseph_update",
    "apply_correction",
    "linear_update",
    "UpdateDiagnostics",
    "no_update_diagnostics",
    # ContactUpdater seams (ported suite)
    "contact_jacobian",
    "contact_residual",
    "contact_update",
    "rotate_measurement_covariance",
    "map_encoder_noise",
    # scan body + trajectory driver
    "JointFilterOutput",
    "ContactFrames",
    "ContactKinematics",
    "InEKFInputs",
    "InEKFCarry",
    "InEKFOutputs",
    "make_step",
    "run",
    "init_carry",
    "contact_position_noise",
    "contact_velocity_noise",
    # EKF orchestrator (G5)
    "InvariantEKF",
    "create",
    "initialize",
    "initialize_from_state",
    "predict",
    "update",
    "gravity_leveling_update",
    "initial_diagnostics",
    # gravity leveling (G4)
    "GravityParams",
    "GravityRef",
    "GravityMeasurement",
    "default_gravity_params",
    "isotropic_gravity_params",
    "init_gravity_ref",
    "update_gravity_reference",
    "is_quasi_static",
    "gravity_jacobian",
    "gravity_measurement_covariance",
    "assemble_gravity_leveling",
    "apply_gravity_leveling",
    "tilt_angle",
    # contact-covariance digest (§5)
    "digest",
    "reconstruct_cov",
    "apply_floor",
    "rotate_to_world",
]
