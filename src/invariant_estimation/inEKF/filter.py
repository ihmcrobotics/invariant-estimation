r"""
inEKF/filter.py
===============
The scan body — ``step`` — and the trajectory driver ``run``.

Everything below this file is a pure function over an explicit ``(X, P)`` carry
(I10); this is the one place they are composed into a tick.  The tick is
deliberately boring::

    propagate  (IMU)                    §3
      -> contact FK update  (joint KF + ContactNet)   §4
      -> gravity leveling   (accelerometer, gated)    G4

Constant-graph contract (I7)
----------------------------
``step`` must trace to the **same jaxpr** regardless of which contacts are
trusted, whether the quasi-static gate is open, or whether pitch is observable.
Nothing here branches on data:

* an untrusted contact keeps its measurement rows but gets ``R_LARGE`` noise and
  a zeroed residual — never a dropped row (a zeroed ``R`` row makes ``S``
  singular; §6 trap);
* the gravity gate multiplies into ``K`` as a float mask, so a closed gate leaves
  ``(X̂, P)`` bit-for-bit unchanged rather than skipping work;
* ``N``, the joint count, and the contact count are static, fixed at build time.

`tests/inEKF/test_filter.py` asserts the jaxpr hash is identical across
differing contact masks and gate states — the I7 proof, which G9 reuses.

The joint-filter boundary
-------------------------
`JointFilterOutput` is the seam the joint KF feeds through: ``(q̂, q̇̂, Σ_q, Σ_q̇)``.
Per the boundary contract those enter **only** on the correction side, always
pre-multiplied by a kinematic Jacobian::

    N^p_i = J_{C_i}(q̂) Σ_q J_{C_i}ᵀ         position FK noise  (used)
    N^v_i = J_{Ċ_i}(q̂) Σ_q̇ J_{Ċ_i}ᵀ        contact velocity noise (see TODO)

They never reach the propagation ``Φ`` or the inertial ``Q``.  The kinematics
themselves come from a caller-supplied `ContactKinematics` callable — the
``robot/`` seam — closed over at build time so it is static under ``jit``.  MJX
implements it at G1; a fixture can implement it today.
"""
from typing import NamedTuple, Protocol

import jax
from jax import Array
import jax.numpy as jnp

from .contact import digest
from .correct import (
    UpdateDiagnostics,
    innovation,
    linear_update,
    map_encoder_noise,
    measurement_noise,
)
from .ekf import InvariantEKF
from .gravity_update import (
    GravityRef,
    assemble_gravity_leveling,
    init_gravity_ref,
    is_quasi_static,
    update_gravity_reference,
)
from .propagate import propagate
from .state import InEKFState

# Noise substituted for an untrusted contact.  Large enough that the update is
# numerically a no-op on that block, small enough that S stays well conditioned
# in float64 (CLAUDE.md §4).
R_LARGE = 1.0e12


# ---------------------------------------------------------------------------
# Boundary types
# ---------------------------------------------------------------------------

class JointFilterOutput(NamedTuple):
    """What the joint KF hands the InEKF each tick (CLAUDE.md §1 deliverable 1).

    Attributes
    ----------
    q : Array, shape (n_joints,)
        Filtered joint positions ``q̂``.
    q_dot : Array, shape (n_joints,)
        Filtered joint velocities ``q̇̂``.
    sigma_q : Array, shape (n_joints, n_joints)
        Joint-position covariance ``Σ_q``.  Full matrix, not a diagonal: the
        joint KF's covariance is genuinely coupled through the mass matrix, and
        ``J Σ_q Jᵀ`` needs the off-diagonals.
    sigma_q_dot : Array, shape (n_joints, n_joints)
        Joint-velocity covariance ``Σ_q̇``.  Routed as its own term, never folded
        into ``Σ_q`` — see the TODO in `step`.

    The base gyro-bias estimate ``b̂`` is deliberately **not** here: the InEKF
    consumes already-bias-corrected IMU (I1), so the correction happens upstream.
    """
    q: Array
    q_dot: Array
    sigma_q: Array
    sigma_q_dot: Array


class ContactFrames(NamedTuple):
    """Kinematics evaluated at ``q̂`` — the ``robot/`` seam's output.

    Attributes
    ----------
    y : Array, shape (N, 3)
        Body-frame base→contact vectors ``h_{p,i}(q̂)``: the FK measurement.
    J : Array, shape (N, 3, n_joints)
        Contact-point position Jacobians ``J_{C_i}(q̂)``.
    J_dot : Array, shape (N, 3, n_joints)
        Their time derivatives ``J_{Ċ_i}``, for the velocity noise term.
    """
    y: Array
    J: Array
    J_dot: Array


class ContactKinematics(Protocol):
    """FK + Jacobians at the filtered joint state.

    Closed over by `make_step` at build time, so it is static under ``jit`` and
    may hold whatever the implementation needs (an MJX model, a fixture chain).
    It must be jit-able and differentiable, and must not branch on traced data.
    """

    def __call__(self, q: Array, q_dot: Array) -> ContactFrames: ...


class InEKFInputs(NamedTuple):
    """Per-tick scan input (``xs``).

    Attributes
    ----------
    omega : Array, shape (3,)
        **Bias-corrected** gyro ``ω̄`` — drives the propagation (I1).
    accel : Array, shape (3,)
        Bias-corrected IMU specific force ``ā``.  Gravity is added internally.
    raw_omega : Array, shape (3,)
        **Raw** gyro, for the gravity reference and its rotation gate.  Kept
        separate on purpose: `testRotationGateUsesRawGyroNotBiasCorrupted`
        requires the gate to see the uncorrected signal.
    joint : JointFilterOutput
        The joint-KF boundary (above).
    contact_mask : Array, shape (N,)
        Per-contact trust in ``[0, 1]``, from the **previous** tick's trusted set
        (§4 phase ordering).  0 ⇒ that contact's FK update is masked out.
    contact_chol : Array, shape (N, 3, 3)
        ContactNet Cholesky factors ``L_{C_i}``; `contact.digest` reconstructs
        and floors them.  Default heuristic: a constant diagonal factor.
    """
    omega: Array
    accel: Array
    raw_omega: Array
    joint: JointFilterOutput
    contact_mask: Array
    contact_chol: Array


class InEKFCarry(NamedTuple):
    """Scan carry: the filter state plus the gravity reference."""
    state: InEKFState
    gravity_ref: GravityRef


class InEKFOutputs(NamedTuple):
    """Per-tick emitted diagnostics — the YoVariable analogue (§4).

    Stacked over time by `run`.  These are the seam the consistency evaluation
    (G10) and ContactNet's trust features read.
    """
    state: InEKFState
    contact_innovation: Array              # (3N,)
    contact_diagnostics: UpdateDiagnostics
    gravity_diagnostics: UpdateDiagnostics
    tilt_angle: Array                      # scalar
    quasi_static: Array                    # scalar float mask


# ---------------------------------------------------------------------------
# Measurement-noise routing (the joint-KF boundary, §4.2)
# ---------------------------------------------------------------------------

def contact_position_noise(J: Array, sigma_q: Array) -> Array:
    """``N^p_i = J_{C_i} Σ_q J_{C_i}ᵀ`` for every contact — vmapped, no loop."""
    return jax.vmap(map_encoder_noise, in_axes=(0, None))(J, sigma_q)


def contact_velocity_noise(J_dot: Array, sigma_q_dot: Array) -> Array:
    """``N^v_i = J_{Ċ_i} Σ_q̇ J_{Ċ_i}ᵀ`` for every contact.

    Kept as its own term, never folded into ``N^p``: they are noises on two
    different measurements.  See the TODO in `step` for why it is not yet
    consumed.
    """
    return jax.vmap(map_encoder_noise, in_axes=(0, None))(J_dot, sigma_q_dot)


def mask_contact_noise(Np: Array, contact_mask: Array) -> Array:
    r"""Blend per-contact noise toward ``R_LARGE`` for untrusted contacts (§4).

    ``mask = 1`` keeps ``N^p_i``; ``mask = 0`` substitutes ``R_LARGE · I₃``.  The
    rows stay — dropping them would change the shape (I7), and *zeroing* them
    would make ``S`` singular (§6 trap).  As ``R_LARGE → ∞`` the posterior tends
    to the one that excludes the contact entirely.
    """
    big = R_LARGE * jnp.eye(3)
    w = contact_mask.reshape(-1, 1, 1)
    return w * Np + (1.0 - w) * big


# ---------------------------------------------------------------------------
# The scan body
# ---------------------------------------------------------------------------

def make_step(ekf: InvariantEKF, kinematics: ContactKinematics):
    """Build the jitted scan body for a given filter wiring and robot model.

    ``ekf`` and ``kinematics`` are closed over (static); everything that varies
    per tick arrives through `InEKFInputs`.

    Returns
    -------
    callable
        ``step(carry, inputs) -> (carry, outputs)`` — the `lax.scan` body.
    """
    gravity_params = ekf.gravity_params

    def step(carry: InEKFCarry, inputs: InEKFInputs) -> tuple[InEKFCarry, InEKFOutputs]:
        state, gravity_ref = carry

        # -- 1. propagate on the bias-corrected IMU (§3) --------------------
        sigma_c = digest(inputs.contact_chol, ekf.params)
        state = propagate(state, inputs.omega, inputs.accel, sigma_c, ekf.params)

        # -- 2. contact FK update (§4) --------------------------------------
        frames = kinematics(inputs.joint.q, inputs.joint.q_dot)

        # Joint-KF covariance enters here and only here, through the Jacobian.
        Np = contact_position_noise(frames.J, inputs.joint.sigma_q)
        Np = mask_contact_noise(Np, inputs.contact_mask)

        # TODO(N^v / zero-velocity): `contact_velocity_noise(frames.J_dot,
        # inputs.joint.sigma_q_dot)` is the noise on the contact *zero-velocity*
        # constraint.  That constraint is a separate measurement block with its
        # own H rows stacked below the position block — it is NOT folded into
        # N^p.  Deferred (design §10, still open); `sigma_q_dot` is carried
        # through the boundary so adding it later is a change here only.

        nu = innovation(state, frames.y)
        # Zero the residual on untrusted contacts as well as inflating R, so a
        # masked contact contributes nothing rather than a large-but-nonzero pull.
        nu = (nu.reshape(-1, 3) * inputs.contact_mask.reshape(-1, 1)).reshape(-1)

        state, contact_diagnostics = linear_update(
            state, ekf.params.H, nu, measurement_noise(Np)
        )

        # -- 3. gravity leveling (G4), gated --------------------------------
        gate = is_quasi_static(
            gravity_ref, inputs.accel, inputs.raw_omega, gravity_params
        ).astype(jnp.float64)
        meas = assemble_gravity_leveling(
            gravity_ref, state, inputs.accel, gravity_params
        )
        state, gravity_diagnostics = linear_update(
            state, meas.H, meas.residual, meas.R,
            cond_max=gravity_params.cond_max, gate=gate,
        )
        gravity_ref = update_gravity_reference(
            meas.ref, inputs.accel, inputs.raw_omega, ekf.params.dt, gravity_params
        )

        outputs = InEKFOutputs(
            state=state,
            contact_innovation=nu,
            contact_diagnostics=contact_diagnostics,
            gravity_diagnostics=gravity_diagnostics,
            tilt_angle=meas.tilt_angle,
            quasi_static=gate,
        )
        return InEKFCarry(state=state, gravity_ref=gravity_ref), outputs

    return step


# ---------------------------------------------------------------------------
# Trajectory driver
# ---------------------------------------------------------------------------

def init_carry(state: InEKFState) -> InEKFCarry:
    """Fresh scan carry: the given state and an unseeded gravity reference."""
    return InEKFCarry(state=state, gravity_ref=init_gravity_ref())


def run(
    ekf: InvariantEKF,
    kinematics: ContactKinematics,
    state: InEKFState,
    inputs: InEKFInputs,
) -> tuple[InEKFCarry, InEKFOutputs]:
    """Run the filter over a trajectory with `lax.scan`.

    Parameters
    ----------
    ekf : InvariantEKF
    kinematics : ContactKinematics
    state : InEKFState
        Initial state.
    inputs : InEKFInputs
        Every field carries a leading time axis of length ``T``.

    Returns
    -------
    carry : InEKFCarry
        Final state and gravity reference.
    outputs : InEKFOutputs
        Per-tick diagnostics, stacked along a leading time axis.

    The whole scan is differentiable end-to-end, so BPTT into ContactNet flows
    through it — that is why the filter holds no trainable parameters.
    """
    return jax.lax.scan(make_step(ekf, kinematics), init_carry(state), inputs)
