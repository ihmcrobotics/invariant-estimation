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

* the gravity gate multiplies into ``K`` as a float mask, so a closed gate leaves
  ``(X̂, P)`` bit-for-bit unchanged rather than skipping work;
* contact condition is a *continuous covariance*, never a shape change (below);
* ``N``, the joint count, and the contact count are static, fixed at build time.

`tests/inEKF/test_filter.py` covers this: no data-dependent branch survives
tracing, and the jitted step never recompiles as inputs change.

.. _no-contact-mask:

DECISION — there is no contact mask.  Read this before debugging a foot.
------------------------------------------------------------------------
**Contact condition is expressed *only* through ``Σ_C``** (the ContactNet
Cholesky factor, which reaches the contact block of ``Q_d`` via `contact.digest`).
There is deliberately **no per-foot trust mask and no measurement kill-switch**
in this filter.  If you are debugging a foot that "should have been ignored",
this is the thing you are looking for, and it is absent on purpose.

Why: the FK measurement ``y_i = R̂ᵀ(d̄_i − p̄)`` is *not wrong* during swing — the
encoders still locate the foot relative to the base perfectly well.  What breaks
in swing is the assumption that ``d_i`` is world-static, and that assumption
lives in the **process** noise, not the measurement noise.  Inflating ``Σ_C`` is
therefore the physically correct lever; masking the measurement treats a true
observation as false.

Measured (`test_large_contact_covariance_isolates_a_swing_foot`): with
``Σ_C = 1.0`` for 100 swing ticks, an 8 cm foot displacement is absorbed **96%**
into the contact anchor and perturbs the base by 3.7 mm — a **7.6x** attenuation
versus the same foot planted.  The residual base motion is not a leak: it is
``P_pp/(P_pp + P_dd) ≈ 8%`` of the residual, which is what Bayes says belongs to
the base under that prior, and it shrinks further as the swing continues.

``Σ_C`` is also strictly more expressive than a scalar trust weight: a full
covariance can say "this foot slides along the surface but not through it",
which no single number can.

Consequences to keep in mind:

* A swing foot **does** still receive a small, correct base correction.  That is
  intended.  If it is too large, ``Σ_C`` is too small — do not reach for a mask.
* CLAUDE.md §4's ``R_LARGE = 1e12·I₃`` masking rule is **not** about this filter.
  It governs the **joint KF's stance anchors** (§2 "trusted feet → anchors",
  oracle checked in the G7 stacked-oracle port).  An earlier version of this
  module imported that mechanism here by mistake; see PORT_NOTES.md.
* An encoder fault is ``Σ_q``, not a contact concern.

CLAUDE.md I2 (contacts permanently in state) and §7 (contact condition expressed
through ``Σ_C``) are the governing invariants.

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

from .contact import digest, rolling_anchor_density
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
from .reseed import (
    LatchState,
    advance_latch,
    init_latch,
    pre_reseed_residual,
    reseed_contacts,
)
from .state import InEKFState

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
    omega_rel : Array, shape (N, 3), optional
        Each contact's **foot** angular velocity *relative to the base*, expressed
        in the body frame ``B``:  ``ω_rel = vee(Ċ Cᵀ)`` with ``C(q) = R_Bᵀ R_sole``.

        Read only by the rolling-anchor density (`contact.rolling_anchor_density`)
        and only on a build with ``rolling.enabled``.  Optional, so a kinematics
        fixture that does not supply it still satisfies the protocol; a build with
        the term enabled will fail loudly on the ``()`` default.

        Note this is a *measured* quantity — encoders differentiated through FK —
        not a filter state, which is the whole reason the rolling-anchor term
        needs no contact inference.
    """
    y: Array
    J: Array
    J_dot: Array
    omega_rel: Array = ()


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
    contact_chol : Array, shape (N, 3, 3)
        ContactNet Cholesky factors ``L_{C_i}``; `contact.digest` reconstructs
        and floors them into ``Σ_C``.  **This is the only contact-condition
        input** — see the DECISION note in the module docstring.  Firm contact ⇒
        small; slip ⇒ anisotropic; swing ⇒ large.  Default heuristic: a constant
        diagonal factor, inflated for swing feet.
    contact_prob : Array, shape (N,), optional
        Per-contact contact signal in [0, 1], read **only** by the touchdown
        re-seed latch (`reseed.advance_latch`).  Required when the filter is
        built with ``reseed.enabled``; ignored, and safely left at its ``()``
        default, otherwise.

        This is **not** a contact mask and does not contradict the DECISION note
        above: it never gates a measurement and never scales a gain.  It selects
        the *instant* at which a contact slot is re-anchored — a
        re-initialisation, not an update — which is a question about timing that
        ``Σ_C`` cannot answer, because ``Σ_C`` shapes a gain along directions the
        contact innovation can see and the re-seed acts on the one it cannot.
    """
    omega: Array
    accel: Array
    raw_omega: Array
    joint: JointFilterOutput
    contact_chol: Array
    contact_prob: Array = ()


class InEKFCarry(NamedTuple):
    """Scan carry: the filter state, the gravity reference, the re-seed latch.

    ``latch`` is ``None`` (an empty pytree, costing the carry nothing) on a build
    with the re-seed disabled.
    """
    state: InEKFState
    gravity_ref: GravityRef
    latch: LatchState | None = None


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
    reseed_fire: Array                     # (N,) float mask, all-zero when off
    reseed_residual: Array                 # (N,) pre-reseed residual norm [m]


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
        state, gravity_ref, latch = carry

        # -- 1. kinematics ---------------------------------------------------
        # Hoisted above the propagation because the rolling-anchor density needs
        # this tick's measured foot angular velocity, and it feeds the contact
        # blocks of Q_d.  Also lets the re-seed (2a) re-anchor onto this tick's
        # FK, which is what gives the zero-release property in the update below.
        frames = kinematics(inputs.joint.q, inputs.joint.q_dot)

        # -- 2. propagate on the bias-corrected IMU (§3) ---------------------
        sigma_c = digest(inputs.contact_chol, ekf.params)
        if ekf.rolling.enabled:
            # ω_foot = ω_base + ω_rel, both in the body frame B — exactly the
            # frame `digest` emits, so `Ad_X̂` in `build_Qd` rotates the whole
            # thing to world in one step.  Purely measured: gyro + encoders.
            omega_foot = inputs.omega[None, :] + jnp.asarray(frames.omega_rel)
            sigma_c = sigma_c + rolling_anchor_density(omega_foot, ekf.rolling)
        state = propagate(state, inputs.omega, inputs.accel, sigma_c, ekf.params)

        # -- 3. contact FK update (§4) --------------------------------------

        # Joint-KF covariance enters here and only here, through the Jacobian.
        # Note there is no per-contact mask: contact condition rides entirely in
        # Σ_C (the DECISION note above).
        Np = contact_position_noise(frames.J, inputs.joint.sigma_q)

        # -- 2a. touchdown re-seed (reseed.py), between predict and update ---
        # `enabled` is a build-time Python flag, so a disabled build traces to
        # the exact graph it did before this feature existed (I7).
        if ekf.reseed.enabled:
            reseed_residual = pre_reseed_residual(state, frames.y)
            latch, reseed_fire = advance_latch(latch, inputs.contact_prob, ekf.reseed)
            state = reseed_contacts(state, frames.y, Np, reseed_fire)
        else:
            reseed_fire = jnp.zeros(ekf.N, dtype=jnp.float64)
            reseed_residual = jnp.zeros(ekf.N, dtype=jnp.float64)

        # TODO(N^v / zero-velocity): `contact_velocity_noise(frames.J_dot,
        # inputs.joint.sigma_q_dot)` is the noise on the contact *zero-velocity*
        # constraint.  That constraint is a separate measurement block with its
        # own H rows stacked below the position block — it is NOT folded into
        # N^p.  Deferred (design §10, still open); `sigma_q_dot` is carried
        # through the boundary so adding it later is a change here only.

        nu = innovation(state, frames.y)
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
            reseed_fire=reseed_fire,
            reseed_residual=reseed_residual,
        )
        return InEKFCarry(
            state=state, gravity_ref=gravity_ref, latch=latch
        ), outputs

    return step


# ---------------------------------------------------------------------------
# Trajectory driver
# ---------------------------------------------------------------------------

def init_carry(state: InEKFState, ekf: InvariantEKF | None = None) -> InEKFCarry:
    """Fresh scan carry: the given state and an unseeded gravity reference.

    ``ekf`` is optional and only consulted for the re-seed wiring: pass it to get
    an armed `LatchState` on a build with ``reseed.enabled``.  Omitting it (or
    passing a build with the re-seed off) leaves ``latch=None``, which is the
    pre-existing carry exactly.
    """
    enabled = ekf is not None and ekf.reseed.enabled
    return InEKFCarry(
        state=state,
        gravity_ref=init_gravity_ref(),
        latch=init_latch(state.N) if enabled else None,
    )


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
    return jax.lax.scan(make_step(ekf, kinematics), init_carry(state, ekf), inputs)
