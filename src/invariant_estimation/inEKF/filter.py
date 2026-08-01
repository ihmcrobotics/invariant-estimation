r"""The scan body — ``step`` — and the trajectory driver ``run``.

Everything below this file is a pure function over an explicit ``(X, P)`` carry (I10);
this is the one place they are composed into a tick, and the tick is deliberately
boring::

    propagate  (IMU)                    §3
      -> contact FK update  (joint KF + ContactNet)   §4
      -> gravity leveling   (accelerometer, gated)    G4

Constant-graph contract (I7)
----------------------------
``step`` must trace to the **same jaxpr** regardless of which contacts are trusted,
whether the quasi-static gate is open, or whether pitch is observable.  Nothing here
branches on data: the gravity gate multiplies into ``K`` as a float mask, so a closed
gate leaves ``(X̂, P)`` bit-for-bit unchanged rather than skipping work; contact
condition is a *continuous covariance*, never a shape change (below); and ``N``, the
joint count and the contact count are static, fixed at build time.
`tests/inEKF/test_filter.py` covers this.

.. _no-contact-mask:

DECISION — there is no contact mask.  Read this before debugging a foot.
------------------------------------------------------------------------
**Contact condition is expressed *only* through ``Σ_C``** (the ContactNet Cholesky
factor, which reaches the contact block of ``Q_d`` via `contact.digest`).  There is
deliberately **no per-foot trust mask and no measurement kill-switch** in this filter.
If you are debugging a foot that "should have been ignored", this is the thing you are
looking for, and it is absent on purpose.

Why: the FK measurement ``y_i = R̂ᵀ(d̄_i − p̄)`` is *not wrong* during swing — the
encoders still locate the foot relative to the base perfectly well.  What breaks in
swing is the assumption that ``d_i`` is world-static, and that assumption lives in the
**process** noise, not the measurement noise.  Inflating ``Σ_C`` is therefore the
physically correct lever; masking the measurement treats a true observation as false.

Measured (`test_large_contact_covariance_isolates_a_swing_foot`): with ``Σ_C = 1.0``
for 100 swing ticks, an 8 cm foot displacement is absorbed **96%** into the contact
anchor and perturbs the base by 3.7 mm — a **7.6x** attenuation versus the same foot
planted.  The residual base motion is not a leak: it is ``P_pp/(P_pp + P_dd) ≈ 8%`` of
the residual, which is what Bayes says belongs to the base under that prior, and it
shrinks further as the swing continues.  ``Σ_C`` is also strictly more expressive than
a scalar trust weight: a full covariance can say "this foot slides along the surface
but not through it", which no single number can.

Consequences to keep in mind:

* A swing foot **does** still receive a small, correct base correction.  That is
  intended.  If it is too large, ``Σ_C`` is too small — do not reach for a mask.
* CLAUDE.md §4's ``R_LARGE = 1e12·I₃`` masking rule is **not** about this filter.  It
  governs the **joint KF's stance anchors** (oracle checked in the G7 stacked-oracle
  port).  An earlier version of this module imported that mechanism here by mistake.
* An encoder fault is ``Σ_q``, not a contact concern.

I2 (contacts permanently in state) and §7 (contact condition expressed through
``Σ_C``) are the governing invariants.

Two contact covariance sockets — do not conflate them
-----------------------------------------------------
The decision above governs contact *condition* and is unchanged.  It is not the only
place a contact covariance enters, and the two are different physics:

===================  ====================  ==============================
``contact_chol``     process, ``Q_d``      *Is this foot world-static?*
``contact_meas_chol``  measurement, ``N``  *How well do we know where it is?*
===================  ====================  ==============================

``contact_chol`` answers the swing/slip question and is the lever the DECISION argues
for.  ``contact_meas_chol`` is additive on the encoder term,
``N_i = J_{C_i} Σ_q J_{C_i}ᵀ + Σ_{C_i}``, and models sole compliance and contact-point
geometry — uncertainty that is present in *firm* stance and that the encoder term
structurally cannot express.

**ContactNet learns the process one** (`pipeline/main_estimator.py`), as of
2026-07-29; network_plan.md §1's measurement socket, which runs 1–4 were trained on,
is superseded.  The argument is structural rather than empirical: for one contact
``H = [0 0 I −I]``, so ``K = P Hᵀ (H P Hᵀ + N)⁻¹`` and ``N`` appears *only inside the
inverted factor*.  It scales the correction and reweights residual axes, but it cannot
change how a residual is apportioned between the base and the anchor — and that
apportionment, which is pure prior and hence pure process noise, is what an integrated
velocity bias lives on.  CoCo-InEKF Eq. (5) puts the learned covariance in the same
place (the network is called inside Prediction, Alg. 1 line 1).

``contact_meas_chol`` is consequently **unused**: it stays at zeros, which is the
pre-ContactNet filter bit-for-bit.  It is kept rather than deleted because it is a real
and separate physical quantity, it is what runs 1–4 were trained into, and
`ContactUpdaterTest`'s ported cases exercise it.

The joint-filter boundary
-------------------------
`JointFilterOutput` is the seam the joint KF feeds through: ``(q̂, q̇̂, Σ_q, Σ_q̇)``.  Per
the boundary contract those enter **only** on the correction side, always
pre-multiplied by a kinematic Jacobian::

    N^p_i = J_{C_i}(q̂) Σ_q J_{C_i}ᵀ         position FK noise  (used)
    N^v_i = J_{C_i}(q̂) Σ_q̇ J_{C_i}ᵀ         contact velocity noise (see
                                            `contact_velocity_noise`)

They never reach the propagation ``Φ`` or the inertial ``Q``.  The kinematics
themselves come from a caller-supplied `ContactKinematics` callable — the ``robot/``
seam — closed over at build time so it is static under ``jit``.
"""
from typing import NamedTuple, Protocol

import jax
from jax import Array
import jax.numpy as jnp

from .contact import digest, reconstruct_cov
from .correct import (
    UpdateDiagnostics,
    innovation,
    linear_update,
    map_encoder_noise,
    measurement_noise,
    rotate_measurement_covariance,
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
from .reseed import LatchState, expand_per_foot, init_latch, reseed_step
from .state import InEKFState


class JointFilterOutput(NamedTuple):
    """What the joint KF hands the InEKF each tick.

    Attributes
    ----------
    q : Array, shape (n_joints,)
        Filtered joint positions ``q̂``.
    q_dot : Array, shape (n_joints,)
        Filtered joint velocities ``q̇̂``.
    sigma_q : Array, shape (n_joints, n_joints)
        Joint-position covariance ``Σ_q``.  Full matrix, not a diagonal: the joint
        KF's covariance is genuinely coupled through the mass matrix, and ``J Σ_q Jᵀ``
        needs the off-diagonals.
    sigma_q_dot : Array, shape (n_joints, n_joints)
        Joint-velocity covariance ``Σ_q̇``.  Routed as its own term, never folded into
        ``Σ_q`` — see `contact_velocity_noise`.

    The base gyro-bias estimate ``b̂`` is deliberately **not** here: the InEKF consumes
    already-bias-corrected IMU (I1), so the correction happens upstream.
    """
    q: Array
    q_dot: Array
    sigma_q: Array
    sigma_q_dot: Array


class ContactFrames(NamedTuple):
    """Kinematics evaluated at ``q̂`` — the ``robot/`` seam's output.

    ``y`` ``(N, 3)`` are the body-frame base→contact vectors ``h_{p,i}(q̂)`` (the FK
    measurement); ``J`` ``(N, 3, n_joints)`` the contact-point position Jacobians
    ``J_{C_i}(q̂)``; ``J_dot`` their time derivatives ``J_{Ċ_i}``.
    """
    y: Array
    J: Array
    J_dot: Array


class ContactKinematics(Protocol):
    """FK + Jacobians at the filtered joint state.

    Closed over by `make_step` at build time, so it is static under ``jit`` and may
    hold whatever the implementation needs (an MJX model, a fixture chain).  It must be
    jit-able and differentiable, and must not branch on traced data.
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
        **Raw** gyro, for the gravity reference and its rotation gate.  Kept separate
        on purpose: `testRotationGateUsesRawGyroNotBiasCorrupted` requires the gate to
        see the uncorrected signal.
    joint : JointFilterOutput
        The joint-KF boundary.
    contact_chol : Array, shape (N, 3, 3)
        Cholesky factors of the **stance-anchor slip process noise**; `contact.digest`
        reconstructs and floors them into the ``Σ_C`` that reaches the contact block of
        ``Q_d``.  **This is the only contact-condition input** — see the DECISION note
        in the module docstring.  Firm contact ⇒ small; slip ⇒ anisotropic; swing ⇒
        large.  Default heuristic: a constant diagonal factor, inflated for swing feet.

        **This is what ContactNet learns** (since 2026-07-29): it sets how fast an
        anchor is allowed to drift, which is the apportionment the sink lives on.
        `contact.apply_floor` is the safety bound on it, and is safety-critical now
        that the supplier is learned.
    contact_meas_chol : Array, shape (N, 3, 3)
        Cholesky factors of the **contact FK measurement noise** ``Σ_C``, added to the
        encoder term: ``N_i = J_{C_i} Σ_q J_{C_i}ᵀ + Σ_{C_i}``.  Sole compliance,
        contact-point geometry, foot deformation: real uncertainty in *where the foot
        is*, which the encoder term alone does not model.

        **Zeros in every shipped path**, which is the pre-ContactNet filter exactly.  A
        caller may still drive it — `experiments/replay_eval.py --socket meas` does, to
        score the old checkpoints — and the update consumes it unchanged.
    contact_prob : Array, shape (K,), optional
        Per-foot contact probability, read **only** when the filter was built with
        touchdown re-seed enabled (`InvariantEKF.reseed`); `expand_per_foot` maps it to
        the ``N`` contacts.  Defaults to ``()`` — an empty pytree node, so on every
        shipped path it contributes no leaf, costs nothing, and leaves `InEKFInputs`
        serialising exactly as it did before the field existed.
    """
    omega: Array
    accel: Array
    raw_omega: Array
    joint: JointFilterOutput
    contact_chol: Array
    contact_meas_chol: Array
    contact_prob: Array = ()


class InEKFCarry(NamedTuple):
    """Scan carry: the filter state, the gravity reference, and the re-seed latch.

    `latch` is ``None`` unless the filter was built with re-seed enabled — again an
    empty pytree node, so a carry built the old way is structurally what it was.
    """
    state: InEKFState
    gravity_ref: GravityRef
    latch: "LatchState | None" = None


class InEKFOutputs(NamedTuple):
    """Per-tick emitted diagnostics — the YoVariable analogue, stacked over time by
    `run`.  These are the seam the consistency evaluation (G10) and ContactNet's trust
    features read.
    """
    state: InEKFState
    contact_innovation: Array              # (3N,)
    contact_diagnostics: UpdateDiagnostics
    gravity_diagnostics: UpdateDiagnostics
    tilt_angle: Array                      # scalar
    quasi_static: Array                    # scalar float mask
    reseed_fire: Array = ()                # (N,) 1.0 where a contact re-anchored
    reseed_residual: Array = ()            # (N,3) how far the old anchor had drifted


def contact_position_noise(J: Array, sigma_q: Array) -> Array:
    """``N^p_i = J_{C_i} Σ_q J_{C_i}ᵀ`` for every contact — vmapped, no loop."""
    return jax.vmap(map_encoder_noise, in_axes=(0, None))(J, sigma_q)


def contact_velocity_noise(J: Array, sigma_q_dot: Array) -> Array:
    r"""``N^v_i = J_{C_i} Σ_q̇ J_{C_i}ᵀ`` for every contact.

    **Zero call sites — deliberately.**  TODO(N^v / zero-velocity): this is the noise
    on a contact *zero-velocity* constraint, which would be a SEPARATE measurement
    block with its own ``H`` rows — never folded into ``N^p``, since the two are noises
    on different measurements.  Deferred: neither Lucas's derivation nor CoCo-InEKF
    (whose only correction is its Eq. (8), the FK position update with
    ``N = J Σ_q Jᵀ``) has a velocity-level measurement, and the constant ``H_v`` such a
    block would want is NOT exactly state-independent — see TODO.md.  ``sigma_q_dot``
    stays plumbed through the boundary so adding it later is a change in `make_step`
    only.

    **The Jacobian is ``J_{C_i}``, not ``J_{Ċ_i}``** — corrected 2026-07-29.  The world
    velocity of contact ``i`` is ``v + R(ω × h_i + J_{C_i} q̇)``, so the sensitivity of a
    measured contact velocity to ``q̇`` is the *position* Jacobian.  Two independent
    checks: ``J_{Ċ} Σ_q̇ J_{Ċ}ᵀ`` has units ``m²/s⁴`` (not a velocity covariance), and
    `pipeline.main_estimator`'s MJX kinematics returns ``J_dot = 0``, so the old pairing
    would have been identically zero on the deployment path while looking wired.
    """
    return jax.vmap(map_encoder_noise, in_axes=(0, None))(J, sigma_q_dot)


def make_step(ekf: InvariantEKF, kinematics: ContactKinematics):
    """Build the jitted scan body ``step(carry, inputs) -> (carry, outputs)``.

    ``ekf`` and ``kinematics`` are closed over (static); everything that varies per
    tick arrives through `InEKFInputs`.
    """
    gravity_params = ekf.gravity_params

    def step(carry: InEKFCarry, inputs: InEKFInputs) -> tuple[InEKFCarry, InEKFOutputs]:
        state, gravity_ref = carry.state, carry.gravity_ref

        # -- 1. propagate on the bias-corrected IMU (§3) --------------------
        # `contact_chol` is ContactNet's output on the deployment path; `digest`
        # is unchanged by that and does not know the difference. The floor it
        # applies is what bounds a mis-prediction — see `contact.apply_floor`.
        sigma_c = digest(inputs.contact_chol, ekf.params)
        state = propagate(state, inputs.omega, inputs.accel, sigma_c, ekf.params)

        # -- 2. contact FK update (§4) --------------------------------------
        frames = kinematics(inputs.joint.q, inputs.joint.q_dot)

        # Joint-KF covariance enters here and only here, through the Jacobian.
        # Note there is no per-contact mask: contact condition rides entirely in
        # the *process* Σ_C (the DECISION note above).
        Np = contact_position_noise(frames.J, inputs.joint.sigma_q)

        # The FK measurement noise, additive per contact and in the same frame as
        # Np.  Zero on every shipped path (ContactNet drives `contact_chol`
        # instead), so this reduces to the pre-ContactNet filter exactly.  No
        # floor is applied: S = H P Hᵀ + N needs only N PSD (H P Hᵀ is already
        # SPD — see `kalman_gain`).
        Nc = reconstruct_cov(inputs.contact_meas_chol)

        # -- 2b. touchdown re-seed (ablation; OFF unless `ekf.reseed` is set) ---
        # Placed between propagate and the contact update — Java's call site — and
        # BEFORE `innovation`, so the update this tick sees the re-anchored state.
        # For a contact that just fired that means a zero residual and a zero
        # rotation correction (the zero-release property), which is the point: a
        # re-seed replaces the correction it would otherwise have provoked rather
        # than adding to it.
        #
        # The FK covariance handed to the re-seed is the SAME `Np + Nc` the update
        # uses. It has to be: the re-seed is placing the anchor with that
        # measurement, so the anchor's new uncertainty is that measurement's.
        latch, fire, reseed_residual = carry.latch, (), ()
        if ekf.reseed is not None:
            latch, state, fire, reseed_residual = reseed_step(
                ekf.reseed, latch, state,
                expand_per_foot(inputs.contact_prob, ekf.N), frames.y, Np + Nc,
            )

        nu = innovation(state, frames.y)

        # ᴮ -> ᵂ.  `innovation` returns `R̂y - (d̂-p̂)`, which is a WORLD-frame
        # residual, while `Np` and `Nc` are both body-frame.  `S = H P Hᵀ + N`
        # therefore needs `R̂ N R̂ᵀ` (Java `ContactUpdater.computeMeasurementCovariance`).
        #
        # This was missing until 2026-07-28 and is INVISIBLE for isotropic noise,
        # where `R̂(σ²I)R̂ᵀ = σ²I` -- measured 6.4e-22, i.e. machine zero.  It is
        # not a no-op for anisotropic noise: an anisotropic slip covariance
        # diag(1e-3, 1e-3, 1e-8) shifts the Kalman gain by 2.7e-3 relative.
        # Anisotropy is exactly what ContactNet exists to produce, and the network
        # cannot compensate because §1 forbids it from seeing R̂.  Conjugation on
        # the PRIOR state, matching `innovation`.
        N_world = rotate_measurement_covariance(state, Np + Nc)

        state, contact_diagnostics = linear_update(
            state, ekf.params.H, nu, measurement_noise(N_world)
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
            reseed_fire=fire,
            reseed_residual=reseed_residual,
        )
        return InEKFCarry(state=state, gravity_ref=gravity_ref, latch=latch), outputs

    return step


def init_carry(state: InEKFState, *, reseed: bool = False) -> InEKFCarry:
    """Fresh scan carry: the given state and an unseeded gravity reference.

    `reseed=True` adds a disarmed `LatchState` — disarmed so the first touchdown does
    not re-anchor anchors that `init_state` only just placed from the same FK.
    """
    return InEKFCarry(state=state, gravity_ref=init_gravity_ref(),
                      latch=init_latch(state.N) if reseed else None)


def run(
    ekf: InvariantEKF,
    kinematics: ContactKinematics,
    state: InEKFState,
    inputs: InEKFInputs,
) -> tuple[InEKFCarry, InEKFOutputs]:
    """Run the filter over a trajectory with `lax.scan`.

    Every field of ``inputs`` carries a leading time axis of length ``T``; the returned
    outputs are stacked along the same axis.  The whole scan is differentiable
    end-to-end, so BPTT into ContactNet flows through it — that is why the filter holds
    no trainable parameters.
    """
    return jax.lax.scan(make_step(ekf, kinematics), init_carry(state), inputs)
