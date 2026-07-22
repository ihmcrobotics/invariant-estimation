"""
jointKF/filter.py
=================
The tick: `predict -> encoder update -> stacked gyro/anchor update`, and a
`lax.scan` over a trajectory.

This is the orchestrator the Java `computeJointState()` (phase 1) and
`computeImuBiases(feet)` (phase 2) map onto.  It owns three things that are
easy to get wrong in ways no single module can see, and that therefore live
here rather than being distributed:

**1. Phase ordering of the trusted-feet mask.**  The *previous* tick's trust set
drives *this* tick's anchors (CLAUDE.md §4): the mask is written at the end of
step `k` and read at the start of `k+1`.  This is not a stylistic choice.  The
contact signal is derived from the same sensors the filter is about to consume,
so using this tick's mask would let a foot's contact decision and the
measurement it gates be correlated through the noise, which quietly biases the
bias estimate — the one quantity the anchor exists to make observable.  Carrying
the mask in the scan state is also what keeps it a *value* rather than a
Python-level decision (I7).

**2. Two separate updates, not one stacked block.**  Encoders and the gyro/anchor
stack are applied as sequential Joseph updates rather than one concatenated
measurement.  Algebraically the two agree only if the noises are independent —
which they are — but they differ completely under *gating*: one stacked update
means a single ill-conditioned gyro row throws the encoders away too.  Splitting
them means a foot in swing, or a NaN on one IMU, costs the filter only the
channel that actually went bad.  The `cond(S)` gate makes this a behavioural
difference, not a numerical one.

**3. Everything is a fixed-shape mask.**  No branch in this file depends on a
traced value, so the jaxpr is identical whichever feet are on the ground and
whichever gates fire (I7).  What that genuinely buys is *no recompilation* in a
vmapped/scanned MJX rollout; it does not by itself prove the masks are right,
which is what the ported tests are for.

Model quantities as inputs
--------------------------
`step` takes the model-derived quantities (`J_rel`, `R_rel`, `M`, the anchor
Jacobians) as *arguments* rather than evaluating a `RobotModel` internally.  Two
reasons.  It keeps the filter free of any simulator dependency, matching the seam
discipline the InEKF already follows; and MJX's kinematics tracing cost grows
sharply with chain depth (PORT_NOTES G1: ~1.3 s to jit a 4-link chain, ~240 s for
10), so the caller must stay free to evaluate the model once per tick, batch it
under `vmap`, or precompute a trajectory — decisions that belong to the rollout,
not to the filter.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from . import anchors as anchors_mod
from . import measure, predict as predict_mod, process
from .state import JointKFBuild, JointKFParams, JointKFState


class SensorInputs(NamedTuple):
    """One tick of proprioception.

    Attributes
    ----------
    encoders : (n,)
        Measured joint positions, in filter state order.
    gyros : (m, 3)
        Per-IMU angular rate, each **in its own measurement frame** — the frame
        `b_omega` is stored in, which is what makes the child bias block `+I3`.
    qd_unfiltered : (n_u,)
        Measured velocities of the base->foot chain joints that are not filter
        states (the Alex ankles), in `anchor_unfiltered_mask` column order.
    contact : (K,)
        This tick's contact/trust signal per anchor slot. Consumed on the NEXT
        tick — see the module docstring on phase ordering.
    """

    encoders: Array
    gyros: Array
    qd_unfiltered: Array
    contact: Array


class ModelInputs(NamedTuple):
    """Model-derived quantities for one tick (see the module docstring)."""

    J_rel: Array                       # (n_pairs, 3, n)   relative gyro Jacobians
    R_rel: Array                       # (n_pairs, 3, 3)   child <- parent rotations
    anchor_jac: anchors_mod.AnchorJacobians
    M: Array | None = None             # (nv, nv) mass matrix; None => scalar CWNA


class FilterCarry(NamedTuple):
    """Scan state: the filter's `(x, P)` plus the one-tick-delayed trust mask."""

    state: JointKFState
    trusted_feet: Array                # (K,) previous tick's mask


class TickDiagnostics(NamedTuple):
    """Per-tick observables (CLAUDE.md §4: the tests read these, so they are seam).

    The Java filter publishes these as YoVariables; here they are a pytree so a
    scan can stack them over a trajectory without any host callback.

    The per-channel NIS fields are kept **separate** rather than merged into one
    array.  Java dispatches its diagnostics on an exact-match label string
    (``"encoder"`` vs ``"encoderVelocity"``), which cannot cross a jit boundary;
    separate fields port the *observable* that dispatch existed to provide — one
    channel structurally cannot publish into another's diagnostic.  The
    direct-velocity channel's cross-talk guard (`velocity.py`) asserts exactly
    this, and it only holds end-to-end because the encoder channel writes the
    encoder field and nothing else does.
    """

    encoder_nis: Array
    encoder_nis_per_joint: Array
    encoder_applied: Array
    stacked_nis: Array
    stacked_nis_per_row: Array
    stacked_applied: Array
    encoder_cond: Array
    stacked_cond: Array
    qa_max_diag: Array
    qa_tripped: Array
    active_anchors: Array


def init_carry(build: JointKFBuild, params: JointKFParams, q0: Array | None = None) -> FilterCarry:
    """Seed the scan carry.  Feet start untrusted.

    Untrusted rather than trusted is deliberate and matches the Java on-ground
    init gate's intent: the base bias is observable *only* through the anchor, so
    seeding as if a foot were planted when the robot is actually hanging asserts
    an observation that was never made.
    """
    from .state import init_state

    return FilterCarry(
        state=init_state(build, params, q0),
        trusted_feet=jnp.zeros(build.n_anchors, dtype=jnp.float64),
    )


def step(
    carry: FilterCarry,
    sensors: SensorInputs,
    model: ModelInputs,
    build: JointKFBuild,
    params: JointKFParams,
) -> tuple[FilterCarry, TickDiagnostics]:
    """One filter tick.  Pure: `(carry, inputs) -> (carry, diagnostics)`.

    Order is predict, then encoders, then the gyro/anchor stack.  Encoders go
    first because they are the best-conditioned channel and are never gated in
    practice, so the gyro update linearises `J_ang(q̂)` at the freshest `q̂`
    available — the only place in the tick where the EKF's linearisation point
    can be improved for free.
    """
    state = carry.state

    # -- predict -----------------------------------------------------------
    F = predict_mod.build_transition(build, params)
    Q = process.build_process_noise(build, params, model.M,
                                    rotor=process.ROTOR_IN_MASS_MATRIX)
    state = predict_mod.predict(state, F, Q)

    qa_diag = jnp.diag(Q[build.n_joints:2 * build.n_joints,
                         build.n_joints:2 * build.n_joints]) / params.dt

    # -- encoder update ----------------------------------------------------
    H_enc = measure.encoder_jacobian(build)
    R_enc = measure.encoder_noise(build)
    state, enc_info = update_channel(state, H_enc, sensors.encoders, R_enc, params)

    # -- stacked gyro + anchor update --------------------------------------
    # `carry.trusted_feet` is the PREVIOUS tick's mask (module docstring §1).
    anchor = anchors_mod.anchor_block(
        build, params, model.anchor_jac,
        gyro_base=sensors.gyros[build.base_imu],
        qd_unfiltered=sensors.qd_unfiltered,
        trusted_feet=carry.trusted_feet,
    )
    stacked = measure.build_stacked(
        build, params, gyros=sensors.gyros,
        trusted_feet=carry.trusted_feet,
        J_rel=model.J_rel, R_rel=model.R_rel, anchor=anchor,
    )
    state, stk_info = update_channel(state, stacked.H, stacked.z, stacked.R, params)

    diagnostics = TickDiagnostics(
        encoder_nis=enc_info.nis,
        encoder_nis_per_joint=enc_info.nis_per_row,
        encoder_applied=enc_info.was_applied,
        stacked_nis=stk_info.nis,
        stacked_nis_per_row=stk_info.nis_per_row,
        stacked_applied=stk_info.was_applied,
        encoder_cond=enc_info.condition_proxy,
        stacked_cond=stk_info.condition_proxy,
        qa_max_diag=jnp.max(qa_diag),
        qa_tripped=(jnp.max(qa_diag) > params.qa_max).astype(jnp.float64),
        active_anchors=jnp.sum((carry.trusted_feet > 0.0).astype(jnp.float64)),
    )
    return FilterCarry(state=state, trusted_feet=sensors.contact), diagnostics


def update_channel(state, H, z, R, params):
    """Thin alias for the shared Joseph update, so every channel gates alike."""
    from .update import joseph_update

    return joseph_update(state, H, z, R, params)


def run(
    carry: FilterCarry,
    sensors: SensorInputs,
    model: ModelInputs,
    build: JointKFBuild,
    params: JointKFParams,
) -> tuple[FilterCarry, TickDiagnostics]:
    """Scan `step` over a trajectory.

    `sensors` and `model` are pytrees with a leading time axis.  The scan is over
    *values*, so the compiled graph is one tick regardless of trajectory length —
    which is the point of keeping every gate a mask.
    """
    def body(c, inputs):
        s, mdl = inputs
        return step(c, s, mdl, build, params)

    return jax.lax.scan(body, carry, (sensors, model))
