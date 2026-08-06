from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp

from ..inEKF.filter import InEKFInputs, init_carry, make_step
from ..inEKF.state import InEKFState
from .losses import l2_velocity, beta_nll_from_diagnostics, pose_l2
from .network import ContactNetParams, forward

class Segment(NamedTuple):
    """
    One training sample: `L` consecutive filter tasks.

    A sesgment is a **trajectory**, not a window. `H` (history per evaluation)
    and `L` are INDEPENDENT.

    ``inputs`` has a leading time axis of length ``L`` on each leaf.
        Its ``contact_chol`` field - the stance anchor **process noise**,
        is  placeholder, and it is overwritten from the output of the network.

    ``windows`` is ``(L, N_c, H, F)`` normalized feature windows; 
    ``state0`` is the initial state of the filter at the start of the segment.
    ``v_true`` is the groudn truth base velocity in world frame.
    ``R_true`` is the rotation matrix of ground truth attitude, meant to convert
        the world frame base velocity to body frame.
    ``p_true`` is the ground truth base position in world frame, used by the
        segment-relative position loss (`losses.l2_position`).
    """
    inputs: InEKFInputs
    windows: Array
    state0: InEKFState
    v_true: Array
    R_true: Array
    p_true: Array


def contact_factors(
    params: ContactNetParams,
    windows :Array,
    eps: float
) -> Array:
    """
    Network over every tick and contact, all at once.

    ``(L, N_c, H, F) -> (L, N_c, 3,3)``

    The network only sees sensor history, never the filter state.

    The flatten is ``(H, F) -> H * F ``.
    """
    L, N_c = windows.shape[0], windows.shape[1]
    flat = windows.reshape(L, N_c, -1) # last dim is D_in = H * F, the flattening
    over_contacts = jax.vmap(forward, in_axes=(None, 0, None))
    over_time = jax.vmap(over_contacts, in_axes=(None, 0, None))
    return over_time(params, flat, eps)

# objective -> (use_pos, use_ori) for the composite pose objectives. Resolved at
# build time (outside the traced region) so the branch never enters the graph.
_POSE_OBJECTIVES = {
    "l2_vel_pos": (True, False),
    "l2_vel_ori": (False, True),
    "l2_vel_pos_ori": (True, True),
}
VALID_OBJECTIVES = ("beta_nll", "l2_velocity") + tuple(_POSE_OBJECTIVES)


def make_segment_loss(ekf, kinematics, eps, beta=0.5, objective="l2_velocity",
                      remat=True, w_pos=0.0, w_ori=0.0):
    """Build the per-segment loss: ``(params, segment) -> (loss, (outputs, carry))``.

    A factory matching `make_step`: `ekf`, `kinematics` and the scalars are static
    and closed over, so the callable is differentiable in `params` and vmappable
    over segments.

    ``objective`` is selected at build time, outside the traced region, so it puts
    no branch in the graph. ``l2_velocity`` reproduces CoCo-InEKF; ``beta_nll`` is
    the Seitzer innovation NLL; ``l2_vel_pos`` / ``l2_vel_ori`` / ``l2_vel_pos_ori``
    add the segment-relative position and/or orientation terms
    (`losses.pose_l2`), weighted by the run-frozen ``w_pos`` / ``w_ori``. ``remat``
    wraps the scan body in `jax.checkpoint` (``prevent_cse=False`` is the correct
    setting under `scan`).
    """
    step = make_step(ekf, kinematics)
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"unknown objective {objective}")
    use_pos, use_ori = _POSE_OBJECTIVES.get(objective, (False, False))

    def segment_loss(params: ContactNetParams, segment: Segment, carry0=None):
        L_c = contact_factors(params, segment.windows, eps)
        inputs = segment.inputs._replace(contact_chol=L_c)
        c0 = init_carry(segment.state0) if carry0 is None else carry0
        carry, outputs = jax.lax.scan(step, c0, inputs)
        if objective == "beta_nll":
            d = outputs.contact_diagnostics
            dof = 3 * outputs.state.d.shape[-2]
            loss = beta_nll_from_diagnostics(
                d.nis, d.logdet_s, d.applied, beta, dof
            )
        elif objective == "l2_velocity":
            loss = l2_velocity(
                outputs.state.v, outputs.state.R, segment.v_true, segment.R_true
            )
        else:
            loss = pose_l2(
                outputs.state.v, outputs.state.R, outputs.state.p,
                segment.v_true, segment.R_true, segment.p_true,
                w_pos=w_pos, w_ori=w_ori, use_pos=use_pos, use_ori=use_ori,
            )
        return loss, (outputs, carry)
    return segment_loss

def make_warm_in(ekf, kinematics, sigma_0: float | None = None):
    r"""``(state0, inputs) -> carry``: run the filter forward without training on it.

    `dataset.ChainedBatcher` uses this to grow a freshly seeded chain's error to
    its natural level before the chain contributes a gradient.

    ``sigma_0=None`` (the default) warms in on the recorded heuristic
    ``inputs.contact_chol``, i.e. on the shipped filter, for the same reason
    `online.make_provider`'s fallback defers to it: this socket has no
    "reproduces the shipped filter" constant to hold, and a constant at the
    *stance* value is `freeze_contact_chol` (see there for the measurement),
    which would grow the chain's error under a filter the trained network never
    runs inside.

    Passing a float restores the old behaviour, broadcasting ``σ₀·I₃`` over the
    warm-in slice. Note that `ContactNetConfig.sigma_0` is a **measurement**-socket
    number; that argument does not transfer, so reusing it here is a deliberate
    choice and not a default.

    Built here rather than in `dataset` so that module keeps its "no MJX, no
    estimator build" property.
    """
    step = make_step(ekf, kinematics)

    @jax.jit
    def warm_in(state0, inputs: InEKFInputs):
        if sigma_0 is not None:
            inputs = inputs._replace(contact_chol=jnp.broadcast_to(
                sigma_0 * jnp.eye(3, dtype=jnp.float64), inputs.contact_chol.shape))
        carry, _ = jax.lax.scan(step, init_carry(state0), inputs)
        return carry

    return warm_in

def make_batch_loss(*args, **kwargs):
    """`make_segment_loss` vmapped over a batch of `B` segments.

    ``in_axes=(None, 0)``: one shared weight set, one independent trajectory per
    batch element. A batch of `B` segments is ``B * L * N_c`` forward passes but
    only **B independent samples** -- size the batch by this, not by the forward
    pass count.
    """
    segment_loss = make_segment_loss(*args, **kwargs)

    def batch_loss(params: ContactNetParams, batch: Segment, carry0=None):
        if carry0 is None:
            losses, aux = jax.vmap(segment_loss, in_axes=(None, 0))(params, batch)
        else:
            losses, aux = jax.vmap(segment_loss, in_axes=(None, 0, 0))(
                params, batch, carry0)
        return jnp.mean(losses), aux

    return batch_loss
