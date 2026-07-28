from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp

from ..inEKF.filter import InEKFInputs, init_carry, make_step
from ..inEKF.state import InEKFState
from .losses import beta_nll_from_diagnostics, l2_velocity
from .network import ContactNetParams, forward

class Segment(NamedTuple):
    """
    One training sample, for `L` consecutive filter ticks.

    A segment is a *trajectory*, not a window. `H` (history per evaluation) 
    and `L` are independent axes - don't mix them up!

    Attributes
    ----------
    inputs : InEKFInputs
        Every leaf carries a leading time axis of length ``L``. Its
        ``contact_meas_chol`` field is a **placeholder** -- `segment_loss`
        overwrites it with the network's actual output. Its ``contact_chol`` field is
        the *stance anchor* process noise, held at a constant for training so
        the network can't lean on the sim's ground truth stance anchor.

    windows: Array, shape (L, N_c, H, F)
        Per-tick, per-contact normalized feature windows.

    state0: InEKFState
        Filter state at segment start, reseeded from ground truth.

    v_true: Array, shape (L, 3)
        Ground-truth base velocity, **world frame** (`SimSensorReader.truth()["v"]`
        already is), for the L2 baseline objective.

    R_true: Array, shape (L, 3, 3)
        Ground-truth attitude. Needed because `losses.l2_velocity` compares in
        each side's OWN body frame -- with a single shared rotation the whole
        conversion would cancel and be a no-op. See that docstring.
    """
    inputs: InEKFInputs
    windows: Array
    state0: InEKFState
    v_true: Array
    R_true: Array

def contact_factors(params: ContactNetParams, windows: Array, eps: float) -> Array:
    """
    Network over every tick and contact at once: (L, N_c, H, F) -> (L, N_c, 3, 3)

    The network only sees sensor history, never the actual filter state, so nothing
    depends on the scan carry and the whole time axis evaluates in one batched pass, 
    rather than `L` sequential ones.

    The flatten is ``(H, F) -> H * F`` row major, i.e. **H-major**: history index
    outer, channel inner. That ordering is a convention that is shared with the Java
    port and a few files of the python port here, namely `features.py, normalize.py`,
    and `export.py`.
    """
    L, N_c = windows.shape[0], windows.shape[1]
    flat = windows.reshape(L, N_c, -1) # (L, N_c, D_in)
    over_contacts = jax.vmap(forward, in_axes=(None, 0, None)) # shared weights
    over_time = jax.vmap(over_contacts, in_axes=(None, 0, None))
    return over_time(params, flat, eps)

def make_segment_loss(ekf, kinematics, eps, beta = 0.5, objective="beta_nll", remat=True):
    """
    Build the per-segment loss.

    A "factory", matching `make_step`: `ekf`, `kinematics` and the scalars are all static
    and get closed over, so the callable only takes ``(params, segment)`` -- this is
    differentiable in `params` and vmappable over segments.

    Parameters
    ----------
    objective : {"beta_nll", "l2_velocity"}
        Run 1 reproduces the original CoCo-InEKF work with ``l2_velocity``; run 2 onward
        uses the new beta_nll. This is selected at build time, and doesn't put a branch in
        the computation graph, and stays outside the JIT-traced region.
    remat : bool
        Wrap the scan body in `jax.checkpoint`. ``prevent_cse=False`` is the
        setting for a remat'd function under `scan`.
    """
    if objective not in ("beta_nll", "l2_velocity"):
        raise ValueError(f"Unknown objective {objective!r}")

    step = make_step(ekf, kinematics)
    if remat:
        step = jax.checkpoint(step, prevent_cse=False)

    def segment_loss(params: ContactNetParams, segment: Segment, carry0=None):
        # Network first, over the full segment - see `contact_factors`
        L_c = contact_factors(params, segment.windows, eps)

        # The one field ContactNet has - the inputs:
        inputs = segment.inputs._replace(contact_meas_chol=L_c)

        # `carry0=None` re-seeds from `segment.state0` (run-1 behaviour, and what
        # the standalone tests use).  `ChainedBatcher` passes the previous
        # segment's final carry instead, so the segment starts at the error the
        # filter actually accumulated rather than at zero -- see dataset.py (f).
        c0 = init_carry(segment.state0) if carry0 is None else carry0
        carry, outputs = jax.lax.scan(step, c0, inputs)

        d = outputs.contact_diagnostics
        if objective == "beta_nll":
            per_tick = beta_nll_from_diagnostics(d.nis, d.logdet_S, beta)
            loss = jnp.mean(per_tick) #NOTE: mean handled outside of beta-NLL, inside of L2.
        else:
            # Body-frame, each side by its OWN attitude -- see `l2_velocity`.
            loss = l2_velocity(
                outputs.state.v, outputs.state.R, segment.v_true, segment.R_true
            )
        return loss, (outputs, carry)

    return segment_loss


def make_warm_in(ekf, kinematics, sigma_0: float):
    r"""``(state0, inputs) -> carry``: run the filter forward without training on it.

    `dataset.ChainedBatcher` uses this to grow a freshly seeded chain's error to
    its natural level before the chain contributes a gradient.  ``Σ_C`` is held
    at the network's initialization ``σ₀²I`` — exact at step 0, and an
    approximation at later re-seeds that decays over an episode ten times longer
    than the warm-in.

    Built here rather than in `dataset` so that module keeps its "no MJX, no
    estimator build" property.
    """
    step = make_step(ekf, kinematics)

    @jax.jit
    def warm_in(state0, inputs: InEKFInputs):
        L_c = jnp.broadcast_to(
            sigma_0 * jnp.eye(3, dtype=jnp.float64),
            inputs.contact_meas_chol.shape)
        carry, _ = jax.lax.scan(
            step, init_carry(state0), inputs._replace(contact_meas_chol=L_c))
        return carry

    return warm_in

def make_batch_loss(*args, **kwargs):
    """
    `make_segment_loss` vmapped over a batch of `B` segments.

    ``in_axes=(None, 0)``: one shared weight set, one independent trajectory per
    batch element. A batch of `B` segments is`B * L * N_c` forward passes, but
    only **B independent samples** - size the batch by this, not by the forward pass
    count.
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
