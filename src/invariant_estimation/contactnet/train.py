from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import optax

from .network import ContactNetParams

class Metrics(NamedTuple):
    """
    Per-step scalars, reduced *inside* the JIT loop.

    `aux` from the rollout is the full `InEKFOutputs` -- `state.P` alone is
    (B, L, 3N+9, 3N+9), ~7 MB at B=32, L=128, N=2. Returning it whole forces a
    device-to-host transfer every iteration. Reduce here, and just give the scalar.

    Attributes
    ----------
    loss: Array, scalar
        For `l2_velocity` this is a progress metric. For `beta_nll` it is NOT.
    grad_norm: Array, scalar
        Global L2 norm of the gradient *prior* to clamping. Logged, as clipping is the
        stability "lever" and cannot be tuned against an unobserved number. Always under 
        `max_norm` => the clip is inert; always over => learning rate is the problem, not clipping.
    nis_over_dof: Array, scalar
        Mean NIS over the measurement dimension. The contact update is the stacked 3N vector,
        so NIS ~ chi^2(3N) and a *calibrated* filter sits at 1.0; above is overconfident, below
        is conservative. This is beta-NLL's real progress metric.
    applied_frac: Array, scalar
        Fraction of ticks whose update passed the cond(S) gate. Below 1.0 means updates are being
        skipped and the loss is scoring innovations that never corrected the state at all.
    cond_proxy_max: Array, scalar
        Worst conditioning proxy in the batch, against `cond_max`.
    """
    loss: Array
    grad_norm: Array
    nis_over_dof: Array
    applied_frac: Array
    cond_proxy_max: Array

def decay_mask(params: ContactNetParams):
    """
    True where weight decay applies - in the trunk only.

    Decay on the head drags `Sigma` back toward the initialization and fights
    the `ln det` term that does the calibration for beta-NLL. `head` is a subtree,
    so `_replace` does this directly, without using strings.
    """
    return jax.tree.map(lambda _: True, params)._replace(
        head=jax.tree.map(lambda _: False, params.head)
    )

def make_optimizer(
    peak_lr,
    total_steps,
    warmup_steps,
    max_norm=1.0,
    weight_decay=0.0
):
    """
    The order here is important: `clip_by_global_norm` runs **first**, on the raw gradient.
    `optax.chain` applies left to right, and clipping *after* Adam would clip already normalized
    updates -- Adam's output is roughly unit scale per parameter by construction, so that clip would
    do nothing.

    Warmup is required. At the initialization, the head weights are *zero*, so the trunk's gradient
    is *exactly* zero and only the head moves first. Adam normalizes per parameter, so that first head
    update has magnitude ~lr no matter how small the gradient is, and a full size first step discards the
    "refinement from a known good point" that we want to use.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=warmup_steps, decay_steps=total_steps
    )
    
    return optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay, mask=decay_mask)
    )

def make_train_step(batch_loss, tx, dof: int):
    """
    Build the JIT train step.

    Parameters
    ----------
    batch_loss: callable
        `(params, batch) -> (lOss, aux)` from `rollout.make_batch_loss`
    tx: optax.GradientTransformation
    dof: int
        Measurement dimension `3 * N_contacts`, for `nis_over_dof`.
    """
    @jax.jit
    def train_step(params, opt_state, batch):
        #has_aux=True nests as ((loss, aux),grads)
        (loss, aux), grads = jax.value_and_grad(batch_loss, has_aux=True)(params,batch)
        grad_norm = optax.global_norm(grads)

        # AdamW needs `params`: decoupled decay is computed against them.
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        d = aux.contact_diagnostics
        # Plain mean, never a NaN: a NaN reaching the *metrics* is a good thing
        # to see, but not something to use as noise.
        metrics = Metrics(
            loss=loss,
            grad_norm=grad_norm,
            nis_over_dof=jnp.mean(d.nis) / dof,
            applied_frac=jnp.mean(d.applied),
            cond_proxy_max=jnp.max(d.condition_proxy)
        )
        return params, opt_state, metrics
    return train_step

def save_params(path: str, params: ContactNetParams) -> None:
    """
    Training checkpoint - **not** the Java artifact.

    Leaves go in `jax.tree.leaves` order, which is structural and stable for a
    NamedTuple, so `load_params` can rebuild the tree from just a reference.
    """
    np.savez(path, *[np.asarray(x) for x in jax.tree.leaves(params)])

def load_params(path: str, like: ContactNetParams) -> ContactNetParams:
    """
    Rebuild params from a checkpoint, taking tree structure from `like`.
    """
    with np.load(path) as z:
        leaves = [jnp.asarray(z[k], dtype=jnp.float64) for k in z.files]
    return jax.tree.unflatten(jax.tree.structure(like), leaves)

def train(
    params,
    batch_loss,
    batches,
    *,
    peak_lr=1e-4,
    total_steps=1000,
    warmup_steps=100,
    max_norm=1.0,
    weight_decay=0.0,
    dof=6,
    log_every=10
):
    """
    Run the training loop over an iterable of batches.

    `batches` yields `rollout.Segment` pytrees with a leading batch axis.
    This module deliberately owns no data loading (belongs to `features.py` and a loader),
    which is what lets the finite difference test drive a four-tick rollout without this.
    """
    tx = make_optimizer(peak_lr, total_steps, warmup_steps, max_norm, weight_decay)
    opt_state = tx.init(params)
    step = make_train_step(batch_loss, tx, dof)

    history = []
    for i, batch in enumerate(batches):
        params, opt_state, metrics = step(params, opt_state, batch)
        history.append(metrics)
        if log_every and i % log_every == 0:
            print(f"step {i:5d} loss {float(metrics.loss): .6e}"
                    f"|g| {float(metrics.grad_norm):.3e}"
                    f"NIS/dof {float(metrics.nis_over_dof):.3e}"
                    f"applied {float(metrics.applied_frac):.2e}")
    return params, opt_state, history
