from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import optax

from .network import ContactNetParams


class Metrics(NamedTuple):
    """Per-step scalars, reduced inside the JIT loop (returning aux whole forces a
    device-to-host transfer of the full InEKFOutputs every iteration).

    loss           -- progress metric for l2_velocity.
    grad_norm      -- global L2 norm PRIOR to clipping (the clip's tuning signal).
    nis_over_dof   -- contact update is the stacked 3N vector, NIS ~ chi^2(3N);
                      a calibrated filter sits at 1.0.
    applied_frac   -- fraction of ticks whose update passed the cond(S) gate.
    cond_proxy_max -- worst conditioning proxy in the batch, against cond_max.
    skipped        -- fraction of ticks whose update was skipped due to NaN or Inf.
    """
    loss: Array
    grad_norm: Array
    nis_over_dof: Array
    applied_frac: Array
    cond_proxy_max: Array
    skipped: Array


def decay_mask(params: ContactNetParams):
    """True where weight decay applies -- the trunk only. Decay on the head drags
    Sigma back toward init and fights the ln det calibration term."""
    return jax.tree.map(lambda _: True, params)._replace(
        head=jax.tree.map(lambda _: False, params.head)
    )


def make_optimizer(peak_lr, total_steps, warmup_steps, max_norm=1.0, weight_decay=0.0):
    """Clip-then-AdamW with a warmup-cosine schedule. Order matters: clip runs
    FIRST on the raw gradient (clipping after Adam clips unit-scale updates).
    Warmup is required: at init the head is zero so the trunk gradient is zero and
    only the head moves; Adam would take a full-size first step otherwise."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=warmup_steps, decay_steps=total_steps
    )
    return optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay, mask=decay_mask)
    )


def make_train_step(batch_loss, tx, dof: int):
    """Build the JIT train step from rollout.make_batch_loss's batch_loss, an optax
    transformation, and the measurement dimension dof = 3 * N_contacts."""
    @jax.jit
    def train_step(params, opt_state, batch, carry0=None):
        (loss, aux), grads = jax.value_and_grad(batch_loss, has_aux=True)(
            params, batch, carry0)
        grad_norm = optax.global_norm(grads)
        grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(grad_norm), g, 0.0), grads)
        skipped = jnp.where(jnp.isfinite(grad_norm), 0.0, 1.0)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        outputs, carry = aux
        d = outputs.contact_diagnostics
        metrics = Metrics(
            loss=loss,
            grad_norm=grad_norm,
            nis_over_dof=jnp.mean(d.nis) / dof,
            applied_frac=jnp.mean(d.applied),
            cond_proxy_max=jnp.max(d.condition_proxy),
            skipped=skipped
        )
        return params, opt_state, metrics, carry
    return train_step


def save_params(path: str, params: ContactNetParams) -> None:
    """Training checkpoint. Leaves in jax.tree.leaves order (stable for a NamedTuple)."""
    np.savez(path, *[np.asarray(x) for x in jax.tree.leaves(params)])


def load_params(path: str, like: ContactNetParams) -> ContactNetParams:
    """Rebuild params from a checkpoint, taking tree structure from `like`."""
    with np.load(path) as z:
        leaves = [jnp.asarray(z[k], dtype=jnp.float64) for k in z.files]
    return jax.tree.unflatten(jax.tree.structure(like), leaves)


def train(params, batch_loss, batches=None, *, batcher=None, peak_lr=1e-4,
          total_steps=1000, warmup_steps=100, max_norm=1.0, weight_decay=0.0,
          dof=6, log_every=10):
    """Run the training loop. Two modes:

    * batches  -- an iterable of rollout.Segment pytrees with a leading batch axis,
      each re-seeded from ground truth (run 1).
    * batcher  -- a dataset.ChainedBatcher, which carries (X_hat, P) between steps so
      a segment starts wherever the filter actually got to (run 2 onward).

    Exactly one of the two must be supplied.
    """
    if (batches is None) == (batcher is None):
        raise ValueError("supply exactly one of `batches` or `batcher`")

    tx = make_optimizer(peak_lr, total_steps, warmup_steps, max_norm, weight_decay)
    opt_state = tx.init(params)
    step = make_train_step(batch_loss, tx, dof)

    stream = (iter(batches) if batcher is None
              else (batcher.batch() for _ in range(total_steps)))

    history, reseeds = [], 0
    for i, item in enumerate(stream):
        if batcher is None:
            batch, carry0 = item, None
        else:
            batch, carry0 = item
        params, opt_state, metrics, carry = step(params, opt_state, batch, carry0)
        if batcher is not None:
            reseeds += batcher.update(carry)
        history.append(metrics)
        if log_every and i % log_every == 0:
            print(f"step {i:5d} loss {float(metrics.loss): .6e}"
                  f" |g| {float(metrics.grad_norm):.3e}"
                  f" NIS/dof {float(metrics.nis_over_dof):.3e}"
                  f" applied {float(metrics.applied_frac):.2e}"
                  + (f" reseeds {reseeds:4d}" if batcher is not None else ""))
    return params, opt_state, history
