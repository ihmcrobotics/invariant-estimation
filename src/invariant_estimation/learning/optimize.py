"""Small offline optimizer; no ContactNet, simulator, or deployment dependency."""
from dataclasses import dataclass
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def body_velocity_l2(rotation, velocity, truth_rotation, truth_velocity):
    """Mean squared body-frame velocity-vector error (not per-component MSE).

    Arrays end in (3,3)/(3,); leading dimensions are time and optional batch.
    Caller must remove invalid MoCap samples before constructing the loss.
    """
    estimated = jnp.einsum("...ji,...j->...i", rotation, velocity)
    truth = jnp.einsum("...ji,...j->...i", truth_rotation, truth_velocity)
    return jnp.mean(jnp.sum((estimated - truth)**2, axis=-1))


@dataclass(frozen=True)
class FitResult:
    theta: object
    losses: tuple[float, ...]
    best_step: int


def fit_scalars(loss_fn: Callable, initial_theta, *, steps=100, learning_rate=0.03):
    """Full-batch Adam, return best finite TRAINING iterate, no test-set access.

    A caller should use a validation split to select hyperparameters/checkpoints.
    Numeric noise recovery is not guaranteed by L2: gains can be non-identifiable.
    Nonfinite loss/gradient aborts loudly rather than exporting plausible junk.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    theta = jnp.asarray(initial_theta, dtype=jnp.float64)
    if theta.ndim != 1 or not np.isfinite(np.asarray(theta)).all():
        raise ValueError("initial_theta must be a finite vector")
    value_grad = jax.jit(jax.value_and_grad(loss_fn))
    m, v = jnp.zeros_like(theta), jnp.zeros_like(theta)
    history, best_loss, best_theta, best_step = [], math.inf, theta, 0
    for step in range(steps + 1):
        loss, grad = value_grad(theta)
        loss = float(loss)
        if not math.isfinite(loss) or not np.isfinite(np.asarray(grad)).all():
            raise FloatingPointError(f"nonfinite loss/gradient at step {step}")
        history.append(loss)
        if loss < best_loss:
            best_loss, best_theta, best_step = loss, theta, step
        if step == steps:
            break
        m = 0.9*m + 0.1*grad
        v = 0.999*v + 0.001*grad**2
        t = step + 1
        theta = theta - learning_rate*(m/(1-0.9**t))/(jnp.sqrt(v/(1-0.999**t)) + 1e-8)
    return FitResult(best_theta, tuple(history), best_step)
