import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

def l2_velocity(
    v_est: Array,
    R_est: Array,
    v_true: Array,
    R_true: Array, #TODO: should this be `jaxlie.SO3`?
) -> Array:
    """CoCo's objective (run 1): mean_k || R_est[k]^T v_est[k] - R_true[k]^T v_true[k] ||^2.

    Both velocities arrive in WORLD frame; each is rotated into its OWN body frame,
    so a correct velocity paired with a WRONG attitude still produces loss (attitude
    error reaches ContactNet's gradient). `rollout.make_segment_loss` calls this on
    (L, ...) segment arrays, so the einsum/reduction MUST be batched -- '...ji,...j'
    and mean-over-time of the per-tick squared error, matching the validated
    reference losses.py. (take-two shipped a non-batched 'ji,j' + linalg.norm that
    raised on the batched call.)
    """
    body_estimated = jnp.einsum("...ji,...j->...i", R_est, v_est)
    body_true = jnp.einsum("...ji,...j->...i", R_true, v_true)
    return jnp.mean(jnp.sum((body_estimated - body_true) ** 2, axis=-1))

#TODO: add beta NLl once ready.
