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
    body_estimated = jnp.einsum("ji,j->i", R_est, v_est)
    body_true = jnp.einsum("ji,j->i", R_true, v_true)
    return jnp.linalg.norm(body_estimated - body_true)

#TODO: add beta NLl once ready.
