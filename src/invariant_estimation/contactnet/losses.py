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

def beta_nll_from_diagnostics(
    nis,
    logdet_S,
    applied,
    beta,
    dof
):
    """
    Per-tick Gaussian innovation NLL, beta-weighted (Seitzer et al. 2022, "On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks"). 
    This is averaged over applied+finite ticks, with NIS and the logdet_S defined as:
        NIS = nu^T S^{-1} nu, logdet_S = log(det(S)), where nu is the innovation and S is the innovation covariance.
    These are both from the same cholesky in linear_update, with DOF = 3 * N_c
    """
    finite = (applied > 0) & jnp.isfinite(nis) & jnp.isfinite(logdet_S)
    nis_c = jnp.where(finite, nis, 0.0)
    logdet_c = jnp.where(finite, logdet_S, 0.0)

    per_tick = 0.5 * (nis_c + logdet_c)
    weight = jax.lax.stop_gradient(
        jnp.exp((beta / dof * logdet_c))
    )
    term = jnp.where(finite, weight * per_tick, 0.0)

    denom = jnp.maximum(jnp.sum(finite), 1.0)
    return jnp.sum(term) / denom
