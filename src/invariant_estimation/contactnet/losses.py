import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

def l2_velocity(v_est: jax.Array, v_true: jax.Array) -> jax.Array:
    """
    CoCo objective for mean squared velocity error.

    Baseline only. NOTE: you need to feed in the BODY FRAME velocities here,
    otherwise this loss function is incorrect according to the CoCo paper.
    """
    return jnp.mean(jnp.sum((v_est - v_true) **2, axis=-1))

def gaussian_nll(nu: jax.Array, S: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Gaussian negative log likelihood of one innovation, via Cholesky factorization.

    Returns ``(nll, logdet_S)``; the caller reuses ``logdet_S`` for the beta
    reweight rather than recomputing a determinant on a second numerical path.

    The constant ``0.5 * k * log(2*pi)`` is dropped, as it has no parameter
    dependence, so it contributes nothing but a constant scaling factor.

    Args:
        nu: (3,) innovation.
        S: (3,3) innovation covariance, S = H P H^T + N_bar
    """
    S = 0.5 * (S + S.T)
    L = jnp.linalg.cholesky(S)

    z = solve_triangular(L, nu, lower=True)
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

    return 0.5 * (jnp.sum(z**2) + logdet_S), logdet_S

def beta_nll(nu: jax.Array, S: jax.Array, beta: float = 0.5) -> jax.Array:
    """
    Beta-NLL for one innovation.

    ``logdet`` is the term that constrains **absolute** scale: it balances the
    quadratic term at ``S ~ E[nu nu^T]``, i.e. calibration by construction. This is
    what L2 for velocity does not have.

    ``beta`` interpolates: 0 is pure NLL, 1 gives L2 like gradient magnitudes 
    while keeping calibration - 0.5 is the default value.

    Args:
        nu: (3,) innovation.
        S: (3,3) innovation covariance.
        beta: reweight exponent.
    """
    nll, logdet_S = gaussian_nll(nu, S)

    # det(S)**beta, built from the logdet already built above.
    # The stop_gradient here is important, as this cancels the S^(-1) gradient *shrinkage*
    # that would drive plain NLL to just inflate the variance in order to escape hard samples. Gradient flowing
    # *through* it restores the pathology.
    weight = jax.lax.stop_gradient(jnp.exp(beta * logdet_S))

    return weight * nll

def beta_nll_from_diagnostics(nis: jax.Array, logdet_S: jax.Array, beta: float = 0.5) -> jax.Array:
    """
    Beta-NLL read off UpdateDiagnostics

    `inEKF.correct.linear_update` already yields a Cholesky factored S to produce both scalars,
    so this needs no factorization here, and can just return the full loss. 
    """
    nll = 0.5 * (nis + logdet_S)
    weight = jax.lax.stop_gradient(jnp.exp(beta * logdet_S))
    return weight * nll
