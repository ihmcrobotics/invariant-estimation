import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

def l2_velocity(v_est: jax.Array, R_est: jax.Array,
                v_true: jax.Array, R_true: jax.Array) -> jax.Array:
    """
    CoCo's objective (run 1): mean squared **body-frame** velocity error.

    The filter state is world-centric ``SE_{N+2}(3)``, so ``v_est`` and ``v_true``
    both arrive in WORLD frame. This rotates each into its OWN body frame before
    comparing::

        L = mean_k || R_est[k]^T v_est[k]  -  R_true[k]^T v_true[k] ||^2

    Each by its own attitude, and that is the whole point. Rotating BOTH sides by
    the same matrix would be a no-op -- rotations preserve norms, so
    ``||R^T(a - b)|| == ||a - b||`` -- and the loss would be identical to
    comparing in world frame. What makes the body-frame form a *different*
    objective is that a correct velocity paired with a WRONG attitude now
    produces loss, where the world-frame form scores it zero. Attitude error
    therefore reaches ContactNet's gradient, which is the paper's behaviour.

    Sanity check worth keeping in mind: when ``R_est == R_true`` this reduces
    exactly to the world-frame form, since the shared rotation cancels.

    Baseline only. Sigma reaches this loss solely through the Kalman gain, so
    only *ratios* of Sigma are constrained -- absolute scale is free, and an
    L2-trained network can pass RMSE while failing NEES. That gap is why
    `beta_nll` exists.

    Args:
        v_est: (..., 3) filter base velocity, world frame.
        R_est: (..., 3, 3) filter attitude.
        v_true: (..., 3) ground-truth base velocity, world frame
            (``SimSensorReader.truth()["v"]`` is already world frame).
        R_true: (..., 3, 3) ground-truth attitude.
    """
    # '...ji,...j->...i' is R^T v directly -- no transpose materialised per tick.
    b_est = jnp.einsum("...ji,...j->...i", R_est, v_est)
    b_true = jnp.einsum("...ji,...j->...i", R_true, v_true)
    return jnp.mean(jnp.sum((b_est - b_true) ** 2, axis=-1))

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
