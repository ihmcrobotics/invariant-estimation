import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

def l2_velocity(v_est: jax.Array, R_est: jax.Array,
                v_true: jax.Array, R_true: jax.Array) -> jax.Array:
    """CoCo's objective (run 1): ``mean_k || R_est[k]^T v_est[k] - R_true[k]^T v_true[k] ||^2``.

    The filter state is world-centric ``SE_{N+2}(3)``, so both velocities arrive
    in WORLD frame; each is rotated into its OWN body frame before comparing, and
    that is the whole point. Rotating BOTH sides by the same matrix would be a
    no-op -- ``||R^T(a - b)|| == ||a - b||`` -- and identical to comparing in
    world frame. The body-frame form is a *different* objective because a correct
    velocity paired with a WRONG attitude now produces loss, so attitude error
    reaches ContactNet's gradient. When ``R_est == R_true`` it reduces exactly to
    the world-frame form.

    Baseline only. Sigma reaches this loss solely through the Kalman gain, so
    only *ratios* of Sigma are constrained -- absolute scale is free, and an
    L2-trained network can pass RMSE while failing NEES. That gap is why
    `beta_nll` exists.
    """
    # '...ji,...j->...i' is R^T v directly -- no transpose materialised per tick.
    b_est = jnp.einsum("...ji,...j->...i", R_est, v_est)
    b_true = jnp.einsum("...ji,...j->...i", R_true, v_true)
    return jnp.mean(jnp.sum((b_est - b_true) ** 2, axis=-1))

def gaussian_nll(nu: jax.Array, S: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Gaussian NLL of one ``(3,)`` innovation under ``S = H P H^T + N_bar``, via Cholesky.

    Returns ``(nll, logdet_S)``; the caller reuses ``logdet_S`` for the beta
    reweight rather than recomputing a determinant on a second numerical path.
    The parameter-independent ``0.5 * k * log(2*pi)`` is dropped.
    """
    S = 0.5 * (S + S.T)
    L = jnp.linalg.cholesky(S)

    z = solve_triangular(L, nu, lower=True)
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

    return 0.5 * (jnp.sum(z**2) + logdet_S), logdet_S

def beta_nll(nu: jax.Array, S: jax.Array, beta: float = 0.5) -> jax.Array:
    """Beta-NLL for one innovation.

    ``logdet`` is the term that constrains **absolute** scale: it balances the
    quadratic term at ``S ~ E[nu nu^T]``, i.e. calibration by construction, which
    L2 velocity does not have. ``beta`` interpolates: 0 is pure NLL, 1 gives
    L2-like gradient magnitudes while keeping calibration.
    """
    nll, logdet_S = gaussian_nll(nu, S)

    # det(S)**beta, built from the logdet above.  The stop_gradient cancels the
    # S^(-1) gradient *shrinkage* that would drive plain NLL to just inflate the
    # variance to escape hard samples; gradient flowing *through* it restores the
    # pathology.
    weight = jax.lax.stop_gradient(jnp.exp(beta * logdet_S))

    return weight * nll

def beta_nll_from_diagnostics(nis: jax.Array, logdet_S: jax.Array, beta: float = 0.5) -> jax.Array:
    """Beta-NLL read off `UpdateDiagnostics` — `inEKF.correct.linear_update` already
    factored ``S`` to produce both scalars, so no factorization is needed here."""
    nll = 0.5 * (nis + logdet_S)
    weight = jax.lax.stop_gradient(jnp.exp(beta * logdet_S))
    return weight * nll
