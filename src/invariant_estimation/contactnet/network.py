from typing import NamedTuple

import jax
import jax.numpy as jnp

class NetworkLayer(NamedTuple):
    W: jax.Array # (out, in)
    b: jax.Array # (out,)

class ContactNetParams(NamedTuple):
    trunk: tuple[NetworkLayer, ...] # hidden layers
    head: NetworkLayer # -> 6 cholesky elements

def _softplus_inv(y):
    """
    Inverse of the softplus function, defined as ``log(exp(y)-1)``.
    """
    return jnp.log(jnp.expm1(y))

def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def init(
    key,
    d_in: int,
    widths: tuple[int, ...],
    sigma_0: float,
    eps: float
) -> ContactNetParams:
    """
    Initialize the network *at* the analytical filter.
    """
    # Build time: init runs once, on the host, and not JIT.
    if d_in <= 0:
        raise ValueError(f"d_in must be positive, got {d_in}")
    if not widths or any(w <= 0 for w in widths):
        raise ValueError(f"widtghs must be non-empty and positive, got {widths}")
    if not sigma_0 > eps:
        # softplus inverse is nan otherwise, so this is a live failure mode.
        raise ValueError(f"need sigma_0 > eps, got sigma_0={sigma_0}, eps={eps}")

    sizes = (d_in, *widths)
    keys = jax.random.split(key, len(widths))

    trunk = tuple(
        NetworkLayer(
            W = jax.random.normal(k, (n_out, n_in)) * jnp.sqrt(2.0 / n_in),
            b = jnp.zeros(n_out),
        )
        for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:])
    )

    # Zero weights => the output is the bias for any input.
    # The diagonal entries invert the softplus eps so that diag(L) = sigma_0.
    # The off diagonals stay zero, so the Sigma_C is diagonal at  initialization, which makes sense.
    # The filter assumese an isotropic covariance, so this lines up with that.
    head = NetworkLayer(
        W = jnp.zeros((6, widths[-1])),
        b = jnp.concatenate([jnp.full(3, _softplus_inv(sigma_0 - eps)), jnp.zeros(3)]),
    )

    params = ContactNetParams(trunk=trunk, head=head)

    # At the entry point, let's check that we are maintaining float64.
    bad = [x.dtype for x in jax.tree.leaves(params) if x.dtype != jnp.float64]
    if bad:
        raise TypeError(f"float64 is reqired (is jax_enable_x64 set?); got {bad}")

    return params

def forward(params: ContactNetParams, x: jax.Array, eps: float) -> jax.Array:
    """
    Map one contact's flatten feature window to its covariance.

    Args:
        params: network parameters.
        x: (d_in, ) flattened, already normalized feature window, H * F.
        eps: softplus floor on the Cholesky diagonal.

    Returns:
        L, (3,3), yields symmetric positive definite covariance (SPD) by construction.
    """
    h = x
    for layer in params.trunk:
        h = gelu(layer.W @ h + layer.b)
    o = params.head.W @ h + params.head.b

    # Lower triangular L: softplus withn floor on the diagonal keeps it strictly postiive.
    # The off diagonals stay unconstrained, but L is kept full rank, so Sigma_C is SPD, not PSD
    d = jax.nn.softplus(o[:3]) + eps
    L = jnp.array(
        [
            [d[0], 0.0, 0.0],
            [o[3], d[1], 0.0],
            [o[4], o[5], d[2]]
        ]
    )
    # We put the actual output of the head into the lower triangular bit,
    # and the diagonal is the softplus gate.
    # return L @ L.T # return the actual covariance
    return L # inputs take cholesky factors

