from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

class NetworkLayer(NamedTuple):
    W: Array
    b: Array

class ContactNetParams(NamedTuple):
    trunk: tuple[NetworkLayer, ...] # hidden layers
    head: NetworkLayer # -> 6 cholesky elements

def _softplus_inv(y):
    """Inverse softplus, ``log(exp(y)-1)``."""
    return jnp.log(jnp.expm1(y))

# Positive parameterisations for diag(L). The choice is RECORDED per run and must be
# read back when loading a checkpoint: the head's raw outputs mean different things
# under each, so misreading it silently rescales Sigma_C.
#
#   softplus  the original. For r << 0 it IS exp (relative sensitivity dlogL/dr = 1),
#             so it is well behaved at the tight end -- but it goes LINEAR above zero,
#             where dlogL/dr decays as 1/r. Reaching the analytic swing value
#             (per-axis std 10) needs r=+10, at sensitivity 0.10.
#   exp       dlogL/dr = 1 everywhere. The same target is r=+2.30 at sensitivity 1.00.
#
# Sigma_C spans ~1e10 between stance and swing (analytic: tr 3e-8 -> 3e2), which is a
# scale parameter, so the log parameterisation is the natural one. Measured on the
# softplus run: the head achieved 3.6 of the 19.2 raw units that span requires (19%).
DIAG_PARAMS = ("softplus", "exp")


def _diag_inv(y, kind):
    """Raw head-bias value that makes diag(L) == y under `kind`."""
    if kind == "softplus":
        return _softplus_inv(y)
    if kind == "exp":
        return jnp.log(y)
    raise ValueError(f"unknown diag_param {kind!r}; expected one of {DIAG_PARAMS}")


def _diag_fwd(o, kind):
    if kind == "softplus":
        return jax.nn.softplus(o)
    if kind == "exp":
        return jnp.exp(o)
    raise ValueError(f"unknown diag_param {kind!r}; expected one of {DIAG_PARAMS}")


def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def init(
    key,
    d_in: int,
    widths: tuple[int, ...],
    sigma_0: float,
    eps: float,
    diag_param: str = "softplus",
) -> ContactNetParams:
    """Initialize the network *at* the analytical filter. Runs once, on the host, not under jit."""
    #WARNING: this could be a problem.
    if d_in <= 0:
        raise ValueError(f"d_in must be positive, got {d_in}")
    if not widths or any(w <= 0 for w in widths):
        raise ValueError(f"widths must be non-empty and positive, got {widths}")
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

    # Zero weights => the output is the bias for any input.  The diagonal entries
    # invert the softplus eps so diag(L) = sigma_0; the off-diagonals stay zero,
    # so Sigma_C is diagonal at initialization, matching the filter's isotropic
    # assumption.
    head = NetworkLayer(
        W = jnp.zeros((6, widths[-1])),
        b = jnp.concatenate([jnp.full(3, _diag_inv(sigma_0 - eps, diag_param)),
                             jnp.zeros(3)]),
    )

    params = ContactNetParams(trunk=trunk, head=head)

    # I8 at the entry point.
    bad = [x.dtype for x in jax.tree.leaves(params) if x.dtype != jnp.float64]
    if bad:
        raise TypeError(f"float64 is required (is jax_enable_x64 set?); got {bad}")

    return params

def forward(params: ContactNetParams, x: jax.Array, eps: float,
            diag_param: str = "softplus") -> jax.Array:
    """One contact's ``(d_in,) = H*F`` normalized feature window → its ``(3,3)`` Cholesky factor.

    ``eps`` is the softplus floor on the diagonal, which makes ``L L^T`` SPD (not
    merely PSD) by construction.
    """
    h = x
    for layer in params.trunk:
        h = gelu(layer.W @ h + layer.b)
    o = params.head.W @ h + params.head.b

    # Lower-triangular L: softplus with a floor keeps the diagonal strictly
    # positive, so L stays full rank even with unconstrained off-diagonals.
    d = _diag_fwd(o[:3], diag_param) + eps
    L = jnp.array(
        [
            [d[0], 0.0, 0.0],
            [o[3], d[1], 0.0],
            [o[4], o[5], d[2]]
        ]
    )
    return L # the filter's inputs take Cholesky factors, not covariances
