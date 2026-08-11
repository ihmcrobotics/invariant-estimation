import math
from dataclasses import dataclass
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
# read back when loading a checkpoint (`checkpoint.config_for_checkpoint`): the head's
# raw outputs mean different things under each, so misreading it silently rescales
# Sigma_C.
#
#   softplus     the original. For r << 0 it IS exp (relative sensitivity dlogL/dr = 1),
#                so it is well behaved at the tight end -- but it goes LINEAR above
#                zero, where dlogL/dr decays as 1/r. Reaching the analytic swing value
#                (per-axis std 10) needs r=+10, at sensitivity 0.10.
#   exp          dlogL/dr = 1 everywhere. The same target is r=+2.30 at sensitivity
#                1.00. MEASURED: the span did open (12.8 -> 21.9 raw units) but the run
#                diverged -- p99 raw +12.5 means a per-axis Sigma_C of 2.7e5, at which
#                the contact update switches itself off (`applied` ~ 0.001) and the
#                loss rises. Unbounded exp has nothing to stop it.
#   bounded_exp  a sigmoid in LOG space between `lo` and `hi`:
#                    L_ii = exp(ln(lo) + (ln(hi) - ln(lo)) * sigmoid(r))
#                Holds near-uniform relative sensitivity through the interior
#                (dlogL/dr = (ln hi - ln lo) * sigmoid'(r), i.e. ~4 per raw unit at the
#                midpoint against softplus's 0.1-0.8) and saturates SMOOTHLY at the
#                ends instead of overflowing. Not a hard clip: a clip kills the
#                gradient at the bound, this one only shrinks it. The defaults span
#                1e-5 -> 1e2 with headroom either side of the analytic 1e-4 -> 1e1.
#
# Sigma_C spans ~1e10 between stance and swing (analytic: tr 3e-8 -> 3e2), which is a
# scale parameter, so the log parameterisation is the natural one. Measured on the
# softplus run: the head achieved 3.6 of the 19.2 raw units that span requires (19%).
DIAG_PARAMS = ("softplus", "exp", "bounded_exp")

DIAG_LO_DEFAULT = 1.0e-5
DIAG_HI_DEFAULT = 1.0e2


@dataclass(frozen=True)
class DiagSpec:
    """The diag(L) parameterisation as ONE object: kind plus, for `bounded_exp`, the
    range it spans.

    Bounds are LINEAR per-axis standard deviations (the units of `sigma_0`), logged
    internally; `lo`/`hi` are ignored by `softplus` and `exp`. They travel with `kind`
    because they are equally load-bearing at load time -- reading a checkpoint under
    different bounds rescales its head exactly the way reading it under the wrong kind
    does, and neither is detectable from the weights.

    Frozen (hashable) so it is safe as a static argument, and NOT a pytree, so it stays
    an opaque leaf if it ever reaches a transformed function.
    """
    kind: str
    lo: float = DIAG_LO_DEFAULT
    hi: float = DIAG_HI_DEFAULT

    def __post_init__(self):
        if self.kind not in DIAG_PARAMS:
            raise ValueError(
                f"unknown diag_param {self.kind!r}; expected one of {DIAG_PARAMS}")
        if not 0.0 < self.lo < self.hi:
            raise ValueError(
                f"need 0 < lo < hi for the diag bounds, got lo={self.lo}, hi={self.hi}")

    @property
    def log_bounds(self) -> tuple[float, float]:
        """`math.log`, NOT `jnp.log`. This is read inside `_diag_fwd`, which runs under
        `jit`, where every `jnp` op on a constant is staged into the jaxpr as a tracer
        and `float()` of it raises `ConcretizationTypeError`. Plain Python floats keep
        the bounds compile-time constants, which is also what I3 wants."""
        return math.log(self.lo), math.log(self.hi)


def _as_spec(diag) -> DiagSpec:
    """Accept a full `DiagSpec`, or a bare kind string for the kinds that have no
    bounds to lose.

    `bounded_exp` is refused as a bare string ON PURPOSE. Every other way of getting
    the bounds wrong is silent -- that is the whole of N5 -- and a caller that forwards
    `cfg.diag_param` instead of `cfg.diag_spec` would quietly fall back to the module
    defaults while the config says something else. Constructing `DiagSpec("bounded_exp")`
    explicitly is still fine: that is a caller stating it wants the defaults.
    """
    if isinstance(diag, DiagSpec):
        return diag
    if diag == "bounded_exp":
        raise ValueError(
            "bounded_exp carries bounds, so a bare 'bounded_exp' would silently mean "
            f"the defaults ({DIAG_LO_DEFAULT}, {DIAG_HI_DEFAULT}). Pass cfg.diag_spec, "
            "or DiagSpec('bounded_exp', lo, hi) to be explicit.")
    return DiagSpec(diag)


def _diag_inv(y, diag):
    """Raw head-bias value that makes diag(L) == y under `diag`."""
    spec = _as_spec(diag)
    if spec.kind == "softplus":
        return _softplus_inv(y)
    if spec.kind == "exp":
        return jnp.log(y)
    if spec.kind == "bounded_exp":
        lo_, hi_ = spec.log_bounds
        if not spec.lo < y < spec.hi:
            # Host-side, at init: a target outside the range has no finite raw value,
            # and clamping it would silently start the run saturated at a bound.
            raise ValueError(
                f"bounded_exp cannot represent diag(L)={y}: it is outside the range "
                f"({spec.lo}, {spec.hi}). Widen diag_lo/diag_hi or move sigma_0.")
        u = (jnp.log(y) - lo_) / (hi_ - lo_)
        return jnp.log(u) - jnp.log1p(-u)          # logit(u)
    raise ValueError(f"unknown diag_param {spec.kind!r}; expected one of {DIAG_PARAMS}")


def _diag_fwd(o, diag):
    spec = _as_spec(diag)
    if spec.kind == "softplus":
        return jax.nn.softplus(o)
    if spec.kind == "exp":
        return jnp.exp(o)
    if spec.kind == "bounded_exp":
        lo_, hi_ = spec.log_bounds
        return jnp.exp(lo_ + (hi_ - lo_) * jax.nn.sigmoid(o))
    raise ValueError(f"unknown diag_param {spec.kind!r}; expected one of {DIAG_PARAMS}")


def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def init(
    key,
    d_in: int,
    widths: tuple[int, ...],
    sigma_0: float,
    eps: float,
    diag_param: "str | DiagSpec" = "softplus",
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
    # invert the parameterisation at (sigma_0 - eps) so diag(L) = sigma_0 exactly
    # under EVERY choice of `diag_param`; the off-diagonals stay zero, so Sigma_C is
    # diagonal at initialization, matching the filter's isotropic assumption. Keeping
    # init identical across parameterisations is what lets `sigma_0` mean the same
    # thing in every run.
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
            diag_param: "str | DiagSpec" = "softplus") -> jax.Array:
    """One contact's ``(d_in,) = H*F`` normalized feature window → its ``(3,3)`` Cholesky factor.

    ``eps`` is the floor added under the positive parameterisation of the diagonal,
    which makes ``L L^T`` SPD (not merely PSD) by construction.
    """
    h = x
    for layer in params.trunk:
        h = gelu(layer.W @ h + layer.b)
    o = params.head.W @ h + params.head.b

    # Lower-triangular L: a positive parameterisation with a floor keeps the diagonal
    # strictly positive, so L stays full rank even with unconstrained off-diagonals.
    d = _diag_fwd(o[:3], diag_param) + eps
    L = jnp.array(
        [
            [d[0], 0.0, 0.0],
            [o[3], d[1], 0.0],
            [o[4], o[5], d[2]]
        ]
    )
    return L # the filter's inputs take Cholesky factors, not covariances
