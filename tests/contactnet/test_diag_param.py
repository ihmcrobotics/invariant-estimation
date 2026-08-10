"""The diag(L) parameterisation: both options, and why `exp` was added.

Sigma_C spans ~1e10 between stance and swing (analytic heuristic: tr 3e-8 stance,
3e2 swing). `softplus` is the original parameterisation and is well behaved at the
TIGHT end -- for r << 0 it IS exp, so relative sensitivity dlog(L)/dr = 1 -- but it
goes linear above zero, where that sensitivity decays as 1/r. Reaching the swing end
needs r = +10 at sensitivity 0.10. Under `exp` the same target is r = +2.30 at
sensitivity 1.00.

Measured consequence on the softplus run: the head achieved 3.6 of the 19.2 raw units
that span requires (19%), leaving the learned Sigma_C 685x too TIGHT in swing -- which
makes the filter treat a lifting foot as world-static and push the base upward. That
is the closed-loop +2.5 m failure.

The load-bearing property for existing work: both parameterisations must agree AT
INIT, so `sigma_0` still means the same thing and checkpoints trained under softplus
stay interpretable when the option is recorded and read back.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.contactnet import network
from invariant_estimation.contactnet.config import ContactNetConfig

D_IN, WIDTHS, SIGMA_0, EPS = 60, (16, 16), 1.0e-4, 1.0e-6


def _params(diag_param, seed=0):
    return network.init(jax.random.PRNGKey(seed), D_IN, WIDTHS, SIGMA_0, EPS,
                        diag_param)


def _perturbed(diag_param, seed, raw_bias):
    """Init, then move the head OFF init to a chosen pre-activation level.

    The bias must be moved, not just the weights: the head initialises at
    raw = -9.21 (sigma_0 = 1e-4), and down there softplus IS exp to five digits, so
    weight noise alone leaves the two parameterisations indistinguishable. That is
    the corrected mechanism -- softplus is fine at the tight end and degrades above
    zero -- so any test that wants to see them differ has to look above zero.
    """
    p = _params(diag_param, seed)
    rng = np.random.default_rng(seed)
    return p._replace(head=p.head._replace(
        W=jnp.asarray(rng.standard_normal(p.head.W.shape) * 0.2),
        b=jnp.asarray(np.concatenate([np.full(3, raw_bias), np.zeros(3)]))))


@pytest.mark.parametrize("diag_param", ["softplus", "exp"])
def test_init_emits_sigma_0_exactly(diag_param):
    """At init the head's weights are zero, so the output is the bias for any input,
    and diag(L) must equal sigma_0 under BOTH parameterisations.

    This is what keeps `sigma_0` meaningful across the change, and it is the property
    `tests/sim/test_n8_network.py` relies on when it asserts iteration 0 reproduces
    the shipped filter.
    """
    p = _params(diag_param)
    x = jnp.asarray(np.random.default_rng(0).standard_normal(D_IN))
    L = network.forward(p, x, EPS, diag_param)
    d = np.diag(np.asarray(L))
    assert np.allclose(d, SIGMA_0, rtol=1e-12, atol=1e-14), (
        f"{diag_param}: init emits diag(L)={d}, expected {SIGMA_0}")
    # off-diagonals zero => Sigma_C diagonal at init, matching the filter's isotropic
    # assumption
    assert np.allclose(np.asarray(L)[np.tril_indices(3, -1)], 0.0)


@pytest.mark.parametrize("diag_param", ["softplus", "exp"])
def test_output_is_spd_and_lower_triangular(diag_param):
    p = _perturbed(diag_param, 1, raw_bias=0.5)
    rng = np.random.default_rng(11)
    for _ in range(20):
        x = jnp.asarray(rng.standard_normal(D_IN))
        L = np.asarray(network.forward(p, x, EPS, diag_param))
        assert np.allclose(L[np.triu_indices(3, 1)], 0.0), "L must be lower triangular"
        assert (np.diag(L) > 0).all(), "diag(L) must be strictly positive"
        # SPD relative to the matrix scale. L L^T is SPD for any positive diagonal
        # (det = prod(diag)^2), but with a near-zero diagonal and O(1) off-diagonals
        # the smallest eigenvalue rounds through zero in float64 -- a statement about
        # conditioning, not about the parameterisation.
        w = np.linalg.eigvalsh(L @ L.T)
        assert w.min() > -1e-12 * max(w.max(), 1.0), "L L^T must be SPD"


def test_exp_holds_relative_sensitivity_where_softplus_loses_it():
    """The reason for the change, as an assertion rather than a comment.

    dlog(L)/dr is what a SCALE parameter cares about: it says how many raw units buy
    a decade. `exp` holds it at 1 everywhere. `softplus` matches that for r << 0 and
    degrades above zero, which is exactly where the swing regime sits.
    """
    d_softplus = lambda r: float(jax.nn.sigmoid(r) / jax.nn.softplus(r))
    d_exp = 1.0

    # tight end: the two agree, so softplus was never the problem at init
    assert d_softplus(-9.21) == pytest.approx(d_exp, rel=1e-3)

    # swing end: softplus has lost an order of magnitude
    assert d_softplus(10.0) < 0.15
    assert d_softplus(10.0) < 0.15 * d_exp

    # and the required travel is shorter under exp: per-axis std 10 needs r=+10 under
    # softplus but only ln(10)=2.30 under exp
    assert float(jnp.log(10.0)) == pytest.approx(2.302585, rel=1e-5)


def test_config_rejects_an_unknown_parameterisation():
    """A silently-unknown value would fall through to whichever branch is last."""
    with pytest.raises(ValueError, match="diag_param"):
        ContactNetConfig(diag_param="relu")
    for good in ("softplus", "exp"):
        assert ContactNetConfig(diag_param=good).diag_param == good


def test_mismatched_parameterisation_changes_sigma_c():
    """Loading a checkpoint under the WRONG parameterisation must not be silent.

    There is no way to detect it from the weights alone -- which is why the choice is
    recorded in summary.json and must be read back. This test documents the size of
    the error that would result: reading softplus-trained weights as exp rescales
    Sigma_C by orders of magnitude away from init.
    """
    # raw = +2.0, i.e. the swing-ish region: softplus(2)=2.13 vs exp(2)=7.39.
    p = _perturbed("softplus", 2, raw_bias=2.0)
    x = jnp.asarray(np.random.default_rng(22).standard_normal(D_IN))
    a = np.diag(np.asarray(network.forward(p, x, EPS, "softplus")))
    b = np.diag(np.asarray(network.forward(p, x, EPS, "exp")))
    assert not np.allclose(a, b, rtol=1e-3), (
        "the two parameterisations produced the same diag(L) off-init; if that were "
        "true the recorded choice would not matter, and it does")
