"""The diag(L) parameterisation: all three options, and why each was added.

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

`exp` was then MEASURED and it diverged: the span did open (12.8 -> 21.9 raw units) but
p99 raw hit +12.5, i.e. a per-axis Sigma_C of 2.7e5, at which the contact update
switches itself off (`applied` ~ 0.001) and the loss rises. Nothing bounds exp.

`bounded_exp` is that same log parameterisation confined to [lo, hi] by a sigmoid in
log space. It is not a clip -- a clip has zero gradient at the bound, this only shrinks
it -- and the default 1e-5 -> 1e2 brackets the analytic 1e-4 -> 1e1 with a decade of
headroom either side. Across that analytic span it holds dlogL/dr >= 1.97, versus
softplus's 0.10 at the swing end.

The load-bearing property for existing work: ALL parameterisations must agree AT INIT,
so `sigma_0` still means the same thing and checkpoints trained under softplus stay
interpretable when the option is recorded and read back (`test_checkpoint_readback.py`
covers the reading-back half).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.contactnet import network
from invariant_estimation.contactnet.config import ContactNetConfig

D_IN, WIDTHS, SIGMA_0, EPS = 60, (16, 16), 1.0e-4, 1.0e-6


def _spec(diag_param):
    """Kind -> the object the network takes. `bounded_exp` refuses a bare string so a
    caller cannot lose its bounds by accident (see `network._as_spec`)."""
    return network.DiagSpec(diag_param)


def _params(diag_param, seed=0):
    return network.init(jax.random.PRNGKey(seed), D_IN, WIDTHS, SIGMA_0, EPS,
                        _spec(diag_param))


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


@pytest.mark.parametrize("diag_param", network.DIAG_PARAMS)
def test_init_emits_sigma_0_exactly(diag_param):
    """At init the head's weights are zero, so the output is the bias for any input,
    and diag(L) must equal sigma_0 under BOTH parameterisations.

    This is what keeps `sigma_0` meaningful across the change, and it is the property
    `tests/sim/test_n8_network.py` relies on when it asserts iteration 0 reproduces
    the shipped filter.
    """
    p = _params(diag_param)
    x = jnp.asarray(np.random.default_rng(0).standard_normal(D_IN))
    L = network.forward(p, x, EPS, _spec(diag_param))
    d = np.diag(np.asarray(L))
    assert np.allclose(d, SIGMA_0, rtol=1e-12, atol=1e-14), (
        f"{diag_param}: init emits diag(L)={d}, expected {SIGMA_0}")
    # off-diagonals zero => Sigma_C diagonal at init, matching the filter's isotropic
    # assumption
    assert np.allclose(np.asarray(L)[np.tril_indices(3, -1)], 0.0)


@pytest.mark.parametrize("diag_param", network.DIAG_PARAMS)
def test_output_is_spd_and_lower_triangular(diag_param):
    p = _perturbed(diag_param, 1, raw_bias=0.5)
    rng = np.random.default_rng(11)
    for _ in range(20):
        x = jnp.asarray(rng.standard_normal(D_IN))
        L = np.asarray(network.forward(p, x, EPS, _spec(diag_param)))
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
    for good in network.DIAG_PARAMS:
        assert ContactNetConfig(diag_param=good).diag_param == good


# --------------------------------------------------------------------------------
# bounded_exp: the properties it exists for
# --------------------------------------------------------------------------------

BSPEC = network.DiagSpec("bounded_exp")


def test_bounded_exp_cannot_leave_its_range():
    """The one thing unbounded `exp` could not do.

    `exp` at the diverged run's p99 raw output (+12.5) is 2.7e5 per axis, which is
    where the contact update switched itself off. Saturation has to be total: no raw
    value, however extreme, may escape [lo, hi], and none may produce a non-finite.
    """
    r = jnp.asarray([-1.0e3, -50.0, -12.5, 0.0, 12.5, 50.0, 1.0e3])
    d = np.asarray(network._diag_fwd(r, BSPEC))
    assert np.isfinite(d).all()
    # The bound holds to the round-trip of exp(log(bound)) -- one ulp, not a slack
    # anyone can train through.
    assert (d >= BSPEC.lo * (1 - 1e-12)).all(), d
    assert (d <= BSPEC.hi * (1 + 1e-12)).all(), d

    # the specific failure it retires, side by side
    at_p99 = float(network._diag_fwd(jnp.asarray(12.5), BSPEC))
    assert at_p99 <= BSPEC.hi
    assert float(network._diag_fwd(jnp.asarray(12.5), "exp")) > 1.0e5


def test_bounded_exp_holds_sensitivity_across_the_analytic_span():
    """Why bounding does not give the range back with one hand and take it with the
    other: over the span that matters -- analytic stance 1e-4 to analytic swing 1e1 --
    relative sensitivity stays within a factor 2.1 of its peak, and 20x above what
    softplus offers at the swing end.

    Sensitivity is the derivative that matters for a SCALE parameter: dlog(L)/dr says
    how many raw units buy a decade.
    """
    dlog = jax.grad(lambda r: jnp.log(network._diag_fwd(r, BSPEC)))
    r_stance = float(network._diag_inv(1.0e-4, BSPEC))
    r_swing = float(network._diag_inv(1.0e1, BSPEC))

    assert float(dlog(0.0)) == pytest.approx(4.03, rel=1e-2)         # peak
    for r in (r_stance, r_swing):
        assert float(dlog(r)) > 1.9                                  # ends of the span

    # softplus at the same swing TARGET (raw +10) is an order of magnitude worse
    d_softplus_swing = float(jax.nn.sigmoid(10.0) / jax.nn.softplus(10.0))
    assert float(dlog(r_swing)) > 15.0 * d_softplus_swing

    # and the whole analytic span is 3.6 raw units wide, against softplus's 19.2 --
    # the measured gap the softplus run only covered 19% of
    assert abs(r_swing - r_stance) == pytest.approx(3.58, abs=0.05)


def test_bounded_exp_inverse_is_exact_on_the_span():
    """`_diag_inv` is what puts init exactly at `sigma_0`; if the round trip drifts,
    every parameterisation stops meaning the same thing at iteration 0."""
    for y in np.logspace(-4, 1, 11):
        r = network._diag_inv(float(y), BSPEC)
        assert float(network._diag_fwd(r, BSPEC)) == pytest.approx(y, rel=1e-12)


def test_bounded_exp_refuses_a_target_outside_its_range():
    """Silently clamping would start the run saturated at a bound with sigma_0 no
    longer meaning what it says."""
    for y in (BSPEC.lo / 10.0, BSPEC.hi * 10.0):
        with pytest.raises(ValueError, match="outside the range"):
            network._diag_inv(y, BSPEC)


def test_config_rejects_bounds_that_do_not_bracket_sigma_0():
    """The init bias is a logit of (log sigma_0 - log lo)/(log hi - log lo): outside
    the range there is no finite value for it."""
    with pytest.raises(ValueError, match="diag_lo"):
        ContactNetConfig(diag_lo=0.0)
    with pytest.raises(ValueError, match="diag_lo"):
        ContactNetConfig(diag_lo=1.0, diag_hi=1.0e-3)
    with pytest.raises(ValueError, match="sigma_0"):
        # sigma_0 defaults to 1e-4, so a floor above it has no representable init
        ContactNetConfig(diag_param="bounded_exp", diag_lo=1.0e-3)
    # ... and the same bounds are legal for the parameterisations that ignore them
    assert ContactNetConfig(diag_param="softplus", diag_lo=1.0e-3).diag_lo == 1.0e-3


@pytest.mark.parametrize("diag_param", network.DIAG_PARAMS)
def test_forward_traces_under_jit(diag_param):
    """Every deployed and trained call site is inside `jit` (`train_step`,
    `make_provider`'s scan), and eager evaluation does not exercise that.

    This caught a real one: `DiagSpec.log_bounds` computed `float(jnp.log(lo))`, which
    is fine eagerly and raises `ConcretizationTypeError` under `jit`, because every
    `jnp` op on a constant inside a trace is staged into the jaxpr as a tracer. The
    unit tests were all eager, so the bounded_exp training run died at step 0.
    """
    p = _params(diag_param)
    spec = _spec(diag_param)
    f = jax.jit(lambda params, x: network.forward(params, x, EPS, spec))
    x = jnp.asarray(np.random.default_rng(4).standard_normal(D_IN))
    L = np.asarray(f(p, x))
    assert np.isfinite(L).all()
    assert np.allclose(np.diag(L), SIGMA_0, rtol=1e-12, atol=1e-14)
    # and gradients flow, which is the other thing training needs from it
    g = jax.jit(jax.grad(lambda params, x: jnp.sum(network.forward(params, x, EPS, spec))
                         ))(p, x)
    assert np.isfinite(np.asarray(g.head.b)).all()


def test_a_bare_bounded_exp_string_is_refused():
    """The one N5 failure that can be made structural rather than documented: a caller
    forwarding `cfg.diag_param` instead of `cfg.diag_spec` would silently get the
    module-default bounds while the config says otherwise. Every other mis-load is
    invisible; this one raises."""
    p = _params("bounded_exp")
    x = jnp.asarray(np.random.default_rng(0).standard_normal(D_IN))
    with pytest.raises(ValueError, match="cfg.diag_spec"):
        network.forward(p, x, EPS, "bounded_exp")
    # the kinds with no bounds to lose still take a bare string
    network.forward(_params("softplus"), x, EPS, "softplus")


def test_bounds_travel_with_the_kind():
    """`DiagSpec` exists so the bounds cannot be lost on the way to the network: a
    checkpoint read under different bounds is rescaled exactly the way one read under
    the wrong kind is, and neither is visible in the weights."""
    p = _perturbed("bounded_exp", 3, raw_bias=1.0)
    x = jnp.asarray(np.random.default_rng(33).standard_normal(D_IN))
    wide = np.diag(np.asarray(network.forward(p, x, EPS, BSPEC)))
    narrow = np.diag(np.asarray(network.forward(
        p, x, EPS, network.DiagSpec("bounded_exp", 1.0e-5, 1.0e1))))
    assert not np.allclose(wide, narrow, rtol=1e-3)

    cfg = ContactNetConfig(diag_param="bounded_exp", diag_lo=1.0e-5, diag_hi=1.0e1)
    assert cfg.diag_spec == network.DiagSpec("bounded_exp", 1.0e-5, 1.0e1)


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
