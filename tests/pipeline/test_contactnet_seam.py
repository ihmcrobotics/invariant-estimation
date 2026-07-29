r"""`with_contactnet` — the deployment seam into the fused estimator.

The network drives `InEKFInputs.contact_chol`, the **stance-anchor process
noise** — since 2026-07-29; it drove `contact_meas_chol` through run 4.  These
are the properties that make attaching a network safe rather than merely
possible.

The three that matter:

1. **Detaching is exact.** With no ContactNet the carry is still the 2-tuple and
   the trajectory is bit-identical to the pre-ContactNet filter.  Everything
   already gated on G9 depends on that.
2. **The warm-up is exact too.** The provider emits the caller's own heuristic
   `sensors.contact_chol` until its ring buffer holds a full window, and that is
   what `_boundary` passes analytically — so an attached filter reproduces the
   unattached one for the first `span` ticks and then diverges.  That gives a
   sharp, checkable switchover instead of a smeared transient.

   The fallback **inverted** when the socket moved, and the old one is now a trap
   rather than a nicety: zeros in the process socket assert every anchor,
   including a foot in flight, perfectly world-static.
   `test_warmup_fallback_is_the_heuristic_not_zeros` is the regression.
3. **`Sigma_C` actually reaches the filter.** A seam that silently dropped the
   network's output would pass 1 and 2 and be useless.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect)
from invariant_estimation.contactnet import features, network, normalize, online
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.pipeline import main_estimator as me

from .test_main_estimator import N, _sensors, _stack, fused  # noqa: F401


def _seq(n_ticks: int, n_j: int):
    """A trajectory with `torques` and `q_unfiltered` wired.

    `_sensors` leaves both empty (the analytic filter never reads them), but
    ContactNet's feature vector is
    ``(omega, accel, q_sub, tau_sub, p, v)`` — so a seam test that left `torques`
    empty would exercise a shape the deployed path never sees.
    """
    out = []
    for k in range(n_ticks):
        q = 0.05 * np.sin(0.02 * k + np.arange(n_j))
        s = _sensors(q=q, gyro=np.array([0.01, -0.02, 0.005]))
        out.append(s._replace(
            q_unfiltered=jnp.zeros(0, dtype=jnp.float64),
            torques=jnp.asarray(3.0 * np.cos(0.03 * k + np.arange(n_j)),
                                dtype=jnp.float64)))
    return _stack(out)


@pytest.fixture(scope="module")
def sensors_seq(fused):
    return _seq(80, fused.n_joints)


def _cfg(F: int) -> ContactNetConfig:
    """Short window so a test trajectory can outrun the warm-up."""
    with __import__("warnings").catch_warnings():
        # A 6 ms window aliases the torque channel; irrelevant here, where the
        # subject is the seam and not the bandwidth.
        __import__("warnings").simplefilter("ignore")
        return ContactNetConfig(F=F, sigma_0=1.0e-4, H=4, window_span_s=0.006,
                                dt=1.0e-3, widths=(8, 8))


def _attach(fused, *, perturb: float):
    """Attach a ContactNet whose head is moved off init by `perturb`.

    At initialization the head is zero and `Sigma_C = sigma_0^2 I` — negligible
    against `N`, so an unperturbed network would be indistinguishable from the
    analytic filter and every test below would pass vacuously.
    """
    subchain = np.tile(np.arange(min(3, fused.n_joints)), (fused.n_contacts, 1))
    F = 12 + 2 * subchain.shape[1]
    cfg = _cfg(F)

    # Constants are irrelevant to the seam; use unit scaling so the test does not
    # depend on a fitted artifact.
    consts = normalize.NormConstants(
        mean=np.zeros(F), std=np.ones(F), names=tuple(f"c{i}" for i in range(F)),
        floored=(), n_ticks=1, source="test")

    params = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    params = params._replace(head=params.head._replace(
        W=params.head.W + perturb, b=params.head.b + perturb))
    return me.with_contactnet(fused, params, cfg, consts, subchain), cfg


def test_no_contactnet_leaves_the_carry_and_trajectory_untouched(fused, sensors_seq):
    """Default `contactnet=None` is the pre-ContactNet filter, exactly."""
    carry = me.init_fused_carry(fused, q0=jnp.zeros(fused.n_joints))
    assert len(carry) == 2, "analytic carry must stay a 2-tuple"
    _, out = me.run_fused(fused, carry, sensors_seq)
    # The measurement socket is unused and stays zero...
    assert bool(jnp.all(out.inekf_inputs.contact_meas_chol == 0.0))
    # ...and the process socket carries the heuristic through untouched.
    assert np.array_equal(np.asarray(out.inekf_inputs.contact_chol),
                          np.asarray(sensors_seq.contact_chol))


def test_attached_carry_gains_a_third_slot(fused):
    cn, cfg = _attach(fused, perturb=0.05)
    carry = me.init_fused_carry(cn, q0=jnp.zeros(cn.n_joints))
    assert len(carry) == 3
    assert carry[2].buf.shape == (online.span_ticks(cfg), cn.n_contacts, cfg.F)
    # The first two slots are untouched by attaching.
    base = me.init_fused_carry(fused, q0=jnp.zeros(fused.n_joints))
    for a, b in zip(jax.tree.leaves(carry[:2]), jax.tree.leaves(base)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_warmup_is_bit_identical_then_diverges(fused, sensors_seq):
    r"""Attached == unattached for `span` ticks, then not.

    Both halves matter.  Identity during warm-up says the fallback really is the
    analytic heuristic rather than something close to it; divergence after says
    `Sigma_C` reaches the filter at all.  A seam that dropped the network's
    output would pass the first half alone.
    """
    cn, cfg = _attach(fused, perturb=0.05)
    span = online.span_ticks(cfg)
    T = sensors_seq.encoders.shape[0]
    if T <= span + 5:
        pytest.skip(f"fixture trajectory {T} ticks is shorter than warm-up {span}")

    _, base = me.run_fused(fused, me.init_fused_carry(
        fused, q0=jnp.zeros(fused.n_joints)), sensors_seq)
    _, got = me.run_fused(cn, me.init_fused_carry(
        cn, q0=jnp.zeros(cn.n_joints)), sensors_seq)

    chol = np.asarray(got.inekf_inputs.contact_chol)
    heur = np.asarray(sensors_seq.contact_chol)
    assert np.array_equal(chol[:span - 1], heur[:span - 1]), (
        "warm-up did not emit the analytic heuristic")
    assert not np.array_equal(chol[span - 1:], heur[span - 1:]), (
        "Sigma_C never reached the filter")
    # The measurement socket is not touched by attaching a network any more.
    assert bool(jnp.all(got.inekf_inputs.contact_meas_chol == 0.0))

    # States agree bit-for-bit through the warm-up...
    assert np.array_equal(np.asarray(base.v[:span - 1]),
                          np.asarray(got.v[:span - 1]))
    # ...and stop agreeing once the network takes over.
    assert not np.array_equal(np.asarray(base.v[span:]), np.asarray(got.v[span:]))


def test_warmup_fallback_is_the_heuristic_not_zeros(fused, sensors_seq):
    r"""The §7 named trap, as a test.

    ``contact_meas_chol = zeros`` meant "the shipped filter" on the measurement
    socket.  The same idiom on the process socket means ``Sigma_C = 0``: every
    anchor, swing feet included, asserted perfectly world-static.  That is the
    run-1 failure mode, and it would be invisible in a test that only checked
    "the warm-up is constant" or "the graph did not change".

    Asserted against the *heuristic value the caller supplied*, not against a
    literal, so a future change to `sim.sensors`' stance/swing constants cannot
    quietly make this vacuous.
    """
    cn, cfg = _attach(fused, perturb=0.05)
    span = online.span_ticks(cfg)
    if sensors_seq.encoders.shape[0] <= span + 5:
        pytest.skip("fixture trajectory is shorter than the warm-up")

    _, got = me.run_fused(cn, me.init_fused_carry(
        cn, q0=jnp.zeros(cn.n_joints)), sensors_seq)
    warm = np.asarray(got.inekf_inputs.contact_chol[:span - 1])

    assert np.abs(warm).max() > 0.0, (
        "warm-up emitted zeros into the PROCESS socket — that pins every anchor, "
        "including swing feet, as perfectly world-static")
    assert np.array_equal(warm, np.asarray(sensors_seq.contact_chol[:span - 1]))


def test_attached_step_stays_one_constant_graph(fused, sensors_seq):
    r"""I7: attaching must not introduce a data-dependent branch or shape.

    Compared as **jaxprs**, matching `test_jaxpr_constant_across_contact_and_gate`.
    Lowered HLO is the wrong oracle here: it carries `sdy.sharding` annotations
    that differ between a `device_put` carry and one returned eagerly from a
    step, so it reports a difference in *commitment* as a difference in graph —
    the same trap `init_fused_carry`'s comment describes.

    Both the ring-buffer roll and the `ready` fallback are the kind of thing that
    would be written as a Python branch by default; either would change the
    graph between the warm-up and the steady state.
    """
    cn, cfg = _attach(fused, perturb=0.05)
    span = online.span_ticks(cfg)
    carry = me.init_fused_carry(cn, q0=jnp.zeros(cn.n_joints))
    step = me.make_fused_step(cn)

    def jaxpr(c, k):
        return str(jax.make_jaxpr(step)(c, jax.tree.map(lambda a: a[k], sensors_seq)))

    first = jaxpr(carry, 0)

    # Walk past the warm-up so `ready` has flipped, then compare again.
    c = carry
    for k in range(span + 2):
        c, _ = step(c, jax.tree.map(lambda a: a[k], sensors_seq))
    assert bool(c[2].n >= span), "did not actually cross the warm-up boundary"

    assert jaxpr(c, span + 3) == first, "graph changed across the warm-up boundary"


def test_subchain_contact_count_must_match_the_estimator(fused):
    """A mismatched subchain indexes different slots — reject at build."""
    subchain = np.zeros((fused.n_contacts + 1, 3), dtype=int)
    cfg = _cfg(18)
    consts = normalize.NormConstants(
        mean=np.zeros(18), std=np.ones(18),
        names=tuple(f"c{i}" for i in range(18)), floored=(), n_ticks=1,
        source="test")
    params = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    with pytest.raises(ValueError, match="contacts"):
        me.with_contactnet(fused, params, cfg, consts, subchain)
