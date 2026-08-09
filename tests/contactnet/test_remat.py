"""Property tests for rematerialization of the BPTT scan (`make_segment_loss`).

Why this file exists: `make_segment_loss` accepted a `remat` argument, documented it
as wrapping the scan body in `jax.checkpoint`, and never applied it. All four
L2-options arms (`results/2026-08-06_*`) trained with `remat=True` in their
`summary.json` and no rematerialization in the graph. A value test cannot catch
that -- remat is mathematically identity, so a dead flag and a live one agree on
every number. Hence the three guarantees here:

  * **identity**  -- remat must NOT change the loss or any gradient (1e-12). This is
    the invariant that makes it safe to enable; if it ever fails, remat is silently
    changing the estimator and every run under it is suspect.
  * **liveness**  -- the remat primitive must actually appear in the jaxpr when the
    flag is set, and must NOT when it is clear. This is the test that would have
    caught the original defect.
  * **effect**    -- it must actually buy activation memory, measured statically
    from the compiled executable rather than from a runtime high-water mark
    (`peak_bytes_in_use` never resets within a process, so two in-process
    measurements are not comparable).
"""
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.contactnet import dataset, network, rollout
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.sim import collect

# The remat primitive as JAX spells it in a jaxpr (jax 0.10.x). Pinned rather than
# fuzzy-matched: a rename should fail loudly here and be re-pinned deliberately,
# because a silent miss turns this file back into the green-while-broken state it
# was written to end.
REMAT_PRIMITIVE = "remat2"

CONTACTS_PER_FOOT = 4          # N=8, the deployed geometry the ablation trains on


def _a_pool_rollout():
    """Any collected N=8 rollout: the `n8fix` ablation pool, else the older n8 one."""
    for pattern in ("*_n8fix_seed*.npz", "*_n8_seed*.npz"):
        hits = sorted(glob.glob(os.path.join("data", pattern)))
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="module")
def scene():
    """(params, segment, ekf, kinematics, cfg) on a real rollout, at a small L.

    A synthetic segment cannot exercise this: remat wraps the *filter* scan body, so
    the property is only meaningful against the real InEKF step and real inputs.
    L=16/H=10 keeps the module under a few seconds while still scanning.
    """
    path = _a_pool_rollout()
    if path is None:
        pytest.skip("no N=8 rollout in data/ (collection not run here)")

    cfg = ContactNetConfig(L=16, H=10)
    c = collect.build_collector(contacts_per_foot=CONTACTS_PER_FOOT, verbose=False)
    dataset.build_channel_cache([path], c, verbose=False)
    norm = dataset.fit_normalization([path])
    prep = dataset.prepare([path], norm, cfg)[0]
    P0 = dataset.measure_p0(c.fused, prep, cfg)
    seg = dataset.make_segment(prep, t0=prep.t_lo, cfg=cfg, P0=P0)
    params = network.init(jax.random.PRNGKey(0), d_in=cfg.d_in, widths=cfg.widths,
                          sigma_0=cfg.sigma_0, eps=cfg.eps)
    return params, seg, c.fused.ekf, c.fused.kinematics, cfg


def _loss_fn(ekf, kinematics, cfg, remat, objective="l2_velocity"):
    return rollout.make_segment_loss(
        ekf, kinematics, cfg.eps, beta=cfg.beta, objective=objective, remat=remat)


# Gradient-agreement tolerance, in relative terms. NOT tuned to the observation --
# derived from the conditioning, then checked against it:
#
#   upper bound   float64 through the filter's own worst conditioning,
#                 eps * cond(S) ~ 2.2e-16 * 1.09e9 ~ 2e-7        (results.md, §3)
#   MEASURED      remat on vs off, L=16, N=8                      5.2e-12
#   noise floor   repeat run, and `everything_saveable` (remat's graph wrapping
#                 with nothing actually recomputed)               0.0, bit-identical
#
# The floor being exactly zero localises the residual: it is not run-to-run jitter
# and not the checkpoint wrapper, it is XLA fusing the RECOMPUTED copy of the body
# differently from the original forward -- i.e. reassociation, which is the expected
# and only numerical consequence of rematerialization. 1e-9 sits ~200x above the
# measurement (not flaky) and ~200x below the conditioning bound, so a real semantic
# break -- a truncated backward pass, a stray `stop_gradient` -- still lands orders
# of magnitude outside it.
GRAD_RTOL = 1.0e-9


@pytest.mark.parametrize("objective", ["l2_velocity", "l2_vel_pos_ori"])
def test_remat_is_identity(scene, objective):
    """remat=True and remat=False must agree on the loss AND every gradient leaf.

    This is the whole safety argument for turning it on: `jax.checkpoint` recomputes
    the scan body's interior on the backward pass instead of storing it, which is a
    memory/compute trade and NOT an approximation. Truncated BPTT or a stray
    `stop_gradient` would show up here as a gradient mismatch far outside `GRAD_RTOL`.
    """
    params, seg, ekf, kin, cfg = scene
    w = dict(w_pos=1.0, w_ori=1.0) if objective != "l2_velocity" else {}

    def run(remat):
        f = rollout.make_segment_loss(
            ekf, kin, cfg.eps, beta=cfg.beta, objective=objective, remat=remat, **w)
        (loss, _aux), grads = jax.value_and_grad(f, has_aux=True)(params, seg)
        flat = np.concatenate([np.asarray(g).ravel() for g in jax.tree.leaves(grads)])
        return float(loss), flat

    loss_off, g_off = run(False)
    loss_on, g_on = run(True)

    # Guard the guard: a zero gradient would make the comparison below vacuous.
    assert np.all(np.isfinite(g_off)) and np.abs(g_off).max() > 0.0, (
        f"{objective}: reference (remat=False) gradient is zero or non-finite -- "
        f"the identity check below would pass on nothing")

    assert loss_on == pytest.approx(loss_off, rel=GRAD_RTOL, abs=1e-14), (
        f"{objective}: remat changed the LOSS ({loss_on} vs {loss_off}) -- it is "
        f"supposed to be identity up to reassociation")
    rel = np.abs(g_on - g_off) / (1.0 + np.abs(g_off))
    assert rel.max() < GRAD_RTOL, (
        f"{objective}: remat changed the GRADIENT (max rel {rel.max():.3e} at leaf "
        f"index {int(np.argmax(rel))}) -- every run trained under it is suspect")


def test_remat_residual_is_recomputation_not_wrapping(scene):
    """The on/off gradient residual must come from RECOMPUTATION, nothing else.

    `everything_saveable` applies the identical `jax.checkpoint` wrapper but saves
    every intermediate, so nothing is re-evaluated and the arithmetic is literally
    the same sequence of ops. It must therefore be bit-identical to no remat at all.

    This is what makes `GRAD_RTOL` an argument rather than a fudge factor: it pins
    the noise floor at exactly zero, so the 5e-12 seen with real remat is attributable
    to re-evaluating the body and to nothing else. If this ever goes non-zero, the
    checkpoint wrapper itself has started changing results and the tolerance above
    stops being justified.
    """
    params, seg, ekf, kin, cfg = scene
    from invariant_estimation.contactnet.rollout import contact_factors
    from invariant_estimation.contactnet.losses import l2_velocity
    from invariant_estimation.inEKF.filter import init_carry, make_step

    def grads_under(wrap):
        step = wrap(make_step(ekf, kin))

        def f(p, s):
            inp = s.inputs._replace(contact_chol=contact_factors(p, s.windows, cfg.eps))
            _carry, out = jax.lax.scan(step, init_carry(s.state0), inp)
            return l2_velocity(out.state.v, out.state.R, s.v_true, s.R_true)

        g = jax.grad(f)(params, seg)
        return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(g)])

    plain = grads_under(lambda s: s)
    saved = grads_under(lambda s: jax.checkpoint(
        s, prevent_cse=False, policy=jax.checkpoint_policies.everything_saveable))

    assert np.array_equal(plain, grads_under(lambda s: s)), (
        "the un-remat'd gradient is not reproducible run-to-run -- there is a source "
        "of nondeterminism here that GRAD_RTOL was not sized for")
    assert np.array_equal(plain, saved), (
        "jax.checkpoint with everything_saveable changed the gradient even though it "
        "recomputes nothing -- the wrapper itself is perturbing the arithmetic, so "
        "GRAD_RTOL's derivation (reassociation-from-recompute only) no longer holds")


def test_remat_flag_reaches_the_jaxpr(scene):
    """The flag must be LIVE, not merely accepted.

    The regression this pins: `make_segment_loss` took `remat`, documented it, and
    dropped it on the floor -- so `summary.json` claimed rematerialization that the
    graph never had. Asserting on values cannot detect that (see test above);
    asserting on the graph can.
    """
    params, seg, ekf, kin, cfg = scene

    def jaxpr_of(remat):
        f = _loss_fn(ekf, kin, cfg, remat)
        return str(jax.make_jaxpr(f)(params, seg))

    assert REMAT_PRIMITIVE in jaxpr_of(True), (
        f"remat=True produced a jaxpr with no {REMAT_PRIMITIVE!r} primitive -- the "
        f"flag is dead again, exactly as it was for the A-D arms")
    assert REMAT_PRIMITIVE not in jaxpr_of(False), (
        f"remat=False still contains {REMAT_PRIMITIVE!r} -- the flag does not turn off")


def test_remat_actually_saves_activation_memory(scene):
    """Compiled scratch memory for the BACKWARD pass must drop when remat is on.

    Measured from the compiled executable (`memory_analysis().temp_size_in_bytes`),
    not from a runtime peak: `peak_bytes_in_use` is a per-process high-water mark
    that never resets, so an in-process before/after comparison silently reports the
    first measurement twice. The static number is also backend-portable, so this
    test is meaningful on CPU-only CI.

    Asserted loosely (>10%) -- the exact ratio is the probe's job
    (`scripts/remat_probe.py`), the DIRECTION is the property.
    """
    params, seg, ekf, kin, cfg = scene

    def temp_bytes(remat):
        f = _loss_fn(ekf, kin, cfg, remat)
        g = jax.jit(lambda p, s: jax.grad(f, has_aux=True)(p, s)[0])
        return g.lower(params, seg).compile().memory_analysis().temp_size_in_bytes

    off, on = temp_bytes(False), temp_bytes(True)
    assert on < 0.9 * off, (
        f"remat did not reduce compiled scratch memory (on={on/1e6:.1f} MB vs "
        f"off={off/1e6:.1f} MB) -- it is in the graph but buying nothing, which at "
        f"L=512 is the difference between training and an OOM")
