"""Gate D: ContactNet at N=8 -- shapes, SPD, iteration-0 reproduction, gradients.

`network.init/forward` are per-contact and `rollout.contact_factors` already vmaps
over the contact axis, so N=8 needs no code change. What has to be PROVEN is that
the invariants survive the wider axis:

  * Sigma_C is (8,3,3) and every block SPD;
  * the iteration-0 bias trick still reproduces the analytic baseline, i.e. at
    init the network emits exactly sigma_0 * I -- if that drifts, run 1 no longer
    starts from the shipped filter and every comparison against R0 is confounded;
  * gradients are finite AND NONZERO through the N=8 BPTT scan (a zero gradient
    trains nothing while looking perfectly healthy).

The research risk recorded in `test_n8_corners.py` is the reason the last test
here checks whether Sigma_C can DIFFER across a foot's corners at all: corners
share q/qd/tau and differ only through FK, so this bounds what the net could
possibly learn. It asserts capability, not that training achieves it.
"""

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.contactnet import network, rollout
from invariant_estimation.contactnet.config import ContactNetConfig

N8 = 8


@pytest.fixture(scope="module")
def cfg():
    return ContactNetConfig()


@pytest.fixture(scope="module")
def params(cfg):
    return network.init(jax.random.PRNGKey(0), d_in=cfg.d_in, widths=cfg.widths,
                        sigma_0=cfg.sigma_0, eps=cfg.eps)


def test_sigma_c_is_8x3x3_and_spd(params, cfg):
    """vmapped over 8 contacts: (L,8,3,3) Cholesky factors, every L L^T SPD."""
    rng = np.random.default_rng(0)
    L = 5
    windows = jnp.asarray(rng.normal(size=(L, N8, cfg.H, cfg.F)))
    Lc = rollout.contact_factors(params, windows, cfg.eps)

    assert Lc.shape == (L, N8, 3, 3), f"expected (L,8,3,3), got {Lc.shape}"
    for t in range(L):
        for i in range(N8):
            S = np.asarray(Lc[t, i] @ Lc[t, i].T)
            np.testing.assert_allclose(S, S.T, atol=1e-12)
            assert np.linalg.eigvalsh(S).min() > 0.0, f"Sigma_C[{t},{i}] not PD"


def test_iteration_zero_reproduces_the_analytic_baseline_at_n8(params, cfg):
    """At init the net must emit sigma_0 * I exactly, for ALL 8 contacts.

    This is the `softplus^-1(sigma_0 - eps)` bias trick. If it does not hold at
    N=8, run 1 does not start from the shipped filter and R3-vs-R0 is confounded
    by a different starting point rather than by the thing under test.
    """
    rng = np.random.default_rng(7)
    windows = jnp.asarray(rng.normal(size=(3, N8, cfg.H, cfg.F)))
    Lc = np.asarray(rollout.contact_factors(params, windows, cfg.eps))

    target = cfg.sigma_0 * np.eye(3)
    for t in range(Lc.shape[0]):
        for i in range(N8):
            np.testing.assert_allclose(Lc[t, i], target, atol=1e-9, err_msg=(
                f"contact {i} does not start at sigma_0*I -- iteration-0 "
                f"reproduction is broken at N=8"))


def test_the_network_can_distinguish_corners_at_all(params, cfg):
    """Capability bound for the Gate D research risk.

    Feeds two windows differing ONLY in the FK block and asserts the output moves.
    If even a large FK difference cannot change Sigma_C, then "corners come out
    identical" after training would be an architectural dead end rather than a
    finding about what is learnable. (At init the net is deliberately constant, so
    this uses perturbed params -- the question is representational capacity.)
    """
    key = jax.random.PRNGKey(3)
    p = network.init(key, d_in=cfg.d_in, widths=cfg.widths,
                     sigma_0=cfg.sigma_0, eps=cfg.eps)
    # perturb the output head off its constant initialisation
    p = jax.tree.map(lambda a: a + 0.05 * jax.random.normal(key, a.shape), p)

    rng = np.random.default_rng(11)
    base = rng.normal(size=(cfg.H, cfg.F))
    other = base.copy()
    other[:, -6:] += 1.0                      # the FK block: p_bc / v_bc channels

    a = np.asarray(network.forward(p, jnp.asarray(base.ravel()), cfg.eps))
    b = np.asarray(network.forward(p, jnp.asarray(other.ravel()), cfg.eps))
    assert np.abs(a - b).max() > 1e-6, (
        "an FK-only feature change did not move Sigma_C at all -- corners could "
        "never be distinguished, which would make the N=8 premise unreachable")


# ---------------------------------------------------------------------------
# gradients through the real N=8 scan
# ---------------------------------------------------------------------------

def _an_n8_rollout():
    hits = sorted(glob.glob(os.path.join("data", "*_n8_seed*.npz")))
    return hits[0] if hits else None


@pytest.mark.parametrize("objective", ["l2_velocity", "beta_nll"])
def test_gradients_are_finite_and_nonzero_through_the_n8_scan(objective):
    """d(loss)/d(params) through an L-step BPTT scan at N=8, both objectives.

    Uses a REAL collected N=8 rollout: the synthetic-window tests above cannot
    exercise the filter scan, and a zero/NaN gradient here is the failure that
    would waste the whole training run while every other test stayed green.
    """
    path = _an_n8_rollout()
    if path is None:
        pytest.skip("no N=8 rollout in data/ yet (Gate E collection not finished)")

    import invariant_estimation  # noqa: F401  (x64)
    from invariant_estimation.contactnet import dataset, normalize
    from invariant_estimation.sim import collect

    cfg = ContactNetConfig(objective=objective, L=16, H=10)
    c = collect.build_collector(contacts_per_foot=4, verbose=False)
    assert c.fused.n_contacts == N8

    dataset.build_channel_cache([path], c, verbose=False)
    norm = dataset.fit_normalization([path])
    prep = dataset.prepare([path], norm, c, cfg)[0]
    P0 = dataset.measure_p0(c.fused, prep, cfg)
    seg = dataset.make_segment(prep, t0=prep.warmup + 1, cfg=cfg, P0=P0)

    params = network.init(jax.random.PRNGKey(0), d_in=cfg.d_in, widths=cfg.widths,
                          sigma_0=cfg.sigma_0, eps=cfg.eps)
    seg_loss = rollout.make_segment_loss(
        c.fused.ekf, c.fused.kinematics, cfg.eps, beta=cfg.beta, objective=objective)

    (loss, _aux), grads = jax.value_and_grad(seg_loss, has_aux=True)(params, seg)

    assert np.isfinite(float(loss)), f"{objective}: loss is not finite"
    flat = np.concatenate([np.asarray(g).ravel() for g in jax.tree.leaves(grads)])
    assert np.all(np.isfinite(flat)), f"{objective}: non-finite gradient entries"
    assert np.abs(flat).max() > 0.0, (
        f"{objective}: gradient is identically zero -- training would run happily "
        f"and learn nothing")
