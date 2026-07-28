r"""`contactnet/online.py` — the deployed window must be the trained window.

The single property this file exists for: the ``(N_c, H, F)`` window produced
one tick at a time from a ring buffer must equal the one
`features.make_feature_windows` produces from the whole trajectory.  If they
differ, the deployed network sees an input distribution it was never trained on,
and the measured benefit (run 2: 3.3x in body-frame velocity, 8.5x in height,
`experiments/replay_eval.py`) silently disappears with nothing raising.

Exact equality is not the bar and cannot be — `features.boxcar` cumsums over the
whole rollout while the online form cumsums over a 400-tick buffer, and a
difference of two large partial sums is not the same float as a difference of two
small ones.  `dataset.prepare`'s docstring makes the same point about slicing.
The bar is agreement to float64 accumulation error, and the tests assert their
own non-vacuity so "agrees" cannot pass by both sides being constant.

Everything runs on a fabricated trajectory and the `tests/inEKF` fixture
kinematics; nothing here needs MJX or a 130 MB rollout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect)
from invariant_estimation.contactnet import features, network, normalize, online
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.pipeline.main_estimator import FusedSensors

N_C = 2
J_SUB = 6
N_J = 8
T = 600


def _cfg(**kw) -> ContactNetConfig:
    """Small but structurally identical: H=5, stride=8 -> span = 40 ticks."""
    base = dict(F=12 + 2 * J_SUB, sigma_0=1.0e-4, H=5, window_span_s=0.032,
                dt=1.0e-3, widths=(16, 16))
    base.update(kw)
    return ContactNetConfig(**base)


SUBCHAIN = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
BASE_IMU = 1


class _FK(NamedTuple):
    y: jnp.ndarray


def _kinematics(q, qd):
    """Stand-in for `ContactKinematics`: a smooth nonlinear function of q.

    Nonlinear on purpose — the FK channel is the one `features` documents as
    real information rather than a rescaling of the q history, so a linear stub
    would make the window agreement easier than it is in the real pipeline.
    """
    y = jnp.stack([
        jnp.array([jnp.sin(q[i]) + 0.3 * jnp.cos(q[i + 1]),
                   jnp.cos(q[i]) * 0.5,
                   -0.9 + 0.1 * jnp.sin(q[i + 2])])
        for i in (0, 1)
    ])
    return _FK(y=y)


def _sensors(seed: int = 0) -> FusedSensors:
    """The real `FusedSensors` NamedTuple, with a leading time axis.

    The real type and not a stand-in: it is a pytree, so `jax.tree.map` slices a
    tick out of it and `lax.scan` can carry it — which is exactly the path the
    deployed provider takes.
    """
    r = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    q = 0.4 * np.sin(0.01 * t + np.arange(N_J)[None, :]) + 0.01 * r.standard_normal((T, N_J))
    return FusedSensors(
        encoders=jnp.asarray(q[:, :5]),
        q_unfiltered=jnp.asarray(q[:, 5:]),
        torques=jnp.asarray(20.0 * np.sin(0.02 * t + np.arange(N_J)[None, :])),
        gyros=jnp.asarray(r.standard_normal((T, 3, 3)) * 0.1),
        accel_base=jnp.asarray(r.standard_normal((T, 3)) * 0.5 + np.array([0, 0, 9.81])),
        qd_unfiltered=jnp.zeros((T, 0)),
        contact=jnp.zeros((T, N_C)),
        contact_chol=jnp.zeros((T, N_C, 3, 3)),
    )


def _tick(sensors, k):
    return jax.tree.map(lambda a: a[k], sensors)


@pytest.fixture(scope="module")
def fitted():
    """Norm constants fitted on the offline channels, as `train_contactnet` does."""
    cfg = _cfg()
    s = _sensors()
    chan = features.make_contact_channels(SUBCHAIN, BASE_IMU, _kinematics, cfg.dt)(s)
    # Real channel names: `normalize`'s per-channel noise-floor table is keyed by
    # prefix and rejects unknown channels outright, so placeholders would not
    # exercise the same floors the trained constants were fitted under.
    names = features.channel_names()
    assert len(names) == cfg.F, f"fixture F={cfg.F} != {len(names)} real channels"
    return cfg, s, chan, normalize.fit(chan, names, source="test")


def _offline_windows(cfg, chan, nc):
    """The training reference: normalize, then `features.window`.

    `window` applies the boxcar **itself** (`features.window`: ``smoothed =
    boxcar(channels, stride)`` before the gather), so calling `boxcar` here too
    would smooth twice.  `dataset.prepare` pre-boxcars only because
    `make_segment` then gathers directly instead of going through `window`.
    """
    return features.window(normalize.apply(jnp.asarray(chan), nc),
                           cfg.H, cfg.stride)


def _run_online(cfg, sensors, nc):
    """Drive the ring buffer tick by tick; return windows and readiness."""
    step = online.make_online_features(SUBCHAIN, BASE_IMU, _kinematics, cfg, nc)
    st = online.init_state(cfg, N_C)
    wins, ready = [], []
    for k in range(T):
        st, w, r = step(st, _tick(sensors, k))
        wins.append(np.asarray(w))
        ready.append(bool(r))
    return np.stack(wins), np.asarray(ready)


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------

def test_online_window_matches_the_offline_window(fitted):
    r"""Past the warm-up, the ring-buffer window equals `features.window`.

    Kills: a stride/offset error in the gather, a boxcar aligned to the wrong
    end of its support, normalize applied after the boxcar instead of before,
    and a buffer written oldest-last.
    """
    cfg, sensors, chan, nc = fitted
    off = np.asarray(_offline_windows(cfg, chan, nc))
    on, ready = _run_online(cfg, sensors, nc)

    span = online.span_ticks(cfg)
    assert ready[span - 1:].all() and not ready[:span - 1].any()

    err = np.abs(on[ready] - off[ready])
    scale = max(1.0, float(np.abs(off[ready]).max()))
    assert err.max() / scale < 1e-10, f"max abs diff {err.max():.3e}"

    # Non-vacuity 1: the windows are not constant, so equality is a real
    # constraint rather than two arrays of zeros agreeing.
    assert float(off[ready].std()) > 0.1
    # Non-vacuity 2: a one-tick offset does NOT match, so the alignment is
    # actually pinned.
    shifted = off[ready][:-1]
    assert np.abs(on[ready][1:] - shifted).max() / scale > 1e-3


def test_warmup_emits_the_analytic_fallback_not_a_garbage_window(fitted):
    r"""Before the buffer fills, the provider returns **zeros** exactly.

    Zeros is what `pipeline.main_estimator._boundary` passes in the analytic
    filter, so the warm-up reproduces the pre-ContactNet filter bit-for-bit. The
    warm-up window is built on zero-padded history — a region of input space the
    network never saw — so emitting its output would be a silent transient at
    every filter start.  Kills: a provider that runs the network regardless of
    `ready`.
    """
    cfg, sensors, chan, nc = fitted
    params = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    # Move the head off its initialization so "network output" and "fallback"
    # are distinguishable — at init they are equal by construction (§4).
    params = params._replace(head=params.head._replace(
        W=params.head.W + 0.05, b=params.head.b + 0.05))

    step = online.make_provider(SUBCHAIN, BASE_IMU, _kinematics, cfg, nc, params)
    st = online.init_state(cfg, N_C)
    span = online.span_ticks(cfg)
    want = np.zeros((3, 3))

    out = []
    for k in range(span + 5):
        st, L = step(st, _tick(sensors, k))
        out.append(np.asarray(L))

    for k in range(span - 1):
        assert np.allclose(out[k], np.broadcast_to(want, (N_C, 3, 3)), atol=0), (
            f"tick {k} inside warm-up did not emit the fallback")
    # And it really does switch over — otherwise the test passes on a provider
    # that returns the fallback forever.
    assert not np.allclose(out[-1], np.broadcast_to(want, (N_C, 3, 3)), atol=1e-9)


def test_provider_output_matches_the_offline_network_output(fitted):
    r"""End to end: online `Sigma_C` equals what training-path windows produce.

    The window test above is the mechanism; this is the quantity the filter
    actually consumes, so it is the one that would be wrong if anything between
    the window and the Cholesky factor disagreed.
    """
    cfg, sensors, chan, nc = fitted
    params = network.init(jax.random.PRNGKey(1), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    params = params._replace(head=params.head._replace(
        W=params.head.W + 0.02, b=params.head.b + 0.02))

    off = _offline_windows(cfg, chan, nc)
    off_L = jax.vmap(jax.vmap(lambda x: network.forward(params, x, cfg.eps)))(
        off.reshape(T, N_C, -1))

    step = online.make_provider(SUBCHAIN, BASE_IMU, _kinematics, cfg, nc, params)
    st = online.init_state(cfg, N_C)
    got = []
    for k in range(T):
        st, L = step(st, _tick(sensors, k))
        got.append(np.asarray(L))
    got = np.stack(got)

    span = online.span_ticks(cfg)
    ref = np.asarray(off_L)[span - 1:]
    err = np.abs(got[span - 1:] - ref)
    assert err.max() / max(1e-12, float(np.abs(ref).max())) < 1e-9

    # Non-vacuity: Sigma_C is genuinely varying over the trajectory.
    sd = np.sqrt(np.einsum("tnij,tnij->tn", ref, ref))
    assert sd.std() / sd.mean() > 1e-3


def test_state_is_fixed_shape_across_ticks(fitted):
    """The carry must not change shape — it goes inside the fused scan (I7)."""
    cfg, sensors, chan, nc = fitted
    step = online.make_online_features(SUBCHAIN, BASE_IMU, _kinematics, cfg, nc)
    st = online.init_state(cfg, N_C)
    shapes = [tuple(x.shape for x in jax.tree.leaves(st))]
    for k in range(50):
        st, _, _ = step(st, _tick(sensors, k))
        shapes.append(tuple(x.shape for x in jax.tree.leaves(st)))
    assert len(set(shapes)) == 1, f"carry changed shape: {set(shapes)}"


def test_online_step_is_jittable_and_scannable(fitted):
    """It has to survive `jit` and `lax.scan` to reach the fused step at all."""
    cfg, sensors, chan, nc = fitted
    step = online.make_online_features(SUBCHAIN, BASE_IMU, _kinematics, cfg, nc)

    def body(st, s):
        st, w, r = step(st, s)
        return st, w

    _, wins = jax.jit(lambda s: jax.lax.scan(
        body, online.init_state(cfg, N_C), s))(sensors)
    assert wins.shape == (T, N_C, cfg.H, cfg.F)
    assert bool(jnp.all(jnp.isfinite(wins)))

    ref, _ = _run_online(cfg, sensors, nc)
    assert np.abs(np.asarray(wins) - ref).max() < 1e-12
