from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import invariant_estimation # noqa: F401
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet.dataset import _segment_window_indices, valid_start_range
from invariant_estimation.contactnet.features import window, window_indices

T, N_C, F, H = 40, 2, 3, 5


def _stream(seed=0):
    """(T, N_c, F) with every entry distinct, so a gather bug cannot alias."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(T, N_C, F)))


def test_window_shape_and_newest_last():
    w = window(_stream(), H)
    assert w.shape == (T, N_C, H, F)
    # index -1 of the history axis is the CURRENT tick; the network's input
    # ordering depends on this and nothing else asserts it.
    assert jnp.array_equal(w[:, :, -1, :], _stream())


def test_window_is_consecutive_raw_ticks():
    """No smoothing, no decimation: row k is literally x[k-H+1 : k+1]."""
    x = _stream()
    w = window(x, H)
    for k in range(H - 1, T):
        expect = jnp.swapaxes(x[k - H + 1:k + 1], 0, 1)     # (N_c, H, F)
        assert jnp.array_equal(w[k], expect)


def test_head_rows_clamp_at_tick_zero():
    x = _stream()
    w = window(x, H)
    assert jnp.array_equal(w[0], jnp.broadcast_to(x[0][:, None, :], (N_C, H, F)))
    # row H-2 has exactly one clamped sample at the oldest slot
    assert jnp.array_equal(w[H - 2][:, 0, :], x[0])
    assert jnp.array_equal(w[H - 2][:, 1, :], x[0])


def test_window_indices_are_consecutive():
    idx = np.asarray(window_indices(T, H))
    assert idx.shape == (T, H)
    unclamped = idx[H - 1:]
    assert np.all(np.diff(unclamped, axis=1) == 1), "history samples must be adjacent ticks"
    assert np.array_equal(unclamped[:, -1], np.arange(H - 1, T))


def test_segment_gather_matches_full_stream_window():
    """The property the global boxcar used to buy, now free.

    A training segment gathers its windows out of the whole normalized stream;
    validation windows the whole stream at once. Those must be the same numbers,
    or the net is scored on an input distribution it was never trained on.
    """
    cfg = ContactNetConfig(H=H, L=8, B=2, warm_in_s=0.5, episode_s=2.0)
    x = _stream(1)
    full = window(x, cfg.H)
    t_lo, _ = valid_start_range(T, warmup=3, cfg=cfg)
    for t0 in (t_lo, t_lo + 1, T - cfg.L):
        seg = np.asarray(x)[_segment_window_indices(t0, cfg)]   # (L, H, N_c, F)
        seg = np.swapaxes(seg, 1, 2)                            # (L, N_c, H, F)
        assert np.array_equal(seg, np.asarray(full[t0:t0 + cfg.L]))


def test_segment_window_indices_reject_a_clamping_start():
    cfg = ContactNetConfig(H=H, L=8, B=2, warm_in_s=0.5, episode_s=2.0)
    import pytest
    with pytest.raises(ValueError, match="clamp"):
        _segment_window_indices(cfg.H - 2, cfg)
