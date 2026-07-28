r"""Per-tick ContactNet inference — the deployment counterpart of `features.py`.

`features.make_feature_windows` builds ``(T, N_c, H, F)`` from a whole recorded
trajectory: it vmaps FK over the full leading axis, cumsums the boxcar over all
``T``, and gathers every window at once.  That shape is right for training and
impossible online, where tick ``k`` must be produced from tick ``k``'s sensors
and a bounded amount of history.

This module is the online form: a fixed-size ring buffer of normalized channels,
advanced one tick at a time, from which the same ``(N_c, H, F)`` window is
gathered.  Everything is fixed-shape and branch-free so it composes into the
fused `lax.scan` without breaking I7.

**The property that matters is agreement with the training path.**  If the
window a deployed network sees differs from the window it was trained on, the
input distribution has silently shifted and the measured gain (run 2: 3.3x in
velocity, 8.5x in height) evaporates with nothing raising.
`tests/contactnet/test_online.py` asserts that agreement against
`features.window` over a real rollout.

Two deliberate differences from training, both bounded and both tested:

* **Not bit-identical, by construction.**  `features.boxcar` cumsums over the
  whole rollout; here it cumsums over a 400-tick buffer.  A difference of two
  large partial sums is not the same floating-point number as a difference of
  two small ones — `dataset.prepare`'s docstring makes the same point about
  slicing.  Agreement is ~1e-13 relative, not exact.
* **A warm-up, instead of `window_indices`' clamp.**  Training segments are
  chosen past the lead-in so their windows never clamp (`make_segment` raises if
  they would), which means the clamped branch is a code path the network was
  never trained on.  Rather than reproduce it, `OnlineFeatures` reports
  ``ready`` only once the buffer holds `span_ticks` real samples, and the caller
  falls back to the analytic ``sigma_0**2 I`` until then.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from . import normalize as norm_mod
from .config import ContactNetConfig
from .network import ContactNetParams, forward


class OnlineState(NamedTuple):
    r"""Ring buffer of normalized per-tick channels, plus what the FK diff needs.

    Attributes
    ----------
    buf : Array, shape (span, N_c, F)
        Normalized channels, oldest first.  ``span = (H-1)*stride + stride`` —
        enough that the boxcar feeding the *oldest* gathered sample has its full
        `stride` ticks of support.
    prev_p : Array, shape (N_c, 3)
        Previous tick's body-frame contact FK, for the causal first difference
        that produces the ``B v`` channels.  Held separately because the buffer
        stores the *normalized* value and the difference is taken on the raw one.
    n : Array, scalar int
        Ticks pushed so far, saturating.  Drives `ready`.
    """
    buf: Array
    prev_p: Array
    n: Array


def span_ticks(cfg: ContactNetConfig) -> int:
    """Raw ticks the window needs: the gather span plus the boxcar's support."""
    return (cfg.H - 1) * cfg.stride + cfg.stride


def init_state(cfg: ContactNetConfig, n_c: int) -> OnlineState:
    """Empty buffer.  `ready` is False until `span_ticks` real ticks arrive."""
    return OnlineState(
        buf=jnp.zeros((span_ticks(cfg), n_c, cfg.F), dtype=jnp.float64),
        prev_p=jnp.zeros((n_c, 3), dtype=jnp.float64),
        n=jnp.asarray(0, dtype=jnp.int32),
    )


def make_online_features(subchain, base_imu: int, kinematics, cfg: ContactNetConfig,
                         constants: norm_mod.NormConstants):
    r"""Factory → ``step(state, sensors) -> (state, window, ready)``.

    A factory for the same reason `features.make_contact_channels` is: the graph
    topology, the normalization constants and the window geometry are all static
    and get closed over, so the returned callable takes only the carry and one
    tick of `pipeline.main_estimator.FusedSensors`.

    Channel order is `features.make_contact_channels`' verbatim —
    ``(omega, accel, q_sub, tau_sub, p, v)`` — and must stay that way: it is the
    same ordering `normalize` and the trained weights were fitted under.

    Returns
    -------
    callable
        ``(OnlineState, FusedSensors) -> (OnlineState, (N_c, H, F), bool)``
    """
    subchain = jnp.asarray(subchain)
    n_c, _ = subchain.shape
    mean = jnp.asarray(constants.mean, dtype=jnp.float64)
    std = jnp.asarray(constants.std, dtype=jnp.float64)
    span = span_ticks(cfg)
    # Gather offsets from the END of the buffer: the newest sample is last.
    idx = span - 1 - (cfg.H - 1 - jnp.arange(cfg.H)) * cfg.stride

    def channels_now(sensors, prev_p) -> tuple[Array, Array]:
        """One tick of raw channels ``(N_c, F)``, and this tick's FK ``p``."""
        q_all = jnp.concatenate([sensors.encoders, sensors.q_unfiltered], axis=-1)
        q_sub = q_all[subchain]                              # (N_c, J_sub)
        tau_sub = sensors.torques[subchain]                  # (N_c, J_sub)

        # RAW base IMU — never the bias-corrected gyro (I1: that is a filter
        # output, and the whole feature set is filter-state-free by design).
        omega = jnp.broadcast_to(sensors.gyros[base_imu], (n_c, 3))
        accel = jnp.broadcast_to(sensors.accel_base, (n_c, 3))

        p = kinematics(q_all, jnp.zeros_like(q_all)).y      # (N_c, 3)
        v = (p - prev_p) / cfg.dt

        row = jnp.concatenate([omega, accel, q_sub, tau_sub, p, v], axis=-1)
        return row, p

    def step(state: OnlineState, sensors):
        row, p = channels_now(sensors, state.prev_p)

        # First real tick has no predecessor, so its finite difference is not
        # defined; `make_contact_channels` sets v[0] = 0 and so does this.  It
        # only ever affects ticks inside the warm-up, which are not emitted.
        row = row.at[:, -3:].set(jnp.where(state.n == 0, 0.0, row[:, -3:]))

        row_n = (row - mean) / std                           # (N_c, F)
        buf = jnp.concatenate([state.buf[1:], row_n[None]], axis=0)
        n = jnp.minimum(state.n + 1, span)

        # Boxcar then gather — the same order as `dataset.prepare`, which
        # normalizes, smooths the whole stream, and only then windows.
        c = jnp.cumsum(buf, axis=0)
        c = jnp.concatenate([jnp.zeros_like(c[:1]), c], axis=0)
        smoothed = (c[cfg.stride:] - c[:-cfg.stride]) / cfg.stride   # (span-s+1, ...)
        # `smoothed[j]` is the average ending at buf index `j + stride - 1`.
        win = smoothed[idx - (cfg.stride - 1)]               # (H, N_c, F)

        return (OnlineState(buf=buf, prev_p=p, n=n),
                jnp.swapaxes(win, 0, 1),                     # (N_c, H, F)
                n >= span)

    return step


def make_provider(subchain, base_imu: int, kinematics, cfg: ContactNetConfig,
                  constants: norm_mod.NormConstants, params: ContactNetParams):
    r"""Factory → ``step(state, sensors) -> (state, contact_meas_chol)``.

    The deployment seam: one tick of sensors in, the ``(N_c, 3, 3)`` Cholesky
    factor `InEKFInputs.contact_meas_chol` wants out.

    Before the buffer is full this returns the analytic ``sigma_0 * I`` — the
    network's own initialization, and the value under which
    `dataset.measure_p0` and every pre-ContactNet test were taken, so the
    warm-up period reproduces the shipped filter rather than an arbitrary guess.
    The fallback is a `jnp.where`, not a branch, so the graph stays constant (I7).
    """
    feats = make_online_features(subchain, base_imu, kinematics, cfg, constants)
    n_c = jnp.asarray(subchain).shape[0]
    fallback = jnp.broadcast_to(
        cfg.sigma_0 * jnp.eye(3, dtype=jnp.float64), (n_c, 3, 3))

    def step(state: OnlineState, sensors):
        state, win, ready = feats(state, sensors)
        L = jax.vmap(forward, in_axes=(None, 0, None))(
            params, win.reshape(n_c, -1), cfg.eps)          # (N_c, 3, 3)
        return state, jnp.where(ready, L, fallback)

    return step
