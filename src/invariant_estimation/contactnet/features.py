r"""
contactnet/features.py
======================
Per-contact feature extraction and history windowing for ContactNet
(``network_plan.md`` §2, §9 step 2).

Two halves, deliberately separated:

* **Windowing** (`window_indices`, `window`) — pure plumbing with an exact
  oracle in ``tests/contactnet/test_features.py``.  Mechanical.
* **Channel extraction** (`contact_channels`) — **HAND-AUTHOR**.  Which
  channels, which subchain joints, what ordering: convention-bound, with no
  cheap oracle.  A wrong choice here produces a network that trains, converges
  and exports, and is simply worse, with nothing to fail.

This module imports nothing from ``inEKF.state`` on purpose.  ContactNet sees
**sensor history only** (§1); if the signatures here cannot see filter state,
that invariant cannot be broken by accident.
"""

import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Windowing (mechanical — oracle-tested)
# ---------------------------------------------------------------------------

def window_indices(T: int, H: int) -> Array:
    r"""``(T, H)`` index matrix: row ``k`` holds the ticks feeding the window at ``k``.

    ``idx[k, h] = k - (H - 1) + h``, so every row **ends** at ``k``.  Causality is
    structural, not incidental: ``idx[k, h] <= k`` holds for every entry, and
    ``tests/contactnet/test_features.py`` asserts exactly that.  A window that
    peeks at ``k + 1`` trains beautifully and cannot be deployed, and nothing
    anywhere raises — which is why it gets its own oracle.

    Lower-clamped at 0, so rows before ``H - 1`` repeat the earliest sample.
    Prefer feeding ``H - 1`` ticks of lead-in and slicing them off afterwards:
    clamping fabricates windows the network never encounters at deployment, and
    it does so exactly where the filter state was freshly reseeded, so the two
    artifacts compound.

    Only the lower bound is clamped.  An upper clamp would *hide* a
    future-peeking bug rather than expose it — JAX silently clamps
    out-of-bounds gathers, so the test is the only thing standing between you
    and a leak.

    Parameters
    ----------
    T : int
        Number of ticks.
    H : int
        History length per evaluation.

    Returns
    -------
    Array, shape (T, H)
    """
    k = jnp.arange(T)[:, None]                      # (T, 1)
    h = jnp.arange(H)[None, :]                      # (1, H)
    return jnp.maximum(k - (H - 1) + h, 0)


def window(channels: Array, H: int) -> Array:
    r"""``(T, N_c, F)`` per-tick channels → ``(T, N_c, H, F)`` windows.

    One gather on a constant-shape index matrix: jit-safe, no scan, and it lands
    directly in the layout `rollout.contact_factors` expects.

    The gather produces ``(T, H, N_c, F)`` — the index axis lands where the time
    axis was, pushing contacts right — so one ``swapaxes`` is required.  Skipping
    it interleaves contacts into the history axis, which (unlike the H/F flatten
    ordering) is wrong in Python too, not just in the Java port.

    Parameters
    ----------
    channels : Array, shape (T, N_c, F)
        Per-tick, per-contact channels from `contact_channels`, already
        normalized (see `feature_windows`).
    H : int
        History length per evaluation.

    Returns
    -------
    Array, shape (T, N_c, H, F)
    """
    if channels.ndim != 3:
        raise ValueError(f"expected (T, N_c, F), got shape {channels.shape}")
    idx = window_indices(channels.shape[0], H)      # (T, H)
    gathered = channels[idx]                        # (T, H, N_c, F)
    return jnp.swapaxes(gathered, 1, 2)             # (T, N_c, H, F)


# ---------------------------------------------------------------------------
# HAND-AUTHOR boundary (network_plan.md §2)
#
# Everything below is convention-bound.  Author it by hand, then freeze it.
# ---------------------------------------------------------------------------

CHANNEL_NAMES: tuple[str, ...] = ()
"""Per-subchain-joint channel names, in order.  TODO(Lucas).

Frozen forever once chosen: `normalize.py`'s ``(F,)`` constants, `export.py`'s
manifest and the Java forward pass all index against this ordering.
"""

CONTACT_SUBCHAIN: tuple[tuple[int, ...], ...] = ()
"""Per-contact joint indices, base→foot.  TODO(Lucas).

Must be the same length for every contact, or the ``(N_c, H, F)`` shape is a
lie — `contact_channels` asserts it.
"""


def contact_channels(sensors) -> Array:
    r"""Per-tick, per-contact raw channels: ``→ (T, N_c, F)``, ``F = C * J_sub``.

    HAND-AUTHOR.  Two rules it must obey:

    1. **Sensor history only** (§1).  Never `InEKFState`, `InEKFCarry`, or
       anything derived from the estimate.  Violating this breaks group
       affinity silently while the code keeps running.

    2. **Body / contact frame only.**  Not a style preference: rotating a
       quantity to world requires ``R̂``, which *is* filter state.  Rule 1
       therefore decides the frame question — gyro and accel stay as the IMU
       measures them.

    Suggested starting channels per subchain joint: ``q``, ``q̇``, ``τ``.
    Whatever is chosen, record it in `CHANNEL_NAMES` and never reorder.
    """
    raise NotImplementedError(
        "contact_channels is hand-authored — see network_plan.md §2 and §9 step 2"
    )


def feature_windows(sensors, H: int) -> Array:
    r"""``sensors → (T, N_c, H, F)``, ready for `rollout.Segment.windows`.

    Composes `contact_channels` with `window`.  Normalization is **not** applied
    here: `normalize.py` owns the frozen per-channel constants and runs on the
    ``(T, N_c, F)`` channels *before* windowing, so a window never mixes
    normalized and raw values — and so the H-major flatten in
    `rollout.contact_factors` cannot misalign the ``(F,)`` constants.
    """
    return window(contact_channels(sensors), H)
