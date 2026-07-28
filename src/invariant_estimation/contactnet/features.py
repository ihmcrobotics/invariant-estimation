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

import jax
import jax.numpy as jnp
import numpy as np
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

ALEX_FOOT_CHAINS: tuple[tuple[str, ...], ...] = (
    ("LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
     "LEFT_ANKLE_Y", "LEFT_ANKLE_X"),
    ("RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y",
     "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X"),
)
"""Base→foot joint chain per contact, in order.

Only the first four of each are joint-KF **states**; the two ankles are the
off-path joints, so a chain spans two different sensor arrays — which is exactly
why `build_subchain_indices` resolves into one concatenated index space rather
than carrying two.
"""

JOINT_LABELS: tuple[str, ...] = ("hip_x", "hip_z", "hip_y", "knee_y", "ankle_y", "ankle_x")
"""Side-agnostic labels for `ALEX_FOOT_CHAINS`, for `channel_names`."""


def build_subchain_indices(joint_names, unfiltered_names, foot_chains=ALEX_FOOT_CHAINS):
    r"""Resolve joint NAMES to indices into ``concat(filtered, unfiltered)``.

    One index space for both ``q`` and ``τ``: `FusedSensors.torques` already
    arrives as that concatenation, so building ``q`` the same way leaves a single
    convention instead of two that can silently drift apart.

    Resolved by name, never by assuming an index coincidence — the same
    discipline `sim.sensors.SimSensorReader` applies at the plant boundary, and
    for the same reason: a permuted gather still produces a network that trains.

    Returns
    -------
    np.ndarray, shape (N_c, J_sub)
    """
    order = list(joint_names) + list(unfiltered_names)
    pos = {n: i for i, n in enumerate(order)}
    if len(pos) != len(order):
        raise ValueError("duplicate joint name across the filtered and unfiltered sets")

    widths = {len(c) for c in foot_chains}
    if len(widths) != 1:
        raise ValueError(f"every foot chain must be the same length, got {widths}")

    idx = []
    for chain in foot_chains:
        missing = [n for n in chain if n not in pos]
        if missing:
            raise KeyError(f"joints not present in the model: {missing}")
        idx.append([pos[n] for n in chain])
    return np.asarray(idx, dtype=int)


def channel_names(joint_labels: tuple[str, ...] = JOINT_LABELS) -> tuple[str, ...]:
    r"""The frozen channel ordering — must match `make_contact_channels` exactly.

    `normalize.py`'s ``(F,)`` constants, `export.py`'s manifest and the Java
    forward pass all index against this.
    """
    return (
        "base_gyro_x", "base_gyro_y", "base_gyro_z",
        "base_accel_x", "base_accel_y", "base_accel_z",
        *(f"q_{s}" for s in joint_labels),
        *(f"tau_{s}" for s in joint_labels),
        "p_bc_x", "p_bc_y", "p_bc_z",
        "v_bc_x", "v_bc_y", "v_bc_z",
    )


def make_contact_channels(subchain, base_imu: int, kinematics, dt: float):
    r"""Factory → ``contact_channels(sensors) -> (T, N_c, F)``.

    Channel order, per contact *i* (CoCo's observation vector)::

        o_i = ( ᴮω(3), ᴮa(3), q(J_sub), τ(J_sub), ᴮp_{B→C_i}(3), ᴮv_{B→C_i}(3) )

    so ``F = 12 + 2·J_sub`` — **24** for Alex, giving ``D_in = H·F = 480``.

    Every term is body-frame, so no filter state is required — and that is
    forced, not chosen: expressing any of it in world needs ``R̂`` (§1).

    A factory for the same reason as `make_step`: ``subchain``, ``base_imu``,
    ``kinematics`` and ``dt`` are static and get closed over, so the returned
    callable takes only ``sensors``.

    Parameters
    ----------
    subchain : (N_c, J_sub) int array
        From `build_subchain_indices`.
    base_imu : int
        IMU ordinal of the star centre (`FusedEstimator.base_imu`).
    kinematics : ContactKinematics
        The ``robot/`` seam — same object `make_step` closes over.
    dt : float
        Tick period, for the ``ᴮv`` finite difference.
    """
    subchain = jnp.asarray(subchain)
    n_c, j_sub = subchain.shape

    def contact_channels(sensors) -> Array:
        """``sensors`` with a leading time axis ``T`` on every leaf."""
        if sensors.q_unfiltered.shape[-1] == 0:
            raise ValueError(
                "q_unfiltered is empty — build the estimator with "
                "contact_fk_unfiltered=True.  The base→foot chain includes the "
                "ankles, which are off-path joints and only reach FusedSensors "
                "under that flag."
            )
        # One index space for q and tau (see `build_subchain_indices`).
        q_all = jnp.concatenate([sensors.encoders, sensors.q_unfiltered], axis=-1)
        tau_all = sensors.torques
        if tau_all.shape[-1] != q_all.shape[-1]:
            raise ValueError(
                f"torques width {tau_all.shape[-1]} != concat(q) width "
                f"{q_all.shape[-1]} — the two must share one index space"
            )
        T = q_all.shape[0]

        q_sub = q_all[:, subchain]                       # (T, N_c, J_sub)
        tau_sub = tau_all[:, subchain]                   # (T, N_c, J_sub)

        # RAW base IMU.  Never the bias-corrected gyro: that correction is the
        # joint KF's b̂_ω (I1), i.e. a filter output.
        omega = jnp.broadcast_to(
            sensors.gyros[:, base_imu, :][:, None, :], (T, n_c, 3))
        accel = jnp.broadcast_to(sensors.accel_base[:, None, :], (T, n_c, 3))

        # ᴮp = FK(q), body frame.  `y` does not depend on q̇ (only `J_dot` does),
        # so zeros keep this path entirely joint-KF-free.  This is the only
        # nonlinear channel: FK(q) is not recoverable from q history by the
        # first dense layer at any H, so it is real information, not rescaling.
        zero_qd = jnp.zeros_like(q_all[0])
        p = jax.vmap(lambda q: kinematics(q, zero_qd).y)(q_all)   # (T, N_c, 3)

        # Causal first difference of ᴮp — deliberately NOT J q̇, which would
        # reintroduce the joint-KF coupling this feature set exists to avoid.
        # v[0] = 0: no earlier sample exists.
        v = jnp.concatenate(
            [jnp.zeros((1, n_c, 3), dtype=p.dtype), jnp.diff(p, axis=0) / dt], axis=0)

        return jnp.concatenate([omega, accel, q_sub, tau_sub, p, v], axis=-1)

    return contact_channels


def make_feature_windows(subchain, base_imu: int, kinematics, dt: float, H: int):
    r"""Factory → ``feature_windows(sensors) -> (T, N_c, H, F)``.

    Composes `make_contact_channels` with `window`, ready for
    `rollout.Segment.windows`.

    Normalization is **not** applied here: `normalize.py` owns the frozen
    per-channel constants and runs on the ``(T, N_c, F)`` channels *before*
    windowing, so a window never mixes normalized and raw values — and so the
    H-major flatten in `rollout.contact_factors` cannot misalign the ``(F,)``
    constants.
    """
    channels = make_contact_channels(subchain, base_imu, kinematics, dt)

    def feature_windows(sensors) -> Array:
        return window(channels(sensors), H)

    return feature_windows
