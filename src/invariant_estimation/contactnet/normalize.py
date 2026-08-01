r"""Frozen per-channel standardization for ContactNet features (``network_plan.md`` §5.5).

Applied to the ``(T, N_c, F)`` channels **before** windowing.  That ordering is
load-bearing: normalizing after `rollout.contact_factors`' H-major flatten would
misalign the ``(F,)`` constants against the layout.

§5.5 rules out both alternatives for the same reason each time. *Per-window*
standardization strips absolute magnitude, which is precisely what signals slip —
a sliding foot has larger excursions than a planted one. *Running* statistics add
state and create a train/deploy mismatch.  The output is an **artifact**, not
code — see `NormConstants`.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

# Per-channel sensor NOISE std (not signal level), keyed by a
# `features.channel_names()` prefix.  Measured on the 2026-07-17 Alex001 log
# (PORT_NOTES.md, "H is not 20").  They exist to stop `fit` dividing a channel by
# its own near-zero calibration variance.
NOISE_FLOOR: dict[str, float] = {
    "base_gyro": 3.0e-3,      # rad/s   -- measured 1.5-4e-3
    "base_accel": 4.5e-2,     # m/s^2   -- measured 0.04-0.05
    "q_": 4.0e-6,             # rad     -- LOWER BOUND; the log's raw_q is
                              #            pre-filtered, so intrinsic encoder
                              #            noise is probably 1-2e-5
    "tau_": 2.0e-1,           # N.m     -- median; per-joint 0.10-0.28
    "p_bc": 5.0e-6,           # m       -- propagated, J sigma_q, not measured
    "v_bc": 7.1e-3,           # m/s     -- propagated, = sqrt(2)*sigma_p/dt; see
                              #            `channel_floor` for why there is no
                              #            1/stride here
}


def channel_floor(
    names: tuple[str, ...],
    table: dict[str, float] = NOISE_FLOOR
) -> Array:
    """Per-channel std floor, ``(F,)``, by **longest**-prefix match on ``names``.

    Per-channel rather than scalar because the floors span five orders of
    magnitude (4e-6 rad to 0.2 N.m).

    Longest-prefix, not largest-value: a more specific key must win even when it
    is *smaller*.  With ``{"q_": 4e-6, "q_ankle": 1e-8}``, ``q_ankle_y`` resolves
    to 1e-8.  Taking the max instead would silently ignore every specific
    override that tightened a floor.

    Note on ``v_bc`` -- read this before "correcting" its floor downward.
    ``v_bc`` is a first difference at 1 kHz, so it amplifies position noise by
    ``sqrt(2)/dt``: from ``sigma_p = 5e-6 m`` that is ``7.1e-3 m/s``, three
    orders above the position floor.  It is tempting to divide that by ``stride``,
    because `features.window` boxcar-averages before subsampling and the two
    telescope::

        boxcar_s(diff(p)/dt)[k] = (p[k] - p[k-s]) / (s*dt)

    **That reduction does not apply here.**  `fit` and `apply` run on the
    ``(T, N_c, F)`` channels *before* windowing, so at the moment the floor is
    compared against ``raw_std`` the boxcar has not happened; the amplification
    is the full ``sqrt(2)/dt``.  An earlier version of this table carried
    ``1e-6`` here, which is ``7071x`` too small and therefore inert: the floor
    could never fire for ``v_bc``.  It would have gone unnoticed while walking,
    and failed exactly the case the floor exists for -- a standing calibration
    set, where ``v_bc`` is almost entirely noise.

    Raises ``KeyError`` if any channel matches no prefix: a new channel without a
    floor is a gap, not a default.
    """
    out = []
    for n in names:
        hit = [(len(k), v) for k, v in table.items() if n.startswith(k)]
        if not hit:
            raise KeyError(f"no noise floor for channel {n!r}; add it to the table")
        out.append(max(hit)[1])          # longest prefix wins
    return jnp.asarray(out, dtype=jnp.float64)

@dataclass(frozen=True)
class NormConstants:
    """Frozen per-channel standardization: computed once over a calibration set,
    written to disk, loaded at train *and* deploy, and baked into the Java export.

    Constants that cannot be traced back to the run that produced them cannot be
    debugged, so the provenance fields travel *with* the numbers.

    A frozen dataclass rather than a NamedTuple on purpose: this is never a
    pytree. A NamedTuple carrying `names` would make those strings pytree
    *leaves*, and any `jax.tree.map` over it would treat them as arrays — the
    same trap `ContactNetParams` avoids by holding only arrays.
    """
    mean: Array
    """(F,) per-channel mean over the calibration set."""

    std: Array
    """(F,) per-channel std, already floored (see `fit`)."""

    names: tuple[str, ...]
    """== `features.channel_names()`. The guard `apply` checks against."""

    n_ticks: int
    """Calibration-set size, ``T * N_c``. Provenance only."""

    source: str
    """Which rollouts, when, what gait. Free text -- but write something."""

    floored: tuple[str, ...]
    """Channels whose raw std fell below the floor.

    A prompt to look, not a status line: it means either the channel is genuinely
    at the noise level, or the calibration set never exercised it.  Measured
    example of the latter -- a standing-only set floors ``base_gyro_y/z``, which
    move perfectly well while walking.
    """

    def __post_init__(self):
        F = len(self.names)
        if self.mean.shape != (F,) or self.std.shape != (F,):
            raise ValueError(
                f"mean/std must be ({F},) to match names, got "
                f"{self.mean.shape} and {self.std.shape}"
            )
        if not bool(jnp.all(self.std > 0)):
            raise ValueError("every std must be > 0 (apply a floor in fit)")


def fit(channels: Array, names: tuple[str, ...], *, floor: Array | None = None, source: str = "") -> NormConstants:
    """Compute frozen constants from a ``(T, N_c, F)`` calibration set.

    Reduces over BOTH time and contacts: the contacts share one weight set
    (§3.1), so they must share one normalization, or contact 0 and contact 1
    arrive at the network in different units.

    The floor is the load-bearing part.  On a standing calibration set some
    channels have near-zero variance, and dividing by those builds a huge gain on
    what is currently just sensor noise; the robot then walks, those channels
    move for real, and they saturate the trunk.  The floor is therefore set
    against SENSOR NOISE, not against calibration variance.
    """
    if channels.ndim != 3:
        raise ValueError(f"Expected (T, N_c, F), got {channels.shape}")
    T, N_c, F = channels.shape
    if F != len(names):
        raise ValueError(f"channels have F={F} but {len(names)} names were given.")
    floor = channel_floor(names) if floor is None else floor
    mean = channels.mean(axis=(0,1))
    raw_std = channels.std(axis=(0,1))
    std = jnp.maximum(raw_std, floor)

    hit = tuple(n for n, r, f in zip(names, raw_std, floor) if r < f)
    return NormConstants(
        mean=mean,
        std=std,
        names=tuple(names),
        n_ticks=int(T * N_c),
        source=source,
        floored=hit
    )

def apply(channels: Array, c: NormConstants) -> Array:
    """``(channels - mean) / std`` over the last axis, with a shape guard.

    Refuses rather than broadcasts.  A stale artifact whose ``F`` no longer
    matches `features.channel_names()` is exactly the silent failure this module
    exists to prevent: NumPy would happily broadcast a mismatched trailing axis in
    some shape combinations, and the result would train fine and be wrong.

    NOT optional despite being cheap: §7 bakes these constants into the Java
    export, and any disagreement between the training transform and the deployed
    one is undetectable from either side alone.
    """
    if channels.shape[-1] != len(c.names):
        raise ValueError(
            f"Channels have F={channels.shape[-1]} but the constants were fit "
            f"for F={len(c.names)} ({c.source!r}) -- stale artifact?"
        )
    return (channels - c.mean) / c.std

def save(path: str, c: NormConstants) -> None:
    """Write the artifact to ``.npz``. Provenance travels with the numbers.

    ``allow_pickle`` stays off on the `load` side, so everything here must be a
    plain array -- which is why `names`, `source` and `floored` are stored as
    string arrays rather than as an object blob.
    """
    np.savez(
        path,
        mean=np.asarray(c.mean),
        std=np.asarray(c.std),
        names=np.asarray(c.names),
        floored=np.asarray(c.floored),
        n_ticks=np.asarray(c.n_ticks),
        source=np.asarray(c.source)
    )

def load(path: str) -> NormConstants:
    """Read the artifact back, restoring float64 and the tuple fields.

    ``allow_pickle=False`` on purpose: these constants are loaded at deploy, and
    a pickle is arbitrary code execution.  It also forces the save side to stay
    plain-array, which is what makes the file readable from Java (§7).
    """
    with np.load(path, allow_pickle=False) as z:
        return NormConstants(
            mean = jnp.asarray(z["mean"], dtype=jnp.float64),
            std = jnp.asarray(z["std"], dtype=jnp.float64),
            names=tuple(str(s) for s in z["names"]),
            n_ticks=int(z["n_ticks"]),
            source=str(z["source"]),
            floored=tuple(str(s) for s in z["floored"])
        )
