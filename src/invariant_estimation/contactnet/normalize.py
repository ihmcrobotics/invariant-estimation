r"""
Per-channel standardization for the ContactNet feature set.

Applied to the ``(T, N_c, F)`` channels.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
import numpy as np

# Noise floors from the working Alex001 log of the InEKF running on hardware.
NOISE_FLOOR: dict[str, float] = {
    "base_gyro": 3.0e-3,  # rad/s
    "base_accel": 4.5e-2, # m/s^2
    "q_": 4.0e-6,         # rad
    "qd_": 5.0e-3,         # rad
    "tau_": 2.0e-1,       # N.m
    "p_bc": 5.0e-6,       # m
    "v_bc": 7.1e-3        # m/s
}

def channel_floor(
    names:tuple[str, ...],
    table: dict[str, float] = NOISE_FLOOR
) -> Array:
    out = []
    for n in names:
        hit = [(len(k), v) for k, v in table.items() if n.startswith(k)]
        if not hit:
            raise KeyError(f"No noise floor for channel {n!r}; add it to the table.")
        out.append(max(hit)[1])
    return jnp.asarray(out, dtype=jnp.float64) # ensure float64 for stability in training and eval, needs to be done this way per literature.

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

    def __post__init(self):
        F = len(self.names)
        if self.mean.shape != (F,) or self.std.shape != (F,):
            raise ValueError(f"Mean/std shape mismatch: {self.mean.shape} vs {self.std.shape} vs {F}.")
        if not bool(jnp.all(self.std > 0.0)):
            raise ValueError(f"Std must be positive: {self.std}.")

def fit(channels: Array, names: tuple[str, ...], *, floor: Array | None = None, source: str = "") -> NormConstants:
    """Compute frozen constants from a ``(T, N_c, F)`` calibration set.

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
    if channels.shape[-1] != len(c.names):
        raise ValueError(f"Channel count mismatch: {channels.shape[-1]} vs {len(c.names)}")
    return (channels - c.mean) / c.std

def save(path, str, c: NormConstants) -> None:
    """
    Write the normalization constants to disk as a NPZ file. 
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
    with np.load(path, allow_pickle=False) as z:
        return NormConstants(
            mean=jnp.asarray(z["mean"],dtype=jnp.float64),
            std=jnp.asarray(z["std"],dtype=jnp.float64),
            names=tuple(str(s) for s in z["names"]),
            floored=tuple(str(s) for s in z["floored"]),
            n_ticks=int(z["n_ticks"]),
            source=str(z["source"])
        )

