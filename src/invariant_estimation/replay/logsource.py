r"""Pull named channels out of an IHMC SCS2 hardware log as plain float64 arrays.

The input half of the Java-parity harness.  The output half -- what the Java
estimator actually produced on that run -- comes out of the *same* log through
the same reader, so inputs and reference outputs are tick-aligned by
construction, with no clock to reconcile.

The `robotData.bsz` decoder is **not vendored**: it is ~700 lines of binary
format parsing that already exist and are already exercised in the `ihmc-log`
skill's `ihmclog.py`, and a second copy would have to track logger version bumps.
This module locates that file instead, trying ``$IHMCLOG`` (an override for CI or
a moved checkout) then ``~/.claude/skills/ihmc-log/ihmclog.py``.  If neither
resolves, every entry point raises `LogToolUnavailable` with the remedy in the
message and the parity tests **skip** rather than fail: a missing log tool is a
missing fixture, not a broken port.

The sensor-processing chain (the trap this module exists to encode)
-------------------------------------------------------------------
The estimator does **not** consume ``raw_q_<joint>``.  IHMC's `SensorProcessing`
runs a chain of stages and publishes each one, so a single joint's position
appears in the log several times::

    raw_q_LEFT_KNEE_Y            <- straight off the encoder
    filt_q_LEFT_KNEE_Y_sp0       <- after the alpha filter        (stage 0)
    stiff_q_LEFT_KNEE_Y_sp1      <- after elasticity compensation (stage 1)  <-- estimator input

`AlexEstimatorLogReplay.java` feeds ``raw_*`` into a live `SensorProcessing`
instance and lets it recompute the chain.  This port has no `SensorProcessing`,
so it takes the equivalent shortcut: read the chain's **last published stage**
directly.  That is strictly more faithful than re-implementing the filters, and
it is why `joint_channel` resolves by highest ``_spN`` rather than by name.

Feeding ``raw_*`` instead would put a one-stage filtering difference between the
Java estimator and this one, which shows up as a small, configuration-dependent
parity error that looks exactly like a filter bug and is not one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np

__all__ = [
    "LogToolUnavailable",
    "LogWindow",
    "ihmclog",
    "joint_channel",
    "imu_channels",
    "read_window",
]

_DEFAULT_TOOL = Path.home() / ".claude" / "skills" / "ihmc-log" / "ihmclog.py"

# `stage_q_JOINT_spN` -- the published output of SensorProcessing stage N.
_STAGE_RE = re.compile(r"^(?P<stage>[a-zA-Z]+)_(?P<chan>q|qd|tau)_(?P<joint>.+)_sp(?P<idx>\d+)$")


class LogToolUnavailable(RuntimeError):
    """The `ihmclog` decoder could not be located."""


@lru_cache(maxsize=1)
def ihmclog() -> ModuleType:
    """Import the `ihmc-log` skill's decoder, or explain how to get it."""
    candidates = [Path(os.environ["IHMCLOG"])] if os.environ.get("IHMCLOG") else []
    candidates.append(_DEFAULT_TOOL)
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("ihmclog", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["ihmclog"] = module
            spec.loader.exec_module(module)
            return module
    raise LogToolUnavailable(
        "could not find ihmclog.py (tried "
        + ", ".join(str(c) for c in candidates)
        + "). Install the 'ihmc-log' skill or set $IHMCLOG to its path."
    )


@dataclass(frozen=True)
class LogWindow:
    """A tick-aligned slice of one log: ``time`` plus one array per channel."""

    log_dir: Path
    time: np.ndarray
    """(T,) seconds from the start of the log."""
    tick: np.ndarray
    """(T,) global tick index -- the join key back into the raw log."""
    channels: dict[str, np.ndarray]
    """``variable name -> (T,) float64``."""
    dt: float

    def __getitem__(self, name: str) -> np.ndarray:
        return self.channels[name]

    def stack(self, names: Sequence[str]) -> np.ndarray:
        """(T, k) column stack, in the order given -- assembles a joint vector or an
        IMU triple without a Python loop at the call site."""
        return np.column_stack([self.channels[n] for n in names])

    def has(self, *names: str) -> bool:
        return all(n in self.channels for n in names)


def joint_channel(reader, joint: str, chan: str) -> str:
    """The variable the *estimator* saw for ``<chan>`` (``q``/``qd``/``tau``) of
    ``<joint>``: the highest-numbered ``_spN`` stage published for it, falling back
    to ``raw_<chan>_<joint>`` only when the log has no processing chain (older
    builds, or a channel `SensorProcessing` passes through untouched)."""
    names = frozenset(reader.hs.names)
    best_idx, best_name = -1, None
    for name in names:
        m = _STAGE_RE.match(name)
        if m and m.group("joint") == joint and m.group("chan") == chan:
            idx = int(m.group("idx"))
            if idx > best_idx:
                best_idx, best_name = idx, name
    if best_name is not None:
        return best_name

    raw = f"raw_{chan}_{joint}"
    if raw in names:
        return raw
    raise KeyError(f"log has no '{chan}' channel for joint '{joint}'")


def imu_channels(imu: str) -> dict[str, list[str]]:
    """The raw gyro/accel variable names for one IMU, in XYZ order.

    *Unprocessed* is correct here, unlike the joint channels: `SensorProcessing`'s
    IMU stages are published under a different registry, and the raw triple keeps
    the bias path inside the port under test rather than borrowing Java's answer.
    """
    return {
        "gyro": [f"gyroscope_{imu}{a}" for a in "XYZ"],
        "accel": [f"accelerometer_{imu}{a}" for a in "XYZ"],
    }


def read_window(
    log_dir: str | Path,
    names: Sequence[str],
    *,
    start: float = 0.0,
    end: float | None = None,
    stride: int = 1,
    cache_dir: str | Path | None = None,
) -> LogWindow:
    """Decode ``names`` over ``[start, end)`` at ``stride`` ticks, with caching.

    Decoding is the expensive step by orders of magnitude -- a 630 s Alex log is
    131 GB uncompressed, and even a seeked, strided read of a few dozen channels
    costs one zstd frame decompress per touched batch.  The result is memoised to
    an ``.npz`` keyed by a hash of every argument that can change the numbers.

    ``cache_dir`` defaults to ``<log_dir>/.parity-cache`` when writable, else a
    temp dir -- logs often live on read-only or shared storage.
    """
    log_dir = Path(log_dir)
    names = list(dict.fromkeys(names))  # de-dup, order-preserving

    key = hashlib.sha256(
        "\0".join([str(log_dir.resolve()), repr(start), repr(end), repr(stride), *names]).encode()
    ).hexdigest()[:16]

    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir(log_dir)
    cache_file = cache_root / f"window-{key}.npz"
    if cache_file.exists():
        blob = np.load(cache_file, allow_pickle=False)
        return LogWindow(
            log_dir=log_dir,
            time=blob["__time__"],
            tick=blob["__tick__"],
            channels={n: blob[n] for n in names if n in blob.files},
            dt=float(blob["__dt__"][0]),
        )

    mod = ihmclog()
    reader = mod.LogReader(str(log_dir))
    dt = float(reader.hs.dt)
    end_sec = reader.n_ticks * dt if end is None else end

    # gather -> (ticks, time, data, var_indices, var_types); data is (n_ticks, n_vars).
    ticks, time, data, _, _ = mod.gather(reader, names, stride, start, end_sec, 0)
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    channels = {n: np.ascontiguousarray(data[:, i]) for i, n in enumerate(names)}

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            __time__=time,
            __tick__=np.asarray(ticks, dtype=np.int64),
            __dt__=np.array([dt]),
            **channels,
        )
    except OSError:
        pass  # a read-only log store is not a reason to fail the read

    return LogWindow(
        log_dir=log_dir,
        time=time,
        tick=np.asarray(ticks, dtype=np.int64),
        channels=channels,
        dt=dt,
    )


def _default_cache_dir(log_dir: Path) -> Path:
    if os.access(log_dir, os.W_OK):
        return log_dir / ".parity-cache"
    return Path(os.environ.get("TMPDIR", "/tmp")) / "invariant-parity-cache" / log_dir.name
