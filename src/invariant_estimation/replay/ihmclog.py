r"""Minimal reader for IHMC SCS2 robot logs (`robotData.bsz` + `robotData.dat`).

Ported from the authoritative Java implementation rather than reverse-engineered:
`us.ihmc.scs2.session.log.LogDataReader` (scs2-session-logger) for the container
and record layout, `us.ihmc.robotDataLogger.LogIndex` for the index, and
`IDLYoVariableHandshakeParser` for variable ordering. Where this disagrees with
those, they are right.

Scope: read named YoVariables over a tick window, as float64. That is the whole
API `invariant_estimation.replay.logsource` needs. Joint states, graphics,
reference frames, enums-as-strings and the websocket/live paths are all out.

Format
------
``robotData.log`` (java .properties) declares everything variable:

    variables.compressed=true
    variables.compressionType=zstd          # or snappy / none
    variables.compressionBatchSize=25       # ticks per compressed frame
    variables.validTicksInLastBatch=16
    variables.index=robotData.dat
    variables.data=robotData.bsz

``handshake.yaml`` is plain YAML. Its ``variables:`` list order **is** the record
order, and ``dt`` is the sample period. ``joints:`` sizes the tail of each record.

``robotData.dat`` is 16 bytes per **batch** (not per tick), big-endian:
``(int64 timestamp, int64 offset into robotData.bsz)``. A batch's compressed size
is the gap to the next offset; the last runs to end-of-file.

Each batch decompresses to ``batch_size * tick_size`` bytes, where

    tick_size = (1 + n_variables + n_joint_state_variables) * 8

and one tick is big-endian int64:

    [0]                    timestamp
    [1 .. 1+n_variables)   one per YoVariable, as **raw long bits**
    [1+n_variables .. ]    joint states

"Raw long bits" is the trap worth naming: a double is stored as its IEEE-754 bit
pattern in an int64, so reading the int64 as a number gives a huge meaningless
integer rather than an obviously-wrong value. Interpretation is by the
handshake's declared ``type``.
"""
from __future__ import annotations

import io
import re
import struct
from pathlib import Path

import numpy as np

#: Joint-state variable counts, from `SixDoFState`/`OneDoFState.numberOfStateVariables`.
JOINT_STATE_SIZES = {"SiXDoFJoint": 13, "SixDoFJoint": 13, "OneDoFJoint": 2}

#: How a raw int64 becomes a number, keyed by the handshake's `type`.
_INT_TYPES = {"IntegerYoVariable", "LongYoVariable", "BooleanYoVariable", "EnumYoVariable"}

# The optional "- " matters: in `variables:` the key sits on its own line, but in
# `joints:` it is the first key of the list item (`  - name: "PELVIS_LINK"`).
_NAME_RE = re.compile(r'^\s*(?:-\s+)?name:\s*"(?P<value>[^"]*)"\s*$')
_TYPE_RE = re.compile(r'^\s*(?:-\s+)?type:\s*"(?P<value>[^"]*)"\s*$')
_DT_RE = re.compile(r"^\s*dt:\s*(?P<value>[0-9.eE+-]+)\s*$")


class Handshake:
    """Variable names, types and ordering, plus ``dt`` — parsed from handshake.yaml.

    Line-scanned rather than YAML-parsed: these files reach 8 MB and only three
    of their sections are wanted. The scan tracks which top-level section it is
    in, because ``joints:`` entries carry ``name:``/``type:`` keys too and would
    otherwise be picked up as variables.
    """

    def __init__(self, path):
        self.names: list[str] = []
        self.types: list[str] = []
        self.joint_types: list[str] = []
        self.dt: float | None = None

        section = None
        pending_name = None
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                # dt BEFORE the section test: `  dt: 0.001` is indented exactly like a
                # section header and would otherwise be consumed as one.
                if self.dt is None:
                    dt = _DT_RE.match(line)
                    if dt:
                        self.dt = float(dt.group("value"))
                        continue
                # A section header has nothing after the colon; a scalar key does.
                header = re.match(r"^  (?P<section>[A-Za-z]+):\s*$", line)
                if header:
                    section = header.group("section")
                    pending_name = None
                    continue
                if section not in ("variables", "joints"):
                    continue
                name = _NAME_RE.match(line)
                if name:
                    pending_name = name.group("value")
                    continue
                kind = _TYPE_RE.match(line)
                if kind and pending_name is not None:
                    if section == "variables":
                        self.names.append(pending_name)
                        self.types.append(kind.group("value"))
                    else:
                        self.joint_types.append(kind.group("value"))
                    pending_name = None

        if not self.names:
            raise ValueError(f"{path}: no variables found; unexpected handshake format")
        if self.dt is None:
            raise ValueError(f"{path}: no dt found")

        self.index = {name: i for i, name in enumerate(self.names)}
        if len(self.index) != len(self.names):
            # Simple names are not unique across registries. The first wins, matching how
            # logsource addresses channels; a name that collides is reported so a caller
            # asking for an ambiguous one is not silently handed the wrong registry's copy.
            self.duplicated = {n for n in self.names if self.names.count(n) > 1} if len(self.names) < 50000 else set()
        else:
            self.duplicated = set()

    @property
    def n_variables(self) -> int:
        return len(self.names)

    @property
    def n_joint_state_variables(self) -> int:
        total = 0
        for kind in self.joint_types:
            if kind not in JOINT_STATE_SIZES:
                raise ValueError(f"unknown joint type {kind!r}; cannot size a record")
            total += JOINT_STATE_SIZES[kind]
        return total


def _read_properties(path) -> dict[str, str]:
    values = {}
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


class LogReader:
    """Random-access reader over one log directory."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        properties = _read_properties(self.log_dir / "robotData.log")
        self.properties = properties

        self.hs = Handshake(self.log_dir / properties.get("variables.handshake", "handshake.yaml"))

        self.compressed = properties.get("variables.compressed", "false").lower() == "true"
        self.compression = properties.get("variables.compressionType", "none").strip().lower() or "none"
        self.batch_size = max(1, int(properties.get("variables.compressionBatchSize", "1") or 1))
        valid_last = int(properties.get("variables.validTicksInLastBatch", "0") or 0)
        self.valid_ticks_in_last_batch = valid_last if self.batch_size > 1 and valid_last > 0 else self.batch_size

        self.tick_size = (1 + self.hs.n_variables + self.hs.n_joint_state_variables) * 8

        self.data_path = self.log_dir / properties.get("variables.data", "robotData.bsz")
        index_path = self.log_dir / properties.get("variables.index", "robotData.dat")
        raw = np.fromfile(index_path, dtype=">i8")
        if raw.size % 2:
            raise ValueError(f"{index_path}: not a whole number of 16-byte entries")
        entries = raw.reshape(-1, 2)
        self.batch_timestamps = np.ascontiguousarray(entries[:, 0])
        self.batch_offsets = np.ascontiguousarray(entries[:, 1])
        if self.batch_offsets.size == 0:
            raise ValueError(f"{index_path}: empty index")

        size = self.data_path.stat().st_size
        self.batch_sizes = np.empty(self.batch_offsets.size, dtype=np.int64)
        self.batch_sizes[:-1] = np.diff(self.batch_offsets)
        self.batch_sizes[-1] = size - self.batch_offsets[-1]
        if np.any(self.batch_sizes <= 0):
            raise ValueError(f"{index_path}: non-monotonic offsets")

        self.n_batches = int(self.batch_offsets.size)
        self.n_ticks = (self.n_batches - 1) * self.batch_size + self.valid_ticks_in_last_batch
        self._stream = open(self.data_path, "rb", buffering=0)
        self._cached_batch = -1
        self._cached_array = None

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _decompress(self, payload: bytes) -> bytes:
        if not self.compressed or self.compression == "none":
            return payload
        if self.compression == "zstd":
            import zstandard

            expected = self.batch_size * self.tick_size
            # max_output_size, not a streaming reader: these frames are written without a
            # content-size header, so the decompressor has to be told how big a batch is.
            return zstandard.ZstdDecompressor().decompress(payload, max_output_size=expected)
        raise NotImplementedError(f"compression {self.compression!r} is not implemented")

    def batch(self, index: int) -> np.ndarray:
        """One decompressed batch as ``(ticks_in_batch, values_per_tick)`` big-endian int64."""
        if not 0 <= index < self.n_batches:
            raise IndexError(f"batch {index} out of range [0, {self.n_batches})")
        if index == self._cached_batch:
            return self._cached_array

        self._stream.seek(int(self.batch_offsets[index]))
        payload = self._stream.read(int(self.batch_sizes[index]))
        if len(payload) != int(self.batch_sizes[index]):
            raise IOError(f"short read for batch {index}")
        raw = self._decompress(payload)
        if len(raw) % self.tick_size:
            raise ValueError(
                f"batch {index} decompressed to {len(raw)} bytes, not a multiple of the "
                f"{self.tick_size}-byte tick record; handshake and data disagree"
            )
        array = np.frombuffer(raw, dtype=">i8").reshape(-1, self.tick_size // 8)
        self._cached_batch, self._cached_array = index, array
        return array

    def tick(self, index: int) -> np.ndarray:
        """One tick's raw int64 record."""
        if not 0 <= index < self.n_ticks:
            raise IndexError(f"tick {index} out of range [0, {self.n_ticks})")
        batch = self.batch(index // self.batch_size)
        within = index % self.batch_size
        if within >= batch.shape[0]:
            raise IndexError(f"tick {index} is past the valid ticks in its batch")
        return batch[within]


def _interpret(column: np.ndarray, kind: str) -> np.ndarray:
    """Raw int64 bits -> numbers, by the handshake's declared type."""
    if kind in _INT_TYPES:
        return column.astype(np.float64)
    # Everything else is a double stored as its IEEE-754 bit pattern.
    return column.astype("<i8").view(np.float64).astype(np.float64)


def gather(reader: LogReader, names, stride: int = 1, start: float = 0.0,
           end: float | None = None, _unused: int = 0):
    """Decode ``names`` over ``[start, end)`` at ``stride`` ticks.

    Returns ``(ticks, time, data, var_indices, var_types)`` with ``data`` shaped
    ``(n_ticks, len(names))`` in the order ``names`` was given -- the contract
    `logsource.read_window` consumes.

    **``start``/``end`` and ``time`` are in DIFFERENT units, deliberately.** The window
    is sliced by tick index: ``start`` and ``end`` are divided by the handshake's
    declared ``dt``, so ``start=110.0`` means tick 110000 regardless of what the clock
    says. ``time``, however, is built from the logged timestamps, because a dropped or
    repeated controller tick has to be visible and a tick-derived axis cannot show one.

    On a log whose real cadence matches its declared ``dt`` the two coincide and nobody
    notices. No Alex log checked so far does: one 2026-09 log runs 8% slow against its
    declaration and a 2026-07 one 14% fast, so tick 110000 there carries a timestamp
    near 94.5 s, not 110 s. Neither number is wrong; they answer different questions.
    `LogWindow` exposes both (`time` and `nominal_time`) plus `measured_dt` so the
    discrepancy can be seen rather than discovered.
    """
    names = list(names)
    missing = [n for n in names if n not in reader.hs.index]
    if missing:
        raise KeyError(f"log has no variables named {missing}")
    ambiguous = [n for n in names if n in reader.hs.duplicated]
    if ambiguous:
        raise KeyError(
            f"variable names are ambiguous across registries: {ambiguous}; "
            "this reader addresses variables by simple name only"
        )

    dt = reader.hs.dt
    stride = max(1, int(stride))
    first = max(0, int(round(start / dt)))
    last = reader.n_ticks if end is None else min(reader.n_ticks, int(round(end / dt)))
    if first >= last:
        raise ValueError(f"empty tick window [{first}, {last})")

    columns = np.array([reader.hs.index[n] + 1 for n in names], dtype=np.int64)  # +1: [0] is the timestamp
    types = [reader.hs.types[reader.hs.index[n]] for n in names]

    ticks = np.arange(first, last, stride, dtype=np.int64)
    raw = np.empty((ticks.size, columns.size), dtype=">i8")
    timestamps = np.empty(ticks.size, dtype=np.int64)
    for row, tick_index in enumerate(ticks):
        record = reader.tick(int(tick_index))
        timestamps[row] = record[0]
        raw[row] = record[columns]

    data = np.empty((ticks.size, columns.size), dtype=np.float64)
    for i, kind in enumerate(types):
        data[:, i] = _interpret(raw[:, i], kind)

    # Time from the logged timestamps, not from tick * dt: a dropped or repeated controller
    # tick would otherwise go unnoticed and shift everything after it.
    time = (timestamps - timestamps[0]).astype(np.float64) * 1.0e-9
    return ticks, time, data, columns - 1, types
