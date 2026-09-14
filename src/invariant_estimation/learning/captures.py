"""Versioned multi-session manifest, separate from Java calibration.CaptureSet.

One session is indivisible. All windows/augmentations derived from it inherit
its partition. Paths are relative to the manifest directory; hashes identify
content even when files are renamed. Calibration reuse is allowed, raw stream
reuse across partitions is not. This stores a synchronization contract, not a
claim that clocks or frame registration were actually calibrated correctly.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str

    def __post_init__(self):
        _nonempty(self.path, "artifact path")
        p = PurePosixPath(self.path)
        if p.is_absolute() or ".." in p.parts or "\\" in self.path or str(p) == ".":
            raise ValueError("artifact path must be a safe manifest-relative POSIX path")
        if not isinstance(self.sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hex digits")

    def verify(self, root):
        root = Path(root).resolve()
        path = (root / self.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact resolves outside manifest root")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.sha256:
            raise ValueError(f"artifact hash mismatch: {self.path}")


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    capture_group: str  # acquisition/run identity shared by all derived exports
    motion: str
    captured_at: str  # timezone-aware ISO 8601
    robot_log: Artifact
    mocap_log: Artifact
    calibration: Artifact
    model: Artifact  # URDF/model hash, same rationale as CalibrationResult.Provenance
    synchronization: Artifact  # clock mapping AND spatial registration metadata
    clock_domain: str
    world_frame: str
    base_frame: str
    start_ns: int  # interval in the documented synchronized clock domain
    end_ns: int
    joint_names: tuple[str, ...]
    imu_names: tuple[str, ...]

    def __post_init__(self):
        for name in ("session_id", "capture_group", "motion", "clock_domain", "world_frame", "base_frame"):
            _nonempty(getattr(self, name), name)
        when = datetime.fromisoformat(self.captured_at)
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("captured_at must include a timezone")
        if type(self.start_ns) is not int or type(self.end_ns) is not int or self.end_ns <= self.start_ns:
            raise ValueError("session interval must be integer nanoseconds with end > start")
        for name in ("joint_names", "imu_names"):
            values = tuple(getattr(self, name))
            if not values or any(not isinstance(v, str) or not v for v in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
            object.__setattr__(self, name, values)
        for name in ("robot_log", "mocap_log", "calibration", "model", "synchronization"):
            if not isinstance(getattr(self, name), Artifact):
                raise ValueError(f"{name} must be an Artifact")


@dataclass(frozen=True)
class SessionSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self):
        for name in ("train", "validation", "test"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.train or not self.test:
            raise ValueError("train and test must each contain at least one whole session")


@dataclass(frozen=True)
class CaptureManifest:
    sessions: tuple[CaptureSession, ...]
    split: SessionSplit
    created_at: str
    source_revision: str
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "sessions", tuple(self.sessions))
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported capture manifest schema")
        _nonempty(self.source_revision, "source_revision")
        if datetime.fromisoformat(self.created_at).utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        ids = [s.session_id for s in self.sessions]
        assigned = self.split.train + self.split.validation + self.split.test
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate session ID")
        if len(assigned) != len(set(assigned)) or set(assigned) != set(ids):
            raise ValueError("split must assign every session exactly once")
        partitions = {sid: part for part in ("train", "validation", "test")
                      for sid in getattr(self.split, part)}
        owners = {}
        for session in self.sessions:
            part = partitions[session.session_id]
            # Hashes have no role prefix: even accidentally swapping input roles
            # must not conceal raw-data leakage. Group protects derived exports.
            keys = ("group:" + session.capture_group,
                    "raw:" + session.robot_log.sha256, "raw:" + session.mocap_log.sha256)
            for key in keys:
                if key in owners and owners[key] != part:
                    raise ValueError(f"capture leakage across partitions: {session.session_id}")
                owners[key] = part

    def partition(self, name):
        if name not in ("train", "validation", "test"):
            raise ValueError("unknown partition")
        by_id = {s.session_id: s for s in self.sessions}
        return tuple(by_id[sid] for sid in getattr(self.split, name))

    def to_json(self):
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text):
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        data = json.loads(text, object_pairs_hook=unique_keys)
        sessions = []
        for entry in data["sessions"]:
            item = dict(entry)
            for key in ("robot_log", "mocap_log", "calibration", "model", "synchronization"):
                item[key] = Artifact(**item[key])
            sessions.append(CaptureSession(**item))
        return cls(**{**data, "sessions": tuple(sessions), "split": SessionSplit(**data["split"])})

    @classmethod
    def load(cls, path, *, verify_artifacts=True):
        path = Path(path)
        result = cls.from_json(path.read_text(encoding="utf-8"))
        if verify_artifacts:
            for session in result.sessions:
                for name in ("robot_log", "mocap_log", "calibration", "model", "synchronization"):
                    getattr(session, name).verify(path.parent)
        return result

    def save(self, path):
        """Exclusive create: never silently replace a frozen experiment split."""
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(self.to_json())


def deterministic_split(sessions, *, n_test, n_validation=0, seed=0):
    """Order-independent seeded WHOLE-SESSION split; never random window split.

    Related/derived exports must first be consolidated to one session. Reject
    them rather than silently assigning correlated captures to different sets.
    For a deliberate motion holdout, construct SessionSplit explicitly instead.
    """
    sessions = tuple(sessions)
    if type(n_test) is not int or type(n_validation) is not int or n_test < 1 or n_validation < 0:
        raise ValueError("invalid held-out session counts")
    if n_test + n_validation >= len(sessions):
        raise ValueError("at least one training session must remain")
    for key in (lambda s: s.session_id, lambda s: s.capture_group,
                lambda s: s.robot_log.sha256, lambda s: s.mocap_log.sha256):
        values = [key(s) for s in sessions]
        if len(values) != len(set(values)):
            raise ValueError("consolidate related captures before automatic splitting")
    ordered = sorted(sessions, key=lambda s: (
        hashlib.sha256(f"{seed}:{s.session_id}".encode()).hexdigest(), s.session_id))
    ids = tuple(s.session_id for s in ordered)
    return SessionSplit(ids[n_test+n_validation:], ids[n_test:n_test+n_validation], ids[:n_test])
