from dataclasses import replace
import hashlib
import json
import pytest

from invariant_estimation.learning.captures import (
    Artifact, CaptureSession, CaptureManifest, SessionSplit, deterministic_split,
)


def artifact(name):
    return Artifact(name, hashlib.sha256(name.encode()).hexdigest())


def session(name):
    return CaptureSession(name, name, "walking", "2026-09-15T10:00:00-05:00",
                          artifact(name+"/robot.bin"), artifact(name+"/mocap.bin"),
                          artifact("calibration.json"), artifact("model.urdf"),
                          artifact(name+"/sync.json"), "aligned_robot_monotonic_ns",
                          "registered_zup", "pelvis", 100, 1000, ("hip",), ("base", "shin"))


def manifest(sessions=None, split=None):
    sessions = tuple(session(str(i)) for i in range(5)) if sessions is None else sessions
    split = deterministic_split(sessions, n_test=1, n_validation=1, seed=23) if split is None else split
    return CaptureManifest(sessions, split, "2026-09-14T22:00:00Z", "git:synthetic-fixture")


def test_roundtrip_and_order_independent_session_split(tmp_path):
    m = manifest()
    assert CaptureManifest.from_json(m.to_json()) == m
    assert deterministic_split(tuple(reversed(m.sessions)), n_test=1, n_validation=1, seed=23) == m.split
    assert len(m.partition("train")) == 3
    assert len(m.partition("test")) == 1
    path = tmp_path / "manifest.json"
    m.save(path)
    assert CaptureManifest.load(path, verify_artifacts=False) == m
    with pytest.raises(FileExistsError):
        m.save(path)


@pytest.mark.parametrize("kind", ["same_id", "same_group", "same_robot", "same_mocap", "cross_role"])
def test_leakage_rejected_even_after_rename(kind):
    a, b = session("a"), session("b")
    changes = {
        "same_id": {"session_id": a.session_id}, "same_group": {"capture_group": a.capture_group},
        "same_robot": {"robot_log": replace(a.robot_log, path="renamed/robot.bin")},
        "same_mocap": {"mocap_log": a.mocap_log}, "cross_role": {"robot_log": a.mocap_log},
    }
    b = replace(b, **changes[kind])
    with pytest.raises(ValueError):
        manifest((a, b), SessionSplit(("a",), (), ("b",)))


def test_shared_calibration_is_allowed_but_split_omissions_are_not():
    a, b, c = session("a"), session("b"), session("c")
    manifest((a, b), SessionSplit(("a",), (), ("b",)))
    with pytest.raises(ValueError, match="exactly once"):
        manifest((a, b, c), SessionSplit(("a",), (), ("b",)))
    with pytest.raises(ValueError, match="exactly once"):
        manifest((a, b), SessionSplit(("a",), ("a",), ("b",)))


def test_automatic_split_rejects_fragments():
    a, b, c = session("a"), session("b"), session("c")
    with pytest.raises(ValueError, match="consolidate"):
        deterministic_split((a, replace(b, capture_group="a"), c), n_test=1)


def test_hash_verification_and_safe_paths(tmp_path):
    # All artifacts in this fixture contain their own relative path as bytes.
    m = manifest()
    for s in m.sessions:
        for name in ("robot_log", "mocap_log", "calibration", "model", "synchronization"):
            a = getattr(s, name)
            path = tmp_path / a.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(a.path.encode())
    path = tmp_path / "manifest.json"
    m.save(path)
    assert CaptureManifest.load(path) == m
    (tmp_path / m.sessions[0].robot_log.path).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        CaptureManifest.load(path)
    for invalid in ("/absolute", "../escape", "a/../../escape", "a\\b", "."):
        with pytest.raises(ValueError):
            Artifact(invalid, "0"*64)


def test_strict_schema_time_and_duplicate_keys():
    m = manifest()
    with pytest.raises(ValueError, match="unsupported"):
        CaptureManifest.from_json(m.to_json().replace('"schema_version": 1', '"schema_version": 2'))
    with pytest.raises(ValueError, match="duplicate JSON"):
        CaptureManifest.from_json('{"sessions": [], "sessions": []}')
    with pytest.raises(ValueError, match="timezone"):
        replace(m.sessions[0], captured_at="2026-09-15T10:00:00")
    with pytest.raises(ValueError, match="integer nanoseconds"):
        replace(m.sessions[0], end_ns=m.sessions[0].start_ns)
    with pytest.raises(ValueError):
        deterministic_split(m.sessions, n_test=5)
