"""Fixtures for the Java-parity harness.

Every test here needs a real hardware log, which is a multi-gigabyte artifact
that cannot live in the repo.  The whole module therefore **skips** rather than
fails when the log or the `ihmclog` decoder is absent: on a machine without the
log store this is a missing fixture, not a regression.

Point the suite at a different run with::

    ALEX_PARITY_LOG=/opt/ihmc/LogData/incoming/<dir> uv run pytest tests/replay
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# The 2026-07-17 Alex001 walking run (16:01-16:12, 630 s). Chosen because both
# filters were live and fully instrumented: jointKFNumberOfIMUs=8,
# NumberOfFilteredJoints=9, StateDimension=42 = 2*9 + 3*8.
DEFAULT_LOG = "/opt/ihmc/LogData/incoming/20260717_160126_Alex001UnifiedControlProcess"

# The joint KF's state order, read off the log's own variable order
# (jointKF_q_<joint>). Not alphabetical and not the URDF order -- it is the order
# the Java filter assigned, so it is the order every comparison must use.
FILTERED_JOINTS = (
    "SPINE_Z",
    "LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
    "RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y",
)

# A quiet-ish window well after startup transients, inside continuous walking.
WINDOW = (200.0, 210.0)


@pytest.fixture(scope="session")
def log_dir() -> Path:
    path = Path(os.environ.get("ALEX_PARITY_LOG", DEFAULT_LOG))
    if not (path / "robotData.bsz").exists():
        pytest.skip(f"no hardware log at {path} (set $ALEX_PARITY_LOG)")
    return path


@pytest.fixture(scope="session")
def ihmclog_module():
    from invariant_estimation.replay.logsource import LogToolUnavailable, ihmclog

    try:
        return ihmclog()
    except LogToolUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def model_spec(log_dir):
    """Alex, converted from the description that shipped inside this very log."""
    from invariant_estimation.config import load_config
    from invariant_estimation.model.urdf2mjcf import convert_log_model

    jk = load_config()["joint_kf"]
    return convert_log_model(
        log_dir,
        rotor_inertia=jk["rotor_inertia"],
        rotor_inertia_default=jk["rotor_inertia_default"],
    )
