r"""Port of `JointLevelKFSingularInnovationDiagnosticTest.java` (2 tests) — gate G8.
Marked **"Adapt (message text)"** in `TEST_SUITE_MAP.md`.

Java's `describeSingularInnovation(label, reason, H, R)` returns a human-readable
string and the tests assert on substrings (`"near-singular"`, `"cond(S)"`,
`"gyro pair 0"`, the base IMU's sensor name, `"encoder q of joint <name>"`).
CLAUDE.md §5 / `CONTRACT_CARD.md` §7 say to **port the observable, not the
message**: the observable is the mapping from the near-null eigenvector of
`S = H P H^T + R` back to the physical measurement rows that carry it.  So
`diagnostics.describe_singular_innovation` returns structured attribution and the
assertions below are on that structure; `.summary()` still renders a log line,
and it is spot-checked but not pinned.

Why the attribution is two-part
-------------------------------
Note what the second Java scenario demands.  Both `H` rows put their weight on
**joint 0's** column, and the expected message names joint 0 — for *both* rows.
So a diagnostic that reports "row 1 => encoder 1" (the row ordinal) would produce
a plausible message naming the wrong joint.  Attribution therefore reads the row
block *and* the row's dominant state column, and for the diagonal channels the
column wins.  That is the whole content of the second test.
"""
import numpy as np
import pytest

from invariant_estimation.jointKF.diagnostics import describe_singular_innovation
from invariant_estimation.jointKF.state import default_params
from invariant_estimation.jointKF.update import joseph_update
from invariant_estimation.jointKF.state import JointKFState

import jax.numpy as jnp

from ._oracles import SHAPES, stub_build

PARAMS = default_params()


# ---------------------------------------------------------------------------
# The degenerate gyro pair
# ---------------------------------------------------------------------------

def test_diagnostic_names_the_degenerate_gyro_pair():
    """`testDiagnosticNamesTheDegenerateGyroPair` — `singlePair(1234, 10, 1, 9)`.

    `H` is 3 x dim with rows 0 and 1 **identical**: the pair's first two gyro axes
    would be measuring the same linear combination of the state, which is exactly
    the rank deficiency a mis-wired or duplicated IMU axis produces on hardware.
    With `R = 1e-6 I`, the direction `(1, -1, 0)/sqrt(2)` has innovation variance
    `2e-6` against `O(1)` elsewhere — near-singular, and the update gets gated.
    """
    build = stub_build(SHAPES[0])                 # n = 8, m = 2, one pair
    dim = build.dim

    c = np.arange(dim, dtype=float)
    H = np.zeros((3, dim))
    H[0] = np.sin(0.31 * (c + 1.0))
    H[1] = H[0]                                    # the induced rank deficiency
    H[2] = np.cos(0.17 * (c + 2.0))
    R = 1.0e-6 * np.eye(3)
    P = np.eye(dim)

    report = describe_singular_innovation(
        build, H, R, P,
        channel="stacked",
        label="stackedGyroUpdate",
        reason="test-induced rank deficiency",
    )

    # The matrix really is near-singular in the sense the gate cares about.
    assert report.min_eigenvalue < 1.0e-5
    assert report.condition_number > 1.0e6

    # The degenerate direction is the difference of rows 0 and 1, and nothing
    # else: two rows, equal weight, opposite sign.
    assert {r.row for r in report.rows} == {0, 1}
    assert all(r.weight == pytest.approx(0.5, abs=1e-9) for r in report.rows)
    assert report.null_vector[0] * report.null_vector[1] < 0.0
    assert abs(report.null_vector[2]) < 1e-9

    # ...and it is attributed to gyro pair 0, naming both IMUs — including the
    # base IMU, which is Java's `getBaseIMU().getSensorName()` assertion.
    assert all(r.channel == "gyro_pair" and r.ordinal == 0 for r in report.rows)
    base_name = build.imu_names[build.base_imu]
    assert all(base_name in r.name for r in report.rows)

    text = report.summary()
    assert "near-singular" in text and "cond(S)" in text and "gyro pair 0" in text
    assert base_name in text


def test_the_gate_actually_fires_on_that_measurement():
    """The diagnostic must describe a situation the filter really refuses.

    Port-specific, and the reason the two halves belong in one file: a diagnostic
    that names a row nobody rejected is decoration.  Feeding the same `(H, R, P)`
    through `joseph_update` must trip the `cond(S)` gate and leave `(x, P)`
    bit-identical.
    """
    build = stub_build(SHAPES[0])
    dim = build.dim
    c = np.arange(dim, dtype=float)
    H = np.zeros((3, dim))
    H[0] = np.sin(0.31 * (c + 1.0))
    H[1] = H[0]
    H[2] = np.cos(0.17 * (c + 2.0))
    R = 1.0e-6 * np.eye(3)

    state = JointKFState(x=jnp.zeros(dim), P=jnp.eye(dim))
    post, info = joseph_update(state, jnp.asarray(H), jnp.zeros(3), jnp.asarray(R), PARAMS)

    assert float(info.was_applied) == 0.0, "the duplicated row must gate the update out"
    assert float(info.condition_proxy) > PARAMS.cond_s_max
    assert np.array_equal(np.asarray(post.x), np.asarray(state.x))
    assert np.array_equal(np.asarray(post.P), np.asarray(state.P))
    assert np.isnan(float(info.nis)), "a gated update publishes no NIS"


# ---------------------------------------------------------------------------
# The degenerate encoder joint
# ---------------------------------------------------------------------------

def test_diagnostic_names_the_degenerate_encoder_joint():
    """`testDiagnosticNamesTheDegenerateEncoderJoint` — `singlePair(4321, 6, 1, 5)`, n = 4.

    Both rows observe **joint 0's** position column.  Java's expected message is
    `"encoder q of joint " + filteredJoints[0].getName()`, i.e. joint 0 for both
    rows — so the attribution must come from the column, not the row ordinal.
    """
    build = stub_build(SHAPES[1])                 # NUM_CHAIN_JOINTS = 6 => n = 4
    dim = build.dim

    H = np.zeros((2, dim))
    H[0, 0] = 1.0
    H[1, 0] = 1.0                                  # row 1 also observes joint 0
    R = np.diag([1.0e-6, 1.0e-6])
    P = np.eye(dim)

    report = describe_singular_innovation(
        build, H, R, P,
        channel="encoder",
        label="encoder",
        reason="test-induced rank deficiency",
    )

    assert report.min_eigenvalue < 1.0e-5
    assert {r.row for r in report.rows} == {0, 1}

    joint0 = build.joint_names[0]
    for r in report.rows:
        assert r.channel == "encoder"
        assert r.ordinal == 0, "attribution must follow the column, not the row index"
        assert r.name == f"encoder q of {joint0}"
        assert r.state_index == 0
        assert r.state_name == f"q of {joint0}"

    text = report.summary()
    assert "near-singular" in text and f"encoder q of {joint0}" in text


def test_attribution_ignores_the_healthy_rows():
    """A third, well-conditioned row must not be implicated.

    Port-specific. The Java scenarios have only the degenerate rows, so a
    diagnostic that simply listed every row would pass both of them while being
    useless for its actual job — finding one bad sensor among forty.
    """
    build = stub_build(SHAPES[1])
    dim = build.dim

    H = np.zeros((3, dim))
    H[0, 0] = 1.0
    H[1, 0] = 1.0
    H[2, build.n_joints + 1] = 1.0                 # a clean, independent row
    R = 1.0e-6 * np.eye(3)
    P = np.eye(dim)

    report = describe_singular_innovation(build, H, R, P, channel="encoder", label="encoder")
    assert {r.row for r in report.rows} == {0, 1}, "the healthy row must not be named"
    assert sum(r.weight for r in report.rows) == pytest.approx(1.0, abs=1e-9)
