"""Graph resolution — `jointKF/build.py`.

No Java analogue: the Java filter does this work in its constructor and the suite
only observes it indirectly (through `n`, `m`, and the anchor count). It gets its
own tests here because the port has a harder job — it must also fix `K_max` and
every mask shape at build time (invariants I2, I7), and a wrong chain resolution
would silently change the state dimension rather than raise.

The tree fixtures are hand-written serial chains, deliberately NOT MJX: the build
logic is pure graph work, and testing it against a tree whose answers can be read
off by eye is what makes these oracles independent.
"""
import logging

import numpy as np
import pytest

from invariant_estimation.jointKF.build import (
    KinematicTree,
    build_joint_kf,
    check_pair_graph,
    joints_between,
)


def serial_chain(n_joints: int, tau_max: float | None = None) -> KinematicTree:
    """Floating base + `n_joints` hinges in series, mirroring the Java fixture.

    Body 0 is the world, body 1 the floating base, and hinge `j` drives body
    `j + 2`. So "the IMU after joint `j`" sits on body `j + 2`, which is what
    makes the expected joint counts below readable by inspection.
    """
    return KinematicTree(
        joint_names=tuple(f"JOINT_{j}" for j in range(n_joints)),
        joint_body=np.arange(n_joints) + 2,
        body_parent=np.concatenate([[0, 0], np.arange(1, n_joints + 1)]),
        joint_dof=np.arange(n_joints) + 6,          # free joint takes DoF 0..5
        base_dofs=np.arange(6),
        site_body={f"imu_after_{j}": j + 2 for j in range(n_joints)}
        | {f"foot_after_{j}": j + 2 for j in range(n_joints)},
        tau_max=np.full(n_joints, np.nan if tau_max is None else tau_max),
    )


# ---------------------------------------------------------------------------
# Chain resolution — the state dimension depends on getting this exactly right
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "chain, parent, child, expected_n",
    [
        (10, 1, 9, 8),   # SHAPES[0]  n=8
        (6, 1, 5, 4),    # SHAPES[1]  n=4
        (4, 0, 3, 3),    # SHAPES[2]  n=3
    ],
)
def test_joints_between_matches_the_java_shape_table(chain, parent, child, expected_n):
    """`n` = joints strictly between the two IMU links = `child - parent`.

    NOTE — a genuine contradiction inside `TEST_SUITE_MAP.md`: its prose says
    `n = child - parent - 1`, but its own shape table says `(10, 1, 9) -> n=8`,
    i.e. `child - parent`. The shape table wins, because it is what the Java
    fixtures actually construct and therefore what the ported tests must
    reproduce; the prose formula would give 7 and change every state dimension in
    the suite. Recorded in PORT_NOTES.md.
    """
    tree = serial_chain(chain)
    js = joints_between(tree, tree.site_body[f"imu_after_{parent}"],
                        tree.site_body[f"imu_after_{child}"])
    assert len(js) == expected_n
    assert js == list(range(parent + 1, child + 1))


def test_two_pairs_sharing_an_imu_union_their_chains():
    """SHAPES[3]: pairs (a,b) and (b,c) share the middle IMU -> n=8, m=3.

    This is the shared-base-IMU star that invariant I6 is about, and the only
    shape whose `n` is a genuine union rather than a single path.
    """
    tree = serial_chain(10)
    build = build_joint_kf(
        tree,
        imu_sites=["imu_after_1", "imu_after_5", "imu_after_9"],
        pairs=[(0, 1), (1, 2)],
    )
    assert (build.n_joints, build.n_imus, build.n_pairs) == (8, 3, 2)
    assert build.dim == 2 * 8 + 3 * 3 == 25
    # The shared IMU has ONE bias 3-vector, not one per pair -- the whole point.
    assert build.pair_child_bias_col(0) == build.pair_parent_bias_col(1)


def test_pair_velocity_masks_are_disjoint_for_a_shared_imu_star():
    """Each pair sees only its own sub-chain's velocities; together they cover n."""
    tree = serial_chain(10)
    b = build_joint_kf(tree, ["imu_after_1", "imu_after_5", "imu_after_9"],
                       [(0, 1), (1, 2)])
    mask = np.asarray(b.pair_velocity_mask)
    assert mask.shape == (2, 8)
    assert np.all(mask.sum(axis=0) == 1.0), "no joint may be claimed by two pairs"
    assert b.pair_velocity_cols(0) == tuple(range(8, 12))
    assert b.pair_velocity_cols(1) == tuple(range(12, 16))


# ---------------------------------------------------------------------------
# Structural rejection — each of these makes S singular, not merely inaccurate
# ---------------------------------------------------------------------------

def test_self_pair_is_rejected():
    with pytest.raises(ValueError, match="self-pair"):
        check_pair_graph([(0, 0)], 2, [2, 6])


def test_same_link_pair_is_rejected():
    """Both IMUs on one body: no joint between them, so three rank-0 rows."""
    with pytest.raises(ValueError, match="same|body"):
        check_pair_graph([(0, 1)], 2, [4, 4])


def test_cycle_in_the_pair_graph_is_rejected():
    """A cycle breaks the tree assumption the L*Sigma*L^T bookkeeping rests on."""
    with pytest.raises(ValueError, match="cycle"):
        check_pair_graph([(0, 1), (1, 2), (2, 0)], 3, [2, 4, 6])


def test_star_topology_is_accepted():
    """Alex's actual topology -- every pair against the base IMU -- is a tree."""
    check_pair_graph([(0, 1), (0, 2), (0, 3)], 4, [2, 4, 6, 8])


# ---------------------------------------------------------------------------
# Anchors: the F/U split (the Alex ankle case)
# ---------------------------------------------------------------------------

def test_anchor_chain_splits_filtered_from_unfiltered_joints():
    """`singlePairFootBeyondIMUs`: IMUs after joints 1 and 5, foot after joint 9.

    Joints 2..5 are filter states; joints 6..9 lie on the base->foot chain but
    are NOT states (the Alex ankles, no foot IMUs). Their measured velocity
    enters the anchor row as a known input, so the U split must capture them --
    that is what `R_anchor = Sigma_eps + J_U diag(sigma_qd^2) J_U^T` is built on.
    """
    tree = serial_chain(10)
    b = build_joint_kf(tree, ["imu_after_1", "imu_after_5"], [(0, 1)],
                       foot_sites=["foot_after_9"])
    assert b.n_joints == 4                       # joints 2..5
    assert b.n_anchors == 1
    f = np.asarray(b.anchor_filtered_mask)
    u = np.asarray(b.anchor_unfiltered_mask)
    assert f.shape == (1, 4) and u.shape == (1, 4)
    assert np.all(f == 1.0), "all four filtered joints lie on the base->foot chain"
    assert u.sum() == 4, "joints 6..9 are on the chain but unfiltered"


def test_anchor_count_is_fixed_at_build_not_by_contact():
    """K_max is a build constant (I2): a foot landing changes a MASK, not a shape."""
    tree = serial_chain(10)
    b = build_joint_kf(tree, ["imu_after_1", "imu_after_9"], [(0, 1)],
                       foot_sites=["foot_after_5", "foot_after_9"])
    assert b.n_anchors == 2
    assert b.n_stacked_rows == 3 * (1 + 2)
    assert b.anchor_row0 == 3                    # anchors follow all pair rows


# ---------------------------------------------------------------------------
# Parameter resolution + loud fallbacks
# ---------------------------------------------------------------------------

def test_sigma_tau_falls_back_when_the_effort_limit_is_absent():
    """`sigma_tau_i = alpha_i * tau_max_i`, else the 5.0 N.m fallback (I9)."""
    tree = serial_chain(6)                                  # tau_max all NaN
    b = build_joint_kf(tree, ["imu_after_1", "imu_after_5"], [(0, 1)])
    assert np.allclose(np.asarray(b.sigma_tau), 5.0)

    tree2 = serial_chain(6, tau_max=200.0)
    b2 = build_joint_kf(tree2, ["imu_after_1", "imu_after_5"], [(0, 1)])
    # These synthetic joint names match no alpha override -> the 0.15 default.
    assert np.allclose(np.asarray(b2.sigma_tau), 0.15 * 200.0)
    assert not np.allclose(np.asarray(b2.sigma_tau), np.asarray(b.sigma_tau))


def test_unwired_encoders_are_reported_loudly(caplog):
    """A silent fallback here means a joint under-trusts its encoder by ~1e4."""
    tree = serial_chain(6)
    with caplog.at_level(logging.WARNING):
        b = build_joint_kf(tree, ["imu_after_1", "imu_after_5"], [(0, 1)])
    assert not any(b.encoder_wired)
    assert np.allclose(np.asarray(b.encoder_var), 5.0e-5)
    assert "UNDER-trust" in caplog.text
    assert "JOINT_2" in caplog.text


def test_zero_gyro_sigma_is_floored_at_build(caplog):
    """Alex historically ran with unset SensorNoiseParameters => Sigma exactly 0.

    A zero Sigma removes the innovation-covariance floor on the pure-bias rows,
    collapsing lambda_min(S) and diverging P through the Joseph K R K^T loop.
    Flooring happens once at build, never per tick (I7).
    """
    tree = serial_chain(6)
    with caplog.at_level(logging.WARNING):
        b = build_joint_kf(tree, ["imu_after_1", "imu_after_5"], [(0, 1)],
                           gyro_sigma=lambda name: np.zeros((3, 3)))
    assert np.allclose(np.asarray(b.gyro_sigma), 1.0e-6 * np.eye(3))
    assert "floor" in caplog.text.lower()

    # A wired Sigma above the floor trace must pass through untouched.
    wired = np.diag([4.0e-4, 1.0e-6, 2.5e-5])
    b2 = build_joint_kf(tree, ["imu_after_1", "imu_after_5"], [(0, 1)],
                        gyro_sigma=lambda name: wired)
    assert np.allclose(np.asarray(b2.gyro_sigma)[0], wired)


def test_nuisance_dofs_are_the_base_six_plus_gap_joints():
    """The Schur nuisance set. Getting this wrong changes Lambda, hence Qa.

    A **gap** joint lies on a `root -> filtered` path without being a filter
    state, exactly Java's `collectSpanningJoints` minus the filtered set. In this
    fixture the pair brackets joints 2..5, so joints **0 and 1** are the gap: the
    base cannot reach a filtered joint without passing through them, and they
    genuinely accelerate, so they must be marginalised.

    Joints 6..9 are a different animal. They are unfiltered members of the
    base->foot *anchor* chain, but they hang BELOW every filtered joint and are
    therefore off the root->filtered paths. Java locks them into the composited
    ignored-subtree inertia; marginalising them models them as free to
    accelerate and shrinks `Lambda`. This test previously asserted exactly that
    wrong set -- on Alex (where joints 6..9 are the ankles) it cost 1.7% on
    `diag(Qa)` against the hardware log. The two sets coincide on a chain with no
    off-path joints, which is why nothing else here noticed.
    """
    tree = serial_chain(10)
    b = build_joint_kf(tree, ["imu_after_1", "imu_after_5"], [(0, 1)],
                       foot_sites=["foot_after_9"])
    nuisance = np.asarray(b.dof_nuisance)
    assert list(nuisance[:6]) == list(range(6)), "floating base's 6 DoF come first"
    # joints 0..1 are the gap joints -> DoF 6..7 (hinge j has DoF j+6)
    assert sorted(nuisance[6:]) == [6, 7]
    assert list(np.asarray(b.dof_joint)) == [8, 9, 10, 11]   # joints 2..5
    # ...and the anchor chain's unfiltered joints (6..9) are published separately,
    # in anchor_unfiltered_mask column order, NOT folded into the nuisance set.
    assert list(np.asarray(b.dof_anchor_unfiltered)) == [12, 13, 14, 15]


def test_empty_chain_is_rejected():
    """A pair whose IMUs are adjacent leaves no joints -> no state to filter."""
    tree = serial_chain(6)
    tree.site_body["same_a"] = 3
    with pytest.raises(ValueError):
        build_joint_kf(tree, ["same_a", "same_a"], [(0, 1)])
