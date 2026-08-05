"""Gate A: N=8 corner contact sites, per-corner FK, and shape/order plumbing.

The premise of the whole N=8 experiment is that four sites per foot resolve to
FOUR DISTINCT places on the sole. If they collapse to one point, the filter still
compiles, still runs, and is silently just N=2 with a redundant state -- so the
distinctness assertion here is the load-bearing one (plan Gate A STOP).

Every claim is checked against an INDEPENDENT computation: corner offsets are
re-derived from the SCS2 box half-extents rather than read back from the function
under test, and FK positions are compared against the sites' own model-frame
placement, not against a second call to the same code path.

The N=2 assertions in this file are the regression guard required by invariant 3:
`contacts_per_foot=1` must reproduce the shipped sole pair EXACTLY.
"""

import numpy as np
import pytest

from invariant_estimation.pipeline import main_estimator as me


# ---------------------------------------------------------------------------
# offsets: derived from the box, not pasted
# ---------------------------------------------------------------------------

def test_corner_offsets_are_the_bottom_face_of_the_scs2_box():
    """Re-derive the 4 corners from the half-extents; must match element-wise.

    Independent recomputation -- catches a sign flip or a z taken at the box
    CENTER instead of its BOTTOM (which would float the contacts 2.75 cm up).
    """
    corners = np.asarray(me.alex_foot_corner_offsets())
    cx, cy, cz = me.ALEX_FOOT_BOX_CENTER
    hx, hy, hz = me.ALEX_FOOT_BOX_HALF

    assert corners.shape == (4, 3)
    expected = np.array([[cx + sx * hx, cy + sy * hy, cz - hz]
                         for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
    np.testing.assert_allclose(corners, expected, rtol=0, atol=0)

    # The plan's arithmetic, stated independently: x spans heel..toe, y spans the
    # width, and every corner sits on the ONE bottom plane.
    assert sorted(set(np.round(corners[:, 0], 6))) == [-0.085, 0.175]
    assert sorted(set(np.round(corners[:, 1], 6))) == [-0.07, 0.07]
    assert sorted(set(np.round(corners[:, 2], 6))) == [-0.0775]


def test_the_sim_foot_geom_is_built_from_the_same_constant():
    """Invariant 6: sim collision box and estimator corner FK share one source.

    Parses the geom string `run_policy` actually emits and checks it against the
    constant -- so editing the box in one place can never leave the other behind.
    """
    import run_policy as rp

    for body, typ, size, pos, _quat in rp.SCS2_COLLISION_GEOMS:
        if body not in ("LEFT_FOOT", "RIGHT_FOOT"):
            continue
        assert typ == "box"
        np.testing.assert_allclose([float(v) for v in size.split()],
                                   me.ALEX_FOOT_BOX_HALF, rtol=0, atol=0)
        np.testing.assert_allclose([float(v) for v in pos.split()],
                                   me.ALEX_FOOT_BOX_CENTER, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# site naming and ordering
# ---------------------------------------------------------------------------

def test_n2_site_set_is_untouched():
    """Regression (invariant 3): the shipped sole pair must not move."""
    assert me.alex_foot_sites(1) == ("left_sole", "right_sole")
    assert me.alex_extra_sites(1) == me.ALEX_EXTRA_SITES
    assert me.alex_site_names(1) == me.alex_site_names()


def test_n8_sites_are_foot_major_and_N_is_derived():
    """Order must be L,L,L,L,R,R,R,R -- `build_subchain_indices` assumes it."""
    sites = me.alex_foot_sites(4)
    assert len(sites) == 8, "N must be 2 * contacts_per_foot, never hardcoded"
    assert all(s.startswith("left") for s in sites[:4])
    assert all(s.startswith("right") for s in sites[4:])

    extra = me.alex_extra_sites(4)
    for i, off in enumerate(me.alex_foot_corner_offsets()):
        assert extra[f"left_c{i}"] == ("LEFT_FOOT", off)
        assert extra[f"right_c{i}"] == ("RIGHT_FOOT", off)


@pytest.mark.parametrize("bad", [0, 2, 3, 5])
def test_unsupported_contact_counts_are_rejected_loudly(bad):
    """A typo'd count must raise at BUILD time, not produce a silent wrong N."""
    with pytest.raises(ValueError):
        me.alex_foot_sites(bad)


# ---------------------------------------------------------------------------
# THE oracle: FK actually resolves per corner
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fused8():
    import run_policy as rp
    return me.build_alex_fused_estimator_from_urdf(rp.URDF, contacts_per_foot=4)


def test_kinematics_emits_eight_distinct_contact_points(fused8):
    """Gate A STOP: the four corners of a foot must be FAR APART, not merely !=.

    Pairwise separation is asserted against the box dimensions (>0.05 m, well
    under the 0.14/0.26 m true spacings but far above any FK noise). If this
    fails the sites collapsed and N=8 is N=2 wearing a hat -- halt, per the plan.
    """
    n_j = fused8.n_joints
    q = np.zeros(n_j)
    y = np.asarray(fused8.kinematics(q, np.zeros(n_j)).y)

    assert y.shape == (8, 3), f"expected (8,3) contact FK, got {y.shape}"

    for foot, sl in (("left", slice(0, 4)), ("right", slice(4, 8))):
        pts = y[sl]
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        off = d[~np.eye(4, dtype=bool)]
        assert off.min() > 0.05, (
            f"{foot} corners collapsed (min pairwise {off.min():.4f} m); "
            f"offsets used = {me.alex_foot_corner_offsets()}")

    # The two feet are distinct bodies: no left corner may coincide with a right one.
    cross = np.linalg.norm(y[:4, None, :] - y[None, 4:, :], axis=-1)
    assert cross.min() > 0.05


def test_corner_spacing_reproduces_the_box_at_the_home_pose(fused8):
    """FK separations must equal the BOX edge lengths -- an independent check.

    At q=0 the foot frame is axis-aligned with the world, so the corner spacings
    are exactly 2*hx and 2*hy. This catches offsets applied in the wrong frame
    (e.g. rotated into the sole frame twice), which distinctness alone would miss.
    """
    n_j = fused8.n_joints
    y = np.asarray(fused8.kinematics(np.zeros(n_j), np.zeros(n_j)).y)
    hx, hy, _ = me.ALEX_FOOT_BOX_HALF

    for sl in (slice(0, 4), slice(4, 8)):
        pts = y[sl]
        # order is (x-,y-), (x-,y+), (x+,y-), (x+,y+)
        np.testing.assert_allclose(np.linalg.norm(pts[1] - pts[0]), 2 * hy, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(pts[2] - pts[0]), 2 * hx, atol=1e-9)
        # all four on one horizontal plane
        np.testing.assert_allclose(pts[:, 2], pts[0, 2], atol=1e-9)


def test_n8_estimator_reports_eight_contacts(fused8):
    """N reaches the estimator as 8 -- derived from the site dict, not a literal."""
    assert fused8.n_contacts == 8


def test_subchain_is_foot_major_and_fk_is_the_only_discriminator(fused8):
    """Foot-major L,L,L,L,R,R,R,R, and the four corners of a foot SHARE a subchain.

    This is not a bug -- it is the documented premise of Gate D's research risk:
    corners of one foot see identical q/qd/tau (same leg joints) and identical IMU,
    so the ONLY thing distinguishing their feature vectors is the FK block (p, v).
    Pinning it here means that if the trained Sigma_C comes out identical across a
    foot's corners, we can point at this test and call it a finding rather than
    hunting for a plumbing bug.
    """
    from invariant_estimation.contactnet import features as F
    from invariant_estimation.sim.sensors import _dof_joint_names

    # the same unfiltered set `dataset.py` resolves, so this exercises the real path
    unf = _dof_joint_names(fused8.model.mj_model,
                           np.asarray(fused8.build.dof_anchor_unfiltered, dtype=int))
    idx = F.subchain_for(fused8, unf)

    assert idx.shape[0] == 8, f"expected 8 subchain rows, got {idx.shape[0]}"
    # foot-major: rows 0-3 are one foot, 4-7 the other
    assert np.array_equal(idx[0], idx[1]) and np.array_equal(idx[0], idx[3]), \
        "corners of the LEFT foot must share a joint subchain"
    assert np.array_equal(idx[4], idx[7]), \
        "corners of the RIGHT foot must share a joint subchain"
    assert not np.array_equal(idx[0], idx[4]), \
        "left and right feet must NOT share a subchain -- ordering is not foot-major"


def test_tangent_is_rotation_first_at_n8():
    """Invariant 1 at N=8: `[phi; rho_v; rho_p; rho_d_i]`, SE_10(3), tangent 33.

    This is the one that "compiles and runs while wrong" (plan invariant 1), so it
    is checked BEHAVIOURALLY, not by reading the index constants back: a unit
    tangent in each slot is pushed through `exp` and we assert WHICH part of the
    group element it moved. A translation-first exp passes the constants but fails
    here.
    """
    import jax.numpy as jnp
    from invariant_estimation.inEKF import state as s
    from invariant_estimation.inEKF import group

    N = 8
    st = s.InEKFState.identity(N)
    assert st.N == N
    assert st.dim == 3 * N + 9 == 33
    assert st.group_size == N + 5 == 13

    assert s.ROTATION_TANGENT_INDEX == 0
    assert s.BASE_VELOCITY_TANGENT_INDEX == 3
    assert s.BASE_POSITION_TANGENT_INDEX == 6
    for i in range(N):
        assert st.contact_tangent_index(i) == 9 + 3 * i

    eye = jnp.eye(N + 5)

    # slot 0..2 must drive the ROTATION block and nothing else
    xi = jnp.zeros(33).at[0].set(0.3)
    X = group.exp_SEn3(xi, N)
    assert not jnp.allclose(X[:3, :3], eye[:3, :3], atol=1e-9), \
        "tangent slot 0 did not rotate -- exp is translation-first"
    assert jnp.allclose(X[:3, 3:], 0.0, atol=1e-9), \
        "a pure rotation tangent moved a translation column"

    # slot 6..8 must drive base POSITION (column 4), leaving R at identity
    xi = jnp.zeros(33).at[6].set(0.7)
    X = group.exp_SEn3(xi, N)
    np.testing.assert_allclose(np.asarray(X[:3, :3]), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.asarray(X[:3, 4]), [0.7, 0.0, 0.0], atol=1e-12)

    # each contact slot must drive ITS OWN column (5 + i) and no other
    for i in range(N):
        xi = jnp.zeros(33).at[st.contact_tangent_index(i)].set(0.5)
        X = np.asarray(group.exp_SEn3(xi, N))
        np.testing.assert_allclose(X[:3, 5 + i], [0.5, 0.0, 0.0], atol=1e-12)
        others = [c for c in range(3, N + 5) if c != 5 + i]
        np.testing.assert_allclose(X[:3, others], 0.0, atol=1e-12)
