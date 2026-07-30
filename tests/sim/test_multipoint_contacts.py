r"""Toe/heel contact points — the `N > K` path.

CoCo-InEKF anchors **two** contact points per foot, and so does this port when
built with `contact_sites=`. The reason is observability, not fidelity: a single
contact point at the sole centre carries **no information about foot
orientation**, so nothing in the filter observes rotation about the vertical
except through the base. Two points 0.197 m apart do.

Measured (`PORT_NOTES.md`, "Toe/heel contact points"): on the analytic filter
alone, with no ContactNet at all, this takes the closed-loop yaw error from 1.168°
to 0.897° in the tail and the vertical sink from −3.19 m to −0.57 m.

Four properties, in the order they would break:

1. **`K` and `N` are decoupled.** The joint KF keeps one stance anchor per foot;
   only the InEKF gets the extra points. Two anchors on one *rigid* foot are
   kinematically locked, so stacking them as independent rows in the joint KF's
   anchor block would double-count the bias information with nothing modelling the
   lock — the same class of error as the block-diagonal `R_g` trap (I6).
2. **`N = K` is bit-for-bit the old filter.** The default must not move.
3. **The geometry is the Java sole plate**, not the collision box.
4. **The load split is real**: a contact forward of the sole centre loads the toe
   slot and one behind it loads the heel slot, resolved from the estimator's own
   site names so the sim cannot disagree with the filter about which slot is which.
"""

from __future__ import annotations

import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect, must precede arrays)
from invariant_estimation.pipeline import main_estimator as me


# ---------------------------------------------------------------------------
# 3. Geometry
# ---------------------------------------------------------------------------

def test_toe_heel_sites_sit_on_the_java_sole_plate():
    r"""Heel at x = −0.052, toe at x = +0.145, both on the sole plane z = −0.072.

    ``ACTUAL_FOOT_LENGTH = 0.197`` with ``FOOT_BACK = 0.052``, so the plate spans
    ``[−0.052, +0.145]`` in the ``*_FOOT`` frame and `ALEX_SOLE_OFFSET` is its
    centre. The **collision box** is more generous (x from −0.085 to +0.175) and is
    deliberately *not* what these follow: anchoring at the box corners would put
    the contact points 3.3 cm beyond the physical plate.
    """
    assert me.ALEX_HEEL_X == pytest.approx(-0.052)
    assert me.ALEX_TOE_X == pytest.approx(0.197 - 0.052)
    # The sole centre is the midpoint of the two, by construction.
    assert (me.ALEX_HEEL_X + me.ALEX_TOE_X) / 2.0 == pytest.approx(me.ALEX_SOLE_OFFSET[0])
    # Separation is the plate length.
    assert me.ALEX_TOE_X - me.ALEX_HEEL_X == pytest.approx(0.197)

    for name, (body, off) in me.ALEX_CONTACT_OFFSETS.items():
        side = "LEFT" if name.startswith("left") else "RIGHT"
        assert body == f"{side}_FOOT"
        assert off[1] == 0.0, "toe/heel are on the foot's centreline"
        assert off[2] == pytest.approx(-me.ALEX_ANKLE_HEIGHT), (
            "toe/heel must be on the SAME sole plane as the sole site, or the two "
            "builds anchor at different heights and are not comparable")
        assert off[0] == pytest.approx(
            me.ALEX_TOE_X if "toe" in name else me.ALEX_HEEL_X)


def test_contact_sites_are_foot_major_and_named_for_the_split():
    """`sim.sensors` splits on the names, so the ordering contract is load-bearing."""
    assert me.ALEX_CONTACT_SITES == ("left_heel", "left_toe", "right_heel", "right_toe")
    # Foot-major: contact 2f+s belongs to foot f.
    for f, side in enumerate(("left", "right")):
        for s in range(2):
            assert me.ALEX_CONTACT_SITES[2 * f + s].startswith(side)
    # Every contact site is in the model's site table, or the FK cannot find it.
    assert set(me.ALEX_CONTACT_SITES) <= set(me.alex_site_names())
    assert set(me.ALEX_FOOT_SITES) <= set(me.alex_site_names())


def test_contact_sites_are_always_emitted_so_a_rollout_stays_readable():
    """The sites exist in every build; whether they are USED is `contact_sites`.

    One site table for both builds means a model built for `N = 2` and one built
    for `N = 4` agree on every *other* site's ordinal, so an `N = 2` rollout stays
    readable against an `N = 4` model. An unused site costs one FK row.
    """
    for nm in me.ALEX_CONTACT_SITES:
        assert nm in me.ALEX_EXTRA_SITES


# ---------------------------------------------------------------------------
# 1 + 2. K/N decoupling, on the fixture estimator
# ---------------------------------------------------------------------------

def _build(toe_heel: bool):
    """The real Alex estimator. Module-scoped via the caching in `_pair`."""
    import run_policy as rp
    return me.build_alex_fused_estimator_from_urdf(
        rp.cycloid_forearm_urdf(rp.URDF), dt=1.0e-3,
        contact_fk_unfiltered=True, toe_heel=toe_heel)


@pytest.fixture(scope="module")
def _pair():
    return _build(False), _build(True)


def test_toe_heel_gives_the_inekf_four_contacts_and_the_joint_kf_two(_pair):
    base, th = _pair
    assert (base.n_anchors, base.n_contacts) == (2, 2)
    assert (th.n_anchors, th.n_contacts) == (2, 4), (
        "the joint KF must keep ONE anchor per foot -- two anchors on a rigid foot "
        "are kinematically locked and stacking them double-counts the bias info")
    # The anchor sites are untouched, element for element.
    assert np.array_equal(base.foot_site_ords, th.foot_site_ords)
    # ...and the contact sites are the toe/heel ones.
    names = tuple(th.model.site_names)
    assert tuple(names[o] for o in th.contact_site_ords) == me.ALEX_CONTACT_SITES
    # The InEKF really is sized N, not K.
    assert th.ekf.N == 4
    assert th.ekf.params.H.shape == (12, 21)      # (3N, 3N+9)


def test_the_two_contact_points_are_a_real_lever_arm(_pair):
    r"""Heel→toe separation is 0.197 m in the contact FK, at the home pose.

    This is what makes foot orientation observable at all — and it has to survive
    the FK, not just the site table, because the sites are declared in the
    ``*_FOOT`` frame and consumed in the InEKF body frame.
    """
    _, th = _pair
    n = th.n_joints + th.n_aux
    y = np.asarray(th.kinematics(np.zeros(n), np.zeros(n)).y)
    assert y.shape == (4, 3)
    for f in range(2):
        heel, toe = y[2 * f], y[2 * f + 1]
        assert np.linalg.norm(toe - heel) == pytest.approx(0.197, abs=1e-9)
        assert toe[0] > heel[0], "the toe must be FORWARD of the heel"
        # Same foot => same lateral offset and same height.
        assert toe[1] == pytest.approx(heel[1])
        assert toe[2] == pytest.approx(heel[2])
    # Left and right are mirrored laterally.
    assert y[0][1] == pytest.approx(-y[2][1])


def test_N_equals_K_reproduces_the_two_contact_filter_exactly(_pair):
    r"""`contact_sites=None` must be the old build, not merely a similar one.

    Every recorded rollout and trained checkpoint is `N = 2`; a silent change here
    invalidates all of them. Compared as the FK the InEKF actually consumes plus
    the constant `H`, which is where an off-by-one in the site table would land.
    """
    base, _ = _pair
    n = base.n_joints + base.n_aux
    y = np.asarray(base.kinematics(np.zeros(n), np.zeros(n)).y)
    assert y.shape == (2, 3)
    names = tuple(base.model.site_names)
    assert tuple(names[o] for o in base.contact_site_ords) == me.ALEX_FOOT_SITES
    assert base.ekf.params.H.shape == (6, 15)

    # The sole sites are the MIDPOINT of their toe/heel pair -- so the N=2 filter
    # anchors exactly where the N=4 one straddles, and the two are comparable.
    _, th = _pair
    y4 = np.asarray(th.kinematics(np.zeros(n), np.zeros(n)).y)
    for f in range(2):
        assert np.allclose(y[f], 0.5 * (y4[2 * f] + y4[2 * f + 1]), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. The load split
# ---------------------------------------------------------------------------

def test_point_load_split_is_resolved_from_the_estimator_site_names():
    """`(foot, fore) -> slot` comes from the filter's own sites, not a hardcode."""
    import mujoco
    import run_policy as rp
    from invariant_estimation.sim.sensors import SimSensorReader

    pol = rp.load_policy("baseline")
    m = rp.build_sim_model(pol, with_visuals=False, with_imu_sensors=True)
    th = _build(True)
    r = SimSensorReader(m, th, foot_geoms=rp.FOOT_GEOMS, dt=1.0e-3)

    assert r.point_trust is not None, "N != K must engage the per-point trust"
    assert r.n_points == 4
    # left=(geom 0), right=(geom 1); False=aft/heel, True=fore/toe.
    assert r.point_slot == {(0, False): 0, (0, True): 1, (1, False): 2, (1, True): 3}
    assert r.sole_centre_x == pytest.approx(me.ALEX_SOLE_OFFSET[0])

    base = _build(False)
    r2 = SimSensorReader(m, base, foot_geoms=rp.FOOT_GEOMS, dt=1.0e-3)
    assert r2.point_trust is None, "N == K must keep the single per-foot path"
    assert r2.point_slot == {}


def test_point_loads_sum_to_the_foot_loads():
    r"""Conservation: no normal force is dropped or double-counted by the split.

    Both are clipped to [0, 1] independently, so the check is done below
    saturation — a fully-loaded foot clips to 1.0 while its two points sum to 1.0
    only by coincidence. Asserted on a settled stance, where the load is shared.
    """
    import mujoco
    import run_policy as rp
    from invariant_estimation.sim.sensors import SimSensorReader

    pol = rp.load_policy("baseline")
    m = rp.build_sim_model(pol, with_visuals=False, with_imu_sensors=True)
    maps = rp.make_maps(m, pol)
    th = _build(True)
    r = SimSensorReader(m, th, foot_geoms=rp.FOOT_GEOMS, dt=1.0e-3)

    d = mujoco.MjData(m)
    d.qpos[maps["ALL_QADR"]] = maps["ALL_HOME"]
    d.qpos[0:3] = [0.0, 0.0, rp.lowest_foot_to_root_height(m, d)]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)

    seen_unsaturated = False
    for k in range(400):
        d.ctrl[maps["ALL_AID"]] = maps["ALL_HOME"]
        mujoco.mj_step(m, d)
        fl, pl = r.foot_loads(d), r.point_loads(d)
        assert pl.shape == (4,)
        assert np.all(pl >= 0.0) and np.all(pl <= 1.0)
        for f in range(2):
            pair = pl[2 * f] + pl[2 * f + 1]
            if fl[f] < 0.99 and pair < 0.99:
                seen_unsaturated = True
                assert pair == pytest.approx(fl[f], abs=1e-9), (
                    f"foot {f}: points sum to {pair} but the foot reads {fl[f]} -- "
                    f"the split lost or duplicated normal force")
    assert seen_unsaturated, "never observed an unsaturated tick; test was vacuous"


# ---------------------------------------------------------------------------
# 5. The ContactNet feature table must agree with N
# ---------------------------------------------------------------------------

def test_subchain_is_one_row_per_contact_point_not_per_foot():
    r"""`make_contact_channels` reads ``N_c`` from the subchain's shape while
    ``p``/``v`` come from the contact FK, so the two must agree.

    `ALEX_FOOT_CHAINS` has one entry per *leg*, so the table is `(2, 6)` by
    default and must become `(4, 6)` for toe/heel — foot-major, with each leg's
    chain repeated, because heel and toe share that leg entirely.

    A silently-permuted or wrongly-sized subchain still produces a network that
    trains, which is why this is asserted rather than assumed.
    """
    from invariant_estimation.contactnet import features as F

    jn = [n for c in F.ALEX_FOOT_CHAINS for n in c[:4]]
    un = [n for c in F.ALEX_FOOT_CHAINS for n in c[4:]]

    one = F.build_subchain_indices(jn, un)
    assert one.shape == (2, 6)
    two = F.build_subchain_indices(jn, un, contacts_per_foot=2)
    assert two.shape == (4, 6)
    # Foot-major: [L, L, R, R]. Heel and toe of one foot are the SAME chain.
    assert np.array_equal(two[0], two[1]) and np.array_equal(two[2], two[3])
    assert np.array_equal(two[0], one[0]) and np.array_equal(two[2], one[1])
    # ...and the two feet are different chains, or the split is meaningless.
    assert not np.array_equal(two[0], two[2])

    with pytest.raises(ValueError, match="contacts_per_foot"):
        F.build_subchain_indices(jn, un, contacts_per_foot=0)


def test_subchain_for_derives_the_count_from_the_estimator(_pair):
    """`subchain_for` is the single place N is divided by the number of feet."""
    from invariant_estimation.contactnet import features as F
    from invariant_estimation.sim import collect

    base, th = _pair
    for fused, want in ((base, (2, 6)), (th, (4, 6))):
        un = collect._unfiltered_names(
            type("C", (), {"fused": fused, "dt": 1.0e-3})())
        assert F.subchain_for(fused, un).shape == want

    # A contact count that is not a whole number per foot must not be guessed at.
    bad = type("F", (), {"n_contacts": 3, "build": base.build})()
    with pytest.raises(ValueError, match="whole number per foot"):
        F.subchain_for(bad, [])
