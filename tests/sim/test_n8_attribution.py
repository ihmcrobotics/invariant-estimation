"""Gate B: per-corner contact attribution and per-corner trust.

Two things must be true and they fail in opposite directions:

  * **Conservation** -- the per-corner normal forces must sum back to the per-foot
    force read INDEPENDENTLY off `mj_contactForce` on the foot geom. Catches
    contacts silently dropped by a bad bucket index.
  * **Discrimination** -- a foot in partial contact must report SOME corners
    untrusted. A test that only ever confirms "flat stance trusts all four" agrees
    on a zero: it passes just as happily if `corner_of` returns a constant.

The tilt case is built as a minimal deterministic box-on-plane model rather than
the walking robot, so it cannot go flaky on gait phase.
"""

import mujoco
import numpy as np
import pytest

from invariant_estimation.pipeline import main_estimator as me
from invariant_estimation.sim.sensors import ContactTrust


# ---------------------------------------------------------------------------
# bucketing, against a hand-computed expectation
# ---------------------------------------------------------------------------

def _reader_at(contacts_per_foot):
    """A `SimSensorReader` on the real Alex sim model at the given contact count."""
    import run_policy as rp
    from invariant_estimation.sim.sensors import SimSensorReader

    policy = rp.load_policy("baseline")
    fused = me.build_alex_fused_estimator_from_urdf(
        rp.URDF, contacts_per_foot=contacts_per_foot)
    m = rp.build_sim_model(policy, with_visuals=False, with_imu_sensors=True)
    return m, fused, SimSensorReader(
        m, fused, foot_geoms=rp.FOOT_GEOMS, dt=rp.DT, noise=None)


@pytest.fixture(scope="module")
def sim8():
    return _reader_at(4)


def test_slot_count_is_derived_from_the_estimator(sim8):
    _m, fused, r = sim8
    assert r.contacts_per_foot == 4
    assert fused.n_contacts == 8
    assert r.trust.n_feet == 8, "trust must run per SLOT, not per foot"


def test_corner_bucketing_matches_the_fk_corner_order(sim8):
    """A point placed AT each FK corner must bucket to that corner's index.

    This is the agreement invariant 6 is really about: it ties `corner_of`'s
    bucket j to `alex_foot_corner_offsets()[j]` by construction, using the foot's
    live pose rather than assuming identity.
    """
    m, _fused, r = sim8
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r._xmat, r._xpos = d.xmat, d.xpos

    for k in range(len(r.foot_gids)):
        bid = r.foot_bids[k]
        R = d.xmat[bid].reshape(3, 3)
        for j, off in enumerate(me.alex_foot_corner_offsets()):
            # nudge inward so the point is strictly inside its own quadrant
            cx, cy, _ = r.foot_box_center
            local = np.array(off, float)
            local[0] += 0.01 * np.sign(cx - local[0])
            local[1] += 0.01 * np.sign(cy - local[1])
            world = d.xpos[bid] + R @ local
            assert r.corner_of(k, world) == j, (
                f"corner {j} of foot {k} bucketed to {r.corner_of(k, world)}")


# ---------------------------------------------------------------------------
# conservation: corners sum to the foot
# ---------------------------------------------------------------------------

def test_corner_loads_sum_to_the_independent_per_foot_reading(sim8):
    """Settle the robot, then check per-corner sums == per-foot force.

    `foot_loads` re-reads the contacts rather than summing the corner array, so
    the two sides are genuinely independent computations of the same quantity.
    The normalisers differ by exactly `contacts_per_foot`, which is undone here.
    """
    import run_policy as rp

    m, _fused, r = sim8
    policy = rp.load_policy("baseline")
    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    loop.set_height_target(loop.height_target)
    for _ in range(150):                       # settle into a standing double stance
        loop.control_tick()

    d = loop.d
    per_corner = r.contact_forces(d)                     # raw N, per corner
    per_foot = r.foot_forces(d)                          # raw N, per foot (independent)

    cpf = r.contacts_per_foot
    summed = per_corner.reshape(len(r.foot_gids), cpf).sum(axis=1)
    assert per_foot.sum() > 0.1 * r.weight, "robot is not actually standing on its feet"
    np.testing.assert_allclose(summed, per_foot, rtol=1e-9, atol=1e-9)
    # and the robot's weight is actually accounted for, so this is not a 0 == 0 pass
    np.testing.assert_allclose(per_foot.sum(), r.weight, rtol=0.15)


def test_flat_stance_trusts_all_four_corners(sim8):
    """The normaliser must be scaled so a planted foot trusts every corner.

    Without the `/ cpf` in `contact_loads` each corner sees ~0.25 of the foot load,
    never clears the 0.35 Schmitt `enter`, and every corner stays untrusted forever.
    """
    import run_policy as rp

    m, _fused, r = sim8
    policy = rp.load_policy("baseline")
    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    loop.set_height_target(loop.height_target)
    trusted = np.zeros(8)
    for _ in range(300):
        loop.control_tick()
        trusted = r.trust.update(r.contact_loads(loop.d))

    assert trusted.sum() >= 6, (
        f"standing robot trusts only {trusted.sum():.0f}/8 corners: {trusted}; "
        f"loads={r.contact_loads(loop.d)}")


# ---------------------------------------------------------------------------
# discrimination: partial contact must be representable
# ---------------------------------------------------------------------------

def test_a_tilted_box_leaves_some_corners_untrusted():
    """THE discrimination check: an edge-loaded foot must NOT trust all corners.

    Minimal deterministic model -- a foot-sized box resting tilted on a plane, so
    only one edge touches. If `corner_of` collapsed to a constant, or the trust
    were still per-foot, every corner would report the same and this fails.
    """
    hx, hy, hz = me.ALEX_FOOT_BOX_HALF
    # Posed, then read WITHOUT stepping. Two dead ends this avoids: a free box
    # levels itself within a few hundred steps (the test then silently re-runs the
    # flat case), and a wedge to hold the tilt puts BOTH ends in contact. The
    # freejoint is required -- MuJoCo discards contacts between bodies with no DOFs
    # between them, so a truly static body reports ncon == 0 -- but `mj_forward`
    # only runs collision detection, it does not integrate, so the pose is held.
    #
    # Pitched -12 deg about y with the centre at z=0.052: the heel edge
    # (local x=-hx) penetrates ~2 mm, the toe edge sits ~5 cm clear.
    xml = f"""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" condim="3"/>
        <body name="foot" pos="0 0 0.052" euler="0 -12 0">
          <freejoint/>
          <geom name="box" type="box" size="{hx} {hy} {hz}" condim="3" mass="10"/>
        </body>
      </worldbody>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon > 0, "the tilted box never touched the floor"

    gid = m.geom("box").id
    bid = m.geom_bodyid[gid]
    R = d.xmat[bid].reshape(3, 3)

    # Which corner buckets do the actual contact points fall into? This is the
    # attribution RULE under test; force is irrelevant to it.
    buckets = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 != gid and c.geom2 != gid:
            continue
        loc = R.T @ (c.pos - d.xpos[bid])
        buckets.add(2 * int(loc[0] > 0.0) + int(loc[1] > 0.0))

    assert buckets, "no contact attributed to any corner"
    assert buckets <= {0, 1}, (
        f"heel-only contact attributed to buckets {sorted(buckets)}; expected a "
        f"subset of the two heel corners {{0,1}} -- attribution is not discriminating")
    assert len(buckets) < 4, "partial contact collapsed onto all four corners"

    # and the trust machine must propagate that asymmetry rather than flattening it:
    # the loaded buckets are driven high, the airborne ones at zero.
    t = ContactTrust(n_feet=4, dt=0.001, ema_tau=0.0)
    p = np.array([1.0 if j in buckets else 0.0 for j in range(4)])
    for _ in range(200):
        out = t.update(p)
    assert 0 < out.sum() < 4, f"trust did not preserve the partial contact: {out}"
    np.testing.assert_array_equal(out > 0, p > 0)


# ---------------------------------------------------------------------------
# N=2 regression
# ---------------------------------------------------------------------------

def test_n2_contact_loads_are_bit_identical_to_foot_loads():
    """Invariant 5: at `contacts_per_foot=1` nothing about the trusted run moves."""
    import run_policy as rp

    m, _fused, r = _reader_at(1)
    assert r.contacts_per_foot == 1
    policy = rp.load_policy("baseline")
    loop = rp.Loop(m, policy, rp.make_maps(m, policy))
    loop.set_height_target(loop.height_target)
    for _ in range(120):
        loop.control_tick()

    np.testing.assert_array_equal(r.contact_loads(loop.d), r.foot_loads(loop.d))
