"""The foot-sole frame — the InEKF's contact anchor and the joint KF's stance anchor.

Why this file exists
--------------------
`ALEX_EXTRA_SITES` emitted `left_sole`/`right_sole` at the `*_FOOT` link origin until
2026-07-26. That origin is the **ankle-roll** frame, so both foot sites sat 7.2 cm above the
ground and 4.65 cm behind the sole centre, while Java's `InvariantMainStateEstimator` anchors
contacts at `referenceFrames.getSoleFrame(side)`.

It survived a hardware-validated suite because the only test that looked at foot-site geometry
(`tests/replay/test_fused_real_model.py::test_urdf_site_fk_is_bit_identical_to_java`) compares the
training URDF's site FK against the *log model's* site FK — both built from the same
`ALEX_EXTRA_SITES`. The sole was therefore only ever compared against itself and would have matched
at any offset. So the checks here deliberately reach the sole plane by routes that do NOT go through
`ALEX_EXTRA_SITES`:

    the constant  -> the Java numbers, written out (AlexV1PhysicalProperties)
    the geometry  -> where the robot's feet actually come to rest on a floor, from contact
"""
from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import pytest

from invariant_estimation.pipeline import main_estimator as me

# tests/model/ -> repo root -> assets/. Vendored so this file's fixture runs on a bare clone.
ASSETS_URDF = pathlib.Path(__file__).resolve().parents[2] / "assets" / "alex_with_imus.urdf"

# AlexV1PhysicalProperties, transcribed. `soleToAnkleFrameTransforms` is
#   translation = (ACTUAL_FOOT_LENGTH / 2 - FOOT_BACK, 0, -ANKLE_HEIGHT)
# with the rotation left commented out, i.e. a pure translation in the ankle-roll frame.
JAVA_ANKLE_HEIGHT = 0.072
JAVA_ACTUAL_FOOT_LENGTH = 0.197
JAVA_FOOT_BACK = 0.052

# SCS2's MuJoCo foot collidable (AlexSimulationCollisionModel): 0.26 x 0.14 x 0.055 at
# (0.045, 0, -0.05) in the ankle-roll frame, so its underside sits at -0.0775.
FOOT_BOX_HALF = "0.13 0.07 0.0275"
FOOT_BOX_POS = "0.045 0 -0.05"
# How far the sole plane sits above that box underside: -0.0720 - (-0.0775).
# Deliberately a LITERAL and not `-JAVA_ANKLE_HEIGHT - box_underside`: computing it from
# JAVA_ANKLE_HEIGHT makes the expectation move with the very number under test, so an ankle
# height mis-transcribed into *both* this file and the source would sail through (verified --
# it did). Written out, this is the one figure here that cross-checks two independent Java
# classes, and either transcription drifting breaks the check below.
FOOT_BOX_SOLE_CLEARANCE = 0.0055


@pytest.fixture(scope="module")
def alex_mjcf():
    """The Alex MJCF from the RL training URDF, vendored at `assets/alex_with_imus.urdf`.

    Override the path with $ALEX_URDF to run against a working copy. This used to default to
    `~/Documents/`, i.e. it skipped everywhere but Lucas's laptop — and then only
    `test_sole_offset_matches_the_java_transform` still constrained anything, the two tests that
    catch a broken `extra_sites` emission or a correlated mis-transcription disappearing into
    skips. The skip is kept for the $ALEX_URDF override (a missing model is a missing fixture, not
    a failure, same trade `tests/replay/` makes), but off the default path it can no longer fire.
    """
    path = pathlib.Path(os.environ.get("ALEX_URDF", str(ASSETS_URDF)))
    if not path.exists():
        pytest.skip(f"no Alex URDF at {path} (set $ALEX_URDF)")
    return me.alex_spec_from_urdf(path).mjcf


def _site_pos(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert sid >= 0, f"site {name} missing from the model"
    return tuple(float(v) for v in model.site_pos[sid])


def test_sole_offset_matches_the_java_transform():
    """The constant itself, against the Java source numbers -- not against our own model."""
    expected = (JAVA_ACTUAL_FOOT_LENGTH / 2.0 - JAVA_FOOT_BACK, 0.0, -JAVA_ANKLE_HEIGHT)
    assert me.ALEX_ANKLE_HEIGHT == pytest.approx(JAVA_ANKLE_HEIGHT, abs=0.0)
    assert me.ALEX_SOLE_OFFSET == pytest.approx(expected, abs=1e-12)
    # The offset is NOT purely vertical: the sole frame sits at the centre of the foot, forward of
    # the ankle. A z-only offset is the plausible half-fix, so pin the forward term too.
    assert me.ALEX_SOLE_OFFSET[0] > 0.04, "the sole's forward offset from the ankle is missing"


def test_soles_are_offset_from_the_link_origin_and_base_body_is_not(alex_mjcf):
    """Plumbing, all three axes: the constant actually reaches the emitted MJCF site.

    Deliberately self-referential -- it compares the model against `ALEX_SOLE_OFFSET`, so it says
    nothing about whether that offset is *right* (that is the test above, and the ground check
    below). What it alone catches is the offset being dropped or mangled between the table and the
    MJCF, on x and y as well as z, and `base_body` silently acquiring an offset of its own.
    """
    model = mujoco.MjModel.from_xml_string(alex_mjcf)
    for name in me.ALEX_FOOT_SITES:
        assert _site_pos(model, name) == pytest.approx(me.ALEX_SOLE_OFFSET, abs=1e-12)
    # The InEKF body frame IS the pelvis link origin; it must not pick up an offset.
    assert _site_pos(model, "base_body") == pytest.approx((0.0, 0.0, 0.0), abs=0.0)


def test_sole_plane_sits_where_the_foot_meets_the_ground(alex_mjcf):
    """Kinematic, exact: the sole site must sit on the plane the foot's underside rests on.

    No dynamics -- an unactuated drop just collapses the legs and measures nothing. Instead pose the
    robot, place it so the lowest foot-box corner is exactly on z = 0, and check the sole site
    stands `FOOT_BOX_SOLE_CLEARANCE` above it. At the old (ankle-origin) offset the sole floats
    0.0775 m up instead of 0.0055 m.

    This is the only check here that reaches the sole *height* without going through
    `ALEX_SOLE_OFFSET` or `JAVA_ANKLE_HEIGHT`: the expectation comes from the collision-box numbers
    plus the written-out clearance, so it still fires when the ankle height is wrong everywhere at
    once. The guard below is what keeps those two Java transcriptions pinned to each other.
    """
    box_underside = float(FOOT_BOX_POS.split()[2]) - float(FOOT_BOX_HALF.split()[2])   # -0.0775
    assert -JAVA_ANKLE_HEIGHT - box_underside == pytest.approx(FOOT_BOX_SOLE_CLEARANCE, abs=1e-12), (
        "AlexV1PhysicalProperties.ANKLE_HEIGHT and the AlexSimulationCollisionModel foot box "
        "disagree about where the sole plane is; one of the two transcriptions has drifted"
    )
    expected_residual = FOOT_BOX_SOLE_CLEARANCE

    root = ET.fromstring(alex_mjcf)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for body in ("LEFT_FOOT", "RIGHT_FOOT"):
        box = ET.SubElement(bodies[body], "geom")
        box.set("name", f"{body}_box"); box.set("type", "box")
        box.set("size", FOOT_BOX_HALF); box.set("pos", FOOT_BOX_POS)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    for name, angle in (("HIP_Y", -0.35), ("KNEE_Y", 0.7), ("ANKLE_Y", -0.35)):
        for side in ("LEFT", "RIGHT"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}")
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = angle
    data.qpos[2] = 1.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    def box_bottom(body):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_box")
        return float(data.geom_xpos[gid][2] - model.geom_size[gid][2])

    # Stand it on the floor: shift the root down so the lower foot box just touches z = 0.
    data.qpos[2] -= min(box_bottom("LEFT_FOOT"), box_bottom("RIGHT_FOOT"))
    mujoco.mj_forward(model, data)

    for site, body in zip(me.ALEX_FOOT_SITES, ("LEFT_FOOT", "RIGHT_FOOT")):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        residual = float(data.site_xpos[sid][2]) - box_bottom(body)
        assert residual == pytest.approx(expected_residual, abs=1e-9), (
            f"{site} sits {residual:+.4f} m above the foot's underside; expected "
            f"{expected_residual:+.4f} m. The sole frame is not the contact plane "
            f"(at the ankle origin this is {-box_underside:+.4f} m)."
        )
        # And therefore it is on the ground the robot stands on, to within that 5.5 mm.
        assert float(data.site_xpos[sid][2]) == pytest.approx(FOOT_BOX_SOLE_CLEARANCE, abs=1e-9)
