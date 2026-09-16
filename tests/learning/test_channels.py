"""Channel-map resolution against a log's own variable list.

Fixtures reproduce the stage layout observed on a real Alex log
(`20260916_101745_Alex001UnifiedControlProcess`): leg joints carry an elasticity
stage, SPINE_Z does not, and no joint has a stiff velocity. That asymmetry is the
whole reason this module exists, so it is what the fixtures encode.
"""
import pytest

from invariant_estimation.learning.channels import alex_channel_map
from invariant_estimation.replay.logsource import handshake_names, resolve_joint_channel

JOINTS = ("LEFT_KNEE_Y", "SPINE_Z")
IMUS = ("pelvis_imu", "torso_imu")
CONTACTS = ("LEFT_FOOT", "RIGHT_FOOT")


def realistic_names():
    """The stage layout a real Alex log actually publishes."""
    names = {
        # Leg joint: raw -> alpha filter -> elasticity compensation.
        "raw_q_LEFT_KNEE_Y", "filt_q_LEFT_KNEE_Y_sp0", "stiff_q_LEFT_KNEE_Y_sp1",
        # Decoys that end in something other than the stage index.
        "stiff_q_LEFT_KNEE_Y_sp1_Deflect", "filt_q_LEFT_KNEE_Y_sp0HasBeenCalled",
        # Velocity stops at the alpha filter for every joint.
        "raw_qd_LEFT_KNEE_Y", "filt_qd_LEFT_KNEE_Y_sp0",
        # Spine: no elasticity stage at all.
        "raw_q_SPINE_Z", "filt_q_SPINE_Z_sp0",
        "raw_qd_SPINE_Z", "filt_qd_SPINE_Z_sp0",
    }
    for imu in IMUS:
        names |= {f"gyroscope_{imu}{a}" for a in "XYZ"}
        names |= {f"accelerometer_{imu}{a}" for a in "XYZ"}
    names |= {f"is{c}FootTrusted" for c in CONTACTS}
    return names


def build(names=None, **kwargs):
    kwargs.setdefault("base_imu", "pelvis_imu")
    kwargs.setdefault("sensor_processing", "test fixture")
    return alex_channel_map(names if names is not None else realistic_names(),
                            JOINTS, IMUS, CONTACTS, **kwargs)


class TestStageResolution:
    def test_each_joint_gets_its_own_last_stage_not_a_shared_suffix(self):
        """The headline case: on a real log these two joints end at DIFFERENT stages, so any
        uniform rule is wrong for one of them."""
        channels = build()
        assert channels.positions["LEFT_KNEE_Y"] == "stiff_q_LEFT_KNEE_Y_sp1"
        assert channels.positions["SPINE_Z"] == "filt_q_SPINE_Z_sp0"

    def test_velocity_resolves_independently_of_position(self):
        """Position reaches sp1 while velocity stops at sp0; reusing the position suffix for
        velocity would ask for a channel that does not exist."""
        channels = build()
        assert channels.velocities["LEFT_KNEE_Y"] == "filt_qd_LEFT_KNEE_Y_sp0"

    def test_companion_variables_are_not_mistaken_for_stages(self):
        """`_Deflect` and `HasBeenCalled` share the stage prefix but are not the measurement."""
        channels = build()
        assert "Deflect" not in channels.positions["LEFT_KNEE_Y"]
        assert "HasBeenCalled" not in channels.positions["LEFT_KNEE_Y"]

    def test_a_log_with_no_processing_chain_falls_back_to_raw(self):
        names = {"raw_q_LEFT_KNEE_Y", "raw_qd_LEFT_KNEE_Y", "raw_q_SPINE_Z", "raw_qd_SPINE_Z"}
        for imu in IMUS:
            names |= {f"gyroscope_{imu}{a}" for a in "XYZ"} | {f"accelerometer_{imu}{a}" for a in "XYZ"}
        names |= {f"is{c}FootTrusted" for c in CONTACTS}
        assert build(names).positions["LEFT_KNEE_Y"] == "raw_q_LEFT_KNEE_Y"

    def test_a_higher_stage_wins_regardless_of_its_prefix(self):
        """Selection is by stage INDEX, not by a hardcoded list of stage names, so a new
        processing stage is picked up without editing this code."""
        names = realistic_names() | {"newstage_q_SPINE_Z_sp7"}
        assert build(names).positions["SPINE_Z"] == "newstage_q_SPINE_Z_sp7"


class TestSensorChannels:
    def test_imu_and_accel_triples_are_in_xyz_order(self):
        channels = build()
        assert channels.gyros["pelvis_imu"] == (
            "gyroscope_pelvis_imuX", "gyroscope_pelvis_imuY", "gyroscope_pelvis_imuZ")
        assert channels.accel == (
            "accelerometer_pelvis_imuX", "accelerometer_pelvis_imuY", "accelerometer_pelvis_imuZ")

    def test_trust_and_probability_share_the_binary_channel_by_default(self):
        """Permitted on purpose -- a binary trust signal IS a valid [0,1] probability, and no
        continuous contact probability is published on the Alex logs checked so far. What must
        never happen is silently thresholding a smoothed signal, which is why supplying a real
        probability channel is opt-in."""
        channels = build()
        assert channels.contact_probability == channels.anchor_trust

    def test_an_explicit_probability_format_is_used_when_one_really_exists(self):
        names = realistic_names() | {f"{c}ContactProbability" for c in CONTACTS}
        channels = build(names, probability_format="{contact}ContactProbability")
        assert channels.contact_probability["LEFT_FOOT"] == "LEFT_FOOTContactProbability"
        assert channels.contact_probability != channels.anchor_trust


class TestFailsLoudly:
    def test_a_missing_joint_channel_names_the_joint(self):
        names = {n for n in realistic_names() if "SPINE_Z" not in n}
        with pytest.raises(KeyError, match="SPINE_Z"):
            build(names)

    def test_a_missing_imu_names_the_imu_and_the_channels(self):
        names = {n for n in realistic_names() if "torso_imu" not in n}
        with pytest.raises(KeyError, match="torso_imu"):
            build(names)

    def test_a_missing_trust_channel_names_the_contact(self):
        names = {n for n in realistic_names() if n != "isRIGHT_FOOTFootTrusted"}
        with pytest.raises(KeyError, match="RIGHT_FOOT"):
            build(names)

    def test_a_base_imu_outside_the_imu_set_is_rejected(self):
        with pytest.raises(ValueError, match="base_imu"):
            build(base_imu="head_imu")

    def test_blank_provenance_is_rejected(self):
        with pytest.raises(ValueError, match="processing chain"):
            build(sensor_processing="   ")


class TestResolvedMapIsUsable:
    def test_the_map_passes_the_adapters_own_validation(self):
        """required_channels() is what prepare_session calls; a map that cannot survive it is
        not a map, however plausible it looks."""
        required = build().required_channels()
        assert "stiff_q_LEFT_KNEE_Y_sp1" in required
        assert "filt_q_SPINE_Z_sp0" in required
        assert len(required) == len(set(required)), "required channels must be de-duplicated"


class TestAgainstARealLog:
    """Skipped unless a real log is present; this machine has them under /opt/ihmc/LogData."""

    LOG = "/opt/ihmc/LogData/incoming/20260916_101745_Alex001UnifiedControlProcess"

    def _names(self):
        import pathlib
        if not (pathlib.Path(self.LOG) / "handshake.yaml").exists():
            pytest.skip(f"no log at {self.LOG}")
        return handshake_names(self.LOG)

    def test_handshake_names_are_readable_without_the_binary_decoder(self):
        """The point of reading handshake.yaml directly: the `ihmclog` decoder is only needed
        for robotData.bsz, so a channel map can be resolved on a machine that lacks it."""
        names = self._names()
        assert len(names) > 1000
        assert "raw_q_LEFT_KNEE_Y" in names

    def test_the_real_logs_stage_layout_is_what_the_fixtures_claim(self):
        """Pins the fixtures above to reality -- if a future robot build changes the chain,
        this fails here rather than silently invalidating every test in this file."""
        names = self._names()
        assert resolve_joint_channel(names, "LEFT_KNEE_Y", "q") == "stiff_q_LEFT_KNEE_Y_sp1"
        assert resolve_joint_channel(names, "SPINE_Z", "q") == "filt_q_SPINE_Z_sp0"
        assert resolve_joint_channel(names, "LEFT_KNEE_Y", "qd") == "filt_qd_LEFT_KNEE_Y_sp0"
