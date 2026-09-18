"""The touchdown-lag path: Java artifact -> loaded lag -> advanced trust channel.

The failure these guard is not a crash. A trust channel shifted the wrong way, by the wrong
number of ticks, or with a fabricated transition at the window edge produces a replay that runs
perfectly and scores plausibly -- wrong by exactly the quantity this path exists to correct.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from invariant_estimation.learning.contact_timing import (TouchdownLag, advance_trust,
                                                          load_touchdown_lag)
from invariant_estimation.replay.logsource import LogWindow


# A literal in the exact shape TouchdownLagEstimator.writeArtifact emits. The Java side pins the
# same keys in TouchdownLagArtifactTest; if the two ever disagree, one of these fails.
ARTIFACT = {
    "schema_version": 1,
    "kind": "touchdown_lag",
    "log": "/opt/ihmc/LogData/incoming/example",
    "dt": 0.001,
    "max_lag_seconds": 1.0,
    "pairs": [
        {"side": "LEFT", "contact": "LEFT_FOOT", "marker_signal": "leftMarkerOnGround",
         "force_signal": "leftHasFootHitGroundFiltered", "lag_seconds": 0.018,
         "peak_score": 0.981, "runner_up_score": 0.912, "confident": True},
        {"side": "RIGHT", "contact": "RIGHT_FOOT", "marker_signal": "rightMarkerOnGround",
         "force_signal": "rightHasFootHitGroundFiltered", "lag_seconds": -0.002,
         "peak_score": 0.640, "runner_up_score": 0.635, "confident": False},
    ],
}


def write_artifact(directory, mutate=None):
    data = json.loads(json.dumps(ARTIFACT))
    if mutate:
        mutate(data)
    path = Path(directory) / "touchdownLag.json"
    path.write_text(json.dumps(data))
    return path


class FakeChannelMap:
    anchor_trust = {"LEFT_FOOT": "isLEFT_FOOTFootTrusted", "RIGHT_FOOT": "isRIGHT_FOOTFootTrusted"}


def window(**channels):
    T = len(next(iter(channels.values())))
    return LogWindow(log_dir=Path("/nonexistent"), time=np.arange(T) * 1e-3,
                     tick=np.arange(T), channels={k: np.asarray(v, float) for k, v in channels.items()},
                     dt=1e-3)


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------

class TestLoad:
    def test_a_java_shaped_artifact_loads(self, tmp_path):
        lag = load_touchdown_lag(write_artifact(tmp_path))
        assert lag.lag_seconds == {"LEFT_FOOT": 0.018, "RIGHT_FOOT": -0.002}
        assert lag.confident == {"LEFT_FOOT": True, "RIGHT_FOOT": False}
        assert lag.dt == 0.001

    def test_the_wrong_kind_is_rejected_at_load_not_applied_as_zero(self, tmp_path):
        path = write_artifact(tmp_path, lambda d: d.update(kind="learned_noise"))
        with pytest.raises(ValueError, match="kind"):
            load_touchdown_lag(path)

    def test_an_unknown_schema_version_is_rejected(self, tmp_path):
        path = write_artifact(tmp_path, lambda d: d.update(schema_version=2))
        with pytest.raises(ValueError, match="schema_version"):
            load_touchdown_lag(path)

    def test_a_duplicate_contact_is_rejected(self, tmp_path):
        def dup(d):
            d["pairs"].append(dict(d["pairs"][0]))
        with pytest.raises(ValueError, match="duplicate"):
            load_touchdown_lag(write_artifact(tmp_path, dup))


# ---------------------------------------------------------------------------------------------
# lag -> ticks
# ---------------------------------------------------------------------------------------------

class TestAdvanceTicks:
    def lag(self):
        return TouchdownLag(log="x", dt=0.001,
                            lag_seconds={"LEFT_FOOT": 0.018, "RIGHT_FOOT": -0.002},
                            confident={"LEFT_FOOT": True, "RIGHT_FOOT": False})

    def test_positive_lag_becomes_a_positive_advance_at_the_sessions_dt(self):
        # 18 ms at dt=1 ms -> 18 ticks; at dt=2 ms -> 9. The session's dt governs, not the
        # artifact's, because the correction is applied to the session's tick axis.
        assert self.lag().advance_ticks("LEFT_FOOT", 0.001) == 18
        assert self.lag().advance_ticks("LEFT_FOOT", 0.002) == 9

    def test_a_flat_peak_measurement_is_refused_by_default(self):
        with pytest.raises(ValueError, match="not confident"):
            self.lag().advance_ticks("RIGHT_FOOT", 0.001)
        # and the override is explicit, not a softer default
        assert self.lag().advance_ticks("RIGHT_FOOT", 0.001, allow_unconfident=True) == -2

    def test_an_unknown_contact_names_what_the_artifact_does_have(self):
        with pytest.raises(KeyError, match="LEFT_FOOT"):
            self.lag().advance_ticks("PELVIS", 0.001)


# ---------------------------------------------------------------------------------------------
# applying the advance
# ---------------------------------------------------------------------------------------------

class TestAdvanceTrust:
    def test_the_step_moves_earlier_by_exactly_the_advance(self):
        trust = np.zeros(100); trust[60:] = 1.0     # touchdown detected at tick 60
        w = window(isLEFT_FOOTFootTrusted=trust, other=np.arange(100))
        out = advance_trust(w, FakeChannelMap(), {"LEFT_FOOT": 18})

        got = out.channels["isLEFT_FOOTFootTrusted"]
        assert got[41] == 0.0 and got[42] == 1.0, "the step must now sit at tick 60-18=42"

    def test_the_tail_holds_the_last_value_rather_than_fabricating_a_liftoff(self):
        trust = np.zeros(100); trust[60:] = 1.0
        w = window(isLEFT_FOOTFootTrusted=trust)
        got = advance_trust(w, FakeChannelMap(), {"LEFT_FOOT": 18}).channels["isLEFT_FOOTFootTrusted"]
        assert (got[-18:] == 1.0).all(), "zero-filling here would invent a contact transition at the edge"

    def test_a_negative_advance_delays_and_holds_the_head(self):
        trust = np.zeros(100); trust[60:] = 1.0
        w = window(isRIGHT_FOOTFootTrusted=trust)
        got = advance_trust(w, FakeChannelMap(), {"RIGHT_FOOT": -5}).channels["isRIGHT_FOOTFootTrusted"]
        assert got[64] == 0.0 and got[65] == 1.0
        assert (got[:5] == 0.0).all()

    def test_the_input_window_is_untouched_and_other_channels_are_shared(self):
        trust = np.zeros(50); trust[20:] = 1.0
        w = window(isLEFT_FOOTFootTrusted=trust, other=np.arange(50))
        out = advance_trust(w, FakeChannelMap(), {"LEFT_FOOT": 3})

        assert w.channels["isLEFT_FOOTFootTrusted"][21] == 1.0 and w.channels["isLEFT_FOOTFootTrusted"][17] == 0.0
        assert out.channels["other"] is w.channels["other"], "untouched channels must be shared, not copied"

    def test_zero_advance_changes_nothing(self):
        trust = np.zeros(50); trust[20:] = 1.0
        w = window(isLEFT_FOOTFootTrusted=trust)
        out = advance_trust(w, FakeChannelMap(), {"LEFT_FOOT": 0})
        assert out.channels["isLEFT_FOOTFootTrusted"] is w.channels["isLEFT_FOOTFootTrusted"]

    def test_an_advance_longer_than_the_window_is_an_error_not_a_constant_channel(self):
        w = window(isLEFT_FOOTFootTrusted=np.zeros(10))
        with pytest.raises(ValueError, match="exceeds"):
            advance_trust(w, FakeChannelMap(), {"LEFT_FOOT": 10})

    def test_a_contact_the_map_does_not_know_is_an_error(self):
        w = window(isLEFT_FOOTFootTrusted=np.zeros(10))
        with pytest.raises(KeyError, match="PELVIS"):
            advance_trust(w, FakeChannelMap(), {"PELVIS": 1})

    def test_the_result_stays_binary_so_the_adapters_own_check_still_passes(self):
        trust = np.zeros(100); trust[30:70] = 1.0
        w = window(isLEFT_FOOTFootTrusted=trust)
        got = advance_trust(w, FakeChannelMap(), {"LEFT_FOOT": 7}).channels["isLEFT_FOOTFootTrusted"]
        assert np.isin(got, [0, 1]).all()
