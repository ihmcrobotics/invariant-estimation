"""The vendored SCS2 log reader.

Two kinds of check, because neither alone is enough:

* **Synthetic round trip** -- a log is written here, byte by byte, to the format
  the Java reader defines, then decoded. Every value is known exactly, so this
  pins the container, the batching, the big-endian int64 records and the
  long-bits interpretation with no external dependency. It is the only test that
  can run anywhere.
* **Real log** -- decoded values must be physically what they claim to be
  (gravity on the accelerometer, radians on a knee). A format error that
  survived the round trip because the fixture shares its misunderstanding cannot
  survive gravity coming out at 9.8. Skipped when no log is present.

What is NOT covered: the full Java-parity gate in `test_java_parity.py`, which
compares this port against what the Java estimator published in a log. No log
now on this machine contains the `jointKF_*` variables it needs -- the robot is
not currently running that filter, and the 2026-07-17 log the harness was
written against has been deleted from the store. That gate remains the real
acceptance test for the port; it is unrun, not passed.
"""
import struct
from pathlib import Path

import numpy as np
import pytest

from invariant_estimation.replay import ihmclog

REAL_LOG = Path("/opt/ihmc/LogData/incoming/20260916_101745_Alex001UnifiedControlProcess")


def write_log(directory, variables, ticks, *, batch_size=4, joints=(), compress=True):
    """Write a log in the Java format. ``variables`` is [(name, type)], ``ticks`` is
    [(timestamp, [raw_int64_per_variable])]."""
    directory.mkdir(parents=True, exist_ok=True)

    lines = ["---", "us::ihmc::robotDataLogger::Handshake:", "  variables:"]
    for name, kind in variables:
        lines += [f'  - registry: 1', f'    name: "{name}"', f'    type: "{kind}"']
    if joints:
        lines.append("  joints:")
        for name, kind in joints:
            lines += [f'  - name: "{name}"', f'    type: "{kind}"']
    lines += ["  dt: 0.001", "  registries:", '  - name: "root"']
    (directory / "handshake.yaml").write_text("\n".join(lines) + "\n")

    joint_state_values = sum(ihmclog.JOINT_STATE_SIZES[k] for _, k in joints)
    per_tick = 1 + len(variables) + joint_state_values

    batches, current = [], []
    for timestamp, values in ticks:
        assert len(values) == len(variables)
        record = [timestamp] + list(values) + [0] * joint_state_values
        current.append(struct.pack(f">{per_tick}q", *record))
        if len(current) == batch_size:
            batches.append(b"".join(current))
            current = []
    valid_last = batch_size
    if current:
        valid_last = len(current)
        # A short final batch is still written at full length -- the reader is told how many
        # of its ticks are valid rather than inferring it from the size.
        current += [b"\0" * (per_tick * 8)] * (batch_size - len(current))
        batches.append(b"".join(current))

    payloads = []
    for raw in batches:
        if compress:
            import zstandard

            payloads.append(zstandard.ZstdCompressor().compress(raw))
        else:
            payloads.append(raw)

    offsets, position = [], 0
    for payload in payloads:
        offsets.append(position)
        position += len(payload)
    (directory / "robotData.bsz").write_bytes(b"".join(payloads))
    (directory / "robotData.dat").write_bytes(
        b"".join(struct.pack(">qq", ticks[i * batch_size][0], offsets[i]) for i in range(len(batches))))

    (directory / "robotData.log").write_text(
        "variables.compressed=" + ("true" if compress else "false") + "\n"
        + ("variables.compressionType=zstd\n" if compress else "variables.compressionType=none\n")
        + f"variables.compressionBatchSize={batch_size}\n"
        f"variables.validTicksInLastBatch={valid_last}\n"
        "variables.index=robotData.dat\nvariables.data=robotData.bsz\n"
        "variables.handshake=handshake.yaml\n")
    return directory


def as_bits(value):
    """A double as the int64 bit pattern the format stores."""
    return struct.unpack(">q", struct.pack(">d", value))[0]


class TestSyntheticRoundTrip:
    def _log(self, tmp_path, **kwargs):
        variables = [("pos", "DoubleYoVariable"), ("count", "IntegerYoVariable"),
                     ("flag", "BooleanYoVariable")]
        ticks = [(1_000_000 * (t + 1), [as_bits(0.25 + t), t * 10, t % 2]) for t in range(10)]
        return write_log(tmp_path / "log", variables, ticks, **kwargs), ticks

    def test_doubles_come_back_as_doubles_not_as_their_bit_patterns(self, tmp_path):
        """The trap the format sets: a double is stored as int64 bits, so reading the integer
        gives ~4.6e18 rather than 0.25 -- a number so wrong it looks like a different variable."""
        directory, _ = self._log(tmp_path)
        reader = ihmclog.LogReader(directory)
        _, _, data, _, _ = ihmclog.gather(reader, ["pos"])
        np.testing.assert_allclose(data[:, 0], [0.25 + t for t in range(10)])

    def test_integer_and_boolean_types_are_read_as_numbers_not_bit_patterns(self, tmp_path):
        directory, _ = self._log(tmp_path)
        reader = ihmclog.LogReader(directory)
        _, _, data, _, _ = ihmclog.gather(reader, ["count", "flag"])
        np.testing.assert_allclose(data[:, 0], [t * 10 for t in range(10)])
        np.testing.assert_allclose(data[:, 1], [t % 2 for t in range(10)])

    def test_columns_follow_the_requested_order_not_the_handshake_order(self, tmp_path):
        directory, _ = self._log(tmp_path)
        reader = ihmclog.LogReader(directory)
        _, _, data, _, _ = ihmclog.gather(reader, ["flag", "pos"])
        np.testing.assert_allclose(data[:, 1], [0.25 + t for t in range(10)])

    def test_a_partial_final_batch_exposes_only_its_valid_ticks(self, tmp_path):
        """10 ticks at batch size 4 leaves a final batch with 2 real ticks and 2 of padding;
        reading the padding would hand the caller zeros that look like real samples."""
        directory, _ = self._log(tmp_path, batch_size=4)
        reader = ihmclog.LogReader(directory)
        assert reader.n_ticks == 10
        _, _, data, _, _ = ihmclog.gather(reader, ["pos"])
        assert data.shape[0] == 10
        assert data[-1, 0] == pytest.approx(9.25)

    def test_time_comes_from_the_logged_timestamps(self, tmp_path):
        """Not from tick*dt: a dropped controller tick would otherwise go unnoticed and shift
        everything after it."""
        directory, ticks = self._log(tmp_path)
        reader = ihmclog.LogReader(directory)
        _, time, _, _, _ = ihmclog.gather(reader, ["pos"])
        expected = [(t[0] - ticks[0][0]) * 1e-9 for t in ticks]
        np.testing.assert_allclose(time, expected)

    def test_stride_and_window_select_the_right_ticks(self, tmp_path):
        directory, _ = self._log(tmp_path)
        reader = ihmclog.LogReader(directory)
        indices, _, data, _, _ = ihmclog.gather(reader, ["pos"], stride=3)
        assert indices.tolist() == [0, 3, 6, 9]
        np.testing.assert_allclose(data[:, 0], [0.25, 3.25, 6.25, 9.25])

    def test_an_uncompressed_log_reads_identically(self, tmp_path):
        directory, _ = self._log(tmp_path, compress=False)
        reader = ihmclog.LogReader(directory)
        _, _, data, _, _ = ihmclog.gather(reader, ["pos"])
        np.testing.assert_allclose(data[:, 0], [0.25 + t for t in range(10)])

    def test_joint_states_shift_the_record_without_shifting_the_variables(self, tmp_path):
        """Joint states sit AFTER the variables, so their presence changes the record size but
        must not change where a variable is found. Getting this wrong reads a neighbour."""
        variables = [("pos", "DoubleYoVariable")]
        ticks = [(1_000_000 * (t + 1), [as_bits(t + 0.5)]) for t in range(8)]
        directory = write_log(tmp_path / "j", variables, ticks,
                              joints=[("PELVIS", "SiXDoFJoint"), ("KNEE", "OneDoFJoint")])
        reader = ihmclog.LogReader(directory)
        assert reader.hs.n_joint_state_variables == 15  # 13 + 2
        _, _, data, _, _ = ihmclog.gather(reader, ["pos"])
        np.testing.assert_allclose(data[:, 0], [t + 0.5 for t in range(8)])

    def test_an_unknown_variable_is_named_in_the_error(self, tmp_path):
        directory, _ = self._log(tmp_path)
        reader = ihmclog.LogReader(directory)
        with pytest.raises(KeyError, match="nope"):
            ihmclog.gather(reader, ["pos", "nope"])


@pytest.mark.skipif(not REAL_LOG.exists(), reason=f"no hardware log at {REAL_LOG}")
class TestAgainstARealLog:
    """Physical checks a wrong format cannot pass by coincidence."""

    def test_the_record_size_agrees_with_the_data(self):
        """The strongest structural check available: a batch must decompress to a whole number
        of records of the size the handshake implies. A miscounted variable or joint state
        makes this fail immediately rather than silently shifting every column."""
        reader = ihmclog.LogReader(REAL_LOG)
        batch = reader.batch(0)
        assert batch.shape[1] * 8 == reader.tick_size
        assert batch.shape[0] == reader.batch_size

    def test_the_accelerometer_reads_gravity(self):
        """Nothing about a wrong column offset or a wrong bit interpretation produces 9.8 on
        the z axis of a standing robot."""
        reader = ihmclog.LogReader(REAL_LOG)
        names = [f"accelerometer_pelvis_imu{a}" for a in "XYZ"]
        _, _, data, _, _ = ihmclog.gather(reader, names, stride=500, end=10.0)
        magnitude = np.linalg.norm(data, axis=1)
        assert 9.0 < magnitude.mean() < 10.5, f"specific force magnitude {magnitude.mean()}"

    def test_a_joint_position_is_in_radians(self):
        reader = ihmclog.LogReader(REAL_LOG)
        _, _, data, _, _ = ihmclog.gather(reader, ["raw_q_LEFT_KNEE_Y"], stride=500, end=10.0)
        assert np.all(np.abs(data) < np.pi), "a knee angle outside +/-pi is not radians"

    def test_the_processing_chain_is_self_consistent_across_three_columns(self):
        """`stiff_q = raw_q + Deflect` must hold sample by sample. This is the strongest
        available check without the Java oracle: it constrains THREE decoded columns against
        each other, so any column misalignment, or any wrong bit interpretation, breaks it.

        (An earlier version of this test merely asserted the two stages differ. They are in
        fact bit-identical wherever the deflection is zero -- 340 of 600 sampled ticks on this
        log -- so that assertion was both weaker and wrong.)
        """
        reader = ihmclog.LogReader(REAL_LOG)
        names = ["raw_q_LEFT_KNEE_Y", "stiff_q_LEFT_KNEE_Y_sp1", "stiff_q_LEFT_KNEE_Y_sp1_Deflect"]
        _, _, data, _, _ = ihmclog.gather(reader, names, stride=50, end=30.0)
        raw, stiff, deflection = data[:, 0], data[:, 1], data[:, 2]

        # MINUS: the deflection is removed to recover the true joint angle from the measured,
        # deflected one. Determined from the data, not assumed -- the opposite sign is off by
        # 2x the deflection and would have looked like a plausible small disagreement.
        np.testing.assert_allclose(stiff, raw - deflection, atol=0.0, rtol=0.0,
                                   err_msg="the elasticity stage does not equal raw - deflection")
        assert np.abs(deflection).max() > 0.0, (
            "the deflection is zero at every sample, so this check is vacuous here")

    def test_a_boolean_variable_is_exactly_zero_or_one(self):
        reader = ihmclog.LogReader(REAL_LOG)
        _, _, data, _, _ = ihmclog.gather(reader, ["isLEFT_FOOTFootTrusted"], stride=500, end=10.0)
        assert set(np.unique(data)) <= {0.0, 1.0}

    def test_timestamps_advance_by_about_the_sample_period(self):
        reader = ihmclog.LogReader(REAL_LOG)
        _, time, _, _, _ = ihmclog.gather(reader, ["raw_q_LEFT_KNEE_Y"], end=1.0)
        steps = np.diff(time)
        assert steps.min() > 0, "timestamps must be strictly increasing"
        assert abs(np.median(steps) - reader.hs.dt) < 0.5 * reader.hs.dt
