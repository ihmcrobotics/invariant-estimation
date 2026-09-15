"""The rehearsal itself, run small enough for CI.

These are the assertions that would have to hold before anyone spends robot time on this
pipeline. They are deliberately about the MACHINERY -- gradients reach the channels, the
split is honoured, the export validates -- and not about accuracy, because the data is
generated under exactly the assumptions the filters make. See `synthetic.py`'s docstring.
"""
import json

import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.learning.artifact import load_artifact
from invariant_estimation.learning.dryrun import dry_run
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.learning.optimize import body_velocity_l2
from invariant_estimation.learning.synthetic import generate_session
from invariant_estimation.learning.two_stage import run

SMALL = dict(n_sessions=3, n_test=1, ticks=48, steps=12, learning_rate=0.1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return dry_run(tmp_path_factory.mktemp("dryrun"), **SMALL)


class TestGeneratedData:
    def test_the_sensors_are_derived_from_the_truth_not_drawn_independently(self):
        """The point of `synthetic` over the existing fixtures: if the sensor stream and the
        truth were unrelated, no noise setting could reduce the loss and a 'successful' training
        run would mean nothing. A filter run on this data must track the truth's magnitude."""
        session = generate_session(1, ticks=96)
        spec = NoiseSpec(tuple(session.build.imu_names), arm=7)
        _, out = run(spec.initial_theta(), spec, session.build, session.joint_params, session.ekf,
                     session.kinematics, jnp.eye(3), session.carry, session.inputs)

        estimated = float(jnp.mean(jnp.linalg.norm(out.base.state.v, axis=-1)))
        truth = float(jnp.mean(jnp.linalg.norm(session.truth_velocity, axis=-1)))
        assert np.isfinite(estimated)
        assert 0.3 * truth < estimated < 3.0 * truth, "the filter must be tracking, not diverging"

    def test_a_stationary_upright_base_would_read_gravity_with_the_filters_sign(self):
        """Pins the accelerometer convention: `v̇ = R a_body + g`, so a static upright base reads
        +9.81 on z. A sign flip here silently inverts every gravity-leveling update."""
        session = generate_session(3, ticks=4)
        first_accel = np.asarray(session.inputs.accel_body[0])
        assert first_accel[2] > 5.0, f"expected roughly +g on z, got {first_accel}"

    def test_different_seeds_give_genuinely_different_trajectories(self):
        a, b = generate_session(1, ticks=32), generate_session(2, ticks=32)
        assert not np.allclose(np.asarray(a.truth_velocity), np.asarray(b.truth_velocity))


class TestRehearsal:
    def test_training_improves_every_held_out_session(self, result):
        """The one assertion that matters: the optimizer never saw these sessions, and the
        fitted parameters still beat the baseline on them. A pipeline that only improved its
        training partition would pass every unit test in the repo and still be useless."""
        held_out = result.held_out()
        assert held_out, "the split must actually hold something out"
        for score in held_out:
            assert score.fitted < score.baseline, (
                f"session {score.session_id} got worse: {score.baseline} -> {score.fitted}")

    def test_training_actually_moved_the_parameters(self, result):
        """Guards the vacuous pass: if theta stayed at zero every scale would be exactly 1.0 and
        the losses above would be identical rather than improved."""
        assert not np.allclose(np.asarray(result.theta), 0.0)
        assert any(abs(scale - 1.0) > 0.01 for scale in result.scales.values())

    def test_every_channel_is_present_and_within_its_declared_bounds(self, result):
        expected = {f"imu_gyro:{n}" for n in ("base", "mid", "tip")} | {
            "base_gyro_q", "base_accel_q", "contact_q", "contact_fk_r",
            "gravity_roll_r", "gravity_pitch_r"}
        assert set(result.scales) == expected
        for name, scale in result.scales.items():
            assert 0.01 <= scale <= 100.0, f"{name} escaped the bound: {scale}"

    def test_the_run_exports_an_artifact_the_strict_reader_accepts(self, result):
        """Completes the chain: a fit that cannot be serialized and re-read is not deployable."""
        reloaded = load_artifact(result.artifact_path)
        assert reloaded["arm"] == 7
        assert reloaded["scales"] == pytest.approx(result.scales)
        assert reloaded["provenance"]["objective"] == "body_velocity_l2"

    def test_the_artifact_records_which_sessions_were_trained_on(self, result):
        """Provenance is what makes a number auditable later; overlapping splits are rejected."""
        provenance = json.loads(result.artifact_path.read_text())["provenance"]
        train, test = set(provenance["train_sessions"]), set(provenance["test_sessions"])
        assert train and test and not (train & test)
        assert {s.session_id for s in result.held_out()} == test

    def test_the_report_is_honest_about_what_generated_data_can_show(self, result):
        text = result.report()
        assert "held-out sessions improved" in text
        assert "not a robot" in text, "the report must not read as a claim about real performance"


def test_the_split_selects_the_data_not_just_a_label(tmp_path):
    """A manifest whose split changes must change which trajectories are trained on. If the
    session id did not seed the data, the split would be decorative and held-out would be a lie."""
    # Seeds 4 and 3 are verified (offline, over all 3-session splits) to hold out sessions '0' and
    # '1' respectively. Asserted rather than branched around: a conditional skip here would let a
    # future change to deterministic_split's hashing silently retire the one check this test is for.
    first = dry_run(tmp_path / "a", split_seed=4, **SMALL)
    second = dry_run(tmp_path / "b", split_seed=3, **SMALL)

    first_ids = {s.session_id for s in first.held_out()}
    second_ids = {s.session_id for s in second.held_out()}
    assert first_ids != second_ids, "fixture seeds must select different held-out sessions"
    assert not np.allclose(np.asarray(first.theta), np.asarray(second.theta)), (
        "training on a different partition must produce a different fit")
