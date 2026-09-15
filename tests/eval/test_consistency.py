import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.eval.consistency import (
    ChannelConsistency, anis_band, channel_consistency, chi2_cdf, chi2_quantile,
    format_report, nis_consistency_loss, two_stage_consistency,
)


# Published chi-squared critical values -- a textbook table, used as the oracle rather than
# another library, so this pins the implementation against the literature and not against
# whatever SciPy happens to be installed (SciPy is not a declared dependency here).
CRITICAL_95 = {1: 3.841, 2: 5.991, 3: 7.815, 5: 11.070, 10: 18.307, 20: 31.410}
CRITICAL_05 = {1: 0.0039, 2: 0.1026, 5: 1.145, 10: 3.940, 20: 10.851}


class TestChiSquared:
    @pytest.mark.parametrize("dof,expected", sorted(CRITICAL_95.items()))
    def test_upper_critical_values_match_the_published_table(self, dof, expected):
        assert chi2_quantile(0.95, dof) == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize("dof,expected", sorted(CRITICAL_05.items()))
    def test_lower_critical_values_match_the_published_table(self, dof, expected):
        assert chi2_quantile(0.05, dof) == pytest.approx(expected, abs=5e-4)

    def test_quantile_inverts_the_cdf(self):
        for dof in (1.0, 3.0, 12.0, 250.0):
            for p in (0.01, 0.25, 0.5, 0.9, 0.999):
                assert float(chi2_cdf(chi2_quantile(p, dof), dof)) == pytest.approx(p, abs=1e-6)

    def test_rejects_degenerate_arguments(self):
        with pytest.raises(ValueError):
            chi2_quantile(0.0, 3)
        with pytest.raises(ValueError):
            chi2_quantile(1.0, 3)
        with pytest.raises(ValueError):
            chi2_quantile(0.5, 0)


class TestBand:
    def test_the_band_brackets_the_expected_value_and_tightens_with_more_samples(self):
        dof = 3.0
        narrow = anis_band(dof, samples=10_000)
        wide = anis_band(dof, samples=10)
        for lower, upper in (narrow, wide):
            assert lower < dof < upper, "the band must contain the expectation it is testing against"
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]), "more samples must tighten the band"

    def test_a_single_sample_band_is_just_the_chi_squared_interval(self):
        lower, upper = anis_band(4.0, samples=1, alpha=0.05)
        assert lower == pytest.approx(chi2_quantile(0.025, 4.0))
        assert upper == pytest.approx(chi2_quantile(0.975, 4.0))

    def test_rejects_an_empty_sample_count(self):
        with pytest.raises(ValueError):
            anis_band(3.0, samples=0)


class TestChannelConsistency:
    """Statistical validation: genuine chi-squared draws must pass, perturbed ones must fail
    in the correct DIRECTION. A harness that reports 'consistent' for everything, or that
    confuses overconfident with conservative, passes neither of these."""

    @staticmethod
    def _draws(dof, n, seed=0, factor=1.0):
        return np.random.default_rng(seed).chisquare(dof, size=n) * factor

    def test_true_chi_squared_innovations_are_reported_consistent(self):
        dof, n = 3.0, 20_000
        report = channel_consistency("contact", self._draws(dof, n), np.ones(n), dof)
        assert report.verdict == "consistent", report.describe()
        assert report.samples == n
        assert report.ratio == pytest.approx(1.0, abs=0.05)

    def test_innovations_larger_than_predicted_read_as_overconfident(self):
        dof, n = 3.0, 20_000
        report = channel_consistency("contact", self._draws(dof, n, factor=4.0), np.ones(n), dof)
        assert report.verdict == "overconfident", report.describe()
        assert report.ratio > 1.0

    def test_innovations_smaller_than_predicted_read_as_conservative(self):
        dof, n = 3.0, 20_000
        report = channel_consistency("contact", self._draws(dof, n, factor=0.25), np.ones(n), dof)
        assert report.verdict == "conservative", report.describe()
        assert report.ratio < 1.0

    def test_gated_and_pre_first_update_samples_are_excluded_not_averaged_in(self):
        """A gated tick's NIS is stale or NaN. Averaging it in would move the verdict; both
        must be dropped, and the reported sample count must say so."""
        dof = 3.0
        good = self._draws(dof, 5_000, seed=7)
        nis = np.concatenate([np.full(10, np.nan), good, np.full(500, 1.0e6)])
        applied = np.concatenate([np.zeros(10), np.ones(good.size), np.zeros(500)])

        report = channel_consistency("contact", nis, applied, dof)
        assert report.samples == good.size, "only applied, finite samples may enter the average"
        assert report.anis == pytest.approx(good.mean(), rel=1e-12)
        assert report.verdict == "consistent"

    def test_an_applied_but_nonfinite_sample_is_still_dropped(self):
        """Applied and NaN together would make the mean NaN and silently destroy the verdict."""
        dof = 3.0
        good = self._draws(dof, 1_000, seed=3)
        nis = np.concatenate([good, [np.nan]])
        applied = np.ones(nis.size)
        report = channel_consistency("contact", nis, applied, dof)
        assert report.samples == good.size
        assert np.isfinite(report.anis)

    def test_a_channel_that_never_applied_reports_no_data_rather_than_nan_verdict(self):
        report = channel_consistency("gravity", np.full(100, np.nan), np.zeros(100), 3.0)
        assert report.verdict == "no-data"
        assert report.samples == 0
        assert np.isnan(report.anis)

    def test_effective_samples_widens_the_band_for_correlated_ticks(self):
        """A 1 kHz log's consecutive ticks are not independent; the raw count makes the band
        far too tight. Declaring fewer effective samples must loosen it, not the average."""
        dof, n = 3.0, 20_000
        nis = self._draws(dof, n, factor=1.06)
        optimistic = channel_consistency("contact", nis, np.ones(n), dof)
        honest = channel_consistency("contact", nis, np.ones(n), dof, effective_samples=200)

        assert honest.anis == pytest.approx(optimistic.anis), "only the band moves, never the statistic"
        assert (honest.upper - honest.lower) > (optimistic.upper - optimistic.lower)
        assert optimistic.verdict == "overconfident" and honest.verdict == "consistent"

    def test_applied_mask_is_treated_as_nonzero_not_exactly_one(self):
        """The mask is a float that survives jit; a smoothed or weighted value must still count."""
        dof = 3.0
        values = self._draws(dof, 1_000, seed=11)
        report = channel_consistency("contact", values, np.full(values.size, 0.5), dof)
        assert report.samples == values.size

    def test_rejects_mismatched_shapes_and_bad_dof(self):
        with pytest.raises(ValueError, match="matching shapes"):
            channel_consistency("contact", np.ones(5), np.ones(4), 3.0)
        with pytest.raises(ValueError, match="dof"):
            channel_consistency("contact", np.ones(5), np.ones(5), 0.0)


class TestReportFormatting:
    def test_the_report_states_the_direction_so_a_reader_knows_which_way_to_tune(self):
        report = channel_consistency("contact", np.full(100, 12.0), np.ones(100), 3.0)
        text = format_report([report])
        assert "overconfident" in text
        assert "ratio" in text and "Q/R" in text

    def test_a_no_data_channel_says_so_instead_of_printing_nan(self):
        text = ChannelConsistency("gravity", 3.0, 0, float("nan"), float("nan"), float("nan"), "no-data").describe()
        assert "no applied updates" in text
        assert "nan" not in text.lower()


class TestConsistencyLoss:
    def test_the_loss_is_zero_exactly_at_the_expected_nis(self):
        assert float(nis_consistency_loss(jnp.full(50, 3.0), jnp.ones(50), 3.0)) == pytest.approx(0.0, abs=1e-30)

    def test_the_loss_is_symmetric_in_the_ratio_and_normalized_by_dof(self):
        """Normalizing by dof before squaring is what lets channels of different measurement
        dimension be summed without the widest one dominating the total."""
        double = float(nis_consistency_loss(jnp.full(10, 6.0), jnp.ones(10), 3.0))
        half = float(nis_consistency_loss(jnp.full(10, 1.5), jnp.ones(10), 3.0))
        assert double == pytest.approx(1.0)
        assert half == pytest.approx(0.25)
        wide = float(nis_consistency_loss(jnp.full(10, 24.0), jnp.ones(10), 12.0))
        assert wide == pytest.approx(double), "a 2x ratio must cost the same at any dof"

    def test_a_gated_nan_sample_does_not_poison_the_loss(self):
        """0.0 * NaN is NaN -- masking by multiplication alone would make the whole loss, and
        every gradient through it, NaN the first time an update is gated."""
        nis = jnp.array([3.0, jnp.nan, 3.0])
        applied = jnp.array([1.0, 0.0, 1.0])
        assert float(nis_consistency_loss(nis, applied, 3.0)) == pytest.approx(0.0, abs=1e-30)

    def test_an_applied_nan_is_also_survived(self):
        nis = jnp.array([3.0, jnp.nan, 3.0])
        assert np.isfinite(float(nis_consistency_loss(nis, jnp.ones(3), 3.0)))

    def test_no_applied_samples_costs_nothing_rather_than_dividing_by_zero(self):
        assert float(nis_consistency_loss(jnp.full(5, jnp.nan), jnp.zeros(5), 3.0)) == 0.0

    def test_it_is_differentiable_and_the_gradient_points_toward_consistency(self):
        """The whole reason this exists in JAX: it has to survive jit and BPTT as a regularizer."""
        def loss(scale):
            return nis_consistency_loss(jnp.full(64, 3.0) * scale, jnp.ones(64), 3.0)

        grad = jax.jit(jax.grad(loss))
        assert np.isfinite(float(grad(2.0)))
        assert float(grad(2.0)) > 0.0, "above the expectation, shrinking the innovations must reduce the loss"
        assert float(grad(0.5)) < 0.0, "below it, the gradient must point the other way"
        assert float(grad(1.0)) == pytest.approx(0.0, abs=1e-12)


class TestTwoStageIntegration:
    """`two_stage_consistency` reads the diagnostics the filters actually emit. Its assumptions
    about their shapes (NaN-padded per-row stacks, float applied masks) are structural claims
    about another module, so they are pinned against a real rollout rather than a mock."""

    @staticmethod
    def _rollout(ticks=40):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from learning.test_pipeline_integration import _synthetic_session_data
        from invariant_estimation.learning.noise import NoiseSpec
        from invariant_estimation.learning.two_stage import run

        build, params, ekf, kin, carry, inputs = _synthetic_session_data(5, ticks=ticks)
        spec = NoiseSpec(tuple(build.imu_names), arm=7)
        _, out = run(spec.initial_theta(), spec, build, params, ekf, kin, jnp.eye(3), carry, inputs)
        return build, out

    def test_every_channel_a_real_rollout_publishes_is_scored(self):
        build, out = self._rollout()
        reports = two_stage_consistency(out.joint_diagnostics, out.base,
                                        n_contacts=1, n_joints=build.n_joints)
        assert {r.channel for r in reports} == {"encoder", "contact", "gravity", "stacked"}
        for report in reports:
            assert report.verdict != "no-data", f"{report.channel} produced no applied updates"
            assert np.isfinite(report.anis) and report.dof > 0
            assert report.lower < report.dof < report.upper

    def test_the_stacked_channels_dof_is_read_from_the_diagnostic_not_assumed(self):
        """Its row count depends on how many anchors were active, so a hardcoded dof would be
        wrong the moment the anchor configuration changes."""
        build, out = self._rollout()
        stacked = next(r for r in two_stage_consistency(out.joint_diagnostics, out.base,
                                                        n_contacts=1, n_joints=build.n_joints)
                       if r.channel == "stacked")
        per_row = np.asarray(out.joint_diagnostics.stacked_nis_per_row)
        assert stacked.dof == float(np.isfinite(per_row).sum(axis=-1)[0])

    def test_the_contact_channel_uses_three_rows_per_contact(self):
        build, out = self._rollout()
        contact = next(r for r in two_stage_consistency(out.joint_diagnostics, out.base,
                                                        n_contacts=2, n_joints=build.n_joints)
                       if r.channel == "contact")
        assert contact.dof == 6.0
