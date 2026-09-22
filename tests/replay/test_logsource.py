

import pytest  # noqa: E402


class TestTheTwoTimeAxes:
    """`start`/`end` slice by tick index; `time` comes from the timestamps. They disagree on
    every real Alex log, and a caller who does not know that will select the wrong window --
    silently, because both numbers are plausible seconds."""

    def window(self, dt=0.001, cadence=0.001, ticks=(100, 101, 102, 103)):
        import numpy as np
        from pathlib import Path
        from invariant_estimation.replay.logsource import LogWindow
        tick = np.asarray(ticks, dtype=np.int64)
        time = (tick - tick[0]).astype(float) * cadence
        return LogWindow(log_dir=Path("/nonexistent"), time=time, tick=tick,
                         channels={"x": np.zeros(len(tick))}, dt=dt)

    def test_nominal_time_is_tick_times_dt_from_the_start_of_the_log(self):
        import numpy as np
        w = self.window(ticks=(100, 101, 102))
        np.testing.assert_allclose(w.nominal_time, [0.100, 0.101, 0.102], atol=1e-12)
        # ... and `time` is measured from the WINDOW, so it starts at zero. Different origins,
        # which is the other half of why the two cannot be used interchangeably.
        np.testing.assert_allclose(w.time, [0.0, 0.001, 0.002], atol=1e-12)

    def test_measured_dt_reports_the_cadence_the_timestamps_actually_show(self):
        w = self.window(dt=0.001, cadence=0.000859)      # the 2026-07 Alex log
        assert w.measured_dt == pytest.approx(0.000859, rel=1e-9)
        assert w.dt == 0.001, "the declared period is left alone; the two are reported separately"

    def test_measured_dt_is_per_tick_regardless_of_stride(self):
        """A strided window must still report a comparable cadence, or the check that matters --
        measured against declared -- silently scales with how the window was sampled."""
        w = self.window(dt=0.001, cadence=0.000859, ticks=(100, 110, 120, 130))
        # time steps are 10 ticks apart; dividing by the tick span recovers the per-tick value
        assert w.measured_dt == pytest.approx(0.000859, rel=1e-9)

    def test_a_single_sample_window_reports_nan_rather_than_a_made_up_cadence(self):
        import math
        assert math.isnan(self.window(ticks=(100,)).measured_dt)

    def test_the_two_axes_diverge_on_a_real_cadence_mismatch(self):
        """The failure this guards: asking for `start=110.0` gets tick 110000, whose wall clock
        is ~94.5 s on a log running 14% fast. Both are 'seconds'; only one is what was meant."""
        w = self.window(dt=0.001, cadence=0.000859, ticks=(110000, 110001))
        assert w.nominal_time[0] == pytest.approx(110.0)
        # the same tick's wall-clock offset into the log is not 110 s, and the ratio says so
        assert w.measured_dt / w.dt == pytest.approx(0.859, rel=1e-6)
