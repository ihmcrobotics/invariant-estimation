"""Regression for FIX_CHECKLIST B1: `terrain.waves` must consume `seed`.

The pre-fix `waves` ignored `seed` and returned a byte-identical heightfield for
every seed, so a training waves rollout and the held-out waves rollout shared the
exact same terrain — a silent train/val leak. The checklist's own failing oracle
was `max|field(sᵢ) − field(s₀)| = 0.0` across seeds; these tests assert it is now
strictly positive, and that the fix did not steepen the (mild) relief band.
"""
import numpy as np
import pytest

from invariant_estimation.sim import terrain

SEEDS = list(range(6))


def test_waves_consumes_seed():
    f0 = terrain.waves(seed=SEEDS[0])
    diffs = [float(np.max(np.abs(terrain.waves(seed=s) - f0))) for s in SEEDS[1:]]
    # the exact oracle the checklist reported as 0.00000 pre-fix
    assert min(diffs) > 0.0, "waves(seed) still returns identical fields across seeds"


def test_waves_fields_pairwise_distinct():
    fields = [terrain.waves(seed=s) for s in SEEDS]
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            assert float(np.max(np.abs(fields[i] - fields[j]))) > 0.0


def test_waves_deterministic_per_seed():
    assert np.array_equal(terrain.waves(seed=3), terrain.waves(seed=3))


def test_waves_relief_band_preserved():
    """Fix keeps the tier mild: peak height within the pre-fix band + jitter, and
    non-degenerate (not flattened). Pre-fix peak was 0.10 m; ±20% amplitude jitter
    caps it at ~0.12 m."""
    for s in SEEDS:
        f = terrain.waves(seed=s)
        assert f.shape == (terrain.N, terrain.N)
        peak = float(f.max())
        assert 0.06 <= peak <= 0.13, f"seed {s}: peak {peak} outside the mild band"
        assert float(f.min()) >= 0.0


def test_flat_still_seed_independent():
    assert np.array_equal(terrain.flat(seed=0), terrain.flat(seed=7))
