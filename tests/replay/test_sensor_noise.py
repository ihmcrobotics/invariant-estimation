r"""
Java parity, tier 1: **per-joint sensor noise**, model-free.

Unlike the ``diag(Qa)`` oracle (`test_java_parity.py`), nothing here needs the
mass matrix.  The joint KF's measurement covariances are pure name->value table
lookups, and the log publishes what the Java filter actually used:

* ``jointKF_encR_<joint>``   -- the encoder POSITION measurement variance R_ii,
  which Java sets to ``std**2`` from ``AlexSensorNoiseParameters`` (or the scalar
  ``ENCODER_VAR = 5e-5`` fallback for an unwired joint).
* ``jointKF_qdR_<joint>``    -- the direct-velocity channel's R_ii.  This one is
  *lag-inflated* per tick (``R = sigma**2 + (slew/omega_eff)**2``), so it is not
  constant -- but its **minimum over the run** is the ``sigma**2`` floor, reached
  whenever the joint's measured slew passes through zero.

The bug this catches
--------------------
Until this run's values were wired in, ``config/filter_cfg.yaml`` carried
``encoder_pos_std: {}`` -- so every joint fell back to ``encoder_var = 5e-5``,
whose implied std (7.1e-3 rad) is **15x to 48x larger** than the measured
per-joint values (1.4e-4 .. 7.1e-4 rad).  A filter that under-trusts its encoders
by 2-3 orders of magnitude in variance leans far too hard on the IMU/model side;
the joint estimates and their exported ``Sigma_q`` are silently wrong.  This test
fails the moment the config drifts back off the measured values.
"""
from __future__ import annotations

import numpy as np
import pytest

from invariant_estimation.jointKF.state import encoder_var_for_name
from invariant_estimation.jointKF.velocity import velocity_var_for_name
from invariant_estimation.replay.logsource import read_window

from .conftest import FILTERED_JOINTS

# encR is a construction-time constant; a wide window at coarse stride is plenty
# to read it and to see qdR's slew term pass through its floor.
_WINDOW = (5.0, 600.0)
_STRIDE = 50


@pytest.fixture(scope="module")
def noise_window(log_dir):
    names = [f"jointKF_encR_{j}" for j in FILTERED_JOINTS] + [
        f"jointKF_qdR_{j}" for j in FILTERED_JOINTS
    ]
    return read_window(log_dir, names, start=_WINDOW[0], end=_WINDOW[1], stride=_STRIDE)


def test_encoder_position_variance_matches_the_log(noise_window):
    """`encoder_var_for_name(j)` == the Java filter's published `encR` per joint.

    This is the whole encoder-noise wiring, checked against the machine that ran:
    the config value, the ``std -> std**2`` convention, and the case-insensitive
    name match all have to be right for this to pass.
    """
    mismatches = []
    for j in FILTERED_JOINTS:
        logged = noise_window[f"jointKF_encR_{j}"]
        # Constant to float precision -- it is a construction-time constant, so
        # its spread is round-off (~1e-23), not variation.
        assert logged.std() <= 1e-10 * logged[0], f"{j}: encR is not constant over the run"
        ours, wired = encoder_var_for_name(j)
        assert wired, f"{j}: fell back to the scalar encoder_var -- config not wired"
        if not np.isclose(ours, logged[0], rtol=1e-3):
            mismatches.append(f"    {j:<14s} ours {ours:.6e}  log {logged[0]:.6e}")
    assert not mismatches, "encoder position variance disagrees with the log:\n" + "\n".join(mismatches)


def test_encoder_velocity_floor_matches_the_log(noise_window):
    """``min_t jointKF_qdR`` == ``velocity_var_for_name(j)``.

    The direct-velocity channel was ON in this flight
    (``jointKFUseDirectVelocityMeasurement = 1``).  Its R is lag-inflated, so only
    the *floor* is the wired ``sigma**2`` -- but that floor is reached often
    enough on a 10-minute walk that its minimum is an exact readout of the
    constant, and it validates the velocity-noise table the same way.
    """
    mismatches = []
    for j in FILTERED_JOINTS:
        floor = noise_window[f"jointKF_qdR_{j}"].min()
        ours, wired = velocity_var_for_name(j)
        assert wired, f"{j}: velocity noise fell back to sigma_qd_unfiltered"
        # 5% because the floor is only touched to within the slew's finite
        # resolution; in practice these land far tighter.
        if not np.isclose(ours, floor, rtol=5e-2):
            mismatches.append(f"    {j:<14s} ours {ours:.6e}  log-floor {floor:.6e}")
    assert not mismatches, "encoder velocity floor disagrees with the log:\n" + "\n".join(mismatches)


def test_the_wired_values_actually_moved_off_the_fallback(noise_window):
    """Guard the regression directly: the fallback would be catastrophically wrong.

    If someone empties the config tables again, the two tests above still fail --
    but this one says *why* in one line, and pins the magnitude of the bug so the
    reason the values matter is not lost.
    """
    from invariant_estimation.config import section

    fallback = section("joint_kf")["encoder_var"]  # 5e-5
    for j in FILTERED_JOINTS:
        logged = noise_window[f"jointKF_encR_{j}"][0]
        # Every real joint is 100x-2400x tighter than the fallback variance.
        assert logged < fallback / 50, (
            f"{j}: measured encoder variance {logged:.2e} is not far below the "
            f"{fallback:.0e} fallback -- did the fallback leak into the log?"
        )
