r"""Port of `JointLevelKFDirectVelocityMeasurementTest.java` (5 tests) — gate G8 —
plus two port-specific tests the Java scenarios structurally cannot see.

The channel under test (`jointKF/velocity.py`) fuses the drive's own published
joint velocity.  Its whole content is the noise model: the published signal is
the output of a firmware first-order low-pass, so it is *lagged*, and for a
first-order filter the identity

    u - y = y_dot / omega_c            (EXACT)

says the lag error is the published signal's own slope over the corner frequency.
Hence `R_ii(t) = sigma_i^2 + (dhat_i / omega_eff,i)^2` with `dhat` an estimate of
that slope: a floor when the joint holds a constant rate, inflated during a
transient.

Two things the Java class does not constrain
--------------------------------------------
1. **The 5 Hz smoothing of `dhat`.**  `lagInflationTracksMeasuredSlewExactly`
   drives a noiseless constant / ramp / constant signal.  On a *noiseless* input
   the raw finite difference and the smoothed one both converge to `slope` and
   both are exactly zero on a constant — so deleting the smoother passes all
   three phases.  What the smoother exists for is the noise: raw, the finite
   difference has variance `2 sigma^2/dt^2`, which at 1 kHz inflates `R` by ~2
   orders of magnitude at quiet standing, the regime the channel is worth having
   in.  `test_smoothing_keeps_quiet_standing_R_at_the_floor` is the constraining
   test; the ramp is the accuracy test.
2. **First-tick priming.**  Java's fixture starts from `z = 0` so it never
   differences a first sample against an unset history.  The port carries a
   `primed` float for it (I7), tested below.

Deviations: RNG (per the map — only sample means are asserted); YoVariable
lookups become `ChannelDiagnostics` fields; the label-string dispatch becomes
separate struct fields (see `velocity.py`).  Trial counts, envelopes, tolerances,
`cornerHz`, `slope`, and the phase lengths are verbatim.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.jointKF.diagnostics import per_joint_nis
from invariant_estimation.jointKF.state import JointKFState, default_params
from invariant_estimation.jointKF.update import joseph_update
from invariant_estimation.jointKF.velocity import (
    advance_slew,
    build_velocity_channel,
    init_velocity_carry,
    no_diagnostics,
    velocity_jacobian,
    velocity_noise,
    velocity_update,
    velocity_var_for_name,
)

from ._oracles import SHAPES, assert_all_close, stub_build, vel_sigma_for

#: The channel is OFF by default (`config/filter_cfg.yaml`); every test here
#: enables it explicitly, which is also the map's `useDirectVelocityMeasurement`.
PARAMS = default_params(direct_velocity_enabled=True)

#: Java `SIGMA_QD_FALLBACK` — 0.1 rad/s, i.e. 0.01 (rad/s)^2.
SIGMA_QD_FALLBACK = 0.1

DT = 1.0e-3
SHAPE = SHAPES[0]                       # NUM_CHAIN_JOINTS = 10 => n = 8, m = 2
TRIALS = 4000
ENVELOPE = 4.0 * np.sqrt(2.0 / TRIALS)
SEED = 47_001

#: The config section the channel builder reads. Supplied explicitly so the test
#: is independent of whatever the (currently empty) `encoder_vel_std` sidecar
#: holds — Java passes `velSigmaFor` as a function argument for the same reason.
BASE_CFG = {"sigma_qd_unfiltered": SIGMA_QD_FALLBACK, "encoder_vel_std": {}}


def _build():
    return stub_build(SHAPE)


def _channel(vel_std=None, corner_hz=None, build=None):
    """Fixture `singlePairWithDirectVelocity(SEED, 10, 1, 9, ..., velSigmaFor, cornerFn)`.

    `corner_hz=None` is Java's `cornerFn = null`: static `R`, no lag inflation.
    """
    build = build if build is not None else _build()
    if vel_std is None:
        vel_std = {nm: vel_sigma_for(nm) for nm in build.joint_names}
    return build_velocity_channel(build, PARAMS, vel_std=vel_std,
                                  corner_hz=corner_hz, cfg=BASE_CFG)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_velocity_measurement_model_is_wired():
    """`velocityMeasurementModelIsWired`: `H = [0 | I | 0]` exactly, `R` per joint."""
    build = _build()
    channel = _channel(build=build)
    n, dim = build.n_joints, build.dim

    H = np.asarray(velocity_jacobian(build))
    expected_H = np.zeros((n, dim))
    expected_H[:, n:2 * n] = np.eye(n)
    # Tolerance 0.0: these are structural ones and zeros, not a computation.
    assert np.array_equal(H, expected_H), "H_qd must be exactly [0 | I_n | 0]"

    R = np.asarray(velocity_noise(channel))
    expected = np.array([vel_sigma_for(nm) ** 2 for nm in build.joint_names])
    assert_all_close(np.diag(R), expected, 1.0e-15, "R_qd diagonal")
    assert np.all(R - np.diag(np.diag(R)) == 0.0), "R_qd must be strictly diagonal"
    assert len(np.unique(np.diag(R))) > 1, "per-joint means per-joint (I9)"

    # Java reads `jointKF_qdR_<name>`; the port publishes the same numbers as a
    # diagnostics field, so assert the published copy agrees with `R` itself.
    state = JointKFState(x=jnp.zeros(dim), P=jnp.eye(dim))
    _, _, diag, _ = velocity_update(state, jnp.zeros(n), build, PARAMS,
                                    channel, init_velocity_carry(build))
    assert_all_close(diag.velocity_r_diag, expected, 1.0e-15, "published qdR")


def test_unmatched_joint_falls_back_to_sigma_qd():
    """`unmatchedJointFallsBackToSigmaQdUnfiltered`: 0.1 rad/s => 0.01, tol 1e-15.

    Java injects the unmatched joint by mapping its name to `NaN`; the port reads
    a non-finite STD as absent (same convention as `encoder_var_for_name`).
    """
    build = _build()
    names = build.joint_names
    unmatched = names[0]
    vel_std = {nm: (np.nan if nm == unmatched else vel_sigma_for(nm)) for nm in names}

    resolved = [velocity_var_for_name(nm, BASE_CFG, vel_std) for nm in names]
    wired = [w for _, w in resolved]
    assert wired[0] is False and all(wired[1:]), "only the unmatched joint may be unwired"

    R = np.asarray(velocity_noise(_channel(vel_std=vel_std, build=build)))
    expected = np.array([SIGMA_QD_FALLBACK ** 2 if nm == unmatched else vel_sigma_for(nm) ** 2
                         for nm in names])
    assert_all_close(np.diag(R), expected, 1.0e-15, "R_qd with one unmatched joint")


# ---------------------------------------------------------------------------
# Lag inflation
# ---------------------------------------------------------------------------

def _run_slew(channel, build, z_sequence):
    """Scan `advance_slew` over a measurement sequence, returning every `R` diagonal.

    Uses `lax.scan` rather than a Python loop deliberately: the carry is the
    object under test, and scanning it is the same code path the filter step will
    use — a Python loop would silently tolerate a carry that is not a pytree.
    """
    def body(carry, z):
        carry = advance_slew(channel, carry, z)
        return carry, jnp.diag(velocity_noise(channel, carry))

    return jax.lax.scan(body, init_velocity_carry(build), jnp.asarray(z_sequence))


def test_lag_inflation_tracks_measured_slew_exactly():
    """`lagInflationTracksMeasuredSlewExactly` — the deterministic three-phase scenario.

    Phase 1 (100 ticks, `z = 0`)      : `R = sigma^2`, tol 1e-12 (the floor).
    Phase 2 (300 ticks, ramp slope 2) : `R = sigma^2 + (slope*invOmega)^2`, rel 1e-3.
    Phase 3 (2000 ticks, `z` frozen)  : `R < 1.01*sigma^2` (decays, never latches).
    """
    build = _build()
    corner_hz = 10.0
    inv_omega = 1.0 / (2.0 * np.pi * corner_hz)
    slope = 2.0
    channel = _channel(corner_hz=corner_hz, build=build)
    n = build.n_joints
    sigma2 = np.array([vel_sigma_for(nm) ** 2 for nm in build.joint_names])

    quiet = np.zeros((100, n))
    ramp = np.repeat((slope * DT * np.arange(1, 301))[:, None], n, axis=1)
    frozen = np.repeat(ramp[-1][None, :], 2000, axis=0)

    _, R_diag = _run_slew(channel, build, np.concatenate([quiet, ramp, frozen]))
    R_diag = np.asarray(R_diag)

    # Phase 1 — the floor. Exactly the floor, in fact: a constant measurement has
    # zero slope, so the lag term is identically zero, not merely small.
    assert_all_close(R_diag[99], sigma2, 1.0e-12, "phase 1 (quiet) R")

    # Phase 2 — the lag term is the ramp's slope over the corner. 300 ticks of a
    # 5 Hz smoother leaves alpha^300 = 8e-5 of the initial transient.
    expected = sigma2 + (slope * inv_omega) ** 2
    rel = np.abs(R_diag[399] - expected) / expected
    assert np.all(rel < 1.0e-3), f"phase 2 (swing) R off by {rel.max():.2e} relative"

    # Stated as the inflation itself, not just the total: for the noisiest joint
    # the floor is 1.2e-3 and the lag term 1.0e-3, so a test on the total alone
    # would still pass with the inflation halved.
    inflation = R_diag[399] - sigma2
    assert np.all(np.abs(inflation - (slope * inv_omega) ** 2) < 2.0e-3 * (slope * inv_omega) ** 2)

    # Phase 3 — no latch-up. The Java bound is 1.01x; the port's decay is
    # alpha^2000 ~ 1e-27, so this is comfortable by 25 orders and would only fail
    # if the inflation were held rather than tracked.
    assert np.all(R_diag[-1] < sigma2 * 1.01), f"inflation latched: {R_diag[-1] / sigma2}"


def test_smoothing_keeps_quiet_standing_R_at_the_floor():
    """Port-specific: the 5 Hz smoother, which the noiseless scenario cannot see.

    Quiet standing with a *noisy* measurement (`z ~ N(0, sigma^2)`, no real slew).
    The raw finite difference then has variance `2 sigma^2/dt^2`, so an unsmoothed
    `dhat` inflates `R` by `~2 sigma^2 invOmega^2/dt^2` — a factor of
    `2 (invOmega/dt)^2 = 5e2` here.  The smoothed one stays within a few percent
    of the floor.

    This is the test that would fail if `advance_slew` stopped smoothing; the
    deterministic ramp above would not.
    """
    build = _build()
    corner_hz = 10.0
    inv_omega = 1.0 / (2.0 * np.pi * corner_hz)
    channel = _channel(corner_hz=corner_hz, build=build)
    n = build.n_joints
    sigma = np.array([vel_sigma_for(nm) for nm in build.joint_names])

    rng = np.random.default_rng(SEED + 29)
    z = rng.standard_normal((3000, n)) * sigma

    _, R_diag = _run_slew(channel, build, z)
    R_diag = np.asarray(R_diag)

    settled = R_diag[500:].mean(axis=0)

    # Analytic expectation. The cascade (finite difference) -> (one-pole, alpha)
    # has impulse response `(1-a)` at k=0 and `(1-a)(a^k - a^(k-1))` after, so
    #     var(dhat) = (sigma^2/dt^2) (1-a)^2 [1 + (1-1/a)^2 a^2/(1-a^2)]
    # which at a = exp(-2*pi*5*dt) is 9.7e-4 sigma^2/dt^2. Through invOmega^2
    # that is 0.25 sigma^2 of inflation — a 25% overstatement of R at standing,
    # tolerable. UNSMOOTHED it would be 2 (invOmega/dt)^2 = 5.1e2, i.e. R wrong
    # by a factor of 500. The bound below sits between those two by 300x.
    a = np.exp(-2.0 * np.pi * default_params().lag_slew_smoothing_hz * DT)
    predicted = (1.0 - a) ** 2 * (1.0 + (1.0 - 1.0 / a) ** 2 * a ** 2 / (1.0 - a ** 2))
    predicted = 1.0 + predicted * (inv_omega / DT) ** 2
    assert predicted == pytest.approx(1.246, abs=0.01)

    ratio = settled / sigma ** 2
    assert np.all(ratio < 1.5), f"quiet-standing R inflated: {ratio}"
    assert np.all(ratio > 1.05), "sanity: the smoothed slew is not identically zero"

    raw_ratio = 1.0 + 2.0 * (inv_omega / DT) ** 2
    assert raw_ratio > 300.0 * 1.5, "sanity: the unsmoothed variant is far outside the bound"


def test_first_tick_does_not_manufacture_a_slew():
    """Port-specific: the `primed` carry (I7).

    The filter can be enabled mid-run, at a non-zero standing velocity.  Without
    priming, tick 0 differences against `z_prev = 0` and reports `z/dt` — at 1 kHz,
    a 0.1 rad/s velocity becomes a phantom 100 rad/s^2 slew that the 5 Hz smoother
    then needs ~30 ms to forget, inflating `R` through the whole of it.
    """
    build = _build()
    channel = _channel(corner_hz=10.0, build=build)
    n = build.n_joints
    sigma2 = np.array([vel_sigma_for(nm) ** 2 for nm in build.joint_names])

    z0 = jnp.full(n, 0.1)
    carry = advance_slew(channel, init_velocity_carry(build), z0)
    assert np.all(np.asarray(carry.dhat) == 0.0), "first sample must not be differenced"
    assert_all_close(jnp.diag(velocity_noise(channel, carry)), sigma2, 1.0e-15, "tick-0 R")
    assert float(carry.primed) == 1.0


# ---------------------------------------------------------------------------
# NIS
# ---------------------------------------------------------------------------

def _run_nis_trials(noise_scale: float, trials: int = TRIALS, seed: int = SEED + 13):
    """Java `runNISTrials(noiseScale, trials)` — `cornerFn = null`, so `R` is static.

    Prior: `x = 0`; `P` diagonal with `1e-4` on positions, `0.25*R_ii` on
    velocities, `1e-6` on bias (verbatim).  Truth is drawn from the prior's
    velocity block, so `nu_i = eps_i + e_i ~ N(0, S_ii)` exactly as on the encoder
    channel — one property, two channels.
    """
    build = _build()
    channel = _channel(build=build)
    n, dim = build.n_joints, build.dim

    R = np.asarray(velocity_noise(channel))
    r = np.diag(R)
    x = np.zeros(dim)
    P = np.diag(np.concatenate([np.full(n, 1.0e-4), 0.25 * r,
                                np.full(dim - 2 * n, 1.0e-6)]))
    prior = JointKFState(x=jnp.asarray(x), P=jnp.asarray(P))

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((trials, n)) * np.sqrt(0.25 * r)
    err = rng.standard_normal((trials, n)) * noise_scale * np.sqrt(r)
    z = jnp.asarray(x[n:2 * n] + eps + err)

    carry = init_velocity_carry(build)

    def one_trial(z_t):
        _, _, diag, info = velocity_update(prior, z_t, build, PARAMS, channel, carry)
        return diag, info

    diags, infos = jax.vmap(one_trial)(z)
    return np.asarray(diags.velocity_nis), np.asarray(z), diags, infos, prior, build


def _assert_trial_invariants(nis, z, diags, infos, prior, n):
    assert np.all(np.isfinite(nis)), "per-trial NIS must be finite"
    assert np.all(nis >= 0.0)
    assert np.all(np.asarray(infos.was_applied) == 1.0), "no velocity update may be gated here"
    assert_all_close(np.asarray(diags.velocity_innovation)[0],
                     z[0] - np.asarray(prior.x)[n:2 * n], 1.0e-15,
                     "trial 0 signed innovation")
    # Cross-talk guard (Java's label dispatch): the position channel's statistic
    # must still be NaN — the velocity channel may not publish into it.
    assert np.all(np.isnan(np.asarray(diags.encoder_nis))), "velocity channel wrote encoder NIS"
    assert np.all(np.isnan(np.asarray(diags.encoder_innovation)))


def test_direct_velocity_nis_is_chi_square_consistent_at_the_wired_noise():
    """`directVelocityNISIsChiSquareConsistentAtTheWiredNoise`: |mean - 1| < 0.089."""
    nis, z, diags, infos, prior, build = _run_nis_trials(1.0)
    _assert_trial_invariants(nis, z, diags, infos, prior, build.n_joints)

    mean = nis.mean(axis=0)
    worst = np.max(np.abs(mean - 1.0))
    assert worst < ENVELOPE, f"per-joint mean NIS off by {worst:.4f} > {ENVELOPE:.4f}: {mean}"


def test_direct_velocity_nis_catches_understated_R():
    """`directVelocityNISCatchesUnderstatedR`: noise at 3x sigma => mean NIS > 4 (7.4)."""
    nis, z, diags, infos, prior, build = _run_nis_trials(3.0)
    _assert_trial_invariants(nis, z, diags, infos, prior, build.n_joints)

    mean = nis.mean(axis=0)
    assert np.all(mean > 4.0), f"understated R not detected: {mean}"
    assert_all_close(mean, np.full_like(mean, 7.4), 7.4 * ENVELOPE, "mean NIS at 3x noise")


def test_a_gated_velocity_update_reports_no_statistic():
    """Port-specific: the NaN convention survives the shared gate.

    A non-finite measurement is skipped by `joseph_update`'s finite mask; the
    per-joint NIS must then be NaN rather than 0, for the same reason
    `UpdateInfo.nis` is — a skipped update has no consistency statistic, and zero
    reads as "perfectly consistent" to anything downstream.
    """
    build = _build()
    channel = _channel(build=build)
    n, dim = build.n_joints, build.dim
    state = JointKFState(x=jnp.zeros(dim), P=jnp.eye(dim))

    z = jnp.zeros(n).at[2].set(jnp.nan)
    post, _, diag, info = velocity_update(state, z, build, PARAMS, channel,
                                          init_velocity_carry(build))
    assert float(info.was_applied) == 0.0
    assert np.all(np.isnan(np.asarray(diag.velocity_nis)))
    assert_all_close(post.x, state.x, 0.0, "gated update must leave x bit-identical")
    assert_all_close(post.P, state.P, 0.0, "gated update must leave P bit-identical")


def test_diagnostics_start_as_nan_on_every_channel():
    """`no_diagnostics` is all-NaN: nothing measured is reported as nothing measured."""
    diag = no_diagnostics(_build())
    for field, value in diag._asdict().items():
        assert np.all(np.isnan(np.asarray(value))), f"{field} must initialise to NaN"


def test_channel_is_off_by_default():
    """The config default is OFF (`config/filter_cfg.yaml`) — a regression guard.

    Enabling this channel in sim would fuse a perfect derivative under a noise
    model calibrated for firmware lag, which is not a small error: it is an
    over-confident measurement of a quantity the filter is trying to estimate.
    """
    assert default_params().direct_velocity_enabled is False
    assert default_params().lag_slew_smoothing_hz == pytest.approx(5.0)
