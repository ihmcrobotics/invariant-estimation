r"""Port of `JointLevelKFEncoderNISConsistencyTest.java` (4 tests) — gate G8.

The class checks two things that look unrelated and are not: that the **per-joint**
encoder variance reaches `R_enc` (invariant I9 — never one uniform number), and
that the resulting innovation statistic is chi-square consistent.  The second is
what makes the first meaningful: a wired-but-wrong variance produces a perfectly
well-formed filter whose only symptom is that its stated uncertainty is a lie, and
the NIS is the only quantity in the filter that can catch that.

The property
------------
Prior `(x, P)` with `P` diagonal, encoder `z_i = q_true,i + e_i`, and truth drawn
**from the prior** — `q_true,i = x_i + eps_i`, `eps_i ~ N(0, P_ii)`.  Then

    nu_i = z_i - x_i = eps_i + e_i ~ N(0, P_ii + R_ii) = N(0, S_ii)

so `NIS_i = nu_i^2 / S_ii ~ chi^2_1`, mean 1, variance 2.  Note this is a
statement about the *marginal* of each row; it holds whatever correlations `S`
carries off the diagonal, which is why it survives being applied to a channel the
filter treats jointly.

What this file does NOT prove
-----------------------------
Recorded here because `PORT_NOTES.md` already lists five tests in this project
that passed against wrong implementations, and this class is a sixth candidate:
at the map's fixed `P = R/4`, swapping the **prior** `S` for the **posterior** one
in the NIS denominator moves the mean from 1.000 to 1.042, against a 4-sigma
envelope of 0.089.  The chi-square test is structurally blind to it.  That is why
`test_S_is_the_prior_innovation_covariance` exists below — it constrains the same
trap deterministically, to 1e-15, instead of statistically at 4.7% of the
detection threshold.

Deviation from Java: RNG.  The map states that only the sample mean over 4000
trials is asserted, so Java's exact `Random(SEED+7)` draws are not required.
Trial count (4000), envelope (`4*sqrt(2/4000)`), `P/R` ratio, noise scales and
thresholds are verbatim.  The trial loop is `vmap`ped, not Python-looped (repo
convention).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.jointKF.diagnostics import per_joint_nis
from invariant_estimation.jointKF.measure import encoder_jacobian, encoder_noise
from invariant_estimation.jointKF.state import (
    JointKFState,
    default_params,
    encoder_var_for_name,
)
from invariant_estimation.jointKF.update import joseph_update

from ._oracles import SHAPES, assert_all_close, nis_quadratic_form, sigma_for, stub_build

PARAMS = default_params()

#: Java `ENCODER_VAR_FALLBACK` — the scalar an unmatched joint falls back to.
ENCODER_VAR_FALLBACK = 5.0e-5

#: `NUM_CHAIN_JOINTS = 10` in Java, i.e. shape `n8_m2` (n = 8, m = 2).
SHAPE = SHAPES[0]

TRIALS = 4000
#: 4-sigma on the sample mean of `TRIALS` draws from chi^2_1 (variance 2).
ENVELOPE = 4.0 * np.sqrt(2.0 / TRIALS)

SEED = 31_001


def _build(encoder_var=None):
    """Fixture `singlePairWithEncoderNoise(SEED, 10, 1, 9, sigmaFor)`.

    Pure linear algebra: the encoder channel is `H = [I | 0]` with a diagonal `R`,
    so no geometry enters and `stub_build` is the right fixture (an MJX chain
    would cost seconds of tracing to supply Jacobians nothing here reads).
    """
    if encoder_var is None:
        encoder_var = np.array([sigma_for(nm) ** 2 for nm in _names()])
    return stub_build(SHAPE, encoder_var=np.asarray(encoder_var, dtype=float))


def _names():
    return stub_build(SHAPE).joint_names


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_per_joint_encoder_variance_is_wired_into_R_enc():
    """`perJointEncoderVarianceIsWiredIntoREncAndPublished`, tol 1e-15.

    Java also asserts the YoVariable `jointKF_encR_<name>` carries the same
    number; the port has no YoVariables, and `R_enc` *is* the published object
    (`measure.encoder_noise` is the `getEncoderNoise` seam), so the two assertions
    collapse into one.
    """
    build = _build()
    R = np.asarray(encoder_noise(build))
    n = build.n_joints

    assert R.shape == (n, n)
    expected = np.array([sigma_for(nm) ** 2 for nm in build.joint_names])
    assert_all_close(np.diag(R), expected, 1.0e-15, "R_enc diagonal")

    # Off-diagonals EXACTLY zero, not merely small: encoder errors on separate
    # joints share no mechanism, and a leaked off-diagonal would quietly make the
    # per-joint NIS below a statement about a correlated channel.
    off = R - np.diag(np.diag(R))
    assert np.all(off == 0.0), f"{np.count_nonzero(off)} non-zero off-diagonal entries"

    # I9: the variances must actually differ across joints, or "per-joint" is
    # vacuous and a uniform scalar would pass everything above.
    assert len(np.unique(np.diag(R))) > 1


def test_unmatched_joint_falls_back_to_scalar_encoder_var():
    """`unmatchedJointFallsBackToScalarEncoderVar`, tol 1e-15.

    Drives `state.encoder_var_for_name` — the real build-time resolution — rather
    than hand-building the array, since the fallback (and the `wired=False` flag
    that makes `build.py` log it) is the thing under test.  Java injects the
    unmatched joint by mapping its name to `NaN`; the port treats a non-finite or
    non-positive STD as absent, which is the same injection.
    """
    names = _names()
    unmatched = names[0]
    cfg = {
        "encoder_var": ENCODER_VAR_FALLBACK,
        "encoder_pos_std": {nm: (np.nan if nm == unmatched else sigma_for(nm)) for nm in names},
    }

    resolved = [encoder_var_for_name(nm, cfg) for nm in names]
    var = np.array([v for v, _ in resolved])
    wired = [w for _, w in resolved]

    assert wired[0] is False and all(wired[1:]), "only the unmatched joint may be unwired"
    expected = np.array([ENCODER_VAR_FALLBACK if nm == unmatched else sigma_for(nm) ** 2
                         for nm in names])
    assert_all_close(var, expected, 1.0e-15, "fallback-resolved encoder variance")

    R = np.asarray(encoder_noise(_build(var)))
    assert_all_close(np.diag(R), expected, 1.0e-15, "R_enc with one unmatched joint")


# ---------------------------------------------------------------------------
# The NIS trials
# ---------------------------------------------------------------------------

def _run_nis_trials(noise_scale: float, trials: int = TRIALS, seed: int = SEED + 7):
    """Java `runNISTrials(noiseScale, trials)` — vmapped over trials.

    Prior: `x = 0`; `P` diagonal with `P_ii = 0.25*R_ii` on positions, `1e-2` on
    velocities, `1e-6` on bias (verbatim from the map).  Each trial draws truth
    from the prior and noise at `noise_scale` times the *wired* sigma, so
    `noise_scale = 3` is the "R understated by 9x" scenario.

    Returns `(mean_nis, nis, nu, info)` — the per-trial arrays come back too
    because the map asserts inside the loop (finiteness, and trial 0's signed
    innovation).
    """
    build = _build()
    n, dim = build.n_joints, build.dim

    H = encoder_jacobian(build)
    R = encoder_noise(build)
    r = np.asarray(jnp.diag(R))

    x = np.zeros(dim)
    P = np.diag(np.concatenate([0.25 * r,
                                np.full(n, 1.0e-2),
                                np.full(dim - 2 * n, 1.0e-6)]))
    prior = JointKFState(x=jnp.asarray(x), P=jnp.asarray(P))

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((trials, n)) * np.sqrt(0.25 * r)      # truth ~ prior
    err = rng.standard_normal((trials, n)) * noise_scale * np.sqrt(r)
    z = jnp.asarray(x[:n] + eps + err)

    def one_trial(z_t):
        _, info = joseph_update(prior, H, z_t, R, PARAMS, label="encoder")
        return per_joint_nis(info.nu, info.S), info

    nis, info = jax.vmap(one_trial)(z)
    return np.asarray(nis), np.asarray(z), info, prior, H, R


def _assert_trial_invariants(nis, z, info, prior, n):
    """The asserts Java makes *inside* `runNISTrials`, on every trial."""
    assert np.all(np.isfinite(nis)), "per-trial NIS must be finite"
    assert np.all(nis >= 0.0), "NIS is a squared quantity"
    assert np.all(np.asarray(info.was_applied) == 1.0), "no encoder update may be gated here"

    # Trial 0: the signed innovation is exactly `z - x_prior` (the prior residual,
    # not the posterior one — which is what the whole NIS convention rests on).
    assert_all_close(np.asarray(info.nu)[0], z[0] - np.asarray(prior.x)[:n], 1.0e-15,
                     "trial 0 signed innovation")


def test_encoder_nis_is_chi_square_consistent_at_the_wired_noise():
    """`encoderNISIsChiSquareConsistentAtTheWiredNoise`: |mean - 1| < 0.089 per joint."""
    nis, z, info, prior, H, R = _run_nis_trials(1.0)
    _assert_trial_invariants(nis, z, info, prior, len(np.diag(np.asarray(R))))

    mean = nis.mean(axis=0)
    worst = np.max(np.abs(mean - 1.0))
    assert worst < ENVELOPE, f"per-joint mean NIS off by {worst:.4f} > {ENVELOPE:.4f}: {mean}"


def test_encoder_nis_catches_understated_R():
    """`encoderNISCatchesUnderstatedR`: noise at 3x sigma => mean NIS > 4.

    The analytic mean is `(P + 9R)/(P + R) = 7.4` at `P = R/4`.  The assertion is
    the loose `> 4.0` from Java: this test exists to prove the statistic *reacts*
    to a mis-stated `R`, not to pin its value.
    """
    nis, z, info, prior, H, R = _run_nis_trials(3.0)
    _assert_trial_invariants(nis, z, info, prior, len(np.diag(np.asarray(R))))

    mean = nis.mean(axis=0)
    assert np.all(mean > 4.0), f"understated R not detected: {mean}"
    # And the value is where the algebra says, which the Java threshold does not
    # check: 7.4 +- a 4-sigma envelope scaled by the mean.
    assert_all_close(mean, np.full_like(mean, 7.4), 7.4 * ENVELOPE, "mean NIS at 3x noise")


# ---------------------------------------------------------------------------
# The deterministic counterpart to the chi-square test (see the module docstring)
# ---------------------------------------------------------------------------

def test_S_is_the_prior_innovation_covariance():
    """`UpdateInfo.S` is `H P^- H^T + R` on the PRIOR `P`, to 1e-15.

    Port-specific, and the test that actually constrains CLAUDE.md §6's
    "NIS on the posterior" trap.  With the map's `P = R/4` the posterior variant
    shifts the mean NIS by 4.2%, less than half the 4-sigma envelope of the
    chi-square test above — so without this assertion the trap would be exercised
    on 8000 samples and caught by none of them.
    """
    build = _build()
    n, dim = build.n_joints, build.dim
    H = encoder_jacobian(build)
    R = encoder_noise(build)
    r = np.asarray(jnp.diag(R))

    x = 0.01 * np.arange(1, dim + 1)
    P = np.diag(np.concatenate([0.25 * r, np.full(n, 1.0e-2), np.full(dim - 2 * n, 1.0e-6)]))
    prior = JointKFState(x=jnp.asarray(x), P=jnp.asarray(P))
    z = jnp.asarray(x[:n] + 3.0e-4)

    post, info = joseph_update(prior, H, z, R, PARAMS, label="encoder")

    Hn, Pn, Rn = np.asarray(H), np.asarray(P), np.asarray(R)
    S_prior = Hn @ Pn @ Hn.T + Rn
    S_post = Hn @ np.asarray(post.P) @ Hn.T + Rn
    assert_all_close(np.asarray(info.S), S_prior, 1.0e-15, "innovation covariance")
    # The two differ by exactly the factor the chi-square test cannot resolve:
    # `S_post_ii/S_prior_ii = (P_post + R)/(P + R) = 1.2/1.25` at `P = R/4`, i.e.
    # 4% — against an 8.9% envelope. Asserted RELATIVELY, because in absolute
    # terms the gap is 4e-8 and would look like rounding.
    rel = np.abs(np.diag(S_prior) - np.diag(S_post)) / np.diag(S_prior)
    assert np.all(rel > 0.03), f"posterior-S variant is only {rel.max():.1%} away"

    # The scalar NIS is the same quadratic form on the same prior objects, and the
    # per-joint statistic is its diagonal marginal.
    assert float(info.nis) == pytest.approx(nis_quadratic_form(info.nu, S_prior), rel=1e-12)
    assert_all_close(per_joint_nis(info.nu, info.S),
                     np.asarray(info.nu) ** 2 / np.diag(S_prior), 1.0e-15, "per-joint NIS")
