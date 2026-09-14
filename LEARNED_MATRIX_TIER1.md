# Learned matrix: Tier 1 item 2 — 2026-09-14

## Ownership and status

Codex owns scalar-noise learning: learning/noise.py, two_stage.py, optimize.py,
and tests/learning/test_noise.py. Claude owns parity (item 1) and multi-session
data (item 3).

**Claude review, 2026-09-14 (post-handoff):** captures.py/test_captures.py
verified -- independently re-read for correctness (leakage-detection logic,
path-safety checks, deterministic-split hashing) and re-run in isolation
(10/10 passed, not just included in the full-suite count). No bugs found.
Also independently re-verified noise.py/two_stage.py/optimize.py's own claims
by direct code reading (bias-subtract-before-rotation order, mask-zeroed
gradient for frozen channels, Adam bias-correction math, body_velocity_l2's
exact cancellation of the filter's unobservable global yaw mode via R^T v in
both estimate and truth) -- no bugs found there either.

**Found the one real gap: captures.py and noise.py/two_stage.py/optimize.py
had zero code connecting them.** Added `tests/learning/test_pipeline_integration.py`:
builds a real (hash-verified, save/load-round-tripped) `CaptureManifest` over
synthetic sessions, drives `fit_scalars` from ONLY the manifest's train
partition, and confirms the held-out loss is computed from the manifest's
test partition specifically (changing which session the split calls "test"
changes the held-out number -- caught and fixed a vacuous version of this
assertion during review: two candidate seeds happened to select the same
held-out session, which would have silently skipped the one check the test
existed for). This is a wiring/seam test, not an accuracy or real-data claim
-- see that file's own docstring for the scope boundary.

Full suite after this change: see the run recorded below.

No existing filter kernel, Java file, config or lockfile changed. A local .venv
was installed with uv sync --frozen --no-default-groups --group dev.
Java loading (item 4), the real capture-to-JAX-input loader, real-data
training and robot activation remain pending -- robot time starts
2026-09-15, and the `ihmclog` decoder `tests/replay/`'s own harness needs is
not installed on this machine, so neither can be exercised yet regardless.

## Parameter contract for Claude / Java integration

All values are dimensionless **variance multipliers**, not standard deviations.
The dimensional baseline and anisotropy stay fixed. NoiseSpec.imu_names must
match JointKFBuild.imu_names exactly, including order.

| Channel | Destination |
|---|---|
| imu_gyro:<name> | raw per-IMU build.gyro_sigma[i] |
| base_gyro_q | ekf.params.gyro_var |
| base_accel_q | ekf.params.accel_var |
| contact_q | low-level sigma_c OR trajectory contact_chol times sqrt(scale) |
| contact_fk_r | correction-only Sigma_q copy, hence s * J Sigma_q J.T |
| gravity_roll_r | enabled roll measurement variance |
| gravity_pitch_r | enabled pitch measurement variance |

Per-IMU scaling intentionally replaces independent per-pair scaling: the
existing L Sigma L.T implementation preserves all shared-sensor pair and anchor
cross blocks. Scale the baseline AFTER its acquisition/flooring; do not silently
re-floor learned values differently in Java.

FK R is a global multiplier of propagated joint covariance, **not an additive
constant R**. Java's constant-only measurement provider is not an equivalent
deployment hook. Joint means and the actual JointKF covariance are unchanged;
only a correction-side copy of Sigma_q is scaled. Contact Q is separate.

Scale = exp(b*tanh(theta/b)), b=log(max_scale), default max_scale=100.
Zero theta is exactly baseline. Arm 4 freezes all; arm 5 learns raw-IMU scales;
arm 6 learns six InEKF scalars; arm 7 learns both. Frozen channels remain 1.
Initial P, gates, dt, H, Phi, disabled-pitch variance and sigmaTau are not learned.
Contact covariance floors remain fixed. These are static learned scales, not a
new ContactNet network.

## Usage

    from invariant_estimation.learning.noise import NoiseSpec
    from invariant_estimation.learning.two_stage import run
    from invariant_estimation.learning.optimize import body_velocity_l2, fit_scalars

    spec = NoiseSpec(tuple(build.imu_names), arm=7)
    def loss(theta):
        final, out = run(theta, spec, build, joint_params, ekf, kinematics,
                         imu_to_body, initial_carry, training_inputs)
        return body_velocity_l2(out.base.state.R, out.base.state.v,
                                truth_rotation, truth_velocity)
    fit = fit_scalars(loss, spec.initial_theta(), steps=100)
    scales = spec.scales(fit.theta)

Construct/run the parameterized scan INSIDE the differentiated loss, with theta
as a dynamic JAX argument. Do not close over a fixed theta before differentiation.
fit_scalars accepts any scalar differentiable loss; a separately validated
consistency regularizer can be added. It does not implement a new NEES metric.

TwoStageInputs.model uses the existing joint filter's model-input contract.
If model quantities depend on the learned trajectory, evaluate them consistently
in the full rollout; precomputed values are a frozen-linearization approximation.
The adapter is not a robot-log decoder or synchronization implementation.
imu_to_body explicitly maps base gyro measurement frame to pelvis/body frame.
Subtract the updated JointKF bias BEFORE rotation. accel_body must already be
bias-corrected and expressed in that body frame. Raw gyro drives gravity gating.
Each independent session needs a fresh TwoStageCarry and training-only
normalization; do not leak state, covariance or fitted statistics across splits.

## Verification

Command from this repository:

    .venv/bin/python -m pytest tests/learning/test_noise.py tests/inEKF/test_filter.py tests/inEKF/test_contact_updater.py tests/jointKF/test_filter.py -q

Result: **47 passed in 32.35 s** (13 new + 34 existing).
Tests cover bounded scales, arm freezing, shared-IMU cross covariance, unchanged
baseline, BPTT gradients from every channel to pelvis velocity L2, finite-
difference agreement, synthetic noisy-input loss reduction and nonfinite failure.
Item 3 tests were not run by Codex and are not included in this claim.

**Claude, full-repo re-verification after review + integration test, 2026-09-14:**

    .venv/bin/python -m pytest tests/ -q --ignore=tests/replay

Result: **619 passed, 0 failed** (tests/replay excluded: those need a real
hardware log plus the `ihmclog` decoder, neither present on this machine --
confirmed by direct attempt, not assumed). Includes item 3's 10 tests
(captures.py, now verified) and the 2 new integration tests. Also independently
re-ran `tests/learning/test_captures.py` alone (10/10) and the full `tests/inEKF/`
suite alone (287, including item 1's parity test) to confirm no test was only
passing as a side effect of import/fixture ordering in the combined run.

L2 need not uniquely identify physical Q/R. A separate Gaussian likelihood test
recovers empirical variance where that parameter IS identifiable; this is not
a claim that arbitrary filter Q/R can be recovered from pelvis error alone.

## Pending parity gates

- Common math parity is separate from full policy parity: Python contact/gravity
  ordering and omitted Java contact/reseed policies must not be hidden by tolerance.
- Verify raw-IMU covariance floors and pair/anchor cross blocks on Java.
- Low-level sigma_c and scan contact_chol are alternative process-noise entry
  points; do not apply contact_q twice.
- Match the J Sigma_q J.T route in Java before transferring contact_fk_r.
- No cross-language equality, held-out improvement, or robot safety claim yet.
