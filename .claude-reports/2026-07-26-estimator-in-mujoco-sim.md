# Estimator in the loop with the policy, in MuJoCo — 2026-07-26 night

**Ask:** wire the estimator to run in the loop with the RL policy in the MuJoCo sim, as it already
does on hardware, so its performance is visible in simulation. Debugging a bonus.

**Status: done and pushed.** Branch `estimator-in-sim`, draft PR
[#5](https://github.com/ihmcrobotics/invariant-estimation/pull/5), based on `model_integration`.
The robot walks 19.4 m in 30 s driven by its own estimate. One real port bug was found and fixed
on the way; it halved the attitude error the policy consumes.

```bash
git fetch && git checkout estimator-in-sim && uv sync
uv run python run_estimator.py --policy baseline                                   # viewer
uv run python run_estimator.py --policy baseline --headless --ticks 1500 --vx 0.6  # 30 s walk
```

First call spends ~45 s tracing MJX; the viewer then runs at ~1/4 speed (~20 ms of estimator per
5 ms physics step, CPU-only jaxlib). Headless 30 s takes ~2 min.

---

## 1. What was built

`run_policy.py` feeds the policy ground truth out of `MjData`. `run_estimator.py` runs the same
sim with the fused estimator in the observation path: simulated IMUs and encoders in, the policy's
`base_ang_vel` and `projected_gravity` out of the filter. Those two terms are the whole of what the
estimator supplies on hardware — the other 92 observation entries are commands, raw encoders and
the last action — so this is the real arrangement, not a sim-only approximation.

One control tick, in order:

```
advance the estimator over the DECIMATION physics steps that just happened   (one scan call)
  -> obs, with the estimated terms substituted -> policy -> DECIMATION * mj_step
```

The estimator runs at the 200 Hz physics rate and the substeps are advanced in a single jitted
`lax.scan`, so the estimate the policy reads is current rather than a control period stale, and
the compiled graph stays constant (I7).

| file | what it is |
|---|---|
| `src/invariant_estimation/sim/sensors.py` | plant → estimator boundary: MJCF sensor injection, `MjData` → `FusedSensors`, contact trust, optional IMU corruption |
| `src/invariant_estimation/sim/estimator_loop.py` | `EstimatorRuntime`: owns the carry and the jitted multi-substep advance; publishes an `EstimateView` |
| `run_estimator.py` | CLI (viewer + headless), A/B flags, scoring vs the sim's own state |
| `tests/sim/` | 24 tests |
| `run_policy.py` | two hooks only: `build_sim_model(with_imu_sensors=)` and `build_obs(est=)` |

Three boundary conventions, each pinned by a test, because each is the kind of error that yields a
filter that runs, converges, and is quietly wrong:

1. A MuJoCo `gyro` reports in its **site frame** — which is exactly the estimator's per-IMU "own
   measurement frame". No rotation is applied at the reader; `R_mount` at the fused boundary does
   that, and rotating twice is a bug this project has already paid for once.
2. A MuJoCo `accelerometer` reports **specific force** (a standing robot reads +9.81 up, a falling
   one reads 0) — precisely what `FusedSensors.accel_base` wants.
3. The contact signal is one tick delayed **by the filter**, not by the harness.

Contact comes from real per-foot normal force (`f_n / 0.5·m·g`) through the Java Schmitt
thresholds and a 40 ms dwell, with immediate release — the asymmetry that stops a bouncing
touchdown from anchoring early and poisoning the bias gauge.

## 2. Results (30 s at vx = 0.6, tail-RMS over the last half; `experiments/sim_runs/`)

| | estimate-driven | truth-driven (A/B) | + IMU noise |
|---|---|---|---|
| tilt error — what the policy reads | **0.81°** | 0.80° | 0.85° |
| attitude error | 0.83° | 0.98° | 0.86° |
| base gyro | 0.003 rad/s | 0.002 | 0.003 |
| base velocity | 0.115 m/s | 0.114 | 0.117 |
| base position drift | 2.20 m | 2.07 | 2.21 |
| distance walked | 19.4 m | 18.3 m | 19.5 m |

**Closing the loop costs nothing.** Estimate-driven and truth-driven score the same, so the filter
is not being destabilised by its own feedback, and the gait is unchanged. **IMU corruption costs
0.03°** — a constant per-IMU gyro bias plus white noise, and the joint KF finds the bias, which is
the invariant (I1) the whole two-filter split exists to serve.

Standing is much tighter than walking: 0.0002 m/s velocity error and a constant 0.012 m position
offset, no drift.

## 3. The bug found: the InEKF's contact FK ignored the ankles

**Symptom.** 2.8 m of base position drift over 20 m walked, and 1.4° of attitude error — about 10x
the hardware-parity noise floor for roll/pitch.

**Root cause.** `_make_contact_kinematics` evaluates the base→sole FK from the 9 filtered joints
only; `MjxModel.qpos` widens that vector by pinning every other joint at `qpos0`. The ankles are
off-path, so the contact FK always believed them to be at zero. Measured over a walk: the ankles
travel **0.66 rad**, and the contact FK error is 3.4 cm mean with a **5.3 cm swing over each gait
cycle**. A constant offset would be harmless — contacts are seeded consistently — but the swing is
not: a planted foot appears to slide 5 cm every step, and a filter whose contacts are stationary by
construction can only explain that as base motion.

Java anchors contacts at the live sole frame (`referenceFrames.getSoleFrame`), so the pinning is a
port gap, not a design choice. It was invisible to the hardware parity harness because that
compared roll/pitch, which gravity leveling holds, and the error lands mostly on velocity/position.

**Fix.** `build_fused_estimator(contact_fk_unfiltered=True)` feeds the measured off-path joints to
the contact FK and widens `Σ_q` with their encoder variance, so `N = J Σ_q Jᵀ` still accounts for
every joint the measurement depends on (widening is not bookkeeping — without it the filter claims
the ankle contribution is noise-free). The seed uses the same augmented vector, or the standing FK
offset would arrive as a step at tick 1.

| tail-RMS | ankles pinned | ankles measured |
|---|---|---|
| tilt error | 1.40° | **0.81°** |
| attitude error | 1.61° | **0.83°** |
| base velocity | 0.143 m/s | 0.115 |
| base position drift | 2.84 m | 2.20 |

**Default is OFF** in the library, so every existing gate keeps the numbers it was recorded
against; the CLI defaults it on. **Flipping the library default is your call** — it changes
hardware behaviour, and the honest argument for it is that Java already does this.

The mass matrix deliberately still sees `qpos0`: that pinning reproduces Mecano compositing the
ignored subtree's inertia once at construction, is worth 14% on `diag(Qa)`, and is a separate
concern from kinematics. The fix does not touch it.

## 4. What I checked

* `tests/sim` — 24 passed. Includes two mutation checks, because a suite that only asserts "the
  robot stands" would pass just as happily with the estimator computed and thrown away:
  * a deliberately corrupted (15°-rolled) estimate **must** change the robot's behaviour — verified
    it genuinely fails when the wiring is cut (ran it with the estimate disconnected: fails with
    "the policy's action did not change when the estimated attitude did");
  * `--source truth` must be bit-identical to `run_policy`, so the A/B control is a true control.
* Adding IMU sensors leaves the dynamics bit-identical (`assert_array_equal` on `qpos` after 200
  steps) — otherwise every A/B against `run_policy` would be invalid.
* Gyro sensors checked against `mj_objectVelocity` in world, rotated by the site — an independent
  path, componentwise, so an axis permutation cannot hide behind a matching norm.
* I7 in the loop: the lowered HLO is identical across a contact-state change.
* The full existing suite (`uv run pytest tests --ignore=tests/sim`): **628 passed**, unchanged —
  the `contact_fk_unfiltered` default-off keeps every recorded gate number intact.

Two incidental facts worth keeping, both of which broke a test I wrote before I understood them:
the home-pose position servos **cannot hold Alex up** (left passive it sinks from z=0.92 to 0.19 in
~5 s; only the policy stands it up), and an accelerometer in free fall correctly reads **zero** —
the robot is in free fall at the seeded pose, 1 mm above the floor.

## 5. Still open

* **Base position drift, ~11% of distance walked.** Two natural suspects, in order: the InEKF has
  no contact zero-velocity constraint (`J_dot = 0`, deferred in `inEKF/filter.py`), and
  `contact_floor = 1e-4 m²` is a permissive per-tick slip allowance. Neither reaches the policy —
  position and velocity are not in the observation — so this is estimator quality, not gait risk.
* **Joint velocity error ~0.8 rad/s peak while walking** (positions are excellent, 8e-4 rad). The
  direct-velocity channel is off by default in sim; it was what took hardware q̇ to ~3%.
* **Speed.** ~20 ms/step is MJX FK + CRB over 49 links on CPU-only jaxlib. A CUDA jaxlib, or the
  `lax.scan`-over-bodies idea in the port-status notes, is the lever if real-time matters.
* The A/B and noise runs in `experiments/sim_runs/` prefixed `walk_est`/`walk_noise`/
  `walk_truthdriven` (no `_fk` suffix) predate the contact-FK fix; the `_fk` ones are current.
