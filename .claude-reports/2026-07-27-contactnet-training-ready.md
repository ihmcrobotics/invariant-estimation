# ContactNet — ready-to-train report (2026-07-27, overnight)

**Goal:** get the ContactNet training pipeline to the point where a real run can be
launched with one command in the morning. Not to train it.

**Branch:** `contact-net`. Commits are local, not pushed.

---

## TL;DR for the morning

**You can launch a real training run immediately:**

```bash
uv run python train_contactnet.py train --steps 10000 --objective l2_velocity \
    --B 32 --no-remat --out artifacts/contactnet_run1.npz --p0 artifacts/p0.npz
```

~75 min. Data (1.5 GB, 12 rollouts), the frozen normalization constants, and the
measured `P0` all already exist on disk.

**Read these three before you do, in this order:**

1. **A real filter bug was found and fixed** — the contact measurement noise was
   never rotated to world. Invisible under isotropic noise (6.4e-22), a 2.7e-3
   relative gain error under the anisotropy ContactNet exists to produce.
   Training against the unfixed path would have been meaningless.
2. **β-NLL does not train on the real model** — the detached weight is 5.2e-21
   and AdamW's `eps` swallows it. Root cause is `det(S)^β` scaling as `σ^{2kβ}`.
   Does not block run 1 (L2), must be decided before run 2. **Do not just patch
   the magnitude** — the semantics are also wrong.
3. **`nis_over_dof` moved away from 1** in the smoke train (1.46 → 0.034) and
   that is correct behaviour under L2, not a failure. Explained below.

Full suite **704 passed**. Everything committed locally on `contact-net`, nothing
pushed. Theory PDF: `~/Documents/contactnet_theory.pdf` (21 pp).

---

## Status by phase

| phase | state | commit |
|---|---|---|
| 0. Rates — sim to 1 kHz, stride from duration | ✅ done | `83ede89`, `e8a75df` |
| 1. Terrain into library code, 64 m hfield | ✅ done | `c4f3f98` |
| **Filter bug found and fixed** (frame conjugation) | ✅ done | `de73c49` |
| Regression test for the filter bug | ✅ done | `61cd0b0` |
| 2. Collector | ✅ done | `2e17add` |
| 3. Calibration (frozen norm constants) | ✅ done, gate passed | `<dataset>` |
| 4. Segmentation, B measurement, smoke train | ✅ done | `<dataset>` |
| Theory PDF (`~/Documents`) | ✅ 21 pp | n/a |

---

## Phase 0 — rates

### What was wrong

The estimator ships at 1 kHz (`CLAUDE.md` §8) and is validated against a 1 kHz
hardware log, but the sim ran at **200 Hz** (`run_policy.DT = 0.005`). That gap
became load-bearing once ContactNet's history window started deriving its tick
stride from `DT`: the same config meant a 393 ms window at 1 kHz and a **1.96 s**
window with a 12.5 Hz Nyquist at 200 Hz, with nothing raising.

### What changed

* `DT` 0.005 → 0.001 and `DECIMATION` 4 → 20, so `CONTROL_DT` stays exactly 0.02 —
  the rate the policy was **trained** at, which is the real invariant. The two
  constants move together; `DT` alone gives 250 Hz control, a trap already recorded
  in the comment above them.
* `stride` is no longer a config field. It is derived from `window_span_s` and `dt`,
  with `window_span_seconds` / `effective_rate_hz` / `nyquist_hz` exposed so the
  bandwidth argument is inspectable rather than implied.

### Verification

* Full suite **653 passed** at 1 kHz.
* Three `tests/sim/test_sensors.py` failures were **test bugs, not regressions** —
  `ContactTrust` accumulates its 40 ms dwell in *seconds* and was already
  rate-correct; the tests hardcoded 20/20/30 ticks. Tests in the same file that
  derived their counts from `DT` all passed, which is what identified the cause.
* Fixed tests verified **rate-independent**: 15 passed at 1 kHz *and* at 200 Hz.
* Config behaviour bit-preserved at `dt=1e-3` (stride 8, span 393, `d_in` 1200);
  at `dt=5e-3` the same config now gives stride 2 — seconds carry over, not ticks.

### Known limitation (documented, not hidden)

Window spans quantize to multiples of `(H-1)·dt` = 49 ms at the defaults, so the
achieved span can differ from the request by up to one quantum. Inherent to an
integer stride. `window_span_seconds` reports the achieved value.


---

## Phase 1 — terrain

`experiments/terrain_stage1_walk.py` duplicated `run_policy`'s model assembly.
`build_sim_model` now takes `floor=` (`PlaneFloor` / `HeightfieldFloor`), so the
collision set, contact params and actuators are single-sourced. Terrain lives in
`sim/terrain.py`; `EXTENT` 16 → 64 m (16 m gives only ~21 s of walking from a
centre start, not enough once a warm-up prefix is discarded).

Walk re-measured at 1 kHz, all four IsaacLab sub-terrains upright 20 s, within
±0.16 m travel and ±0.3° tilt of the 200 Hz reference.

**Two findings worth knowing:**

* `waves` wavelength was extent-dependent. The prototype spread `num_waves=2`
  over its whole field, so growing EXTENT to 64 m stretched the wavelength 8 m →
  32 m — a gentle ramp still labelled "waves", i.e. a **silently easier
  terrain**. Now per 8 m IsaacLab tile.
* The obvious terrain check is structurally wrong: comparing resting height flat
  vs `hard_stepping` differs by 0.8 mm, because `hard_stepping` has
  `platform=0.5` — a flat spawn pad *by design*. Replaced with the lower foot's
  sole plane tracked against `terrain.sample()` over the whole walk.

14 new tests, mutation-checked against 7 mutants.

---

## The filter bug — read this one

**`inEKF/filter.py` was mixing frames in the contact update, and it is the single
most important thing found tonight.**

`correct.innovation` returns a **world-frame** residual (`R̄y − (d̄−p̄)`, as its own
docstring states), while `Np = J_C Σ_q J_Cᵀ` and ContactNet's `Σ_C` are
**body-frame**. They were passed into `linear_update` unconjugated.

The port already contained `rotate_measurement_covariance`, documented as the
port of Java `ContactUpdater.computeMeasurementCovariance`, saying outright that
this conjugation is required — and never called it.

Why 653 tests passed over it, measured with ‖R̂−I‖ = 2.82:

| `Σ_C` | max\|N_body − N_world\| | relative Kalman-gain error |
|---|---|---|
| isotropic `1e-6·I` | **6.4e-22** | 1.2e-21 |
| anisotropic `diag(1e-3, 1e-3, 1e-8)` | 3.3e-4 | **2.7e-3** |

`R̂(σ²I)R̂ᵀ = σ²I` exactly, so under the shipped isotropic default the bug is
machine-zero invisible. It is *not* a no-op for anisotropy — and anisotropy is
ContactNet's entire justification. Training against the unfixed path would have
optimised a network through a filter discarding the thing it was learning.

Fixed; full suite **667 passed**. Worth an anisotropic regression test when the
ContactNet suite is written.

**Two smaller fixes from the same review pass:** `normalize.NOISE_FLOOR`'s `v_bc`
was 7071× too small and therefore inert (it assumed a `1/stride` reduction that
does not apply, because `fit`/`apply` run *before* windowing); and two dangling
doc citations, one to a PORT_NOTES table that did not exist (now written) and one
to a test file that does not exist (now marked NOT YET WRITTEN).

**Flagged, not acted on:** β-NLL's `β` is dimension-dependent — `det(S)^0.5` at
k=6 weights as `s³`, not the source formulation's per-dimension `s^0.5`, and
`losses.py` has two entry points at different `k` where the same β means
different things. Doesn't block run 1 (L2 baseline). Decide before run 2.


---

## Phases 2–4 — collector, calibration, dataset, smoke train

### Measured numbers you asked for

| quantity | value | how |
|---|---|---|
| Joint-KF warm-up | **16 000 ticks (16 s)** | 1 s bias drift < 5% of excursion, worst terrain |
| `diag(Σ_q)` plateau | 3.2 s | the quantity that actually reaches the contact update |
| Sim throughput | **3.2 wall-s per sim-s** | 20 cores, CPU-only jaxlib |
| `B` | **32, no remat** | 0.44 s/step; not memory-bound |

`B` is not memory-bound at all — ~8.5 MB per unit `B`, and `B = 64` peaks at
2.9 GB of 94 GB. The largest allocation in the process is the **XLA compiler**,
not the gradient, because the scan carry is a 15×15 covariance and the network
runs *outside* the scan. remat saves 43% of execution memory at `B = 64` but
costs +250 MB compile peak and **+47% step time**, so it is a net loss here.

### Calibration gate: PASSED

`floored` is **empty** over 1.104M pooled post-warm-up samples, by a wide margin
(closest is `base_gyro_x` at 32× its floor). The standing-only failure this gate
exists to catch does not occur on the gait-spanning set. Physical spot checks:
`base_accel_z` 9.807, `p_bc_z` −0.866 m, `q_knee_y` 0.99 rad vs `tau_knee_y`
−57.6 N·m.

### Smoke train — passed, with one result that needs reading correctly

200 steps, `l2_velocity`, `B = 32`: loss down **82×**, gradients finite and
non-zero throughout, no NaN, `applied_frac = 1.000` every step.

**§4 init parity held end-to-end on real feature windows**: `Σ_C − σ₀²I` =
1.3e-15 relative, trunk gradient **exactly 0.0 bitwise** on every leaf, head
gradient 0.949. That is the property the whole initialization design exists for,
now verified through the real data pipeline rather than on synthetic input.

**`nis_over_dof` went 1.46 → 0.034 — away from 1.** This is correct under L2 and
was predicted: `losses.l2_velocity`'s own docstring says Σ reaches an L2 loss
only through the Kalman gain, so only *ratios* are constrained and the optimiser
buys accuracy by inflating absolute scale. Under L2 the loss is the criterion and
NIS is a diagnostic. Closing that gap is precisely what β-NLL is for — which is
why finding #2 above matters.

**Watch this on the long run:** the conditioning proxy climbed 1.9 → 1.8e8 in 200
steps against `cond_max = 1e9`. No update has been gated yet, but 10 000 steps
will reach it, and a gated update leaves `(X, P)` unchanged — so `applied_frac`
dropping below 1.0 is the signal to stop and look.

---

## Open items, ranked

1. **β-NLL semantics** (above). Blocks run 2, not run 1.
2. **Shin-IMU bias observability.** Six of eight IMU bias states converge to
   0.007–0.025 rad/s of truth; `left/right_shin_imu` settle **0.21 rad/s off** —
   5× the injected bias and 97% of the total error. Neither reaches the InEKF
   (I1 takes the base IMU) nor ContactNet (raw gyro), so it does not block, but
   it is a real joint-KF question.
3. **`contact_chol` frozen at stance forces ContactNet to reject swing feet
   through the measurement covariance**, which `inEKF/filter.py`'s DECISION note
   argues is the wrong lever. Coherent only if the deployed filter also drops the
   heuristic swing inflation — otherwise the two double-count at deploy.
4. **`make_contact_channels` reaches 38 GB at T = 62 000.** Use
   `collect.contact_channels_chunked`. Anything else running the feature path
   over a full rollout needs the same treatment.
5. **`network_plan.md` §8's ~240K parameter budget is stale** — at `H = 50`,
   `d_in = 1200` gives 374 790 params. It is a Java inference-cost question, so
   re-decide it rather than silently exceed it.
6. `make_segment_loss` cold-starts the gravity reference each segment (τ = 5 s
   against 0.128 s segments), so it is a constant offset, not a transient — but
   it is a train/deploy difference.
7. Pre-existing, unrelated: 6 `ruff` errors in `jointKF/measure.py` and two test
   files, including a full `L Σ Lᵀ` computed and discarded at `measure.py:326`.
