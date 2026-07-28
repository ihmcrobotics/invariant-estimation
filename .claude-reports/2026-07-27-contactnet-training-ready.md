# ContactNet — ready-to-train report (2026-07-27, overnight)

**Goal:** get the ContactNet training pipeline to the point where a real run can be
launched with one command in the morning. Not to train it.

**Branch:** `contact-net`. Commits are local, not pushed.

---

## TL;DR for the morning

*(filled in at the end — see "Status" below for the live picture)*

---

## Status by phase

| phase | state | commit |
|---|---|---|
| 0. Rates — sim to 1 kHz, stride from duration | ✅ done | `83ede89`, `e8a75df` |
| 1. Terrain into library code, 64 m hfield | ✅ done | `c4f3f98` |
| **Filter bug found and fixed** (frame conjugation) | ✅ done | `de73c49` |
| 2. Collector | ⏳ in progress | — |
| 3. Calibration (frozen norm constants) | ⏳ pending | — |
| 4. Segmentation, B measurement, smoke train | ⏳ pending | — |
| Theory PDF (`~/Documents`) | ✅ 19 pp, being amended | n/a |

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
