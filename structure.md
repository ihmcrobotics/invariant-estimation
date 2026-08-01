# structure.md — what is in this repository, and where the knowledge lives

A map of the code, assembled 2026-08-01 by reading every module's comments and
docstrings. It exists because this repo puts **load-bearing content in comments**:
measured numbers, refuted hypotheses, and named traps that appear nowhere else.
This file is the index to that; it is not a substitute for reading a module before
changing it.

**Authority, unchanged by this document.** `CLAUDE.md` (invariants I1–I10, gates
G1–G10) and `TEST_SUITE_MAP.md` (the binding Java-suite contract) govern. Where
this map and those disagree, they win and this file is wrong.

**Reading order for a newcomer:** `RUNNING.md` (how to run anything) → this file
(what exists) → `DESIGN_DECISIONS.md` (things that look like bugs and aren't) →
`PORT_NOTES.md` tail (what was measured recently).

---

## 1. The shape of the thing

Two filters in series, a learned covariance module hanging off the second, and a
MuJoCo/MJX harness to exercise both.

```
  MuJoCo plant ──> sim/sensors.py ──> jointKF/  ──(q̂, Σ_q, b̂_ω)──> inEKF/ ──> estimate
                        │                                            ↑
                        │                                     contactnet/ (Σ_C)
                        └──> sim/collect.py ──> data/*.npz ──────────┘
```

- **`jointKF/`** — joint-space KF over `(q, q̇, b_ω)`. Schur-complement process
  noise, stacked gyro measurement over the IMU graph, stance anchors.
- **`inEKF/`** — world-centric right-invariant InEKF on `SE_{N+2}(3)`. Consumes
  bias-corrected IMU (I1); contacts are permanent state slots (I2).
- **`contactnet/`** — MLP emitting a per-contact covariance Cholesky, trained by
  BPTT through the differentiable InEKF.
- **`pipeline/main_estimator.py`** — fuses the two into one `lax.scan` body with a
  constant XLA graph (I7).
- **`sim/`, `model/`, `replay/`** — the plant, the URDF→MJX model seam, and the
  hardware-log parity harness.

---

## 2. Source packages

### `inEKF/`

`group.py` supplies rotation-first Lie primitives (I4); `state.py` holds the carry
plus the precomputed constants `Φ` and `H` that make the design work (I3/I6);
`propagate.py` and `correct.py` are the predict/update halves, with
`correct.linear_update` as the *single* gain/Joseph/gate path every measurement
goes through — which is what makes the ported EKF-delegation tests exact to 1e-12.
`ekf.py` is pure wiring; `filter.py` composes one tick and carries the package's
most consequential design record.

| file | what it is |
|---|---|
| `group.py` | Γ₀/Γ₁/Γ₂, exp/log/Ad. Small angles use the **double-`where`** trick (clamped θ² for the analytic branch so `0*NaN` can't backprop, raw for the Taylor branch). |
| `state.py` | `InEKFState`/`InEKFParams`, `build_Phi`, `build_H`. `N` deliberately **not stored** (inferred from `d.shape[0]`) to avoid recompiles. |
| `propagate.py` | Exact-mean group propagation + `Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt`. `Ad` taken at the **prior** (Java predict ordering). |
| `correct.py` | Innovation, gain, Joseph, `linear_update`. `logdet_S` derives from the same Cholesky as `nis` "so it cannot drift" — exposed for β-NLL. |
| `contact.py` | ContactNet Cholesky → Σ_C digest. **Additive** floor, not an eigenvalue clamp, so BPTT stays finite at singular Σ. |
| `gravity_update.py` | Accelerometer roll/pitch update. Rank-2 `H`, exact null along `e_z`, complementary reference τ = 5 s. |
| `reseed.py` | Touchdown congruence + `TouchdownReseedLatch`. Off by default. |
| `ekf.py` | Immutable wiring; Java's getters become the returned `UpdateDiagnostics` pytree. |
| `filter.py` | The scan body and `run`. Holds the no-contact-mask decision and the two-socket table. |

**Load-bearing:**

- **No contact mask, on purpose.** FK during swing is not wrong; what breaks is
  world-staticity, which lives in **process** noise. Measured: Σ_C = 1.0 for 100
  swing ticks absorbs **96%** of an 8 cm foot displacement into the anchor,
  perturbing the base by **3.7 mm** — 7.6× attenuation vs planted. The residual
  `P_pp/(P_pp+P_dd) ≈ 8%` is correct Bayes, not a leak.
- **`contact.apply_floor` is safety-critical.** At `contact_floor = 1e-6` the
  closed-loop filter measured **−15 m drift and 18° tilt** and the robot falls; a
  pinned swing foot is **10.2× worse** in body-frame velocity than not using
  contacts at all. Two floors act here (network `eps = 1e-6` on `diag(L)`, and the
  physical `floor`) — do not stack them.
- **Body→world rotation of `N` was missing until 2026-07-28.** Invisible for
  isotropic noise (6.4e-22) but shifts the gain 2.7e-3 relative for anisotropic
  slip — exactly what ContactNet produces, and the net cannot compensate because it
  never sees `R̂`.
- **Gravity `R` triad is built about the *predicted* direction `R̂ᵀe_z`**, not the
  measured one; the measured one misaligns the null directions by the very tilt
  being corrected (~10× slower pitch convergence).
- **Regression F.3:** the horizontal quasi-static gate resolves against the
  sensor-driven `ĝ_ref`, never `R̂ᵀe_z`. The old version locked leveling out above
  **2.92°** of tilt.
- **The reseed latch exists because of hardware log `20260717_112516`**, where
  contact probability pulses 1→0→1 *inside one foot strike*. `P_θd = P_θp` is the
  zero-release condition.
- **`contact_velocity_noise` uses `J_C`, not `J_Ċ`** (corrected 2026-07-29):
  `J_Ċ Σ_q̇ J_Ċᵀ` has units m²/s⁴, and MJX returns `J_dot = 0`, so the old pairing
  was identically zero while looking wired.
- `raw_omega` is carried separately from bias-corrected `omega` solely because
  `testRotationGateUsesRawGyroNotBiasCorrupted` requires the gate to see the
  uncorrected signal.

### `jointKF/`

State `x = [q; q̇; b_ω] ∈ R^{2n+3m}`, where **m = distinct IMUs, not pairs** (I6).
`state.py` is the frozen contract; `build.py` turns names into indices once in
plain Python (I7). One tick is `predict.py` → `process.py` (Schur → `Λ_eff` → Gram
`Qa` → Van Loan) → two **sequential** Joseph updates via `update.py`, fed by
`measure.py` and `anchors.py`. Model quantities are always passed *in* — no module
here imports a simulator.

| file | what it is |
|---|---|
| `state.py` | Layout, `JointKFParams`, `JointKFBuild`, `SEAM_MAP`. `_ci_get` is **exact** case-insensitive; `_substring_lookup` is **longest-key-first**. Conflating them makes `LEFT_HIP_X` inherit `HIP_X`. |
| `build.py` | Graph resolution, union-find acyclicity, F/U anchor split. Anchor chains root at the **base IMU's body**, not the world. |
| `process.py` | `Λ = M_jj − M_jb M_bb⁻¹ M_bj`, `Λ_eff`, `Qa = YYᵀ`, Van Loan. `QA_MAX` surfaces, never rescales. |
| `predict.py` | `F = I + AΔt` exact (`A² = 0`). Bias↔joint blocks structurally zero — coupling there would *manufacture* observability. |
| `measure.py` | Encoder rows, stacked pair rows, `mixing_operator`. |
| `anchors.py` | The only absolute observation of gyro bias. `R_anchor = Σ_ε + J_U diag(σ²)J_Uᵀ`; `Σ_ε` is the ContactNet socket. |
| `update.py` | The single gated Joseph update. `cond(S)` proxy over **informative rows only**; NIS on prior `P`. |
| `velocity.py` | Optional direct-q̇ channel, lag inflation `R = σ² + (d̂/ω_eff)²`. Off by default. |
| `diagnostics.py` | `per_joint_nis` in-jit; host-side structured row attribution instead of Java's message string. |

**Load-bearing:**

- **I6 is invisible on 3 of 4 test shapes.** For isotropic `Σ = σ²I`, per-pair
  block-diagonal `R_g` and `LΣLᵀ` agree to **2.6e-20**. Only anisotropic `Σ` or a
  shared-IMU star separates them.
- **Anchor rows belong in the congruence — a deliberate divergence from Java.**
  Building `R_anchor` as a separate diagonal block (what Java and `CLAUDE.md` §2
  prescribe) drops the base-IMU noise *and* the anchor↔pair cross-terms: the
  stacked oracle failed **12/12 trials**, off by 2e-4…9e-4 against a 1e-5 tolerance.
- **Gap joints ≠ anchor-chain unfiltered joints.** Alex's ankles are
  anchor-unfiltered but *off* the root→filtered paths. Worth **1.7%** on
  `diag(Qa)`. The two sets coincide on a serial chain, which is why unit fixtures
  never caught it.
- **Rotor double-add.** Armature folds into `qM` pre-Schur, so `Λ` *is already*
  `Λ_eff`; re-adding starves `Qa` by **~4× on distal joints** with no error raised.
  Without the rotor term, `Λ⁻²` carries diagonal outliers to ~1.6e6 — the Alex002
  velocity-covariance blow-up.
- **`R_LARGE` and `cond_s_max` are mutually destructive as configured.** Counting
  masked anchor rows gives `cond(S) ~ 4e11 ≫ 1e9`, gating the *entire* stacked
  update (gyro rows included) on every swing tick. Hence the informative-row mask.
- **`Qa` must NOT be symmetrised.** The Gram form is bit-exactly symmetric on XLA;
  a `0.5(A+Aᵀ)` would turn `tol = 0.0` symmetry assertions into tautologies.
- **QA_MAX rescaling was actively harmful:** the superseded uniform rescale starved
  global `Q` (hips down ~6 orders), *causing* the `S` singularity it was meant to
  prevent.
- **Sequential, not concatenated, updates:** one stacked block means a single bad
  gyro row throws the encoders away. Encoders go first so `J_ang(q̂)` linearises at
  the freshest `q̂`.
- Gated updates use `jnp.where` on the *whole carry*, not just `K = 0` — Joseph
  re-derivation is not bit-identical. NaN sanitisation sets `R → I`, never `0`.

### `contactnet/`

`config.py` is one frozen, never-traced dataclass; `features.py`/`normalize.py`
build and standardise the 24-channel sensor-only windows; `network.py` is a
trunk+Cholesky-head MLP initialised *at* the analytic filter; `losses.py` +
`rollout.py` scan the filter over an `L`-tick segment; `dataset.py` is a 3-pass
pipeline (MJX cache → frozen norm constants → pure-NumPy batching); `online.py` is
the deployment ring-buffer counterpart of `features.py`.

**Window geometry** — `H = 50`, `window_span_s = 0.392`, `stride` **derived** as
`round(span/((H-1)dt))` = 8, so a rate change cannot silently change the window.
`features.boxcar` is the anti-alias filter for that stride and is composed *inside*
`window` so the two cannot be separated by accident.

**Load-bearing:**

- **Frozen `contact_chol` is 10.2× worse** than not using contacts at all at the
  128 ms horizon. Primary cause of run 1's collapse.
- **Refuted, three times over:** "passing the real `contact_chol` hands the network
  a free ground-truth contact flag." The network's input is the 24 channels;
  `contact_chol` enters only the filter's *process model*.
- **`sigma_0 = 1e-4` is a measurement-socket number and does not transfer.** On the
  process socket a constant `σ₀·I` at iteration 0 *is* `freeze_contact_chol`. The
  config says outright: choose deliberately before the next run; nothing enforces it.
- **Bandwidths (2026-07-17 log, 1 kHz):** joint pos f99 = 1.10 Hz, torque 4.25 Hz,
  gyro 53.9 Hz, accel 155.6 Hz; stride fundamental 0.183 Hz (5.47 s). Sensors latch
  at 500 Hz behind a 2-tick hold, so at `stride = 1` every second sample repeats.
- **Accelerometer:** ~24% of its power is above 50 Hz, and walking carries **43×**
  more of it than standing. Boxcar chosen over an IIR because it is **stateless**
  and therefore portable to EJML.
- **`v_bc` floor trap:** an earlier table had 1e-6, **7071× too small** and
  therefore inert.
- **Memory:** vmapping full-body MJX FK over `T = 62 000` reached **38 GB RSS**;
  hence the chunked cache.
- **`warmup_ticks = 16 000`** is the measured joint-KF gyro-bias plateau.
- **`episode_s = 43.0` is a ceiling, not a choice:** 62 s rollout − 16 s warm-up
  = 45.5 s of legal starts, minus `warm_in_s`.
- **Chain phase staggering:** seeding all B chains at the same tick makes them
  re-seed in a synchronised wave forever, destroying most of the B samples.
- Heel and toe share `q_sub`/`tau_sub` entirely and are distinguished only by
  `p`/`v`: `p_x` is near-constant **+0.1475** (toe) vs **−0.0495** (heel), which is
  the shared-weight network's "which point am I" identifier.

### Core, model, pipeline, replay, entry points

`__init__.py` flips `jax_enable_x64=True` at import — **process-global**, so it
forces float64 on anything else in the process. `config.py` exists partly because
**PyYAML is YAML 1.1: `1.0e9` parses as the string `"1.0e9"`**, surfacing thousands
of lines later as a dtype error inside jit.

`model/urdf2mjcf.py` converts URDF (or the `model.sdf` inside an SCS2 log) to MJCF
carrying armature, IMU/sole/toe-heel sites and full-precision joint origins.
`model/mjx_model.py` is the one place FK, site angular Jacobians and `M(q)` enter.
`pipeline/main_estimator.py` is the fused scan body.

Entry points: `run_policy.py` (estimator-free sim + ONNX policy; all sim magic
numbers live here), `run_estimator.py` (filter in the observation path, scored
against truth), `train_contactnet.py` (cache/norm/p0/train), `sim_scaffold.py`
(superseded, and **not self-contained** — its `PCFG` points at an absolute
`persona_rl` path).

**Load-bearing:**

- **Off-path joints must stay at `qpos0` for `M(q)`** to match Java: Mecano
  composites ignored-subtree inertia once at `q = 0`; live angles move `diag(Qa)`
  by up to **14%**. But nuisance-*eliminating* off-path joints instead of locking
  them is worth **58%**. Java is knowingly stale here; the port matches Java
  because the replay suite measures parity.
- **Contact FK must use live ankles**: pinned ankles swing the base→sole vector
  **5.3 cm over a gait cycle**, so a planted foot appears to slide that much per
  step. Kinematics and inertia deliberately diverge on this point.
- **`R_mount`** verified against Java to **1e-18**. **Sole offset**
  `(0.197/2 − 0.052, 0, −0.072)`; emitting soles at the link origin put contacts
  **7.2 cm above ground and 4.65 cm behind** the sole centre.
- **G9 landmines:** `imu_bias_process_var` 1e-4 is the Java *unit-test* value —
  flight is 0.0, else fused bias is ~200× too noisy. Flight's
  `contactMeasurementVariance = 1e-4` has no port analogue; exposed as
  `contact_meas_var`, default 0.0.
- **Toe/heel is deliberately not coupled to anchors**: two anchors on one rigid
  foot double-count bias information with no cross-covariance — the same class of
  error as block-diagonal `R_g` (I6).
- **`mjOBJ_XBODY`, not `mjOBJ_BODY`**: with `flg_local=1` MuJoCo resolves
  `mjOBJ_BODY` in the *inertial* frame, permuting Alex's pelvis axes. This was the
  bug that made every policy fall.
- **URDF `rpy` is extrinsic XYZ; MJCF `euler` is intrinsic** — `urdf2mjcf` never
  emits `euler`, always `quat`. The pelvis IMU is yawed 90°, so this is not a
  small-angle detail.
- **ORT pinning is the real-time fix**: the default session adds **9 threads** on a
  20-thread box. Interleaved A/B: 0.83×/0.67× → 1.12×/1.24× real time.
- **A 630 s Alex log is 131 GB uncompressed**, hence `logsource`'s `.npz` cache.
  The estimator consumes the highest `_spN` stage, never `raw_*`.

### `sim/`

`sensors.py` is the hub — plant→estimator boundary, contact trust, toe/heel load
split, `EarlyRelease`. `collect.py` runs the *open-loop* variant (policy on truth,
filter along for the ride) and carries the DR knobs; `estimator_loop.py` is the
*closed-loop* one (policy reads the estimate).

**Load-bearing:**

- **Sampling-rate trap:** a naive loop around `control_tick` samples at 50 Hz
  (`DECIMATION = 20`) and *looks* right — arrays have a time axis and loss goes
  down. The test fails on runs of 20 identical values.
- **Two of eight bias states don't converge:** six IMUs land 0.007–0.025 rad/s from
  a 0.042 rad/s injected bias; `left/right_shin_imu` settle **0.21 rad/s off** =
  97% of the vector error. The base IMU — the only one the InEKF consumes (I1) —
  converges to 0.007 rad/s. Open joint-KF observability question.
- **`off_dwell = 0` fragments stances:** a 30 s walk reports **306 "liftoffs"** on
  two feet against a real ~2 steps/s; MuJoCo drops the contact set for 1–5 ticks
  mid-stance.
- **Threading traps:** the queue is a `deque` with **no maxlen** (maxlen drops the
  *oldest* sample — the worst possible); chunks must be exactly `substeps` or XLA
  retraces (~11 s stall); back-pressure waits on `_alive()`, not `_error is None`
  — mutation testing found this, and **the test that should have caught it hung
  instead of failing**.
- **Off-field is the silent failure:** past the hfield edge MuJoCo *clamps*, so the
  robot walks an infinite extrusion of the boundary row and every downstream array
  stays finite.
- `truth()` uses `mjOBJ_XBODY` for the reason above.
- **N = 4 is baked into a dataset** (`(T,N,3,3)` arrays) — an N=2 dataset cannot be
  replayed under an N=4 filter.

---

## 3. `experiments/` — the measurement layer

One-off scripts, each carrying its own result in its module docstring, often
including the refutation of the hypothesis it was written to confirm. Pre-training
gates (`alpha_sweep`, `phase_lock`, `measure_tstar`, `sigma_bound`, `slip_probe`)
decide whether a dataset or objective is worth a GPU hour; post-training scorers
(`check_sigma`, `replay_eval`, `reseed_table`, `summarise_runs`) decide whether a
checkpoint helped.

**The results that changed a design decision:**

- **Release *timing*, not anchor *tightness*, controls the vertical sink.** Arm B
  19× at a 100-tick lead (−0.03750 → −0.00200 m/s); arm C flat across 1e-4→1e-2
  stance values. And the proposed mechanism was **refuted by the script's own
  output**: the Schmitt trigger releases *at* liftoff, so `P_dd` is already at its
  swing value on the first swing tick. The real asymmetry is stance vs swing,
  ~1e6 in velocity gain. (`process_socket_ablation.py`)
- **Touchdown impact defeats a naive running-peak release:** peak normal force is
  2.0–10.8× the stance median, firing the latch over 70–86% of stance and scoring
  *worse* than baseline. Hence `IMPACT_BLANK_TICKS = 150`.
- **The L2 objective has no interior optimum in Σ_C unless `T* < L·dt`.** Run 1
  converged to "ignore the feet", 3835× velocity-gain suppression.
  (`measure_tstar.py`, `alpha_sweep.py`)
- **A gain ratio cannot distinguish degenerate from correct.** Run 2's z axis is
  suppressed 3115× and is nonetheless **8.5× better in height** than the heuristic.
  Only running the filter settles it. (`check_sigma.py` → `replay_eval.py`)
- **Refuted:** slip is not measurable post hoc from saved rollouts — the 0.29 m/s
  "slip" is sole-site roll, not sliding. Instrumentation had to move into the
  collector. (`dataset_stats.py`)
- **The learned covariance is a stride-phase clock:** R² = 0.721 pooled (0.790
  single-rollout) on phase vs 0.020 on contact trust. (`phase_lock.py`)
- **The command envelope is not the binding constraint on diversity** — nothing
  fell on any command axis anywhere; the only unusable region is the policy's own
  deadband. (`dr_envelope.py`)
- **`v_bc` amplifies encoder noise 1414× over an analytic `J q̇`**; its noise floor
  sits at or above the median loaded contact speed, so the channel carries ~1 bit.
- **GPU wins the batch-1 float64 estimator loop 1.5×** with accuracy identical to
  three decimals, against every a-priori argument. (`bench_estimator_device.py`)
- **Per-env terrain under vmap does not recompile in MJX.** (`mjx_terrain_probe.py`)
- **`z_bias_diag.py` refuted itself:** a naive tick-wise `np.gradient` gives
  −0.0096 m/s, but that is boundary contamination; with 50-tick erosion the drift
  changes sign to +0.0007 m/s and is statistically zero. Per-tick standard errors
  understate by ~25× because 1 kHz samples within a stance are near-perfectly
  correlated.

---

## 4. `tests/` — what is actually locked in

A 1:1 port of a 176-test Java suite (`tests/inEKF/`, `tests/jointKF/`) plus
port-specific gates (everything else). Oracles are written in plain NumPy from
closed forms and never call the code under test. Green proves the **algebra** and
the **state-machine semantics**, and that gates and masks are bit-exact no-ops. It
does **not** prove the model is Alex (that is G1/G3) nor that the filters are
consistent on real gait (G10).

Only `tests/replay/` needs external data (a ~9 GB IHMC log plus the `ihmclog`
decoder) and it **skips cleanly**. `tests/sim/test_collect*.py` and
`test_estimator_loop.py` are `@pytest.mark.slow`; two `test_ghost.py` tests need
EGL + ffmpeg.

**The decisive oracles:**

- `tests/jointKF/test_stacked_oracle.py` — the stacked Joseph update ≡ the
  nuisance-`ω_base`-marginalised raw-gyro reference KF. 12 trials, tol 1e-5.
- `tests/inEKF/test_propagate.py` — asserts as an **inequality** that dropping
  `Ad_X̂` changes the answer (the §6 trap).
- `tests/pipeline/test_process_socket_wiring.py` — Σ_C lands in `Q_d` as exactly
  `R̂ Σ_C R̂ᵀ Δt` (rtol 1e-12), using a **strongly anisotropic** Σ_C so a frame
  error cannot pass.
- `tests/model/test_sole_frame.py` — reaches the sole offset by two independent
  routes, with `FOOT_BOX_SOLE_CLEARANCE = 0.0055` written as a literal so a
  correlated mis-transcription cannot pass.

**Coverage gaps the tests admit about themselves** — this is the section to read:

- **I6 is exercised nine times but constrained twice.** On three of four `SHAPES`,
  block-diagonal `R_g` equals `LΣLᵀ` to 2.6e-20.
  `test_isotropic_single_pair_cannot_constrain_i6` asserts the degeneracy on
  purpose.
- **`test_encoder_nis.py` names itself a candidate false-pass.** Substituting the
  posterior `S` for the prior moves the NIS mean 1.000 → 1.042 against a 4σ
  envelope of 0.089 — the χ² test is structurally blind to it.
- **`test_sole_frame.py` documents a test that passed against a wrong
  implementation for months**: it compared the sole site against *itself*.
- **`test_mjx_model.py` corrects `CLAUDE.md` §2's shorthand:** "armature never
  touches `M_bb`" is true only of the base six DoFs. An armed *gap* joint lands
  inside `M_bb`, so `Λ_eff = Λ + diag(rotor_j)` does not hold in general.
- **`_cache_size()` is never the I7 oracle** — three files explain that the jit
  cache is a process-global LRU which a full run evicts. The real check is lowered
  HLO or jaxpr text.
- **`test_collect.py::test_the_recorded_inekf_inputs_are_what_the_filter_consumed`
  is explicitly weaker than its neighbours**: "NOT mutation-checked… the
  assertions are written to fail under both, but that is an argument, not a
  measurement."
- **Skips that hide missing modules:** `test_transition_noise.py` and
  `test_standing_stability.py` use `pytest.importorskip`, so they vanish silently
  rather than fail if the module is absent.
- **`test_collect_dr.py::test_the_shipped_datasets_realise_only_a_handful_of_mu`
  reads `data/dr5/…` with no skip guard** — it errors on a bare clone (`data/` is
  gitignored).

---

## 5. Documentation inventory and consolidation proposal

Nothing here has been moved or deleted. This is a proposal.

| file | lines | status |
|---|---|---|
| `RUNNING.md` | 1286 | LIVE — the most-used file |
| `PORT_NOTES.md` | 4551 | LIVE (front half historical, tail live) |
| `TEST_SUITE_MAP.md` | 1312 | LIVE — binding contract |
| `CLAUDE.md` | 308 | LIVE — auto-loaded |
| `DESIGN_DECISIONS.md` | 181 | LIVE (§4 lightly stale) |
| `TODO.md` | 189 | LIVE, one stale line |
| `artifacts/RUNS.md` + `run9/experiments/*.md` | 291 | LIVE — the run contract |
| `docs/theory/*`, `docs/notes/*` | ~700 | LIVE |
| `README.md` | 17 | LIVE (thin) |
| `CONTRACT_CARD.md` | 145 | HISTORICAL — fan-out is over |
| `EXPERIMENTS.md` | 350 | HISTORICAL — the policy bug, solved |
| `branch_out.md` | 367 | HISTORICAL — Phase 0/1 executed |
| `network_plan.md` | 271 | HISTORICAL — §1 already banner'd |
| `.claude-reports/*` (5) | 2607 | HISTORICAL |
| `JOINTKF_PORT_PLAN.md` | 321 | **STALE** — says "G6–G8 not started" |
| `POLICY_DEBUG.md` | 398 | **STALE** — its conclusion is wrong; superseded by EXPERIMENTS.md |
| `TERRAIN.md` | 303 | **findings live, Stage 0 blocker false** — GPU jaxlib is installed |

**Overlaps, with the authority named:**

- `EXPERIMENTS.md` ⊃ `POLICY_DEBUG.md` — same bug, same sessions. POLICY_DEBUG's
  "sim-to-sim gap" conclusion is *wrong*. Authoritative: `EXPERIMENTS.md`.
- `branch_out.md` §3 ≡ `TODO.md` §3 — TODO.md is later and *corrects* branch_out's
  claim that `H_v` is exactly state-independent. Authoritative: `TODO.md`.
- `JOINTKF_PORT_PLAN.md` §3–4 ≡ `CONTRACT_CARD.md` §7–8 ≡ `CLAUDE.md` §5 — the
  mutation-testing rule appears three times verbatim.
- The five `.claude-reports` ⊂ the `PORT_NOTES.md` tail — the reports keep the
  *method* and the failed intermediate steps; PORT_NOTES keeps the conclusion.

**Proposal:**

1. **Split `PORT_NOTES.md` at the G10 boundary** into `PORT_NOTES.md` (Java
   reconciliation, frozen) and `EXPERIMENT_LOG.md` (the dated ContactNet/sim
   sections, still appended to). *This is the highest-value single change:* two
   documents with different lifecycles are sharing one 4551-line file.
2. **Archive to `docs/history/`, do not delete:** `POLICY_DEBUG.md`,
   `EXPERIMENTS.md`, `JOINTKF_PORT_PLAN.md`, `CONTRACT_CARD.md`, `branch_out.md`,
   `network_plan.md`, and `.claude-reports/*`. Each carries a refuted-hypothesis
   record that exists nowhere else.
3. **Fold into survivors:** `CONTRACT_CARD.md` §5–§6 → `DESIGN_DECISIONS.md`;
   still-open parts of `branch_out.md` and `TERRAIN.md` Stages 2–4 → `TODO.md`.
4. **Correct in place:** `TERRAIN.md` Stage 0 (GPU jaxlib *is* installed),
   `DESIGN_DECISIONS.md` §4 (ContactTrust *is* ported, in `sim/sensors.py`).
5. **Genuinely redundant, droppable outright:** only `docs/README.md` (39 lines)
   into `docs/index.md`.

**Must NOT be merged or moved:**

- **`CLAUDE.md` (root) and `jointKF/CLAUDE.md`** — read automatically by the agent
  harness *by path*. Moving them silently removes them from every future agent's
  context.
- **`TEST_SUITE_MAP.md`** — a contract, not documentation; cited by test docstrings.
- **`artifacts/RUNS.md` and per-run `experiments/README.md`** — RUNS.md explicitly
  requires chat-independence per run directory.
- **`docs/theory/anchor_release_timing.md`** — the only derivation of *why* the
  sink exists, and the theoretical basis for the early-release work.

---

## 6. Defects and staleness surfaced while building this map

Verified directly (I read the code):

- **`sim/collect.py:460` — `_command_schedule` never read `dr.command`.**
  `command: {enabled: false}` silently randomised anyway, while `max_speed` (the
  only other reader) returned the `abs(vx)` bound, sizing the off-field pre-flight
  for a robot walking straight. **Fixed 2026-08-01**, with two regression tests in
  `tests/sim/test_collect_dr.py` and a mutation check. No shipped dataset affected
  — every `collect_dr*.yaml` sets `enabled: true`.
- **`jointKF/build.py:376` — `rotor_inertia=rotor if not use_armature_for_rotor
  else rotor`** is a no-op ternary; the flag documented at length has no effect.
- **`contactnet/export.py` and `contactnet/__init__.py` are both 0 bytes.** The
  Java export path (plan §7, the cross-language oracle) does not exist, while
  `normalize.py`, `features.channel_names` and `rollout.contact_factors` all
  document their layouts as "what `export.py`'s manifest indexes against".

Reported by the survey and worth confirming before acting:

- `inEKF/propagate.py`'s module docstring contradicts its own `build_Qd` twice: on
  frame (world vs body Σ_C) and on whether `Q_d` is the exact integral or the
  first-order form. `build_Qd` implements the first-order form and says so.
- `jointKF/anchors.py` says "this port follows Java" on `R_anchor`; `measure.py`
  deliberately does not, with a 12/12-trial oracle failure justifying it.
- `robot.py` cites `jointKF/CLAUDE.md` §2/§3b/§7/§9 and states `Q_a = σ_τ² M⁻²`,
  contradicting the Schur-complement form in `CLAUDE.md` I1.
- `inEKF/__init__.py` says reseed is "deferred by decision"; it was implemented
  2026-07-30 and `__init__.py` exports none of its symbols.
- `correct.contact_update`'s return annotation is a 2-tuple; it returns 3.
- `filter.run` calls `init_carry(state)` with no `reseed=`, so the trajectory
  driver cannot drive the reseed path even when `ekf.reseed` is set.
- `run_estimator.py --max-backlog-ticks` help says "default 5"; the default is 2.
- `main_estimator.FusedSensors.contact_chol` asserts `N == K`, false whenever
  `toe_heel=True` (N=4, K=2).
- `CLAUDE.md` §2b describes `config/alex_jointkf.yaml` and `config/alex_inekf.yaml`;
  the repo has a single `config/filter_cfg.yaml`.
- Dangling references: `Z_BIAS_FACTS.md` (cited 4×, never committed),
  `tests/experiments/test_slip_probe.py` (cited as guarding a duplication; the
  directory does not exist).
- `sim/terrain.py`'s `VSCALE = 0.005` is defined with an explanatory comment and
  never used.

One survey claim I checked and **rejected**: the causality oracle for
`features.window_indices` was reported as unwritten, but
`tests/contactnet/test_dataset.py:193` has
`test_window_indices_are_causal_and_never_clamp`. It tests
`dataset._segment_window_indices`, so the coverage may be narrower than PORT_NOTES
specifies, but it is not absent.
