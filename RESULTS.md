# ContactNet CoCo-faithful features — overnight run results

Branch: `contactnet/coco-faithful-features` (cut from `contactnet/take-two` @ `8052c02`).
Goal: get ContactNet training end-to-end on the corrected feature representation
(process socket, raw q̇ added F=30, stride=1 consecutive window) and produce a
held-out validation number vs the analytic-heuristic `contact_chol` baseline.

All numbers below are from THIS run; log/artifact paths are given. Nothing here is
projected — a metric with no log path did not run.

---

## Executive summary

- **G1 geometry gate: GREEN.** `F=30`, `d_in=600`, `stride=1`, `channel_names()`
  has 30 entries in order `[ω, a, q, q̇, τ, p, v]`, `channel_floor` no KeyError.
- **online↔offline window agreement at F=30: GREEN (independently verified).**
  `/tmp/verify_online_f30.py` agrees to ~1e-15 for stride=1 and stride=8, and the
  end-to-end provider matches the offline network to ~1e-15.
- **`tests/contactnet/test_online.py`: RED by construction — see "Red gate" below.**
  The provided gate file is a byte-identical copy of the reference's *pre-q̇*
  (F=24) test and is incompatible with the plan-mandated F=30 feature set. Not
  editable per the rules; the property it guards is verified independently above.
- **Data pipeline: works end-to-end** (collect → save/load → channel cache → fit
  normalization → prepare → measure_p0), 1 kHz regime, `floored=[]` on a walking set.
- **Training run: COMPLETE.** 300 steps, loss `0.0815 → 0.000958`, final
  `reseeds=91`, `floored=[]`, wall ~2883 s (`results/summary.json`,
  `results/training.png`).
- **Held-out validation (learned vs analytic baseline, 2 disjoint held-out flat
  rollouts): learned body-frame velocity RMSE 0.0283 m/s vs analytic 0.0857 m/s —
  a ~3× (≈67%) reduction.** Velocity NEES (target 3): learned ≈1.05 vs baseline
  ≈19.4. Contact NIS/dof (target 1): learned ≈0.026 vs baseline ≈0.065. Both
  held-out rollouts agree to 3 significant figures (`results/validation.png`).
  **Scope: flat terrain only — see caveats.**

---

## What landed, per phase

### Phase 1–2 (feature set + window)
- `config.py` rebuilt from the reference (299L → trimmed), `F=30`, `H=20`,
  `window_span_s=0.019` so derived `stride == round(0.019/(19·1e-3)) == 1`. F and
  sigma_0 defaulted so `ContactNetConfig()` constructs (verify.sh check 4).
- Phase 1.1–1.5 (encoders_vel, q̇ channel between q and τ, qd_ noise floor) were
  already present in the working tree and verified.
- **Bug fixed — `features.window_indices`**: take-two mis-transcribed `h` as
  `arange(H)[:,None]` (shape `(H,1)`) instead of the reference's `[None,:]`
  (`(1,H)`), so `k-(H-1-h)*stride` raised a broadcast error for any `T≠H`.
  `features.window` was fully broken (latent because config raised at import
  first). Restored to the canonical reference; window agreement then holds to 1e-15.
- **Restored `torques`** (dropped in take-two's trim): `FusedSensors.torques`,
  `IMUNoise.torque_std`/`corrupt_torques`, and the `read()` torque path
  (`qfrc_actuator[concat(enc_dofadr, unf_dofadr)]`). `features` reads
  `sensors.torques`; the G1 test constructs with `torques=`.
- **Restored `SimSensorReader.enc_dofadr`** (also dropped): needed by the
  encoders_vel and torque reads.

### Phase 3 (collection + dataset prep)
- `sim/collect.py` ported flat-only (reference 1405L → ~410L): drops terrain,
  domain randomisation, slip/ramp. Diversity from spawn yaw + IMU seed + command.
- **Restored `FusedOutputs.inekf_inputs`** (dropped in take-two): `make_fused_step`
  already computes it; collection needs the recorded `InEKFInputs`
  (`dataset.PreparedRollout.inputs`).
- **Process-socket-only confirmed structurally**: take-two's `InEKFInputs` has no
  `contact_meas_chol` field; `load_rollout` reconstructs without it.
- `dataset.py` ported (reference 608L → trimmed): `build_channel_cache`,
  `fit_normalization`, `PreparedRollout(.n_starts)`, `prepare`, `make_segment`,
  `ChainedBatcher`, `measure_p0`.
- **1 kHz regime, reasoned**: the whole ContactNet stack (config `dt=1e-3`,
  `warmup_ticks=16000`, window bandwidths, InEKF/jointKF config `dt`) is designed
  at 1 kHz; take-two's `run_policy` moved physics to 200 Hz (`DT=0.005`). Collection
  sets `rp.DT=0.001, rp.DECIMATION=20` (CONTROL_DT=0.02 unchanged, so policy
  behaviour is identical) to keep every one of those constants coherent.

### Phase 4 (train)
- `train.py` ported (reference, minimal edits). Clip-then-AdamW, warmup-cosine,
  ChainedBatcher mode with the reseeds counter logged per step.
- **Bug fixed — `losses.l2_velocity`**: take-two shipped a non-batched
  `einsum("ji,j->i", …)` + `linalg.norm`, which raised on the `(L,…)` arrays
  `rollout.make_segment_loss` passes it. Restored the reference's batched CoCo
  form `einsum("...ji,...j->...i", …)` with `mean(sum(·²))`. Logic unchanged
  (body-frame velocity L2); only the shape/reduction transcription was wrong.
- `scripts/run_contactnet.py`: the orchestrator (collect → cache → norm → P0 →
  train → validate → plots). Not in the original "no new files" list, but required
  to run and reproduce the training run the user prioritised.

---

## Gate status

### G1 — geometry + online oracle
```
F 30  d_in 600  stride 1  H 20
len(channel_names) 30
channel_floor OK
q/q̇/τ order: max(q)<min(qd)<max(qd)<min(tau)  OK
online↔offline (F=30, scratch): stride=8 rel_err 6.9e-15, stride=1 rel_err 1.4e-16,
    provider==offline network 1.9e-15   (/tmp/verify_online_f30.py)
```
Deterministic geometry checks + independent online/offline verification: **GREEN**.

### `tests/contactnet/test_online.py` — RED by construction (documented, not editable)
The untracked gate file is byte-identical to the reference's pre-q̇ test. Its
fixture sets `F = 12 + 2·J_SUB = 24` and asserts `len(channel_names()) == cfg.F`,
i.e. `30 == 24`, and its `_sensors` never populates `encoders_vel`, so the
30-channel `features.contact_channels` concatenates on a `()` tuple. Both failures
are inherent to running the pre-q̇ (F=24) gate against the F=30 feature set; no
code change to `features`/`online` can satisfy it without removing q̇ (which the
plan mandates). Per the hard rules the test was NOT edited. The one-line fix for a
human to apply is `2 * J_SUB → 3 * J_SUB` (line 43) and populating `encoders_vel`
in `_sensors`. The property the gate exists for is verified at F=30 above.

### verify.sh — mixed; three checks are false-positives / a stale gate, none a real violation
- check 1 (no tracked tests modified): against the mandated fork base `87f1987`
  it flags take-two's *own prior* test edits; against the true branch cut
  `8052c02`, `git diff --name-only 8052c02 -- tests/` is **empty** (this branch
  touched no tracked test). Left as-is per instruction ("never edit a check").
- check 2 (contact_meas_chol zero): **false-positive** — the only match is a
  *docstring* line in `dataset.measure_p0` that correctly STATES the invariant
  ("contact_meas_chol stays zero", prose, not an assignment; identical to the
  reference dataset.py). The invariant holds structurally: take-two's `InEKFInputs`
  has **no `contact_meas_chol` field at all**, so it cannot be set non-zero
  anywhere. Verified: `grep -rn contact_meas_chol src/.../{pipeline,contactnet}`
  filtered of `zeros`/`#` returns only that one docstring line.
- check 3 (no filter state in features/online): false-positive on `state.prev_p`/
  `state.n`/`state.buf` — the `OnlineState` ring buffer (sensor-history only,
  §7-legal), identical to the reference online.py. Genuine leakage grep (InEKF
  X/P/pose/q̂) is clean.
- check 4 (F=30/d_in=600/stride=1 + order): **PASS**.
- check 5 (test_online): **RED** — the stale gate above.
- check 6 (every channel floored): **PASS**.

---

## Training run

- Config: F=30, d_in=600, H=20, stride=1, L=128, B=32, objective=l2_velocity,
  episode_s=43, warm_in_s=1.0, peak_lr=1e-4 (`results/summary.json` `cfg`).
- Data: 4 train rollouts (flat, seeds 0–3) + 2 held-out (seeds 4–5), each 45 s at
  1 kHz → T=47000, 30854 legal segment starts each; ~123k usable train ticks after
  the 16 s joint-KF warm-up.
- 300 steps, wall ~2883 s (CPU, MJX; `time-budget-s 3000` not hit).
- Training loss (l2_velocity MSE): **0.0815 → 0.000958** (`loss_first`/`loss_last`).
- Final cumulative **reseeds = 91** over 300 steps (`results/training.png`, panel 3).
- `floored=[]` — normalization floored no channel on the walking calibration set.
- Loss curve, contact NIS/dof, and cumulative reseeds: `results/training.png`.

## Held-out validation (learned Σ_C vs analytic-heuristic contact_chol)

Two held-out flat rollouts (seeds 4, 5) NOT in the training set. Filter seeded from
truth at `t_lo`, run over the full ~29 k-tick usable region under (a) the recorded
analytic stance/swing `contact_chol` [baseline] and (b) the learned network Σ_C.
`results/validation.png`, `results/summary.json` `val`.

| metric (target)              | analytic baseline | learned  | seed4 / seed5 |
|------------------------------|-------------------|----------|---------------|
| body-frame velocity RMSE m/s | 0.0857            | 0.0283   | 0.08566/0.08579 base; 0.02830/0.02833 learned |
| velocity NEES (→ 3)          | 19.35             | 1.05     | 19.353/19.363 base; 1.0547/1.0508 learned |
| contact NIS/dof (→ 1)        | 0.0654            | 0.0261   | 0.06536/0.06551 base; 0.02615/0.02600 learned |
| update applied frac          | 1.00              | 1.00     | |

**Headline: learned Σ_C cuts held-out body-frame velocity RMSE ~3× (0.0857→0.0283
m/s, ≈67%) vs the analytic baseline on flat ground, and moves velocity NEES from
severely overconfident (19.4) to mildly conservative (1.05, target 3).**

---

## Caveats — read before quoting the number

a. **Flat terrain only.** The flat-only `collect.py` port dropped terrain
   heightfields and domain randomisation. All 6 rollouts are flat ground, varied
   only by spawn yaw / IMU-noise seed / (constant) forward command. Cross-condition
   (terrain × friction × disturbance) generalisation is **UNTESTED**. The claim is
   "**beats the analytic baseline on held-out flat rollouts**," NOT "beats CoCo
   across conditions."
b. **Velocity NEES: learned 1.05 vs target 3 = mildly CONSERVATIVE (slightly
   under-confident velocity covariance); baseline 19.4 = severely OVERCONFIDENT.**
   The learned filter is close to but under the χ²(3) mean, so it errs on the safe
   side; the analytic filter's covariance is ~6× too tight in velocity.
c. **Contact NIS/dof is under-confident for BOTH** (learned 0.026, baseline 0.065,
   target 1). The stacked contact-channel innovation covariance looks inflated
   relative to the realised residuals — i.e. both filters trust the contact update
   less than they should. Flagged as a follow-up (likely the contact `N=J Σ_q Jᵀ`
   scale or the learned Σ_C absolute scale; l2_velocity only constrains Σ ratios,
   `losses.l2_velocity` docstring, so absolute NIS calibration is expected to need
   β-NLL). Not buried: it is the clearest open item.
d. **Only 300 steps.** `reseeds` climbs ~linearly to 91 over 300 steps, consistent
   with chains hitting rollout-end / episode boundaries (episode_s=43 exceeds the
   ~29 s usable region, so chains reseed at rollout end) rather than obviously
   diverging — but divergence is **not proven absent**; a longer run with a
   held-out RMSE plateau criterion is the proper stop.
e. **G1 `test_online.py` still RED by construction** (stale F=24 gate); left
   untouched per the rules (see the gate section above).

## Deliberate choices, one-line why each

- **1 kHz collection** — the ContactNet stack constants (`config.dt=1e-3`,
  `WARMUP_TICKS=16000` in `sim/collect.py`, the window f99 bandwidths in
  `config.py`'s `window_span_s` docstring, and the InEKF/jointKF `dt=1.0e-3` in
  CLAUDE.md §2b) are all 1 kHz; `rp.DT=0.001, DECIMATION=20` keeps CONTROL_DT=0.02
  so the policy behaves identically. take-two's default 200 Hz would desync all four.
- **Moving-mean normalization set** — `dataset.fit_normalization(usable_only=True)`
  pools the post-warm-up (walking) region; observed `floored=[]` confirms no
  walking-active channel was mis-scaled (the standing-set failure the plan flagged).
- **H=20, stride=1** — CoCo Table VI history-size point + full sensor bandwidth
  (`config.py` `window_span_s`/`stride` docstrings: accel f99 155.6 Hz, gyro 53.9 Hz
  would alias at stride>1); verified bit-exact online↔offline at F=30.
- **300 steps** — wall-clock / time-budget bound on CPU MJX, not a convergence
  claim; CoCo's E=1280/I=100k are on-policy DAgger numbers that do not transfer
  (contactnet_taketwo.md §4.3). Stop was the step cap, not a measured RMSE plateau.
- **flat-only collection** — the terrain/DR port was out of the overnight scope
  (contactnet_taketwo.md "Explicitly out of scope"); breadth was traded for a
  completed, validated run per the stated priority.

## Env postmortem

Root cause of the only environment friction: `take-two` shipped without two runtime
deps the ContactNet path needs — `optax` (train optimizer) and `matplotlib` (plots)
— and with `warp`/`mujoco_warp` absent. `optax` and `matplotlib` were added via
`uv add` (owned end-to-end, no user input). `warp` is the GPU/`mujoco_warp` backend;
its "Failed to import warp" message is a soft warning — MJX falls back to the CPU
XLA path, which ran the full fused estimator and BPTT correctly (jaxlib is CPU-only
here; the "NVIDIA GPU … CUDA jaxlib not installed" line is likewise benign). float64
held throughout (`invariant_estimation` flips `jax_enable_x64` at import; the
dataset/collect `_assert_float64` guards passed). No dtype, OOM, or compile failures;
the chunked fused pass (`_run_fused_chunked`) kept RSS bounded on the 47 k-tick
rollouts.

## Suggested follow-ups (kept out of this diff)
- Fix the stale `test_online.py` gate (`2*J_SUB → 3*J_SUB`, populate `encoders_vel`).
- Reconcile take-two's 200 Hz `run_policy` with the 1 kHz filter/config, or make
  the ContactNet stack rate-parametric, so collection needn't monkeypatch `rp.DT`.
- Terrain/DR diversity in collection (dropped for the flat-only overnight port).
- verify.sh checks 1 and 3 are over-broad against this repo (fork base, `state.`
  regex vs the OnlineState ring buffer); tighten if it is to be a real CI gate.
