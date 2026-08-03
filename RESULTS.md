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
- **Training run: <FILL>.**
- **Held-out validation (learned vs analytic baseline): <FILL>.**

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
- check 2 (contact_meas_chol zero): **PASS**.
- check 3 (no filter state in features/online): false-positive on `state.prev_p`/
  `state.n`/`state.buf` — the `OnlineState` ring buffer (sensor-history only,
  §7-legal), identical to the reference online.py. Genuine leakage grep (InEKF
  X/P/pose/q̂) is clean.
- check 4 (F=30/d_in=600/stride=1 + order): **PASS**.
- check 5 (test_online): **RED** — the stale gate above.
- check 6 (every channel floored): **PASS**.

---

## Training run

<FILL: steps run, wall clock, loss first→last, final NIS/dof, reseeds rate,
log path results/summary.json, plot results/training.png>

## Held-out validation (learned Σ_C vs analytic-heuristic contact_chol)

<FILL: per held-out rollout — body-frame velocity RMSE learned vs baseline,
velocity NEES (target 3), contact NIS/dof (target 1). plot results/validation.png>

---

## Deliberate choices, with reasons

- **1 kHz collection** — coherence with `dt=1e-3` / `warmup_ticks=16000` / filter
  config dt (all 1 kHz); CONTROL_DT unchanged so policy behaviour is identical.
- **Normalization fit on the walking (train) rollouts, post-warmup region** —
  `dataset.fit_normalization(usable_only=True)`; `floored=[]` confirms no
  walking-active channel was floored (the failure mode the plan flagged).
- **`objective="l2_velocity"`** — CoCo Eq. 9, the process-socket run-1 objective.
- **episode_s** — <FILL with the value used and why, tied to rollout length>.

## Env postmortem

<FILL one paragraph>

## Suggested follow-ups (kept out of this diff)
- Fix the stale `test_online.py` gate (`2*J_SUB → 3*J_SUB`, populate `encoders_vel`).
- Reconcile take-two's 200 Hz `run_policy` with the 1 kHz filter/config, or make
  the ContactNet stack rate-parametric, so collection needn't monkeypatch `rp.DT`.
- Terrain/DR diversity in collection (dropped for the flat-only overnight port).
- verify.sh checks 1 and 3 are over-broad against this repo (fork base, `state.`
  regex vs the OnlineState ring buffer); tighten if it is to be a real CI gate.
