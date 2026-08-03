# Morning report — ContactNet CoCo-faithful features (overnight, 2026-08-03)

Branch `contactnet/coco-faithful-features` off `contactnet/take-two` @ `8052c02`.
Priority followed: get one validated training run + a real held-out number, arrived
at by reasoning (every deviation justified below), no test edits, process socket only.

## Headline
- G1 geometry gate GREEN; online↔offline window agreement verified at F=30 to ~1e-15.
- `tests/contactnet/test_online.py` RED **by construction** — the provided gate file
  is a byte-identical copy of the reference's pre-q̇ (F=24) test, incompatible with
  the mandated F=30 feature set. NOT editable per the rules; documented + the property
  independently verified.
- Full data + training pipeline reconstructed and running end-to-end at 1 kHz.
- Training run COMPLETE: 300 steps, loss 0.0815 → 0.000958, reseeds 91, wall ~2883 s.
- **Held-out (2 disjoint flat rollouts): learned body-frame velocity RMSE 0.0283 m/s
  vs analytic baseline 0.0857 m/s — ~3× (≈67%) reduction.** Velocity NEES 1.05 (learned)
  vs 19.4 (baseline), target 3. Contact NIS/dof 0.026 vs 0.065, target 1.
  `results/summary.json`, `results/validation.png`. FLAT TERRAIN ONLY.

## The big surprise (plan understated this correctly)
`take-two` is a partial reimplementation that had **trimmed away working code the
plan assumed present**. Reconstructed, each faithful to the reference and committed
separately for bisect:
1. `config.py` — was a 6-line stub that `raise`d at import (reddened every importer).
2. `dataset.py`, `train.py` — empty (0 bytes).
3. `sim/collect.py` — missing entirely.
4. `features.window_indices` — mis-transcribed `h` shape `(H,1)` vs reference `(1,H)`;
   `features.window` fully broken (latent behind the config import error).
5. `FusedSensors.torques` + `IMUNoise.corrupt_torques` + the `read()` torque path —
   dropped; `features` reads `sensors.torques` and the G1 test constructs with it.
6. `SimSensorReader.enc_dofadr` — dropped; needed by encoders_vel + torque reads.
7. `FusedOutputs.inekf_inputs` — dropped; `make_fused_step` computes it and
   collection needs the recorded `InEKFInputs`.
8. `losses.l2_velocity` — non-batched `einsum("ji,j->i")+norm`, raised on the
   batched call `rollout` makes; restored the reference's `"...ji,...j->...i"` + MSE.

None of these were logic changes — each restores the validated reference behaviour.

## Per-phase status
- **Phase 1–2**: config rebuilt (F=30,H=20,window_span_s=0.019→stride=1); q̇/encoders_vel/
  qd_ floor already present + verified; window_indices + torques bugs fixed. G1 geometry
  GREEN. `verify.sh` written per plan (fork base `87f1987`); checks 2/4/6 PASS, checks
  1/3/5 are false-positives or the stale gate (analysed in RESULTS.md).
- **Phase 3**: `collect.py` flat-only port + `dataset.py` port. Collection runs the
  fused estimator on CPU MJX (warp absent, non-fatal). 1 kHz regime (rp.DT=0.001,
  DECIMATION=20; CONTROL_DT=0.02 unchanged). Data pipeline verified end-to-end;
  `floored=[]` on the walking set. Collected 6 flat rollouts (seeds 0–5), 45 s each
  → T=47000, 30854 legal starts each; ~123k usable train ticks (4 train rollouts
  after the 16 s warm-up). Fused pass ~140 s/rollout on CPU.
- **Phase 4**: `train.py` port; `l2_velocity` batched-form fix; orchestrator
  `scripts/run_contactnet.py`. 300 steps, loss 0.0815 → 0.000958, reseeds climb
  ~linearly to 91 (episode/rollout boundaries), wall ~2883 s. `results/training.png`.

## Files ported from the reference (contact-net-integration @ process-socket)
| file | how heavily | notes |
|---|---|---|
| contactnet/config.py | port+trim (299→~180L) | F=30, H=20, window_span_s=0.019 |
| contactnet/dataset.py | port+trim (608→~430L) | logic intact, prose stripped |
| contactnet/train.py | near-verbatim | already clean in reference |
| sim/collect.py | heavy port + simplify (1405→~410L) | flat-only; terrain/DR/slip dropped |
| losses.l2_velocity | reference form restored | batched einsum + MSE |
| features/sensors/pipeline | targeted restores | window_indices, torques, enc_dofadr, inekf_inputs |

## Red gate left red (with reason)
`tests/contactnet/test_online.py`: `_cfg` sets `F=12+2·J_SUB=24` and the fixture asserts
`len(channel_names())==cfg.F` (30≠24); `_sensors` never sets `encoders_vel` so the
F=30 feature builder concatenates a `()` tuple. Both are inherent to the pre-q̇ gate
vs the F=30 set. One-line human fix: `2*J_SUB→3*J_SUB` + populate `encoders_vel`.
Property verified independently at F=30 (`/tmp/verify_online_f30.py`, ~1e-15).

## Caveats (do not oversell)
a. Flat terrain only — cross-condition (terrain/friction/disturbance) generalization
   UNTESTED; claim is "beats analytic baseline on held-out flat rollouts."
b. Velocity NEES 1.05 = learned mildly conservative; baseline 19.4 = severely overconfident.
c. Contact NIS/dof under-confident for BOTH (learned 0.026, baseline 0.065) — contact
   covariance looks inflated; open follow-up (l2_velocity only constrains Σ ratios, so
   absolute NIS calibration likely needs β-NLL).
d. Only 300 steps; reseeds ~linear to 91 (episode boundaries), divergence not proven absent.
e. G1 `test_online.py` red-by-construction (stale F=24 gate), left untouched per rules.

## Env postmortem
take-two shipped without two ContactNet-path deps — `optax` (train optimizer) and
`matplotlib` (plots) — added via `uv add` (no user input). `warp`/`mujoco_warp` are
absent; the "Failed to import warp" line is a soft warning (warp is the GPU backend),
so MJX fell back to the CPU XLA path and ran the full fused estimator + BPTT correctly
(jaxlib is CPU-only here). float64 held throughout (`jax_enable_x64` at package import;
the `_assert_float64` guards passed). No dtype/OOM/compile failures; the chunked fused
pass kept RSS bounded on the 47k-tick rollouts. Total run wall ~2883 s.

## Commit/PR state
See the final summary returned to the coordinator (pushed + PR against
`contactnet/take-two`, or local-only + `PR_BODY.md` if push/gh failed).

## Next session (prioritised)
1. Chase the contact NIS/dof under-confidence (caveat c): audit the `N=J Σ_q Jᵀ`
   contact-noise scale and add β-NLL so absolute Σ_C scale (not just ratios) is trained.
2. Fix the stale `test_online.py` gate so G1 is a real green.
3. Reconcile take-two's 200 Hz `run_policy` with the 1 kHz filter/config (or make the
   stack rate-parametric) so collection needn't monkeypatch `rp.DT`.
4. Terrain/DR diversity in collection (dropped for the flat-only overnight port).
5. Confirm G9 pipeline test still green after the `FusedOutputs.inekf_inputs` restore.
