# ContactNet overnight run — RESULTS

Run dir: `results/2026-08-04_00-06-04_overnight/` (`results/latest` → here)
Branch `contactnet/take-two` @ `7a5d0ed`. Backend: **GPU** (NVIDIA RTX 4070, `jax[cuda13]`).

Every number below is from `summary.json`, `history.npy`, or the per-run logs in
`closed_loop/`; paths are given inline.

---

## 1. Did it finish?

**Completed all 8000 training steps** (`summary.json:steps_run=8000`), reaching the
`--steps` limit, NOT the time budget. Training-loop wall clock ~02:44→03:35; full
process (collection→caching→normalize→train→validate) `wall_s=12831.6`
(`summary.json`). Held-out validation ran on all six modes; `params.npz`,
`history.npy`, `summary.json`, and both plots were written (03:32–03:39).

## 2. Held-out per-mode validation — learned vs analytic baseline

Source: `summary.json` → `val`. Velocity RMSE is body-frame [m/s] (lower better);
NEES is world-frame 3-DoF velocity (**target 3**); NIS/dof is contact (**target 1**).

| mode | vel RMSE base | vel RMSE learned | learned<baseline? | NEES base | NEES learned | NIS/dof base | NIS/dof learned |
|------|------:|------:|:--:|------:|------:|------:|------:|
| forward   | 0.0939 | **0.0274** | YES (3.4×) | 23.882 | 0.740 | 0.0737 | 0.0194 |
| backward  | 0.0361 | 0.0440 | **NO (0.82×)** | 1.474 | 1.857 | 0.0100 | 0.0491 |
| lateral_L | 0.0536 | **0.0478** | YES (1.12×) | 6.398 | 1.960 | 0.0232 | 0.0530 |
| lateral_R | 0.0533 | **0.0469** | YES (1.14×) | 6.183 | 2.374 | 0.0241 | 0.0482 |
| turn_L    | 0.0367 | **0.0338** | YES (1.09×) | 2.521 | 1.132 | 0.0175 | 0.0380 |
| turn_R    | 0.0365 | **0.0322** | YES (1.13×) | 2.445 | 1.293 | 0.0177 | 0.0433 |

**Headline:** learned beats baseline on velocity RMSE in **5 of 6 modes**. The one
regression is **backward** (learned 0.0440 vs baseline 0.0361). On NEES calibration
the learned model is closer to a consistent filter in every mode (baseline forward
NEES 23.9 is severely overconfident; learned 0.74); learned NEES sits in
[0.74, 2.37] (mildly under target 3), baseline in [1.47, 23.9].

## 3. Closed-loop — ContactNet in the loop (`run_estimator.py --contactnet`)

Each mode run once, headless, **500 control ticks (10 s), deterministic, no IMU
noise**, learned `contact_chol` from `params.npz` driving the live InEKF (which
feeds `projected_gravity`/`base_ang_vel` back to the policy). Baseline = the
analytic stance/swing heuristic (`--stance-chol 1e-4 --swing-chol 1e1`). Logs +
histories in `closed_loop/cl_<mode>_<baseline|learned>.{log,npz}`. `fell` = a
non-finite state was hit during the run. Errors are estimate-vs-sim-truth.

| mode | vErr rms base | vErr rms learned | tilt tail base | tilt tail learned | pErr rms base | pErr rms learned | fell (either)? |
|------|------:|------:|------:|------:|------:|------:|:--:|
| forward   | 0.0876 | **0.0489** | 0.511 | 0.160 | 0.403 | 0.111 | no |
| backward  | 0.0494 | 0.0776 | 0.196 | 0.256 | 0.032 | 0.215 | no |
| lateral_L | 0.0635 | 0.0956 | 0.333 | 0.370 | 0.190 | 0.269 | no |
| lateral_R | 0.0616 | 0.1739 | 0.338 | 0.454 | 0.183 | 0.667 | no |
| turn_L    | 0.0445 | 0.0617 | 0.311 | 0.376 | 0.103 | 0.103 | no |
| turn_R    | 0.0439 | 0.0526 | 0.339 | 0.343 | 0.104 | 0.149 | no |

vErr [m/s], tilt [deg], pErr [m]; "tail" = rms over the last half.

**In closed loop the learned model beats baseline only in `forward`** (vErr 0.56×,
pErr 0.28×, tilt-tail 0.31×). In the other five modes baseline is better (learned
vErr 1.2×–2.8× worse; worst is `lateral_R` at 2.82×). **No run fell** in either
configuration. This differs from the open-loop per-mode result in §2, where learned
won 5/6 — i.e. the gain the offline replay shows on recorded rollouts does not carry
into the closed loop except for forward walking.

Confirmed at a longer horizon (**1500 ticks / 30 s**, `closed_loop/cl30_*`): forward
vErr baseline 0.0884 → learned 0.0458 (0.52×); lateral_R baseline 0.0607 → learned
0.1420 (2.34×). Same direction as the 10 s runs, so this is not a short-horizon
artifact.

## 4. Training curve

Source: `history.npy` (cols: loss, |g|, NIS/dof, applied, reseeds), `training.png`.

- loss: first **0.14416** (step 0) → step 500 **0.00692** → **min 0.000991 @ step 4613** → last **0.002065** (step 7999)
- `applied` fraction: **1.00 at every step** (min = 1.0); no steps skipped by the non-finite guard
- reseeds: 1 → **1487**, monotone/smooth (no spikes)
- contact NIS/dof: 2.86 (step 0) → ~0.02–0.06 for the bulk of the run
- **No NaN/Inf** in loss at any step

Milestones: step 100 loss 0.671 (LR still ramping, `--warmup-steps` 50); step 1000
0.00376; step 4000 0.00229; step 6000 0.00304.

## 5. Anomalies / monitoring notes

- **Launch gate (log `results/overnight.log`):** `cfg: F=30 d_in=600 H=20 (window
  ... consecutive ticks) L=128 B=32 objective=l2_velocity`; **`floored channels: []`**
  (empty — no channel floored, none of the moving channels flagged); step-0 loss
  finite (0.14416). All three pass.
- No divergence: `applied` never dropped below 1.0; no reseed spike; no non-finite
  loss (monitored every ≤5 min across the whole run).
- Closed-loop numbers are **single deterministic 10 s walks per mode**, not
  seed-averaged.

## 6. Deviations from the original prompt (all per explicit user instruction)

The prompt's premise ("collection + caches done, pure-training run, don't touch
code") did not hold on disk; the user authorized the changes below.

- **Data was incomplete.** The prior smoke was Ctrl-C'd mid-collection: only train
  seeds 0–26 existed, no val seeds, no caches. Collection was re-run (train 0–31 +
  val 900–905, all 60 s). Caches are rebuilt every run by `dataset.build_channel_cache`
  regardless — nothing was pre-existing.
- **Validation modes.** `VAL_MODES` in `scripts/run_contactnet.py` was hardcoded to
  seeds 4–7 (inside the train range), so the prompt's `--val-seeds 900–903` would
  have produced four `"mixed"` rollouts and no per-mode data. Per the user, it was
  extended to the six held-out motions on disjoint seeds **900–905**
  (forward/backward/lateral_L/lateral_R/turn_L/turn_R). Diff: `run_contactnet.py`
  (+10/−9).
- **GPU.** Switched from CPU fallback to GPU via the project's own
  `gpu = ["jax[cuda13]"]` extra (`uv sync --extra gpu`).
- **Closed-loop wiring.** Added a `--contactnet PARAMS.npz` path to `run_estimator.py`
  (+107): a `ContactNetRuntime` that runs `contactnet.online.make_provider` as a scan
  over the substep batch and splices the learned `contact_chol` into the fused step
  (network output depends only on sensors, so the fused step is otherwise untouched).
  Config is `ContactNetConfig()` defaults (F=30, H=20) — the config this run trained
  under. Includes a direct npz load for the norm constants (the run's
  `save_norm` omits the `n_ticks`/`source` provenance fields that `normalize.load`
  requires; both are unused by inference).

## 7. Artifacts

- `summary.json` — per-mode validation, cfg, args, wall time
- `params.npz` — trained ContactNet weights (1.77 MB)
- `history.npy` — (8000, 5) training history
- `norm_constants.npz`, `P0.npy` — normalization + initial covariance
- `training.png`, `validation.png` — training curve; per-mode base-vs-learned bars
- `closed_loop/cl_<mode>_<baseline|learned>.{log,npz}` — 12 closed-loop runs
- Training log: `results/overnight.log`
