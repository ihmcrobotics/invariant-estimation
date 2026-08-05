# ContactNet — Overnight Plan: get a working training run on the CoCo-faithful feature set

**Goal (one sentence):** get ContactNet training end-to-end on the corrected feature representation — **process socket, raw q̇ added, `stride=1` consecutive window** — and produce a held-out validation number proving the new branch trains and at least matches the analytic baseline. This is a *pipeline de-risking* run, not a bid to beat CoCo.

**Definition of done by 9 AM (in priority order):**
1. All code edits below applied; `tests/contactnet/test_online.py` green (online↔features agreement ≈1e-13).
2. Rollouts re-collected with q̇, channel caches rebuilt at F=30, normalization refit on a **moving** set.
3. One training run completes on the new feature set with a decreasing loss.
4. `RESULTS.md` with held-out velocity RMSE + NEES vs. the analytic (heuristic `contact_chol`) baseline.

If (1)–(3) land and (4) shows *parity or better* vs. the analytic filter, the night is a success. Beating CoCo is explicitly **out of scope** (see "Realistic ceiling").

---

## Working discipline — read first

**Branch + PR. Never commit to the working branch directly.**
- Cut a fresh branch off the current one (e.g. `contactnet/coco-faithful-features`). Do all work there.
- Commit in small, labeled steps — roughly one phase or one file per commit — so a reviewer can bisect.
- End the run by opening a **draft PR** (do not merge, do not push to `main` or the existing working branch). The PR description maps commits → the phases below and pastes the gate outputs (`test_online`, shape/`d_in` checks) and the `RESULTS.md` numbers.
- Anything risky or uncertain goes in its own commit labeled `[NEEDS REVIEW]`, not folded into a working commit.

**Simplest thing that works. No architectural changes.**
- The tasks below are the whole job. Do them and stop. Do **not** refactor, rename, "clean up," restructure modules, add abstractions, introduce dependencies, or change any interface beyond what a task literally specifies.
- Minimal diff: the smallest edit that satisfies the task and passes its gate. If a task is 3 lines, do not write 30. No new files except `RESULTS.md`.
- If you think a larger change is warranted, do **not** make it — write it under "Suggested follow-ups" in `RESULTS.md` and leave the code minimal. A smaller correct diff is always preferred over a larger clever one.

**No unverified claims. Hard rule — this is the main failure mode to guard against.**
- Every non-trivial statement in a comment, commit message, or `RESULTS.md` must be backed by exactly one of: a line reference in this repo, a passing test, or an equation/section in CoCo (arXiv 2605.15122). If you cannot cite it, do not write it.
- Do not invent file names, function names, config keys, test names, tolerances, or numbers. If you need one, `grep` for it first; if it does not exist, say so explicitly rather than fabricating a plausible-looking one.
- `RESULTS.md` reports **what ran and what was measured**, with log paths — not a narrative of what probably happened. "Val RMSE = X over N steps (log: …)" is allowed; "the network learned to detect slip" is not, unless a metric shows it.
- If a gate is red, report it red and stop. **Never** edit a test, loosen a tolerance, or delete an assertion to get green. Modifying anything under `tests/` to pass is prohibited; if a test looks wrong, note it and stop.

---

## Hard invariants — DO NOT violate these while "fixing" anything

These are correctness properties, not preferences. If a change would break one, stop and leave a note in `RESULTS.md` instead.

- **Process socket only.** ContactNet output drives `InEKFInputs.contact_chol` (the contact-anchor random-walk block of `Q_d`, per CoCo Alg. 1 line 1 / Eq. 5). `contact_meas_chol` **stays zero everywhere** — do NOT re-introduce a measurement-noise `Σ_C` term. CoCo's measurement noise is `N = J_{C_i} Σ_q J_{C_i}ᵀ` only (Eq. 8).
- **Features are filter-state-free.** No filter estimates (state, covariance, joint-KF `q̂`/`q̇̂`) may enter the network. Raw IMU, FK from **raw** encoders, **raw sensor** q̇. This preserves the InEKF's group-affine invariance; violating it silently breaks the filter while the code keeps running.
- **One canonical channel order, identical in every file:** `[ω(3), a(3), q(J_sub), q̇(J_sub), τ(J_sub), p(3), v(3)]` → CoCo's `o = (ᴮω, ᴮa, q, q̇, τ, ᴮp_{B→Ci}, ᴮv_{B→Ci})`. Order must match byte-for-byte across `features.channel_names()`, `features.contact_channels`, `online.channels_now`, and the normalization artifact.
- **float64 everywhere** at and below the dataset boundary. `_assert_float64` guards it; do not downcast.
- **Do not touch** `losses.py` or `network.py` logic (validated) beyond what's needed to wire the new `d_in`. Use `objective="l2_velocity"` (CoCo's Eq. 9).
- **Do not reseed every segment from ground truth** (force-teacher). CoCo §III-B shows it degrades learning; `ChainedBatcher`'s carried `(X̂,P)` is correct — leave it.
- **Do not** switch to per-window normalization, and **do not** re-enable the boxcar/`stride>1` window.

---

## Phase 1 — Feature set: add raw q̇ (F: 24 → 30)

Parallelizable across agents where noted. Target: `F = 12 + 3·J_sub = 30` (J_sub = 6).

**1.1 `pipeline/main_estimator.py` — `FusedSensors`** *(prereq for everything)*
Add the field (defaulted like `torques` so existing construction sites don't break):
```python
encoders_vel: Array = ()
```

**1.2 `sim/sensors.py` — `SimSensorReader.read()`**
Raw filtered-joint velocity. `enc_dofadr` already exists (used by the torque line). Use the **dof** index array, not `enc_qadr`:
```python
enc_vel = d.qvel[self.enc_dofadr].copy()          # after: qd_u = d.qvel[self.unf_dofadr].copy()
# in the `if self.noise is not None:` block, alongside qd_u:
enc_vel = self.noise.corrupt_velocities(enc_vel)
# in the FusedSensors(...) return:
encoders_vel=enc_vel,
```

**1.3 `contactnet/features.py` — `contact_channels` + `channel_names`** *(parallel with 1.2)*
```python
# contact_channels, next to the tau width-check:
qd_all = jnp.concatenate([sensors.encoders_vel, sensors.qd_unfiltered], axis=-1)
if qd_all.shape[-1] != q_all.shape[-1]:
    raise ValueError(f"qd width {qd_all.shape[-1]} != concat(q) width {q_all.shape[-1]}")
qd_sub = qd_all[:, subchain]                       # (T, N_c, J_sub)

# final concat — q̇ BETWEEN q and τ:
return jnp.concatenate([omega, accel, q_sub, qd_sub, tau_sub, p, v], axis=-1)
```
```python
# channel_names(): insert between the q and tau blocks
*(f"q_{s}"   for s in joint_labels),
*(f"qd_{s}"  for s in joint_labels),   # NEW
*(f"tau_{s}" for s in joint_labels),
```
Update the docstring: `F = 12 + 3·J_sub = 30`. **Leave the `p` line unchanged** — `kinematics(q, zeros).y` is position-only (q̇-independent); the zeros are the unused velocity arg.

**1.4 `contactnet/online.py` — `channels_now`** *(must mirror 1.3 exactly)*
```python
qd_all = jnp.concatenate([sensors.encoders_vel, sensors.qd_unfiltered], axis=-1)
qd_sub = qd_all[subchain]
row = jnp.concatenate([omega, accel, q_sub, qd_sub, tau_sub, p, v], axis=-1)
```
Field name is **`encoders_vel`** (plural) everywhere — a singular typo throws `AttributeError`.

**1.5 `contactnet/normalize.py` — `NOISE_FLOOR`**
```python
"qd_": 5.0e-3,   # rad/s — = sim encoder_vel_std; stack convention (jointKF qdR min = encoder_vel_std²)
```
Prefix is collision-free (`"qd_hip_x".startswith("q_")` is False). Without this, `channel_floor` raises `KeyError` by design.

**1.6 `contactnet/config.py`**
`F` is a hand-set field, not derived: `F: int = 30`. (`d_in = F*H` is a derived property — it updates itself.)

---

## Phase 2 — Window revert: `stride = 1`, consecutive ticks (config-only)

Rationale for the agent: CoCo Table VI (history-size ablation) shows shorter, full-bandwidth windows win; the informative slip signal is in the fast IMU channels (accel f99 155.6 Hz, gyro 53.9 Hz), which decimation at `stride=8` aliases (Nyquist 62.5 Hz). `stride=1` preserves them and makes CoCo's Table V/VI a valid reference.

**2.1 `contactnet/config.py`**
```python
H: int = 20                 # CoCo Table VI point (was 50)
window_span_s: float = 0.019   # ≈ (H-1)*dt so the derived `stride` property rounds to 1
```
At `stride=1`: `features.boxcar` short-circuits to identity, `window_indices` returns consecutive ticks, `valid_start_range` lead-in shrinks to `H-1`. No code edits needed in `features.py`/`online.py`/`dataset.py` — all geometry is cfg-derived.

**2.2 (optional, bit-exactness)** add an `s == 1: return x` early-return to `online.py`'s ring-buffer boxcar so it matches `features.boxcar` exactly rather than to ~1e-13. Non-blocking.

**Gate G1 (after Phase 1–2):**
- `python -m pytest tests/contactnet/test_online.py` → green.
- `len(features.channel_names()) == 30`; `config.d_in == 600`; `config.stride == 1`.
- `normalize.channel_floor(features.channel_names())` returns without `KeyError`.

---

## Phase 3 — Data regeneration (the real gate; sequential)

Existing rollouts predate `encoders_vel` (filtered-joint velocity was never saved), so caches **cannot** be rebuilt from them — the sim must be re-run.

**3.1 Verify `sim/collect.py` save path is generic over `FusedSensors._fields`** (the load path at ~L1104 already is). If it hand-picks fields, add `encoders_vel`. If generic, no edit.

**3.2 Re-collect a MODEST but DIVERSE rollout set.** Prioritize breadth over volume — this fixed cache stands in for CoCo's per-iteration aggregation, and diversity (terrain × friction × disturbance × motion phase) is what it buys. Suggested first pass: enough rollouts to give a few hundred thousand usable ticks total, spread across conditions. **Do not** spend the whole night collecting one giant homogeneous set. Log wall-clock; if collection exceeds ~3–4 h, cut it and proceed to train on what exists.

**3.3 Rebuild channel caches** (`build_channel_cache`) → F=30 caches. `prepare`'s name-check rejects stale F=24 caches loudly, so nothing mixes.

**3.4 Fit normalization on a WALKING/MOVING set** (`fit_normalization` → `norm_constants.npz`). **Not a standing set** — standing floors every channel that only moves while walking (gyro y/z, leg q̇/τ, v_bc), which then arrive at the trunk mis-scaled. Inspect the `floored` field of the artifact: if it lists moving channels, the calibration set was too static — recollect a moving one.

**Gate G2 (after Phase 3):**
- `prepare(...)` runs; `PreparedRollout.n_starts > 0` for every rollout.
- cache `names == norm.names`; `apply` does not raise.
- artifact `floored` contains no channel that should be active while walking.

---

## Phase 4 — Train + validate

**4.1** Confirm `train.py` sizes the network from `cfg.d_in` (=600) and reads the updated cfg. Confirm `objective="l2_velocity"`.

**4.2** Run training via `ChainedBatcher` (carried `(X̂,P)`, truncated BPTT, Adam). Set `episode_ticks` to a sane drift horizon (CoCo dancing ≈100 s; at 1 kHz that's 100k ticks — but cap to rollout length). **Log the `reseeds` counter** each step: a climbing rate mid-run means chains are diverging, not episodes ending.

**4.3 Stop criterion is a metric, not an iteration count.** CoCo's E=1280/I=100k are on-policy DAgger numbers and do **not** transfer to this offline loop — do not target them. Stop on held-out velocity-RMSE plateau or wall-clock (leave ≥30 min for validation before 9 AM).

**4.4 Validation → `RESULTS.md`.** On held-out rollouts (not in the training cache), report:
- Velocity RMSE, learned vs. analytic-heuristic `contact_chol` baseline.
- NEES (target: toward 1; report the number regardless).
- Loss curve summary, final `reseeds` rate, any gate that went red.

**Success = learned matches or beats the analytic baseline on held-out velocity RMSE.** That proves the process-socket + q̇ + stride=1 pipeline trains correctly. A large margin is not expected overnight.

---

## Verification (Layer 1) — deterministic gate checks

Promote the gates into one script, `scripts/verify.sh` (the only new file besides `RESULTS.md`), and run it at G1, G2, and before opening the PR. These are mechanical asserts, not judgment — an agent cannot talk its way past a grep. Any failure = stop and report, do not "fix" the check.

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. No test was touched to force a pass (prohibited).
if git diff --name-only origin/HEAD... | grep -q '^tests/'; then
  echo "FAIL: files under tests/ were modified"; exit 1
fi

# 2. Measurement socket stays zero — process socket only.
#    The estimator path must not set contact_meas_chol to anything non-zero.
if grep -rn "contact_meas_chol" src/invariant_estimation/pipeline src/invariant_estimation/contactnet \
     | grep -v "zeros" | grep -v "#" | grep -q .; then
  echo "FAIL: contact_meas_chol referenced non-zero outside a zeros/comment"; exit 1
fi

# 3. No filter state leaked into the feature builders (invariance).
if grep -Enr "state\.|carry\.|\.P\b|q_hat|qhat|xhat|x_hat" \
     src/invariant_estimation/contactnet/features.py \
     src/invariant_estimation/contactnet/online.py | grep -q .; then
  echo "FAIL: a filter-state symbol appears in features.py/online.py"; exit 1
fi

# 4. Feature geometry is CoCo-faithful: F=30, d_in=600, stride=1.
python - <<'PY'
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet import features
cfg = ContactNetConfig()          # defaults are the training config
assert cfg.F == 30,        f"F={cfg.F}, expected 30"
assert cfg.d_in == 600,    f"d_in={cfg.d_in}, expected 600"
assert cfg.stride == 1,    f"stride={cfg.stride}, expected 1 (window not consecutive)"
names = features.channel_names()
assert len(names) == 30,   f"channel_names has {len(names)}, expected 30"
# q̇ block sits between q and tau, in order.
qd = [i for i,n in enumerate(names) if n.startswith("qd_")]
q  = [i for i,n in enumerate(names) if n.startswith("q_")]
tau= [i for i,n in enumerate(names) if n.startswith("tau_")]
assert q and qd and tau and max(q) < min(qd) < max(qd) < min(tau), "q/q̇/τ order wrong"
print("OK: geometry F=30, d_in=600, stride=1, channel order q,q̇,τ")
PY

# 5. online.py and features.py agree (the load-bearing oracle).
python -m pytest -q tests/contactnet/test_online.py

# 6. Normalization has a floor for every channel (no KeyError), incl. qd_.
python - <<'PY'
from invariant_estimation.contactnet import features, normalize
normalize.channel_floor(features.channel_names())   # raises if any channel unfloored
print("OK: every channel has a noise floor")
PY

echo "verify.sh: all Layer-1 checks passed"
```

Notes for the agent:
- Adjust the `git diff` base (`origin/HEAD...`) to whatever the branch was cut from; the intent is "nothing under `tests/` changed on this branch."
- Checks 2–4 encode the three invariants most likely to be silently broken (measurement socket, filter-state leakage, window geometry). If one fails, the fix is in the code the check points at — never in the check.
- This is Layer 1 only: deterministic, un-hallucinatable. It does not judge whether the normalization set was moving or whether `RESULTS.md` overclaims — those stay human review items in the PR.

---

## If you get stuck
- A red `test_online` after Phase 1 almost always means a **channel-order mismatch** between `features` and `online` — diff the two concat lines and `channel_names()`.
- Shape error at the network input → `config.F` not 30, or a stale cache/artifact.
- NaNs in training → check float64 held (`_assert_float64`), then the reseed rate; a diverging chain poisons later steps but should auto-reseed.
- **Do not** resolve any blocker by re-adding the measurement socket, putting filter state into features, changing the loss, or reseeding from truth each segment. If those look tempting, the real bug is elsewhere — stop and write it up.

## Explicitly out of scope tonight
Hardware/export (`export.py`), the learned moving-mean extension, β-NLL and other losses, domain-randomization sweeps, beating CoCo's numbers. Reproduce the baseline first.
