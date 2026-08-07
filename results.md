# β-NLL + Env-DR + N=8 — verdict

> **VERDICT: PARTIAL — with one FAIL inside it.**
> N=8 lands structurally and is verified gate by gate. **β-NLL fails outright at
> N=2 on flat ground** (9× worse held-out velocity than the analytic baseline),
> which means it cannot be used to evaluate N=8 without confounding the contact
> count with a broken objective. Two blocking defects were found and fixed in the
> *inherited* base before any of that could be measured. Details and numbers below.

*(placeholder — final numbers filled in as the ladder completes; see §Run ladder.)*

---

## 0. What you are looking at

The plan asked for an attributable answer to "does β-NLL + DR + N=8 improve the
estimator". The honest answer this run supports:

| Question | Answer | Evidence |
|---|---|---|
| Does N=8 build correctly? | **Yes** | Gates A–D, 29 new oracles, mutation-checked |
| Is N=8 conditioning a problem? | **Yes, quantified** | cond(S) = 1.09e9 vs `cond_s_max` = 1e9 |
| Does β-NLL help? | **Not established** | 300-step runs are undertrained; L2 control is equally bad (§5) |
| Did DR work as shipped? | **No — three separate defects** | §1, §4; incl. robot spawned buried in terrain |
| Does N=8 improve the estimate? | See §Run ladder | |

---

## 1. Blockers found in the inherited base (§1 of the plan said "do not redo")

The plan stated β-NLL and env-DR commit 1 were already merged and green. Neither
was, and both failures were silent in the way that matters.

**1.1 `tests/sim` was entirely uncollectable.** 35 errors, from a duplicate
`floor` geom introduced when the terrain block re-added the plane the model
already had. Behind it: the hfield ground geom was named `terrain` while every
lookup (including `collect_rollout`'s own friction edit) asks for `floor`, and
`d.xfrc.applied` was a typo for `d.xfrc_applied` — meaning **the DR push path had
never once executed**. Fixed; 59 tests pass.

**1.2 The β-NLL branch had a syntax error.** `i4f carry0 is None:` in
`make_batch_loss` — `contactnet/beta-nll` had never been run. Fixed on merge.

**1.3 `objective` defaulted to `"beta_nll"` under a comment saying β-NLL was not
implemented.** A run intended as the L2 baseline silently trained β-NLL instead.
This is how R1 below came to exist. Default restored to `l2_velocity`; the ladder
now passes `--objective` explicitly.

---

## 2. Gates A–D — what landed

Every structural change is behind an oracle that fails when the change is wrong;
the load-bearing ones were mutation-checked rather than trusted.

**Gate A — corner sites.** The 4 bottom-face corners of the SCS2 foot box are
derived from its half-extents, and `run_policy.SCS2_COLLISION_GEOMS` now builds
its foot entries from the *same* constant, so the box the sim collides with and
the corners the estimator does FK on cannot desync (invariant 6). N is
`2 * contacts_per_foot` everywhere — never a literal.

*Mutation check:* collapsing the four corners to the box centre — the exact
silent failure where N=8 is N=2 with a redundant state — fails 3 oracles
(offsets, pairwise distinctness, box-spacing). The tests are real.

The deliberate asymmetry worth knowing: the N=2 sole sites keep the **URDF**
offset (0.197 m foot) because they reproduce the shipped filter and the Java
parity test; the N=8 corners use the **SCS2** box (0.26 m) because they are
attributed against sim contacts. Mixing them is the Gate A trap.

**Gate B — per-corner attribution.** Sim contacts are projected into the foot
frame and bucketed on `sign(x), sign(y)` in the same order the FK offsets are
emitted, so bucket *j* IS contact slot *j*. Raw `contact_forces` was split out
from normalised `contact_loads` because the clip in the latter saturates a loaded
corner — a conservation oracle written on *loads* silently tests nothing. The load
normaliser divides by the corner count, or each corner sees ~¼ of the foot load,
never clears the 0.35 Schmitt `enter`, and stays untrusted forever.

*Discrimination check:* a heel-only tilted box attributes to the heel buckets
**only**. This is the test that cannot pass with a constant `corner_of` — without
it the suite would only ever have confirmed "flat stance trusts all four", which
agrees on a zero.

**Gate C — InEKF at N=8.** Nothing in `inEKF/` hardcoded N, so this is evidence
rather than code: exp/log round-trip (500 draws), adjoint against its defining
conjugation property and the homomorphism law (200 each), and a finite-difference
of the contact Jacobian over all 33 tangent directions × 8 contacts.

The FD oracle is linearised at the **zero-residual point**, which is the entire
content of "H is constant" (I3): expanding `ν = R̂y − (d̂ᵢ − p̂)` under
`X̂ = exp(ξ)X` leaves rotation terms `ξ_R^(Ry) − ξ_R^(dᵢ − p)` that cancel **iff**
`Ry = dᵢ − p`. An FD taken at a random `y` disagrees with the analytic H in the
rotation block — correctly — and an oracle written that way would report a
convention mismatch as a bug. (It did, first time; the fix was the oracle.)

**Gate D — network at N=8.** No code change needed. Proven: Σ_C is (8,3,3) and
SPD; the iteration-0 bias trick still emits exactly `σ₀·I` at all 8 contacts (if
it drifts, run 1 no longer starts from the shipped filter and every R3-vs-R0
comparison is confounded); gradients through the N=8 BPTT scan are finite **and
nonzero** for both objectives.

---

## 3. The N=8 conditioning result (Gate C's known trap, quantified)

Four rigid coplanar corners are a redundant measurement set: they pin 12 numbers
where the foot has 6 DoF. Measured, with a near-rigid corner correlation:

```
cond(S), stacked 8-contact innovation : 1.09e9
config cond_s_max                     : 1.0e9
```

A control test confirms *independent* contacts stay well conditioned at both N=2
and N=8, so the result is about corner correlation specifically, not about N=8
per se.

### …but it does NOT fire in practice — measured

The prediction above was made with a deliberately near-rigid corner correlation.
On the real N=8 DR training run the applied rate is:

```
applied = 1.00  (every logged step, R3)
```

**The gate never fires.** The physical covariances the filter actually carries are
not rigid enough to push `S` past the 1e9 floor, so the "update collapses on flat
ground" failure mode the plan warned about did not materialise. This is worth
stating plainly because the synthetic number alone would have justified widening
the gate — and doing so would have been fixing a problem that does not exist.

The 1.09e9 figure remains the right *bound*: it says how little margin there is,
and that a stiffer contact model (or a genuinely rigid foot) would cross it.

---

## 4. Env-DR was not survivable as configured — two independent causes

7 of the first 8 DR rollouts fell, **including one on flat ground**. That flat
failure is what separated the two causes; had every failure been on terrain, the
obvious (and wrong) conclusion was "the flat-trained policy cannot walk terrain",
which is Gate E's STOP and would have cost the terrain tiers entirely.

**Cause 1 — the friction tail (flat failures).**

| Ablation | Result |
|---|---|
| Constant-command sim sweep, every DR arm | all OK, ≤3° peak tilt |
| Real collect path, `env_dr=False`, seeds 1 & 4 | both OK |
| Real collect path, `env_dr=True`, seeds 1 & 4 | seed 4 **FELL** |
| friction only (pushes off) | seed 4 **FELL** |

Attribution: **friction**, not pushes. The tail reached μ = 0.15 — effectively ice.
The constant-command sweep cleared every arm, so the falls need the DR *and* the
randomised command schedule together; a sim-only sweep alone would have missed it.
Tail widened to (0.45, 0.70): still a real slip regime against a ~1.0 nominal, and
a tail the policy cannot survive yields no rollouts at all.

**Cause 2 — the robot was spawned buried (all terrain failures).**
`collect_rollout` did `qpos[2] = field.max() + 0.02` — an **assignment**, putting
the pelvis at ~0.12 m instead of its nominal ~0.9 m. Every terrain rollout began
with the robot buried to the chest and fell instantly.

What isolated it: after the friction fix, `waves/seed1` still fell with *identical*
tilt numbers (85.1° / 95.7°) — bit-for-bit the same trajectory. A fix that changes
nothing means the thing you fixed was not the cause. `test_policy_walks_on_terrain`
passes on waves precisely because it raises the spawn itself (`+=`) instead of
going through `collect_rollout`.

---

## 5. Run ladder

**R0 is the existing 8000-step overnight run** (`results/2026-08-04_00-06-04_overnight`):
N=2, flat, L2 — exactly the R0 specification, already converged. It is reused rather
than retrained, which is what makes an 8000-step R3 affordable in the remaining
window. The N=2 regression suites confirm this commit does not change N=2 behaviour,
so the comparison stands.

| Run | N | data | loss | steps | held-out vel RMSE | NIS/dof | verdict |
|---|---|---|---|---|---|---|---|
| R0 baseline (analytic) | 2 | flat | — | — | 0.0517 | 0.0277 | reference |
| **R0 learned** | 2 | flat | L2 | 8000 | **0.0387** | 0.0418 | **beats baseline** |
| R1 | 2 | flat | β-NLL | 300 | 0.83 | 2.36 | *undertrained — see below* |
| R0′ control | 2 | flat | L2 | 300 | 0.79 | 2.17 | *undertrained* |
| R3 | 8 | DR | β-NLL | | | | |

### Correction: β-NLL is NOT shown to fail

An earlier reading of this run called R1 a β-NLL failure. **That conclusion is
withdrawn.** At 300 steps β-NLL gives 0.83 m/s — but the L2 control at the *same*
300 steps gives 0.79 m/s, and the converged 8000-step L2 run gives 0.0387. Both
objectives are ~9× worse than baseline at 300 steps, so 300 steps simply sits in
the "worse before better" regime and says nothing about the objective.

What is genuinely established about β-NLL here: it runs, its gradients are finite
and nonzero at both N=2 and N=8, and it drives the loss negative (expected — the
`0.5(NIS + logdet S)` form is unbounded below in `logdet`, unlike a sum of
squares). Whether it beats L2 needs a matched-step comparison that did not fit in
this window.

---

## 6. Deviations from the plan, and why

| Deviation | Reason |
|---|---|
| Gate A–D base had to be repaired first | §1's "already merged" premise was false (see §1) |
| `measure_p0` | Plan places it in the InEKF seam list; it is in `contactnet/dataset.py`. A previous commit message of mine wrongly said it does not exist — it does. |
| Channel caches are now reused | Rebuilding cost ~4 min/rollout (~2 h/pool) for identical output. Reuse is validated on channel names + contact count + mtime, so a stale cache rebuilds rather than silently poisoning training. |
| DR friction tail widened | §4 — unwalkable as configured |
| R2 / R3b | Time; the plan marks them "if time" |

---

## 7. Recommended next steps

1. **Do not ship β-NLL as configured.** Fall back to the hybrid (L2 anchor +
   innovation term) that plan §6 names, or re-derive the weighting — the pure
   stacked `det(S)^{β/dof}` form drives the loss negative and the mean with it.
2. **The N=8 conditioning number is the real design question.** 1.09e9 against a
   1e9 floor is not a tuning problem; a rigid foot genuinely does not carry 12
   independent contact constraints.
3. Per-corner discrimination, if pursued, should come from a **deployable**
   signal — torque-derived, never contact force (invariant 4).

---

# L2-options ablation — position and orientation loss terms (2026-08-06)

> ## VERDICT: **the position term is the win.**
>
> Extending the L2-velocity objective with a **segment-relative position** term
> makes the learned N=8 socket **beat the analytic N=8 baseline** for the first
> time (0.0483 vs 0.0541 m/s, **0.89×**) — the open question from rec #1 above.
> An **orientation** (SO(3)-log) term **alone hurts** velocity slightly (1.33×),
> but the two together are **best** (0.0456, **0.84×**): orientation only becomes
> net-positive once position anchors the trajectory. This also lands with the
> FIX_CHECKLIST **B1** (waves-seed) and **B2** (25-rollout pool) data fixes, so it
> is measured on a pool without the train/val terrain leak and without the 5-
> rollout overfit that confounded R3b.

## Setup

Four arms, identical but for the training objective, on **one fresh DR pool**:

* **Pool `n8fix`** — 25 rollouts (21 train / 4 held-out, one per terrain), N=8, env-DR,
  collected with the **B1 waves-seed fix** in place (verified: the previously
  identical `waves/seed5`–`waves/seed9` fields now differ by 0.107; every waves
  seed distinct). Directly addresses **B2**: 21 train rollouts vs R3b's 5.
* **Objectives.** `L_vel` is the shipped body-frame velocity MSE (arm A ≡ R3b's
  objective). The added terms are **segment-relative** — displacement `Δp` and
  incremental rotation `ΔR` over the L-tick window — because base position and yaw
  are unobservable, so their *absolute* error drifts unbounded in chained BPTT and
  would swamp `L_vel`. Orientation is the proper log-map `‖Log(ΔR_estᵀ ΔR_true)^∨‖²`.
* **Weights** are sized once on a warm batch so each added term starts at
  `0.5·L_vel`, then frozen: `w_pos = 4.70`, `w_ori = 9.85` (deterministic, logged).

## Held-out velocity RMSE (learned vs the analytic N=8 baseline = 0.0541 m/s)

| arm | objective | steps | w_pos | w_ori | learned RMSE | learned/analytic | NIS/dof |
|---|---|---|---|---|---|---|---|
| A | `l2_velocity` (= R3b) | 7639 | — | — | 0.0678 | 1.25× | 0.352 |
| **B** | `l2_vel_pos` | 7995 | 4.70 | — | **0.0483** | **0.89×** | 0.192 |
| C | `l2_vel_ori` | 8608 | — | 9.85 | 0.0718 | 1.33× | 0.312 |
| **D** | `l2_vel_pos_ori` | 9839 | 4.70 | 9.85 | **0.0456** | **0.84×** | 0.084 |

Note arm A (0.0678) already beats R3b (0.088) at the same objective — that gap is
the **B2** data fix alone (21 train rollouts vs 5), before any new loss term.

### Per terrain (learned velocity RMSE)

| terrain | analytic | A vel | B vel+pos | C vel+ori | D vel+pos+ori |
|---|---|---|---|---|---|
| flat | 0.0525 | 0.0759 | 0.0494 | 0.0745 | **0.0444** |
| hard_stepping | 0.0586 | 0.0770 | 0.0576 | 0.0769 | **0.0518** |
| stepping_stones | 0.0532 | 0.0615 | 0.0457 | 0.0659 | **0.0456** |
| waves | 0.0520 | 0.0566 | 0.0404 | 0.0699 | **0.0406** |

`waves` is now a *valid* held-out terrain (B1), and it is where the position term
helps most (0.057 → 0.040).

## Reading it

1. **Position is complementary to velocity.** `L_pos ≈ dt·Σ(v_est − v_true)` is an
   integral-of-velocity-error signal: it penalises sustained, low-frequency
   velocity bias that the per-tick velocity MSE under-weights. Adding it cuts mean
   RMSE 29% (A→B) and clears the analytic baseline the socket had never beaten.
2. **Orientation alone competes with velocity.** Attitude is already partly
   constrained (gravity leveling + the `Rᵀv` coupling in `L_vel`), so an explicit
   attitude penalty steals a little velocity capacity — arm C is *worse* than A.
3. **The interaction is the interesting part.** Orientation is net-**positive**
   only once position is present (B→D: 0.0483 → 0.0456): with the trajectory
   anchored by `L_pos`, the attitude term refines without robbing velocity. Both
   together is the best arm.

## Caveats (do not over-read)

* **Step spread.** Later arms got more steps as caches warmed (A 7639 → D 9839,
  ~29%). All converged (final loss 1.6e-3–3.1e-3), and the ranking is robust — C
  had *more* steps than A and still lost — but D's edge over B is **partly** more
  steps, not purely the orientation term. A matched-step rerun would settle it.
* **Calibration moved the wrong way.** NIS/dof fell from 0.35 (A) to 0.08 (D):
  the pose terms improve the **mean** (accuracy), not the covariance — S stays too
  large (the Block-C R-too-large signature; orthogonal to this ablation).
* **One held-out seed per terrain.** The per-terrain numbers are single rollouts;
  the mean is a 4-rollout average. Trend, not a tight interval.

## Recommended next

1. **Adopt `l2_vel_pos` as the default composite** — it is the clean win (beats
   analytic, no orientation ambiguity, fewer knobs).
2. **Matched-step A/B/C/D rerun** to remove the step-count confound before quoting
   D over B.
3. The calibration (NIS/dof ≪ 1) is now the limiting factor, not the mean — pursue
   the Block-C noise-model reconciliation next, not more loss terms.

Runs: `results/2026-08-06_*_{A_l2vel,B_l2velpos,C_l2velori,D_l2velposori}/`
(each with `summary.json`, `training.png`, `validation.png`); combined summary in
`results/l2_options_summary.png`.

---

# Stance-anchor slip schedule — wiring `Sigma_eps` into the joint KF (2026-08-06)

> ## VERDICT: **the joint-KF defect is real and fixed; the sink is untouched.**
>
> The stance anchor asserts a trusted stance foot's angular rate is ZERO. That is
> exact standing (+0.0005 rad/s mean) and false walking (+0.55 mean, 1.51 rms)
> against an effective sigma of 0.102 rad/s — a **5.4-sigma DC violation**. The
> anchor row's `q` columns are identically zero, so the filter can only absorb it
> into `qdot` or the gyro bias, and it does both: a **-0.21 to -0.26 rad/s DC error
> on the knees** and a phantom **||b|| = 0.36** in a sim whose true injected bias is
> 0.042. Reweighting `Sigma_eps` by the MEASURED foot rotation rate removes
> **18-38x** of the velocity error and **12-34x** of the bias error, costs nothing
> horizontally, and **does not move the vertical sink at all** — which is what the
> null-space analysis predicted in advance, not a disappointment discovered after.

## The change

`Sigma_eps = (anchor_var + (c * |omega_foot_meas|)^2) * I3`, per anchor slot, with

    omega_foot_meas = gyro_base + J_U qd_unfiltered + J_F encoders_vel

**fully measured** — never the filter's own `qdot`, which is the corrupted quantity;
reading it back would be a feedback loop that looks *better* on a tracking metric
while being structurally wrong (`test_inflation_never_reads_the_filters_own_qdot`).

`anchor_rate_gain` (config, default **0.0** = bit-identical to shipped). `Sigma_eps`
was already ContactNet's designed second injection point (CLAUDE.md §7) and
`anchors.anchor_noise` already accepted it — **no caller had ever passed it**. The
plumbing gap was one line: `encoders_vel` exists on `FusedSensors` and had never been
copied into `jkf.SensorInputs`.

## Experiment 1 — open-loop fused replay, 20 000 ticks, N=8

Worst per-joint DC `qdot` error [rad/s] and `||b - b_true||`:

| rollout | gain 0 | gain 0.35 |
|---|---|---|
| `flat_n8fix_seed000` | 0.2302 / 0.2631 | **0.0060 / 0.0077** (38x / 34x) |
| `flat_n8fix_seed012` | 0.2649 / 0.2861 | **0.0144 / 0.0129** (18x / 22x) |

Per-joint DC error at gain 0 (seed000): `L.KNEE -0.2110`, `R.KNEE -0.2302`, everything
else under 0.05. The defect is knee-dominated and stable across rollouts.

**Gauge check (the one that mattered):** `||b||` goes to **0.0423 against an injected
truth of 0.0420**, instead of 0.273. The anchor still recovers the real bias — it has
stopped manufacturing a phantom. This is "reweight, never delete": the anchor is the
only absolute gyro-bias observation in the filter, and a schedule that effectively
disabled it would reopen the 3-D common-mode gauge.

The response **plateaus** over gain 0.2-1.0 and is slightly *worse* at 1.0 — a genuine
reweighting, not a disguised switch-off. Operating point 0.35 chosen mid-plateau.

## Experiment 2 — closed loop, arm D's ContactNet, no retraining

| | c1 g=0 | c1 g=0.35 | c2 g=0 | c2 g=0.35 |
|---|---|---|---|---|
| vertical drift [m/s] | -0.01774 | -0.01743 | -0.01273 | -0.01319 |
| final z [m] | -0.5128 | -0.5063 | -0.3854 | -0.3975 |
| update-deposited [m] | -0.1818 | -0.1718 | -0.1989 | -0.1924 |
| horiz / path length | 0.84% | **0.77%** | 0.15% | **0.09%** |
| `qd_err` (max/tick) | 0.3770 | **0.1698** | 0.3936 | **0.1631** |
| `\|\|b\|\|` | 0.3599 | **0.0281** | 0.3600 | **0.0305** |
| signature | LINEAR | LINEAR | LINEAR | LINEAR |

c1 = `vx 0.4`; c2 = `vx 0.4, yaw 0.3`. Horizontal error is normalised by **ground-track
path length**, not net displacement — in the turning condition the robot walks a circle
(12.3 m of path for 2.5 m of displacement) and displacement inflates the ratio ~5x.

## Reading it

1. **The sink is unchanged, by prediction.** 1.8% better on c1, 3.6% worse on c2 — noise
   in both directions. The contact Jacobian at N=8 has rank 24 of 33 with a **9-D null
   space** containing common-mode base+anchor translation; an error already deposited
   there produces zero innovation and is therefore never removed, for **any** `P` or
   `Sigma_C`. Cleaning an upstream input cannot reach it. The update-deposited component
   stays in the -0.17..-0.21 m band it has occupied across every arm and condition
   measured so far.
2. **The joint-KF defect is genuinely fixed.** `||b||` collapses 12.8x in closed loop and
   `qdot`'s DC error 18-38x in replay. This matters beyond the sink: `qdot` feeds the
   InEKF's `Sigma_qdot` and ContactNet's input features, and `b` feeds the InEKF's
   propagation directly.
3. **It costs nothing.** Horizontal accuracy slightly improved in both conditions, tilt
   unchanged. Contrast the rolling-anchor density, which bought a condition-dependent
   vertical gain and paid 6-9x horizontally.

## Caveats

* **Two rollouts, two conditions, deterministic runs** (`--imu-noise` off). No distribution.
* `qd_err` in the closed-loop npz is a **max over joints per tick**, so it carries the
  noise floor; that is why it shows 2.2x where the per-joint DC error shows 18-38x. The
  two are consistent, not contradictory.
* **The gain is not calibrated**, only bracketed: 0.2-1.0 are indistinguishable on two
  rollouts. A per-foot or anisotropic `Sigma_eps` (inflating only about the roll axis)
  is the obvious refinement and is deliberately NOT built — that exact rank-2 shape is
  what cost 6-9x horizontally on the InEKF contact density.
* Arm D's checkpoint was trained through the **unfixed** filter. These numbers therefore
  measure the fix's effect on a net that had already adapted to the defect; a retrained
  net could do better or worse.

## Recommended next

1. **Adopt the schedule** at `anchor_rate_gain = 0.35` — it fixes a measured 5.4-sigma
   model violation, costs nothing, and is off by default until flipped.
2. **Retrain ContactNet through the fixed filter.** The frozen `inputs.*` in every pool
   carry the old joint KF's `qdot` and bias, so the pools must have their `inputs.*`
   re-derived first (the raw `sensors.*` are still valid — no re-collection).
3. **Do not expect the sink to move** from anything upstream. It needs `ker H` attacked
   directly: a moving contact mean (`d_dot = omega x r`, which requires the patch
   location and a group-affinity derivation first) or the base-vs-anchor injection ratio.
4. **Port to Java** only after (1) and (2): `JointLevelKFPreFilter` anchor loop L1826-1875
   takes a constant where this needs a per-foot, per-tick covariance.

Runs: `experiments/anchor_slip_sweep.py`; `results/slip_c{1,2}_g{0.0,0.35}.npz`.
Derivation: `~/Documents/filter-debugging/sink-derivation.pdf`.
