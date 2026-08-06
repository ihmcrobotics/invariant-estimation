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

# Rolling-anchor density under ContactNet — arm E (2026-08-06)

> ## VERDICT: **NULL RESULT.** Training arm D through the rolling-anchor contact
> density does **not** fix the closed-loop sink, and costs held-out velocity RMSE.
>
> The sink improves 1.75× on one closed-loop condition and gets **1.42× worse** on
> the other — a coin flip on n=2, not an effect. **An earlier report of "1.75×
> better" was read off condition 1 alone and is retracted**; the retraction is the
> useful part of this run. The one reproducible effect is a **horizontal
> regression, 6–9× on both conditions**. And the reason the open-loop numbers do
> not translate is the finding worth keeping: **analytic Σ_C and learned Σ_C are
> substitutes, not complements** — the rolling term buys the *analytic* filter 1.8×
> on `vel_nees_z` and the *learned* filter ~nothing, because ContactNet had already
> absorbed the same effect into its own Σ_C.

## Setup

* **Arm E** = arm D's objective (`l2_vel_pos_ori`), same `n8fix` pool, same N=8
  (`--contacts-per-foot 4`), retrained with `--rolling`: the filter it is trained
  *through* carries `Σ_C += τ σ_r² (‖ω‖² I − ω ωᵀ)`
  (`inEKF/contact.py::rolling_anchor_density`), τ = 0.25 s, σ_r = 0.0985 m — both
  config defaults, untuned. 12000/12000 steps, 8229 s wall, `w_pos = 4.7185`,
  `w_ori = 14.14`.
* **Both arms are evaluated against their own filter.** E's "analytic baseline" is
  the rolling filter with the heuristic Σ_C; D's is the non-rolling one. The
  learned-vs-analytic ratios are therefore within-arm; the cross-arm *learned*
  comparison is the one that carries the rolling effect.
* `summary.json` now carries a top-level **`filter`** block recording the resolved
  build (`{rolling: {enabled, tau, sigma_r}}`). Arms A–D predate it.
  `run_estimator.py::_check_contact_geometry` refuses a rolling mismatch in either
  direction, and `scripts/evaluate_run.py` now reads the same block rather than
  silently rebuilding a non-rolling collector (it did, until this run).

## Held-out (4 terrains, 4 held-out rollouts, means)

| metric | D analytic | D learned | E analytic | E learned | rolling did what |
|---|---|---|---|---|---|
| `vel_rmse` [m/s] | 0.0541 | **0.0456** | 0.0531 | 0.0492 | analytic −2%, learned **+8% worse** |
| `vel_nees` (3 dof) | 4.645 | 2.452 | 3.038 | 2.370 | analytic **1.53×** better, learned 1.03× |
| `vel_nees_z` | 2.753 | 1.341 | 1.525 | 1.218 | analytic **1.80×** better, learned 1.10× |
| `nis_over_dof` | 0.0412 | 0.0845 | 0.0219 | 0.0683 | S grows either way (unchanged story) |

### Per-terrain learned velocity RMSE (each against its own analytic baseline)

| terrain | D analytic | **D learned** | E analytic | **E learned** |
|---|---|---|---|---|
| flat | 0.0525 | **0.0444** | 0.0514 | **0.0468** |
| hard_stepping | 0.0586 | **0.0518** | 0.0564 | **0.0530** |
| stepping_stones | 0.0532 | **0.0456** | 0.0533 | **0.0481** |
| waves | 0.0520 | **0.0406** | 0.0514 | **0.0489** |
| mean | 0.0541 | **0.0456** | 0.0531 | **0.0492** |

E's learned arm loses to D's on **every** terrain, worst on `waves` (0.0406 →
0.0489, the terrain where the position term had helped most). E still beats its
own analytic baseline everywhere, so the socket is not broken — it is just worse.

## Closed loop — the sink, which is what this arm was FOR

30 s (1500 control ticks @ 50 Hz), policy driven by the estimate, `--imu-noise`
off, one deterministic run per arm per condition.

| condition | metric | D (rolling off) | E (rolling on) | |
|---|---|---|---|---|
| vx 0.4 | vertical drift rate | −0.0177 m/s | **−0.0101 m/s** | 1.75× better |
| | final z error | −0.513 m | **−0.290 m** | |
| vx 0.4 + yaw 0.3 | vertical drift rate | −0.0127 m/s | **−0.0180 m/s** | **1.42× worse** |
| | final z error | −0.385 m | **−0.522 m** | |
| vx 0.4 | final horiz err / ground track | 0.103 m / 12.22 m = **0.84%** | 0.974 m / 12.34 m = **7.90%** | **9.4× worse** |
| vx 0.4 + yaw 0.3 | final horiz err / ground track | 0.018 m / 12.28 m = **0.15%** | 0.108 m / 12.41 m = **0.87%** | **5.9× worse** |

Error signature (linear-fit vs √t-fit residual RMS on `z`) stayed **LINEAR
(biased)** on both arms in both conditions. Touchdown concentration (share of
`|Δz|` within ±40 ms of a contact rising edge, over the share of ticks) 1.44× (D)
vs 1.85× (E) on condition 1 — i.e. **more** of the error lands at touchdown with
rolling on, not less.

This is the direct contradiction of the open-loop N=2 result in
`RESEED_ROLLING_ANCHOR_README.md`, where the rolling anchor took the drift to
0.18×/0.11× **and flipped the signature to SQRT (diffusive)** and collapsed the
touchdown concentration 3.0× → 0.2×. None of those three signatures reproduces
here. What changed between the two measurements: N=2 → N=8, open-loop replay →
closed loop, analytic Σ_C → ContactNet Σ_C.

## Σ_C diagnostics (Gate G) — the mechanism behind the null result

`scripts/evaluate_run.py` → `sigma_diag.npz` → `scripts/plot_sigma_c.py`, run for
**both** arms on the **same** held-out rollout (`flat_n8fix_seed028`, first 4000
ticks, foot 0), so the two figures are directly comparable.

| quantity (stance ticks unless noted) | D (rolling off) | E (rolling on) |
|---|---|---|
| median `det(Σ_C)^{1/3}` | 2.38e-7 m | **9.45e-9 m** (25× smaller) |
| swing/stance modulation of `tr Σ_C` | **336×** | **120×** |
| median `cond(Σ_C)` = λ_max/λ_min | 8.3e8 | **2.9e12** |
| median eigenvalues | (4.2e-12, 3.3e-6, 4.5e-3) | (4.8e-16, 1.4e-6, 1.6e-3) |
| per-corner relative spread (script stat) | **0.565** | **0.093** |
| numerically singular ticks (`det ≤ 1e-300`) | 0.00% | **0.88%**, all in swing |

Look at the two `sigma_c_over_stride.png` side by side: **D's Σ_C sweeps ~8 orders
of magnitude over the gait cycle** (1e-7 in stance up to ~1e0 in swing) with the
four corners visibly separated. **E's is nearly flat at ~1e-8** across the whole
cycle, the four corners lie on top of each other, and it drops to numerically
singular for short flickers in mid-swing.

That is finding 3 in the network's own output: given a filter that already
supplies an ω-driven, phase-modulated contact density analytically, the net
**stopped producing one**. It collapsed onto a near-constant, near-rank-2 floor —
including losing most of the per-corner discrimination that was the open question
from the N=8 verdict above (0.565 → 0.093). The learned and analytic terms are
filling the same hole, and only one of them fills it at a time. (The singular
ticks are swing-only, where contacts are untrusted and Σ_C is irrelevant to the
update, so they are a symptom rather than a bug — but a near-rank-2 Σ_C in
**stance**, cond 2.9e12, is not something to leave unwatched.)

## Reading it

1. **The sink is not fixed.** −0.0101 vs −0.0177 on one condition, −0.0180 vs
   −0.0127 on the other. Two conditions, opposite signs, one deterministic run
   each: that is a coin flip, and the honest summary is "no effect established".
   The earlier "1.75× better" headline came from condition 1 alone, before
   condition 2 existed — **retracted**. Any future single-condition drift number
   should be treated the same way until a second condition agrees with it.
2. **The horizontal regression is the one reproducible effect.** 9.4× and 5.9× on
   two independent conditions, same sign, large. *Hypothesis, not a finding:* the
   density `τ σ_r² (‖ω‖² I − ω ωᵀ)` is rank 2 with its null direction along `ω`.
   Walking `ω` is pitch-dominated (`≈ e_y`), so the term inflates Σ_C in exactly
   the `x`–`z` plane and leaves `y` alone — it loosens the anchor along the
   direction of travel. That predicts a horizontal (fore–aft) cost as the price of
   the vertical relief, which is what the table shows. It is falsifiable: split the
   horizontal error into fore-aft and lateral, and check the ratio tracks `ω`'s
   direction.
3. **Analytic and learned Σ_C are substitutes, not complements.** The rolling term
   moves the *analytic* filter a lot (`vel_nees_z` 2.75 → 1.53, **1.8×**) and the
   *learned* filter almost not at all (1.34 → 1.22, 1.10×), while **costing**
   velocity RMSE (0.0456 → 0.0492) despite E getting **22% more training steps**.
   ContactNet had already learned to emit whatever Σ_C the analytic path was
   missing; adding an analytic term that supplies the same thing spends network
   capacity on double-counting, and the residual capacity buys less accuracy. The
   Σ_C diagnostics above show it directly — E's learned Σ_C went flat and
   near-rank-2 where D's modulated over the stride.
   This generalises past the rolling anchor: **any Σ_C improvement measured on the
   analytic filter must be re-measured under the net before it is believed.**

## Caveats (do not over-read)

* **E got 12000 steps, D got 9839** (D was time-truncated). E had **more**
  training and still lost on RMSE, which *strengthens* the negative result rather
  than confounding it — but the two are not step-matched, and that is stated here
  so nobody re-derives it as a surprise.
* **The training losses are not strictly comparable.** At matched step 9839 E's
  loss was 2.40e-3 vs D's 1.88e-3, but `w_ori` is auto-sized per run (14.14 for E
  vs 9.85 for D), so the two composites weight the orientation term differently.
  Ranking by held-out RMSE, not by loss.
* **The closed-loop evidence is n=2 conditions, not a distribution.** One
  deterministic run per arm per condition, `--imu-noise` off. No seeds, no
  terrain variation, no error bars. Two conditions disagreeing is exactly what
  n=2 looks like when the effect is small or absent.
* **`σ_r` was left at the config default 0.0985 m** — the **N=2 sole-centre**
  value. `RESEED_ROLLING_ANCHOR_README.md` caveat 2 says it must be re-derived for
  per-corner anchors before flipping `--rolling` and N=8 on together; the
  `RollingAnchorParams` docstring argues the opposite (the prior on `‖r_i‖` is not
  smaller at a corner — the far edge is still a foot away — what N=8 buys is
  observability of `r_i`, not a tighter prior). **That disagreement is unresolved,
  and this arm ran with it unresolved.** A σ_r sweep is the cheapest way to find
  out which side is right.
* **Closed-loop contact NIS moved the opposite way to held-out NIS/dof** (median
  0.073 → 1.28 on condition 1, 0.052 → 0.96 on condition 2, D → E; held-out
  `nis_over_dof` *fell*, 0.0845 → 0.0683). Unexplained; single runs; noted so it
  is not rediscovered as new.
* **`results/l2_options_summary.png` was NOT regenerated.** It has no generating
  script in the repo, and arm E does not belong on it under its existing
  conventions anyway: the grey bars and the dashed line are the *non-rolling*
  analytic baseline, and E's analytic reference is a different filter. Putting E's
  bar next to D's grey would compare against the wrong baseline — the exact
  confound this section is about. Arms A–D only in that figure.

## Recommended next

1. **Do not adopt `--rolling` for ContactNet training.** Arm D stays the reference
   N=8 checkpoint. The flag remains correct and useful for the *analytic* filter
   (1.8× on `vel_nees_z` is real), just not underneath a trained socket.
2. **Re-derive σ_r for corner anchors, or sweep it** (0.02–0.10 m), and settle the
   README-vs-docstring disagreement before any further rolling arm. Rerunning arm E
   at a σ_r the corner geometry actually justifies is the only version of this
   experiment worth the 2.3 h.
3. **Test the rank-2 hypothesis directly** — decompose the closed-loop horizontal
   error into fore-aft and lateral and check it aligns with `ω`'s null direction.
   Cheap (the `.npz` files already hold `est_p`/`true_p`), and it either promotes
   finding 2 to a mechanism or kills it.
4. **Make the closed-loop sink measurement a distribution, not a run.** ≥4 seeds ×
   ≥2 command conditions before any drift claim is quoted again. This run cost a
   retraction because n=1 looked like a result.

Runs: `results/2026-08-06_12-41-11_E_l2velposori_rolling/` (`summary.json` with
the new `filter` block, `training.png`, `validation.png`, `eval.json`,
`sigma_diag.npz`, `sigma_c_over_stride.png`, `sigma_c_stats.json`,
`closed_loop/contactnet_demo_ghost_n8_rolling.{mp4,gif}`) against
`results/2026-08-06_06-42-29_D_l2velposori/` (same eval artifacts, regenerated
here so the two Σ_C figures are comparable); closed-loop sink traces in
`results/sink{,2}_{D_rolling_off,E_rolling_on}.npz`. Both evals reproduce their
run's own `summary.json` validation numbers to ~1e-12, which is the check that the
rebuilt collector matched the trained filter in each case.
