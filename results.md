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

> **SUPERSEDED 2026-08-10 — do not act on items 1–2.** Both rest on held-out velocity
> RMSE, which was subsequently measured to be *uncorrelated* with vertical drift
> (Spearman +0.05 over nine checkpoints). The matched-step rerun in item 2 was done:
> with the contact R floor live, `l2_vel_pos` came back the **worst** of the four arms
> (+41% vs the analytic baseline), inverting the ranking below. See §8.

1. ~~**Adopt `l2_vel_pos` as the default composite**~~ — see the note above.
2. ~~**Matched-step A/B/C/D rerun**~~ — done; see §8.
3. The calibration (NIS/dof ≪ 1) is now the limiting factor, not the mean — pursue
   the Block-C noise-model reconciliation next, not more loss terms.

Runs: `results/2026-08-06_*_{A_l2vel,B_l2velpos,C_l2velori,D_l2velposori}/` — the
`summary.json` of each is still there; their weights, histories and plots (and the
combined `results/l2_options_summary.png`) were pruned from the working tree on
2026-08-10 and are recoverable from git history at `e84321f~1`. This section's ranking
is superseded twice over anyway (§8 matched-step, §9 closed-loop), so the checkpoints
had no remaining use.


---

## 8. L-ablation and the z-drift pivot (2026-08-09/10)

Full write-ups: `~/Documents/filter-debugging/z-drift-pivot.pdf` (results),
`contact-zero-velocity.pdf` (the structural proposal), `branch-fixes.pdf` (what to
port back). Mechanism detail and the invariants live in `PORT_NOTES.md` and
`CLAUDE.md` §7. This section is the outcome only.

### What was run

4 objectives x L in {128, 256, 512} at **matched 6000 steps** on the `n8fix` pool,
after fixing two dead knobs (`remat`, `contact_meas_var` — see PORT_NOTES). Then a
six-point contact-R-floor sweep, a randomized-motion pool, and a closed-loop check.

### What it showed

* **The objective stops mattering once the horizon is adequate.** At L=128 the four
  arms spread 54% on held-out velocity RMSE; at L=256, 2.5% — all beating the
  analytic baseline. The L2-options ranking above is a low-horizon artifact.
* **The horizon is spent by L=256.** L=512 is worse on drift; NEES_z saturates at
  ~2.0; NIS/dof is flat at ~0.18 across every L. An L=1024 column was queued and
  cancelled.
* **Randomized motion hurt.** Spectrally it worked (gait line 62% -> 34% of in-band
  power) and cost 2.4x on drift when scored on walking.
* **No R floor gives both low drift and consistency.** Every floor that looks good on
  drift is a cancellation across terrains; the only sign-consistent one (1e-3) is the
  worst on both axes. NIS/dof is monotone in the floor.

### The result that invalidates the rest

**Offline replay is not a valid metric for a learned Sigma_C.** Replay and closed-loop
agree for the *analytic* arm and disagree by **21x** on the learned one: the best
checkpoint measures -0.085 m in replay (and "2.3x better than analytic") and
**+2.49 m** closed-loop, against the analytic heuristic's -0.117 m in the same
harness. `scripts/online_offline_oracle.py` confirms the deployed path is faithful to
1e-15, so this is the metric, not a bug.

Everything in this section derived from replay — the floor sweep, the sign analysis,
the arm ranking — therefore needs redoing closed-loop (~3 min per configuration,
*cheaper* than replay). The analytic-baseline rows are unaffected.

### Why the network fails, and the one fix tried

Learned Sigma_C modulates stance->swing by 885x where the analytic heuristic spans
1e10 — 685x too tight in swing, so the filter treats a lifting foot as world-static
and pushes the *base* upward. Error is ~0 standing and appears the moment walking
starts.

`diag_param="exp"` (log-parameterised diag(L)) was added to open that range and
retrained once. **The mechanism worked and the optimisation did not:** span 12.8 ->
21.9 raw units, swing Sigma_C from 685x to **8x** too tight — but held-out RMSE 1.374
vs the analytic 0.0607, loss rising, 30 non-finite steps. `exp` is unbounded (p99
per-axis Sigma_C ~2.7e5, contact update effectively off) and its uniform relative
sensitivity makes `peak_lr=1e-4`, tuned for softplus, too aggressive. Default stays
`softplus`.

### Open

1. ~~Redo the floor sweep closed-loop (~20 min total).~~ Done 2026-08-10, §9.
2. Bounded/retuned `exp` (clamp to ~[1e-4, 1e2], lower LR).
3. The structural option: the contact **zero-velocity constraint**, which adds rows
   to `H` rather than reweighting existing ones. `H` currently has no velocity
   columns, and 98.6% of the sink flows through the contact update's write into
   `v_hat`/`R_hat`. Derivation in `contact-zero-velocity.pdf`.

---

## 9. The closed-loop floor sweep (2026-08-10)

Twelve runs, `scripts/cl_floor_sweep.sh` → `scripts/cl_floor_summary.py`, one 25 s
six-motion clip each, clean sensors, N=8. Redoes closed-loop what §8's replay sweep
did invalidly (N1). Evidence: `results/zdrift_bexp/closed_loop/cmv_*_{analytic,learned}.json`.

    floor     analytic d(e_z)   analytic horiz    learned d(e_z)
    0            -0.198             0.006            +2.713
    3e-5         -0.178             0.021            +2.903
    1e-4         -0.162             0.033            +2.971
    3e-4         -0.141             0.044            +2.852
    1e-3         -0.117             0.060            +2.488
    3e-3         -0.085             0.058            +2.063

**The harness is deterministic and the reference used clean sensors.** Clean-sensor
`cmv=1e-3` reproduces `results/zdrift/closed_loop/best_A_cmv1e-3.json` (+2.4884 m) in
every digit of all seven motions, and the analytic arm at the same floor reproduces the
documented -0.117 m. `--imu-noise` moves the same learned configuration to +1.9750 m —
a 21% level shift, so the settings must never be mixed inside a comparison.

**The analytic response is monotone over four decades, so the sweep does not identify
an operating point.** There is no interior optimum: "minimise |e_z|" selects whichever
floor is the top of the swept range. The vertical gain is bought with horizontal error
(0.006 → 0.058 m, ~10x), which is the mechanism already on record — the floor
de-weights the contact FK measurement rather than modelling anything, and at 1e-3 it is
~30 000x the measured Σ_q. §8's warning that 1e-2 flips the sign upward puts the useful
range's edge just past 3e-3.

**The floor is not the lever for the learned arm either.** It moves learned drift over
2.06–2.97 m, ~30%, and never within an order of magnitude of the analytic ~0.1 m. §8's
replay-derived "the one config lever ContactNet responds to (+31.1%)" survives in
magnitude and not in importance: a 30% modulation of a 20x failure is not a lever.

Every floor is single-signed across the six motions on both arms, so N3's cancellation
trap is not what is happening here — this is a real monotone trade, not four terrains
averaging out.

The bounded-`exp` run therefore trains at **1e-3**, chosen to match the softplus
baseline rather than by the (void) minimisation rule, so that run differs from
`L256_A_l2vel_cmv1e-3` only in the parameterisation and its paired learning rate.

---

## 10. `bounded_exp` — the amplitude hypothesis, confirmed (2026-08-10)

`results/zdrift_bexp/L256_A_cmv1e-3_bexp`: `diag_param=bounded_exp` (sigmoid in log
space over [1e-5, 1e2]) with `peak_lr=3e-5`, otherwise identical to
`L256_A_l2vel_cmv1e-3` — same pool, objective, L, `--no-remat`, floor, 6000 steps.

**Closed loop, `cmv=1e-3`, clean sensors, same 25 s six-motion clip:**

    arm            total d(e_z)    horiz     per-motion signs
    analytic          -0.117       0.060     - only
    bounded_exp       -0.312       0.095     - only
    softplus          +2.488       1.023     + only

**8.0x better than softplus on vertical, ~11x on horizontal, and the sign is
corrected.** The learned filter no longer pushes the base *up* when the feet leave the
ground; it sinks, like the analytic filter, at 2.7x the analytic magnitude. All six
motions are single-signed, so this is not an N3 cancellation.

Training was healthy in every respect the `exp` run was not: loss 0.376 -> 0.00219
(softplus 0.00153), **zero** non-finite steps (exp had 30), `applied` 1.00 throughout
(exp collapsed to ~0.001), and it beats the analytic baseline on all four held-out
terrains (0.0474-0.0633 vs 0.0579-0.0674). Contact NIS/dof 0.062-0.073, looser than
softplus's 0.113-0.137 and closer to the analytic 0.005-0.008.

**The reading.** Amplitude was the dominant cause of the learned arm's failure, and
removing it removed the learned-specific failure mode entirely. What remains is the
*same* error the analytic filter has, not a different one — which is what §8's
null-space argument predicts as the ceiling for anything Sigma_C-shaped: Sigma_C sets
the base-vs-anchor split, and the residual common-mode sink is invisible to `H` from
any split. **This is not a success on its own terms** — -0.312 m over 25 s is ~1.2 cm/s
and not deployable — but it is the first learned Sigma_C that does not invert the sink,
and it relocates the problem from "the network is wrong" to "the measurement model
cannot see this mode".

Evidence: `results/zdrift_bexp/closed_loop/cmv_1e-3_bexp.json`, summary and span plot
in the run directory.

---

## 11. Zero velocity, process noise, and the gravity gate (2026-08-11)

Twenty closed-loop runs on the 25 s six-motion clip, clean sensors, N=8,
`contact_meas_var = 1e-3`. Batches: `scripts/overnight_{zv_qc_sweep,gate_sweep,
zv_kappa_hi,zv_mechanism}.sh`; evidence under `results/zv_qc_2026-08-11/`.
**Four leads, four dead ends** — recorded so none of them is re-derived.

### 11.1 The correction that matters most

The analytic stance Σ_C asserts a contact slip of **0.1414 m/s**, not the 3.2e-3
m/s that has been quoted. `contact.digest` applies `contact_floor` **additively**
and it is 10 000x `stance_chol^2`:

    Sigma_C = stance_chol^2 + contact_floor = 1e-8 + 1e-4 = 1.0001e-4 m^2/s
    sqrt(Sigma_C / dt) = sqrt(1.0001e-4 / 5e-3)             = 0.1414 m/s

which sits at the TOP of the measured 0.07-0.20 m/s foot roll. Two consequences:
the shipped stance trust level is already physically right, and **the "learned
Sigma_C is ~100x looser than analytic" claim does not survive into the deployed
filter** — post-floor, both arms sit at the floor. Measured independently: under
the zero-velocity block the learned arm behaves like kappa ~ 1-2, not kappa ~ 100
(§11.2).

### 11.2 Zero velocity: no trust level is a win, and the gain is a cancellation

kappa multiplies `N^v` (`--nv-scale`); kappa -> inf is "ZV off".

    kappa      e_z      horiz   NIS/dof   tiltRMS
    0.1     +0.2261     0.822    0.0086     0.850   deg
    1       +0.0802     0.669    0.0054     0.815
    10      -0.0764     0.261    0.0026     0.431
    100     -0.1100     0.087    0.0012     0.195
    1e3     -0.1155     0.062    0.0014     0.224
    1e4     -0.1179     0.061    0.0014     0.225
    1e6     -0.1165     0.060    0.0014     0.225   <- reproduces `base` to 4 dp
    off     -0.1165     0.060    0.0014     0.225

Every point is single-signed across the six motions. **No kappa beats the baseline
on both axes**; the best vertical (kappa=10, 0.0764 m, 34% better) costs 4.3x
horizontally. Graceful degradation is verified twice — as a unit-test property
(kappa*|dv| constant over kappa in {1e6, 1e9, 1e12}) and as `zvK1e6` above.

**The mechanism, from `--history` + `scripts/zv_signature.py`** — both predictions
the ZV plan wrote down in advance FAIL:

               e_z      integrated    DEPOSITED    signature
    base     -0.1159      -0.0501      -0.0658     LINEAR (R2 .999 vs sqrt .951)
    kappa=1  +0.0797      +0.1312      -0.0514     LINEAR (R2 .994 vs sqrt .930)
    kappa=10 -0.0754      -0.0111      -0.0643     LINEAR (R2 .998 vs sqrt .962)

The signature never turns sqrt(t), and the update-DEPOSITED null-mode component is
untouched (-0.0658 -> -0.0643 at kappa=10) while the headline swings 0.20 m. What
changes is the velocity-integrated term: ZV injects a POSITIVE vertical velocity
bias (mean dv_z -0.0021 -> +0.0055 m/s) that overshoots the sink. Sweeping kappa
scales that injected bias, which is why e_z passes smoothly through zero between
kappa=1 and 10. **Nothing was starved; something equal and opposite was added.**

Under the learned `bounded_exp` Sigma_C: `zvBexp` -0.1535 / 0.604 against `bexp`
-0.3123 / 0.095 — the same trade (2.0x better vertical, 6.4x worse horizontal),
and worse than the plain analytic filter on both axes.

**Do not retrain ContactNet on top of the zero-velocity block.** Gate Z6 of the
plan ("if it does not improve the analytic arm, training on top of it is
premature") is failed unambiguously.

### 11.3 Non-contact process noise is not the missing tuning

`gyro_var` and `accel_var` swept x0.1/x10 on the analytic arm (drift AND NIS/dof
together): drift moves <=13%, contact NIS/dof stays in the 1e-3 decade (0.0008 to
0.0025 over four decades of Q_c), and the two configurations that improve drift
move NIS in OPPOSITE directions. Reaching NIS/dof = 1 needs ~700x. `S` is not set
by the gyro/accel blocks of `Q_c` — consistent with CLAUDE.md 7a's `H P H^T`
argument, now confirmed from the other side.

### 11.4 The gravity gate never fires because the signal is not gravity

Instrumented per tick in closed loop (`scripts/gravity_gate_trace.py`; the
reconstructed `GravityRef` is verified against the filter's own published
`quasi_static` mask on all 5000 ticks). The gate passes **23 of 4800 walking
ticks (0.479%)**. Walking-tick medians against tolerances:

    norm  ||f|-g|/g   0.084 / 0.05   passes 27.5%
    rot   |w_raw|     0.326 / 0.15   passes 14.3%
    horiz |f_perp|    1.472 / 0.50   passes  2.2%   <- binding
    leave-one-out: dropping horiz reaches only 4.50%

Every median is 1.7-2.9x its tolerance, so the thresholds are not marginally
mis-set: during gait the specific force genuinely is not gravity (median
|f_perp| = 1.47 m/s^2 reads as 8.6 deg of apparent tilt, against a true walking
tilt error of 0.225 deg RMS). Opening the gate anyway (`--gravity-gates`, new)
does not blow up and does not fix anything: past every p99, drift -0.1165 ->
-0.1013 (13%), horizontal 0.060 -> 0.057, tilt RMS 0.225 -> 0.255 deg. This
retires the "cheap and never checked" item in `what-has-been-tried.md` 7.

Full write-up, including the process failure that cost one run:
`.claude-reports/2026-08-11-zero-vel-overnight.md` (local, gitignored).
