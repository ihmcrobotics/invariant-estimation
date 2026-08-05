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

**The conditioning gate is therefore expected to fire on flat ground at N=8.**
Per plan §6 this is escalated, not fixed: widening the floor would mask the very
effect under test, and the learned Σ_C is the intended remedy. A control test
confirms *independent* contacts stay well conditioned at both N=2 and N=8, so the
result is about corner correlation specifically, not about N=8 per se.

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
