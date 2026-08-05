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
| Does β-NLL help? | **No — it is much worse** | R1: vel RMSE 0.83 vs 0.091 m/s baseline |
| Does DR work as configured? | **No — it was unwalkable** | 7/8 rollouts fell, incl. on flat |
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

## 4. Env-DR was not survivable as configured

7 of the first 8 DR rollouts fell — **including one on flat ground**, which is what
ruled out the terrain as the cause.

| Ablation | Result |
|---|---|
| Constant-command sim sweep, every DR arm | all OK, ≤3° peak tilt |
| Real collect path, `env_dr=False`, seeds 1 & 4 | both OK |
| Real collect path, `env_dr=True`, seeds 1 & 4 | seed 4 **FELL** |
| friction only (pushes off) | seed 4 **FELL** |

Attribution: **friction**, not pushes. The tail reached μ = 0.15 — effectively ice.
Note the constant-command sweep cleared every arm: the falls need the DR *and* the
randomised command schedule together, so a sim-only sweep alone would have missed
this. Tail widened to (0.45, 0.70), still a genuine slip regime against a ~1.0
nominal. A tail the policy cannot survive yields no rollouts, which trains nothing.

---

## 5. Run ladder

*(filled in as runs complete)*

| Run | N | data | loss | held-out vel RMSE | NIS/dof | applied | verdict |
|---|---|---|---|---|---|---|---|
| baseline (analytic) | 2 | flat | — | | | | reference |
| R0 | 2 | flat | L2 | | | | |
| R1 | 2 | flat | β-NLL | 0.83 m/s | 2.36 | 1.00 | **FAIL** |
| R3 | 8 | DR | β-NLL | | | | |

**R1 (β-NLL alone, N=2, flat) — FAIL.** Held-out velocity RMSE **0.83 m/s** against
the analytic baseline's **0.091** — 9× worse — with NIS/dof 2.36 vs 0.056 and a
training loss driven negative. This is plan §6's "mean-excuse" failure mode
appearing on the first rung, and it is why an N=8 run under β-NLL would confound
the contact count with a broken objective.

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
