# ContactNet — open theoretical questions

Handoff document, 2026-07-29. Written for someone doing a theory deep dive who
was not present for the experimental work.

Every claim below is backed by a measurement in `PORT_NOTES.md`; this file states
the *questions* and the evidence that motivates them, not the narrative. Where a
number appears, the section of `PORT_NOTES.md` that produced it is named.

**Status of the empirical work.** ContactNet trains, converges, and is validated
closed-loop in sim: vertical drift −3.18 m → −0.40 m over 30 s, tilt 1.20° →
0.25°, nothing regressing. Best checkpoint `artifacts/contactnet_run4.npz`. Four
degenerate or failed runs preceded it. The questions below are what remains after
that, and they are theoretical rather than engineering.

---

## 1. A covariance cannot remove a bias — what should?

**The measurement.** The residual closed-loop sink is a *constant-rate bias*, not
accumulated noise: fitting `dz(t)` against a line beats fitting against `√t` by
16× (analytic filter) and 7× (ContactNet), and `dz/t` is constant to three
significant figures in both arms. ContactNet reduced the *rate* 8× (−0.108 →
−0.0135 m/s) by trusting contacts less.

**The question.** `Σ_C` parameterises how much the filter weights the contact
measurement. It has no representational capacity for a systematic offset in that
measurement, and `l2_velocity` contains no term that would reward correcting one.
So:

* Is the right answer a **bias state** in the InEKF (a per-contact vertical
  offset, estimated), a **model fix** (the FK contact point, §4 below), or a
  **discrete re-anchor** (`reseedContact`, §5)?
* If a bias state: where does it sit in the `SE_{N+2}(3)` structure without
  breaking I1 (bias out of the InEKF state) or I4 (tangent ordering)? Adding a
  translational bias per contact is not obviously an invariant object.
* Does CoCo-InEKF have this problem and hide it? Their Eq. (5) models a zero-mean
  contact-candidate *velocity* in the process model. Zero-mean is an assumption,
  and if the true contact velocity has a systematic vertical component during
  stance (foot roll, penetration), their formulation absorbs it into process
  noise rather than correcting it — which would produce exactly this bias.

---

## 2. L2 constrains the gain sequence and nothing else

**The measurement.** Run 4 finishes at `nis_over_dof ≈ 9.5e-3` — the filter's
covariance is ~100× too large — while its *mean* trajectory is 27-67% better than
run 2 on every metric. The estimate is good and the covariance is not.

**Settled theory** (theory doc §7.2.1, re-derived and verified): the quadratic
term `tr(S⁻¹Σ)` fixes `S` only up to a positive scalar; along `S = αΣ` it is
`k/α`, monotone to zero, so there is **no interior minimum** without a `ln det S`
term, and with coefficient `c` the optimum satisfies `NIS/dof = c` exactly.

**The open questions.**

* β-NLL's `β` is **dimension-dependent as implemented**: `det(S)^β` at `k = 6`
  weights as `s³`, not the source formulation's per-dimension `s^0.5`, and
  `losses.py` has two entry points at different `k` where the same β means
  different things. Re-derive per-dimension, or set `β = β_paper / k`? The second
  is cheaper but changes what every previously recorded β number means.
* The detached weight is **5.2e-21** on the real model and AdamW's `eps` swallows
  it. Offsetting by a constant `c` inside the `stop_grad` is exactly a uniform
  loss rescale (hence optimisation-identical) and fixes the magnitude — but
  leaves the semantics above wrong. Which failure would you rather have?
* **Does the mean regress when the covariance is calibrated?** L2 bought its mean
  accuracy partly *by* inflating scale. β-NLL removes that freedom. Nobody has
  measured whether the 27-67% mean improvement survives.

---

## 3. Identifiability: one gait makes contact condition and stride phase the same variable

**The measurement.** On a single-gait dataset the learned covariance is **79%
explained by gait phase alone** (`R² = 0.721` pooled of `log10 std_z` against
time-since-touchdown; `0.020` from the contact-trust signal). Randomising the
command vector and adding disturbances dropped it to **0.180**.

**The question.** This is an identifiability statement, and it deserves a proper
one. Given features `o = (ω, a, q, τ, p_{B→C}, v_{B→C})` over a window, and a
periodic gait, the map from features to "contact quality" is confounded with the
map from features to phase. Under what conditions on the *excitation* is `Σ_C`
identifiable at all? The empirical answer was "randomise the command", but the
condition ought to be statable — and it would tell us how much randomisation is
enough rather than leaving it to a `phase_lock` R² threshold.

Related: **what is `Σ_C` even supposed to be?** It is used as the covariance of
the contact FK measurement, but the physical quantity it must absorb — sole
compliance, contact-patch geometry, slip — is not obviously zero-mean or Gaussian,
which is the assumption the Kalman update makes about it. §1 above is one
symptom of that gap.

---

## 4. The FK contact point is the sole *site*, not the contact patch

**The measurement.** The reconstructed world velocity of the FK contact point is
**0.29 m/s in deep mid-stance** (trust > 0.95, 60 ms eroded from each edge)
against a 0.406 m/s base speed, with an independent finite difference agreeing to
three digits. A planted foot should read zero. Foot roll carries the sole site
through the world without any sliding; nothing in the recorded signals separates
roll from slip, because the rollouts store neither contact-patch position nor
contact forces.

**Why it is a theory question and not a bug report.** `p_{B→C}` is both (a) the
InEKF's contact measurement and (b) one of ContactNet's 24 input channels. If the
site is systematically offset from the patch, then the measurement is biased (§1)
*and* the network has been trained on a feature that does not mean what its name
says. The right formulation may be a contact point that migrates over the sole
during stance — which changes what "the contact is world-static" asserts, and
therefore what belongs in the process model versus the measurement model.

The `inEKF/filter.py` DECISION note argues contact *condition* belongs in the
process noise and the FK measurement is never wrong during swing. That argument
is about *whether the foot is planted*; it does not cover *where on the foot*.

---

## 5. `reseedContact` was rejected on evidence that may not apply

Deferred deliberately (`inEKF/ekf.py` TODO): *"Lucas measured no meaningful
difference on the real robot (2026-07-21)"*, with
`tests/inEKF/test_invariant_ekf.py:271` asserting its absence. CLAUDE.md §2/§3
still require it and G5 lists `InvariantEKFReseedTest`.

**The question.** That rejection was measured against hardware drift. The sim
failure is **−3.18 m in 30 s**, far larger. A discrete re-anchor does not remove
the bias in §1 but bounds its accumulation, which is a different claim from "makes
no difference". Worth deciding on theory rather than re-measuring blindly: under
what drift regime does a fire-once re-anchor help, and is that regime the one we
are in?

---

## 6. The theory PDF is stale on the point that cost a training run

`~/Documents/contactnet_theory.pdf` (21 pp, written 2026-07-27 by a different
session) is correct where it is explicit, but:

* **§8.1 never states that `x̂₀` is reset to ground truth on every segment.** It
  describes the training segment purely computationally — scan, vmap,
  checkpointing, `H` vs `L`. That single unstated fact made the objective
  degenerate: over a 128 ms horizon from zero error, IMU dead-reckoning beats any
  contact correction, so the optimal `Σ_C` is infinite. Run 1 found it.
* **§7.1's limitation paragraph is correct but incomplete.** It identifies that
  the covariance is unconstrained and treats "at least the gain sequence is
  pinned" as the safeguard. It never asks what the constrained gain sequence *is*
  — which was zero.
* **§7.2.1's no-interior-minimum machinery was aimed only at the β-NLL quadratic
  term.** The same analysis applied to the L2 objective in use would have caught
  run 1 before it ran.

An errata addendum is the single highest-value unwritten document here.

---

## Where the artifacts are

| what | where |
|---|---|
| blow-by-blow with every number | `PORT_NOTES.md` (read the ContactNet sections in order) |
| how to run any of it | `RUNNING.md`, "ContactNet" and "ContactNet in the closed loop" |
| overnight randomisation work | `.claude-reports/2026-07-29-motion-diversity-and-dr.md` |
| identifiability gate | `experiments/phase_lock.py` |
| objective non-degeneracy gate | `experiments/alpha_sweep.py` (raise `--B` to the training batch size) |
| what a checkpoint does to the filter | `experiments/check_sigma.py` (diagnostic) + `experiments/replay_eval.py` (verdict) |
| yaw-invariance derivation | `tests/contactnet/test_loss_invariance.py` docstring |
| best checkpoint | `artifacts/contactnet_run4.npz` + `data/dr/norm_constants.npz` |
