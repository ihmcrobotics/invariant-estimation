# branch_out.md — move ContactNet to the process socket, wire N^v

> **Agent dispatch document.** Written 2026-07-29 for execution by Claude Code on
> a machine with the pinned toolchain (Python ≥3.12, `jax==0.10.2` CUDA, a GPU)
> and working `git`. The analysis that motivates it was done in a sandbox that
> had none of those, so **nothing below has been executed or tested.** Treat every
> code claim as a specification to verify, not a result.
>
> **Companion documents:** `CLAUDE.md` (invariants I1–I10, constant-graph rules),
> `PORT_NOTES.md` §"The diagnostic ran" and §"Run 4 converged" (the measurements),
> `Z_BIAS_FACTS.md` (the structural argument), `TEST_SUITE_MAP.md` (gates).
> **Precedence on conflict:** ported Java tests > paper > Java implementation.

---

## 0. Mission

Branch: **`contactnet/process-socket`**, off the current worktree HEAD.

Three changes, in order, each independently revertable:

1. **Phase 0 — analytic ablation (no training).** Decide whether the process
   socket is even the right lever, before spending a GPU hour. **This phase can
   cancel phases 1–3.** Do not skip it.
2. **Phase 1 — socket move.** ContactNet's `L` feeds `InEKFInputs.contact_chol`
   (the process noise on the stance anchor `d_i`) and nothing else.
   `contact_meas_chol` returns to zeros.
3. **Phase 2 — wire `N^v`.** The contact zero-velocity constraint, using the
   joint KF's `Σ_q̇` that is already plumbed to the boundary and unconsumed.

Then retrain and re-measure with `experiments/z_bias_diag.py`.

### Design decision, already settled — do not re-litigate

**The network outputs `L`; `Σ_C = L Lᵀ` is the process-noise covariance on the
stance anchor `d_i`, and nothing more.** This is CoCo-InEKF Eq. (5) parity
(`Wṗ_Ci = −WR_B · Bw_Ci`, `Bw_Ci ~ N(0, BΣ_Ci)`, network called inside
Prediction, Alg. 1 line 1). It **replaces** the `ContactTrust` heuristic on the
InEKF path. It is not a modulation of the heuristic, not a multiplicative
correction, and not a second head.

`sensors.contact` (the joint-KF stance-anchor trust mask, consumed at
`main_estimator.py:703`) is a **different consumer** and is unchanged. Only
`sensors.contact_chol` is superseded.

### Why (one paragraph — the full argument is in `Z_BIAS_FACTS.md`)

For one contact `H = [0 0 I −I]` over `(R, v, p, d)`, so
`K = P Hᵀ (H P Hᵀ + N)⁻¹` and the `v`-row of `P Hᵀ` is `P_vp − P_vd`. `N` appears
only inside the inverted factor: it scales the correction and reweights residual
axes, but **cannot change how a residual is apportioned between the base and the
anchor.** That apportionment is pure prior, hence pure process noise. Measured:
the sink is an integrated velocity bias (`r = +0.958`, ratio 1.068 over 15
rollouts), it is not the propagation (specific-force error flips sign on the
control set while the sink worsens), and it is not the contact point (sole drift
`+0.00213 ± 0.00034 m/s`, wrong sign, never penetrates). ContactNet has been
holding the one knob that provably cannot reach it.

---

## 1. Phase 0 — the ablation that can cancel this whole branch

**Goal:** determine whether the sink is sensitive to *process-side* covariance
timing at all. No network, no training, recorded data only.

The mechanism under test: at liftoff the foot rises, so `y_z` grows while `d̂` is
still pinned where stance left it ⇒ `ν_z > 0` ⇒ `exp(−Kν)` drives the base down.
The fraction landing on the base is `P_pp/(P_pp + P_dd)`, and `P_dd` is at its
**tightest** exactly then, because the foot just spent a whole stance being told
it was world-static. Late in swing `P_dd` is huge and the descent is absorbed by
the anchor. One rectified downward dose per step per foot.

Build on `experiments/replay_eval.py::run_arm`, which already replays from a
truth seed under a supplied contact factor. Add a `--socket {meas,process}` flag
so the same harness can drive either field, and run:

| arm | `contact_chol` | tests |
|---|---|---|
| **A** | current heuristic, 1e-4 stance / 1e1 swing, Schmitt-switched | baseline |
| **B** | heuristic, but inflated `N` ticks **before** liftoff; sweep `N ∈ {0, 10, 25, 50, 100}` | is it the *timing*? |
| **C** | heuristic with stance value swept `1e-4 → 1e-3 → 1e-2` | is it the tightness? |
| **D** | run 4 on `contact_meas_chol` (unchanged wiring) | control |

Liftoff is known offline from `truth.contact_fn`, so arm B is a backward shift
of the existing mask — no predictor needed. This is deliberately non-causal; it
is a mechanism test, not a deployable filter.

**Metrics** — reuse `experiments/z_bias_diag.py`:
`slope(e_pz)`, `mean(e_vz)`, and their ratio; plus `vel_rms`, `tilt_deg`.

**Decision rule.**
* Sink drops materially in **B** or **C** ⇒ mechanism confirmed, ContactNet
  exonerated, proceed to Phase 1.
* Sink flat across B and C ⇒ **stop.** The process socket is not the lever and a
  retrain would be wasted. Reopen the ranking in `Z_BIAS_FACTS.md` §5 and
  instrument `contact_innovation` (see §6) before touching anything else.

Expect **C to make it worse in the tightening direction**: `contact_floor`
1e-4 → 1e-6 was already measured at −15 m of drift, 18° tilt, robot falls. That
is a *sign prediction* the mechanism makes and the record confirms; if the sweep
does not reproduce it, the harness is wrong, not the theory.

---

## 2. Phase 1 — the socket move

### 2.1 Call sites (exhaustive)

| file:line | now | becomes |
|---|---|---|
| `pipeline/main_estimator.py:722` | `inekf_inputs._replace(contact_meas_chol=meas_chol)` | `_replace(contact_chol=chol)` |
| `contactnet/rollout.py:98` | `segment.inputs._replace(contact_meas_chol=L_c)` | `contact_chol=L_c` |
| `contactnet/rollout.py:141` | `inputs._replace(contact_meas_chol=L_c)` | `contact_chol=L_c` |
| `contactnet/rollout.py` `make_warm_in` | broadcasts `σ₀·I` into `contact_meas_chol` | into `contact_chol` |
| `contactnet/online.py` `make_provider` | docstring + fallback describe the measurement socket | rewrite for process; see §2.3 |
| `contactnet/dataset.py:652` `measure_p0` | `contact_chol` = const, `contact_meas_chol` = 0 | see §2.4 — **P0 must be re-measured** |

Rename the local `meas_chol` at `main_estimator.py:722`; leaving the old name is
how the next reader concludes the measurement socket is still live.

### 2.2 What does **not** change

* `inEKF/contact.py::digest` — already the consumer of `contact_chol`. Same
  `reconstruct_cov → apply_floor` path, different supplier. **No InEKF core edits
  in this phase.**
* `inEKF/propagate.py` — `Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Δt` (I3). The `Ad_X̂` stays.
* `contactnet/features.py`, `normalize.py` — untouched, so the frozen norm
  constants and the checkpoint↔norm pairing discipline carry over verbatim.
* `sensors.contact` → joint-KF anchors. Untouched.

### 2.3 The warm-up fallback inverts

`online.make_provider` currently emits **zeros** before the ring buffer fills,
because zeros in the *measurement* socket reproduce the shipped filter exactly.
**Zeros in the process socket are catastrophic** — `Σ_C = 0` pins every anchor,
including swing feet, as perfectly world-static.

Replace the fallback with the **heuristic value** for those ~400 ticks:
`sensors.contact_chol`, which the caller already has. That keeps the warm-up on
the shipped filter's behaviour rather than on a degenerate one, and it must stay
a `jnp.where`, not a branch (I7).

### 2.4 `contact_floor` becomes safety-critical

`digest` applies `Σ ← Σ + contact_floor·I` with `contact_floor = 1e-4`. With the
heuristic supplying `Σ_C` this was a conditioning nicety. With the **network**
supplying it, this floor is the only thing standing between a mis-prediction and
a pinned swing foot. Document it as such in `inEKF/contact.py`, and note that the
network's own `eps` (on the diagonal of `L`) now floors the same object — two
floors on one quantity. Reconcile explicitly; do not let them silently double.

`PORT_NOTES.md` §"ContactNet seam" says `eps` is the measurement-socket lever and
`contact_floor` the process one. After this change **both act on the process
socket.** That note needs correcting in the same commit.

---

## 3. Phase 2 — the `N^v` zero-velocity block

`filter.py:264` already defines `contact_velocity_noise(J_dot, sigma_q_dot)`.
It has **zero call sites**. `JointFilterOutput.sigma_q_dot` is computed, plumbed
and unconsumed.

### 3.1 The observation matrix is constant — derive it, then assert it

For a static contact, the world velocity of contact `i` is zero:

```
0 = v + R u ,      u = ω̄ × h_{p,i}(q̂) + J_{C_i} q̇̂        (all measured)
```

Residual `r = v̂ + R̂ u`. Under the port's left perturbation `X̂ = exp(ξ)X` (I5),
`v̂ ≈ v + ξ_R × v + ξ_v` and `R̂ ≈ (I + ξ_R×)R`, so

```
r ≈ (v + Ru) + ξ_v + ξ_R × (v + Ru) = ξ_v          since v + Ru = 0 at truth
```

⇒ **`H_v = [0₃ I₃ 0₃ 0₃…]`, exactly constant and state-independent.** The
`ξ_R` coupling cancels identically because the quantity being observed is zero.
This is the same right-invariance property that makes the position block
constant, and it is the *first* observation in the filter with a `ξ_v` block —
today nothing observes velocity directly, which is precisely why the bias
survives.

**Add a test asserting `H_v` is bit-identical across two random states**, mirroring
`testJacobianStructureAndStateIndependence`. If it is not, the derivation above is
wrong and the block must not ship.

### 3.2 It stacks; it does not fold

`correct.py`'s existing TODO is binding: `N^v` is a **separate measurement block
with its own `H` rows**, stacked below the position block. Do **not** add
`J_Ċ Σ_q̇ J_Ċᵀ` into `N^p`. They are noises on two different measurements.

Noise: `R̂ (J_{Ċ_i} Σ_q̇ J_{Ċ_i}ᵀ + gyro term) R̂ᵀ`, rotated to world by the same
`rotate_measurement_covariance` the position block uses, on the **prior** state.

### 3.3 The honest problem with this block

**The FK position measurement is true during swing; the zero-velocity
measurement is not.** `inEKF/filter.py`'s DECISION note argues against masking
because "the encoders still locate the foot relative to the base perfectly well"
— that argument holds for position and **does not transfer** to zero-velocity,
which is simply false for a foot in flight.

So `N^v` needs a swing-inflated `R`, which reintroduces a contact-condition input
on the measurement side. Options, in preference order:

1. **Drive `R_v` from the same `Σ_C` the network emits.** Physically coherent —
   `Σ_C` *is* the contact-velocity covariance in CoCo's formulation, which is
   exactly what this measurement's validity depends on. One learned quantity,
   two consumers, no new socket.
2. Gate `R_v` on the `ContactTrust` Schmitt output (`sensors.contact`).
3. Ship Phase 2 behind a config flag, default **off**, and land it after Phase 1
   is measured.

**Take option 3 for the first branch regardless** — Phase 1 and Phase 2 must be
measurable separately or the retrain result is uninterpretable. Note in
`PORT_NOTES.md` that this block is a **departure from CoCo-InEKF**, which has no
velocity measurement at all (their only correction is Eq. (8)).

---

## 4. Phase 3 — initialization, and the trap that will eat this branch

### 4.1 `sigma_0 = 1e-4` is a measurement-socket number and must not be reused

Its justification (`config.py`) is "three orders below `J Σ_q Jᵀ = 1.26e-5 m²`".
That argument does not exist on the process socket. The heuristic's range there
is `1e-4` (stance) to `1e1` (swing) **as Cholesky factors** — variance `1e-8` to
`1e2`.

### 4.2 The dynamic range is fine; the gradient is not

`network.forward` emits `softplus(o) + eps`. Pre-activations of `−9.2` and `+10`
give `1e-4` and `10.0` — a benign ~20-unit span for a linear head, so "ten orders
of magnitude" is not the problem it sounds like.

The problem is that `d softplus/dx = σ(x) ≈ 1e-4` at the stance end. **The
network receives almost no gradient exactly where it is outputting tight stance
covariances.** Expect slow learning in stance and fast learning in swing. Log
`grad_norm` split by predicted magnitude; if stance is frozen, that is the cause,
and the fix is a reparameterisation (log-scale output head), not more steps.

### 4.3 Constant init is the run-1 configuration — do not ship it

`network.init` sets head `W = 0`, so **iteration 0 emits a constant `σ₀·I` for
every input, at every gait phase.** On the process socket that *is*
`freeze_contact_chol=True`, which `experiments/measure_tstar.py` measured at
**10.2× worse in body-frame velocity than not using contacts at all**, and which
run 1 escaped only by driving `Σ_C → ∞`. Starting there again will reproduce run
1.

Three options:

1. **Supervised warm-start (recommended).** Regress the network onto the
   heuristic `contact_chol` — already recorded in every rollout as
   `inputs.contact_chol` — for a few hundred steps, then switch to BPTT L2. The
   network starts *at* the heuristic including its swing/stance switching, and
   BPTT only has to improve on it. Cheap, and it removes the cold-start problem
   entirely. The features carry the needed information (the torque channels are
   the load signal).
2. **Conservative constant init**, `σ₀` at the swing value (`1e-1`…`1e0`). Every
   foot starts loose; the filter degrades to "no contact information", which is
   the *better* end of the measure_tstar comparison, and the network must learn
   to tighten. Safe but slow.
3. Constant init at the stance value — **this is run 1. Do not.**

Whichever is chosen, `dataset.py`'s `freeze_contact_chol` flag and
`_constant_contact_chol` become obsolete or invert their meaning. Resolve them in
this branch rather than leaving a flag whose docstring describes the old socket.

### 4.4 `P0` must be re-measured

`dataset.measure_p0` burns in with `contact_chol` at the constant and
`contact_meas_chol` at zero, "exactly the conventions a training segment runs
under". After Phase 1 that sentence is false. Re-run it under the new wiring and
write a new `artifacts/p0_*.npz`; **do not reuse `p0_dr.npz`.** A stale `P0` seeds
every segment from the wrong prior and moves `nis_over_dof` for reasons unrelated
to the network.

---

## 5. Gates

Run in order. Do not proceed on a red gate.

| gate | check |
|---|---|
| **P0** | Phase 0 decision rule (§1) satisfied — mechanism confirmed |
| **P1.a** | Full ported suite green (`TEST_SUITE_MAP.md`); no regression from HEAD |
| **P1.b** | **jaxpr-constancy (I7):** `fused_step` hashes identically across differing contact masks and gate states. This is the constant-graph proof and the socket move touches the scan carry |
| **P1.c** | float64 at every boundary (I8); `_assert_float64` still passes in `dataset.prepare` |
| **P1.d** | Network output → `contact_chol` proved **live**: two runs differing only in `--contactnet` are bit-identical through the ring-buffer warm-up and diverge at the first ready tick. This is the cheap wiring proof from `PORT_NOTES.md`; re-run it after any seam change |
| **P2.a** | `H_v` bit-identical across two random states (§3.1) |
| **P2.b** | With `N^v` disabled by flag, the filter is **bit-for-bit** the Phase-1 filter |
| **P3** | `nis_over_dof`, `grad_norm`, `applied_frac` logged; `applied_frac < 1.0` means the conditioning gate is skipping updates and the loss is scoring a fiction |

**Acceptance for the branch** — `experiments/z_bias_diag.py` on held-out
rollouts:

* `slope(e_pz)` materially below run 4's `−0.0135 m/s`;
* the `slope(e_pz) / mean(e_vz)` ratio **stays ≈ 1** (if it departs from 1, the
  sink has stopped being an integrated velocity bias and the mechanism has
  changed — that is a finding, not a pass);
* no regression in `vel_rms`, `tilt_deg`, or worst-window performance.

---

## 6. Instrumentation to add while you are in here

Both are one-field changes to `sim/collect.py` and both close gaps the analysis
had to work around:

1. **Log `contact_innovation`** (already emitted as `InEKFOutputs.contact_innovation`,
   just not saved). Everything in §1 infers the sign and timing of `ν_z` from
   mechanism plus corroborating experiments. Saving it makes the rectified
   per-step dose directly observable.
2. **Log the `P_dd` / `P_pp` diagonal** through a gait cycle. If `P_dd` is flat,
   the §1 asymmetry argument is **wrong** and Phase 0 arm B will show nothing.
   Checking this first is cheaper than running the ablation blind.

---

## 7. Named traps

* **Zeros in the process socket.** The `contact_meas_chol=0` idiom appears in
  `measure_p0`, `make_warm_in`, `online.make_provider`'s fallback, and several
  tests. Every one means "shipped filter" today and "pin all anchors rigid" after
  the move. Grep for `contact_meas_chol` and `zeros_like` together and fix each.
* **Reusing `sigma_0 = 1e-4`** (§4.1) and **constant init** (§4.3).
* **Stale `P0`** (§4.4) and **stale norm constants** — the latter are actually
  fine here (features unchanged), which makes it easy to forget the former is not.
* **Folding `N^v` into `N^p`** (§3.2).
* **`contact_floor` double-counting with the network's `eps`** (§2.4).
* **Dropping `Ad_X̂` from `Q_d`** (I3). The socket move puts a learned quantity
  into `Q_c`; resist any urge to "simplify" the conjugation while in there.
* **The Schmitt trigger is not missing.** `sim/sensors.py::ContactTrust` is a
  live port of `FootSwitchContactProbabilityProvider` (enter 0.35 / stay 0.25 /
  40 ms dwell, immediate release) and it currently drives *both* consumers. Every
  measurement in `PORT_NOTES.md` was taken **with** it running. What is absent is
  `reseedContact` + `TouchdownReseedLatch`. Do not "add the Schmitt trigger".
* **Do not implement the reseed in this branch.** It is a confound. It is also
  the obvious next lever if Phase 0 confirms the mechanism — record that, and
  keep `tests/inEKF/test_invariant_ekf.py::test_reseed_is_not_implemented` green.

---

## 8. Out of scope

The **learned moving-mean extension** — `Bw_Ci ~ N(v̂_c, Σ_C)` rather than
`N(0, Σ_C)`. Both CoCo Eq. (5) and this branch keep the contact noise **zero-mean**,
so neither can represent a genuinely non-zero-mean contact error. Moving the
socket converts an *accumulating* drift into a *bounded offset* if the
apportionment argument holds; it is not a bias correction.

This branch is the **prerequisite** for the moving-mean work (the mean has to
live on the process socket), not a substitute for it. Keep them separate so the
retrain result stays interpretable.

Also out of scope: the accelerometer-bias state. `sim/sensors.py` injects no
accel bias, and the measured specific-force error is *positive* on the control
set that sinks worst — so it cannot be the sim cause, and adding it here would
show zero change while confounding the result. It remains a real **hardware**
exposure (no accel-bias state exists anywhere in the port; the joint KF carries
`b_omega` only), and it needs its own branch and its own argument.
