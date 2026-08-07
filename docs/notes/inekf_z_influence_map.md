# Every place that can move the InEKF's height — and where the sink actually comes from

> Measured 2026-08-06 with `experiments/z_budget.py`. ContactNet is deliberately out
> of scope: every run below uses the analytic stance/swing `Σ_C` heuristic
> (`sim/sensors.py:350`). Companion to `~/Documents/filter-debugging/sink-derivation.pdf`,
> which this note refines rather than replaces.

---

## Part 1 — the map

### 1.1 There are exactly three places `p̂_z` is written

Nothing else in the filter touches the base height. Everything in §1.2 acts
*through* one of these three.

| # | Site | What it writes |
|---|---|---|
| **W1** | `pipeline/main_estimator.py:733` `init_fused_carry` → `inEKF/ekf.py` `initialize` | the seed `p₀`, and the anchors `d₀ = R₀·y(q₀) + p₀` |
| **W2** | `inEKF/propagate.py:91-96` `propagate_mean` | `p ← p + v dt + R̂ Γ₂(ω dt) ā dt² + ½ g dt²` |
| **W3** | `inEKF/correct.py:239` `apply_correction`, reached from `linear_update:357` | `p ← Γ₀(−ξ_R)·p + Γ₁(−ξ_R)(−ξ_p)` |

**W3 fires twice per tick** — once for the contact FK update and once for gravity
leveling (`inEKF/filter.py:274` and `:285`). Both go through the *same*
`linear_update`, which is why they cannot drift apart.

Three properties of W3 that drive everything in Part 2:

* **It is a left multiplication** (`X̂⁺ = exp(−ξ)X̂`, I5), so the correction acts on
  `p` as *a rotation about the world origin plus a translation*. The rotation part
  `Γ₀(−ξ_R)·p − p ≈ −ξ_R × p` **scales with how far the estimate is from the world
  origin.** This is a property of the world-centric right-invariant
  parameterisation, not a bug — but it is a real and unbounded sensitivity.
* **Neither `H` has a `p`-independent path to `v` or `R`.** The contact Jacobian is
  `H_i = [0 | 0 | +I | … −I …]` (`state.build_H:233`, `correct.contact_jacobian:402-405`):
  **zero rotation columns, zero velocity columns.** The gravity Jacobian is
  `H = [−R̂ᵀ(e_z)_× | 0 | 0 | 0]` (`gravity_update.py:284`): rotation only.
  So every write the contact update makes into `R̂` and `v̂` — and every write the
  gravity update makes into `p̂` and `v̂` — arrives *purely through the
  cross-covariance blocks of `P`*, via `K = P Hᵀ S⁻¹`.
* **Contact means are never propagated** (`propagate.py:64`, "their dynamics is all
  noise"). Anchors move only at W3. There is **no touchdown reseed** — deferred by
  decision (`inEKF/ekf.py:29-36`, `inEKF/contact.py:146`) and separately measured
  to be worth nothing (1.01×).

### 1.2 What feeds each write

```
                                     ┌─ g            config inekf.gravity
                                     ├─ dt           build_fused_estimator(dt=rp.DT*est_every)
 W2  p += v dt + R̂ Γ₂ ā dt² + ½g dt²─┤
                                     ├─ v̂ ──────────┐ (state; written by W2 and W3)
                                     ├─ R̂ ──────────┤ (state; written by W2 and W3)
                                     └─ ā  ← _boundary:713  R_mount · sensors.accel_base
                                             ├ IMU lever arm: ā is read at the base IMU
                                             │ SITE, the state tracks base_body_site
                                             ├ R_mount (build-time, from qpos0)
                                             └ NO accelerometer bias state exists anywhere
                                               in either filter — a real structural gap

 W2  R̂ ← R̂ Γ₀(ω dt)   ω ← R_mount·(gyro_base − b_base)     b_base from the joint KF

 W3  ξ = K ν,  K = P Hᵀ S⁻¹,  S = H P Hᵀ + R
     ├ contact FK update  (filter.py:274)
     │   ν = R̂ y − (d̂ − p̂)                       correct.innovation:117
     │   y = FK(q̂, q_unfiltered)                  _make_contact_kinematics:544
     │     ├ q̂            joint KF
     │     └ q_unfiltered  measured ankles (contact_fk_unfiltered; ON in the sim CLI)
     │   R = R̂ (J Σ_q Jᵀ) R̂ᵀ                     correct.map_encoder_noise:445
     │     ├ Σ_q          joint KF covariance
     │     └ contact_meas_var·I folded in         _boundary:717   (default 0.0)
     │   H = constant, no R/v columns             state.build_H:233
     │   cond_max gate masks K                    correct.py:351   (config 1.0e+9)
     ├ gravity leveling   (filter.py:285)
     │   H = −R̂ᵀ(e_z)_×, rank 2, null along e_z   gravity_update.py:284
     │   R = aniso(roll_var 2.5e-3, pitch_var 1.9e-1, pitch_disabled_var 1e4)
     │   gate = quasi_static(norm_tol .05, rot_tol .15 on RAW gyro, horiz_tol .5)
     │   ref = complementary filter, reference_tau 5 s
     └ P   ← the whole covariance history
           ├ init:  rotation_var 1e-2, velocity_var 1e-1, position_var 1e-2,
           │        contact_var 1.0        (config inekf.init)
           ├ Φ P Φᵀ  with Φ constant       state.build_Phi:198  ([p,v]=I dt is what
           │                                turns a v-error into a p-error)
           └ Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ dt  propagate.build_Qd:145
                 ├ gyro_var 1e-4, accel_var 1e-3          config
                 ├ Σ_C = L Lᵀ + contact_floor·I           contact.digest:142
                 │   L = stance_chol 1e-4 / swing_chol 1e1  sim/sensors.py:350
                 │       switched by the Schmitt/dwell ContactTrust state machine
                 │   contact_floor 1e-4 — measured LOAD-BEARING: tightening it to
                 │   1e-6 makes drift −15 m and the robot falls
                 └ Ad_X̂ carries (p)_×R̂ and (v)_×R̂ — Q_d itself grows with |p|
```

### 1.3 The two structural facts that make a height error permanent

1. **`ker H` contains common-mode translation.** With `H_i ξ = ξ_p − ξ_{d_i}`, any
   `ξ` with `ξ_p = ξ_{d_i} = δ` gives `Hξ = 0`. At N=8, `rank H = 24` of 33, null
   dim 9. Error deposited there produces zero innovation forever and no choice of
   `P`, `R` or `Σ_C` can remove it.
2. **No absolute height reference exists.** Anchors are seeded from the base
   estimate (W1) and thereafter moved only by W3. Height is observable only
   *relative to the anchors*; the assembly is free to translate.

---

## Part 2 — the trace

`experiments/z_budget.py` builds an **exactly closing** ledger: every metre of
`e_z = p̂_z − p_z^true` is assigned to one named term and the terms sum back to the
observed error (closure `0.00e+00` to `6.8e-21` on every run below). It also
replays the **shipped** `inEKF/filter.make_step` over the same recorded boundary
inputs and asserts bit-equality with its instrumented copy (`--verify`: `0.000e+00`).

Six position terms (the three write sites, with each update split into its
world-origin rotation lever and its translation), and the dominant one —
`PROP_CARRY = (v̂_z − v_z^true)·dt` — is then expanded through its own closing
velocity budget, so the whole thing collapses into one table in metres.

### 2.1 N=2 (one anchor per foot), `vx = 0.4`, 30 s, analytic Σ_C

Total sink **−2.068 m (−0.0689 m/s)**.

| source | metres | share |
|---|---:|---:|
| **contact update → `v̂_z`, then integrated** | **−1.835** | **88.7 %** |
| **contact update → `p̂_z` directly** (net of lever + translation) | **−0.203** | **9.8 %** |
| attitude error × specific force → `v̂_z` (`PV_ATT`) | −0.305 | 14.7 % |
| `Γ₁` rotation compensation over `dt` | +0.140 | −6.8 % |
| one-sided sampling of `ā` at 200 Hz (`PV_INTEG`) | +0.101 | −4.9 % |
| gravity leveling, **all four channels** | +0.032 | −1.5 % |
| **IMU lever arm** `α×r + ω×(ω×r)`, r = (−0.087, +0.012, −0.081) m | **+0.001** | **0.05 %** |
| propagation 2nd-order remainder | +0.000 | 0.0 % |

**The contact FK update is 98.6 % of the sink** (−2.038 of −2.068 m), and **86 % of
that arrives through the velocity channel** — the channel `H` has no columns for.

This sharpens the earlier stage attribution ("propagation 87 %, contact update 13 %").
Propagation is the *integrator*; the velocity error it integrates is *manufactured
by the contact update*. Both statements are true; only the second is actionable.

Corroborating splits from the same record:

* base moved −0.203 m, mean anchor moved −2.016 m, **common-mode deposit −1.412 m**
  (68 % of the total) — the part `ker H` guarantees is unrecoverable.
* gravity-leveling gate open on **0.42 %** of ticks (25 of 6000); its total
  contribution across all four of its channels is **+0.03 m, i.e. it opposes the sink**.

### 2.2 The world-origin lever arm is real and unbounded — `--x0 50`

Re-running the identical rollout with the robot started 50 m along +x (flat ground,
translation-invariant policy) leaves the **velocity budget bit-identical** and changes
only the two rotation-lever terms:

| | `x0 = 0` | `x0 = 50` |
|---|---:|---:|
| `CONT_ROT` | −0.544 m | **−6.114 m** |
| `CONT_TRANS` | +0.341 m | **+4.948 m** |
| net into `p̂` | −0.203 m | **−1.166 m** |
| total sink | −2.068 m | **−2.857 m (+38 %)** |

Within a single `x0 = 0` run the same scaling shows up over time: `CONT_ROT`'s rate
triples (−0.0067 → −0.0295 m/s) as the robot walks from 1.8 m to 9.8 m out, while
the rate *per metre of lever* is flat (−0.0037 → −0.0030 s⁻¹). The identity is exact:
`corr(CONT_ROT, ξ_y p_x − ξ_x p_y) = −0.99999`.

**Consequence: the drift is not constant. It grows with distance from wherever the
filter was initialised**, because `exp(−ξ)` on the left rotates `p̂` about the world
origin. The driver is a sustained pitch correction — +0.111 rad accumulated over
30 s, mean +1.9e-5 rad/tick — which the contact update writes into `R̂` through `P`
despite `H` having no rotation columns.

### 2.3 N=8 (deployed geometry) — the diagnosis changes

Total sink **−0.215 m (−0.00716 m/s)**, i.e. **9.6× better than N=2**, and no longer
dominated by one term:

| source | metres |
|---|---:|
| one-sided sampling of `ā` (`PV_INTEG`) | −0.245 |
| attitude error × specific force | −0.142 |
| contact update → `p̂_z` (net) | −0.080 |
| contact update → `v̂_z` (net) | **+0.075** (now *opposes*) |
| `Γ₁` rotation compensation | +0.150 |
| gravity leveling, all channels | +0.027 |

The eight-anchor geometry very nearly neutralises the contact update's velocity
leak. What is left is the propagation's own specific-force integration residual and
the attitude-error projection.

**Honest caveat on attributing `PV_INTEG` vs `CV_CONT_*`.** These two are an
injector and its servo: propagation injects a systematic vertical specific-force
error, the contact update sees the resulting gap and removes it. The ledger is
exact, but "cause" is shared — what escapes is the *imbalance*. The standing control
makes this unmistakable.

### 2.4 Standing control, 30 s

Total **+0.0008 m**. `PV_INTEG` = −0.196 m/s and `CV_CONT_TRANS` = +0.196 m/s cancel
to five digits. The loop is a working servo; the sink is entirely a gait phenomenon.
(Consistent with the 2026-07-27 finding that standing does not drift.)

Phase breakdown while walking: single support is 70.6 % of ticks and double support
29.2 %, and the two contribute *opposing* contributions an order of magnitude larger
than the net — the sink is a small residue of a large cancellation across the gait
cycle, which is why per-run numbers are noisy.

### 2.5 The dominant channel cannot be switched off — `--mask-k rotvel`

Zeroing the rotation and velocity rows of the contact update's `K` severs exactly
the cross-covariance path that §2.1 blames, leaving the position and anchor
corrections untouched. Result over the same 30 s:

| | shipped | `--mask-k rotvel` |
|---|---:|---:|
| height error | −2.07 m | **−6.53 m** |
| attitude error, tail RMS | 0.63° | **7.15°** |
| velocity error, RMS | 0.084 m/s | **12.3 m/s** |

**The filter falls apart, and that is the correct result.** In a contact-aided
InEKF the `p − d_i` measurement *is* the only observation of base velocity and the
only non-gravity observation of attitude — both reach the state solely through
`P`'s cross-covariance blocks, because `H` has no columns for them. So the channel
carrying 88 % of the sink is not a leak to be plugged; it is the filter's primary
mechanism, operating slightly off-calibration.

This kills the obvious fix and relocates the lever: **not whether the contact update
writes into `v̂` and `R̂`, but with what gain** — i.e. `P_{v,p}` versus `P_{v,d_i}`,
which is set by `Φ`'s `[p,v] = I dt` block accumulating `P_{v,p} ≈ P_{vv} dt` against
contact blocks fed only by `Σ_C`. That ratio is the same "base-vs-anchor injection
ratio" the null-space derivation named, seen from the velocity side.

### 2.6 Ruled out, quantitatively

| candidate | measured contribution |
|---|---|
| IMU lever arm (`α×r + ω×(ω×r)`) | **+0.001 m of −2.068 m (0.05 %)** — confirms the 2026-07-27 A/B by direct decomposition rather than by difference of runs |
| gravity leveling | **+0.03 m, wrong sign to be the cause**; gate open 0.4 % of walking ticks |
| propagation 2nd-order (`Γ₂ ā dt²`, `½g dt²`) | +0.0003 m |
| gravity constant / `R_mount` | inside `PV_INTEG`, which is +0.10 m at N=2 (opposes) |

---

## What to attack next

1. **The *gain* of the contact update's write into `v̂` and `R̂`** — 98.6 % of the N=2
   sink flows through it, and §2.5 shows it cannot be masked (it is the filter's only
   velocity observation). So the target is the ratio `P_{v,p} : P_{v,d_i}`, set by
   `Φ`'s `[p,v] = I dt` accumulating `P_{v,p} ≈ P_{vv} dt` against contact blocks fed
   only by `Σ_C`. Concretely: sweep `accel_var` (which sets `P_{vv}`, hence the
   numerator) against `stance_chol`/`contact_floor` (the denominator) and watch the
   ledger's `via v: CV_CONT_*` row rather than the end-to-end drift — the ledger
   separates injection from servo response, which a drift number cannot.
2. **The world-origin rotation lever** (§2.2) — unbounded in distance travelled.
   Nothing in the current design bounds it. Re-anchoring the world frame
   periodically (or a body-centric/left-invariant formulation for odometry) are the
   two structural answers.
3. **`PV_INTEG`** at N=8 — a systematic ~6.5 mm/s² vertical specific-force
   integration error whose sign flips between standing and walking, i.e. a
   sampling artifact of an impulsive signal at 200 Hz, not a bias. A trapezoidal or
   two-sample `Γ₁` integration would test it.
4. Everything upstream of the filter (joint KF `q̂`, `q̇̂`, `b̂`, the IMU, gravity
   leveling) is **measured innocent** and stays that way — consistent with the
   earlier four-channel investigation.

## Reproducing

```bash
uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --verify     # §2.1
uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --x0 50      # §2.2
uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4  # §2.3
uv run python experiments/z_budget.py --ticks 1500 --vx 0.0              # §2.4
uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --mask-k rotvel  # §2.5
```

Raw per-tick records: `results/zbudget_{c1,c1_x50,n8,stand}.npz`.
Always pass `--verify` after touching `inEKF/filter.py` or the tracing step — it is
the only thing keeping the instrumented copy honest.
