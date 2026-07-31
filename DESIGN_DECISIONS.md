# Design decisions

Deliberate choices whose *symptoms* look like bugs. If the filter is behaving in
a way that surprises you, check here before digging into the math.

Each entry records **what**, **why**, **what it costs**, and **the test that
guards it** — so a future change that quietly reverses the decision fails loudly
rather than drifting.

For the Java-port reconciliations (tolerances, adapted tests, RNG substitutions)
see `PORT_NOTES.md`. For tuning values see `config/filter_cfg.yaml`.

---

## 1. There is no contact mask — contact condition rides in `Σ_C` alone

**Decided:** 2026-07-22 (Lucas). **Code:** `inEKF/filter.py` (`DECISION` note in
the module docstring). **Guarded by:**
`tests/inEKF/test_filter.py::test_large_contact_covariance_isolates_a_swing_foot`.

### What

The InEKF has **no per-foot trust mask and no measurement kill-switch**. A foot
in swing is expressed *only* by a large `Σ_C` — the ContactNet Cholesky factor,
which reaches the contact block of `Q_d` through `contact.digest`.

If you are hunting for the place that "turns off" a foot: it does not exist, on
purpose.

### Why

The FK measurement `y_i = R̂ᵀ(d̄_i − p̄)` is **not wrong during swing**. The
encoders still locate the foot relative to the base perfectly well. What breaks
in swing is the assumption that `d_i` is world-static — and that assumption lives
in the **process** noise, not the measurement noise.

So inflating `Σ_C` is the physically correct lever. Masking the measurement
treats a true observation as false, and puts the contact condition in two places
at once.

`Σ_C` is also strictly more expressive than a scalar trust weight: a full 3×3
covariance can say *"this foot slides along the surface but not through it"*,
which no single number can.

### Measured

With `Σ_C = 1.0` for 100 swing ticks (`P_dd` grows to 0.11 vs 0.01 planted), an
8 cm foot displacement arriving as a measurement is:

| | base `|Δp|` | absorbed by anchor `d₁` |
|---|---|---|
| large `Σ_C` (swing) | 3.7 mm | 0.0765 / 0.08 = **96%** |
| small `Σ_C` (planted, control) | 27.9 mm | 0.0532 |

**7.6× attenuation** of base corruption, 96% of the discrepancy absorbed by the
anchor.

### What it costs — read this part

* **A swing foot still receives a small base correction.** It is 3.7 mm above,
  and it is *correct*: `P_pp/(P_pp + P_dd) ≈ 8%` of the residual genuinely
  belongs to the base under that prior. It shrinks further as the swing
  continues and `P_dd` keeps growing.
  **If it is too large, `Σ_C` is too small. Do not reach for a mask.**
* There is no hard override for a bad contact measurement. An encoder fault is
  `Σ_q`; everything else is `Σ_C`.
* Swing-foot behaviour now depends on ContactNet (or the heuristic default)
  producing a genuinely large `Σ_C`. A ContactNet that under-predicts swing
  covariance will corrupt the base, and the symptom will look like a filter bug
  rather than a model bug. Check `Σ_C` first.

### The trap this replaced

An earlier version of `filter.py` carried a per-foot `contact_mask` that blended
`R_eff = w·Np + (1−w)·R_LARGE·I`. Two things were wrong with it:

1. **It was a step function pretending to be continuous.** With `Np ~ 1e-6` and
   `R_LARGE = 1e12`, `w = 0.999` already contributed `1e-15` of the information;
   only `w` within ~`1e-12` of 1.0 meant anything. A "0.5 trust" foot was fully
   off.
2. **It was imported from the wrong filter.** CLAUDE.md §4's `R_LARGE = 1e12·I₃`
   masking rule governs the **joint KF's stance anchors** (§2 "trusted feet →
   anchors"; the oracle is checked in the G7 stacked-oracle port), *not* the
   InEKF contact update. The InEKF's governing invariants are **I2** (contacts
   permanently in state) and **§7** (contact condition expressed through `Σ_C`),
   neither of which mentions masking.

`R_LARGE` masking is still correct — for the joint KF at G7, where it belongs.

---

## 2. `Q_d` uses the first-order `·Δt` discretisation, not the exact integral

**Decided:** 2026-07-21 (Lucas). **Code:** `inEKF/propagate.py::build_Qd`
(`TODO(van-loan)`). **Guarded by:**
`tests/inEKF/test_propagate.py::test_build_Qd_keeps_the_adjoint`.

`Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt` — CLAUDE.md **I3** / paper Eq. 38 verbatim,
matching the working Java estimator.

**Cost:** the first-order form overshoots the cross terms relative to the exact
integral (≈3× on position variance, 2× on position–velocity covariance). If
NEES/NEES at G10 shows the covariance is inflated, the exact closed form is a
drop-in replacement — `A` is nilpotent so the integrand is a degree-≤4 polynomial
and the integral is closed-form. Deliberately not done for v1 because the Java
reference does not.

**Do not "clean up" the `Ad_X̂`.** It is not a no-op even for isotropic `Q_g, Q_a`:
`Ad_X̂` carries `(v)_× R̂` and `(p)_× R̂` in its first block-column, so the
conjugation generates genuine cross terms. `test_build_Qd_keeps_the_adjoint`
fails if it is removed.

---

## 3. Touchdown re-seed IS implemented, and is off by default

**Decided:** 2026-07-21 (Lucas) to defer; **implemented 2026-07-30** as an ablation
arm. **Code:** `inEKF/reseed.py`, wired in `inEKF/filter.make_step`.
**Guarded by:** `tests/inEKF/test_reseed.py` (16) and
`tests/inEKF/test_invariant_ekf.py::test_reseed_is_implemented_but_off_by_default`.

`reseed_contacts` re-anchors a contact slot by a covariance congruence
(`P_dd = P_pp + R N Rᵀ`, `P_θd = P_θp`) under a fire-once `TouchdownReseedLatch`
(0.5 / 0.1 / 100 ticks). Both Java test classes are now ported, including the
zero-release property and the 200k-tick chatter property.

**Default `None`**, so it is absent from the traced graph unless asked for: runs
1-6 and every replay and closed-loop number on record were produced without it,
and they stay comparable only while that is true. Enable with
`build_fused_estimator(reseed=True)` or `run_estimator.py --reseed`.

**Why the original deferral was right, now with a sim measurement behind it.**
Lucas measured no meaningful difference on hardware. The port reproduces that, and
the mechanism is now visible: over 10 s of `data/dr5/flat_seed000`, the
pre-re-seed anchor discrepancy at the ticks the latch fires is **0.4 mm mean,
1.2 mm max**. There is almost nothing to re-anchor, because **this InEKF never
releases a contact** — it runs the FK update on all `N` contacts every tick with
no per-foot gate (the DECISION note in `inEKF/filter.py`: contact condition rides
entirely in the process `Σ_C`), so the inflated swing-phase `Σ_C` lets each anchor
track its own foot continuously. The stale-anchor problem a re-seed exists to fix
is already handled by the process-noise mechanism.

**Corollary worth keeping:** a re-seed would matter much more in a filter that
*did* gate the contact update per foot. If that ever changes, revisit this.

**What it cannot do, in any variant:** make global `x`, `y` or yaw observable.
Those are unobservable in a proprioceptive InEKF — `P0`'s
`diag(R) = [7.1e-5, 7.1e-5, 1.0]` is the filter saying so correctly — and a
re-seed changes only the *rate* at which they drift.

---

## 4. Contact trust (`FootSwitchContactProbabilityProvider`) is not ported

**Decided:** 2026-07-21 (Lucas). **Config:** `contact_trust:` in
`config/filter_cfg.yaml`.

The Schmitt-trigger / dwell / EMA trust machine will be validated by comparing
the Python and Java implementations on logged data rather than by porting its 10
unit tests.

**Cost:** the debounce semantics (40 ms dwell, 0.25/0.35 band, impact-blip
rejection) are currently unverified in Python. Note this interacts with decision
1: with no contact mask in the InEKF, trust output feeds the joint KF's stance
anchors, not the InEKF.

---

## 5. The contact zero-velocity constraint is plumbed but not applied

**Code:** `inEKF/filter.py` (`TODO(N^v / zero-velocity)`).

`JointFilterOutput` carries `Σ_q̇`, and `contact_velocity_noise` computes
`N^v = J_Ċ Σ_q̇ J_Ċᵀ` — but nothing consumes it. `N^v` is the noise on the contact
*zero-velocity constraint*, a separate measurement block with its own `H` rows
stacked below the position block. Whether that constraint belongs in v1 is still
open, so it is not invented here.

**Cost:** contact points are constrained in position but not velocity. The
covariance crosses the boundary already, so landing it later is a change in
`filter.step` only.
