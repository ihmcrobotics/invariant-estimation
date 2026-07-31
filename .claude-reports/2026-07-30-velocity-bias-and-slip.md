# Overnight: the vertical velocity bias, and the slip question

Night of 2026-07-30 into 07-31, branch `contactnet/process-socket`.

Every number here is measured in this repo tonight. Where a claim came from somewhere else -- the
CoCo paper, an earlier report, a plausible-sounding mechanism -- I checked it and several did not
survive; those are called out rather than quietly dropped.

**Reproduce the headline:**
```bash
uv run python run_estimator.py --policy baseline --headless --ticks 1500 --vx 0.6 --imu-noise \
    --toe-heel --early-release 0.5 --early-release-source foot --early-release-off-dwell 20
```
Drop the three `--early-release*` flags for the baseline.

**Status: complete.** All four phases ran. Run 8 (the slip-stratified retrain) finished at 03:36
and its result is a clean negative -- see the end of Phase 4.

---

## Executive summary

**Six things, in order of how much they change what you do next.**

1. **The sink is cut 2.24x, live and closed-loop.** A causal early anchor release on the
   **per-foot** normal load, with a short off-dwell so a stance stops fragmenting, takes final
   `dz` from **−0.567 m to −0.253 m over 30 s**, tail tilt 0.240° → 0.149°, velocity 0.0347 →
   0.0294 m/s. Every metric moves the right way. The per-liftoff lead is **159 +- 11 ticks with a
   0.0% miss rate** -- the profile of arm B, the non-causal oracle, reached causally. Shipped
   behind `--early-release`, default OFF, 28 tests, 11 mutants killed. Two things I got wrong
   first and only found by measuring: the per-*contact* load (the finer, more principled signal)
   does nothing, and without the off-dwell a 30 s walk reports 306 "liftoffs" on two feet.

2. **`v_bc` is noise. This is the night's most important measurement.** The contact-velocity
   feature is a 1 kHz finite difference of FK on noisy encoders, and it carries **0.204 m/s of
   noise**. **86% of loaded ticks have a true contact speed below that floor**; swing sits 9x
   above it. As given, the channel reports exactly one bit — swinging or not — which is *precisely*
   the stride-phase clock the learned `Σ_C` has been stuck on since run 2, and the ablation agrees:
   deleting `v_bc` entirely costs 0.002 of slip R². An analytic `J_C q̇` would put the floor at
   0.0036 m/s, **57x lower and 5x below the median loaded speed**, with no re-collection needed.
   **I weakened this claim after checking it** — the same floor is reachable by a 56-tick secant of
   the `p_bc` taps the window already carries, so the information is not absent, it is only
   available at 56 ms of smoothing and as a fine cancellation the BPTT gradient has to discover.
   See §2.3.

3. **Slip is predictable from the features (R² 0.32–0.36 held out by rollout, carried mostly by
   torque) — but the slip-heavy retrain DID NOT HELP, and that is the useful part.** The gate
   said go, `data/dr6` was collected (pooled slip 19.0 %, mu 0.30–0.80 against dr5's 6 values
   clustered at 1.1), `alpha_sweep` passed, and run 8 trained. Result: **slip R² unchanged**
   (0.065 → 0.062 on x), **phase lock worse** (0.100 → 0.271 on z), the CoCo Figure-3 correlation
   **collapsed** (+0.744 → +0.257) and went non-monotone, and on the held-out set it is worse than
   the heuristic **at the very thing it optimised** (`vel_rms` 1.35x). The data-side phase-lock
   number predicted this: recorded slip was never strongly phase-locked on *any* dataset
   (R² 0.12–0.17 on dr4, dr5 and dr6) — it is the network's **output** that is the clock
   (0.39–0.49). **The dataset was not the bottleneck.**

4. **Two premises the investigation started from are wrong.** CoCo-InEKF does *not* report
   centimetre position agreement (0.124 m ATE on their easiest case, over 20 s segments aligned at
   the initial state), their loss is identical to ours, and they do not truth-reseed during
   training — but neither do we (`ChainedBatcher`, `episode_s = 43 s`). And we do **not** have a
   horizontal-drift problem: it is 0.01–0.47 % of travel and does not track slip within a rollout.
   Our problem is vertical, by one to two orders of magnitude.

5. **On the one directly comparable figure with CoCo, we correlate but with the wrong dynamic
   range.** `√tr(Σ_C)` vs contact-point speed: corr +0.744, but only **1.2x** between planted and
   creeping, and a **total range of 2.4x** where the analytic heuristic spans **1e5**. Run 7's
   learned `Σ_C` is nearly constant — the arm-I configuration — which is why it sinks **2.0x worse
   than no network at all** in my own closed-loop run (−114.8 cm vs −56.7 cm), matching the 2.6x
   on mean `e_vz` already on record.

6. **Two failed arms that were more informative than the successes.** (i) A stance-clock
   predictor (`prev_len - lead`) scored **-5.39 m** -- 10x worse than baseline -- because
   fragmented stances made it fire at tick 0 and become arm I; once the fragmentation was fixed it
   became simply redundant. (ii) Offline on `data/dr5`, every load-threshold predictor misses
   31-50% of liftoffs and so does a *perfect* contact-speed sensor (22%), so arm B is not a
   physical predictor being approximated -- it is a covariance-conditioning schedule. That still
   stands; what the off-dwell fixed was the episode bookkeeping, not the anticipation.

---

## THE VIDEOS — `artifacts/video/`

All ghost clips are `--ghost` (full, non-anchored: the translucent robot is drawn at the
**estimated** pose, so its sinking *is* the error), 30 s, 25 fps, `--imu-noise` seed 0,
`--toe-heel`, N=4, identical camera. Only the thing under test differs.

| file | what it shows | `dz` final [cm] | `dz` tail [cm] | tilt tail [deg] | **yaw** tail [deg] | horiz drift [m] | vel rms [m/s] |
|---|---|---|---|---|---|---|---|
| `A_analytic_n4_vx06.mp4` | analytic baseline, `vx = 0.6` | **−56.7** | −42.3 | 0.24 | 0.90 | 0.20 (1.0 % of 18.8 m) | 0.0347 |
| `B_earlyrelease_foot05_vx06.mp4` | + early release, per-foot | −31.3 | −23.4 | 0.16 | 0.60 | 0.14 (0.8 %) | 0.0299 |
| **`E_best_od20_vx06.mp4`** | **+ off-dwell — the best config** | **−25.3** | **−18.8** | **0.15** | 0.66 | 0.16 (0.8 %) | **0.0294** |
| `E_best_od20_vx06_sidebyside.mp4` | the same, ghost displaced 0.8 m — shows the gait rather than the overlay | −25.3 | −18.8 | 0.15 | 0.66 | 0.16 | 0.0294 |
| `C_contactnet_run7_vx06.mp4` | ContactNet run 7 attached, no early release | −114.8 | −85.3 | 0.42 | 0.55 | 0.11 (0.6 %) | 0.0463 |
| `F_contactnet_run8_vx06.mp4` | ContactNet **run 8** (the slip-heavy retrain), no early release | −50.5 | −39.0 | 0.38 | — | — | 0.0349 |
| `A_analytic_n4_yaw08.mp4` | analytic, **turning** `vx 0.4 / yaw 0.8` | **−39.1** | −29.4 | 0.32 | 3.41 | 0.04 | 0.0374 |
| `B_earlyrelease_foot05_yaw08.mp4` | + early release, turning | −25.1 | −18.4 | 0.30 | 2.99 | 0.04 | 0.0362 |
| **`E_best_od20_yaw08.mp4`** | **best config, turning** | **−16.1** | **−12.2** | 0.30 | 3.00 | 0.04 | 0.0362 |
| `D_traindata_dr5_flat_seed000.mp4` | the **training data** — 20 s of `data/dr5`, hollow commands | — | — | — | — | — | — |
| `D_traindata_dr5_hard_stepping_seed001.mp4` | training data, hard stepping | — | — | — | — | — | — |

Read in one line: **the fix takes the sink from 56.7 cm to 25.3 cm walking forward (2.24x) and
from 39.1 cm to 16.1 cm turning (2.42x)**, improving tilt and velocity at the same time and
leaving yaw and horizontal drift unchanged; and **ContactNet run 7 makes it twice as bad as no
network at all** (−114.8 cm), which is the closed-loop consequence of the 2.4x-dynamic-range
`Σ_C` measured in §2.4.

`E_best_od20_*` is `--early-release 0.5 --early-release-source foot --early-release-off-dwell 20`.
The training-data clips are `experiments/render_rollout` re-simulations; they are 20 s of a 60 s
rollout, so the script's own divergence check is not meaningful for them and says so.

Three caveats on those numbers, none of them optional:

* **Global x, y and yaw are unobservable in a proprioceptive InEKF.** No `H` in this filter has a
  world-position or heading row. Those three drift for as long as it runs; a better release
  schedule reduces the *rate* and nothing can eliminate it. The horizontal column is a drift, not
  an error that converges.
* **The turning clips travel 0.3 m net** — the robot circles, so "% of travel" is meaningless
  there and the absolute 0.04 m is what to read. And note what did NOT improve: **yaw is 3.41° at
  baseline and 3.00° with the fix, i.e. unchanged within run-to-run scatter**, and horizontal drift
  is identical at 0.04 m. The release schedule buys the vertical axis and nothing else, which is
  what the theory predicts -- it changes the apportionment of a vertical residual.
* **The closed loop is not bit-reproducible** (PORT_NOTES: two identical invocations differ by
  1.4e-6 in final `dz`). These effects are 1.5–2x, far above that floor, but do not A/B anything
  below ~1e-5 in this harness.

---

## PHASE 1 — the vertical velocity bias

### What was built

`sim.sensors.EarlyRelease` (new, `src/invariant_estimation/sim/sensors.py`) — the live,
causal counterpart of the offline arm E'. A per-contact-point state machine: within a stance it
tracks the post-impact running peak of the normal load and latches a release when the load falls
to `frac` of it, ORed with the heuristic's own swing state so it can only ever loosen. Wired
through `SimSensorReader(early_release=...)` and `run_estimator.py --early-release`, **default
OFF**, so every number on record stays reproducible. `foot_loads` / `point_loads` grew `_raw`
variants because the clipped-at-1.0 signal `ContactTrust` consumes cannot supply a peak.

`tests/sim/test_early_release.py` (new, 28 tests, all green). The load-bearing one asserts
**bit equality with `experiments.process_socket_ablation.causal_early_release`** across
`frac ∈ {0.3,0.5,0.7,0.85} × blank ∈ {0,50,150}` — a live port that merely behaved similarly
would make every recorded arm-E' number uncomparable and nothing would raise. The rest pin the
impact-blanking regression (unblanked ⇒ >60 % of stance released; blanked ⇒ <35 %), the
only-ever-loosens property, the latch, and the latch clearing at touchdown.

**Mutation-checked**, per the standing rule that a green suite proves presence and not
correctness. Eleven mutants, all killed: latch removed (4 fail), fresh-reset removed (16), blanking
removed (10), peak reference ignored (15), peak includes the impact (10), `lead_ticks` zeroed (1),
clock arming without a previous stance (2), `prev_len` off by one (1), clock lead doubled (1),
`off_dwell` ignored (1), off-counter frozen (17).

Regression check: with the flag off, the closed-loop run reproduces the recorded N=4 numbers
**exactly** — final `dz` −0.5673 m, tilt 0.4298/0.2398°, velocity 0.0347 m/s, position 0.3375 m,
matching PORT_NOTES' "Toe/heel contact points" table to every printed digit. The refactor is
neutral.

### (a) Closed-loop: the load SOURCE decides everything (1.81x here; 2.24x once (a2) lands)

30 s at `vx = 0.6`, `--imu-noise`, seed 0, `--toe-heel`, N=4, no ContactNet.

| arm | final `dz` [m] | tail mean `dz` [m] | vel err rms [m/s] | tilt tail [deg] |
|---|---|---|---|---|
| **A analytic baseline** | **−0.5673** | −0.4234 | 0.0347 | 0.2398 |
| E frac 0.3, per-contact load | −0.5577 | −0.4161 | 0.0343 | 0.2338 |
| E frac 0.5, per-contact load | −0.5856 | −0.4363 | 0.0348 | 0.2541 |
| E frac 0.7, per-contact load | −0.6061 | −0.4503 | — | — |
| E frac 0.5, blank 75 | −0.5796 | −0.4329 | 0.0344 | 0.2481 |
| E frac 0.5, blank 250 | −0.5666 | −0.4239 | 0.0349 | 0.2419 |
| E frac 0.5, `peak_mode=prev` | −0.5335 | −0.3954 | 0.0332 | — |
| **E frac 0.5, per-FOOT load** | **−0.3127** | **−0.2342** | **0.0299** | **0.1601** |

**1.81x on the sink, 1.16x on velocity, 1.50x on tail tilt. Not a trade — every metric moves the
right way**, which is the same signature arm B had.

**The load SOURCE is the whole story, and getting it wrong first was the useful part.** My live
port defaulted to the per-**contact** normal load, which is the finer and apparently more
principled signal; the offline arm E' had used the per-**foot** total repeated to both contacts.
Per-contact does nothing (±2 %); per-foot gives 1.81x. The mechanism: at N=4 the toe carries the
load through the whole end of stance, so *its own* running peak is late and `frac · peak` is only
reached at the very last tick — the toe anchor is never released early. The foot's total load
starts falling as soon as the foot begins to unload, and releasing on it loosens **both** of that
foot's anchors together. Splitting the load per point split the peak reference with it and
destroyed the anticipation.

I would not have found this by reasoning; it came from running the parity arm. Both sources are
kept behind `--early-release-source` for that reason.

### (a2) The last piece: the stance was FRAGMENTING. 2.24x, and arm B's lead profile.

Adding the lead diagnostic to `run_estimator.py` exposed the remaining defect immediately: a 30 s
walk reported **306 "liftoffs" on two feet**, against a real cadence near 2 steps/s. The per-foot
normal load momentarily reads zero mid-stance (MuJoCo re-solves the contact set each step and a
foot can have no qualifying contact for a tick or two), so **one stance fragments into several
episodes, and every fragment resets the peak reference and clears the latch** — the two pieces of
state the whole mechanism depends on.

`EarlyRelease.off_dwell` requires N consecutive unloaded ticks to end a stance. Asymmetric in the
opposite direction to `ContactTrust`, which debounces the *entry*; here a spurious *end* is the
expensive error.

| arm | final `dz` [m] | tail | tilt tail [deg] | vel rms | liftoffs / 30 s | **zero-lead %** | lead p50 ± std [ticks] |
|---|---|---|---|---|---|---|---|
| analytic baseline | −0.5673 | −0.4234 | 0.2398 | 0.0347 | — | — | — |
| per-foot, `off_dwell = 0` | −0.3127 | −0.2342 | 0.1601 | 0.0299 | 306 | — | — |
| per-foot, **`off_dwell = 5`** | **−0.2529** | −0.1880 | **0.1492** | **0.0294** | 130 | 10.8 | 142 ± 45 |
| per-foot, **`off_dwell = 20`** | **−0.2529** | −0.1880 | **0.1492** | **0.0294** | **116** | **0.0** | **159 ± 11** |
| per-foot, `off_dwell = 50` | −0.2529 | −0.1880 | 0.1492 | 0.0294 | 116 | 0.0 | 189 ± 11 |
| + stance clock, lead 60 / 100 | −0.2556 / −0.2557 | — | — | — | 116 | 0.0 | 158 ± 19 / 158 ± 23 |
| *(reference)* arm B, non-causal oracle | — | — | — | — | — | 0.7 | 100 ± 23 |

The plateau across `off_dwell ∈ {5, 20, 50}` is worth having: it is not a tuned knob. And the
**stance clock becomes redundant** once the fragmentation is fixed — at lead 60 or 100 it changes
nothing (−0.2556 / −0.2557), because the level test already fires 159 ticks before liftoff, which
is more lead than the clock was asking for. The clock's catastrophic −5.39 m earlier was entirely
the fragmentation: with stances splitting into ~100-tick episodes, `prev_len − lead ≤ 0` for any
lead ≥ 100, so it fired at `k = 0` every time and became arm I.

**2.24x on the sink, 1.61x on tail tilt, 1.18x on velocity.** And the lead profile at
`off_dwell = 20` is **159 ± 11 ticks with a 0.0 % miss rate** — arm B's shape, reached causally.
116 liftoffs over 30 s on two feet is 1.9 steps/s per foot, the correct cadence, so the episode
bookkeeping is finally measuring real stances.

Two honest notes:

* **5 and 20 give identical filter output** (`dz` agreeing to four decimals) and differ only in
  the diagnostic. Not a coincidence: a gap of 6–20 ticks is a moment when the foot really is
  unloaded, so the heuristic's own swing state already sets `Σ_C` loose through the OR. What
  `off_dwell ≥ 5` fixes is the **1–5-tick dropouts, which occur while loaded** and were destroying
  the peak and the latch. `20` is preferred because it also makes the lead statistics read
  correctly, and a diagnostic that lies is worse than none.
* This does **not** overturn §1(b)/§1(c). The zero-lead rate is 0 % because the *episode* is now
  correct, not because the level test became anticipatory: the 159-tick lead is the tail of a
  correctly-measured stance, not a prediction of one. Arm B's 4.3x was measured in replay on
  `dr5`; this 2.24x is closed loop at `vx = 0.6`. Different measurements — do not subtract them.

### (b) Why — and it is not an implementation defect

I measured the **lead distribution per liftoff**, as asked, and used the right definition of
lead: not "first release tick in the stance" but **the length of the released run that ends at
liftoff**. The anchor has to be loose *at* the moment the residual arrives; a release that fires
and then re-tightens is worth nothing, and the first-release metric hides that.

Offline on `data/dr5` (4 rollouts, N=4, per-foot load, heuristic stance blocks):

| arm | lead p10/p50/p90 [ticks] | lead std | **zero-lead %** | loose % of stance |
|---|---|---|---|---|
| **B (non-causal oracle, 25)** | 25 / 25 / 25 | 3 | **0.7** | 6.5 |
| **B (non-causal oracle, 100)** | 51 / 100 / 100 | 23 | **0.7** | 24.2 |
| level f0.35 latch, current peak | 0 / 1 / 46 | 83 | 49.7 | 9.9 |
| level f0.5 latch, current peak (= arm E') | 0 / 3 / 79 | 202 | **41.8** | 19.0 |
| level f0.5 latch, previous stance's peak | 0 / 11 / 445 | 282 | 35.5 | 38.6 |
| level f0.7 latch, current peak | 0 / 21 / 204 | 236 | 31.4 | 32.3 |
| level f0.7 latch, previous peak | 0 / 21 / 467 | 292 | 31.6 | 44.5 |
| stance-clock (`prev_len − 100`) | 0 / 194 / 499 | 265 | 16.6 | 63.5 |

**The binding quantity is not the median lead — it is the fraction of liftoffs that get NO lead at
all.** Arm B gives every liftoff a lead of ~100 ticks with a std of 23 and a 0.7 % miss rate.
Raising `frac` reduces the miss rate but buys its extra loose ticks in the **mid-stance dip**
rather than at liftoff (loose % climbs to 32–45 % while zero-lead only falls to ~31 %) — which is
exactly why offline arm E' at frac 0.85 scored *worse* (−0.01836) than at 0.5 (−0.01101), and it
is the same failure mode as arm I. The closed-loop sweep reproduces the shape: frac 0.3 → 0.5 →
0.7 on the per-contact source goes −0.5577 → −0.5856 → −0.6061, i.e. monotonically *worse* with
more loosening in the wrong place.

Note the table above is on the per-**foot** load, the source that wins closed-loop; it still
misses 31–50 % of liftoffs, so 1.81x is what that miss rate buys. Arm B's 4.3x is what a 0.7 %
miss rate buys. **The miss rate is the binding quantity** — and §1(a2) closes it to 0.0 % for a
different reason than anything in this subsection: the episode detection was wrong, not the
threshold.

### (c) The property test, and what it revealed

The theory note's property test is: **`f` must have collapsed before `ν_z` turns positive.** The
release schedule is what sets `f`, so the causal question is whether any real signal turns
positive early enough. I measured the **true** world contact-point speed
(`p_C = p_true + R_true · y_fk`, 20-tick centred secant) as a release trigger — i.e. a *perfect*
contact-motion sensor, an upper bound on any feature-based predictor:

| speed threshold [m/s] | lead p10/p50/p90 | std | zero-lead % | loose % |
|---|---|---|---|---|
| 0.010 | 1 / 42 / 156 | 69 | 8.6 | 57.7 |
| 0.020 | 0 / 23 / 110 | 50 | 22.3 | 29.5 |
| 0.050 | 0 / 8 / 78 | 35 | 38.3 | 12.9 |
| 0.100 | 0 / 0 / 53 | 25 | 52.7 | 6.0 |

At the loose fraction that arm B uses (24 %), a perfect contact-speed sensor still misses 22 % of
liftoffs and has a median lead of 23 ticks against arm B's 100.

**Therefore: arm B is not a physical predictor that a causal signal is approximating. It is a
covariance-conditioning schedule.** It works by making `Σ_rel` large *before* anything physical
has happened, so that when the liftoff residual arrives `f` is already ~0. The information it
uses — "liftoff is 100 ms away" — is in the **gait plan**, not in the load and not in the
contact-point motion at that instant. No load-threshold predictor can recover it, and the
theory note's own conclusion ("a threshold on load *level*, however low, is structurally late")
understates the problem: a threshold on load *rate* or *ratio* is late too, and so is a perfect
tangential-velocity sensor.

This is a negative result and I am reporting it as one. The offline 1.35x was real but it was a
replay number on hollow-command data with long stances; on a steady 0.6 m/s walk the same
schedule fires too late on a third to a half of liftoffs and nets out at zero.

**What this does NOT overturn:** arm B's 4.3x is still real, and it still says the sink is
controlled by *when* the anchor is released. What has changed is the identity of the deployable
lever: it is a **gait-phase / stride-clock predictor**, which a walking controller has and a
load sensor does not. That is written up as a proposal below, not as a result.

---

## PHASE 2 — the slip question

### 2.1 (task 2b) DECISIVE GATE: slip **is** predictable from the 24 feature channels. Gate PASSES.

`experiments/slip_probe.py` (new). From the cached per-contact channels, predict recorded
`truth.slip_sat` on loaded ticks only (`truth.contact_fn > 0`). Features are both of a foot's
contacts, 8 causal history taps 40 ticks apart (384 dims), subsampled every 20 ticks.
Held out **by rollout** — the honest split, because `mu` is constant within a rollout and a
random within-rollout split lets any probe memorise it.

| dataset | target | by-rollout ridge | by-rollout **GBM** | within-rollout GBM |
|---|---|---|---|---|
| dr5 | `slip_sat` | +0.206 | **+0.359** | +0.462 |
| dr5 | `\|f_t\|/f_n` (mu removed) | +0.197 | +0.392 | +0.529 |
| dr5 | `contact_fn` (control) | +0.150 | +0.035 | +0.188 |
| dr4 | `slip_sat` | +0.188 | **+0.318** | +0.455 |
| dr4 | `\|f_t\|/f_n` | +0.202 | +0.359 | +0.489 |
| dr4 | `contact_fn` (control) | +0.191 | +0.165 | +0.217 |

**R² ≈ 0.32–0.36 held out on unseen rollouts, on both datasets.** That is not zero and it is not
close to zero. The network is **not** structurally blind to slip.

Why this is physically sensible, and it was worth checking rather than asserting: the channels
carry the six leg joint torques and the leg configuration, and in stance the ground reaction
wrench satisfies `tau = J_C^T f`, so `|f_t|/f_n` is recoverable from torque + kinematics up to
the Jacobian conditioning. `mu` is *not* recoverable — it is a DR draw, constant per rollout,
observable only through having already slipped. The `|f_t|/f_n` row is the mu-free target and it
scores **slightly higher** than `slip_sat` (0.392 vs 0.359 on dr5), which is exactly the
signature that prediction stops at the force ratio and the residual gap is the unknown `mu`.

Consequence: **Phase 4 (a slip-heavy retrain) is worth running.** The counterfactual —
"no amount of richer data helps, go add a force-ratio channel to CLAUDE.md §7" — is *refuted*
by these numbers. A tangential/normal force channel would still likely help (it would hand the
network the quantity it currently has to invert a Jacobian to get), but it is an optimisation,
not a prerequisite.

Caveat stated honestly: R² 0.36 is a *ceiling estimate for a 384-dim GBM on 36 k samples*, not
for the network. The network sees a richer window (H=50 at stride 8) but has to learn the
mapping through a BPTT gradient that only rewards body-velocity error — a far weaker teacher
than direct regression.

### 2.2 (task 2a) The learned covariance tracks **phase**, not slip — and only on the x axis

Same script, part (a). R² of `log10 std_axis` on `truth.slip_sat` versus on gait phase, loaded
ticks only, both predictors quantile-binned to the **same 24 bins** so the two columns are
comparable. (`phase_lock.r2_on_phase` uses 300 fixed-width phase bins, which would have handed
phase ~0.1 of advantage for free against a bounded predictor like `slip_sat`.)

| net / data | axis | median std | R² on slip | R² on phase |
|---|---|---|---|---|
| run7 / dr5 | x | 4.94e-2 | 0.065 | **0.490** |
| run7 / dr5 | y | 4.59e-2 | 0.010 | 0.041 |
| run7 / dr5 | z | 1.08e-1 | 0.008 | 0.100 |
| run6_w128 / dr4 | x | 7.37e-2 | 0.070 | **0.386** |
| run6_w128 / dr4 | y | 4.91e-2 | 0.009 | 0.031 |
| run6_w128 / dr4 | z | 1.51e-1 | 0.011 | 0.063 |

Three things fall out, and one of them corrects the premise of the investigation.

1. **The covariance barely tracks slip at all.** R² ≤ 0.07 on every axis of both nets, against a
   probe that reaches 0.32–0.36 from the same inputs. The information is present in the features
   and the network is not using it. This is the central negative result of the night.

2. **What it does track is phase, and only fore-aft.** `std_x` is 0.39–0.49 explained by stride
   phase; `std_y` and `std_z` are ~0.03–0.10. So the "stride-phase clock" finding survives the
   move to the process socket and to N=4, but it has narrowed to a single axis.

3. **The 400x horizontal/vertical anisotropy does NOT reproduce on the process-socket nets.**
   Run 4 (measurement socket, N=2) had median `std_x = 3.85e-4` vs `std_z = 1.67e-1`, a factor
   434. Run 6 and run 7 sit at 2.1x and 2.2x respectively (`std_z / std_x`), with `std_y` the
   tightest axis in both. **The strong form of the hypothesis — "the network never inflates the
   horizontal components" — is therefore false for the nets currently in use.** It was true of
   run 4, on a different socket, at N=2. What *is* true of run 6/7 is weaker and still bad: the
   horizontal components are inflated, but on a stride clock rather than on slip.

   Note the direction this points. `std_y` (lateral) is the least phase-locked *and* the least
   slip-locked axis in both nets — it is close to a learned constant. Lateral slip is the axis a
   turning walk stresses most.

### 2.3 (unplanned) THE MECHANISM: the `v_bc` channel cannot resolve contact motion inside stance

This was not on the plan; it fell out of checking whether our contact-velocity feature carries
what CoCo's Figure 3 correlates against. It is the most important thing measured tonight.

`features.make_contact_channels` builds `v_bc` as a **causal first difference of FK(q) at 1 kHz**,
on noisy encoders — deliberately, "not J q̇, which would reintroduce the joint-KF coupling this
feature set exists to avoid" (`features.py:335`). Differentiating white encoder noise at 1 ms
amplifies it by `sqrt(2)/dt = 1414`.

Measured (`experiments/contact_velocity.py`, new; median over the 12 `data/dr5` rollouts). The
noise is not assumed: for a signal smooth on a 1 ms scale plus white noise, the second difference
`p[k+1] − 2p[k] + p[k−1]` has variance `6 σ_ε²`, so `σ_ε = std(d²p)/√6`.

| quantity | value |
|---|---|
| FK position noise `σ_p` | 1.44e-4 m |
| effective FK gain `‖J‖_eff = σ_p / σ_q` (with `σ_q = 2e-4` rad) | 0.720 m/rad |
| **`v_bc` noise as built** `√2 σ_p / dt` | **0.204 m/s** |
| `v_bc` noise if built as analytic `J_C q̇` (at `σ_q̇ = 5e-3` rad/s) | 0.0036 m/s |
| ratio | **57x** |
| `v_bc` channel std (all axes pooled) | 0.713 m/s |

Now the true world-frame contact-point speed, from `p_C = p_true + R_true · y_fk` differentiated
with a 20-tick centred secant, over 6 rollouts:

| regime | n | p50 | p90 | p99 |
|---|---|---|---|---|
| loaded (`contact_fn > 0`) | 717 872 | **0.0173** | 0.325 | 1.346 |
| unloaded (swing) | 386 128 | **1.834** | 3.063 | 4.106 |

**86.2 % of loaded ticks have a true contact speed below the `v_bc` noise floor.** Swing sits at
1.83 m/s, 9x *above* it.

So `v_bc` can reliably report exactly one bit — *is this point swinging or not* — and nothing
about how a loaded point is moving. That is a stride-phase clock, expressed in a feature channel.
It closes the causal chain:

1. `v_bc` noise floor 0.204 m/s (measured);
2. 86 % of loaded ticks are below it (measured);
3. swing is 9x above it (measured);
4. ⇒ the only reliable information in the channel is swing/stance;
5. ⇒ the learned `Σ_C` is a stride-phase clock (measured, §2.2: R² 0.39–0.49 on phase, ≤0.07
   on slip);
6. CoCo-InEKF Figure 3 correlates their `√tr(Σ_C)` against **contact-point velocity magnitude** —
   a quantity our feature set structurally cannot represent below 0.2 m/s.

An analytic `J_C q̇` would put the floor at 0.0036 m/s, **5x below the median loaded speed** of
0.0173 m/s, i.e. it would move the discriminating regime from "invisible" to "well resolved".

#### I checked my own tidy explanation, and it needs weakening

"The information is not in the features" would be the satisfying version, and it is **not quite
true**. The window carries `p_bc` at 50 taps 8 ticks apart over 392 ms, and a *difference across
taps* is a linear function of the flattened window — representable by the first dense layer.
Measured noise of a `W`-tick secant of the recorded `p_bc`:

| W [ticks] | σ_v [m/s] | SNR vs the 0.0173 m/s median loaded speed |
|---|---|---|
| 1 (= `v_bc` as built) | 0.2037 | **0.08** |
| 8 (one tap spacing) | 0.0255 | 0.68 |
| 20 | 0.0102 | 1.70 |
| 40 | 0.0051 | 3.40 |
| **56** | **0.0036** | 4.76 |
| 392 (the whole window) | 0.0005 | 33.3 |
| **analytic `J_C q̇`** | **0.0036** | **4.81** |

So a **56-tick (56 ms) secant of `p_bc` matches the analytic channel exactly**, and the network
could in principle form it. The correct, weaker claim is therefore:

* the **instantaneous** `v_bc` channel is noise at SNR 0.08 and is worthless as given — that part
  stands, and the ablation confirms it costs 0.002 to delete;
* the information is recoverable from `p_bc`, but only by **trading temporal resolution for
  noise**: 56 ms of smoothing to reach the analytic floor, against a liftoff lead of 159 ticks, so
  a third of the event is smeared;
* and it has to be discovered as a **fine cancellation between two large standardised numbers**
  (`normalize.apply` scales each channel by its own std) through a BPTT gradient that only rewards
  body-velocity error 128 ms downstream.

The case for `J_C q̇` is therefore *"hand it the quantity at full bandwidth"*, not *"the
information is absent"*. That is a real weakening of the argument and I would rather state it than
have it found later.

This is a CLAUDE.md §7 change (it changes `Features`, hence `F` and `d_in`) and it forces a full
re-cache of every dataset and a retrain from scratch. See "What I propose and did not do".

### 2.4 (unplanned) The CoCo Figure-3 analogue — it correlates, and the dynamic range is the problem

CoCo §V-A1 plots `√tr(ᴮΣ_C)` against the contact point's velocity magnitude and reports that the
two "agree". `experiments/contact_velocity.py` computes both for our nets, with the speed taken
from **ground truth** (`p_C = p_true + R_true·y_fk`, 20-tick centred secant) rather than from the
noisy `v_bc` feature.

| net / data | corr(log10 √tr Σ, speed) | R² on binned speed | near-static (<0.05 m/s) | creeping (0.05–0.30) | swinging (>0.30) |
|---|---|---|---|---|---|
| run7 / dr5 | **+0.744** | 0.611 | 0.134 | 0.164 | 0.322 |
| run6_w128 / dr4 | **+0.746** | 0.588 | 0.176 | 0.208 | 0.393 |

The correlation is genuinely strong, and taken alone it would read as "we reproduce Figure 3".
The regime split says otherwise, in two ways:

1. **Between planted (<0.05 m/s) and creeping (0.05–0.30 m/s) the covariance rises only 1.2x.**
   That band is where slip lives. Almost all of the 0.744 comes from the swing/stance step, and
   in a periodic walk contact speed and gait phase are nearly the same variable — so a phase clock
   scores 0.74 on this plot for free. This is the phase-clock finding restated in CoCo's own
   coordinate.

2. **The total dynamic range is 2.4x. The analytic heuristic's is 1e5.** The heuristic switches
   the Cholesky factor 1e-4 (stance) → 1e1 (swing); run 7 spans 0.134 → 0.322. So the learned
   `Σ_C` is **1300x looser than the heuristic in stance and 31x tighter in swing** — very nearly
   the constant-`Σ_C` configuration that arm I measured as catastrophic (velocity rms 1.15 vs
   0.059, tilt 10.4° vs 0.51°). Run 7 does not fall, but it sinks **2.6x worse than the analytic
   filter** (−0.0347 vs −0.0134 m/s), and this is why. CoCo's own Figure 3 shows their std going
   to ~0 in stance and up to ~1.5–2.0 m/s in swing — a range they do not quantify but which is
   visibly orders of magnitude, not 2.4x.

So the head-to-head against CoCo on the one directly comparable figure is: **same sign, same
correlation, wrong dynamic range, and flat exactly where slip is.**

### 2.5 (unplanned) Which channels carry the slip signal — and `v_bc` is free to drop

`slip_probe --group-ablation`, target `slip_sat`, held out by rollout, GBM:

| channels | R² | vs all 24 |
|---|---|---|
| ALL 24 | +0.359 | — |
| only `tau` | **+0.320** | −0.039 |
| only `v_bc` | +0.250 | −0.109 |
| only `q` | +0.236 | −0.123 |
| only `p_bc` | +0.224 | −0.134 |
| only `accel` | +0.215 | −0.143 |
| only `gyro` | +0.192 | −0.167 |
| drop `tau` | +0.322 | −0.037 |
| drop `q` | +0.366 | +0.007 |
| drop `p_bc` | +0.360 | +0.002 |
| **drop `v_bc`** | **+0.356** | **−0.002** |
| drop `accel` | +0.354 | −0.005 |
| drop `gyro` | +0.359 | +0.000 |

**Torque is the primary carrier** — 89 % of the full R² on its own — which is what the physics
says: in stance `tau = J_Cᵀ f`, so `|f_t|/f_n` is recoverable from torque and configuration up to
Jacobian conditioning. The signal is heavily redundant: no family costs more than 0.04 to remove.

And **`v_bc` costs 0.002 to delete.** A channel carrying real contact-motion information would not
be free to drop. That is an independent confirmation of §2.3 from the opposite direction: the
channel is noise-dominated, its 0.250 standalone score is the swing/stance clock leaking through
phase, and it contributes essentially nothing the other channels do not already have.

### 2.6 (task 2c) Attributing the horizontal drift: slip explains it BETWEEN rollouts and not WITHIN

`experiments/slip_attribution.py` (new), read-only over the 12 `data/dr5` rollouts. The
estimator ran during collection, so `aux.est_p` is its own output and nothing is re-run.

**Between rollouts** (n = 12): `corr(slip fraction, horizontal drift rate) = +0.508`,
`corr(mu, drift rate) = −0.523`. Suggestive, and driven by the three low-`mu` rollouts. But read
the scatter, not the number — `hard_stepping_seed002` at `mu = 0.52` and 17.4 % slip drifts
**−0.023 m**, i.e. essentially not at all, while `flat_seed002` at `mu = 0.45` and 20.7 % drifts
+0.172 m. At n = 12 that is one rollout away from no relation.

**Within rollouts** (276 windows of 2 s): `corr(mean cone saturation, drift increment) = +0.028`.
Mean increment in the low-slip quartile +0.0053 m, in the high-slip quartile +0.0068 m, against a
per-window std of 0.0193 m. **Slip does not predict where inside a rollout the drift happens.**
The between-rollout relation is therefore more consistent with `mu` changing the whole gait than
with individual slip events dragging the base.

**And the magnitudes settle the priority.** Horizontal drift over a 46 s rollout is **0.02–0.17 m,
i.e. 0.01–0.47 % of distance travelled**. The closed-loop *vertical* error over 30 s is
**−0.567 m**. Horizontal drift is between one and two orders of magnitude smaller than the
vertical bias, on the same data, and it does not track slip within a run.

So the framing that started the night — "a sliding foot drags the base and we get horizontal
drift the CoCo paper does not have" — is **not supported**. We do not have a horizontal drift
problem. We have a vertical one, and Phase 1 halves it.

---

## PHASE 3 — CoCo-InEKF, and a correction to the premise

I read the paper source directly (`~/Downloads/arXiv-2605.15122v1/body-paper.tex`) rather than
only the PDF, and verified every number quoted below against it.

### The premise "they reach centimetre-level agreement" is wrong

Their headline metric is **linear velocity**, not position. Table III (simulated dancing),
Table IV (simulated ground motions), Table XI (20 real sequences):

| | lin. vel. ATE RMSE [m/s] | position ATE RMSE [m] | orientation ATE RMSE [rad] |
|---|---|---|---|
| CoCo-InEKF, dancing (sim) | **0.046** | 0.124 | 0.031 |
| CoCo-InEKF, ground motions (sim) | 0.099 | 0.342 | 0.069 |
| CoCo-InEKF, real world | 0.0805 | 0.2019 | 0.0302 |
| InEKF, GT contacts (dancing) | 0.176 | 0.422 | 0.033 |
| InEKF, heuristic contacts (dancing) | 2.675 | 17.423 | 0.041 |
| Hybrid Baseline+ (dancing) | 0.121 | **0.111** | 0.027 |

Position is **decimetre for them too** — 12.4 cm on their easiest scenario, 34 cm on ground
motions, 20 cm on hardware. And on *position* they are beaten by their own Hybrid Baseline+
(0.111) and by SET (0.126 on ground motions); they say so, and attribute it to having no position
term in the loss.

Two things about their protocol that must be quoted whenever their numbers are:

* the test sets are **100 sequences of 20 s each** (§IV-B, verbatim: *"The dancing test dataset
  comprises 100 such sequences, each 20 s in duration"*);
* ATE is *"computed over the trajectory **after aligning the initial state**"* (§IV-C).

So their position number is a bounded ≤20 s drift from an aligned start. Our 30 s continuous
closed-loop run with no realignment is a strictly harder measurement. Our N=4 analytic filter
scores base velocity RMS **0.0347 m/s** on that harder protocol, which is *better* than their
best reported velocity ATE (0.046) — though the scenarios are not the same and the comparison is
indicative, not a claim of parity.

**Consequence: the "CoCo gap" as posed is much smaller than assumed, and on velocity may not
exist.** What we actually have that they do not is a vertical position ramp, and their loss
contains nothing that would have penalised one either.

### Verified: what they have, what we have

| | CoCo-InEKF | our port | verdict |
|---|---|---|---|
| loss | Eq. (9), L2 on body-frame velocity | `losses.l2_velocity`, same | **identical** — "L2 is insufficient" is refuted |
| position/orientation loss term | none; explicitly declined | none | same |
| BPTT unroll `L` | 128 @ 600 Hz = **213 ms** | 128 @ 1 kHz = **128 ms** | 1.67x shorter |
| history `H` | 20 @ 600 Hz = **33 ms** | 50 over **392 ms** | **we are 12x longer, and their Table VI says longer HURT** |
| features | `ω, a, q, q̇, τ, p_B→Ci, v_B→Ci` | same minus **`q̇`** | we omit `q̇` |
| filter state as NN input | excluded, deliberately | excluded (CLAUDE.md §7) | same |
| accelerometer bias state | **yes**, `ᴮb_a`, Eqs. (3)/(7) | **none anywhere** | **structural absence on our side** |
| velocity-level / zero-velocity contact constraint | **none** — Alg. 1 has one correction, Eq. (8), the FK *position* measurement | none | **same; the `TODO(N^v)` deferral is correct** |
| touchdown reseed | **none**; contact states are permanent and continuous | off by default | same |
| absolute measurement (map/vision/mocap) | none | none | same |
| `Σ_C` socket | process, body frame, inside Prediction | process, body frame | same |
| contact points `N` | 4 (dancing) / 10 (ground); 4→10→18 gives 0.134→0.099→0.069 | 4 | same for walking |
| training episode `T` | 100 s (dancing), 6 s (ground) | `episode_s = 43.0` | **comparable** |
| truth reseed during training | *"experimented with … observed a degradation"* — they do not | `ChainedBatcher` carries `(X̂,P)`; reseeds only at `episode_ticks`, rollout end, or divergence | **we do not either** |
| compute | E = 1280 envs, 64 600 iters in a 5-day cap, fresh physics per iteration | B = 32, 10 000 iters, 78 min, replayed data | **~250x on segments seen** |
| robot | Lima, 0.84 m, 16.2 kg, 20 DoF, 600 Hz | Alex, 90.5 kg, 1 kHz | not comparable in metres |

Two corrections to things that were believed going in, both of which I checked in the source
rather than accepting:

* **We are not force-teacher-training.** `dataset.ChainedBatcher` (docstring, `dataset.py:461`)
  exists precisely to avoid it, and its own docstring gives the same argument CoCo gives. Only
  run 1 reseeded per segment. `episode_s = 43 s` sits between their two values. So
  "no per-window truth reseeding" is **not** a difference between us and them — it is a
  difference between us-now and us-at-run-1, and it was already fixed.
* **The velocity-level contact constraint is not the answer.** Their Alg. 1 has exactly one
  correction block and it is the FK position measurement, Eq. (8). The zero-velocity assumption
  enters entirely through the process model Eq. (5), whose covariance is the learned quantity —
  which is our architecture exactly. The `TODO(N^v / zero-velocity)` at `inEKF/filter.py:392`
  should stay deferred.

### Ranked: what actually explains the difference in the learned `Σ_C`

Ranked by evidence, not by how satisfying the story is. Note the ranking is now about *why our
`Σ_C` is a phase clock and theirs tracks contact velocity*, because the position/velocity gap
itself turns out to be much smaller than assumed.

1. **Our contact-velocity feature is 57x noisier than it needs to be, exactly in the regime that
   matters.** Measured, §2.3: floor 0.204 m/s, 86% of loaded ticks below it, swing 9x above it,
   and deleting the channel costs 0.002 of slip R². Their Fig. 3 plots `√tr(Σ_C)` against
   contact-point speed; our instantaneous channel cannot resolve the x-axis of their figure below
   0.2 m/s. **CONFIRMED as a difference, and it is the only item that mechanically explains the
   phase clock rather than merely correlating with it — but see the weakening in §2.3: a 56 ms
   secant of the `p_bc` taps reaches the same floor, so the honest form is "badly presented at
   full bandwidth", not "absent".** They feed `q̇` directly (their observation vector `o`); we do
   not carry a measured `q̇` for the filtered joints at all.
2. **Motion diversity.** One gait, one command family ⇒ "contact quality" and "stride phase" are
   the same variable. They train on 81 retargeted dance sequences (5.6–36.1 s) plus a falling
   policy with goal poses changing every 2 s, with friction, disturbance-force and terrain
   randomisation, and their *test* set adds periodic disturbances specifically *"to induce
   slippage"* (§IV-B, verbatim). CONFIRMED as a difference; it caps how much headroom exists.
3. **Training budget, ~250x on segments seen** (their E=1280 × 64 600 fresh-physics iterations
   vs our 32 × 10 000 replayed). CONFIRMED, and it compounds (2). I have written the exact
   queueable run below rather than leaving this as an excuse.
4. **No accelerometer-bias state on our side.** CONFIRMED as a structural absence
   (`grep` for accel bias in `src/` returns nothing; `jointKF/state.py` carries `b_ω` only, and
   the joint KF has no accelerometer measurement so `ā` is never bias-corrected). An
   uncompensated `b_a,z` integrates straight into `v_z`; our −0.0134 m/s is ~1.3 mg of z-bias,
   squarely in IMU range. **However** — `sim/sensors.IMUNoise` injects *no* accel bias, so this
   cannot be the cause of the sink measured **in sim**. It is a hardware exposure and a real gap;
   it is not tonight's bug. Ranked here rather than higher for that reason.
5. **Our history window is 12x longer than theirs and their ablation says that hurts.** H=20 at
   600 Hz = 33 ms; our H=50 over 392 ms. Their Table VI: H=150 (250 ms) was worse for *every*
   method (CoCo 0.052 vs 0.046). This directly contradicts a documented choice of ours
   (`config.py:52-72`). Cheap to ablate. PLAUSIBLE.
6. **Contact-point count** for ground-contact motions (4→10→18 monotone). Irrelevant to walking;
   relevant if we ever collect falls. CONFIRMED but not applicable.
7. **Evaluation protocol.** 20 s aligned segments vs our 30 s free-running episode. Not a
   mechanism, but it means their position numbers and ours are not the same measurement.
   CONFIRMED.

**Consistency with the anchor** (their learned model beats their GT-contacts baseline): their GT
baseline is itself a *threshold* (Table I: xy-velocity ≤ 0.25 m/s **and** height ≤ 0.01 m), so a
foot sliding at 0.24 m/s counts as "in contact". Their stated win (§V-B4, verbatim) is *"the
contact points are not required to be exactly in contact. As soon as a contact candidate is
stationary **along a certain dimension**, this information can be used"* — per-axis partial
stationarity, which no binary flag can express at any threshold. That is a statement about
*graded, directional* covariance driven by *contact-point motion* — i.e. it is item 1 on the list
above, from their side.

### Caveats on their numbers, for the record

* Their real-world "ground truth" is **MoCap fused with IMU through a classical InEKF** (§V-D).
  The velocity reference therefore shares an IMU with the estimator being scored.
* They publish **no numerical `Σ_C` values** — Fig. 3 is `√tr(Σ)` curves and two ellipsoid
  renderings. Our anisotropy ratios cannot be compared against anything they printed.
* They never evaluate a continuous run longer than 20 s.

---

## PHASE 4 — the slip-heavy retrain

Gated ON by §2.1 (slip is predictable, R² 0.32–0.36 held out). Queued and running.

**The lever, and it is a sampling defect not a range defect.** `dr4`/`dr5` draw `mu` i.i.d. from
a stream keyed on the *seed alone*; `friction_range_by_terrain` changes only the range, so the
same underlying uniform is reused and 12 rollouts realise ~6 distinct values. Measured on `dr5`:
`mu ∈ {1.179, 1.089, 0.447, 1.181, 1.100, 0.522}` — **eight of the twelve rollouts are above
mu = 1.0**, and the 0.20–0.35 band where slip is 26–44 % was never sampled at all.

**What was built.** `DomainRandomization.friction_grid_by_terrain` — a deterministic per-terrain
`mu` grid indexed by seed. The uniform draw is still consumed, so the push and command schedules
stay bit-identical and `dr6 − dr5` is exactly the friction change. Five tests in
`tests/sim/test_collect_dr.py`, including one that asserts the *defect* against the shipped data
(≤6 distinct mu, none below 0.44, and the replayed value equals `flat_seed000`'s recorded
`friction_mu` to 1e-12) so the fix cannot be silently reverted.

`config/collect_dr6.yaml` — the grid, with the low end spent only where the robot survives it
(`hard_stepping` FELL at 0.20 when measured, so it is floored at 0.30):

| terrain | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| flat | **0.20** | 0.30 | 0.45 |
| waves | **0.20** | 0.35 | 0.55 |
| stepping_stones | 0.30 | 0.40 | 0.60 |
| hard_stepping | 0.30 | 0.45 | 0.80 |

8 distinct `mu`, 5 of 12 rollouts at ≤0.35 against **zero** in dr4/dr5.

`artifacts/run8_pipeline.sh` — collect → cache → norm → P0 → **gate** (`alpha_sweep` at B=32,
`phase_lock`, `slip_probe`) → train (`l2_velocity`, widths 128 64, 10 000 steps) → replay
in-sample and held-out → **re-measure whether it learned slip** (`slip_probe --checkpoint` and the
Figure-3 analogue, so run 8 is directly comparable to run 7's 0.065/0.010/0.008). Everything
except friction is held at run 7's settings.

### What was actually collected — and mu = 0.20 does NOT survive 60 s

**3 of the 12 rollouts fell and were dropped by the collector's own tilt bound**, all of them the
low-mu entries: `flat/seed0` (mu 0.20, fell at 46.4 s), `waves/seed0` (0.20, at 51.7 s),
`hard_stepping/seed0` (0.30, at 25.3 s). The feasibility measurement the grid was built on was
over **8 s** walks; over **60 s with pushes up to 400 N**, mu = 0.20 is not survivable even on
flat, and 0.30 is not survivable on `hard_stepping`. That corrects the friction-feasibility note,
and the grid should floor flat/waves at 0.25–0.30 next time.

The 9 that survived are nonetheless the slip-heavy set that was wanted:

| rollout | mu | slip fraction |
|---|---|---|
| `stepping_stones_seed000` | 0.30 | **33.8 %** |
| `flat_seed001` | 0.30 | **26.6 %** |
| `waves_seed001` | 0.35 | **22.4 %** |
| `flat_seed002` | 0.45 | 20.8 % |
| `hard_stepping_seed001` | 0.45 | 16.0 % |
| `waves_seed002` | 0.55 | 15.9 % |
| `stepping_stones_seed001` | 0.40 | 14.4 % |
| `stepping_stones_seed002` | 0.60 | 11.5 % |
| `hard_stepping_seed002` | 0.80 | 9.4 % |
| **pooled** | 0.30–0.80, 8 distinct | **19.0 %** |

Against dr5, whose per-rollout slip was 6.2–20.8 % with **eight of twelve rollouts at mu > 1.0**
and a minimum mu of 0.447. dr6 has **five of nine rollouts above 20 % slip where dr5's maximum was
20.8 %**, and its mu spans 0.30–0.80 where dr5 clustered at 1.1.

### The gates, and a nuance that matters more than the gates

* **`alpha_sweep` at B=32: PASS.** Interior optimum at `α = 2.154e3` (`Σ_C ≈ 0.215 m`), index 11
  of 12 — the same shape run 7 had. "Ignore the feet" is not the loss-minimising answer, so the
  objective is trainable on this data.
  *Operational note:* the gate **initially did not run at all** — my `run8_pipeline.sh` omitted
  `--toe-heel`, so it built an N=2 filter against an N=4 `P0` and died with
  `dot_general … got (15,) and (21,)`. `run7_pipeline.sh` has the same omission (hence its
  `4_alpha_retry.log`). The pipeline used `set -uo pipefail` without `-e`, so it kept going;
  I re-ran the gate by hand with `--toe-heel` and it passes. **Both pipeline scripts should get
  the flag.**
* **`phase_lock` data-side R²: dr6 = 0.119**, against dr5 = 0.168 and dr4 = 0.144. A 1.4x
  decorrelation, in the right direction but modest.

**The nuance is the important part.** That data-side number has been ~0.12–0.17 on *every*
dataset, including the ones the phase-clock finding was made on. So **the recorded slip was never
strongly phase-locked** — 85 % of its variance is *not* explained by stride phase. What is
phase-locked is the **network's output** (model-side R² 0.39–0.49 for `log10 std_x`). The data
has always contained slip variation decorrelated from phase; the network has not been able to see
it. That is a direct, independent argument for §2.3: the limiting factor is the *feature*, not the
dataset, and it predicts that run 8 will move the needle less than the slip fractions suggest.

* **`slip_probe` on dr6: by-rollout GBM R² = +0.238**, against dr5's +0.359. *Lower*, and that is
  expected rather than disappointing: dr6 spans `mu ∈ [0.30, 0.80]` where dr5 clustered near 1.1,
  so unobservable `mu` now accounts for a larger share of `slip_sat`'s variance. The mu-free
  ratio is the quantity to compare across datasets.

Results land in `artifacts/run8/pipeline.log`. **`beta_nll` is untouched** — `--objective
l2_velocity` throughout. The acceptance test for run 8 is **not** the loss: stage 7 re-runs
`slip_probe --checkpoint` and the Figure-3 analogue so run 8's "R² of `log10 std` on slip" is
directly comparable to run 7's 0.065 / 0.010 / 0.008, and its planted→creeping ratio to run 7's
1.2x. If run 8 lands near run 7 on those, the `v_bc` change (§"What I propose", item 1) is the
next thing to run and the dataset is not the lever.

**How to read run 8 when it lands** (`artifacts/run8/`), in this order — the loss is the least
informative number in the directory:

1. `7_slip_model.log` — R² of `log10 std_{x,y,z}` on slip vs on phase. **Run 7: 0.065 / 0.010 /
   0.008 on slip against 0.490 / 0.041 / 0.100 on phase.** If slip does not rise and phase does
   not fall, more slip in the data did not help and the feature is the bottleneck.
2. `7_fig3.log` — the planted → creeping → swinging medians. **Run 7: 0.134 / 0.164 / 0.322**, a
   1.2x planted→creeping ratio and a 2.4x total range against the heuristic's 1e5. A run that has
   learned something about slip should widen the planted→creeping step first.
3. `6_replay_heldout.log` — out-of-sample on `data/control4`, which is the only generalisation
   number in the pipeline.
4. Then, and only then, close the loop:
   `run_estimator.py … --contactnet artifacts/contactnet_run8.npz --contactnet-norm
   data/dr6/norm_constants.npz`, against run 7's **−114.8 cm** and the analytic **−56.7 cm**.
   A gain ratio called run 6 healthy and then degenerate on this project before; only the filter
   settles it.

My prediction, recorded before the numbers existed so it could be wrong: **run 8 will not move the
slip R² much.** The data-side phase-lock argument above says the slip variation was already there
and uncorrelated with phase in dr4 and dr5; what changed in dr6 is its *magnitude*, not its
*visibility*.

### RUN 8 FINISHED. The prediction holds, and the result is worse than "no change".

| | run 7 (dr5) | **run 8 (dr6, slip-heavy)** |
|---|---|---|
| R² of `log10 std_{x,y,z}` **on slip** | 0.065 / 0.010 / 0.008 | **0.062 / 0.017 / 0.017** |
| R² of the same **on gait phase** | 0.490 / 0.041 / 0.100 | **0.509 / 0.105 / 0.271** |
| Figure-3 corr(log10 √tr Σ, contact speed) | **+0.744** | **+0.257** |
| planted / creeping / swinging median `√tr Σ` | 0.134 / 0.164 / 0.322 | **0.288 / 0.241 / 0.308** |

**Slip R² did not move** (0.065 → 0.062 on x; y and z doubled off a floor of ~0.01, which is
noise). **Phase lock got worse on every axis**, most of all vertically (0.100 → 0.271). And the
contact-speed correlation — the one thing run 7 had in common with CoCo's Figure 3 — **collapsed
from +0.744 to +0.257**, with the regime medians now **non-monotone**: the covariance is *lower*
while the foot is creeping (0.241) than while it is planted (0.288). Total dynamic range 1.28x,
down from run 7's already-poor 2.4x.

Replay, from `artifacts/run8/6_replay_*`:

| metric | in-sample (dr6) heuristic → trained | held out (control4) heuristic → trained |
|---|---|---|
| `vel_rms` (**the thing L2 optimises**) | 0.0774 → 0.0540 (**0.70x, better**) | 0.0290 → 0.0390 (**1.35x, WORSE**) |
| `e_vz` | −0.00921 → −0.00090 (0.10x) | −0.01009 → −0.02036 (2.02x) |
| `pos_rms` | 0.132 → 0.175 (1.33x worse) | 0.156 → 0.290 (1.86x worse) |
| `height_rms` | 0.110 → 0.140 (1.28x worse) | 0.153 → 0.280 (1.83x worse) |

In sample it is the textbook L2 trade — velocity better, position and height worse, exactly what
an objective with no position term rewards. **Held out it is worse at everything, including the
quantity it optimised**, which is `replay_eval`'s own hard-coded failure verdict: *"The trained
Sigma_C is WORSE at the very thing it optimised. Something upstream of the objective is wrong."*

Closed loop, 30 s at `vx = 0.6`, same conditions as every other clip:

| | final `dz` | tilt tail | vel rms |
|---|---|---|---|
| analytic N=4 | −56.7 cm | 0.24° | 0.0347 |
| **+ early release (the fix)** | **−25.3 cm** | **0.15°** | **0.0294** |
| ContactNet run 7 | −114.8 cm | 0.42° | 0.0463 |
| ContactNet run 8 | −50.5 cm | 0.38° | 0.0349 |

Run 8 is **2.3x better than run 7** and finally not a disaster, but it is still *worse than no
network at all* on tilt and only 11 % better on `dz` — against 2.24x from the schedule change
that costs no training at all.

**Verdict: a slip-heavy dataset is not the lever.** The gate that authorised this run (§2.1, slip
is predictable at R² 0.32–0.36) was correct and worth checking, and the run was correct to make —
but the answer is negative, and it is the same answer the data-side phase-lock number predicted.
The bottleneck is not what is in the data. Item 1 of "What I propose" (`J_C q̇` at full bandwidth)
is the next thing to run, and the ranking below stands unchanged.

---

## What I changed

| file | change | tests |
|---|---|---|
| `src/invariant_estimation/sim/sensors.py` | `EarlyRelease` — causal anchor early release (level-on-peak, impact-blanked, latched, optional falling-rate and stance-clock terms); `foot_loads_raw` / `point_loads_raw` because a peak cannot be read off a clipped signal; `SimSensorReader(early_release=…)`, default OFF | `tests/sim/test_early_release.py`, 28 tests, 11 mutants killed |
| `run_estimator.py` | `--early-release{,-blank,-mode,-source,-rate,-lead}`, all defaulting to OFF; `print_release_leads` reports the per-liftoff lead distribution and its zero-lead fraction | covered above |
| `src/invariant_estimation/sim/collect.py` | `friction_grid` / `friction_grid_by_terrain` + `friction_value` — deterministic `mu` stratification that leaves every other RNG stream in place | 5 tests in `tests/sim/test_collect_dr.py` |
| `config/collect_dr6.yaml` | the slip-heavy dataset config | parsed and asserted by the above |
| `experiments/slip_probe.py` | new — the decisive gate (§2b), the phase-vs-slip R² matrix (§2a), the channel ablation (§2e) | — |
| `experiments/contact_velocity.py` | new — the `v_bc` noise floor and the CoCo Figure-3 analogue (§2d) | — |
| `experiments/slip_attribution.py` | new — horizontal drift vs slip, between and within rollouts (§2c) | — |
| `experiments/summarise_runs.py` | new — the video table: `dz`, yaw, horizontal drift per run | — |
| `experiments/early_release_sweep*.sh`, `night_queue.sh`, `artifacts/run8_pipeline.sh` | serial job queues (one `run_estimator` at a time, own `TMPDIR` each) | — |

**Nothing changes behaviour with the new flags off**, and that is checked rather than asserted:
the analytic baseline run reproduces the recorded N=4 numbers to every printed digit.

**Two pre-existing failures, neither mine.** (i)
`tests/sim/test_collect.py::test_every_leaf_is_float64_with_the_full_time_axis` fails with
`inputs.contact_prob: T=0` -- confirmed pre-existing by stashing my changes and re-running. It
comes from the uncommitted reseed work: `InEKFInputs.contact_prob` defaults to `()` and the
collector's leaf enumeration finds a zero-length array. (ii)
`tests/sim/test_estimator_loop.py::test_threaded_mode_agrees_with_synchronous_within_noise` --
the stale threshold already written up in PORT_NOTES on 2026-07-30. Everything else in
`tests/sim` is green: 82 passed.

---

## What I propose and did NOT do

### 1. Give the network a real contact-point velocity (`J_C q̇`), not a finite difference

The highest-value change on the list, and the only one that mechanically explains why our `Σ_C`
is a phase clock. Of the two forms:

* **Option A — add raw `q̇` channels** (`F: 24 → 30`). Closest to CoCo's stated input list.
* **Option B — replace `v_bc` with the analytic `J_C(q) q̇`** (`F` unchanged at 24).

**Option B, on the measurements.** (i) It moves the noise floor from 0.204 m/s to 0.0036 m/s,
i.e. from *above* 86 % of loaded contact speeds to 5x *below* their median. (ii) It changes no
dimension, so the architecture, `d_in` and every parameter count are untouched. (iii) Option A
asks the network to learn the bilinear contraction `J_C(q)·q̇` from a 50-sample window — the same
argument `features.py:330` already makes for why FK(q) is a real channel and not a rescaling
applies against A. (iv) It hands the network exactly the x-axis of CoCo's Figure 3.

**One blocker that has to be named:** `FusedSensors` carries no measured `q̇` for the nine
filtered joints — only `qd_unfiltered` for the four ankles. Using the joint KF's `q̇̂` would break
CLAUDE.md §7 (no filter mean states), so a raw measured-velocity sensor field has to be added.
`IMUNoise.encoder_vel_std = 5e-3` and `corrupt_velocities` already exist for the ankles, so the
model is there.

**And existing datasets do not need re-collecting.** `truth.q_dot` (the 9 filtered joints) and
`sensors.qd_unfiltered` (the 4 ankles, already noise-corrupted) are both recorded, so a measured
`q̇` can be reconstructed offline as `truth.q_dot + N(0, 5e-3²)` and the analytic `v_bc` built in
the `cache` stage. **Cost: a re-cache (~25 min) plus a retrain (~78 min). No re-collection.**
This is a CLAUDE.md §7 change and it invalidates every existing checkpoint's input meaning.

Queueable, after run 8 reports:
```bash
# after implementing the analytic v_bc in features.make_contact_channels
uv run python train_contactnet.py cache --toe-heel --data data/dr6 --force
uv run python train_contactnet.py norm  --toe-heel --data data/dr6 --force
JAX_PLATFORMS=cuda uv run python train_contactnet.py p0 --toe-heel --data data/dr6 \
    --p0 artifacts/p0_dr6_vbc.npz
JAX_PLATFORMS=cuda uv run python -u train_contactnet.py train --toe-heel --data data/dr6 \
    --p0 artifacts/p0_dr6_vbc.npz --steps 10000 --objective l2_velocity --B 32 --no-remat \
    --sigma-0 1e-1 --lr 1e-4 --widths 128 64 --out artifacts/contactnet_run9.npz
```
The acceptance test is not the loss. It is `slip_probe --checkpoint` R² of `log10 std` **on slip**
rising from run 7's 0.065/0.010/0.008, and the Figure-3 correlation in
`experiments/contact_velocity.py` rising from whatever run 7 scores.

### 0. A default I did not change, deliberately

`EarlyRelease.off_dwell` defaults to **0**, which is worse than 20 on every metric. Two reasons I
left it: `0` is what makes the tick-for-tick equivalence test against
`experiments.process_socket_ablation.causal_early_release` meaningful, and changing it would
silently redefine what `--early-release 0.5` means for anyone reading an older log. **The
recommended setting is `--early-release-off-dwell 20`** and it is what every `E_best_*` clip uses.
If you want it to be the default, the place is `SimSensorReader`'s parameter (leaving
`EarlyRelease`'s at 0 so the parity test keeps its meaning) — one line, but it is your call.

### 2. Whether anything is left in the release schedule

With the off-dwell in place the closed-loop lead is 159 +- 11 ticks at a 0.0% miss rate, and
adding a stance clock on top changes nothing (-0.2556 vs -0.2529). So the schedule is no longer
miss-rate-limited and I would NOT spend more on it without a new idea. The remaining gap to arm
B's 4.3x is not a like-for-like comparison anyway (replay on `dr5` vs closed loop at vx = 0.6).
If it is revisited, the thing to try is a *shorter* lead: 159 ticks is 1.6x what arm B used, and
`--early-release 0.7` (a later release, so a shorter lead) was slightly worse, which suggests the
optimum is broad rather than that longer is better.

### 3. Add an accelerometer bias state

CoCo has `ᴮb_a` (their Eqs. 3, 7); we have no accelerometer bias anywhere — the joint KF carries
`b_ω` only and has no accelerometer measurement, so `ā` is never bias-corrected. Our −0.0134 m/s
is ~1.3 mg of z-bias, right in IMU range. **But `sim/sensors.IMUNoise` injects no accel bias, so
this cannot be the cause of the sink measured in sim** — it is a hardware exposure. Ranked
accordingly. Property test if it is done: inject a synthetic constant `b_a,z`, require the
estimate to converge to it and `e_vz → 0`, and require the augmented-state NEES to stay in band.

### 4. Ablate the history window down from 392 ms

Their `H = 20` at 600 Hz is a **33 ms** span; ours is 392 ms. Their Table VI shows H = 150
(250 ms) was worse for *every* method (CoCo 0.052 vs 0.046). This contradicts a documented choice
of ours (`config.py:52-72`, the "H is not 20" argument). Cheap: `--H` and `--window-span-s` are
already flags; it needs a re-cache per span. Two points (e.g. 100 ms and 33 ms) would settle it.

### 5. Do NOT implement the velocity-level contact constraint

The `TODO(N^v / zero-velocity)` at `inEKF/filter.py:392` should stay deferred. CoCo's Alg. 1 has
exactly one correction block and it is the FK *position* measurement Eq. (8); the zero-velocity
assumption lives entirely in their process model Eq. (5), whose covariance is the learned
quantity — our architecture exactly. Verified in the paper source, not inferred.

### A materially larger run, if one is wanted

Compute was not the binding constraint tonight, and I am not offering it as an excuse. The run
that would settle whether our remaining `Σ_C` gap is method or budget:

```bash
# 100k steps at B=128 on dr6 (or dr6+dr5 pooled), l2_velocity, otherwise run-8 settings.
JAX_PLATFORMS=cuda uv run python -u train_contactnet.py train --toe-heel \
    --data data/dr6 --p0 artifacts/p0_dr6.npz \
    --steps 100000 --objective l2_velocity --B 128 --no-remat \
    --sigma-0 1e-1 --lr 1e-4 --widths 128 64 --log-every 200 \
    --out artifacts/contactnet_run8_long.npz
```
At run 7's measured 4697 s for 10 000 steps at B = 32, this is ~13 h at B = 32 and roughly **2–3
days at B = 128** on one 4070 SUPER — which is the same order as CoCo's 5-day cap and would put
segments-seen within ~3x of theirs rather than 250x. It is worth queueing on the Blackwell only
*after* run 8 and the `v_bc` change report, because a longer run over a feature set that cannot
represent contact motion buys the same phase clock, more precisely.

---

## Honest summary of what did and did not work

**Worked**
* Per-foot causal early release plus a short off-dwell: **2.24x on the sink** (1.81x from the
  source alone), and tilt and velocity improve with it. Turning walk 2.42x.
* The decisive slip gate: slip **is** predictable from the current features (R² 0.32–0.36).
* The `v_bc` noise-floor measurement — an unplanned finding that explains the phase clock.
* `mu` stratification: the defect is confirmed against the shipped data and fixed with a test.

**Did not work, and why**
* The slip-heavy retrain (run 8). Slip R² unchanged, phase lock worse, the contact-speed
  correlation collapsed +0.744 -> +0.257 and went non-monotone, and held out it is worse than the
  heuristic at the metric it optimised. The gate that authorised it was right to run; the answer
  is no.
* The per-**contact** load source — the natural, finer signal — does nothing (±2 %). The toe's
  own peak is late, so its ratio test fires only at the last tick.
* Raising `frac`: monotonically worse (−0.5577 → −0.5856 → −0.6061 at 0.3/0.5/0.7). More loose in
  the wrong place is arm I.
* Blanking-window tuning: 75 and 250 ticks are both worse than 150. Not the lever.
* `peak_mode=prev` on the level test: better than `current` (−0.5335) but far behind per-foot, and
  offline it releases 71 % of stance.
* Reproducing arm B causally at all: impossible from load *or* from a perfect contact-speed
  sensor. Arm B is a covariance-conditioning schedule, not a physical predictor.

**Premises that turned out to be wrong, and I checked rather than repeated them**
* "CoCo reaches centimetre-level agreement" — they report **0.124 m** position ATE on their
  easiest scenario, over 20 s segments aligned at the initial state.
* "We force-teacher-train and they don't" — `ChainedBatcher` carries `(X̂,P)` with
  `episode_s = 43 s`, between their 6 s and 100 s.
* "The network never inflates the horizontal `Σ_C`" — true of run 4 (434x anisotropy, measurement
  socket, N=2); run 6 and run 7 sit at 2.1x and 2.2x.
* "A sliding foot drags the base and gives us horizontal drift" — horizontal drift is 0.01–0.47 %
  of travel and does not track slip *within* a rollout. The problem is vertical, by 1–2 orders of
  magnitude.
* "L2 is structurally insufficient" — their Eq. (9) is the same loss.
