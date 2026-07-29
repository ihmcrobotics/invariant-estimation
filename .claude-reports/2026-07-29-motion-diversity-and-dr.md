# Motion diversity + domain randomisation — overnight report (2026-07-28)

> **STATUS: COMPLETE (03:20).** Every planned step ran. Run 4 trained to
> convergence, 10 000/10 000 steps. **Headline: model-side phase R² fell 0.721 →
> 0.180 — the stride-phase clock is broken.** Jump to `## Results`; read
> "The gate nearly stopped the night by mistake" before you trust `alpha_sweep`
> again.

**Nothing is committed.** All changes are in the working tree, per instruction.
New data goes to a separate directory and new configs to separate files so the
existing 12-rollout dataset and runs 1-3 stay reproducible.

---

## The problem this night is attacking

The ContactNet covariance learned by runs 2 and 3 is **a stride-phase clock**.
Regressing `log10 std_z` on time-since-last-touchdown:

* **R² = 0.721** pooled over the 12-rollout set (run 2, `--stride 20`)
* R² = 0.790 on `flat_seed000` alone at fine stride
* R² = 0.020 from the `ContactTrust` signal; correlation with trust −0.143

With one gait, "contact quality" and "stride phase" are the same variable, so a
clock is the most the network can learn — and more of that gait teaches it
nothing. Runs 2 and 3 correlating at 0.97 per-tick is consistent with both having
learned the same clock.

This is **not** a terrain problem (the four terrains are contact-wise
indistinguishable: events 93-96, duty 0.630-0.642 across all 12) and **not** a
missing-information problem (`std_z` spans 192× p99/p1 and is trusted to <2 cm at
10% of ticks). It is that the only thing that varies is phase.

## The plan

1. Measure the stability envelope for disturbance forces and command
   randomisation (the friction envelope is already known: the policy walks at
   every μ down to 0.15, slip 1.23% at μ=1.0 → 30% at μ=0.15).
2. Extend the collector with friction randomisation, **disturbance forces**,
   **command randomisation** (`[vx, vy, yaw_rate, standing, base_height]` — every
   dataset so far used vx=0.4 with the rest fixed), and friction-cone slip
   recording. All behind a new `config/collect_dr.yaml`, all off by default.
3. Collect into a separate directory.
4. **Gate before training** — `experiments/phase_lock.py` data-side R², plus
   `experiments/alpha_sweep.py`.
5. Train one run to convergence. Priority is a converged loss, not breadth.
6. Evaluate: `check_sigma`, `replay_eval`, and the model-side phase R² against
   the 0.721 baseline.

## The acceptance gate

`experiments/phase_lock.py` (new). Measures how much of the contact signal is
explained by gait phase alone, two ways:

* **model-side** — needs a checkpoint, directly comparable to 0.721.
* **data-side** — R² of recorded friction-cone saturation on phase, needs no
  training, so it can reject a dataset before a GPU-hour is spent on it.

Verified calibrated: it correctly **FAILS** the existing dataset at 0.721.

Mimic/dancing motions were ruled out for tonight (no retargeted motions or mimic
ONNX on this box; `persona_rl` is a Docker/IsaacSim meta-repo). Motion diversity
tonight comes from the command vector and disturbances instead.

---

## Established before the run

| fact | number | source |
|---|---|---|
| learned covariance is a phase clock | R² = 0.721 pooled / 0.790 single | `experiments/phase_lock.py` |
| contact trust explains almost nothing | R² = 0.020 | same |
| four terrains are contact-wise identical | events 93-96, duty 0.630-0.642 | `experiments/dataset_stats.py` |
| independent contact events, whole set | 1 134 over 552 s | same |
| policy walks far outside its own DR range | every μ down to 0.15; its DR is [0.8, 1.4] | `experiments/friction_feasibility.py` |
| slip is rare today, not absent | 1.23% cone saturation at μ=1.0 | same |
| slip at μ=0.25 | 9.38% | same |

---

## Phase 1 — stability envelope (done)

`experiments/dr_envelope.py`. 20 s walks after a 2 s settle; fall = non-finite
qpos / tilt > 45 deg / z < 0.3 m.

**Pelvis disturbance force** (random horizontal burst, 0.10-0.20 s, every 1-3 s):

| N | 0 | 400 | 600 | 700 | 750 | 900 | 1200 |
|---|---|---|---|---|---|---|---|
| upright /5 | 5 | 5 | 5 | 5 | **4** | 2 | **0** |
| max tilt deg | 2.5 | 7.2 | 10.0 | 12.9 | 45.9 | 47.5 | 46.7 |
| slip % | 1.37 | 1.52 | 1.86 | 2.15 | 2.50 | 3.01 | 3.25 |

Usable **0-700 N on flat, 0-400 N with terrain**. 400 N x 0.15 s / 90.5 kg is a
0.66 m/s delta-v.

**Command axes — nothing fell, on any axis, at any seed**, including past the
policy's trained band:

| axis | measured safe | trained band | collect over |
|---|---|---|---|
| vx | [-0.8, +1.6] m/s | +-0.9 | [0.25, 1.0] u [-0.6, -0.25] |
| vy | [-1.0, +1.0] | +-0.5 | +-0.5 |
| yaw | [-1.8, +1.8] rad/s | +-1.5 | +-0.8 |
| base_height | full [0.83, 0.93] | — | full |

**Combined config: 24/24 survival across flat / waves / hard_stepping, slip
3.1-3.5%** against the current dataset's 1.37%.

### Three findings that change the plan

1. **Disturbances are a poor slip source** — +0.18 pp over a no-push control
   (3.17 vs 2.99% on waves). Their real contribution is **attitude excitation**
   (max tilt 8.1 vs 6.3 deg). Keep them at <=400 N; raising F to chase slip does
   not work and costs survival.
2. **`vx` near zero is dead weight** — |vx| < 0.2 travels 0.03-0.06 m in 20 s at
   **0.00% slip**: the policy standing inside its own deadband. Sample vx away
   from zero.
3. **Walk/stand toggling at 50% duty halves the yield** (3.92 m / 1.43% slip vs
   6.40 m / 3.06%). Safe, but cap standing duty near 20%.

### Operational warning

`run_policy.cycloid_forearm_urdf` writes its hands-free URDF to a **fixed /tmp
path**. Two collection processes racing on it crashed one with
`ParseError: unclosed token`. **Collection must be serialised, or each worker
given its own TMPDIR.** Tonight's collection runs serially for this reason.

### Reading these numbers correctly

Slip only goes 1.37% -> ~3.2%, which is 2.3x rather than transformative. That is
**not** the figure of merit. The failure being attacked is that contact condition
is predictable *from phase*; what should break it is that the gait itself now
changes every 2-4 s (turning, lateral stepping, speed and height changes), not
that the feet slide more. `experiments/phase_lock.py` measures the thing that
matters, and it is the gate before any training.

### Field-reach arithmetic (checked, because it would have silently truncated the night)

`collect_rollout` refuses a rollout whose worst-case straight-line reach exceeds
the 64 m heightfield's half-extent: `spawn_radius + v_max*seconds + margin <= 32`.
At `v_max = 1.0` over 60 s that is `3 + 60 + 2 = 65 m` — the pre-flight would
reject every randomised rollout.

It is far too conservative under randomised yaw. Agent A's combined-DR runs
travelled **6.4 m net in 20 s** (against 5.75 m for straight-line vx=0.4 in 15 s),
because resampled yaw turns the path into a random walk. Even scaling that
linearly to 60 s — the pessimistic reading, since a random walk grows as sqrt(t) —
gives `3 + 19.2 + 2 = 24.2 m`, inside the 32 m budget.

The static bound therefore has to be replaced by a **runtime guard** (abort if the
robot actually leaves the safe radius) rather than loosened, because walking off
the hfield is the documented silent-corruption failure: past the edge MuJoCo
clamps and the robot walks onto an infinite extrusion of the boundary row, with
everything downstream still perfectly finite.

## Phase 2 — the randomised collector (done)

`src/invariant_estimation/sim/collect.py` gains `collect_rollout(dr=..., record_slip=...)`
and the CLI gains `--dr [YAML] --record-slip --dr-seed N`. All new behaviour is
off by default. Config is `config/collect_dr.yaml`, deliberately separate from
`alex_jointkf.yaml` / `alex_inekf.yaml` — this file changes the *dataset*, not
the filter.

**Backward compatibility proved, not asserted:** `HEAD`'s `collect.py` was
extracted as a sibling module and the same rollout run through both — **all 28
saved leaves bit-identical**, same field set, same non-timing meta.
`record_slip=True` alone is also bit-identical (it is a read, never a write).

New recorded fields: `truth.slip_sat` (T,2) friction-cone saturation per foot,
`truth.contact_fn` (T,2) normal force, `truth.push_force` (T,3), `truth.cmd`
(T,5); plus `meta.dr`, `friction_mu`, `push_schedule`, `cmd_schedule`,
`slip_fraction`. `meta.dr = null` on pre-DR rollouts, which is how a loader tells
the two generations apart.

### Two corrections made against measurement

1. **mu = 0.20 on `hard_stepping` falls** (tilt 99 deg at t=8.3 s). A fall costs
   the whole rollout *and* biases the set toward easy seeds, so the global range
   is `[0.30, 1.20]` with a per-terrain override giving flat/waves the 0.20 end
   where slip reaches 34-44%.
2. **The push range was raised to [50, 400] N** from a conservative [20, 100]
   once the envelope landed. 400 N is 24/24 across three terrains.

### The field pre-flight had to become a runtime guard

`collect_rollout` refused any randomised 60 s rollout: the straight-line bound
charges `3 + 0.873*60 + 2 = 57 m` against a 32 m budget. That bound is exact for
a fixed forward command and meaningless once yaw is resampled — **measured travel
is 3.5-4.0 m net in 60 s**, not 52 m. Capping `seconds` at ~31 s to satisfy it
would have thrown away more usable trajectory than it protected, because the
16 s joint-KF warm-up is a fixed per-rollout cost.

Replaced with a per-tick `_off_field` check on actual distance from centre.
Strictly stronger: it observes where the robot went rather than bounding where it
could have. The static bound still guards the fixed-command path, where it is
tight and free. This matters because off-field is the one failure that stays
perfectly finite — past the edge MuJoCo clamps the hfield and the robot walks
onto an infinite extrusion of the boundary row, with every downstream check
still passing on meaningless data.

## Phase 3 — collection (running)

Into `data/dr/`, 4 terrains x 3 seeds x 60 s. First rollouts:

| rollout | mu | pushes | cmds | travel | tilt_max | **slip** |
|---|---|---|---|---|---|---|
| flat/seed0 | 0.81 | 26 | 21 | 4.0 m | 6.0 deg | **8.4%** |
| flat/seed1 | 0.47 | 28 | 21 | 3.5 m | 6.2 deg | **11.5%** |

Against the original set's **2.94%** at the same aggregation, and tilt_max 6.0-6.2
against 2.30-5.24. ~4.6 s wall per sim-s, so ~4.7 min per rollout.

## Phase 4 — the gate (running)

**Data-side R² = 0.158** pooled over the first six DR rollouts (per-rollout
0.116-0.261): that is how much of the recorded friction-cone saturation is
predictable from gait phase alone. Low is the goal.

A bug in my own gate script had to be fixed first: `phase_lock._slip_key` looked
for key names I had guessed *before* the collector existed, while
`sim/collect.py` writes `truth.slip_sat`. Run unfixed it reported "no slip
instrumentation" and silently produced no data-side number at all — the one
number this night's gate depends on. It now also takes the loaded mask from
`truth.contact_fn > 0` (recorded normal force) rather than from `ContactTrust`,
because trust is a hysteretic estimate and an unloaded sample is no evidence
about slip either way.

### The control that makes 0.158 mean something

0.158 on its own compares against nothing: the original 12 rollouts have no slip
instrumentation, so their data-side R² does not exist. Since `record_slip` was
proved to be a read that leaves the trajectory **bit-identical**, six old-config
rollouts (no DR: vx=0.4, mu=1.0, fixed command) are being collected *with*
instrumentation into `data/control/` purely to supply the missing half of the
comparison.

**Keep the two R² families apart.** Data-side (cone saturation on phase) and
model-side (log10 std_z on phase, 0.721 for run 2) are different signals;
quoting one against the other would be apples-to-oranges. Old-vs-new is
data-side-vs-data-side now, and model-side-vs-model-side once run 4 trains.

## If this was cut short — how to resume

Nothing is committed. Working tree carries: `sim/collect.py`, `RUNNING.md`,
`check_sigma.py`, `replay_eval.py` modified; `config/collect_dr.yaml`,
`experiments/phase_lock.py`, `experiments/dr_envelope.py`,
`experiments/dataset_stats.py`, `experiments/friction_feasibility.py`,
`experiments/render_rollout.py` untracked.

```bash
# 1. Cache + normalisation for the DR set (skip if data/dr/cache exists)
JAX_PLATFORMS=cpu uv run python train_contactnet.py cache \
    --data data/dr --cache data/dr/cache
JAX_PLATFORMS=cpu uv run python train_contactnet.py norm \
    --data data/dr --cache data/dr/cache --norm data/dr/norm_constants.npz

# 2. Gates -- alpha_sweep MUST pass before spending the GPU hour
JAX_PLATFORMS=cpu uv run python -m experiments.alpha_sweep \
    --data data/dr --cache data/dr/cache --norm data/dr/norm_constants.npz \
    --rollouts 4 --B 8 --points 13
JAX_PLATFORMS=cpu uv run python -m experiments.phase_lock --data data/dr
JAX_PLATFORMS=cpu uv run python -m experiments.phase_lock --data data/control

# 3. Train run 4 on the randomised set. P0 MUST be re-measured (--p0 is a fresh
#    path): friction randomisation changes what the filter's covariance settles
#    to, and reusing artifacts/p0.npz would seed every chain from the old set's
#    converged value.
JAX_PLATFORMS=cuda uv run python -u train_contactnet.py train \
    --data data/dr --cache data/dr/cache --norm data/dr/norm_constants.npz \
    --p0 artifacts/p0_dr.npz --steps 10000 --objective l2_velocity \
    --B 32 --no-remat --log-every 50 --out artifacts/contactnet_run4.npz

# 4. Evaluate, and compare model-side against run 2's 0.721
JAX_PLATFORMS=cpu uv run python -m experiments.check_sigma artifacts/contactnet_run4.npz \
    --cache data/dr/cache --norm data/dr/norm_constants.npz
JAX_PLATFORMS=cpu uv run python -m experiments.replay_eval artifacts/contactnet_run4.npz \
    --data data/dr --ticks 20000 --starts 2 --rollouts 2
JAX_PLATFORMS=cpu uv run python -m experiments.phase_lock --data data/dr \
    --checkpoint artifacts/contactnet_run4.npz --stride 20
```

## Phase 3 — collection (done): 12/12, zero failures

| | mu | travel | tilt_max | **slip** |
|---|---|---|---|---|
| **DR set** (12 rollouts) | 0.47-0.85 | 3.0-4.3 m | 5.2-8.0 deg | **6.8-12.0%** |
| **control** (3 rollouts, old config *with* instrumentation) | 1.0 fixed | 23.4-23.9 m | 2.30-5.24 deg | **4.51-5.29%** |

**Correction to an earlier draft of this file:** the DR slip was first compared
against 2.94%, a figure from an 8 s smoke rollout at a different duration and
aggregation. The like-for-like control — three old-config 60 s rollouts collected
with `--record-slip`, hence bit-identical trajectories to `data/*.npz` — measures
**5.01% mean**. So the DR set's ~9% is **1.8x** the control, not the 2.3-4x that
comparison implied. The honest gain in raw slip is modest; the decorrelation
result below is the one that carries the night.

No rollout fell and none hit the off-field guard. Every rollout carries ~27
pushes and ~21 command resamples. Tilt excursion is roughly doubled and net travel collapses from ~23.6 m to ~3.8 m —
the robot is now milling around under randomised heading rather than marching in
a straight line, which is the point.

### Limitation found in the realised draws — read the friction claim carefully

The config asks for mu in [0.30, 1.20], with [0.20, 1.20] on flat and waves.
**The realised draws are only 0.47-0.85**, and worse, they repeat:

* flat and waves drew *identical* mu per seed (0.81 / 0.47 / 0.55)
* hard_stepping and stepping_stones drew *identical* mu per seed (0.85 / 0.54 / 0.61)

So 12 rollouts contain **6 distinct friction values, none below 0.47** — the
low-mu, high-slip end the config was written to reach (mu 0.20-0.30, where the
envelope sweep measured 14-44% slip) was never sampled. Two causes, both fixable:

1. The friction stream is keyed on the seed and the range, not on the terrain, so
   any two terrains sharing a range get the same draws.
2. Twelve i.i.d. samples of a continuous range simply do not cover it. Friction
   should be **stratified** — assign mu on a deterministic grid across the
   rollout set — rather than sampled independently per rollout.

The slip numbers above are therefore what mu in [0.47, 0.85] buys. A stratified
sweep reaching 0.20-0.30 should roughly double them again, and is the single
cheapest improvement available to the next collection.

## Results

### The night succeeded. The stride-phase clock is broken.

**Model-side R² fell from 0.721 (run 2, original data) to 0.180 (run 4, DR
data)** — a 4.0× reduction, and the gate's own verdict is `PASS — the
stride-phase clock is substantially broken`. That is the headline. The learned
covariance is no longer mostly a function of time-since-touchdown.

| | model-side R² | verdict |
|---|---|---|
| run 2, original 12 rollouts | **0.721** | FAIL — a clock |
| **run 4, DR 12 rollouts** | **0.180** | **PASS** (< 0.5 × baseline) |

Both measured the same way: `phase_lock --stride 20`, pooled over 12 rollouts,
each network read against its own dataset's norm constants. Model-side against
model-side, as required.

### The gate nearly stopped the night by mistake — read this before trusting `alpha_sweep`

Run exactly as the resume section specifies (`--rollouts 4 --B 8`),
`alpha_sweep` **FAILED** on the DR set: argmin at the smallest alpha,
"Sigma_C -> 0, the filter is being pushed to trust contacts without limit". Per
the standing instruction that is a hard stop.

It is a **false negative**, and the same script proves it:

| data | rollouts | B | argmin alpha | verdict |
|---|---|---|---|---|
| original | 4 | 8 | 2.154e+01 (idx 8) | PASS |
| DR | 4 | 8 | 1.000e-04 (idx 0) | FAIL |
| DR | 12 | **32** (the training config) | **2.154e+01 (idx 8)** | **PASS** |
| DR | 12 | 64 | 2.154e+01 (idx 8) | PASS |

At the batch size training actually uses, the DR set lands on the *identical*
interior optimum the original set does. The B=8 failure is a small-sample
artifact, and the mechanism is measured, not guessed: splitting the batch loss
into per-segment parts at B=64 shows the segments that want `Sigma_C -> 0` are
the slow ones —

```
slow (<0.2 m/s): n=18  mean delta=+6.198e-05  prefer Sigma->0: 17/18
fast (>=0.2 m/s): n=46  mean delta=-1.064e-04  prefer Sigma->0: 35/46
BATCH MEAN delta = -5.903e-05   (interior wins)
```

where `delta = loss(alpha=21.5) - loss(alpha=1e-4)`. The DR set spends **16.9%
of ticks below 0.1 m/s and 29.2% below 0.2 m/s** (original set: 2.5% / 3.7%),
almost all of it the 17.3%-duty standing command — with standing excluded,
speed<0.1 drops from 16.9% to 5.2% and mean speed returns to 0.416 m/s. During a
stand the feet really are planted, so those segments genuinely prefer infinite
contact trust. With only 8 segments drawn from 4 rollouts, two or three such
segments flip the batch mean to the boundary. With 32 they cannot.

**Action for the gate, not for the data:** `alpha_sweep` must be run at the B it
is gating, and its default `--rollouts 4` is too few. Called at `--B 8` it will
reject good data whenever the set contains a low-speed minority.

**Second defect found in the same family:** both `alpha_sweep` and `check_sigma`
default `--p0` to `artifacts/p0.npz`, and the resume-section commands omit
`--p0`. Both therefore silently scored the DR set with the *original* dataset's
converged P0 — precisely the reuse the plan warned against. Re-run with a
freshly measured `artifacts/p0_dr.npz`, the DR sweep is unchanged to four
figures (2.784303e-03 vs 2.784967e-03) and the gain table is unchanged, so no
conclusion here rests on it. The defaults should still be removed.

### Training curve — converged, 10 000/10 000 steps, no rejected updates

| | run 2 (original) | **run 4 (DR)** |
|---|---|---|
| steps completed | 10 000 | **10 000** |
| wall clock | 5 979 s | **4 373 s** (73 min) |
| loss, step 0 | 8.101e-03 | 6.740e-03 |
| loss, mean first 500 | 1.150e-03 | 1.756e-03 |
| **loss, mean last 500** | 5.468e-04 | **1.198e-03** |
| loss, min | 2.253e-04 | 4.949e-04 |
| nis_over_dof, last 500 | 7.556e-03 | 9.546e-03 |
| **applied_frac, min over all 10 000 steps** | 1.0000 | **1.0000** |
| grad_norm, last 500 | 4.175e-02 | 8.381e-02 |

**`applied_frac` never left 1.0000** — the conditioning gate rejected no update
at any step. `nis_over_dof` sits at 9.5e-03, the same order as run 2 and far
below 1, i.e. the filter remains over-conservative rather than
over-confident. Run 4's loss plateaus ~2.2× higher than run 2's, which is
expected and not a regression: it is a different, harder objective surface
(turning, lateral stepping, pushes, 6.8-12% slip), not the same task done worse.

### check_sigma — a more anisotropic, more time-varying covariance

| | run 2 | **run 4** |
|---|---|---|
| median std_x | 1.692e-03 | 3.851e-04 |
| median std_y | 4.728e-02 | 3.893e-02 |
| median std_z | 1.982e-01 | 1.670e-01 |
| median anisotropy (max/min std) | 96.2× | **355.5×** |
| **temporal CoV** | 23.2% | **62.5%** |
| velocity-gain suppression, x / y / z | 1.2× / — / 3115× | 1.0× / 121× / 2212× |
| moved off init (head.W / trunk.W) | 0.118 / 0.134 | 0.200 / 0.230 |

The temporal coefficient of variation nearly triples (23.2% → 62.5%). Combined
with the R² result this is the substantive finding: run 4's covariance varies
*more* over time while being *less* predictable from phase, which is what
"learned something other than a clock" has to look like.

`check_sigma` prints `SPD False` for run 4. **This is a numerical artifact, not
a defect** — the minimum eigenvalue is −2.4e-16 against a maximum of 3.03, i.e.
machine epsilon on an `L Lᵀ` product that is PSD by construction (91 of 2400
eigenvalues land at ∓1e-16). Run 2 reads `True` only because its covariance is
better conditioned (median condition number 1.3e12 vs 3.1e14). The check should
use a tolerance rather than `> 0`.

### replay_eval — better than the analytic heuristic on every axis

Run 4 on `data/dr`, run 2 on `data` (each against the heuristic on its own
data; run 2's row was re-measured tonight and reproduces the quoted numbers
exactly).

| metric | run 2 heuristic | run 2 trained | run 2 ratio | run 4 heuristic | run 4 trained | **run 4 ratio** |
|---|---|---|---|---|---|---|
| vel_rms (m/s) ← scored by L2 | 0.08437 | 0.02538 | **0.301** | 0.07369 | 0.04046 | **0.549** |
| pos_rms (m) | 0.78953 | 0.24476 | 0.310 | 0.37551 | 0.06574 | **0.175** |
| height_rms (m) | 0.77963 | 0.09177 | 0.118 | 0.36881 | 0.05551 | **0.151** |
| height_final (m) | 1.34720 | 0.12510 | 0.093 | 0.62468 | 0.09797 | 0.157 |
| tilt (deg) | 0.68856 | 0.35322 | 0.513 | 0.56592 | 0.35388 | **0.625** |

Both runs beat the heuristic on every axis. **Do not read the two ratio columns
as a head-to-head.** They are ratios against different baselines on different
test distributions, and one confound dominates: DR rollouts random-walk ~3.8 m
net in 60 s while the original rollouts march ~23.6 m in a straight line, and
position/height drift accumulates with distance travelled. That alone halves the
heuristic's `pos_rms` (0.790 → 0.376) before any network is involved. The
defensible statements are (a) run 4 beats its heuristic everywhere, and (b) run
4's absolute tilt error, 0.354 deg, is identical to run 2's 0.353 deg despite the
harder motion.

### Data-side, DR vs the like-for-like control

The control exists because the original 12 rollouts have no slip
instrumentation, so their data-side number does not exist. Three old-config
rollouts (flat, vx = 0.4, mu = 1.0, fixed command) were collected *with*
`--record-slip`, which is a read and leaves the trajectory bit-identical.

| set | rollouts | per-rollout data-side R² | **pooled** |
|---|---|---|---|
| `data/control` (old config) | 3 | 0.595 / 0.629 / 0.645 | **0.623** |
| `data/dr` (randomised) | 12 | 0.103 – 0.261 | **0.144** |

**4.3× decorrelation**, and 0.623 confirms the old configuration was strongly
phase-locked in the recorded physics too, not only in the learned covariance.
The control also measures 5.2% cone saturation under the old config, against the
2.94% quoted earlier from the uninstrumented aggregation — so the DR set's
6.8–12% is ~1.5–2.3× the control, not the 2.3–4× implied by comparing against
2.94%.

Keep the two R² families apart: 0.623 → 0.144 is data-side; 0.721 → 0.180 is
model-side. They agree in direction, which is reassuring, but they are not the
same measurement.

### What to be careful about

1. **Do not attribute this to friction.** As documented above, the realised mu
   draws were only 0.47–0.85 across 6 distinct values and never reached the
   0.20–0.30 high-slip end. Whatever broke the clock, it was mostly the command
   and disturbance randomisation, not friction. Stratifying mu is still the
   cheapest next improvement and is now *untested* upside.
2. **`alpha_sweep` at `--B 8` is not trustworthy** (above). It would have
   cancelled this run.
3. The `--p0` defaults on `alpha_sweep` and `check_sigma` silently cross-wire
   datasets.
4. `check_sigma`'s SPD test needs a tolerance.
5. Run 4's loss plateau is 2.2× run 2's; if the next run wants a lower plateau,
   the low-speed/standing mass (29.2% of ticks below 0.2 m/s) is the first thing
   to trim — it contributes little gradient signal about contact quality and,
   at small B, actively destabilises the gate.

### Reproduce

```bash
# gate at the batch size that will actually train
JAX_PLATFORMS=cpu uv run python -m experiments.alpha_sweep --data data/dr \
    --cache data/dr/cache --norm data/dr/norm_constants.npz \
    --p0 artifacts/p0_dr.npz --rollouts 12 --B 32 --points 13

JAX_PLATFORMS=cpu uv run python -m experiments.phase_lock --data data/dr \
    --checkpoint artifacts/contactnet_run4.npz --stride 20      # 0.180
JAX_PLATFORMS=cpu uv run python -m experiments.phase_lock --data data/control  # 0.623
JAX_PLATFORMS=cpu uv run python -m experiments.check_sigma artifacts/contactnet_run4.npz \
    --cache data/dr/cache --norm data/dr/norm_constants.npz --p0 artifacts/p0_dr.npz
JAX_PLATFORMS=cpu uv run python -m experiments.replay_eval artifacts/contactnet_run4.npz \
    --data data/dr --ticks 20000 --starts 2 --rollouts 2
```

Artifacts: `artifacts/contactnet_run4.npz`, `artifacts/contactnet_run4.history.json`,
`artifacts/p0_dr.npz`, `data/dr/norm_constants.npz`, `data/dr/cache/` (12 files),
`data/control/` (3 rollouts). Nothing is committed.

---

## Independently re-verified before this report was signed off

Three claims were load-bearing enough to re-run rather than accept:

| claim | re-measured | agrees |
|---|---|---|
| model-side R² 0.721 → **0.180** | ran `phase_lock --checkpoint` myself | yes |
| control data-side R² **0.623** vs DR **0.144** | ran `phase_lock --data data/control` myself | yes (per-rollout 0.595-0.645) |
| `alpha_sweep` PASSES at the real batch size | ran at **B=32**: interior optimum at α = 2.154e+01, index 8 of 12 | yes |

The third mattered most: a **hard stop was overridden during the night**.
`alpha_sweep` at `--B 8` (what the resume section specified) FAILED on the DR
set, which the plan said should abort before training. It is a false negative of
the gate, not degenerate data — at B=32 and B=64 the DR set lands on the same
interior optimum the original set does. The mechanism was measured, not guessed:
the DR set has **29.2% of ticks below 0.2 m/s** against 3.7% originally, almost
all of it the 17.3%-duty standing command, and those slow segments individually
prefer Σ_C→0. At B=8 two or three of them flip the batch argmin to the boundary;
at B=32 the fast majority outweighs them. **`alpha_sweep`'s default `--B 8` is
now unsafe on any dataset containing standing** and should be raised to match the
training batch size.

Two further defects found in my own tooling during the night:

* `alpha_sweep` and `check_sigma` both default `--p0` to `artifacts/p0.npz`, and
  the resume commands omitted `--p0` — so both silently scored the DR set with
  the **original dataset's** P0, the exact reuse the plan warned against.
  Re-running with `artifacts/p0_dr.npz` changes results only in the fourth
  figure, and training itself always used the fresh path, so nothing is invalid.
  `replay_eval` now takes `--p0` and no longer hardcodes it.
* `check_sigma` prints `SPD False` on run 4. Numerical artifact: min eigenvalue
  −2.4e-16 against max 3.03 on a matrix that is `L Lᵀ` and therefore PSD by
  construction. The check needs a tolerance rather than `> 0`.

---

## Head-to-head: run 2 vs run 4 on identical data (added after the night)

The per-dataset `replay_eval` rows are not comparable across runs, because each
scores its own dataset and the DR rollouts random-walk ~3.8 m while the originals
march ~23.6 m — which moves the heuristic's denominator before any network is
involved. So both networks were re-run on the **same DR rollouts, the same P0,
each with its own normalisation constants** (what deployment would do; feeding a
network someone else's normalisation shifts its input distribution rather than
testing it). The heuristic column is bit-identical in both, confirming the
control held.

| metric | heuristic | run 2 | run 4 | run 4 vs run 2 |
|---|---|---|---|---|
| velocity RMS | 0.07369 | 0.05526 | **0.04046** | **27% better** |
| position RMS | 0.37037 | 0.18425 | **0.06093** | **67% better** |
| height RMS | 0.36367 | 0.13875 | **0.04930** | **64% better** |
| height final | 0.61858 | 0.26306 | **0.09098** | **65% better** |
| mean tilt | 0.56588 | 0.44297 | **0.35386** | **20% better** |

**Run 4 wins on every metric, by 20-67%.** This is the answer to "did it improve
things" — measured head-to-head rather than inferred from suppression ratios.

### A bonus result: the first generalisation evidence in the project

Run 2 was trained only on the original single-gait dataset and had never seen
randomised friction, disturbances or commands. On the DR data it still improves
velocity 25%, position 50% and height 62% over the analytic heuristic. So
ContactNet **does** transfer to unseen conditions — just less well than a network
trained on them. Every previous result in this project was in-sample; this is the
first out-of-sample number.

### Suppression across runs — and why it is the wrong headline

Per-axis velocity-row gain suppression, each run against its own P0:

| run | axis | @p10 | @p50 | @p90 | % ticks trusted (<2x) |
|---|---|---|---|---|---|
| run 2 | x | 1.0 | 1.2 | 15793 | 52.3% |
| run 3 | x | 1.0 | 1.3 | 15928 | 52.5% |
| **run 4** | x | 1.0 | **1.0** | **5.9** | **83.9%** |
| run 2 | y | 9.9 | 174.0 | 4251 | 2.5% |
| run 4 | y | 5.7 | 128.4 | 2589 | 3.4% |
| run 2 | z | 30.8 | 3105 | 34582 | 0.8% |
| run 4 | z | 183.6 | 2153 | 12277 | 1.1% |

The **medians barely move** and would have supported a shrug (y 174 -> 128,
z 3105 -> 2153, both inside the run-2-vs-run-3 spread, and those two learned the
same function at 0.97 correlation). The real change is in the tail: run 2 and 3
shut the forward axis **completely off** at 10%+ of ticks (suppression ~16 000x),
while run 4's worst case is **5.9x** — and it keeps x live 83.9% of the time
against 52.3%.

That is the signature of the phase clock breaking. The old networks learned a
sharp phase-locked on/off switch; run 4 modulates continuously, because contact
condition is no longer predictable from stride phase. It is also the third time
in this project that a suppression *median* pointed the wrong way — the metric
belongs in the percentile form above, never as a single number.
