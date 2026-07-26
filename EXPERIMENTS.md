# EXPERIMENTS.md — why the IHMC walking policy falls in `run_policy.py`

Running log of the sim-to-sim investigation: what was measured, with numbers, and what it ruled
out. The point of this file is that **nothing here has to be re-derived**. If you are about to
test something, check the "Ruled out" table first.

**Status (2026-07-26): SOLVED.** Root cause was `mj_objectVelocity(..., mjOBJ_BODY, ..., flg_local=1)`
in `build_obs`, which resolves the velocity in the body's **inertial** frame, not its body frame. For
Alex the pelvis `body_iquat` is ~180° about (1,0,1)/√2, so the policy was fed a `base_ang_vel` with
the **x and z gyro axes swapped and y negated** — the equivalent of bolting the IMU on rotated. Fix:
`mjOBJ_XBODY`, which uses the body frame. One-token change, `run_policy.py`.

After the fix, all three policies hold for 30 s and the baseline reproduces the Java reference almost
exactly:

| | Java (SCS2-MuJoCo) | ours, after the fix |
|---|---|---|
| root height | 0.8918 | 0.895 |
| tilt | < 1.2° | 1.1° |
| `\|action\|` | flat ~1.86 | flat 1.86 |
| `ncon` | 8 | 8 |

And it walks: `vx = 0.3` → 0.25 m/s, `vx = 0.6` → 0.60 m/s, `vy = 0.3` → 0.26 m/s lateral, all
upright over 15 s with tilt < 3°. (A yaw-rate command does not turn — unresolved, see §8.)

**Why this hid for so long, and the lesson.** The term-by-term observation diff against Java reported
`base_ang_vel` as matching, because the state I seeded set only *joint* velocities and left the
free-joint velocity at zero — so both sides read (0,0,0) and the axis permutation was invisible. Every
other term genuinely was bit-identical. `projected_gravity` was correct throughout because it uses
`d.xmat[bid]`, the actual body frame, on a different code path. **A comparison that agrees on a
zero is not a comparison.** Perturb every channel you claim to have verified.

Companion docs: `RUNNING.md` §"Watching an RL policy" (how to run things, the ground-truth table),
`POLICY_DEBUG.md` (earlier sessions).

---

## 0. Reference implementations, and which is authoritative for what

| Source | Location | Authoritative for |
|---|---|---|
| **Java / SCS2-MuJoCo** | `~/workspaces/robot-stuff/alex`, `ihmc-internal-software/.../rlController/`, jar `scs2-mujoco-simulation` | The closest maintained MuJoCo deployment. **This is the thing that works** — match it. |
| **IsaacLab** | `~/alex/persona_rl/projects/ihmc_lab` (+ `shared_lab`) | How the policy was *trained*: obs terms, action scale, gains, default pose, actuator model. |
| `alexander-mujoco` | `~/alex/alexander-mujoco` | **Old MJX training setup — conventions are stale.** Useful for structural plumbing only (how joints/actions are fed). Do NOT copy its gains, contacts, or armature. |

Key entry points: `AlexRLSimulation.startHeadlessMujocoSimulation`, `RLController.doControl`,
`RLJointOutputProcessor`, `ObservationDefinitions`, `RLDesireds`, `RLHeightManager`,
`AlexSimulationCollisionModel`, `MujocoMultiBodyRobotFactory`, and on the training side
`tasks/locomotion/alex_ihmc_walk_env_cfg.py` + `robots/alex.py`.

### Which policy goes with which model

`AlexRLModels` dispatches on the robot version:

- cycloid forearms both sides → `AlexFullWalkingModelDefinition`, dir `2026-07-10_baseline`,
  `getModelName() == "baseline_walking"`, 29 joints. **This is `--policy baseline`.**
- nub forearms → `Alex002NubWalkingModelDefinition`, `getModelName() == "walking_baseline"`,
  23 joints. **Naming trap:** "walking_baseline" is *not* the 29-joint baseline we run.
- `AlexVersion.getPhysicalRealityVersion()` reads `$IHMC_ALEX_UNIT` (default 2 → 002 → NUB).
  Unit 001 = `AlexV2Version.CYCLOID_FOREARMS`.

---

## 1. The Java reference numbers

Produced by `AlexMujocoObsDumpTest` (see §6). Standing, `CYCLOID_FOREARMS`, `baseline_walking`,
SCS2's MuJoCo engine, 10 s:

```
tilt                    < 1.2 deg  (max 1.19)
root height             flat at 0.8918
lowestFootToRootHeight  0.9237 -> 0.8901, then flat
|action|                FLAT at ~1.86 for the whole run, never drifts
base_height command     0.9397 -> 0.8902 over ~0.4 s, then flat
ncon                    8 (two flat feet, 4 corners each)
```

Our harness **before** the gyro fix: `|action|` climbed 1.15 → 7 → 20+, tilt 8° at 0.5 s, over at
~1.3 s. **After** the fix: root height 0.895, tilt 1.1°, `|action|` flat at 1.86, `ncon` 8 — i.e. the
Java column above, reproduced.

---

## 2. Verified IDENTICAL to Java (do not re-litigate)

| Thing | Evidence |
|---|---|
| Body | Java's own compiled MJCF: **90.539488 kg**, `nv=35`, per-body mass / inertia / ipos match ours exactly |
| Observation vector | at matched state `projected_gravity`, `base_velocity_plus_standing`, `joint_pos_rel`, `joint_vel_rel`, `last_action` all diff **0.0**. `base_ang_vel` was **NOT** genuinely covered — see the root cause; it now matches to 1e-4 with a nonzero base twist |
| ONNX network | our `onnxruntime` on Java's observation reproduces Java's action, median **8e-3** (residual = one-tick sampling skew in the CSV, not a real error) |
| Home / default pose | IsaacLab `init_state.joint_pos` == Java `<joint>_q_home` == `policy_cfg` `homePosition`, all 29 joints, **0 mismatches** |
| Action scale | `alex.py: ALEX_JOINT_SCALE = 0.3` == the yaml == ours |
| kp, kd **and** effort limit | `alex.py` `STIFFNESS_*` / `DAMPING_*` / `EFFORT_LIMIT_*` == the `policy_cfg` yaml == ours, **all 29 joints, all three columns** |
| Observation scaling | **no `scale=` and no `clip=`** on any `ObsTerm` in `alex_ihmc_walk_env_cfg.py` — unscaled confirmed at the training source |
| Joint order | `model.getOrderedJointNames()` == the yaml `jointParameters` order == ours |
| Feet flat / contacts | foot geom z-axis **0.000°** off world z at home; `ncon = 8`. (The `ncon ≈ 2` noted earlier is a *symptom* of falling, not a cause.) |
| Sensor path | in sim `raw_q*` == `filt_q*` == the simulated robot's q/qd — the policy sees ground truth, same as we do |
| Contact / solver params | Newton, implicitfast, iterations 25, noslip 5, impratio 1, pyramidal, friction `1 0.05 0.01`, solref `0.02 1`, solimp `0.9 0.99 0.0007 0.5 2`, condim 4 |

---

## 3. Real defects found and FIXED

| Defect | What was wrong | Fix |
|---|---|---|
| **ROOT CAUSE — gyro axes permuted** | `mj_objectVelocity(m, d, mjOBJ_BODY, bid, v6, 1)` resolves the velocity in the body's **inertial** frame. Alex's pelvis `body_iquat` is ~180° about (1,0,1)/√2, so `base_ang_vel` came out with x/z swapped and y negated. `‖∂a/∂base_ang_vel‖ = 4.6`, and it is one of only two base signals the policy gets. | **`mjOBJ_XBODY`** instead of `mjOBJ_BODY` — that flag uses the body frame, which is what Java's `root_AngularVelocity` means. Matches the Java dump to 1e-4. |
| **Wrong body** | `alex_with_imus.urdf` is `FULL_ROBOT_ABILITY_HANDS`: 141 links, 91.5126 kg. The policy requires `CYCLOID_FOREARMS`: 49 links, 90.539488 kg. Extra 0.973 kg at both wrists. | `cycloid_forearm_urdf()` strips the two `*_ABILITY_HAND_ADAPTER` subtrees (92 links). Result matches the Java cycloid model exactly — same mass **and** full inertia tensor on every link. |
| **Almost no collision geometry** | Only 2 foot boxes existed, so the robot sank through the floor once it tipped (pelvis z → −0.80 vs +0.21). | Port SCS2's collision set. |
| **Wrong source for the collision set** | I first ported the 32 URDF `<collision>` primitives. SCS2 does **not** use those — it builds collidables from `AlexSimulationCollisionModel`: **7 geoms** (pelvis/torso/head/2 gripper capsules + 2 foot boxes). | `SCS2_COLLISION_GEOMS`, transcribed from the MJCF SCS2 compiles. |
| **Foot box too small** | URDF box `0.22 × 0.10 × 0.02 @ (0.05, 0, −0.06)`. Java balances on `0.26 × 0.14 × 0.055 @ (0.045, 0, −0.05)` in the ankle-roll frame — 15% shorter and **29% narrower in y**. | Use Java's. |
| **Self-collision was on** | robot geoms *and* floor all at contype/conaffinity 1/1, so the foot boxes caught each other. | Java's grouping: robot 1/2, terrain 2/1 — robot geoms test only against terrain, deliberately. |
| **`base_height` was a constant** | We passed 0.90. Java's `RLHeightManager` ramps it from the measured root-above-lowest-sole to `homeHeight = 0.89` as a zero-end-velocity cubic over `GO_HOME_DURATION = 0.5 s`, settling at 0.8902. `‖∂a/∂base_height‖ = 17.5`, so 0.0098 of offset moves the action by 0.11. | `height_command()` + `lowest_foot_to_root_height()`. Target changed 0.90 → **0.89**. |
| **`foot_rest_height` scanned all box geoms** | Fine with 2 geoms; wrong once pelvis/torso boxes exist. | Scoped to `FOOT_GEOMS` by name. |

Also pinned along the way: `ANKLE_HEIGHT = 0.072` (`AlexV1PhysicalProperties`) — the sole plane sits
that far below the `*_FOOT` body frame. With it, `lowest_foot_to_root_height()` reproduces Java's
`lowestFootToRootHeight` to **1e-5** (0.89026 vs 0.89027).

**None of these fixes make the robot stand.** Measured contribution, baseline policy, 30 s:

| foot box | height command | tilt @0.5 s | fell |
|---|---|---|---|
| URDF (0.22×0.10×0.02) | constant 0.90 | 15.8° | 1.32 s |
| URDF | ramped | 9.1° | 1.46 s |
| SCS2 (0.26×0.14×0.055) | constant | 13.2° | 1.20 s |
| **SCS2** | **ramped** (current) | **7.9°** | 1.34 s |

The height ramp roughly halves early tilt; the bigger foot alone does nothing. Outcome unchanged.

---

## 4. Ruled out by experiment

Each was tried, each still falls at ~1.2–2.6 s. Closed loop, baseline policy, unless noted.

| Hypothesis | Result |
|---|---|
| Ability hands vs cycloid body | both fall; 1.07% mass difference |
| Only-feet vs 32 URDF primitives vs Java's 7 geoms | all fall; changes where it lands, not whether |
| Ground: infinite plane vs Java's 82 tiled 25×25×0.5 boxes | no change |
| Physics rate 200 Hz / 500 Hz / 2 kHz | no change (200 Hz is what training used: `SIM_DT = 0.005`) |
| `armature`: repo rotor table / training values / **0 (Java's)** | no change. Open-loop error is slightly *worse* at armature 0 (0.0112 vs 0.0085 rad at t1) |
| `damping = kd` (our emulation) vs Java's passive 0.05 | `kd` is right: open-loop one-tick error 0.011 vs 0.056 rad with 0.05 |
| Position servo vs explicit torque PD (`τ = clamp(kp·e + kd·(0−q̇), ±τmax)` into `qfrc_applied`, recomputed every physics step, as Java does) | our torque-PD port tracks far *worse* (0.67–1.05 rad at t1). The position servo is the better emulation. |
| Actuation delay 0 / 10 / 20 / 25 / 30 / 40 ms — training uses `DelayedPDActuatorCfg(min_delay=4, max_delay=6)` sim steps = 20–30 ms | no change at any value |
| `LIMIT_RESIDUALS_FROM_PEAK_TORQUES` (clamp the setpoint to `q ± τmax/kp` and feed the *clamped* residual back into `last_action`) | drops `|a|max` 82 → 38, does not stop the fall. **And it never binds while standing** — `limitedResidual == raw residual` exactly in the Java dump. |
| Seeding from Java's own steady state (its q, q̇, root pose, `last_action`) | still leaves it: `|a|` goes 1.86 → 1.65 → 2.03 → … → 38 |
| URDF joint position limits, and ×0.9 soft (training: `soft_joint_pos_limit_factor = 0.9`) | no change. **Java's compiled MJCF has zero joint ranges** |
| URDF joint velocity limits (~9–17 rad/s, `qvel` clamp emulating PhysX `maxJointVelocity`) | no change. Java has none either |
| Effort limits | not a discrepancy: yaml `maxEffort` == training `effort_limit_sim` for all 29 joints. (Note the URDF `<limit effort>` differs — ankles 193.6/145.2 vs 129.0/40.0 — but neither Java nor training uses the URDF value.) |

Earlier sessions also swept, all diverging: kp ×{1..40}, integrator {implicitfast, Euler},
init {clean, settle}, base-frame rotation {I, ±90°, 180°}, stand flag {0,1}, action {±0.3a, +a},
velocity-term signs, base_height across its range, and contact tuning.

---

## 5. Measurements that constrain the problem

### 5a. The policy is stiff in the STATE, not in its own action

Numerical Jacobian of the ONNX output w.r.t. each observation block, at Java's steady state
(largest singular value):

```
||d action / d projected_gravity||  = 24.67
||d action / d base_height||        = 17.53
||d action / d joint_pos_rel||      = 16.76
||d action / d base_velocity+stand|| =  6.30
||d action / d base_ang_vel||       =  4.61
||d action / d joint_vel_rel||      =  2.55
||d action / d last_action||        =  0.92   <-- CONTRACTING
```

A 0.01 rad joint error moves the action by 0.17. With the state frozen, iterating
`a ← π(obs(a))` **converges** (1.862 → 1.96).

### 5b. Open-loop plant comparison

Feed Java's own `qdes` setpoints into our model from Java's initial state, compare q to Java's:

```
t = 0.02 s   max|dq| = 0.0085 rad
t = 0.10 s   max|dq| = 0.035
t = 0.50 s   max|dq| = 0.082
```

Per-joint at t = 0.02 s the error is **almost perfectly symmetric left/right** and concentrated in
the sagittal load-bearing joints: `HIP_Y` ≈ 0.0084, `KNEE_Y` ≈ 0.0052, then arms/wrists ~0.003.
Leg total 0.032, arm total 0.026.

**Caveat on reading this test:** a standing biped is stabilised closed-loop through the floating
base, so an open-loop setpoint replay *cannot* stay upright regardless of plant fidelity — by t = 2 s
every variant has fallen and `|dz| ≈ 0.71 m`. Only the first ~0.1 s is meaningful.

### 5c. Rigid-body dynamics vs Java's compiled MJCF — exactly equal

Loading `/tmp/scs2-mujoco-*/world.xml` alongside our model and putting both in the same
configuration (no simulation, no contact, no actuation):

```
mass matrix, armature removed :  max|dM| = 1.9e-11   (relative 2.1e-13)
bias force (gravity+Coriolis) :  max|dB| = 1.9e-10 N / Nm
subtree CoM                   :  max|d|  = 4.1e-15 m
```

So the kinematics and inertias are **identical**. The *only* mass-matrix difference is our rotor
armature: `M[HIP_Y, HIP_Y]` java 1.9166 vs ours 2.0836 — armature 0.167 on an effective inertia of
1.92, i.e. **+8.7%** on the hip and knee. Java's compiled MJCF sets `<joint armature="0.0"/>`
globally, so armature is a deliberate deviation on our side (the estimator needs the rotor table
for `Qa`). It is worth knowing about but was NOT the cause — removing it changed nothing.

This measurement is what finally cornered the problem: with M, the bias force and the CoM provably
equal, and the actuator, contact set and timestep all swept, the discrepancy had to be in an
observation channel. The only one never genuinely perturbed was `base_ang_vel`.

### 5d. The old synthesis (superseded)

Pre-fix reasoning was: obs identical + network identical + state identical ⇒ divergence must enter
through state evolution, so hunt a small plant mismatch amplified by the state gain of ~17. The
premise was wrong — the observation was *not* identical. Kept here because the sensitivity numbers
in §5a and the open-loop method in §5b remain valid tools.

---

## 6. Reproducing the Java reference

`~/workspaces/robot-stuff/alex/src/test/java/us/ihmc/alex/rlController/AlexMujocoObsDumpTest.java`
— a diagnostic, not a real test. Writes `/tmp/alex_java_obs_dump.csv` (per 50 Hz tick: root
pose/twist, gravity vector, commands, and per-joint q/qd/home/lastAction/residual/qdes, plus the
sensor-processing and controller-side copies of q/qd) and `/tmp/alex_java_yovariables.txt`.

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
cd ~/workspaces/robot-stuff/alex
xvfb-run -a ../gradlew :alex-test:test --offline --tests '*AlexMujocoObsDumpTest*'
```

Gotchas, each of which cost time:

- The test source set is its own gradle project: **`:alex-test:test`**, not `:alex:test` (which
  reports `NO-SOURCE`). Use `../gradlew` from inside `alex/` — it is an included build.
- `AlexStateEstimatorParameters` defaults to `JOINT_KF` but only builds the IMU pairs eagerly on
  `RobotTarget.REAL_ROBOT`, so in SCS the pre-filter constructor throws
  **"Base IMU is null, check the kinematic tree."** Call
  `setJointLevelEstimatorType(ALPHA_COMPLEMENTARY)` (empty pair list ⇒ pass-through) or `JOINT_KF`
  explicitly to trigger the lazy build.
- xvfb is required even headless.
- **Resolve YoVariables by full name.** `q_LEFT_HIP_X` exists on both the simulated robot and the
  controller-core feedback toolbox; `LEFT_HIP_X_q_prev` exists once per instantiated RL model
  (`baseline_walking`, `mout_walking`, …). `RLEstimates` / `RLDesireds` / `RLData` live under the
  **controller** thread (`…HumanoidHighLevelControllerManager.RLControllerState.*`), not the
  estimator thread.
- **SCS2 writes its generated MJCF to `/tmp/scs2-mujoco-*/world.xml`.** This is the single most
  useful artifact in the whole investigation — the actual MuJoCo model Java simulates. Diff against
  it rather than reading the Java that generates it.
- `FeedbackControllerToolbox`'s `q_*`/`qd_*` are zero in RL mode — not the controller's joint state.

---

## 7. Retracted claims

Things previously written down here or in memory that turned out to be **wrong**:

1. *"The `last_action` loop has gain > 1, so the observation is off-distribution."* No —
   `‖∂a/∂a_prev‖ = 0.92`, the map contracts. The action drift is driven by the robot state moving,
   not by action feedback.
2. *"`LIMIT_RESIDUALS_FROM_PEAK_TORQUES` is what keeps Java stable."* It never binds while standing.
3. *"The residual gap is PhysX↔MuJoCo fidelity; the fix is training-side."* Dead — the policy works
   in Java's MuJoCo.
4. *"`--policy standing` balances, so the harness is proven correct."* It holds ~4.5 s then goes
   over; over 30 s the pelvis ends up below the floor. That conclusion rested on a short run.
5. *"SCS2 emits every primitive URDF `<collision>`."* It emits `AlexSimulationCollisionModel`'s 7.
6. *"The `0.26 × 0.14 × 0.055` foot box belongs to SCS2's impulse engine, not the MuJoCo path."*
   It is exactly what the MuJoCo path uses.

---

## 8. Still open

- **Yaw-rate command does not turn.** `cmd[2] = 0.5 rad/s` with `stand = 0` for 15 s produces
  +0.1° of yaw, while `vx`/`vy` track well. `ObservationDefinitions.base_velocity` puts
  `desiredTurningVelocity` in slot 2 of `base_velocity_plus_standing`, which is where we put it, so
  the wiring looks right. Not yet investigated.
- **Our sole sites are at the ankle, not the sole.** `ALEX_EXTRA_SITES` maps `left_sole`/`right_sole`
  to the `*_FOOT` body origin with no offset, so they sit `ANKLE_HEIGHT = 0.072 m` too high
  (root − sole_site = 0.818 where Java reports 0.890). This is estimator territory, not the policy
  harness, and the estimator is hardware-validated — so either the InEKF contact update absorbs the
  offset elsewhere or there is a 7.2 cm contact-point error behind a passing suite. **Unresolved and
  worth checking independently of the policy work.**
- **Armature is a known, deliberate deviation from Java** (+8.7% inertia on hip/knee, §5c). Harmless
  for the policy, but if a future comparison needs bit-parity with Java's dynamics, drop it.
- Deferred, low priority now that it works: `frictionloss` (neither Java nor we have any), Java's
  82-box ground vs our plane, 2 kHz physics.
