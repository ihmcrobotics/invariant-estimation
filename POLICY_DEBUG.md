# POLICY_DEBUG.md — getting `walking_baseline` to walk in our MuJoCo sim

Handoff / resume doc for the `run_policy.py` sim-to-sim debugging (2026-07-23).

> **READ THIS FIRST — CURRENT STATE (2026-07-23, Session 2 = authoritative).**
> Everything in the "Session 1" sections below (Goal → "run_policy.py current state")
> is the ORIGINAL investigation and is **partly superseded** — in particular the
> "foot collision geometry" suspect was DISPROVEN. The authoritative, up-to-date
> conclusions live in the **`## SESSION 2 ...`** sections at the bottom. TL;DR:
>
> - **`run_policy.py` is refactored and policy-agnostic** (reads any policy's
>   `policy_cfg.yaml`; builds obs term-by-term; drives its joints). CLI:
>   `uv run python run_policy.py [--policy standing|baseline|forearms] [--headless]`.
>   See `RUNNING.md`.
> - **The harness is PROVEN CORRECT.** The `standing` policy (`--policy standing`)
>   **balances** (~4 s, tilt <3°). That validates joint order (yaml/BFS), gains,
>   armature, foot box, obs frames/layout, action mapping, dt — no gross bug.
> - **The foot box == SCS2's foot, bit-for-bit** — not a guess, not the bug.
> - **The walking baseline face-plants forward in ~1 s.** Every tuning lever failed
>   (base_height, contact, stiffer/softer PD, dt, interpolation, walk commands). This
>   is a genuine **PhysX→MuJoCo sim-to-sim gap** for the marginal walking gait, NOT an
>   obs/action/param bug. Fix = training-side (domain randomization / fine-tune against
>   MuJoCo, or reproduce IHMC's full deployment loop). Details in Session 2 §"#3 walking".

---

## Goal (Session 1)

Run IHMC's pre-trained **`walking_baseline`** RL policy **directly** (via
`onnxruntime`, no JAX port) inside a **standalone MuJoCo sim** (`run_policy.py` at
the repo root), with **ground-truth observations**, keyboard command now / Xbox
later. The invariant-estimator integration is **deferred** — this is purely "watch
the policy walk in sim." (Estimator work is the actual project and is untouched.)

**Status (Session 1 snapshot; see TL;DR above for current):** the loop runs
end-to-end, finite, renders as the full-mesh robot — but the robot falls over.
Being debugged systematically against a known-good reference (below).

---

## Files

- **`run_policy.py`** (repo root) — the sim + control loop. **(Session 2: refactored to be
  policy-agnostic — see "run_policy.py current state" below and `RUNNING.md`.)** Entry:
  `uv run python run_policy.py [--policy standing|baseline|forearms] [--headless]`.
- **Policy** (the one to use — FULLBODY, 98-in/29-joint):
  `/home/llibshutz/workspaces/robot-stuff/alex/src/main/resources/rl_models/2026-07-10_baseline/{policy.onnx,policy_cfg.yaml}`
  - NAMING TRAP: the Java `getModelName()=="walking_baseline"` is the *nub* variant
    `2026-04-03_..._002_nubs` (inputSize **80** = 23 joints) — dimensionally can't
    drive our 29-joint model. `2026-07-10_baseline` (98/29) is the fullbody match.
- **Robot model:** `/home/llibshutz/Documents/alex_with_imus.urdf` (v2 fullbody, 29
  joints), converted via `main_estimator.alex_spec_from_urdf`. Visual meshes:
  `/home/llibshutz/workspaces/robot-stuff/ihmc-alex-sdk/alex-models/alex_virtual_description/` (v1 meshes).
- **Policy is a bare MLP**: `98→256→256→256→29`, ELU on first three, linear head,
  `x@W.T+b`. `empirical_normalization=False` → feed **raw** obs. float32.

---

## THE REFERENCE — IHMC's working Java MuJoCo sim (the oracle)

The policy **works** in IHMC's own MuJoCo sim, so the bug is in **our** sim setup.
Debug method = diff ours against theirs (same discipline as the Java-vs-Python
estimator port).

- **Launcher:** `~/workspaces/robot-stuff/alex/.../simulation/AlexRLSimulation.java`
  → SCS2 MuJoCo engine.
- **SCS2 MuJoCo engine source** (how it builds the MJCF) — in the gradle cache:
  `~/.gradle/caches/modules-2/files-2.1/us.ihmc/scs2-mujoco-simulation/*/…-sources.jar`
  → `MujocoMultiBodyRobotFactory.java` (option/contact/geom/init), `MujocoTools.java`
  (`appendGeom`), `MujocoTerrainFactory.java` (ground). Extract with `unzip`.
- **Java RL controller obs/action code** (the obs ORACLE):
  `~/workspaces/robot-stuff/ihmc-internal-software/ihmc-closed-source-control/src/main/java/us/ihmc/closedSourceControl/rlController/`
  → `modelDefinition/ObservationDefinitions.java` (each obs term), `RLJointOutputProcessor.java`
  (action→target), `RLEstimates.java` (frames).

### SCS2's MuJoCo config (from `MujocoMultiBodyRobotFactory`)

```
<option integrator="implicitfast" solver="Newton" iterations="25"
        noslip_iterations="5" impratio="1" cone="pyramidal"/>
geom defaults: condim="4" friction="1 0.05 0.01" solref="0.02 1"
               solimp="0.9 0.99 0.0007 0.5 2"
robot geoms: contype=1 conaffinity=2 ;  terrain: contype=2 conaffinity=1
  (robot collides with TERRAIN ONLY — no self-collision)
```
- PD torque applied via **`qfrc_applied`** (NOT MuJoCo actuators).
- Initial joint state **seeded to the half-squat home** (else "immediately collapses").
- Uses the robot's **actual collision shapes** (full set: box/sphere/cyl/capsule),
  **not** a hand-guessed foot box.
- Comments note `solref` default 0.02 is "too soft for the controller; 0.005 the
  sweet spot" (but the param default is 0.02, which is what Alex runs).

### Obs conventions (from `ObservationDefinitions.java`, all VERIFIED = ours)

98 = `[base_ang_vel(3), projected_gravity(3), base_velocity_plus_standing(4),
base_height(1), joint_pos_rel(29), joint_vel_rel(29), last_action(29)]`, in the
29-joint yaml order, no scale/clip:
- `base_ang_vel` = pelvis **body-frame** angular velocity (`getRootAngularVelocity`,
  frame-after-root-joint). Ours: `mj_objectVelocity(local)[:3]`. ✓
- `projected_gravity` = world `(0,0,-1)` rotated into pelvis frame = `Rᵀ·(0,0,-1)`,
  **unit** (not 9.81). ✓
- `base_velocity_plus_standing` = `[vx, vy, turn, stand]` (command, base frame). ✓
- `base_height` = **desired** root height (a command, ~0.88), not measured. ✓
- `joint_pos_rel` = `q − homePosition`; `joint_vel_rel` = `qd`. ✓
- `last_action` = last raw policy output. ✓
- Action: `target = home + 0.3·action`, clamped so `|target−q| ≤ maxEffort/kp`
  (torque-equivalent to our force-clamped actuator). 50 Hz control / 200 Hz physics.

---

## What's been RULED OUT (matches the reference, not the bug)

1. **Physics config** — applied SCS2's `implicitfast`/`Newton`/`condim=4`/`solref`/
   `solimp`/`friction` to `run_policy.build_sim_model`. Did **not** fix it.
2. **Actuator mechanism** — tested `qfrc_applied` PD (`actuators=False` path) ≡ the
   position-servo+damping path. Same failure.
3. **All 7 obs terms + their frames/signs/order** — diffed against the Java oracle,
   all match.
4. **Action processing** — `home + 0.3·action`, torque-limited. Matches.
5. **Pose / balance-ability** — CoM is `x=0.003`, **inside** the foot support
   `[−0.088, 0.132]`. The pose is statically balanceable.
6. **Obs normalization** — none (`empirical_normalization=False`); raw obs correct.

## Failure signature (identical every attempt)

- Policy ON: `|act|` starts ~1.7 (reasonable), tilts **~15° in 0.5 s**, then diverges
  (`|act|→100s`, eventually NaN). Consistent.
- Policy OFF, holding home **rigidly**: tilts **37° in 0.5 s** — *despite CoM over
  the feet*. A statue with CoM over its feet should NOT fall.
- `ncon=0` at the "rest" pose earlier (feet started 1 mm above floor → free-fall);
  user observed **"feet hang from the plane by their tops, reset flies up."**

## STRONGEST REMAINING SUSPECT → the next step  ⛔ SUPERSEDED (foot hypothesis DISPROVEN)

> **This Session-1 conclusion was WRONG.** Kept for the record. Session 2 proved:
> SCS2's MuJoCo foot geom == our box bit-for-bit; the feet give real support; a rigid
> robot stands on them; and the `standing` policy balances on them. Foot geometry is
> NOT the bug. See Session 2 §"foot hypothesis KILLED".

**(historical)** Foot collision geometry — the box `0.11 0.05 0.01 @ 0.05 0 -0.06` was
suspected to be a bad guess. It is in fact exactly what SCS2 emits from the URDF.

## Lower-priority candidates (Session 1)  — resolved in Session 2

- **Joint sign / order**: RESOLVED — the yaml/BFS joint order is correct (the standing
  policy balances with it; the alex.py per-limb `JOINT_NAMES_FULLBODY` order is worse).
- **Physics dt**: RESOLVED — IsaacLab trains at dt=0.005/decim=4 (NOT 0.002); ours
  already matches. dt=0.002 was tried and does not help.

---

## run_policy.py current state  (UPDATED Session 2 — the script was refactored)

`run_policy.py` is now **policy-agnostic**. Key functions: `load_policy(name)` (reads a
`policy_cfg.yaml`), `build_sim_model(policy, with_visuals=True)` (free base + floor + SCS2
physics + foot boxes + per-joint position servos using the policy's kp/kd, others held at
home by the baseline PD), `make_maps`, `build_obs(m,d,policy,maps,cmd,last_action)` (emits
each obs term in the order the policy's `observations` list declares), `Loop`. Registry
`POLICIES = {standing, baseline, forearms}`. Run per `RUNNING.md`. Obs/action/gains/order
are all verified correct (the standing policy balances). The remaining issue is the
PhysX→MuJoCo dynamics gap that the walking gait cannot survive (Session 2 §"#3 walking").

---

## SESSION 2 (2026-07-23 cont.) — foot hypothesis KILLED; config proven faithful

Picking up from "port SCS2 feet." Two subagents pulled the SCS2 collision source and
the actual IsaacLab training env config. Result: **the feet were never the bug, and
`run_policy`'s physics config already matches training.** The policy diverges anyway.

### Foot hypothesis — disproven three ways
1. **SCS2's MuJoCo foot geom == our box, bit-for-bit.** `scs2-mujoco-simulation`'s
   `MujocoMultiBodyRobotFactory` reads the robot's URDF `<collision>` and emits it as
   a MuJoCo geom. Alex's foot collision (in `alex_v1.lowerBody.urdf`) is
   `box 0.22 0.10 0.02` at `(0.05,0,-0.06)` in the FOOT frame → MuJoCo half-extents
   `0.11 0.05 0.01` at `0.05 0 -0.06`, `condim=4`. That is EXACTLY
   `build_sim_model.foot()`. SCS2 does NOT add the 4-corner contact points as geoms
   (those are controller-only). So there is nothing to "port" — we already have it.
2. **The feet provide real support.** At the home pose `ncon=8` (both boxes, 4 corners
   each), contact z ≈ 0, boxes perfectly flat (`geom z-axis=[0,0,1]`), CoM x=0.003 well
   inside support `x∈[-0.088,0.132], y∈[-0.17,0.17]`.
3. **A rigid robot STANDS on them.** With a correct stiff position servo (kp×20, both
   `gainprm[0]` AND `biasprm[1]` scaled — see trap below) and feet pre-touching, the
   robot holds tilt ≈ 3° indefinitely. Feet are fine.
   - TRAP: scaling only `actuator_gainprm[0]` on a MuJoCo `position` actuator turns it
     into a huge constant torque `(scale-1)*kp*home` (force = kp·ctrl − kp·qpos, encoded
     as gainprm[0]=kp, biasprm[1]=−kp). Scale both, or you "prove" a false collapse.

### Config is FAITHFUL to training (all verified against the IsaacLab env + `alex.py`)
Training env = `Isaac-WalkingUneven-Alex-v0`, robot cfg `alex.py`
(`.../ihmc_lab/robots/alex.py`, found in `/opt/ihmc/LogData/@Recycle/.../.persona_rl_code`).
- **kp/kd:** IsaacLab `DelayedPDActuatorCfg` stiffness/damping == `policy_cfg.yaml`
  kp/kd EXACTLY (HIP_X 53.67/8.035, HIP_Y 72.4/10.86, KNEE 72.4/10.86, ANKLE_Y 43/6.45,
  ANKLE_X 20/4.0, …). The yaml kp are the TRAINING gains, NOT "clamp gains".
- **armature:** IsaacLab `ARMATURE_85=0.062, ARMATURE_115=0.167, ARMATURE_68=0.020,
  ARMATURE_S=0.005` (ARMATURE_SCALE=1.0) == our model's `dof_armature` (which the
  estimator writes from the rotor table). Match.
- **dt/decimation:** IsaacLab `sim.dt=0.005, decimation=4` == `run_policy` DT/DECIMATION.
  (The `alex_v1_full_body_mjx.xml` MJCF with dt=0.002 + kp=112/150 + armature=0.004 is an
  OLDER variant — do NOT chase it; it is not this policy's training config.)
- **obs:** no scale on ANY of the 7 terms (confirmed in IsaacLab cfg, the yaml export,
  AND the Java `ObservationDefinitions.java`). Frames confirmed against `RLEstimates.java`:
  base_ang_vel = pelvis body-frame ω; projected_gravity = Rᵀ(0,0,-1) in pelvis frame.
  `base_velocity_plus_standing = [vx,vy,yaw, standflag]`, standflag=0 for a zero-vel walk.
- **action:** `target = default + 0.3*action`, `use_default_offset=True`. Match.
  (`ALEX_JOINT_SCALE=2.0` in the recycle snapshot is a sibling; trust the yaml's 0.3.)
- **default pose:** IsaacLab init_state joint_pos == yaml homePosition (HIP_Y −0.35,
  KNEE 0.7, ANKLE_Y −0.35, SHOULDER_Y 0.15, ELBOW −0.5, SHOULDER_X ±0.05 L/R). Match.
- **masses/inertias:** our MuJoCo model == URDF exactly (total 91.51 kg; per-link mass
  and pelvis inertia match). Model dynamics are faithful.
- **ONNX:** bare MLP 98→256→256→256→29, Elu×3, no normalization/clip/tanh.

### What was swept and FAILED to stop the divergence
Every one of these diverges (tilt → 130–175° within ~1–2 s, |a| blows up):
kp ×{1,3,6,10,20,40}; armature {estimator 0.06–0.167, training-MJCF 0.004}; integrator
{implicitfast/Newton, Euler}; dt {0.005, 0.002}; init {feet-touching clean start,
0.3–0.8 s settle}; base-frame rotation of (ang_vel,proj_grav) {I, Rz90, Rz180, Rz-90};
stand flag {0,1}; action {+0.3a, −0.3a, +a, 0.3a}; velocity-term signs {±ang_vel,
±joint_vel}; qfrc-PD vs position-servo. The FAITHFUL config (nominal yaml kp/kd, our
armature) diverges from a clean upright t=0 start: tilt 0→147° in 2 s.

Note: nominal PD genuinely CANNOT hold the half-squat passively (static sag ≈ τ/kp,
knee needs ~100 Nm / kp 72 ≈ 1.4 rad) — but that is true in IsaacLab too; there the
POLICY holds it. So "statue falls" is EXPECTED, not the bug. The bug is that our policy
loop is (marginally) unstable: from the trace it oscillates 17→21→6→10→27→13→18→46°
with |a|~12 — it's fighting, not diverging monotonically at first — i.e. a small
phase/gain error, not a gross obs error.

### REMAINING SUSPECTS (all obs/action/params now excluded)
The gap must be a concrete difference between OUR MuJoCo and the training/SCS2 setup
that we have NOT reproduced:
1. **Action delay.** IsaacLab uses `DelayedPDActuatorCfg(min_delay,max_delay)` (0–N
   physics-step random delay). We apply instantly. (Absence usually *helps* stability,
   so weak candidate — but it changes loop phase.)
2. **Self-collision.** IsaacLab `enabled_self_collisions=True`; our sim has only 2 foot
   collision geoms (no limb collision), so none. Would need the full limb collision set.
3. **PD evaluation cadence / implicit PhysX drive vs MuJoCo servo.** IsaacLab computes
   the PD once per 50 Hz control step and holds it; MuJoCo re-evaluates per 200 Hz
   substep. (Again usually *more* stable for us.)
4. **The premise itself** — is the policy CONFIRMED to balance in SCS2 from raw
   ground-truth obs + the kp=53 PD alone, with nothing else in the loop (no estimator,
   no whole-body, no gravity-comp, no different low-level gains)? If SCS2's loop has
   more than "obs→MLP→target→PD", the sim-only reproduction is under-specified.

### DECISIVE NEXT STEP (needs a ground-truth trace, not more sweeps)
Sweeping our sim is exhausted. Get ONE working reference rollout and diff it tick-by-tick:
either (a) run IHMC's SCS2 `AlexRLSimulation` and log obs[98] + action[29] + qpos each
control tick, then feed our sim the SAME initial qpos and compare obs/action term-by-term
at t=0,1,2,3; or (b) get an IsaacLab rollout of this exact policy. The first point where
our obs or action diverges from the reference is the bug. Without a reference, we're
guessing in a space where every static parameter already matches.

---

## SESSION 2 (cont.) — SCS2 loop traced; it is NOT bare obs→MLP→PD

Traced the full SCS2 RL control loop in the IHMC Java source (`rlController/` +
`RLSimulationFactory`/`SCS2OutputWriter`/`AvatarLowLevelOutputProcessor`). The loop has
FOUR things our replay lacks:
1. **Obs from the STATE ESTIMATOR, not perfect sim state.** `RLSimulationFactory`
   `setUsePerfectSensors(false)`, estimator mode NORMAL; `RLEstimates` reads the
   controller-side `fullRobotModel` (estimator-driven). Root ω, projected gravity, joint
   q/qd are all estimator outputs. BUT: IsaacLab TRAINED on ground truth + Unoise
   (ang_vel ±0.3, grav ±0.05, jpos ±0.01, jvel ±0.3). So our perfect-state obs is the
   CENTER of the trained distribution — strictly EASIER than SCS2's lagging estimator
   obs. This cannot be why our (cleaner) replay fails. (Paradox flagged below.)
2. **Desired-position interpolation** over each 20 ms control window (JointControlBlender
   at the estimator rate) — we apply a stepped, held 50 Hz target.
3. **Setpoint clamp** vs current (estimated) q: `|q_des − q| ≤ maxEffort/kp` per tick.
4. **`last_action` = the CLAMPED/limited residual** (not the raw MLP output), init to
   `(q−home)` at entry.
Also: 50 Hz control, inference every tick, EFFORT mode with RL kp/kd, `desiredVelocity=0`
(no feed-forward), NaN-hold latch, unstable-velocity damping reducer. NO whole-body/ID/QP.

**Replicated #2+#3+#4 in the replay (`scs2loop.py`): NONE stabilize it.** Interp, clamp,
limited-last_action, and all-three each still diverge to 110–170°.

### Obs fully re-verified (including the one ambiguous term)
- `mj_objectVelocity(PELVIS, local)[:3]` IS the pelvis body-frame angular velocity
  (verified: it returns [angular; linear], and the wx→z/wy→−y/wz→x permutation seen when
  poking `qvel[3:6]` is just MuJoCo's free-joint inertial-frame DOF convention — the obs
  path via mj_objectVelocity is correct). PELVIS_LINK is the free base, xmat=I at spawn.
- projected_gravity, joint_pos_rel, joint_vel_rel, order, layout, no-scale — all confirmed.

### The core paradox (this is the real clue)
The policy tolerates SCS2's **noisy, lagging estimator** obs, so our **clean ground-truth**
obs is strictly within (indeed at the center of) the trained input distribution. Yet the
clean replay diverges. Since obs/action/gains/armature/mass/dt are all proven faithful,
the remaining possibilities are narrow:
- (a) The premise is wrong/incomplete — the policy may NOT actually balance from this
  loop alone in SCS2 (unconfirmed by the user). Worth verifying directly.
- (b) A PHYSICS-RESPONSE difference between our MuJoCo and SCS2's MuJoCo that we have NOT
  reproduced: SCS2 uses the robot's FULL collision set (all limb collision boxes, not just
  2 feet) with self-collision, and a finer sim dt (0.002 vs our 0.005) with sim-rate
  qfrc PD on interpolated targets. Our replay has only 2 foot geoms and 200 Hz servo PD.

### MJX-model ground-truth attempt — BLOCKED
Tried running the policy in the actual IsaacLab-lineage MJX model
(`alex_v1_full_body_mjx.xml`): it FAILS to load in plain MuJoCo — pelvis_link `fullinertia`
is non-PD ('inertia must have positive eigenvalues'), not fixed by `balanceinertia`. It is
also the wrong vintage (27 joints, no grippers; kp 112/150 not the policy's 53/72). Not a
clean oracle without inertia surgery + a joint-mapping shim.

### DECISIVE NEXT STEP (unchanged, now the only path left)
A real tick-by-tick reference trace. Cheap replays are exhausted. Recommended:
instrument SCS2's headless `AlexRLSimulation` MuJoCo entry point to dump, per control
tick: the 98-obs vector, the 29 raw action, and full `qpos`. Then feed our replay the SAME
initial qpos and diff obs/action term-by-term at t=0,1,2,3. The first divergence is the
bug. Secondary cheap probe worth doing first: add the FULL limb collision set + self-
collision + dt=0.002 to the replay (tests possibility (b) without a Java build).

---

## SESSION 2 (cont.) — BREAKTHROUGH: the harness is CORRECT (a standing policy balances)

User confirmed the ground truth is **IsaacLab** (not SCS2), and that ALL the policies in
`rl_models/` work there. So a failure of ALL of them in our replay would be systematic.
Tested the **`20251219_standing18`** policy (a pure Isaac-Standing task, 76-obs/23-joint,
deeper squat home HIP_Y −0.77 / KNEE 1.42) through the same MuJoCo harness:

**IT BALANCES.** With all 29 joints actuated (wrists/grippers held at home) and
`base_height=0.75`, it holds tilt < 3°, z steady at 0.737, |a|≈2–3, for ~4 s before
slowly drifting out (a foot loses contact, ncon→1, then it tips). This **validates the
entire replay**: joint order (the yaml/BFS order IS correct — same order this policy uses),
gains, foot box, obs frames, obs layout, action mapping (`home+0.3a`), dt/decim — all
correct. There is NO gross/systematic bug. (Joint-order experiment separately: yaml-BFS vs
the alex.py per-limb `JOINT_NAMES_FULLBODY` order — BFS is right; FULLBODY diverges worse.)

### `base_height` is a sensitive obs
standing18 holds at bh≈0.75 but falls at 0.70 or 0.788 (a ~4 cm window). For the walking
baseline, a fine bh sweep 0.80–0.92 gives NO stable window — it survives only 0.4–0.6 s at
every height (slightly longer at higher bh). So bh is not the walking-baseline fix.

### Why the walking baseline still fails fast (and standing mostly holds)
- standing18 (Isaac-Standing) has a roomy stability basin → survives ~4 s in our sim.
- The 2026-07-10 baseline + the forearms policies (Isaac-WalkingUneven) fail in ~0.5 s.
- Even standing18 eventually drifts out via a FOOT losing contact → the residual gap is
  **contact / dynamics fidelity between IsaacLab (PhysX) and our MuJoCo**, which a
  marginally-stable WALKING policy cannot survive but a roomy STANDING policy mostly can.
- The walking obs also carries the extra `base_velocity_plus_standing(4)` term the standing
  obs lacks; feeding it [0,0,0,stand] (stand∈{0,1}) does not rescue it.

### Net conclusion (this session)
The replay is fundamentally correct — proven by a policy that balances in it. The remaining
problem is NOT obs/action/gains/order/model/foot. It is (a) a residual PhysX↔MuJoCo
contact/dynamics fidelity gap that hits marginal walking policies hardest, and/or (b) the
walking policies needing to actually walk / the SCS2-style smoothing (estimator lag +
target interpolation) they were robustified against. 

### Concrete next directions (in priority order)
1. **Wire `standing18` as a working demo** in run_policy (it stands ~4 s) — a real,
   watchable baseline and a regression anchor. Then attack the 4 s drift (contact tuning).
2. **Contact fidelity**: tune foot contact (solref/solimp/friction, `condim`, dt=0.002,
   maybe a softer/‘sweet-spot’ solref 0.005 per SCS2 comment) to extend standing18's hold;
   whatever extends it should help the walking policies too.
3. **Walking**: give the baseline a real forward command AND a short in-air/upright
   initialization (or SCS2-style target interpolation + a 1-tick action delay) so a dynamic
   gait can start; static standstill may be out-of-distribution for a WalkingUneven policy.
4. If needed, the definitive check remains an IsaacLab rollout of `standing18` to diff the
   4 s drift tick-by-tick — but the harness itself is now proven, so this is lower urgency.

---

## SESSION 2 (cont.) — #1 DONE (standing demo); #3 walking = sim-to-sim gap, not tunable today

`run_policy.py` refactored to be **policy-agnostic**: reads any policy's `policy_cfg.yaml`
(obs list, joint order, per-joint kp/kd/home/effort, action scale), builds the obs vector
term-by-term, drives its joints (others held at home by baseline PD). CLI:
`uv run python run_policy.py [--policy standing|baseline|forearms] [--headless]`.

**#1 DONE — `--policy standing` visibly balances** (~4 s, tilt <3°, z steady 0.737), then
slowly drifts out (a foot loses contact). This is the working-policy-in-MuJoCo demo.

**#3 walking — NOT achievable by tuning.** Traced the baseline: from a cold standstill it
immediately extends the knees (KNEE action −2.5→−5 ⇒ target ~straight), the pelvis pitches
and drifts FORWARD, it lifts the right foot to step, and **face-plants forward in ~1.1 s**
(walks ~0.6 m first). Every lever fails to sustain it:
- base_height 0.80–0.92: survives 0.4–0.6 s (standing mode), ~1.1 s (walk mode, stand=0).
- contact (solref 0.005/0.01, friction 2.0, condim, dt 0.002): no change to the 4 s standing
  drift, no help to walking.
- stiffer leg tracking (kp ×1.5–3): WORSE (0.2–0.4 s) — amplifies the aggressive extension.
- SCS2 target interpolation over the 20 ms window: no help (0.9 s).
- walk commands vx 0.3/0.5: face-plants forward, no sustained gait.

**Diagnosis:** the WalkingUneven gait is fore-aft unstable in our MuJoCo — a genuine
PhysX→MuJoCo transfer gap. The standing policy (roomy basin, deep squat) survives it for 4 s;
the walking gait (shallow squat, aggressive) does not survive ~1 s. This is orthogonal to
obs/action/gains/order/model/foot (all proven correct by the standing policy balancing).

**Path to walking (beyond tuning, for a later session):**
1. Close the sim gap by RUNNING the policy in an MJX/MuJoCo-backed IsaacLab or fine-tuning /
   domain-randomizing the policy against MuJoCo dynamics (the standard sim-to-sim fix). IHMC's
   deployment likely relies on training-time DR covering the target sim + the full SCS2 stack.
2. OR reproduce IHMC's exact deployment loop (estimator obs + target interpolation + the
   robot's low-level joint controller), which may differ from raw-obs + RL-gain PD in ways
   that matter for the marginal gait — needs the actual SCS2 run to confirm.
3. The harness is proven correct, so a working IsaacLab/MJX rollout to diff against would
   pinpoint the exact dynamics divergence quickly.
