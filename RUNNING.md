# Running this repo

Setup, entry points, and the gotchas that cost time. Written 2026-07-22.

## Setup

```bash
uv sync                     # creates .venv from pyproject.toml + uv.lock
uv run pytest -q            # the whole suite
```

`uv run` activates the venv for you — there is no `source .venv/bin/activate`
step in any workflow below.

`Failed to import warp: No module named 'warp'` on every command is **expected
and harmless**: `mujoco-mjx` probes for the optional Warp backend and falls back
to XLA. Pipe through `| grep -v warp` if it bothers you.

## Test suites

| command | what it covers |
|---|---|
| `uv run pytest -q` | everything |
| `uv run pytest tests/inEKF -q` | gates G2–G5, the ported Java InEKF suite |
| `uv run pytest tests/jointKF -q` | gates G6–G8, the ported Java joint-KF suite |
| `uv run pytest tests/model -q` | the MJX model seam |
| `uv run pytest tests/replay -q` | **Java parity against a hardware log** (below) |

## Java parity against a hardware log

This is the acceptance test that answers "does the Python port agree with the
Java estimator that actually flew". It reads a real SCS2 log, recomputes what the
Java filter published, and diffs.

```bash
uv run pytest tests/replay -q                       # uses the default log
ALEX_PARITY_LOG=/opt/ihmc/LogData/incoming/<dir> uv run pytest tests/replay -q
```

Default log: `/opt/ihmc/LogData/incoming/20260717_160126_Alex001UnifiedControlProcess`
— the 2026-07-17 Alex001 walking run (16:01–16:12, 630 s), chosen because both
filters were live and fully instrumented (`jointKFNumberOfIMUs=8`,
`NumberOfFilteredJoints=9`, `StateDimension=42`).

**Requirements.** The suite *skips* (never fails) without them:

1. The log directory, readable.
2. The `ihmc-log` skill's decoder at `~/.claude/skills/ihmc-log/ihmclog.py`, or
   `$IHMCLOG` pointing at it. The binary-format decoder is deliberately **not**
   vendored — see the header of `src/invariant_estimation/replay/logsource.py`.
3. `zstandard`, which `uv sync` installs as a dev dependency.

**Caching.** The first run decodes from `robotData.bsz` (9.3 GB compressed,
131 GB uncompressed) and memorizes to `<log_dir>/.parity-cache/*.npz`, falling
back to `$TMPDIR` when the log store is read-only. Later runs are ~1 s.

### Gotchas specific to log parity

- **The estimator does not consume `raw_q_*`.** IHMC's `SensorProcessing` runs a
  chain and publishes every stage: `raw_q_X` → `filt_q_X_sp0` → `stiff_q_X_sp1`.
  The filter sees the **last** stage. `logsource.joint_channel` resolves by
  highest `_spN` for exactly this reason; feeding `raw_*` injects a one-stage
  filtering difference that looks like a filter bug and is not one.
- **The robot model comes from inside the log.** `model.sdf` (plain URDF despite
  the extension) ships in every log directory and is by construction the
  description that ran. `model/urdf2mjcf.py` converts it. Do **not** substitute
  the vendored `~/.ihmc/resources/Alex/.../alex_v1_full_body_mjx.xml` — it is a
  different build *and* currently fails to compile (zero-eigenvalue inertias).
- **Variable indices shift between builds.** Always resolve YoVariables by name
  from that log's own handshake, never by index.

## The fused estimator (G9)

`pipeline/main_estimator.py` runs the joint KF and the InEKF back to back as one
constant-XLA-graph `lax.scan` body (`fused_step`). Two entry points:

```python
from invariant_estimation.pipeline import main_estimator as me

# Synthetic / bring-up: build on any MjxModel (see tests/pipeline).
fused = me.build_fused_estimator(model, imu_sites=..., pairs=..., foot_sites=...,
                                 base_imu=0, base_body_site="pelvis_body")

# Real Alex, from the log's model.sdf (topology + frames baked in):
from invariant_estimation.model.urdf2mjcf import convert_log_model
from invariant_estimation.config import load_config
jk = load_config()["joint_kf"]
spec  = convert_log_model(LOG, rotor_inertia=jk["rotor_inertia"],
                          rotor_inertia_default=jk["rotor_inertia_default"],
                          extra_sites=me.ALEX_EXTRA_SITES)   # adds body + sole sites
fused = me.build_alex_fused_estimator(spec)

carry = me.init_fused_carry(fused, q0=...)                   # device-committed carry
carry, outputs = me.run_fused(fused, carry, sensors_over_time)   # lax.scan
```

Test gates:

```bash
uv run pytest tests/pipeline -q              # 9 synthetic scenarios + I7 jaxpr-constancy
uv run pytest tests/replay/test_fused_real_model.py -q   # real Alex + R_mount parity (skips w/o log)
```

**Two landmines are surfaced as arguments**, defaulting to their flight/current
values: `imu_bias_process_var=0.0` (flight; the config's `1e-4` is a test-locked
unit value that makes the fused bias ~200× too noisy) and `contact_meas_var=0.0`
(the current port; set to the flight `1e-4` floor to add the InEKF contact
measurement-noise the port otherwise lacks — affects velocity/position, not
roll/pitch).

**Frames — the one place a G9 bug hides.** Three distinct frames: the base IMU
site (gyro/accel source + joint-KF anchor), the body frame `B` = `base_body_site`
(the pelvis *root* body, what the InEKF's `R` and `invariantRootAngularVelocityBody`
mean), and `R_mount = ᴮR_S` (auto-computed; a +90° yaw on Alex). Verified against
the Java InEKF to 1e-18. **Caveat for a full trajectory replay:** the real InEKF
consumes a *Mahony-prefiltered* pelvis gyro, not the raw `gyroscope_pelvis_imu`
(see `PORT_NOTES.md` "G9 — real model").

## Watching an RL policy in a standalone MuJoCo sim (`run_policy.py`)

`run_policy.py` (repo root) runs an IHMC pre-trained ONNX policy **directly** (via
`onnxruntime`, no JAX/port) inside a plain MuJoCo loop with ground-truth observations.
It is a sim-to-sim demo, **independent of the estimator** (that work is untouched).

```bash
uv run python run_policy.py                       # viewer, default = the STANDING demo
uv run python run_policy.py --headless --ticks 200 # no window; prints tilt / z / |action|
uv run python run_policy.py --policy baseline     # the walking_baseline policy
uv run python run_policy.py --policy baseline --wasd  # WASD window, no controller, no mjpython
```

> **macOS windowing — the two viewers have opposite launcher rules.**
> - **Default** viewer = MuJoCo's Simulate GUI (`launch_passive`): **must** use `mjpython`
>   (`mjpython run_policy.py --policy baseline`). Under plain `python` it can't own the main thread.
> - **`--wasd`** viewer owns its own GLFW window: **must** use plain `python`
>   (`uv run python run_policy.py --policy baseline --wasd`), **never** `mjpython`. `mjpython` runs the
>   script on a secondary thread, so GLFW window creation throws `libc++abi ... NSException`. The
>   script now detects this and prints the fix instead of crashing. `--wasd` is the one that gives you
>   WASD (see below), so this is the command you want.
>
> If `mjpython` dies at startup with `Library not loaded: @executable_path/../lib/libpython3.12.dylib`,
> the uv-managed interpreter's shared lib isn't on any path dyld searches. One-time fix (persists for
> this venv; re-run if you recreate `.venv`):
> ```bash
> LIBDIR=$(.venv/bin/python -c 'import sys,os; print(os.path.join(sys.base_prefix,"lib"))')
> ln -sf "$LIBDIR/libpython3.12.dylib" .venv/lib/libpython3.12.dylib
> ```
> Equivalent per-invocation form: `DYLD_FALLBACK_LIBRARY_PATH="$LIBDIR" mjpython run_policy.py …`.

### Driving it

**Gamepad (preferred).** An Xbox-style pad on `/dev/input/js0`, read straight off the legacy
joystick device — no extra dependency, no thread, drained once per control tick.

| control | effect |
|---|---|
| left stick | `vx` forward/back, `vy` strafe (absolute, scaled to ±0.9 / ±0.5 m/s) |
| right stick X | yaw rate, up to ±1.5 rad/s (≈±86°/s) |
| `RT` / `LT` | raise / lower commanded base height at 0.15 m/s, clamped to the policy's trained band (0.83–0.93 for the walking policies, from `AlexCommandsCfg.base_height`) |
| `A` | toggle the standing flag |
| `B` | stop |
| `START` | height back to the policy default |

Axis numbering is Linux `xpad` (8 axes, 11 buttons, triggers resting at −1.0) and the sign
conventions were checked by hand against the attached pad (2026-07-26). Deadzone is 0.15 because
these sticks rest up to 0.07 off centre.

`uv run python run_policy.py --probe-gamepad` prints the raw axes next to the command they produce
— use it after swapping controllers, since the numbering is per-driver, not universal.

**The policy has its own command deadband** — it stands still below `vx` ≈ 0.25 and `yaw` ≈ 0.5,
then tracks at ratio ≈1.0 above `vx` 0.45 / `yaw` 0.75. So the sticks are *not* mapped linearly:
`_stick_to_command` sends the first bit of travel past the hardware deadzone straight to
`WALK_MIN_*` (0.30 / 0.28 / 0.60), so any real deflection moves the robot instead of silently
commanding a velocity the policy ignores. Trained ranges are `vx` ±0.9, `vy` ±0.5, `yaw` ±1.5.

**Keypad (fallback).** `8`/`2` = ±vx, `4`/`6` = ±vy, `7`/`9` = turn, `5` = stop, `+`/`-` = height,
`0` = standing flag.

> **Letters cannot be viewer keys.** MuJoCo's viewer reserves every letter A–Z (plus `,` `/` `;`
> `'` `\` `` ` ``) for render-flag toggles — the shortcut column of `mjVISSTRING`/`mjRNDSTRING` —
> and it fires its own toggle *in addition* to calling `key_callback`. A WASD mapping therefore
> steers *and* flips wireframe / auto-connect / shadows / static-body. `RESERVED_KEYS` in
> `run_policy.py` is built from those tables at import so a future letter binding fails loudly.

**WASD, in a window (`--wasd`) — no controller, works on the Mac.** The letter restriction above is
a property of the *passive* Simulate GUI, not of MuJoCo. `--wasd` opens our own GLFW window and
renders into it directly, so raw key **press *and* release** reach us with none of Simulate's
render-toggle bindings — hold-to-move, like a game.

```bash
uv run python run_policy.py --policy baseline --wasd
```

| key | effect |
|---|---|
| `W` / `S` | forward / back (`vx = ±0.6`, held) |
| `A` / `D` | strafe left / right (`vy = ±0.4`; `y` is LEFT) |
| `Q` / `E` | turn left / right (`yaw = ±0.9`) |
| `Space` / `Shift` | raise / lower base height (continuous, via `nudge_height`) |
| `R` | height back to the policy default |
| `X` | stop |
| `Esc` | quit |
| mouse | left-drag orbit, right-drag pan, scroll zoom; camera tracks the pelvis |

Held magnitudes sit above the policy's walk deadband (same reason as `WALK_MIN_*`), so a tap moves
the robot rather than commanding the ignored first ~40%. This path does **not** use the passive
viewer, so the Simulate render-flag panel/sliders are unavailable — use the default viewer (no flag)
when you want those. On macOS GLFW must own the main thread, which it does here.

**Terminal (always available).** Single letters mirroring the old WASD (`w`, `s`, `a`, `d`, `q`,
`e`, `x`, `t`, `+`, `-`; `www` = three presses), plus `h 0.85` to set the height target and
`v 0.4 0 0` to set `vx vy yaw` outright.

Any nonzero velocity command clears the standing flag automatically — the policy's
`base_velocity_plus_standing[3]` gates walking, so commanding `vx` while it is set does nothing.

Discrete height changes ease to the new target over 0.5 s (re-seeding `RLHeightManager`'s cubic, as
`goHome()` does); the analog triggers move the command directly via `nudge_height` instead, since
re-seeding the ramp every tick would pin it at t=0 and freeze it. Verified: 0.89 → 0.99 raises the
pelvis 0.896 → 0.943, 0.89 → 0.79 lowers it to 0.820, upright with 8 contacts throughout. The
pelvis tracks roughly half the commanded change — that is the policy's behaviour, not the harness.

**Status: working.** All three policies stand indefinitely and the walking policies walk.

```
--policy baseline   30 s: root z 0.895, tilt 1.1 deg, |action| flat 1.86, ncon 8
--policy standing   30 s: root z 0.743, tilt 1.5 deg, |action| flat 2.92
--policy forearms   30 s: root z 0.905, tilt 0.3 deg, |action| flat 1.91
walking (baseline): vx=0.3 -> 0.25 m/s, vx=0.6 -> 0.60 m/s, vy=0.3 -> 0.26 m/s lateral,
                    15 s each, upright, tilt < 3 deg
```

The baseline numbers reproduce the Java reference (root 0.8918, tilt < 1.2 deg, |action| ~1.86,
ncon 8). A yaw-rate command does not turn yet — see `EXPERIMENTS.md` §8.

**The bug was a frame error in the gyro observation.**
`mj_objectVelocity(m, d, mjOBJ_BODY, bid, v6, 1)` resolves the velocity in the body's **inertial**
frame, not its body frame. Alex's pelvis `body_iquat` is ~180° about (1,0,1)/sqrt(2), so
`base_ang_vel` reached the policy with its **x and z axes swapped and y negated** — the equivalent of
mounting the IMU rotated. `mjOBJ_XBODY` uses the body frame and matches Java's
`RLEstimates.root_AngularVelocity` to 1e-4. If you ever read a body twist out of MuJoCo for a
control or estimation signal, use `mjOBJ_XBODY`; `projected_gravity` was always right because it
goes through `d.xmat`, a different code path.

`experiments/java_parity.py {dynamics,obs,openloop}` holds the three parity checks that found this
— rigid-body model vs Java's compiled MJCF, observation vector vs Java's, and an open-loop setpoint
replay. Reach for them first if a policy starts misbehaving.

`TERRAIN.md` is the plan for uneven-terrain training + vmapped envs in MJX, and the go/no-go for
training in pure MuJoCo at all. Its two feasibility claims are backed by
`experiments/mjx_terrain_probe.py`, which asserts them — run that first.

**`EXPERIMENTS.md` is the full investigation log** — everything measured, everything ruled out, the
Java reference numbers, and the retracted wrong conclusions. Read it before re-testing any
hypothesis about this harness.

### Getting the Java numbers out (the reference dump)

`alex/src/test/java/us/ihmc/alex/rlController/AlexMujocoObsDumpTest.java` — a diagnostic, not a
real test. Runs the RL controller standing on SCS2's MuJoCo engine and writes
`/tmp/alex_java_obs_dump.csv` (per 50 Hz tick: root pose/twist, gravity vector, commands, and
per-joint q/qd/home/last_action/residual/qdes) plus `/tmp/alex_java_yovariables.txt`.

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
cd ~/workspaces/robot-stuff/alex
xvfb-run -a ../gradlew :alex-test:test --offline --tests '*AlexMujocoObsDumpTest*'
```

Gotchas, all of which cost time:
- The test source set is its own project: **`:alex-test:test`**, not `:alex:test` (that reports
  `NO-SOURCE`). `../gradlew` from inside `alex/`, since it is an included build.
- `AlexStateEstimatorParameters` defaults to `JOINT_KF` but only builds the IMU pairs eagerly on
  `RobotTarget.REAL_ROBOT`, so in SCS the pre-filter constructor throws
  *"Base IMU is null, check the kinematic tree."* Call
  `setJointLevelEstimatorType(ALPHA_COMPLEMENTARY)` (empty pair list ⇒ pass-through) or
  `JOINT_KF` explicitly to trigger the lazy build.
- xvfb is required even headless.
- Resolve YoVariables by **full** name. `q_LEFT_HIP_X` exists on both the simulated robot and the
  controller-core feedback toolbox, and `LEFT_HIP_X_q_prev` exists once per instantiated RL model
  (`baseline_walking`, `mout_walking`, …). `RLEstimates`/`RLDesireds`/`RLData` live under the
  **controller** thread (`...HumanoidHighLevelControllerManager.RLControllerState.*`), not the
  estimator thread.
- **SCS2 writes its generated MJCF to `/tmp/scs2-mujoco-*/world.xml`.** That file is the single
  best artifact for model comparison — it is the actual MuJoCo model Java simulates.
- The sensor path is a pass-through in sim: `raw_q*` == `filt_q*` == the simulated robot's q/qd,
  so the policy sees ground truth, same as we do.

### Ground truth: SCS2's MuJoCo backend

The reference implementation is `us.ihmc.scs2.simulation.mujoco` (jar `scs2-mujoco-simulation`,
sources in the Gradle cache), driven by `AlexRLSimulation.startHeadlessMujocoSimulation` and
`AlexRDXTeleoperationUI` (`USE_MUJOCO = true`). What it does, and where we stand:

| Thing | SCS2 MuJoCo | `run_policy.py` |
|---|---|---|
| Body | `AlexV2Version.CYCLOID_FOREARMS` — no hand adapters, 49 links, 90.539488 kg | matches (hands stripped, see below) |
| Collision geoms | every **primitive** URDF `<collision>` → 32 geoms; mesh collisions skipped | matches (`add_collision_geoms`) |
| Self-collision | robot `contype=1 conaffinity=2`, terrain `2/1` → robot geoms test only against terrain | matches |
| `<option>` / contact | `MujocoSimulationParameters` defaults: Newton, implicitfast, iterations 25, noslip 5, impratio 1, pyramidal, friction `1 0.05 0.01`, solref `0.02 1`, solimp `0.9 0.99 0.0007 0.5 2`, condim 4 | matches |
| Collision set, actual | only **7** geoms, from `AlexSimulationCollisionModel` (pelvis/torso/head/2 gripper capsules + 2 foot boxes) — the URDF `<collision>` tags are NOT what SCS2 uses | we port all 32 URDF primitives; harmless on flat ground but not faithful |
| Foot box | **0.26 × 0.14 × 0.055** at `(0.045, 0, −0.05)` in the ankle-roll frame (`newBoxWithSTP`) | 0.22 × 0.10 × 0.02 at `(0.05, 0, −0.06)` — *smaller* in every dimension |
| Ground | 82 tiled 25×25×0.5 boxes (`FlatGroundEnvironment`) | one infinite plane |
| Physics rate | 0.0005 s (2 kHz) | 0.005 s (200 Hz) — matches the IsaacLab training env (`SIM_DT`), not SCS2 |
| Joint armature | 0 (global default); kd comes from the low-level PD, MJCF `damping` is just the URDF's 0.05 | rotor armature from `urdf2mjcf`; `damping = kd` (a faithful emulation — tested) |
| Actuation | explicit torque into `qfrc_applied` at the sim rate, `τ = clamp(kp·(q_d−q) + kd·(0−q̇), ±τmax)` | MuJoCo `position` actuator + joint damping; tracks Java's setpoints better than our own torque-PD port did |
| Commanded root height | `RLHeightManager` ramps 0.9397 → **0.8902** over ~0.4 s | constant 0.90 / 0.75 |

The foot box and the root-height command are the two genuinely unreconciled numbers. Neither fixes
the fall on its own; both are worth correcting anyway since the foot is 29% narrower in y than the
robot Java balances on.

**Model variant.** `alex_with_imus.urdf` is the `FULL_ROBOT_ABILITY_HANDS` assembly (141 links),
but the policies were trained on `CYCLOID_FOREARMS` and `AlexFullWalkingModelDefinition` throws
if the robot version lacks cycloid forearms. `cycloid_forearm_urdf()` drops the two
`*_ABILITY_HAND_ADAPTER` fixed joints and their 92 descendants, which reproduces the Java cycloid
model exactly — same 49 links, same mass and full inertia tensor on every one. Set
`STRIP_ABILITY_HANDS = False` to go back to the hands body (adds 0.973 kg at both wrists).

**Foot collision box** is `0.22 × 0.10 × 0.02` at `(0.05, 0, −0.06)` in the `*_FOOT` frame,
straight from the URDF — i.e. MuJoCo half-extents `0.11 0.05 0.01`. Note this is *not* the
`0.26 × 0.14 × 0.055` box in `AlexSimulationCollisionModel`: that one feeds SCS2's own impulse
engine and the RDX selection model, not the MuJoCo path.

**How it's wired (policy-agnostic).** `load_policy(name)` reads
`rl_models/<...>/policy_cfg.yaml` (obs list, joint order, per-joint kp/kd/home/effort,
action scale); `build_sim_model(policy)` builds a free-base MJCF (estimator MJCF + floor +
SCS2 physics/contact/collision set + per-joint position servos, policy gains for its joints,
baseline gains for the rest); `build_obs` emits each obs term in the order the policy's
`observations` list declares; `Loop` runs it at 50 Hz control / 200 Hz physics. Verified facts:
obs are UNSCALED; `target = home + scale*action`; joint order is the yaml (IsaacLab
breadth-first) order; base_height is a genuinely sensitive input.
Registry: `POLICIES = {standing, baseline, forearms}`. Collision geoms are in viewer group 3
(hidden by default — press `3` to see them over the visual meshes).

**Everything it reads is vendored — a bare clone runs.** No sibling repos, nothing under `~`:

| `assets/…` | size | copied verbatim from |
|---|---|---|
| `alex_with_imus.urdf` | 113 KB | `~/Documents/alex_with_imus.urdf` |
| `rl_models/` (3 policies, `.onnx` + `policy_cfg.yaml`) | 1.5 MB | `alex/src/main/resources/rl_models/` |
| `alex_virtual_description/` (29 visual meshes) | 16.5 MB | `ihmc-alex-sdk/alex-models/alex_virtual_description/` |

Only the meshes this URDF references are vendored (16.5 of the source tree's 43 MB). The 7
ability-hand meshes are deliberately absent: `cycloid_forearm_urdf` deletes those links before the
model is built, and `_add_visual_meshes` already skips any mesh file that is missing.

Each has an env override, for running against a live working copy without editing the source —
`ALEX_URDF` (the same variable the test suite uses), `ALEX_RL_MODELS`, `ALEX_MESHDIR`:

```bash
ALEX_RL_MODELS=~/workspaces/robot-stuff/alex/src/main/resources/rl_models \
  uv run python run_policy.py --policy baseline      # e.g. against a fresh retrain
```

`onnxruntime` is a declared dependency (`pyproject.toml`), so `uv sync` is all the setup there is.

The one still-machine-local file in the repo is `sim_scaffold.py`'s `PCFG`, which points into
`persona_rl`. That scaffold is superseded by `run_policy.py`; its `URDF` now uses `assets/` but the
file as a whole does not run on a bare clone.

## The estimator IN THE LOOP with a policy (`run_estimator.py`)

`run_policy.py` runs the policy on ground truth. **`run_estimator.py` runs the same sim with the
fused estimator in the loop**: simulated IMUs and encoders go in, and the policy's `base_ang_vel`
and `projected_gravity` come out of the filter — the arrangement the real robot runs.

```bash
uv run python run_estimator.py --policy baseline --headless --ticks 1500 --vx 0.6   # 30 s walk
uv run python run_estimator.py --policy baseline                                    # viewer
uv run python run_estimator.py --policy baseline --headless --imu-noise             # noisy IMUs
uv run python run_estimator.py --policy baseline --headless --source truth          # A/B control
uv run python run_estimator.py ... --out run.npz                                    # per-tick log
```

Every run prints an error table against the sim's own state (tilt as the policy sees it,
attitude, gyro, velocity, position drift, joint state) over the whole run and over its last half.

| flag | what it changes |
|---|---|
| `--source ...` | which obs terms come from the estimate: `base_ang_vel`, `projected_gravity`, `joints` (routes the 9 filtered joints through the joint KF), or `truth` for none — the estimator still runs and is still scored, which is the A/B control |
| `--imu-noise` | constant per-IMU gyro bias + white noise on gyros/accel/encoders (`--noise-seed`) |
| `--contact-fk measured\|pinned` | whether the InEKF contact FK uses the measured ankle angles (default) or pins them at `qpos0`, as the library default still does — worth ~2x on attitude error, see below |
| `--stance-chol` / `--swing-chol` | the Σ_C factor for a trusted / airborne foot. The InEKF has **no contact mask**; contact condition rides entirely in Σ_C, so a swing foot needs a large factor or the filter keeps believing it is planted |
| `--contact-meas-var` | flight's `1e-4` contact measurement-noise floor (port default 0) |

**Measured, 30 s at vx = 0.6 (2026-07-26, `experiments/sim_runs/`).** It walks 19–20 m on its
own estimate, and closing the loop costs essentially nothing — estimate-driven and truth-driven
score the same, so the filter is not being destabilised by its own feedback:

| tail-RMS | estimate-driven | truth-driven (A/B) | + IMU noise |
|---|---|---|---|
| tilt error (policy) | 0.81° | 1.33°* | 1.42°* |
| base gyro | 0.003 rad/s | 0.002 | 0.004 |
| base position drift | 2.20 m in 19.4 m | 2.42 m* | 2.85 m* |

\* the A/B and noise columns predate `--contact-fk measured`; rerun them for a like-for-like table.

**Speed (CPU-only jaxlib, measured per tick, not inferred).** A control tick costs **~35 ms**
against its 20 ms real-time budget while walking (~21 ms standing), so the viewer runs at roughly
**0.6x speed** and a 30 s headless run takes ~1 min. Building the estimator and compiling the step
costs ~55 s up front; `make_estimated_loop` compiles eagerly, so that is all paid before the first
tick rather than as an 11 s freeze during it. `--est-every 4` runs the estimator once per control
tick (50 Hz) instead of per physics step — about 1.6x faster, but tilt error degrades 0.81° → 2.0°,
so it is a viewing convenience, not a setting to measure with.

**How it is wired** (`src/invariant_estimation/sim/`): `sensors.py` adds real MuJoCo
`gyro`/`accelerometer` sensors on the 8 estimator IMU sites (site frame = the estimator's
measurement frame; a MuJoCo accelerometer reports SPECIFIC FORCE, which is what the InEKF wants),
reads encoders and foot normal force, and runs the Schmitt/dwell contact trust.
`estimator_loop.py` owns the jitted step and advances it over the 4 physics substeps of each
control tick in ONE scan call, so the estimate the policy reads is current rather than a control
period stale. Gates: `uv run pytest tests/sim -q` (24 tests, ~3 min).

## Exploring a log by hand

The `ihmc-log` skill's CLI is the tool for this; it needs no JVM and no SCS2.

```bash
S=~/.claude/skills/ihmc-log/ihmclog.py
L=/opt/ihmc/LogData/incoming/20260717_160126_Alex001UnifiedControlProcess

python3 $S list $L                          # always first: prints the byte-alignment sanity block
python3 $S vars $L --grep '^jointKF_'       # find variables (regex, ~0.1 s, handshake only)
python3 $S extract $L --vars A,B --stride 25 --start 200 --end 210 --stats -o out.csv
python3 $S plot $L --vars A,B --stride 50 -o out.png
```

`--stride` is in **ticks** (dt = 1 ms, so `--stride 25` = 40 Hz). Never decode a
big log linearly — always stride, or narrow with `--start/--end`.

## Documentation

```bash
uv run docs           # live-reload Sphinx server
uv run docs-build     # build to docs/_build
```

## Repo map

```
config/filter_cfg.yaml          every tunable; [test-locked] values are asserted
src/invariant_estimation/
  config.py                     YAML loader (YAML 1.1: write 1.0e+9, never 1.0e9)
  robot.py                      RobotModel Protocol — the kinematics/inertia seam
  model/urdf2mjcf.py            log's model.sdf -> MJCF
  model/mjx_model.py            the MJX adapter implementing RobotModel
  inEKF/                        the invariant filter (G2–G5, complete)
  jointKF/                      the joint-space pre-filter (G6–G8)
  pipeline/main_estimator.py    the fused joint-KF→InEKF step (G9); ALEX_* topology
  replay/logsource.py           hardware-log reader for the parity harness
tests/pipeline/                 G9 synthetic scenarios + I7 jaxpr-constancy
tests/replay/                   Java parity — the acceptance test (incl. fused real-model)
```

Authoritative design docs: `CLAUDE.md` (spec, invariants I1–I10, gates G1–G10),
`TEST_SUITE_MAP.md` (per-test scenarios/tolerances), `PORT_NOTES.md` (deviations),
`DESIGN_DECISIONS.md` (choices whose symptoms look like bugs).
