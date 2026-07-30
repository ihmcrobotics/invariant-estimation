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

## GPU vs CPU — which to use for what

`jax[cuda12]` is a dependency, so `uv sync` installs a CUDA-enabled jaxlib.
Confirm with:

```bash
uv run python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
JAX_PLATFORMS=cpu uv run ...                          # force CPU for any command
```

**Use the GPU for training. Use the CPU for the test suite.** Measured on this
machine (RTX 4070 SUPER, 20-core CPU), on the real training step at `B=32`,
`L=128`, no remat, with marginal cost separated from setup:

| | marginal | setup | 10k steps |
|---|---|---|---|
| GPU | **0.188 s/step** | 56.8 s (CUDA init) | **31 min** |
| CPU | 0.440 s/step | 39.2 s | 73 min |

Three gotchas, all of which cost time once:

* **A naive total-time comparison inverts the answer.** At 60 steps the GPU
  looks *slower* (68.2 s vs 65.8 s) purely because CUDA init makes its setup
  17 s longer. Always difference two step counts to isolate the marginal cost.
* **Do not run the full suite on GPU.** It took >20 min against 17.5 on CPU and
  was abandoned — a suite of small unit tests is the worst case for a GPU
  (kernel-launch overhead per test). Worse, XLA **preallocates 75% of VRAM**
  regardless of use, so a running suite holds ~10 GB and blocks anything else on
  the device. The targeted subset that matters for training,
  `tests/inEKF tests/contactnet tests/pipeline`, runs on GPU in 5 min (312 tests).

  **It also fails on GPU, and the failures read as real bugs.**
  `tests/sim/test_collect.py` reports 3: `test_chunking_is_exact` asserting
  `np.array_equal` false on two `(1500, 13, 13)` `sigma_q` arrays that print
  identically, and two `JaxRuntimeError: INTERNAL: Autotuning failed ...
  RESOURCE_EXHAUSTED: Out of memory while trying to allocate 182.25MiB`. **All 19
  pass under `JAX_PLATFORMS=cpu`.** The chunked-vs-one-pass check is a
  bit-exactness assertion and GPU autotuning picks its reduction order per
  compilation, so that equality is not a GPU-portable property; the OOMs are the
  75%-preallocation above colliding with anything else on the device. Since a
  bare `uv run pytest` now selects CUDA, **always pin the suite to CPU**:

  ```bash
  JAX_PLATFORMS=cpu uv run pytest -q tests/     # 733 passed in 18:54 (2026-07-28)
  ```
* **Consumer NVIDIA runs FP64 at 1/64 of FP32.** A 4070 SUPER is ~0.5 TFLOPS
  FP64, comparable to this CPU, and I8 mandates float64 at the filter boundary.
  The GPU wins anyway because the workload is latency- and memory-bound rather
  than FLOP-bound — which is also why `B=32` (2.34x) beats the 1.5x previously
  recorded for the batch-1 estimator loop. Do not assume the ratio transfers to
  a differently-shaped workload; measure it.

**Logging a long run: use `python -u`.** Python block-buffers stdout when
redirected, so a `nohup ... > run.log` shows *nothing* for minutes and leaves an
empty file if the run dies. `PYTHONUNBUFFERED=1` / `-u` fixes it.

## Test suites

| command | what it covers |
|---|---|
| `uv run pytest -q` | everything (**run on CPU** — see above) |
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
uv run python run_estimator.py --policy baseline --ticks 1500 --vx 0.6 \
       --video walk.mp4                                                             # 30 s video
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
| `--contactnet CKPT.npz` | attach a trained ContactNet as the contact-noise provider (off = the analytic filter). The run prints `ContactNet: ATTACHED ckpt=… norm=…` at startup so a log is never ambiguous about which arm it is |
| `--contactnet-norm N.npz` | the normalization constants, which **must** be the ones the checkpoint was trained under: run 4 → `data/dr/norm_constants.npz`, run 2 → `data/norm_constants.npz`. A mismatch shifts the network's input distribution and **nothing raises** — the run just quietly measures something else |
| `--video walk.mp4` | record the run offscreen to H.264 (implies `--headless`, `--video-fps` / `--video-size` tune it) |
| `--ghost [mode]` | draw a translucent robot at the estimated state: `full` (default) or `attitude`. Viewer only |
| `--ghost-offset M` | displace the ghost sideways for side-by-side viewing instead of overlaid |
| `--realtime` | run the estimator on its own thread. Viewer only; the estimate goes slightly stale |
| `--max-backlog-ticks` | how far the estimator may fall behind before the sim thread waits (default 2). Samples are never dropped |
| `--wasd` | use the standalone WASD window instead of the passive viewer |

### ContactNet in the closed loop

> **Do not run runs 1–4 through this after the socket move.** They were trained
> on `contact_meas_chol`, where ~1e-4 is a sensible FK measurement std; the same
> output in `contact_chol` is the *stance* value, so every anchor — swing feet
> included — is asserted world-static. That is `freeze_contact_chol`, and in the
> closed loop it is a fall. `run_estimator.py` prints a warning if the checkpoint
> filename looks like one of them. To score an old checkpoint fairly, use
> `experiments/replay_eval.py --socket meas`. The table below was measured
> **before** the move and is kept as the run-4 record.

```bash
uv run python run_estimator.py --policy baseline --headless --ticks 1500 --vx 0.6 \
    --imu-noise --contact-fk measured \
    --contactnet artifacts/contactnet_run5.npz --contactnet-norm data/dr/norm_constants.npz
```

The provider needs `span_ticks(cfg)` = **400** ticks of history (0.4 s at the 1 kHz filter rate,
i.e. 20 control ticks) before it emits anything; until then it falls back to `sensors.contact_chol`,
the analytic heuristic. An attached run is therefore **bit-identical** to an unattached one for the
first 19 control ticks and diverges at tick 19 — that identity-then-divergence is the cheapest proof
the network is actually reaching the filter rather than being silently dropped.

**Measured, 30 s at vx = 0.6, `--imu-noise`, `--contact-fk measured`, 3 noise seeds
(2026-07-29).** The headline is vertical drift, `est_z - true_z`, which the 3D `p_err` norm hides:

| mean of 3 seeds | no ContactNet | run 4 (`data/dr`) | run 2 (`data/`) |
|---|---|---|---|
| final signed dz [m] | **−3.183** | **−0.399** | −0.769 |
| dz RMS (last half) [m] | 2.432 | 0.303 | 0.583 |
| sink rate, last 20 s [m/s] | −0.107 | −0.0135 | −0.026 |
| tilt error, tail RMS [deg] | 1.201 | 0.252 | 0.658 |
| base gyro, tail RMS [rad/s] | 0.0080 | 0.0056 | 0.0062 |
| 3D position drift, final [m] | 3.260 | 0.511 | 2.220 |

**Run 5 — the first PROCESS-socket network** (single seed 0, same command as
above with `--contactnet artifacts/contactnet_run5.npz`). Vertical drift is
**2.1–2.2x better than run 4**; the cost is rotational and it is yaw, which this
filter cannot observe. Full analysis in `PORT_NOTES.md`, "Run 5".

| seed 0 | no ContactNet | run 4 (rec.) | **run 5** |
|---|---|---|---|
| final signed dz [m] | −3.194 | −0.399 | **−0.180** |
| dz RMS (last half) [m] | 2.435 | 0.303 | **0.143** |
| sink rate, last 20 s [m/s] | −0.1079 | −0.0135 | **−0.0064** |
| base velocity error RMS [m/s] | 0.1343 | — | **0.0304** |
| 3D position drift, final [m] | 3.279 | 0.511 | 0.501 |
| tilt error, tail RMS [deg] | 1.210 | **0.252** | 0.370 |
| yaw component, tail [deg] | 1.168 | — | 2.225 |

Run 5 is **not calibrated** (`nis_over_dof` 0.022, closed-loop NIS tail 0.20
against 1.0) — the known `l2_velocity` gap. Good for drift, will not pass G10's
consistency bands.

The robot itself walks fine in every arm (true base height stays 0.88–0.91 m, 19–20 m travelled) —
the sinking is entirely in the estimate. **A parallel-runs gotcha:** `run_policy.cycloid_forearm_urdf`
writes its hands-free URDF to a fixed `tempfile.gettempdir()` path, so N concurrent runs race on
one file and some die with an XML `ParseError`. Give each background run its own `TMPDIR=…`.

### Recording a video

**Seeing the estimate: `--ghost`.** A translucent second robot drawn at the
ESTIMATED state, so the filter's error is visible rather than tabulated. It works
in the viewer *and* (since 2026-07-30) in `--video`:

```bash
# side-by-side, current best checkpoint (N=4, so --toe-heel is REQUIRED)
uv run python run_estimator.py --policy baseline --ticks 1500 --vx 0.6 \
    --imu-noise --contact-fk measured --toe-heel \
    --contactnet artifacts/contactnet_run6_w128.npz \
    --contactnet-norm data/dr4/norm_constants.npz \
    --ghost --ghost-offset 0.9 --video artifacts/video/out.mp4 --video-size 1280x720
```

* `--contactnet-norm` defaults to `data/dr/` (run 4's). **A run-6 checkpoint needs
  `data/dr4/`** — wrong constants shift the input distribution and nothing raises.
* `--ghost-offset` is a lateral displacement in **world Y**, applied after the pose
  is set from the estimate (`ghost.py`, `q[1] += offset`). `0` overlays the two.
  Because it is world-frame rather than body-relative, on a turning walk the ghost
  does not stay beside the robot.
* `--ghost attitude` pins the ghost's position at truth so only ORIENTATION error
  shows. This is the view that makes the yaw cost legible; `full` hides it behind
  the (now small) position error.
* `--ghost` with a bare `--headless` is rejected: there is no scene to draw into.

Three reference clips are in `artifacts/video/` (gitignored, local only):
`1_baseline_N2_no_contactnet.mp4` (ghost sinks 3.19 m),
`2_N4_contactnet_w128.mp4` (holds height, final dz +0.07 m), and
`3_N4_w128_attitude_yaw.mp4` (the 4.74° yaw cost). Same seed, so 1 and 2 are
frame-comparable.

`--video` renders the run offscreen with a chase camera on the pelvis and pipes raw frames into
`ffmpeg` — no `imageio`/`mediapy` dependency and nothing buffered in memory. It needs `ffmpeg` on
`PATH` and an offscreen GL context; the script sets `MUJOCO_GL=egl` for you (it has to happen
*before* `import mujoco`, hence the argv peek at the top of `run_estimator.py`). Override with
`MUJOCO_GL=glfw` if EGL is unavailable. Frame rate is capped by the 50 Hz control loop, so the
default `--video-fps 50` is real time; `--video-size 1920x1080` is the largest the offscreen
buffer is declared for (`<visual><global offwidth/offheight>` in `_add_scene_look`). Recording
costs roughly 15 ms/tick on top of the control tick (~17 ms on CPU — see the speed table below).

The scene look — gradient skybox, blue checkered floor (0.5 m tiles, so a stride can be read off
them), black robot, overhead light — lives in `run_policy._add_scene_look` and rides along with
the visual meshes, so `--headless` runs without `--video` compile exactly the dynamics they did
before: it adds no geom, mass or collision, only textures, materials and a light.

**Measured, 30 s at vx = 0.6 (2026-07-26).** It walks 19–20 m on its
own estimate, and closing the loop costs essentially nothing — estimate-driven and truth-driven
score the same, so the filter is not being destabilised by its own feedback:

| tail-RMS | estimate-driven | truth-driven (A/B) | + IMU noise |
|---|---|---|---|
| tilt error (policy) | 0.81° | 1.33°* | 1.42°* |
| base gyro | 0.003 rad/s | 0.002 | 0.004 |
| base position drift | 2.20 m in 19.4 m | 2.42 m* | 2.85 m* |

\* the A/B and noise columns predate `--contact-fk measured`; rerun them for a like-for-like table.

**Speed (measured per tick, not inferred — 2026-07-27).** This used to read "~35 ms per tick, so
the viewer runs at ~0.6x". **That is fixed: the loop now keeps real time on CPU.** Pinning the
ONNX session to one non-spinning thread (`run_policy._ort_session`) was the whole fix. ORT
defaults to one intra-op thread per core *and spins* after each `Run`, so ~9 spinning threads
were fighting XLA's own pool between control ticks. Interleaved A/B, same process, walking, with
an offscreen render per tick:

| ORT session | p50 | p90 | mean | vs real time |
|---|---|---|---|---|
| default (before) | 24.0 / 29.7 ms | 34.7 / 41.5 | 25.9 / 30.1 | 0.83x / 0.67x |
| pinned, 1 thread, no spin | **17.8 / 16.1 ms** | 31.5 / 24.6 | 20.5 / 17.7 | **1.12x / 1.24x** |

Building the estimator and compiling the step costs ~55 s up front; `make_estimated_loop` compiles
eagerly, so that is paid before the first tick rather than as an 11 s freeze during it.
`--est-every 4` runs the estimator once per control tick (50 Hz) instead of per physics step —
about 1.6x faster, but tilt error degrades 0.81° → 2.0°, so it is a viewing convenience, not a
setting to measure with.

**GPU: it wins, which was not the expected answer.** `uv sync --extra gpu` installs a CUDA jaxlib
alongside the CPU one; `jax.devices()[0]` then picks it up with no code change. Every reason to
expect a *loss* still holds (batch size 1, no `vmap`, hundreds of tiny kernels, float64 at 1/64
rate on a consumer card, a blocking host round-trip per tick) — and it is 1.5x faster anyway,
with the error columns identical to three decimals. Measured on an RTX 4070 SUPER, 250 ticks at
vx = 0.6, three interleaved repeats (`experiments/bench_estimator_device.py`):

| device | build | p50 | xRT | tilt tail-RMS | drift |
|---|---|---|---|---|---|
| cpu | 54 s | 13.59–13.84 ms | 1.45–1.47x | 0.856° | 0.353 m |
| gpu | 71 s | **8.93–9.05 ms** | **2.21–2.24x** | 0.856° | 0.353 m |

Switch backends with the env var, never a flag — the backend must be chosen before `jax` is
imported, and `run_estimator.py` imports the whole chain at module scope:

```bash
JAX_PLATFORMS=cpu  uv run python run_estimator.py --policy baseline    # force CPU
JAX_PLATFORMS=cuda uv run pytest tests/sim -q                          # deliberate parity check
uv run python experiments/bench_estimator_device.py --ticks 250        # re-run the A/B
```

The test suite is pinned to CPU by the repo-root `conftest.py`, so installing the extra cannot
silently move MJX kinematics onto the GPU and shift every tolerance in `tests/sim`,
`tests/pipeline` and `tests/replay`. (Checked: `tests/sim` + `tests/pipeline`, 61 tests, pass on
CUDA too — the pin is precaution, not a workaround for a known failure.) Note `uv lock` resolves
all extras, so `uv.lock` carries a large `nvidia-*` block even though a default `uv sync`
downloads none of it.

### Watching the estimate: the ghost, and `--realtime`

```bash
uv run python run_estimator.py --policy baseline --ghost              # translucent robot at the ESTIMATE
uv run python run_estimator.py --policy baseline --ghost attitude     # pinned at true position
uv run python run_estimator.py --policy baseline --ghost --ghost-offset 1.0   # side by side
uv run python run_estimator.py --policy baseline --ghost --wasd       # WASD window instead
uv run python run_estimator.py --policy baseline --realtime           # estimator on its own thread
```

`--ghost` draws a second, translucent robot at the estimated state — physics-free (a second
`MjData`, `mj_kinematics` only, never `mj_step`ped; a test asserts `qpos` is bit-identical with
the ghost on and off). Overlaid by default, so a good estimate hides inside the real robot and
disagreement reads as a separating shadow; the known missing-touchdown-reseed drift shows up as
the ghost sinking through the floor. Costs 0.08 ms/frame.

Cycle **off → full → attitude** with the **keypad `*`** in the passive viewer, `g` typed at the
terminal, or plain `G` in `--wasd`. (Letters cannot be viewer keys — MuJoCo reserves every one of
A–Z for render toggles, hence the keypad.) `full` shows the whole estimated pose and is the honest
view; `attitude` pins the pelvis at the true position so orientation error is visible without the
ghost drifting off screen.

`--realtime` moves the estimator to its own thread. It is **viewer-only and off by default**:
`--headless --realtime` is a hard error, because the policy then reads a slightly stale estimate
and the run is not reproducible. No sensor sample is ever dropped — when the estimator falls more
than `--max-backlog-ticks` behind, the sim thread waits for it instead (back-pressure), which
degrades to the old sub-real-time behaviour rather than silently changing the filter's input.

Honest note on what it buys: **very little, now that the ORT fix landed.** Measured 0.97x headless
and 1.03x with a render per tick — the sim thread no longer has a deficit to hide. The default
`--max-backlog-ticks` is **2**, not the 5 originally planned: staleness tracks the allowed backlog
almost exactly (age ≈ backlog + 1 ticks) and tilt error degrades sharply past ~3 —
1 → +0.099°, 2 → +0.112°, 3 → +0.456°, 5 → +1.598° against the synchronous run.

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

## ContactNet — the learned contact covariance (`contactnet/`)

Spec: `src/invariant_estimation/contactnet/network_plan.md`. Trains an MLP that
emits a contact covariance `Σ_C` by BPTT through the InEKF.

`optax` is the only added dependency (training-time only — the forward pass is
hand-written because it is transliterated into Java per §7).

**Which socket.** There are two contact covariances and they are not the same
thing (`inEKF/filter.py`, "Two contact covariance sockets"; `PORT_NOTES.md`):

| `InEKFInputs` field | enters | question |
|---|---|---|
| `contact_chol` | process, `Q_d` | is this foot world-static? |
| `contact_meas_chol` | measurement, `N` | how well do we know where it is? |

**ContactNet feeds `contact_chol`** — the process socket — since 2026-07-29.
Runs 1–4 fed `contact_meas_chol`; `network_plan.md` §1 still says so and is
superseded. `contact_meas_chol` is now zeros everywhere, which is the shipped
analytic filter bit-for-bit.

The reason, in one line: `N` sits inside the inverted factor of
`K = P Hᵀ (H P Hᵀ + N)⁻¹`, so it can scale a correction but cannot change how a
residual is split between the base and the anchor — and the drift being trained
out is an integrated velocity bias, which lives entirely on that split. Measured
in `experiments/process_socket_ablation.py`; full argument in `PORT_NOTES.md`,
"ContactNet moves to the process socket".

Three things this changes that will bite if skipped:

* `sigma_0 = 1e-4` is a *measurement*-socket number. On this socket a constant at
  that value is the run-1 configuration — `network.init` zeroes the head, so
  iteration 0 emits a constant `Sigma_C` at every gait phase. **Nothing enforces
  this**; pick the initialization deliberately, see `TODO.md` item 1.
* **`artifacts/p0_dr.npz` is stale.** `measure_p0` runs under different
  conventions now. `artifacts/p0_process_dr.npz` is the re-measured one for
  `data/dr` (1.06% off the stale one, in the position/anchor block); for any
  other dataset run `train_contactnet.py p0 --p0 <path.npz>`.
* `contact_floor` (`config/alex_inekf.yaml`) is now safety-critical: it is the
  only bound on a mis-predicted `Σ_C` pinning a swing foot.

```bash
# Does the process socket even move the thing you are trying to fix?
# Replays recorded rollouts under shifted/retightened anchor noise. No network.
uv run python -m experiments.process_socket_ablation --data data/dr
# Score a checkpoint on either socket (--socket process is the default path now):
uv run python -m experiments.replay_eval artifacts/contactnet_run4.npz --data data/dr
```

### Driving a run — `train_contactnet.py`

From nothing to a trained network is four commands; only the first is slow.

```bash
uv run python -m invariant_estimation.sim.collect --seconds 60 --seeds 0 1 2  # ~50 min, 1.5 GB
uv run python train_contactnet.py cache        # ~25 min: MJX features -> data/cache/ (23 MB each)
uv run python train_contactnet.py norm         # seconds: freezes data/norm_constants.npz
# Measure P0 under the CURRENT conventions first -- p0_dr.npz is stale (TODO.md 2).
JAX_PLATFORMS=cuda uv run python train_contactnet.py p0 --p0 artifacts/p0_process_dr.npz
JAX_PLATFORMS=cuda uv run python -u train_contactnet.py train --steps 10000 \
    --objective l2_velocity --B 32 --no-remat \
    --out artifacts/contactnet_run5.npz --p0 artifacts/p0_process.npz
```

**Use `-u`.** The first launch of run 1 buffered and showed nothing for 3.5
minutes despite 70% GPU load. Chaining, the pass-through process socket, and the
measured `warm_in_s`/`episode_s` are all defaults — the command above is the
current recommended run. `--no-chained` and `--freeze-contact-chol` restore
run-1 behaviour for ablations.

`cache` and `norm` are idempotent and are re-run automatically by `train` when
their artifacts are missing, so once the data exists **the real run is the last
command alone** (≈75 min for 10 000 steps at B=32 on 20 CPU cores). `--force`
rebuilds them. Two more modes:

```bash
uv run python train_contactnet.py check-init   # the §4 init-parity properties, on real windows
uv run python train_contactnet.py measure-b    # peak RSS vs B, remat on/off (subprocess per point)
```

**Pass structure** (`contactnet/dataset.py`). Only pass 1 needs MJX:

| pass | what | cost |
|---|---|---|
| `cache` | `features.make_contact_channels` + contact FK over each rollout → `(T, N_c, F)` and `(T, N_c, 3)` | ~2 min/rollout, 23 MB each |
| `norm` | pools every rollout's post-warm-up region → `normalize.NormConstants` | seconds |
| `prepare` | normalize + global boxcar, per-rollout, in NumPy | ~10 s/rollout, ~210 MB resident each |

Segments are then pure indexing. A segment's windows are **bit-identical** to
`features.window` over the whole rollout, sliced — the boxcar is done once
globally so only a gather remains (`test_segment_windows_are_bit_identical_to_features_window`).

**What the loader guarantees**, each with a mutation-checked test:

* `inputs.contact_chol` is **passed through unchanged** (since 2026-07-28).
  Run 1 froze it at the stance constant and that was the primary cause of its
  collapse — pinning a *swinging* foot as world-static costs 10.2× in body-frame
  velocity error versus not using contacts at all. `--freeze-contact-chol`
  restores the old behaviour for the ablation. The leak argument for freezing
  does not hold: the network's input is the 24 feature channels, and
  `contact_chol` reaches only the filter's *process* model.
* Segment starts avoid the 16 000-tick warm-up **and** the following
  `(H-1)·stride = 392` ticks. 545 772 legal starts over 12 rollouts.
* Batches are drawn from a fresh permutation of the rollouts each step, so a
  batch touches `min(B, n_rollouts)` distinct trajectories. Starts are uniform
  random inside a rollout, never tiled.
* Segments are **chained**, not independently seeded (`dataset.ChainedBatcher`).
  `B` filter chains walk the rollouts in order carrying `(X̂, P)` and the gravity
  reference between steps, so a segment starts at whatever error the filter has
  actually accumulated. See "Run 2 and the chained batcher" below — this is the
  single most important thing to understand before launching a run.
* `state0` is `truth.R/v/p` at the start tick, with contact anchors
  `d = R_true·y(q̂) + p_true` from the FK at the **recorded filter** `q̂` — so the
  first contact residual is exactly zero. The joint KF is *not* reseeded: it ran
  continuously and its outputs are frozen into `inputs`.
* `P0` is **measured**, not chosen: `dataset.measure_p0` runs the InEKF alone over
  3 000 ticks of recorded input under training conventions and takes the converged
  covariance. Measured on `flat/seed0`:
  `diag(R) = [6.9e-5, 6.9e-5, 1.0]` (yaw unobservable, as it must be),
  `diag(v) = 7.6e-4`, `diag(p) = diag(d) = 0.34` (absolute position unobservable;
  only `p − d` is). At that seed the pre-ContactNet filter starts at
  `NIS/dof = 1.46`, i.e. already nearly calibrated.

### Run 2 and the chained batcher — read before launching

Run 1 (10 000 steps, `l2_velocity`) looked healthy by every process metric —
loss down 82×, `applied_frac` 1.000 throughout, gradients finite — and was
**degenerate**. It learned `Σ_C` with a median per-axis std of 0.68 m, four
orders above the `N = J Σ_q Jᵀ` term in the same innovation, suppressing the
contact-update velocity gain **3835×**. The filter had learned to ignore its
feet. Loss curves cannot see this; see the two gates below.

Two causes, both fixed and both now the default:

1. **The frozen process socket** (primary, 10.2×) — above.
2. **Force-teacher seeding** (secondary) — every segment was re-seeded from
   ground truth, so it began at *zero* error and ran 128 ms. Over that horizon
   IMU dead-reckoning beats any contact correction, so the loss-minimising `Σ_C`
   is infinite. `ChainedBatcher` carries the state instead. CoCo-InEKF
   (arXiv 2605.15122 §III-B) reports the same failure for force-teacher seeding.

Three config fields govern chaining, all measured rather than chosen:

| field | default | why |
|---|---|---|
| `warm_in_s` | 1.0 | a fresh chain starts at zero error; the filter's error saturates at ~8.5e-2 m/s and is there by 1 s. Untrained-on. |
| `episode_s` | 43.0 | **the ceiling this dataset allows** — 62 s rollout − 16 s joint-KF warm-up = 45.5 s usable. Setting 100 s would never fire; the rollout-end re-seed trips first. A true 100 s episode needs `--seconds 120`+ at collection. |
| `freeze_contact_chol` | False | run-1 ablation only |

Chains also re-seed on a non-finite carry (a diverged chain would poison every
later step) and their initial episode phase is staggered — seeding them all at
`ticks = warm_in_s` makes them re-seed in a synchronised wave and then march in
lockstep, costing most of the `B` independent samples.

### The two gates — run both, in this order

```bash
uv run python -m experiments.alpha_sweep                       # before training
uv run python -m experiments.check_sigma  artifacts/<run>.npz  # after
uv run python -m experiments.replay_eval  artifacts/<run>.npz  # after — the verdict
```

> **`alpha_sweep` needs `artifacts/p0.npz`, which only `train` creates.** On a
> fresh clone the gate dies with `FileNotFoundError` before it can gate anything:
> it `np.load`s P0 with no fallback, while `train_contactnet.py:286-289` is the
> only code path that measures-and-caches it. Mint P0 first with a throwaway
> 2-step run, which uses train's own code path so the value is identical:
> ```bash
> mkdir -p artifacts
> uv run python train_contactnet.py train --steps 2 --warmup-steps 1 \
>     --p0 artifacts/p0.npz --out artifacts/_p0_probe.npz && rm artifacts/_p0_probe.*
> ```
> `--steps 1` does **not** work: `warmup_steps` clamps to `total_steps` and then
> trips its own `warmup_steps < total_steps` guard.

**`alpha_sweep`** scales `Σ_C` by a global factor and requires an interior
`argmin`. It is theory-doc §7.2.1 Claim 1 aimed at the objective actually in use,
runs in ~30 s, and needs no training. Run-1's config **FAILS** it (monotone to
α=1e4 — loss down 136× purely from disabling contacts); run-2's **PASSES** with
the optimum at `Σ_C ≈ 1 cm`. Run this whenever you change seeding, the horizon,
or the process socket.

**`check_sigma`** reports the per-axis contact gain against initialization. Treat
it as a **diagnostic, not a verdict** — a gain ratio cannot distinguish "switched
off because the objective was degenerate" from "switched off because that
residual direction carries little velocity information". It gave the wrong answer
twice on run 2, in opposite directions.

**`replay_eval`** is the verdict: it runs the filter under the heuristic `σ₀²I`
and under the trained `Σ_C`, identical otherwise, and compares error in what L2
scores *and* what it does not. Run 2, 20 s horizons:

| metric | heuristic | run-2 `Σ_C` | ratio |
|---|---|---|---|
| body-frame velocity RMS | 0.0844 m/s | 0.0254 | 0.301 |
| position RMS | 0.790 m | 0.245 | 0.310 |
| height RMS | 0.780 m | 0.0918 | 0.118 |
| height final | 1.347 m | 0.125 | 0.093 |
| mean tilt | 0.689° | 0.353° | 0.513 |

`nis_over_dof` finishing far below 1 (run 2: 7.5e-3) is **expected** under L2 and
is not a failure — L2 constrains the gain sequence and says nothing about the
covariance. It means the *estimate* is good and the *covariance* is not
calibrated; that gap is what β-NLL exists to close.

### GPU, timing, and what does not help

Measured on an RTX 4070 SUPER (12 GB), B=32, `--no-remat`:

| setting | s/step | 10k steps | 100k steps |
|---|---|---|---|
| `warm_in_s=2, episode_s=20` (run 2 as launched) | 0.598 | 1.66 h | 16.6 h |
| `warm_in_s=1, episode_s=43` (current default) | ~0.36 | ~1.0 h | ~10 h |

Over half of run 2's wall time was re-seed warm-in scans (1118 ms each at
`warm_in_s=2`, 0.305/step). The current defaults cut that to ~0.18/step at
~559 ms.

**Larger `B` does not help on a 12 GB card** — B=32 450 ms, B=64 1333 ms
(2.96×), B=128 2227 ms. Scaling is superlinear, so the GPU is not
under-occupied at B=32 and buying throughput with batch size does not work here.
Re-measure on a bigger card before assuming otherwise.

### The second machine: WSL2 / RTX 4090 / 6 cores (measured 2026-07-28)

A full from-scratch pipeline reproduced run 2's `replay_eval` numbers on a
*different* box. Everything above was measured on the 4070 SUPER / 20-core /
native-Linux machine; this one differs in ways that matter, and not in the
direction you would guess.

| stage | 4070 SUPER, 20c, native | 4090, 6c, WSL2 |
|---|---|---|
| `collect` | 3.2 s/sim-s | **2.63 s/sim-s** (faster) |
| `cache` | ~2 min/rollout | **4.4 min/rollout** (slower) |
| `train` B=32 | 0.36 s/step | **0.632 s/step** marginal (0.68 total) |

**The GPU is not the bottleneck — the host is.** During training the 4090 sits at
**24% utilisation, 73 W of 337 W, P-state P2**, with sustained ~112 MiB/s inbound
host→device traffic. Three consequences:

* **A bigger GPU buys at most ~1.3×.** Util is the fraction of wall time a kernel
  is running, so 24% busy ⇒ deleting *all* GPU time is a `1/0.76` speedup. The
  workload is a `lax.scan` of L=128 **sequentially dependent** filter steps on
  15×15 float64 matrices — many tiny serial kernels, not big GEMMs. Batch size
  widens each kernel but cannot remove the 128 round trips, which is the same
  thing the superlinear `B` scaling above is telling you.
* **`collect` inverts the core count.** It is ~88% `run_fused`, so the GPU absorbs
  it and 6 cores are fine. `cache` and `train` are host-bound and lose badly.
* **Cores scale sublinearly.** 20c → 6c costs only 1.76× on `train`, not 3.3×,
  because part of the host path (Python/JAX dispatch, NumPy segment gathering) is
  single-threaded. Expect ~2×, not more, from a 24-core box. WSL2 vs native is
  confounded with core count in every number here and was not profiled.

**On a 24 GB card `B` is nearly free — the 12 GB result does not transfer.**
`measure-b` on the 4090 (GPU, L=128, ahead-of-time compile):

| B | remat | peak [MB] | compile [s] | step [s] |
|---:|---|---:|---:|---:|
| 8 | off | 2866 | 25.7 | 0.25 |
| 16 | off | 2943 | 26.0 | 0.28 |
| 32 | off | 3052 | 26.1 | 0.30 |
| 64 | off | 3206 | 25.7 | 0.30 |
| 128 | off | 3605 | 25.7 | **0.35** |

**16× the batch for 1.4× the step.** On the 4070 the same sweep was superlinear
(B=32 450 ms → B=64 1333 ms, 2.96×), so "larger `B` does not help" is a 12 GB
statement, not a general one — the instruction above to re-measure on a bigger
card was right. Two caveats before raising it: it buys sample *throughput*, not
sample *diversity* (at 12 rollouts, B=128 is ~10.7 segments per rollout, so the
extra draws increasingly replay trajectories already in the batch — the same
argument that makes more iterations the wrong purchase), and `remat` is still a
net loss (+0.02–0.09 s/step, +250–300 MB compile).

**The host overhead is measured, not inferred.** `measure_one` times a single
compiled `grad_fn` on a **pre-built** batch (`train_contactnet.py:226-228`) — no
chained batcher, no re-seed warm-in, no segment gathering. That step is
**0.30 s** at B=32/no-remat, against the real training loop's **0.632 s/step**.
So **~0.33 s/step, 52% of wall time, is host-side work outside the jitted step**,
decomposing roughly into the documented ~0.18 s/step of re-seed warm-in plus
~0.15 s of batch construction and dispatch. That, not the GPU and not `B`, is the
only thing worth optimising.

**WSL2 gotcha — the startup `CUDA_ERROR_OUT_OF_MEMORY` spam is benign.** XLA tries
to preallocate 75% of VRAM as *one contiguous block* (17.99 GiB) and fails, then
walks down ~10% at a time. Nothing is lost: `bytes_limit` stays at 17.99 GiB and
the allocator grows on demand. Measured on this box with 19 GB free:

| probe | result |
|---|---|
| single 8 GiB block | **fails** |
| 8 × 1 GiB blocks | **succeeds** |
| largest single block (bisected) | **~3.6 GiB** |

So it is a **contiguity limit, not a capacity limit** — an artifact of the
paravirtualised WDDM (`dxgkrnl`) path, not the card. It only bites if one tensor
exceeds ~3.6 GiB; B=32 peaks near 242 MB. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` to silence the spam and make the
grow-on-demand behaviour explicit. Note `nvidia-smi` free-VRAM is *not* the
health check here (see the NVML soname note) — these numbers came from actual
allocation attempts.

### Moving a ContactNet run to another machine

**`data/` and `artifacts/` are both gitignored** (`.gitignore:230,233`), so a
`git pull` on the other box gets you the code and none of the run. Decide per
directory:

| path | size | copy or regenerate |
|---|---|---|
| `data/*.npz` (12 rollouts) | 1.5 GB | **copy** to reproduce a run exactly; regenerating is ~35–40 min but MJX/XLA are not bit-identical across hardware |
| `data/cache/*_feat.npz` | 276 MB | **copy** — cheap, and saves the 25–53 min `cache` stage |
| `data/norm_constants.npz` | 4 KB | copy (or regenerate in seconds) |
| `artifacts/contactnet_run3.npz` | 3 MB | **copy** — this is the deliverable and nothing else reproduces it |
| `artifacts/p0.npz` | 2 KB | either; it now regenerates automatically |

`prepare` needs **both** the raw rollout and its `_feat.npz`: features come from
the cache, but `inputs`/`truth` come from the raw `.npz`. Copying only the cache
is not enough.

**Pulling out of WSL when your SSH endpoint is the Windows host.** The repo lives
on the WSL filesystem but Windows OpenSSH drops you into Windows, not WSL. You do
not need an sshd inside WSL — point rsync at the WSL binary with `--rsync-path`.
Run this **from the target box** (verified working 2026-07-28):

```bash
WINHOST=lucas@<windows-tailscale-ip>
SRC=/home/lucas/Documents/ihmc/invariant-estimation
for d in data artifacts; do                      # BOTH -- see the trap below
  rsync -avh --partial --progress --rsync-path="wsl -d Ubuntu rsync" \
    "$WINHOST:$SRC/$d/" "./$d/"
done
```

* **Name the distro.** `wsl -d Ubuntu`: this box also has a stopped `Ubuntu-22.04`,
  and a bare `wsl` can land in the wrong one.
* **Trap: pull `artifacts/` too.** It is easy to copy only `data/` and lose the
  3.9 MB trained network — the one thing in the whole 1.8 GB that nothing
  regenerates.
* Skip `-z`; `.npz` is already compressed. `--partial` makes a dropped link
  resume instead of restarting 1.8 GB.
* If PowerShell is the default Windows shell and the quoting misbehaves, stream a
  tar instead — no rsync needed on the Windows side:
  ```bash
  ssh "$WINHOST" "wsl -d Ubuntu -- tar -C $SRC -cf - data artifacts" | tar -xf -
  ```

**Pushing straight out of WSL over Tailscale usually fails**, even though the
route works (TCP 22 to the peer is reachable). WSL2 in the default NAT mode is
**not a tailnet node** — its traffic egresses through the Windows host's
Tailscale and arrives with no client identity, so a peer running Tailscale SSH
refuses it regardless of password. Install Tailscale inside WSL, or use key auth.
WSL also starts with no `~/.ssh`; the Windows key at
`/mnt/c/Users/Lucas/.ssh/id_ed25519` cannot be used in place because drvfs
reports every file `0777` and ssh rejects a world-readable key — copy it to
`~/.ssh` and `chmod 600`.

**Verify the transfer.** Generate a manifest on the source and check it on the
target; a truncated `.npz` fails much later and much more confusingly:

```bash
# source
sha256sum artifacts/*.npz artifacts/*.json data/*.npz data/cache/*.npz > transfer-manifest.sha256
# target
sha256sum -c transfer-manifest.sha256
```

**What changes on a native-Linux box:** `XLA_PYTHON_CLIENT_PREALLOCATE=false`
becomes unnecessary (harmless to keep — it costs a little allocation overhead and
buys quieter logs), and neither the 3.6 GiB contiguity ceiling nor the NVML
soname shadowing applies. Expect roughly `collect` ~40 min, `cache` ~25 min,
`train` 10k ≈ 1 h on the 20-core box.

**What does not change:** the pipeline is host-bound everywhere, so do not expect
a bigger GPU or a larger `B` to help — see the util argument above.

**More iterations is probably the wrong purchase.** 100k steps × B=32 = 3.2M
segment draws over ~1 100 independent contact events in the current dataset —
2 900 replays of each. The gap to CoCo-InEKF is ~10⁵ in *sample diversity*
(they regenerate physics every iteration across 1 280 envs), not in iteration
count. Friction randomisation and a wider `vx`/yaw sweep buy more than steps do.

### Sizing `B` — measured, not guessed

`measure-b`, 20 CPU cores, L = 128, ahead-of-time compile so XLA's own peak is a
separate column:

| B | remat | compile Δ [MB] | exec Δ [MB] | process peak [MB] | compile [s] | step [s] |
|---:|---|---:|---:|---:|---:|---:|
| 8 | on | 732 | -13 | 2521 | 24.4 | 0.20 |
| 8 | off | 464 | 24 | 2232 | 16.5 | 0.13 |
| 16 | on | 717 | 35 | 2545 | 24.7 | 0.38 |
| 16 | off | 468 | 101 | 2386 | 17.3 | 0.24 |
| 32 | on | 753 | 135 | 2729 | 24.6 | 0.63 |
| 32 | off | 471 | 242 | 2541 | 16.8 | 0.44 |
| 64 | on | 718 | 305 | 3037 | 24.8 | 1.23 |
| 64 | off | 472 | 533 | 2933 | 16.9 | 0.84 |

Read it as: **`B` is not memory-bound at this `L`.** The forward+backward costs
~8.5 MB per unit of `B` without remat and ~5.7 with; even B = 64 peaks at 2.9 GB
of a 94 GB machine, and the largest single allocation is the XLA *compiler*, not
the gradient. `remat` does what it claims (43% less execution memory at B = 64)
but costs +250 MB of compile and **+47% step time**, so on CPU at L = 128 it is a
net loss. Gradients are identical either way (checked: 4.8e-11 relative).

**Use `B = 32, --no-remat`.** 0.44 s/step, and 32 is ~2.7 segments per rollout —
past that the extra segments are drawn from trajectories already in the batch and
buy less than they cost. Turn `remat` back on when `B·L` grows past ~10 000
tick-segments or on a GPU, where device memory is the binding constraint.

### Collecting the training data (`sim/collect.py`)

Terrain-randomised rollouts → one `.npz` per rollout under `data/` (gitignored):

```bash
uv run python -m invariant_estimation.sim.collect --seconds 60 --seeds 0 1 2   # 4 terrains x 3
uv run python -m invariant_estimation.sim.collect --terrain waves --seeds 0 --seconds 60
uv run python -m invariant_estimation.sim.collect --measure                    # warm-up + timings
```

```python
from invariant_estimation.sim.collect import build_collector, collect_rollout, load_rollout
c = build_collector()                 # ~30 s of MJX trace + XLA compile, ONCE — reuse it
r = collect_rollout("hard_stepping", seed=3, seconds=60.0, collector=c)
r.sensors      # FusedSensors      -> contactnet.features
r.inputs       # InEKFInputs       -> contactnet.rollout.Segment.inputs (incl. joint.sigma_q)
r.truth["v"], r.truth["R"]          # -> the L2 objective
r.meta["warmup_ticks"]              # ticks to DISCARD at the head; nothing is dropped on save
```

The policy runs on ground truth (`run_policy.Loop`); the estimator runs **open loop** over the
recorded stream afterwards, so the dataset does not depend on the filter ContactNet is about to
change. Sensors are sampled every physics tick (1 kHz), not every control tick.

| number | measured |
|---|---|
| cost | **3.2 wall-s per simulated second** on 20 CPU cores: `run_fused` 2.83, sensor read 0.19, `mj_step` + policy 0.12. So a 62 s rollout ≈ 3.5 min, and 12 rollouts ≈ 45 min |
| size | 2.1 MB per 1000 ticks compressed (130 MB per 62 s rollout); `Σ_q`/`Σ_q̇` are 60% of it |
| warm-up | **16 000 ticks (16 s)** — the gyro-bias drift plateau, worst of the four terrains (15.7 s). `Σ_q` itself plateaus in 3.2 s |
| upright / on-field | asserted per rollout; a fall or an excursion past the 64 m hfield **raises**, and nothing is written |

Two hazards the module docstring expands on: the saved `contact_chol` carries the sim's
stance/swing truth (constant it during training), and `make_contact_channels` vmaps the MJX FK
over the whole time axis — it needs ~38 GB on a 62 s rollout, so use
`collect.contact_channels_chunked` for anything that runs the feature path over a full rollout.

#### Domain-randomised collection (`--dr`, `config/collect_dr.yaml`)

The first 12-rollout dataset varied only terrain tilt and came out contact-wise near-identical
(93–96 contact events, stance duty 0.630–0.642 across all twelve), and the learned `Σ_C` was then
**79% explained by gait phase alone**. `--dr` randomises friction, adds pelvis pushes, and
resamples the velocity/height command, so contact quality stops being a function of stride phase:

```bash
uv run python -m invariant_estimation.sim.collect --dr                    # config/collect_dr.yaml
uv run python -m invariant_estimation.sim.collect --dr my.yaml --seconds 30 --dr-seed 7
uv run python -m invariant_estimation.sim.collect --record-slip           # DR off, slip recorded
```

```python
dr, run = collect.load_dr_config()                 # config/collect_dr.yaml -> (config, run kwargs)
r = collect.collect_rollout("flat", seed=0, seconds=60, collector=c, dr=dr)
r.truth["slip_sat"]    # (T, 2) friction-cone saturation |f_t|/(mu f_n), worst contact per foot
r.truth["contact_fn"]  # (T, 2) N; 0 ⇒ no loaded contact, i.e. this sample says nothing about slip
r.truth["push_force"]  # (T, 3) N applied to the pelvis
r.truth["cmd"]         # (T, 5) the live [vx, vy, yaw, standing, base_height]
r.meta["friction_mu"], r.meta["push_schedule"], r.meta["cmd_schedule"], r.meta["slip_fraction"]
```

* **`dr=None` is bit-for-bit the pre-DR collector** — verified: all 28 saved leaves identical to
  `HEAD`'s module on the same rollout. The old `data/*.npz` stay reproducible, and every DR
  rollout carries `meta["dr"]` (`null` on the old ones) so a loader can tell the generations apart.
* **Slip is recorded, not inferred.** `mj_contactForce` per foot contact per physics tick; below
  `normal_force_min_n = 5 N` the cone is meaningless and the sample is marked unloaded. It cannot
  be recovered from a finished rollout, which is why it is here. `--record-slip` gets the DR-off
  baseline (a read only — the trajectory stays bit-identical).
* **The friction floor is terrain-dependent** and a fall costs the whole rollout, so the config
  carries a `range_by_terrain` override. Measured, 8 s walks with pushes and command resampling:

  | terrain | mu → slip fraction |
  |---|---|
  | flat | 1.0 → 2.9% · 0.70 → 3.9% · 0.40 → 8.5% · **0.20 → 26–44%** |
  | waves | 0.20 → 34% |
  | stepping_stones | 0.30 → 14% |
  | hard_stepping | 0.60 → 11% · 0.45 → 19% · 0.30 → 19% · **0.20 → FELL** |

* Cost: the `mj_contactForce` loop adds ~0.11 wall-s per simulated second (sensor read 0.21 →
  0.32 s/sim-s), i.e. ~3% on top of a collection run that `run_fused` already dominates.

**Reading the logs — two traps, both measured (`PORT_NOTES.md`):**

* Under `beta_nll` the **loss is not a progress metric**. It can rise while
  calibration improves by orders of magnitude, because the `stop_gradient`
  β-weight is not being minimised. Watch `metrics.nis_over_dof → 1.0`. Under
  `l2_velocity` the loss *is* monotone — run that baseline first.
* The **first optimiser step is a no-op** (`init_value=0.0` ⇒ `lr(0) == 0`), and
  the second moves the head only (the §4 zero-init head makes the trunk gradient
  exactly zero until the head is nonzero). Not a broken loop.

### Two things measured on the REAL model that the fixture did not show

Both from the first 200-step run (2026-07-28; `PORT_NOTES.md`, "First real
ContactNet runs"), and both change how you read a run:

* **β-NLL does not train on the real model as written.** `S` is in m² and the
  contact block is 6-dimensional, so `logdet S ≈ -93` and the detached β-weight
  `exp(β·logdet S)` is **5.2e-21**. `‖g‖` lands at 5e-18, which AdamW's
  `eps = 1e-8` divides into oblivion: 150 steps moved nothing (`NIS/dof` flat at
  1.3–1.6). This is a units problem, not a gradient problem — the fix is to
  offset the *detached* weight by a constant (a pure loss rescale) or to shrink
  Adam's `eps`. **Do not read a flat β-NLL run as "converged".**
* **Under L2, `NIS/dof` moves away from 1, not toward it.** Measured 1.46 → 0.034
  while the loss fell 82x. `losses.l2_velocity` says exactly why: Σ reaches the
  loss only through the Kalman gain, so only *ratios* are constrained and the
  network is free to inflate the absolute scale. Expected, and the reason β-NLL
  exists — but it means the L2 baseline's success criterion is the **loss**, and
  its `NIS/dof` should be logged as a diagnostic, not a target.

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
  contactnet/                   learned contact measurement covariance (network_plan.md)
  replay/logsource.py           hardware-log reader for the parity harness
tests/pipeline/                 G9 synthetic scenarios + I7 jaxpr-constancy
tests/replay/                   Java parity — the acceptance test (incl. fused real-model)
```

Authoritative design docs: `CLAUDE.md` (spec, invariants I1–I10, gates G1–G10),
`TEST_SUITE_MAP.md` (per-test scenarios/tolerances), `PORT_NOTES.md` (deviations),
`DESIGN_DECISIONS.md` (choices whose symptoms look like bugs).
