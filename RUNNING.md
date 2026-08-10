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

## Training ContactNet (the learned contact-noise socket)

`scripts/run_contactnet.py` is the one-shot orchestrator — **collect → cache →
normalize → train → validate**. It trains a network that emits the InEKF's
per-contact `contact_chol` from sensor-history features only. It writes the
**process** socket (the contact-anchor random-walk block of `Q_d`), never the
measurement socket; features are F=30 (includes the raw q̇ channel) on a `stride=1`
full-rate window, and the sim runs at a coherent 1 kHz (`rp.DT=0.001`,
`rp.DECIMATION=20`).

```bash
uv run python scripts/run_contactnet.py --collect --steps 300 --seconds 45
uv run python scripts/run_contactnet.py --collect --train-seeds 0 1 2 3 --val-seeds 4 5 \
       --steps 300 --warmup-steps 50 --time-budget-s 3000     # the overnight run's args
```

`--collect` re-runs the sim to gather rollouts (it skips any seed already on disk);
drop it to reuse `data/*.npz`. **Pass `--seconds 60` to match the collected set** —
the default is 45 s, and mixing rollout lengths is a silent trap. The val seeds are
disjoint from the train seeds, and held-out validation reports
learned-vs-analytic-baseline **body-frame velocity RMSE**, velocity **NEES**
(target 3), and **contact NIS/dof** (target 1). The per-mode validation labels
(forward/backward/lateral_L/lateral_R/turn_L/turn_R) are keyed to the val seeds in
`VAL_MODES` at the top of `run_contactnet.py` — the `--val-seeds` you pass MUST be
those keys (currently **900–905**) or the rollouts collect with random commands and
label as `"mixed"`.

**GPU (large speedup — do this).** JAX falls back to CPU unless the CUDA plugin is
installed; the project ships the extra. Sync once, then prefix runs with `--extra gpu`:

```bash
uv sync --extra gpu                                   # installs jax-cuda13-plugin (additive)
uv run --extra gpu python scripts/run_contactnet.py … # collection ~2× ; training much faster
uv run --extra gpu python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
```

Note the training clock (`--time-budget-s`, measured from process start) includes
collection + the per-run cache rebuild (`build_channel_cache` is unconditional,
~4 min/rollout, CPU-bound), so budget accordingly or collect in a prior pass.

| path | what |
|---|---|
| `data/flat_seed*.npz` + `data/cache/*_feat.npz` | collected rollouts (~100 MB each) and F=30 feature caches — **gitignored** |
| `results/<YYYY-MM-DD_HH-MM-SS>[_tag]/` | **one directory per run** — every artifact below lands here, so runs never overwrite each other |
| `results/latest` | symlink repointed at the most recent run directory |
| `…/summary.json` | the run's config + held-out metrics (baseline vs learned), plus a `run` block (timestamp, tag, full argv) identifying the run |
| `…/{training,validation}.png` | loss / NIS / reseed curves; held-out RMSE / NEES / NIS bars |
| `…/params.npz`, `…/norm_constants.npz` | trained weights and the **frozen** normalization pair (load them together — a mismatch silently shifts the input distribution) |
| `RESULTS.md` | the written-up validation numbers + caveats (hand-authored, not emitted by the script) |

Name a run with `--tag baseline-redo` (appended to the timestamp) or bypass the
naming entirely with `--out-dir path/to/dir`. The validated run written up in
`RESULTS.md` lives in `results/2026-08-03_11-45-30_coco-faithful-f30/` — it
predates this layout and was moved into it by hand, so its `summary.json` has no
`run` block.

### Loss-function options (velocity + position + orientation)

`--objective` selects the training loss. Beyond `l2_velocity` (body-frame velocity
MSE, the trusted baseline) and `beta_nll`, three composites add **segment-relative**
pose terms on top of the velocity loss:

| objective | loss |
|---|---|
| `l2_velocity` | `L_vel` (unchanged) |
| `l2_vel_pos` | `L_vel + w_pos·L_pos` |
| `l2_vel_ori` | `L_vel + w_ori·L_ori` |
| `l2_vel_pos_ori` | `L_vel + w_pos·L_pos + w_ori·L_ori` |

`L_pos` is the world-frame **displacement** MSE over the segment and `L_ori` is
`‖Log(ΔR_estᵀ ΔR_true)^∨‖²`, the SO(3) log-map error of the **incremental** rotation.
Both are segment-relative on purpose: base position and yaw are unobservable, so
their *absolute* error drifts unbounded in chained BPTT and would swamp `L_vel`
(see `contactnet/losses.py`). The weights default to **auto-measure**: on the first
warm batch each active term is sized to `--pose-weight-ratio` (default 0.5) × `L_vel`
and then frozen for the run (logged to `summary.json`'s `cfg`). Override with
`--w-pos` / `--w-ori`. Validation metrics are objective-independent, so all arms stay
directly comparable.

**Overnight 4-arm ladder.** `scripts/overnight_loss_ladder.sh [STOP_BY_HHMM]` (default
`08:45`) collects a fresh, fixed-terrain (`waves` seed bug fixed) N=8 DR pool under
tag `n8fix`, pre-builds channel caches once, then trains all four arms back-to-back.
It is **deadline-aware** — each arm gets an equal slice of the time left before
`STOP_BY`, so the ladder always finishes on time whatever collection costs:

```bash
nohup scripts/overnight_loss_ladder.sh 08:45 > results/ladder.out 2>&1 &
# env knobs: POOL_TAG CONTACTS SECONDS_PER COLLECT_SEEDS STEPS_CAP WARMUP VAL_RESERVE
```

Last validated run (flat ground, seeds 0–3 train / 4–5 held out): velocity RMSE
**0.086 → 0.028 m/s**, NEES **19.4 → 1.05** vs the analytic `contact_chol`
baseline. Full numbers and caveats (flat-terrain only, etc.) in `RESULTS.md`.

> **The deadline split is a confound.** Each arm gets an equal slice of the
> *remaining* time, and caches warm as the night goes on, so the arms do not get
> equal **steps** (the 2026-08-06 run: A 7639 → D 9839, +29%). Do not compare arms
> from this script without checking `steps_run` in each `summary.json`. Use
> `scripts/l_ablation_ladder.sh` below when you need a matched-step comparison.

### BPTT length `L`, rematerialization, and the contact R floor

Three knobs on `run_contactnet.py` that the loss ladder does not touch:

| flag | default | what it does |
|---|---|---|
| `--L` | 128 | BPTT segment length in **ticks** — how far the gradient traverses the InEKF scan. Independent of `H` (history per network evaluation). |
| `--remat` / `--no-remat` | `True` | Wrap the scan body in `jax.checkpoint(prevent_cse=False)`. |
| `--contact-meas-var` | `1e-4` | InEKF contact-measurement noise floor (`main_estimator` "landmine #2"). |

**`L` is a memory axis, not just a horizon.** Reverse-mode AD stores the scan's
per-tick residuals, so activation memory is `O(L × residuals-per-tick)`; at N=8 the
body's interior (`Φ`, `Ad_X̂`, `Q_d`, `H`, `S`, `K`, the Joseph products — all
33-square) dwarfs the ~9 kB carry crossing each tick. `remat` saves only the carry
and recomputes the interior on the backward pass: `O(L × carry)` memory for one
extra forward evaluation of the body. It is **identity, not an approximation** —
`tests/contactnet/test_remat.py` pins the gradients to 1e-9 relative and pins the
noise floor at bit-equality, so a truncated backward pass cannot hide in it.

> **`remat` was a dead flag until 2026-08-08.** `make_segment_loss` accepted it,
> documented it, and never applied it, so all four 2026-08-06 arms trained with no
> rematerialization while their `summary.json` said `remat: true`. `summary.json`
> now records `remat` and `contact_meas_var` explicitly, and `test_remat.py`
> asserts the primitive is actually in the jaxpr. Measure before choosing:
>
> ```bash
> uv run --extra gpu python scripts/remat_probe.py     # temp/peak MB + s/step per L
> ```

Measured on the RTX 4070 (12 GB), N=8, B=32, `n8fix`, 2026-08-08 — `temp` is compiled
scratch for the train step, `peak` is the runtime device high-water mark:

| L | peak MB (off) | peak MB (on) | s/step (off) | s/step (on) |
|---|---|---|---|---|
| 128 | 1310 | 672 | 0.457 | 0.523 |
| 256 | 2371 | 1081 | 0.958 | 1.123 |
| 512 | 5201 | 2619 | 2.047 | 2.351 |

remat buys a consistent **~2× memory for ~1.16× time**. The operational conclusion:
**L=512 fits on this card without remat** (5.2 GB against ~11.4 GB free), so remat is
headroom rather than a requirement up to L=512 — it is what makes L=1024 (≈10 GB raw)
practical. Note `peak_bytes_in_use` never resets within a process, which is why the
probe runs each cell in its own subprocess; measuring both cells in one process
reports the first one twice.

**The R floor.** `--contact-meas-var` defaults to `0.0` inside `build_collector`
(the port's original behaviour); `run_contactnet.py` now passes `1e-4`, the tuned
value from the 2026-08-07 z-drift study — the one config lever ContactNet
measurably responds to (+31.1%). **Do not raise it to `1e-2`:** that is a
flat-ground cancellation which drifts *upward* on both terrains. The contact-trust
`dwell` knob from the same study is deliberately **not** set here: ContactNet is
measured immune to it (0.9%, it overwrites `contact_chol`), and the
`contact_trust` block in `config/filter_cfg.yaml` is not wired on this branch —
nothing in `src/` reads it and `sim/sensors.ContactTrust` hardcodes `dwell=0.04`.

**Matched-step L ablation.** `scripts/l_ablation_ladder.sh` runs 4 objectives ×
`L ∈ {128, 256, 512}` with **identical `--steps` in every cell**, into one nested
`results/l_ablation/` so twelve run directories do not land loose in `results/`:

```bash
REMAT=on nohup scripts/l_ablation_ladder.sh > results/l_ablation/ladder.out 2>&1 &
# env knobs: REMAT(required) LVALS STEPS WARMUP POOL_TAG CONTACTS CONTACT_MEAS_VAR
#            TIME_BUDGET OUT_ROOT
```

It refuses to start without `REMAT`, skips cells that already have a
`summary.json` (so a re-run resumes), and **fails loudly on any cell whose
`steps_run` ≠ `STEPS`** rather than quietly tabulating a short arm. Two caveats
that belong on any table it produces: pose weights are auto-sized per cell so
`w_pos`/`w_ori` differ across `L` (each term is held at `0.5 × L_vel` at init,
which is what keeps "the same objective" meaningful), and at matched steps `L=512`
sees 4× the trajectory of `L=128`, so *data seen* is not matched — inherent to a
matched-step L ablation.

### Measuring drift, not RMSE — `scripts/drift_backfill.py`

**Do not rank ContactNet arms on held-out velocity RMSE.** Measured over nine
checkpoints, Spearman(|drift_z|, vel RMSE) = **+0.05** — the metric every arm has
been selected on is uncorrelated with the drift we care about.

```bash
uv run --extra gpu python scripts/drift_backfill.py --root results/l_ablation
uv run --extra gpu python scripts/drift_backfill.py --root results/rand_motion \
    --pool n8fix --only L256_A_l2vel     # cross-pool: train anywhere, score on walking
```

Replays each checkpoint over an identical held-out region (truth-seeded, so vertical
error starts at exactly zero) and reports `drift_z` (least-squares slope, m/s),
`final_ez`, `horiz_pct` (against **true** path length), and contact `NIS/dof`. It
prints both rankings and their Spearman correlation.

Three things it does deliberately, each of which would silently invalidate the
comparison otherwise:

* **One region for every cell.** `prepare` sets `t_hi = T − L`, so scoring each cell
  at its own `L` would give different-`L` cells different spans. `EVAL_L` pins one.
* **Each cell's own frozen `norm_constants.npz`**, never refit from the evaluation
  pool. Refitting is invisible for a same-pool cell and silently distribution-shifting
  for a cross-pool one — i.e. wrong for exactly the comparison worth making.
* **`--pool` selects evaluation rollouts, `--root` selects checkpoints.** Training on
  one distribution and scoring on another is a supported, deliberate combination.

`drift_z` and `final_ez` **rank cells differently** — slope answers "where in ten
minutes", accumulated error answers "where now". Pick the one your deployment cares
about; they disagree.

This ranks; it does not explain. `experiments/z_budget.py` (branch
`full-filter/z-debug`) remains the tool that attributes the sink to a specific filter
write and sweeps terrain to catch cancellations.

### The contact R floor — `scripts/zdrift_tonight.sh`, `zdrift_summary.py`

`--contact-meas-var` is the largest lever measured (27% on RMSE; it flipped
ContactNet from 2.5× *worse* than the analytic heuristic on drift to 0.26×). Because
the floor is applied to the **replayed** inputs, an existing checkpoint can be scored
at any floor without retraining (~5 min/point), which is what makes a sweep cheap:

```bash
FLOORS="0 3e-5 1e-4 3e-4 1e-3 3e-3" bash scripts/zdrift_tonight.sh
uv run python scripts/zdrift_summary.py          # incremental: reads whatever exists
```

Read **both** columns. The floor sets innovation covariance directly, so raising it
pushes `NIS/dof` *away* from 1 while it may improve drift — if the two goals pull
apart, that tension is the result, and the combined score hides it.

### Serialize GPU work — `scripts/gpu_lock.sh`

JAX preallocates ~75% of the device per process, so a second JAX job on this 12 GB
card does not run slower — it OOMs, **and takes the first one down with it**. On
2026-08-09 a drift evaluation launched alongside a training cell killed the cell 2.5
minutes in and then died itself, costing the night's queue. Wrap anything that touches
the GPU:

```bash
bash scripts/gpu_lock.sh uv run --extra gpu python scripts/run_contactnet.py …
```

`flock` blocks rather than failing, so a queued job waits its turn. The victim is
whichever process next instantiates a CUDA graph, not the one that over-committed,
which is why care alone is not a control.

### Breaking the stride clock — motion randomization

`collect_dr_pool.py` takes `--cmd-resample-s`, `--cmd-vx/vy/yaw`, `--disturb-rate-hz`.
The default `cmd_resample_s=3.0` is *slower* than the measured ~1.0 s stride, so the
gait settles into a limit cycle and gait phase becomes nearly deterministic from the
sensor window — which lets the network regress Σ_C off phase (phase R² = 0.942)
instead of contact condition.

```bash
bash scripts/rand_motion_check.sh 0.4 0.8        # survivability FIRST — falls yield no data
uv run python scripts/plot_pool_psd.py           # did periodicity actually break?
```

It works spectrally (gait line 62% → 34% of in-band power, sub-gait 9% → 19%) but
**measured worse for deployment**: scored on walking, the randomized-trained net was
2.4× worse on drift slope. Breaking the shortcut removed something useful for the
distribution we deploy into.

**Gate before pushing:** `bash scripts/verify.sh` — the Layer-1 deterministic
checks (F=30, `d_in=600`, `stride=1`, channel order q,q̇,τ; process-socket-only;
nothing under `tests/` modified; the `test_online` geometry). It only greps and
asserts, so a red check means fix the code it points at — never the check.

> **Known red gate:** `tests/contactnet/test_online.py` is a stale F=24 copy of the
> pre-q̇ online test and fails against the F=30 set *by construction* (its `_cfg`
> sets `F=12+2·J_SUB` and `_sensors` never populates `encoders_vel`). One-line human
> fix: `2·J_SUB → 3·J_SUB` + populate `encoders_vel`. Left untouched per the
> no-editing-tests rule; the online↔offline window agreement is verified separately.

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
uv run --extra gpu python run_estimator.py --policy baseline --headless --ticks 500 \
       --vx 0.45 --contacts-per-foot 4 \
       --contactnet results/latest/params.npz                                       # ContactNet in the loop
```

`--contacts-per-foot` must match what the checkpoint trained under — the current
ladder arms (A/B/C/D) are all N=8, so they need `4`. It is not optional and not
inferable: the network's per-contact input is foot-major duplicated, so it accepts
either N without a shape error and simply applies corner-calibrated Σ_C to
whole-sole anchors. `run_estimator.py` reads the training geometry from the
checkpoint's `summary.json` and refuses the mismatch rather than let it run.

Every run prints an error table against the sim's own state (tilt as the policy sees it,
attitude, gyro, velocity, position drift, joint state) over the whole run and over its last half.

| flag | what it changes |
|---|---|
| `--source ...` | which obs terms come from the estimate: `base_ang_vel`, `projected_gravity`, `joints` (routes the 9 filtered joints through the joint KF), or `truth` for none — the estimator still runs and is still scored, which is the A/B control |
| `--imu-noise` | constant per-IMU gyro bias + white noise on gyros/accel/encoders (`--noise-seed`) |
| `--contact-fk measured\|pinned` | whether the InEKF contact FK uses the measured ankle angles (default) or pins them at `qpos0`, as the library default still does — worth ~2x on attitude error, see below |
| `--stance-chol` / `--swing-chol` | the Σ_C factor for a trusted / airborne foot. The InEKF has **no contact mask**; contact condition rides entirely in Σ_C, so a swing foot needs a large factor or the filter keeps believing it is planted |
| `--contactnet PARAMS.npz` (+ `--contactnet-norm`) | run a trained ContactNet in the loop: its learned per-tick `contact_chol` (via `contactnet.online.make_provider`) replaces the analytic stance/swing heuristic. Reads `norm_constants.npz` beside `PARAMS.npz` unless overridden; config is the `ContactNetConfig()` defaults the checkpoint trained under. Run the same command without the flag for the closed-loop A/B |
| `--contacts-per-foot 1\|4` | contact slots per foot: `1` = the shipped N=2 sole pair (default), `4` = the N=8 box corners. Must match a `--contactnet` checkpoint's training geometry, which is enforced against its `summary.json` |
| `--contact-meas-var` | flight's `1e-4` contact measurement-noise floor (port default 0) |
| `--video walk.mp4` | record the run offscreen to H.264 (implies `--headless`, `--video-fps` / `--video-size` tune it) |
| `--ghost [mode]` | draw a translucent robot at the estimated state: `full` (default) or `attitude`. Viewer only |
| `--ghost-offset M` | displace the ghost sideways for side-by-side viewing instead of overlaid |
| `--realtime` | run the estimator on its own thread. Viewer only; the estimate goes slightly stale |
| `--max-backlog-ticks` | how far the estimator may fall behind before the sim thread waits (default 2). Samples are never dropped |
| `--wasd` | use the standalone WASD window instead of the passive viewer |

### Recording a video

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
