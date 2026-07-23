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
131 GB uncompressed) and memoises to `<log_dir>/.parity-cache/*.npz`, falling
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
