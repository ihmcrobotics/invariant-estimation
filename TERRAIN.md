# TERRAIN.md — uneven terrain + vmapped envs in MJX, and whether to train here at all

**The decision this document serves:** can we reproduce IsaacLab's uneven-terrain training for Alex
in pure MuJoCo/MJX, or do we pivot back to IsaacLab?

Three things are already settled, all measured rather than argued:

1. **MJX collides boxes against heightfields** — the make-or-break capability (§2a).
2. **Per-env terrain is data, not graph structure** — 8 envs on 8 different terrains, one traced
   graph, no recompile when the terrain changes (§2b).
3. **The existing policy already walks over IsaacLab's terrain in our sim**, untrained and without
   MJX: all four sub-terrains at 0.38 m/s, upright, tilt ≤ 3.6° (§3 Stage 1).

So feasibility is not the open question. The open questions are **throughput** — which cannot be
judged on this machine yet, jaxlib is CPU-only (§3 Stage 0) — and **MJX-vs-MuJoCo solver parity**
(Stage 2).

Written 2026-07-26, alongside `EXPERIMENTS.md` (the sim-to-sim debugging log) and `RUNNING.md`.

---

## 1. What Alex was actually trained on

`ihmc_lab/tasks/locomotion/alex_ihmc_walk_env_cfg.py`, `TILED_TERRAIN_WITH_HEIGHT_CFG`, selected by
`AlexWalkUnevenEnvCfg` (`terrain_type = "generator"`, `curriculum=True`). Transcribed:

```
size            (8.0, 8.0) m tiles      num_rows 10 (difficulty levels)
border_width    40.0 m                  num_cols 20 (variants per level)
horizontal_scale 0.1 m/px               vertical_scale 0.005 m
slope_threshold 0.75                    seed 1738, use_cache True

sub_terrains (by proportion)
  0.40  stepping_stones       MeshRandomGridTerrain  grid 0.45 m, height 0.00–0.03 m, platform 0.0
  0.15  hard_stepping_stones  MeshRandomGridTerrain  grid 0.75 m, height 0.00–0.07 m, platform 0.5
  0.30  waves                 HfWaveTerrain          amplitude 0.00–0.10 m, num_waves 2.0
  0.15  flat                  MeshPlaneTerrain
```

PLAY variant: 5×10 tiles, 50 envs, `max_init_terrain_level=None` (spawn anywhere in the grid).

Two things to notice before designing anything:

- **This is mild terrain.** 3 cm and 7 cm stepping-stone steps, 10 cm wave amplitude. We are not
  reproducing stairs or gaps.
- **Two of the four types are grids of raised squares** and one is already a heightfield. Only
  `MeshRandomGridTerrain` is nominally mesh-based, and a flush axis-aligned grid of squares is
  exactly representable as a heightfield. That matters a lot for §4.

### The policy is blind

The 98-dim observation is `base_ang_vel(3) + projected_gravity(3) + base_velocity_plus_standing(4)
+ base_height(1) + joint_pos_rel(29) + joint_vel_rel(29) + last_action(29)`. There is **no height
scan**. The env config does instantiate a `height_scanner`, but the policy observation group never
reads it.

So terrain enters through the *physics only*. There is no exteroception to plumb, no ray-casting to
port, and no observation-space change. This removes what is normally the hardest part of a terrain
port.

---

## 2. The two mechanisms, measured

Run `uv run python experiments/mjx_terrain_probe.py` — it asserts both and prints the numbers below.
Re-run it after any `mujoco`/`mjx` upgrade; neither fact is a documented stability guarantee.

### 2a. MJX collides boxes against heightfields

`mujoco 3.10.0` / `mjx` implements 27 geom pairs, including:

```
HFIELD x BOX      HFIELD x CAPSULE    HFIELD x MESH    HFIELD x SPHERE
PLANE  x BOX      BOX x BOX           ...
```

Alex's feet are boxes (`0.26 × 0.14 × 0.055`, from `AlexSimulationCollisionModel`), and the rest of
its collision set is boxes and capsules — all covered. **This was the make-or-break question and it
is a yes.**

### 2b. Per-environment terrain is data, not graph structure

`hfield_data` is a batchable `mjx.Model` field, so terrain can vary per environment inside one
`vmap` without touching the traced graph. 8 envs × 200 steps, each with a different random field:

```
 env    amp   z_start     z_end   ncon
   0   0.00    0.5998    0.0274      4
   1   0.14    0.5998    0.1467      4
   ...
   7   1.00    0.5998    0.1105      4

resting heights differ across envs: 0.0434 m std (min 0.0274, max 0.1613)
all finite: True
traced graphs after the first call: 1
traced graphs after NEW terrain:    1     -> terrain is DATA, not graph structure
```

The idiom is the MuJoCo-Playground `domain_randomize` one — build an `in_axes` tree that is `None`
everywhere except the fields that vary, and `jit` *outside* the `vmap`:

```python
mx = mjx.put_model(model)
in_axes = jax.tree.map(lambda _: None, mx).tree_replace({"hfield_data": 0})
step = jax.jit(jax.vmap(rollout, in_axes=(in_axes, 0)))
step(mx.tree_replace({"hfield_data": fields}), data_batch)     # fields: (N_ENV, nrow*ncol)
```

Distinct resting heights are the load-bearing evidence: it proves each env is colliding against
*its own* terrain, not against a shared one. A version of this check that only asserted "it ran"
would pass with the terrain silently shared.

---

## 3. Staged plan, with the go/no-go at each gate

Do these in order. Each gate is cheap relative to the next, and each can send you back to IsaacLab
before you have sunk the following stage's effort.

### Stage 0 — GPU jaxlib  ⚠️ BLOCKER, do this first

```
jax.devices() -> [CpuDevice(id=0)]
"An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed."
```

**Every throughput number measured on this machine today is meaningless for the training decision.**
MJX's whole argument is thousands of parallel envs on GPU; on CPU it is strictly worse than plain
MuJoCo. Install a CUDA jaxlib and re-run `experiments/mjx_terrain_probe.py` before judging anything.
Gate: `jax.devices()` shows a GPU.

### Stage 1 — one heightfield under the existing (non-MJX) sim  ✅ DONE, PASSED

`uv run python experiments/terrain_stage1_walk.py`. Swaps `run_policy`'s floor plane for a 160×160
heightfield rasterised from §1's parameters, keeps `SCS2_COLLISION_GEOMS` and everything else
untouched, and walks `--policy baseline` at `vx = 0.4` for 20 s across each sub-terrain:

```
  terrain                        relief   travelled          tilt_max   result
  flat (control)                  0.0cm   +7.59 m (0.38 m/s)     2.4    upright 20 s
  waves  a=0.10 n=2              10.0cm   +7.64 m (0.38 m/s)     2.3    upright 20 s
  stepping_stones 0.45m/0.03m     3.0cm   +7.67 m (0.38 m/s)     3.1    upright 20 s
  hard_stepping   0.75m/0.07m     7.0cm   +7.50 m (0.38 m/s)     3.6    upright 20 s
```

So the policy already handles the terrain it was trained on, in our sim, with no retraining and no
MJX. That is the single most encouraging datapoint in this document: whatever else follows is an
engineering/throughput question, not a "can the robot do it" question.

That script's `waves` / `stepping_stones` functions are the reference rasterisers for §4 — lift them
into the MJX path rather than rewriting.

Two things the plane→hfield swap changes, worth carrying forward: an hfield is **finite** (the robot
can walk off the edge — size it for your episode length; 16 m here) and it needs a nonzero base
thickness so it is solid rather than a shell.

### Stage 2 — port the sim to MJX, verify parity on flat ground

MJX is a different solver path from `mujoco.mj_step`. Before trusting rough-terrain rollouts,
reproduce the flat-ground result we already trust: standing at `z ≈ 0.895`, `tilt ≈ 1.1°`,
`|action| ≈ 1.86`, `ncon = 8` (`EXPERIMENTS.md` §1).

Gate: MJX flat-ground standing matches plain MuJoCo to a few mm over 30 s. Expect
`iterations`/`ls_iterations` to need lowering for speed, which changes contact softness — re-check
the gate after any solver change.

Contact budget: in `mujoco 3.10` it is **not** a `Model` field. It is set on the data —
`mjx.make_data(model, naconmax=..., naccdmax=..., njmax=...)` — and defaults are derived from the
model's geom pairs. Raising it is what you do if hfield contacts get clipped; it is a fixed
allocation, so it costs memory per env.

⚠️ Known gotcha, already cost time once: under x64, build scan carries with `mjx.make_data`
(int64) rather than `mjx.put_data` (int32), or the carry dtypes mismatch. See the
`mjx-x64-scan-carry-gotcha` note.

### Stage 3 — N envs, per-env terrain, measure throughput

Extend §2b from a single foot to the full Alex model. Sweep `N_ENV` and record steps/second.

Gate: total throughput beats IsaacLab's for the same robot by enough to justify the port. This is a
**measurement, not a belief** — write the number down next to IsaacLab's.

### Stage 4 — the curriculum

Only worth building once Stage 3 pays. IsaacLab's is a 10-level difficulty ladder with promotion on
performance. In MJX the natural form is: keep `N_LEVELS` pre-rasterised fields on device, hold a
per-env `level` index, and gather `hfield_data` from that index at reset. Gathering keeps the graph
constant; regenerating terrain inside the graph would not.

---

## 4. Representation: one heightfield per env, not a grid of boxes

Rasterise **all four** IsaacLab sub-terrains into `hfield_data` rather than emitting geometry.

Why not boxes: `MeshRandomGridTerrain` at 0.45 m grid over an 8×8 m tile is ~324 squares. As geoms
that is 324 collision candidates per foot per env, and MJX's contact budget is a fixed allocation —
it would dominate cost and force a large `naconmax`. As a heightfield it is **one geom** whose
collision cost is bounded by the sub-grid under each foot box.

Mapping IsaacLab's parameters onto MuJoCo's `<hfield nrow ncol size="rx ry ez bz">`:

| IsaacLab | MuJoCo hfield |
|---|---|
| `size=(8.0, 8.0)` | `rx = ry = 4.0` (radii, not extent) |
| `horizontal_scale=0.1` | `nrow = ncol = 8.0 / 0.1 = 80` |
| terrain max height | `ez` (elevation scale); `hfield_data` is normalised to `[0,1]` and multiplied by `ez` |
| — | `bz` = base thickness below z=0, keep a few cm so the field is solid |

Set `ez` once to the tallest terrain you will ever generate (e.g. 0.15 m) and express every
difficulty level as a fraction of it. Changing `ez` changes the *model*; changing `hfield_data`
does not — and only the latter is batchable.

Four rasterisers to write, matching §1's parameters:

- `flat` — zeros.
- `waves` — `amplitude * sin(2π * num_waves * x / L)`, amplitude 0.00–0.10, `num_waves = 2.0`.
- `stepping_stones` — per-cell random height in `[0, 0.03]`, quantised to `grid_width = 0.45 m`
  blocks (i.e. constant over each 0.45/0.1 = 4.5 px block; round to an integer block size).
- `hard_stepping_stones` — same with grid 0.75 m, height `[0, 0.07]`, plus a flat `platform_width
  = 0.5 m` at the centre (spawn area).

Keep `vertical_scale = 0.005` as an explicit quantisation step if you want to match IsaacLab
exactly — it is what its heightfield backend snaps to, and unquantised terrain is slightly
different terrain.

---

## 5. Domain randomization beyond terrain

`~/alex/alexander-mujoco/training/envs/alexander/randomize.py` already implements this for Alex in
the Brax/MJX idiom — `domain_randomize(sys, rng) -> (batched_sys, in_axes)`, vmapped, randomising
`geom_friction`, `dof_frictionloss`, `dof_armature`, `body_ipos`, `body_mass`. It is the same
`tree_replace` + `in_axes` pattern as §2b, so terrain slots into it as one more batched field.

Reuse the **pattern**, not the values: that repo is an older MJX training setup and
`EXPERIMENTS.md` §0 records that its gains, contacts and armature are stale. The values we trust
come from IsaacLab `robots/alex.py` and the Java, and are already in `run_policy.py`.

Worth adding, since the estimator will eventually consume it: IMU noise. IsaacLab's `NoiseConfig`
for the older MJX env used gyro 0.2, gravity-vector 0.05, joint pos 0.05, joint vel 1.5. The
current IsaacLab Alex env instead puts `Unoise` on the observation terms directly
(`base_ang_vel ±0.3`, `projected_gravity ±0.05`, `joint_pos_rel ±0.01`, `joint_vel_rel ±0.3`).
Use the latter — it is what this policy was trained with.

---

## 6. When to pivot back to IsaacLab

Pivot if any of these holds after Stage 3:

- MJX throughput on GPU does not clearly beat IsaacLab for Alex. The port has no other advantage;
  our reason for wanting MuJoCo is the estimator sharing one model, and that is already true in the
  non-MJX harness (`EXPERIMENTS.md`).
- Flat-ground MJX parity (Stage 2) cannot be reached without solver settings that change contact
  behaviour enough to invalidate the policy.
- The curriculum needs terrain *regenerated* inside the graph rather than gathered from a
  pre-built set.

Conversely, the argument **for** staying in MuJoCo is specific and worth keeping in view: the
estimator, the sim and the policy would share one model and one contact definition, so
training↔estimation inconsistencies become test failures instead of silent bias. That is the whole
reason this repo exists. It is not a throughput argument.

---

## 7. Gotchas already paid for

- **`mj_objectVelocity` frames.** Use `mjOBJ_XBODY`, never `mjOBJ_BODY`, for any body twist used as
  a signal — `mjOBJ_BODY` resolves in the body's *inertial* frame. This silently permuted the gyro
  axes and cost most of a session (`EXPERIMENTS.md`).
- **Foot soles.** `left_sole`/`right_sole` are the contact anchors and sit at
  `ALEX_SOLE_OFFSET = (0.0465, 0, −0.072)` in the ankle-roll frame, not at the link origin. Terrain
  work touches contact, so keep `tests/model/test_sole_frame.py` green.
- **Collision set provenance.** SCS2 uses `AlexSimulationCollisionModel`'s 7 geoms, *not* the URDF
  `<collision>` tags. `run_policy.SCS2_COLLISION_GEOMS` is the transcription.
- **x64 scan carries.** `mjx.make_data`, not `mjx.put_data` (Stage 2).
- **A comparison that agrees on a zero is not a comparison.** Both mistakes that cost the most time
  this month were checks that compared something against itself, or against a zero. When you assert
  "per-env terrain works", assert that the envs *differ*.
