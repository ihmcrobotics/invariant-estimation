# Joint KF port (G6–G8) — execution plan and session handoff

**Written 2026-07-22.** Self-contained: a fresh Claude session should be able to
start from this file alone. Read this, then `CLAUDE.md`, then `TEST_SUITE_MAP.md`.

Companion documents:
- `CLAUDE.md` — the authoritative spec (invariants I1–I10, gates G1–G10, §2 tables)
- `TEST_SUITE_MAP.md` — per-test scenarios, tolerances, seeds, trial counts (binding)
- `PORT_NOTES.md` — what has been ported so far and every deviation, with reasons
- `DESIGN_DECISIONS.md` — deliberate choices whose *symptoms* look like bugs
- `config/filter_cfg.yaml` — every tuning number

---

## 1. Where the project stands

**Done — the InEKF is complete through G5, plus the scan body.** 459 tests green
(`uv run pytest -q`, ~80 s).

| Gate | Status |
|---|---|
| G2 `SEK3UtilsTest`, `InvariantStateTest` | green |
| G3 `InvariantPropagatorTest` | green (+1 port-specific test, see §4) |
| G4 `ContactUpdaterTest`, `GravityLevelingUpdaterTest` | green |
| G5 `InvariantUpdaterTest`, `InvariantEKFTest` | green |
| G5 reseed / contact-trust | **dropped by decision** — `DESIGN_DECISIONS.md` §3, §4 |
| `inEKF/filter.py` scan body + `run` | green, I7 constancy proven |
| **G6–G8 joint KF** | **not started — this document** |
| G1 model layer | partly covered by Phase 0b below |
| G9 pipeline, G10 sim/eval | not started |

56 of the Java suite's ~176 tests are ported; 20 are deliberately dropped; ~99 are
the joint-level suite this plan covers.

**Source layout**

```
src/invariant_estimation/
  config.py            YAML loader for config/filter_cfg.yaml
  robot.py             RobotModel Protocol — NO implementation yet (Phase 0b)
  inEKF/               complete: group, state, propagate, correct, contact,
                       gravity_update, ekf, filter
  jointKF/             SUPERSEDED design — to be rewritten by this plan
tests/
  inEKF/_oracles.py    NumPy-only oracles (the pattern to copy)
  jointKF/             five files to delete, test_filter.py to trim
```

---

## 2. Why this is implement-then-port, not a test port

`src/invariant_estimation/jointKF/` exists (956 lines) but was built to a
superseded design. Against the Java reference —
`/home/llibshutz/workspaces/robot-stuff/ihmc-open-robotics-software/ihmc-state-estimation/src/main/java/us/ihmc/stateEstimation/jointLevel/JointLevelKFPreFilter.java`
(2584 lines) — it is missing nearly all the load-bearing machinery:

| Required by G6–G8 | Present today |
|---|---|
| Schur complement `Λ = M_ff − M_Nfᵀ M_NN⁻¹ M_Nf` | ✗ (`acceleration_cov_mass` is the Rev.1 locked-base `σ_τ²M⁻²`) |
| `Λ_eff` = Λ + rotor-inertia diagonal, Weyl floor | ✗ |
| Gram-form `Qa = Y Yᵀ`, per-joint `σ_τ,i = α_i τ_max,i` | ✗ (single scalar) |
| `QA_MAX` tripwire (surface, never rescale) | ✗ |
| Stance anchors, F/U split, `R_anchor = Σ_ε + J_U diag(σ²)J_Uᵀ` | ✗ |
| `cond(S)` gate + S-pivot floor + skip-not-latch | ✗ |
| Per-joint encoder variance, NIS diagnostics | ✗ |
| Bias **per-IMU** (`m` = distinct IMUs) | ✗ — currently per-**pair** |

The last row is a breaking layout change: I6's exact `LΣLᵀ` on the shared-base-IMU
star requires per-IMU bias, and the G7 stacked oracle cannot pass without it.

**Target:** ~90 ported tests across 14 classes. Both `*AllocationTest` allocation
tests are skipped per the map; `testHotPathStaysFinite` is kept.

---

## 3. Decisions already taken (do not relitigate)

| Decision | Where recorded |
|---|---|
| Existing hand-written jointKF tests are **superseded** — delete the five the ported suite covers, trim `test_filter.py` to JAX-only invariants (jit/grad/vmap) | this plan, Phase 0 |
| `TEST_SUITE_MAP.md` **copied into the repo** | Phase 0 step 1 |
| Stance-anchor bias observability must be **explicitly** tested, not just the anchor algebra | Phase 2, B2 |
| Kinematics and `M(q)` come from **MJX**, not a hand-rolled CRB; the MJX→`RobotModel` adapter lands as **production code now** | Phase 0b |
| Alex URDF will be vendored via **Gitman** (`gitman.yml`, not yet present) so it tracks the lab's upstream; `model/urdf2mjcf.py` consumes it | Phase 0b note |
| All noise parameters are **variances**, never standard deviations | `PORT_NOTES.md` G3 |
| `Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt` — first-order, `Ad` retained | `DESIGN_DECISIONS.md` §2 |
| Update convention `X̂⁺ = exp(−(Kν)^∧)X̂` with `H = [0 0 +I −I]` | `PORT_NOTES.md` G4a |
| No contact mask in the InEKF — condition rides in `Σ_C` | `DESIGN_DECISIONS.md` §1 |

---

## 4. Hard-won lessons this session — read before writing tests

Three tests in this project have passed against **wrong implementations**. Assume
yours will too unless you check.

1. **`InvariantPropagatorTest` cannot distinguish `Γ_1/Γ_2` from plain Euler.**
   Every Java mean scenario has either ω=0 or a=0, and the log-linear test is blind
   to accuracy because log-linearity follows from the propagation being
   *group-affine*, not from it being *correct*. Fixed by adding a
   quadrature-oracle test with ω≠0 **and** a≠0. *Lesson: a test that exercises a
   term is not the same as a test that constrains it.*
2. **The gravity `R` triad was built about the measured direction instead of the
   predicted one.** Every structural test passed; only a convergence-*rate* test
   failed, and only marginally (1.87e-3 vs a 1e-3 bound) — the shape of failure
   that gets "fixed" by loosening a tolerance. *Lesson: when a test barely fails,
   find the mechanism before touching the threshold.*
3. **The jaxpr-constancy tests were oversold.** Traced arrays cannot change a
   jaxpr, so equality across input patterns is near-automatic; what those tests
   actually catch is a data-dependent branch (which raises). *Lesson: state what a
   test proves, not what you hoped it proves.*

**Therefore, mandatory at every phase gate:** mutate the source (flip a sign, drop
the rotor term, zero a block) and confirm the test fails. A green test is not
evidence on its own.

---

## 5. Execution plan

### Structure: phased fan-out

Agents cannot see each other's work, so anything they must agree on has to exist
first. Two things dominate: the **contracts** (state layout, params, config keys)
and the **model seam** (FK, Jacobians, `M(q)` — an oracle four test classes depend
on). So: freeze contracts → build and verify the model seam → fan out → integrate.

Agents own disjoint files. Shared files (`__init__.py`, `config/filter_cfg.yaml`,
`PORT_NOTES.md`, `DESIGN_DECISIONS.md`) belong to the parent session alone.

### Phase 0 — contracts (parent, no agents)

1. `cp ~/Documents/TEST_SUITE_MAP.md ./TEST_SUITE_MAP.md`
2. Extend `config/filter_cfg.yaml` `joint_kf:` with the Java tuning table
   (`JointLevelKFPreFilter.java:70–200`): `encoder_var 5e-5`, `sigma_accel 50.0`,
   `sigma_tau 5.0`, `target_qdd_std 20.0`, `alpha_default 0.15`, the 9 calibrated
   `alpha_overrides`, `qa_max 900.0`, the 16-entry `rotor_inertia` table +
   `default 0.005`, `sigma_gyro_floor 1e-6` (+ `_trace 3e-6`), `cond_s_max 1e9`,
   `init {pos 1e-6, vel 1.0, bias 2.5e-3}`, `anchor_var 4e-4`,
   `sigma_qd_unfiltered 0.1`, `lag_slew_smoothing_hz 5.0`, `imu_bias_process_var 1e-4`.
   Mark `[test-locked]`; extend the checksum test in `tests/test_config.py`.
   **YAML 1.1 trap: write `1.0e+9`, never `1.0e9`** — the latter parses as a
   *string*. `config.py`'s loader guard catches it.
3. Rewrite `jointKF/state.py` to the frozen contract:
   - `x = [q (n) ; q̇ (n) ; b_ω (3m)]`, `dim = 2n + 3m`, **m = distinct IMUs**
   - `JointKFParams` from config; `JointKFBuild` holding static index arrays (pair
     parent/child ordinals, per-pair q̇ columns, anchor chains with F/U split,
     per-joint α/σ_τ/rotor/encoder-var arrays)
   - name-table resolution at build time in plain Python, per I7
4. Delete the five superseded test files; trim `tests/jointKF/test_filter.py`.
5. Write the **contract card** handed verbatim to every agent.

### Phase 0b — MJX model seam (1 agent; parent verifies before Phase 1)

Adds `mujoco` + `mujoco-mjx` to `pyproject.toml` (needed for G1/G9/G10 anyway).

**Production** — `src/invariant_estimation/model/mjx_model.py`, implementing the
existing `robot.RobotModel` Protocol:
- `mass_matrix(q)` → dense `qM`, `(jj, jb, bb, bj)` gather by DoF-index arrays
  resolved at build time (CLAUDE.md §2; I7 — no strings in jit)
- site angular Jacobians, columns gathered by chain index arrays, rotated to the
  child frame (the `J_ang(q̂) S_ab` of §2)
- FK for site poses
- constructed from an MJCF path, so the Gitman-vendored Alex model drops in unchanged

**Test fixture** — `tests/jointKF/_fixture.py` + `_oracles.py`:
- programmatic MJCF for a serial revolute chain, axes cycling X/Y/Z by `i%3`,
  random link offsets, IMU sites — drives the **same adapter** as production
- `apply_consistent_motion(q, q̇)` — zero base twist, gyros = link angular velocity
- `spd(size, seed)` (`sin(i+1+seed)` fill, `m·mᵀ + size·I`),
  `genericH(k, dim, seed)` (`sin(0.37(r·dim+c+1)+seed)`),
  `assert_allclose/symmetric/psd`, and `SHAPES` — four fixtures:
  n=8/m=2, n=4/m=2, n=3/m=2, and n=8/m=3 with two pairs sharing the middle IMU

**Why MJX rather than a hand-rolled CRB.** CLAUDE.md §2 already specifies MJX for
production, so a hand-rolled CRB is a throwaway. More importantly it would make
the G3 armature-equivalence oracle a tautology — the same hand writing both sides.
Because MuJoCo folds `dof_armature` into `qM`, the oracle becomes a genuine
two-model comparison (armature set vs zeroed).

**Verification gate — nothing proceeds until green.** MJX is trusted for CRB but
not blindly: each check reaches `M`/`J` by a route independent of the MJX call
being checked. This is also G1's model-layer sign-off.
- MJX FK vs a hand-rolled serial-chain FK (~15 lines), 20 random `q`, ≤1e-10
- angular Jacobians vs `jax.jacobian` of the FK (autodiff — independent of MJX's
  analytic Jacobian path)
- `qM` vs kinetic energy `½q̇ᵀMq̇` accumulated from body twists
- `qM` diagonal reflects `dof_armature` exactly
- `M` symmetric PD at 20 random configurations
- `apply_consistent_motion` ⇒ relative gyro `= J_ang q̇` exactly

Pin two MJX conventions here rather than discovering them at G9: DoF ordering for
a floating base (free joint's 6 DoF first, then hinges — the nuisance gather
depends on it), and the armature folding above.

### Phase 1 — G6 linear-algebra core (3 agents, parallel)

| Agent | Source | Tests | Java reference |
|---|---|---|---|
| **A1 process** | `jointKF/process.py` | massmatrix_noise (10), rotor_and_gram (3), transition_noise (6) | `updateProcessNoiseFromMassMatrix` L1287–1425, `buildProcessNoise` L1227 |
| **A2 predict** | `jointKF/predict.py` | state (10), predict (7) | `buildConstantTransition` L1220, `predict` L1718 |
| **A3 update** | `jointKF/update.py` | update (6) | `josephUpdate` L1966–2065 |

A1 is the hardest: Schur → `Λ_eff` → Gram `Qa` → Van Loan fill, plus the
armature-equivalence oracle `Schur(qM with armature) ≡ Schur(qM zeroed) +
diag(table)` to 1e-12. That settles CLAUDE.md §2's "armature pre-Schur in MJCF vs
Java's post-Schur add" reconciliation — the one place A1 could silently
double-count rotor inertia (§6 trap).

A3 must implement the gate as **masked K**, so a gated update leaves `(x, P)`
bit-identical. Reuse the proven pattern in
`src/invariant_estimation/inEKF/correct.py::linear_update`.

### Phase 2 — G7 measurement (2 agents, parallel; needs Phase 0b)

| Agent | Source | Tests | Java reference |
|---|---|---|---|
| **B1 stacked gyro** | `jointKF/measure.py` | measurement (11) | `buildStackedMeasurement` L1750–1928 |
| **B2 anchors** | `jointKF/anchors.py` | bias_observability (4) | anchor loop L1826–1875 |

B1 owns the pair rows, the mixing operator `L`, and `R_g = LΣLᵀ` — **I6: the bias
columns of `H_g` ARE `L`**, asserted bit-identical.

B2 owns the anchor rows, the F/U split, and `R_anchor = Σ_ε + J_U diag(σ²_q̇,U) J_Uᵀ`,
masked to `R_LARGE = 1e12·I₃` when inactive. (§4's `R_LARGE` rule belongs *here*,
not in the InEKF — see `DESIGN_DECISIONS.md` §1 for why that mattered.)

**B2 must explicitly test the observability claim, not just the algebra.** Without
anchors, the common-mode bias direction lies in the nullspace of the stacked `H_g`
— assert that directly (rank/nullspace, and unbounded bias-block growth under
repeated predict–update). With one anchor active, assert the gauge is fixed: the
nullspace direction disappears and per-IMU bias covariance converges. That is the
entire reason the anchor exists, and the property that would silently vanish if the
`+I₃` base-IMU `L` block were mis-placed.

### Phase 3 — decisive oracle + wiring (parent, not delegated)

`JointLevelKFStackedOracleTest` (2 tests, 12+8 trials, tol 1e-5): the stacked
Joseph update must equal a nuisance-`ω_base`-marginalized raw-gyro reference KF.
This proves B1+B2+A3 *compose*, and is the one place a wrong answer is not locally
detectable by any single agent — so the parent writes it.

Also `jointKF/filter.py` (`step`/`run`) and `jointKF/build.py` graph resolution
(union-find acyclicity; reject self-pairs and same-link pairs).

### Phase 4 — G8 behaviour (3 agents, parallel)

| Agent | Tests |
|---|---|
| **C1** | filter (6, incl. 20k-tick bias convergence), trajectory (8, incl. NaN hardening + singular-innovation skip) |
| **C2** | standing_stability (5, incl. the one-tick Qa injection ≤ 1.0 (rad/s)² regression gate), singular_innovation (2 — adapt message text → observable) |
| **C3** | encoder_nis (4), direct_velocity (5) |

C3 also implements the direct-q̇ channel (`R_ii = σ_i² + (d̂_i/ω_eff)²`, 5 Hz slew
smoother) — **default off** in config per CLAUDE.md; tests enable it explicitly.

### Phase 5 — integration (parent)

Export wiring, full-suite run, `PORT_NOTES.md` entries per ported class,
`DESIGN_DECISIONS.md` entries for anything deferred, G6/G7/G8 sign-off.

---

## 6. Agent brief template (hand to every agent verbatim)

1. **Read first**: `CLAUDE.md` (§2b tables, I1–I10, §4 constant-graph rules, §6
   traps), `./TEST_SUITE_MAP.md` (your class's section — scenarios, tolerances,
   seeds and trial counts are binding), `DESIGN_DECISIONS.md`, `PORT_NOTES.md`,
   and §4 of this file.
2. **The contract card** (frozen in Phase 0) — state layout, params, config keys,
   fixture API. Do not invent alternatives. If the contract looks wrong, say so in
   your report instead of working around it.
3. **File ownership** — you own exactly the files listed for you. Do **not** touch
   `__init__.py`, `config/filter_cfg.yaml`, `PORT_NOTES.md`, or another agent's
   files. The parent integrates.
4. **Precedence on conflict**: Java tests > paper > Java implementation source.
5. **Preserve** trial counts, seeds and tolerances verbatim. `vmap` the trial loop
   rather than Python-looping (repo convention — see `tests/inEKF/test_sek3_utils.py`).
6. **Oracles must be independent**: NumPy, written from the closed form, never
   calling the code under test. See `tests/inEKF/_oracles.py`.
7. **Mutation-check your decisive assertion** before reporting green (§4).
8. **Report back**: what you implemented; every deviation from the Java with its
   justification; anything the tests do *not* constrain (guesses you had to make);
   any place the Java and the map disagree.

---

## 7. Verification

Run at each phase boundary, not only at the end.

- **Phase 0b gate** — the six model-seam oracle checks. Blocking.
- **Phase 1/2/4 gates** — `uv run pytest tests/jointKF -q`, plus a mutation check
  per agent that the decisive assertion actually discriminates (§4).
- **Phase 3 gate** — the stacked oracle green at tol 1e-5 across 12+8 trials.
- **Phase 5** — `uv run pytest -q` (459 green today; expect ~545), plus the I7
  jaxpr-constancy check extended to the joint-KF step.

## 8. Cost and risk

~9 agents across 4 phases; each needs a substantial brief and produces a module
plus a test file.

**Main schedule risk is Phase 0b** — if the model seam fails verification,
Phases 1–4 stall. Adopting MJX moves risk *earlier* deliberately: convention
surprises (DoF ordering, sparse vs dense `qM`, armature folding) surface against a
4-joint synthetic chain rather than at G9 against full Alex.

**Side benefit:** this closes most of G1's model layer and gives `robot.RobotModel`
its first real implementation. What remains of G1 afterwards is the Alex-specific
URDF → MJCF conversion, fed by the Gitman-vendored URDF.

## 9. What comes after

G9 (`pipeline/main_estimator.py` — fused joint KF → InEKF step, with the
jaxpr-constancy proof) and G10 (`sim/rollout.py`, `eval/consistency.py` — closed-loop
walking, NIS/NEES in χ² bands). The ContactNet socket (§7) also remains: the
`ContactUncertaintyProvider` Protocol and `Features` container are not yet built,
and the second injection point (`Σ_ε` into the joint-KF stance anchors) only
becomes possible once Phase 2 lands.
