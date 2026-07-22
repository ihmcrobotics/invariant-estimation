# Joint-KF contract card — FROZEN in Phase 0

Handed verbatim to every porting agent. Do not invent alternatives to anything
here. If the contract looks wrong, **say so in your report** instead of working
around it — the parent integrates and only the parent changes this file.

---

## 1. State layout (locked by `JointLevelKFStateTest.testXOrdering`)

```
x = [ q (n) ; q_dot (n) ; b_omega (3m) ]   dim = 2n + 3m
```

| segment   | state indices        | meaning                        |
|-----------|----------------------|--------------------------------|
| `q`       | `i`                  | joint position of joint `i`    |
| `q_dot`   | `n + i`              | joint velocity of joint `i`    |
| `b_omega` | `2n + 3k .. +3`      | gyro bias of **IMU** `k`       |

- `n` = filtered joints, `m` = **DISTINCT IMUs** (not pairs).
- **Bias is per-IMU.** Two pairs sharing an IMU share one bias 3-vector. This is
  required by I6 / `testBiasColumnsOfHgAreExactlyL` and by the G7 stacked oracle.
- Bias lives here and nowhere else (I1). The InEKF stays pure `SE_{N+2}(3)`.

## 2. Types — `src/invariant_estimation/jointKF/state.py` (frozen, do not edit)

- `JointKFState(x, P)` — NamedTuple, the pure `(x, P)` carry (I10).
  Views: `.q(n)`, `.q_dot(n)`, `.b_omega(n)`, `.b_omega_imus(n)`,
  `.sigma_q(n)`, `.sigma_q_dot(n)`, `.sigma_b(n)`.
- `JointKFParams` — scalars only, from `config/filter_cfg.yaml` `joint_kf:`.
  Build with `default_params(**overrides)`; an unknown field raises `TypeError`.
- `JointKFBuild` — all name-resolved static structure: index arrays, fixed-shape
  masks, per-joint/per-IMU parameter vectors.
  Index helpers: `.dim`, `.joint_index(i)`, `.velocity_index(i)`, `.bias_col(k)`,
  `.pair_parent_bias_col(e)`, `.pair_child_bias_col(e)`, `.pair_velocity_cols(e)`,
  `.stacked_row_for_pair(e)`, `.anchor_row0`, `.n_stacked_rows`.
- `init_state(build, params, q0=None)` — Java `initialize()`.
- Name tables: `rotor_inertia_for_name`, `alpha_for_name`, `encoder_var_for_name`
  (case-insensitive **substring**, longest key first).
- `SEAM_MAP` — Java test hook → Python callable. Your module must provide the
  callables listed for it.

## 3. Config keys

Everything lives under `joint_kf:` in `config/filter_cfg.yaml`. Read scalars via
`default_params()`, never hard-code. Values marked `[test-locked]` are asserted
by `tests/test_config.py::test_test_locked_values_match_the_java_suite`.

**YAML 1.1 trap: write `1.0e+9`, never `1.0e9`** — the latter parses as a
*string*. `config.py`'s loader guard catches it at load time.

## 4. Module ownership (Phase 1/2 fan-out)

| module | owner | provides |
|---|---|---|
| `jointKF/process.py` | A1 | Schur → `Lambda_eff` → Gram `Qa` → Van Loan `Q` |
| `jointKF/predict.py` | A2 | `build_transition`, `predict` |
| `jointKF/update.py`  | A3 | `joseph_update` (masked-K gate) |
| `jointKF/measure.py` | B1 | encoder rows + stacked gyro `z_g/H_g/R_g`, `L` |
| `jointKF/anchors.py` | B2 | stance-anchor rows, F/U split, `R_anchor` |
| `jointKF/build.py`   | parent | graph resolution, union-find acyclicity |
| `jointKF/filter.py`  | parent | `step` / `run` |

You own **exactly** your files plus your test file. Do **not** touch
`__init__.py`, `config/filter_cfg.yaml`, `state.py`, `PORT_NOTES.md`,
`DESIGN_DECISIONS.md`, or another agent's files. The parent integrates.

## 5. Model seam — `RobotModel`

Kinematics and `M(q)` come from MJX via `model/mjx_model.py` (Phase 0b), which
implements `src/invariant_estimation/robot.py::RobotModel`. Tests drive the
**same adapter** through the synthetic chain fixture in `tests/jointKF/_fixture.py`.
Never hand-roll a CRB mass matrix — it would make the G3 armature-equivalence
oracle a tautology (same hand writing both sides).

**Armature:** MuJoCo folds `dof_armature` into `qM` as an exact diagonal add on
the hinge DoFs, touching neither `M_bb` nor `M_jb` (verified empirically in
Phase 0b). Therefore `Lambda_eff = Lambda + diag(rotor)` comes out of the Schur
complement automatically. **Never add rotor inertia again post-Schur** — that is
the double-add trap (CLAUDE.md §6).

## 6. Constant-XLA-graph rules (I7, CLAUDE.md §4) — non-negotiable

- No data-dependent shapes; no Python branches on traced values inside jit.
- Every gate/skip/anchor-active decision is a **fixed-shape float mask**
  advanced with `jnp.where`.
- Inactive anchor ⇒ residual zeroed **and** `R` block set to `R_LARGE = 1e12 * I3`.
  **Never zero anchor R rows** — that makes `S` singular (CLAUDE.md §6).
- `cond(S)` gate: Cholesky-diagonal proxy `(max L_ii / min L_ii)^2`;
  `gate = (cond < cond_s_max)` as a float; `K <- gate * K`. A gated update must
  leave `(x, P)` **bit-identical** (`testSingularInnovationIsSkippedNotLatched`).
- NaN hardening: per-measurement finite-mask → the same gated-K mechanism.
  Recovery must be automatic; never latch.
- No `inv(S)` — Cholesky solve, and symmetrize `S` first.
- float64 everywhere at the filter boundary (I8). `invariant_estimation/__init__.py`
  sets `jax_enable_x64=True` at import.
- Diagnostics (per-joint NIS, innovation, `Qa` diag, anchor count, gate skip
  counts, cond proxy) are fields of a returned pytree — the tests read them, so
  they are part of the seam surface, not optional logging.

## 7. Test-porting rules (binding — CLAUDE.md §5)

- **Precedence on conflict: Java tests > paper > Java implementation source.**
- Port these oracles **bit-for-bit**:
  - `spd(size, seed)`: fill `m.flat[i] = sin(i + 1 + seed)` row-major, then
    `a = m @ m.T + size * I`.
  - `genericH(k, dim, seed)`: `H[r,c] = sin(0.37*(r*dim + c + 1) + seed)`.
  - `seededPrior` mean patterns — predict: `i+1`, `i+1+100`, `i+1+1000`;
    update: `0.1*(i+1)`.
  - the explicit-inverse reference KF, the information-form nuisance-marginalized
    stacked reference, the LU Schur reference, the quadratic-form NIS.
- **Preserve trial counts, seeds, and tolerances verbatim** from
  `TEST_SUITE_MAP.md`. Fix one Python seed per test.
- Oracles must be **independent**: NumPy, written from the closed form, never
  calling the code under test. See `tests/inEKF/_oracles.py` for the pattern.
- `vmap` the trial loop rather than Python-looping over it (repo convention —
  see `tests/inEKF/test_sek3_utils.py`).
- `tol=0.0` bit-equality → assert determinism against the port's own repeat run.
- Diagnostic **strings** → port the observable (degenerate-row attribution), not
  the message text.
- Skip both `*AllocationTest` allocation tests (JVM `ThreadMXBean`). Keep
  `testHotPathStaysFinite`.

## 8. Mandatory before you report green (JOINTKF_PORT_PLAN §4)

Three tests in this project have already passed against **wrong**
implementations. Assume yours will too unless you check.

**Mutate the source** — flip a sign, drop the rotor term, zero a block — and
confirm your decisive assertion actually fails. Report what you mutated and that
it failed. A green test is not evidence on its own.

Related lessons, worth internalising:
1. A test that *exercises* a term is not a test that *constrains* it.
2. When a test barely fails, find the mechanism before touching the threshold.
3. State what a test proves, not what you hoped it proves.

## 9. Report back

- What you implemented.
- Every deviation from the Java, with its justification.
- Anything the tests do **not** constrain (guesses you had to make).
- Any place the Java source and `TEST_SUITE_MAP.md` disagree.
- Your mutation check: what you broke, and that the test caught it.
