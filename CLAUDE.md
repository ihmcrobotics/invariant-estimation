# CLAUDE.md — Python/JAX Port of the Alex Invariant Estimator

> Machine-readable agent dispatch. Authoritative for the Python build; overwrites
> the prior version (upgrade: Java test suite integrated as the validation backbone).
>
> **Primary math spec:** the paper "Invariant Estimator Framework for Alex"
> (`main.pdf`; known small-equation typos — the framework is authoritative, not
> every symbol).
> **Behavioral contract:** the Java test suite, mapped in **`TEST_SUITE_MAP.md`**
> (176 tests / 28 classes across `jointLevel` + `invariant_estimator`). The port is
> correct when it passes the ported suite, not when it "looks like" the Java code.
> **Runtime-semantics tiebreaker:** `JointLevelKFPreFilter.java` for the joint KF.
> Precedence on conflict: Java **tests** > paper > Java implementation source.
> `TEST_SUITE_MAP.md` is a companion file to this one — per-test scenarios,
> tolerances, and seeds live there; this file encodes only the structure.

---

## 0. Mission — three deliverables + one validation obligation

1. **Joint-space KF** (paper §II, `JointLevelKFPreFilter.java`): (q, q̇, b_ω) filter
   with Schur-complement process noise `Qa = Λ_eff⁻¹ Σ_τ Λ_eff⁻ᵀ`, per-joint α
   equalization, stacked gyro measurement over the IMU graph with exact `LΣLᵀ`
   cross-covariances, stance anchors. Exports (q̂, q̇̂, Σ_q, Σ_q̇, b̂_base).
2. **World-centric right-invariant InEKF** (paper §III, Java `invariant_estimator`):
   SE_{N+2}(3) state, constant-H via the b-selector/Π formalism, **contact FK +
   gravity-leveling updates**, touchdown re-seed with fire-once latch, Schmitt-
   trigger contact trust. The gravity update keeps roll/pitch observable; the
   reseed machinery is part of the tested runtime behavior, not optional.
3. **URDF + MJX + pre-trained policy mapping** so both filters run inside vmapped,
   scanned MJX rollouts with a **constant XLA graph**.
4. **Ported test suite**: the 1:1 Python port of the Java suite per
   `TEST_SUITE_MAP.md` (its "Porting guide" section is binding). Tests are not an
   afterthought per module — each build gate below IS a set of ported test classes.

**Out of scope:** ContactNet internals (Lucas implements; build the socket, §7),
GMO/wrench axis, yaw seeding (`enableYawSeeding=false` in the tested config).

---

## 1. Invariants — silent-failure if violated

- **I1 — Bias out of the InEKF state.** Main state is pure SE_{N+2}(3); bias lives
  in the joint KF (P-A). InEKF consumes bias-corrected `ω̄, ā`.
- **I2 — Contacts permanently in state.** N slots fixed for the filter lifetime;
  static graph, differentiable. (Touchdown handling is **re-seed**, not
  add/remove: `reseedContact` re-anchors an existing slot.)
- **I3 — A and H constant; Q_d state-dependent but error-independent.**
  `Φ = I + AΔt + ½A²Δt²` with blocks `Φ_{v,φ}=g^∧Δt`, `Φ_{p,φ}=½g^∧Δt²`,
  `Φ_{p,v}=IΔt` (this exact Φ is the oracle in `InvariantPropagatorTest.
  buildErrorTransition`). `Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt` (paper Eq. 38) — the
  Ad_X̂ stays; do not "clean it up" to match Hartley's convention.
- **I4 — Tangent ordering rotation-first**, locked by `InvariantStateTest.
  testTangentIndices`: rotation 0, base velocity 3, base position 6, contact i at
  9+3i. Group X is (5+N)×(5+N). Build exp/log/Ad from SO(3) primitives; never a
  translation-first `SE3.exp`.
- **I5 — Update convention** `X̂⁺ = exp(−(Kν)^∧)X̂`, right-invariant error
  `η = X̂X⁻¹` with perturbation `X̂ = exp(ξ)X` (left multiplication — the
  convention every `perturb` helper in the test suite uses). Joseph form; NIS
  computed on the **prior** P and prior residual (`testNormalizedInnovation
  SquaredMatchesQuadraticForm`); NIS initializes to NaN before any update.
- **I6 — Exact `LΣLᵀ` on the stacked gyro measurement.** Bias columns of H_g ARE
  the mixing operator L (`testBiasColumnsOfHgAreExactlyL` asserts bit-identity).
  Block-diagonal per-pair R_g is wrong on the shared-base-IMU star.
- **I7 — Constant XLA graph.** No data-dependent shapes or Python branches in jit.
  Everything the Java code does by reshaping/skipping — anchors, cond(S) gate,
  quasi-static gates, Schmitt trigger, reseed latch — becomes fixed-shape masked
  computation (§4).
- **I8 — float64 at the filter boundary**; assert dtype at every entry point.
- **I9 — Per-joint noise scaling, never uniform.** α_i per joint at
  TARGET_QDD_STD = 20 rad/s²; QA_MAX = 900 is a per-joint tripwire counter.
- **I10 — Pure-function state, free test seams.** Implement both filters as pure
  functions over explicit (x, P) / (X, P) carries. The Java suite's `*ForTest`
  seams (`setStateForTest`, `getTransitionMatrix`, `buildStackedMeasurementForTest`,
  …) then cost nothing: they are just the sub-functions, called directly. The full
  seam list in `TEST_SUITE_MAP.md` §"Test seams" is the required public surface of
  each module — design signatures against it before writing internals.

---

## 2. Parameter acquisition — Java source → Python/MJX source

All string matching resolves to **index arrays at build time**; no strings in jit.

| Quantity | Java source | Python/MJX source | Notes |
|---|---|---|---|
| IMU pairs (parent, child) | `IMUBasedJointStateEstimatorParameters` names → sensor map | **Config YAML**: pair list by MJCF site name → site ids at build | Keep structural asserts: reject self-pairs and same-link pairs (singular S) at build; union-find acyclicity (paper §II-B3). |
| Filtered joint set n | Union of 1-DoF joints on pair chains | Union of hinge joints on each site-pair tree path (`jnt_bodyid`/`body_parentid` walk) | Fixed at build ⇒ state dim `2n+3m` static. |
| Pair Jacobians `J_ang(q̂)S_ab` | `GeometricJacobianCalculator`, angular block, child frame | MJX site angular Jacobians, columns gathered by chain index arrays, rotated to child frame | FD-verified at G2. |
| Rotation blocks `ᴶᵉR_i` in L | Live frames at q̂ | Site rotations from MJX FK | Block layout precomputed as index maps. |
| Mass matrix M(q) | `CompositeRigidBodyMassMatrixCalculator` over considered subsystem | MJX full `qM` (CRB), gather (jj, jb, bb, bj) blocks by DoF-index arrays | Equivalent to the Java considered-subsystem trick. Gap joints → nuisance columns exactly as in Java. |
| Schur complement | `Λ = M_jj − M_jb M_bb⁻¹ M_bj`, Cholesky | Same on gathered blocks, `cho_solve` | Nuisance rotor diag on gap joints, zero on base 6 DoF. |
| Reflected rotor inertia | Hardcoded name-substring table, added **post**-Schur | **MJCF `armature`** (pre-Schur diagonal on jj) — algebraically identical since armature never touches M_bb/M_jb: `(M_jj+diag(arm)) − M_jb M_bb⁻¹ M_bj = Λ + diag(arm) ≡ Λ_eff` | **Prove once at G3 (armature-equivalence oracle), then never add rotor again post-Schur — double-add is the trap.** Table values also asserted by `JointLevelKFRotorAndGramTest.testRotorInertiaTableLookup` (substring, case-insensitive, default 0.005). |
| Effort limits τ_max,i | `getEffortLimitUpper()` | URDF `<limit effort>` carried into the build table | σ_τ,i = α_i τ_max,i; α default 0.15 (test-locked). |
| α_i per joint | Name-substring table + TARGET_QDD_STD equalization | Same table, YAML sidecar → length-n array | Offline calibration helper: `α_i ← α_i·√(TARGET²/diag(Qa)_i)`, 2-3 iterations. |
| Encoder position var | Per-joint std lookup, fallback 5e-5 with boot warning | YAML per-joint → array; warn unwired at build | Fallback/wiring behavior is test-locked (`JointLevelKFEncoderNISConsistencyTest`). |
| Gyro Σ_i per IMU | Sensor params, floored 1e-6 if trace < 3e-6 | YAML per-IMU → blkdiag at build, same floor at build | R_g built from gyro MEASUREMENT noise, never bias random-walk (`testMeasurementNoiseUsesGyroMeasurementCovariance` is the regression). |
| Stance anchor chains | Base→foot chains, F/U split (Alex ankles unfiltered) | Same split from MJCF tree, precomputed index sets; `R_anchor = Σ_ε + J_U diag(σ²_q̇,U) J_Uᵀ` | Σ_ε = 4e-4 (ContactNet slot); σ_q̇,U = 0.1. Both test-locked (`JointLevelKFBiasObservabilityTest`). |
| Trusted feet → anchors | Prev-tick trusted set, on-ground debounce | Sim contact / FootSwitch provider → per-foot float mask in scan carry (prev-tick semantics) | §4 masking. |
| Contact trust | `FootSwitchContactProbabilityProvider`: Schmitt stay 0.25 / enter 0.35, 40 ms dwell, EMA, TrustModes | Port as a scalar state machine in the scan carry | Fully deterministic and portable; 10 tests lock it. |
| Touchdown reseed | `reseedContact` + `TouchdownReseedLatch` (trigger 0.5, rearm 0.1, dwell 100 ticks, fire-once) | Same, masked (§4); reseed congruence `P_dd = P_pp + R N Rᵀ`, `P_θd = P_θp` | Zero-release property: identical measurement right after reseed ⇒ NIS = 0, zero rotation correction. |
| Gravity leveling | `GravityLevelingUpdater`: anisotropic R (roll 2.5e-3, pitch 1.9e-1), quasi-static gates (norm 0.05·g, rot 0.15 rad/s raw gyro, horiz 0.5), complementary gravity reference τ = 5 s, pitch-observable gate | Port verbatim; gates become float masks | 14 tests, no RNG — the most portable behavioral spec in the suite. Gate must use the sensor-driven reference, NOT the estimate's attitude (regression F.3). |
| Direct q̇ channel | Lag inflation `R_ii = σ_i² + (d̂_i/ω_eff)²`, 5 Hz slew smoother | Optional, flag in config; if enabled port the exact formula (`lagInflationTracksMeasuredSlewExactly` is deterministic) | Default off for sim v1. |

### 2b. Config fill-in (`config/alex_jointkf.yaml`)

Seed values transcribed from the Java file and cross-checked against the test-suite
constants table (`TEST_SUITE_MAP.md` §"Filter constants the tests lock in" —
that table is the checksum for this one):

```yaml
dt: 1.0e-3
target_qdd_std: 20.0
alpha_default: 0.15
qa_max: 900.0
sigma_accel_fallback: 50.0        # scalar-CWNA path
sigma_tau_fallback: 5.0
encoder_var_fallback: 5.0e-5
sigma_gyro_floor: 1.0e-6
anchor_var: 4.0e-4                # Sigma_eps — ContactNet slot
sigma_qd_unfiltered: 0.1
imu_bias_process_var: 1.0e-4
init: {pos_var: 1.0e-6, vel_var: 1.0, bias_var: 2.5e-3}
cond_s_max: 1.0e9
rotor_inertia:                    # also written into MJCF armature (see §2)
  HIP_X: 0.062, HIP_Z: 0.02, HIP_Y: 0.167, KNEE: 0.167,
  ANKLE_Y: 0.07, ANKLE_X: 0.05, SPINE: 0.062,
  SHOULDER_Y: 0.067, SHOULDER_X: 0.067, SHOULDER_Z: 0.022, ELBOW: 0.022,
  WRIST_Z: 0.005, WRIST_X: 0.005, GRIPPER_Z: 0.005, NECK_Z: 0.005, NECK_Y: 0.005
  default: 0.005
alpha_overrides: {}               # TODO(Lucas)
encoder_pos_std: {}               # TODO(Lucas): 2026-07-15 PSD values
gyro_sigma: {}                    # TODO(Lucas): AlexSensorNoiseParameters
imu_pairs: []                     # TODO(Lucas): star on base IMU
```

`config/alex_inekf.yaml` (all test-locked): gravity (0,0,−9.81); gyro/accel/contact
variances 1e-4 / 1e-3 / 1e-6; contact measurement var 1e-6; gravity-leveling R
(2.5e-3, 1.9e-1); quasi-static gates (0.05, 0.15, 0.5); gravity-reference τ 5 s;
Schmitt (0.25, 0.35, 40 ms); reseed latch (0.5, 0.1, 100 ticks); initial cov 1.0.

---

## 3. Module map, build order, and test-gate mapping

Each gate = a set of ported test classes green (per `TEST_SUITE_MAP.md` scenarios,
tolerances, and trial counts) plus any listed extra oracle. Do not start a module
on a red upstream gate. The build order below IS the map's "Suggested porting
order" with the MJX stages interleaved.

```
config/     alex_jointkf.yaml, alex_inekf.yaml
model/      urdf2mjcf.py            armature, IMU/sole/contact sites, primitive geoms
lie/        se_k3.py                exp/log/Ad/compose/inverse, rotation-first
inekf/      state.py                InvariantState: group element, tangent indices
inekf/      propagate.py            Eq. 36/37/38 propagation
inekf/      contact_update.py       ContactUpdater: H, residual, R̂NR̂ᵀ, mapEncoderNoise
inekf/      gravity_update.py       GravityLevelingUpdater: anisotropic R, gates, ref
inekf/      reseed.py               reseedContact congruence + TouchdownReseedLatch
inekf/      contact_trust.py        FootSwitchContactProbabilityProvider port
inekf/      ekf.py                  orchestrator: predict/update/reseed wiring, NIS API
jointkf/    build.py                graph resolution, index maps, union-find
jointkf/    process.py              qM gather, Schur, Λ_eff, Qa, Van-Loan blocks
jointkf/    measure.py              encoder + stacked z_g/H_g/R_g + anchors (masked)
jointkf/    filter.py               predict / josephUpdate / scan step
pipeline/   main_estimator.py       jointKF → (q̂,Σ_q,b̂) → InEKF fused step
sim/        rollout.py              MJX vmapped envs, ONNX→Flax policy, contact mask
eval/       consistency.py          NIS/NEES vs ground truth, χ² bands
tests/      <1:1 port of the Java suite per TEST_SUITE_MAP.md>
```

| Gate | Modules | Ported test classes (from `TEST_SUITE_MAP.md`) | Extra oracles |
|---|---|---|---|
| **G1** | model/ | — (no Java analogue) | MJX FK vs independent FK ≤1e-10 at 20 random q; mass/CoM match URDF; qM diagonal reflects armature exactly |
| **G2** | lie/, inekf/state | `SEK3UtilsTest` (exp/log round-trip k=1..3, SE(3) agreement at k=1, adjoint homomorphism + conjugation, 1000 trials each), `InvariantStateTest` (7) | Permutation-restriction oracle vs a translation-first reference Ad, residual exactly zero |
| **G3** | inekf/propagate | `InvariantPropagatorTest` (8: free fall, gravity comp, rotation composition, static contacts, symmetry, zero-noise, **exact log-linear error propagation** at large ξ₀ to 1e-10) | Armature-equivalence: Λ_eff via MJX-qM-Schur ≡ armature-free Λ + diag(table) to 1e-12 |
| **G4** | inekf/contact_update, gravity_update | `ContactUpdaterTest` (9: H state-independence exact, residual formula, R̂NR̂ᵀ, mapEncoderNoise J Σ Jᵀ, learned-flag raises NotImplementedError), `GravityLevelingUpdaterTest` (14) | Programmatic-H-from-b ≡ Table I closed forms (contact and gravity); gravity H rank 2, null along e_z |
| **G5** | inekf/reseed, contact_trust, ekf | `InvariantUpdaterTest` (6, incl. χ²(3) NIS mean over 4000 samples), `InvariantEKFTest` (7, delegation to 1e-12), `InvariantEKFReseedTest` (3: PSD over 50 trials, `P_dd = P_pp + RNRᵀ` & `P_θd = P_θp` exact, zero-release), `FootSwitchContactProbabilityProviderTest` (10), `TouchdownReseedLatchTest` (7, incl. 200k-tick chatter property) | — |
| **G6** | jointkf/build, process, filter (linear-algebra core) | `JointLevelKFStateTest` (10), `JointLevelKFPredictTest` (7), `JointLevelKFUpdateTest` (6, vs explicit-inverse reference KF), `JointLevelKFTransitionNoiseTest` (6), `JointLevelKFRotorAndGramTest` (3) | — |
| **G7** | jointkf/measure + kinematics fixture | Python chain fixture (map's "route 1": serial revolute chain, axes cycling X/Y/Z, FK + angular Jacobians + CRB mass matrix; `applyConsistentMotion` oracle preserved), then `JointLevelKFMeasurementTest` (11), `JointLevelKFMassMatrixNoiseTest` (10), `JointLevelKFBiasObservabilityTest` (4), **`JointLevelKFStackedOracleTest` (2 — THE decisive oracle: stacked Joseph update ≡ nuisance-ω_base-marginalized raw-gyro reference KF, 12+8 trials, tol 1e-5)** | — |
| **G8** | jointkf behavior | `JointLevelKFFilterTest` (6, incl. 20k-tick bias convergence), `JointLevelKFTrajectoryTest` (8, incl. NaN hardening + singular-innovation skip), `JointLevelKFStandingStabilityTest` (5, incl. the one-tick Qa injection ≤ 1.0 (rad/s)² regression gate), `JointLevelKFEncoderNISConsistencyTest` (4), `JointLevelKFDirectVelocityMeasurementTest` (5, if channel enabled), `JointLevelKFSingularInnovationDiagnosticTest` (2, adapt message text), `testHotPathStaysFinite` | — |
| **G9** | pipeline/ | `InvariantMainStateEstimatorTest` analogue (5 scenarios: static equilibrium, no-contact rotation integration, hanging under gravity, free fall no-runaway, poisoned IMU) against the MJX model instead of `RandomFullHumanoidRobotModel` | jaxpr-hash constancy of the fused step across differing contact masks / gate states — the constant-graph proof (I7) |
| **G10** | sim/, eval/ | — | Closed-loop ≥30 s walking; ONNX policy oracle ≤1e-6; NIS/NEES in χ² bands on held-out rollouts with heuristic noise (frozen pre-ContactNet baseline) |

**Skip (per the map):** both `*AllocationTest` allocation tests (JVM ThreadMXBean).
JAX's analogue of the allocation guard is G9's jaxpr-constancy check — no
recompilation IS the port's "no per-tick allocation."

---

## 4. Constant-XLA-graph rules (replaces every dynamic Java behavior)

- **Anchors:** always 3(E+K_max) rows. Inactive anchor ⇒ residual zeroed AND
  R block set to R_LARGE (1e12·I₃) via `jnp.where` on the per-foot mask. Never
  zero R rows (singular S). Oracle: masked posterior → K-excluded analytic
  posterior as R_LARGE→∞ (checked inside the G7 stacked-oracle port).
- **cond(S) gate:** Cholesky-diagonal proxy `(max L_ii/min L_ii)²`;
  `gate = (cond < COND_S_MAX)` float; `K ← gate·K`. The Java suite's
  `testSingularInnovationIsSkippedNotLatched` requires exactly-unchanged (x, P)
  when gated — masked K achieves this bit-for-bit.
- **NaN hardening:** `testTransientNonFiniteInputRecovers` requires bad
  measurements skipped, never propagated, no latch. Per-measurement finite-mask
  → same gated-K mechanism; recovery must be automatic.
- **Schmitt trigger, reseed latch, quasi-static gates, pitch gate:** all scalar
  state machines / boolean gates → float carries in the scan state, advanced with
  `jnp.where`. The Java tests define their exact transition semantics (dwell
  counters reset on any high tick; latch disarms on fire; TrustMode switch changes
  output selection only, state machine keeps advancing underneath).
- **Reseed:** `reseedContact` is a fixed-shape covariance congruence, executed
  every tick and blended by the latch-fire mask (fire=0 ⇒ identity congruence).
- **Trusted-feet phase ordering:** previous tick's trust set drives this tick's
  anchors — mask written at end of step k, read at start of k+1.
- **Name tables:** resolved in `build.py` (plain Python) to index/param arrays
  closed over by the jitted step; loud fallback logging at build (Java parity).
- **No `inv(S)`:** Cholesky solve; symmetrize S first.
- **Diagnostics:** YoVariable-published values (per-joint encNIS/qdR, innovation,
  Qa diag, anchor count, gate skip counts, NIS, `wasLastUpdateApplied`,
  condition proxy) become fields of a returned diagnostics pytree — the tests
  read these, so they are part of the seam surface (I10), not optional logging.

---

## 5. Test-porting rules (binding, condensed from the map's Porting guide)

- **Port bit-for-bit:** the deterministic oracles — `spd(size,seed)`
  (`sin(i+1+seed)` fill, `m·mᵀ + size·I`), `genericH(k,dim,seed)`
  (`sin(0.37(r·dim+c+1)+seed)`), seeded-prior mean patterns, the explicit-inverse
  reference KF, the information-form nuisance-marginalized stacked reference, the
  LU Schur reference, the quadratic-form NIS, and `buildErrorTransition`.
- **RNG:** Java streams are not reproducible in NumPy and don't need to be — every
  randomized test is property-based (oracle recomputed from the same draw) or
  statistical with wide envelopes. **Preserve trial counts** (1000 Lie-group, 500
  round-trip, 4000 NIS, 50 reseed-PSD, 200k latch, 12/8 stacked-oracle); fix one
  Python seed per test. `String.hashCode()`-based maps: reimplement
  `h = 31h + c` or substitute any deterministic per-joint map.
- **Kinematics fixture:** route 1 — reimplement the serial revolute chain (axes
  cycling X/Y/Z by i%3) with FK, angular Jacobians, CRB mass matrix. Tests depend
  only on self-consistency between fixture and filter kinematics, never on
  Mecano's specific random geometry. Keep `applyConsistentMotion` (zero base
  twist; gyros = link angular velocity from commanded q/q̇) — it is the oracle
  guaranteeing encoder/gyro consistency that the tracking tolerances assume.
- **Adapt:** `tol=0.0` bit-equality → assert determinism against the port's own
  repeat run; exact symmetry holds if the port also symmetrizes `0.5(A+Aᵀ)`.
  Diagnostic strings → port the observable (degenerate-row attribution), not the
  string. LU-vs-Cholesky loosened tolerances (3e-3, 1e-4) may tighten on a
  single-path NumPy port.
- **Tolerances and constants:** verbatim from the map. The constants table there
  is the acceptance checksum for §2b — a mismatch fails the port, not the test.

---

## 6. Named traps

- **Armature double-add** (§2): armature in MJCF + post-Schur diag = rotor counted
  twice. G3 equivalence oracle retires it.
- **Tangent ordering** (I4): translation-first exp/Ad. G2 permutation oracle +
  `testTangentIndices`.
- **Q_d "cleanup"** (I3): removing Ad_X̂ to match Hartley. The paper's
  error-independent vs state-independent distinction is the design.
- **NIS on the posterior**: NIS must use prior P and prior residual; the
  quadratic-form test fails otherwise.
- **Block-diagonal R_g** (I6) on the shared-base-IMU star.
- **Zeroed anchor rows** instead of R_LARGE masking: singular S.
- **Update sign** (I5): `exp(+(Kν)^∧)` passes easy tests, diverges under
  transients; the δ-injection consistency argument (paper Eq. 51) and the
  residual-reduction tests are the detectors.
- **Gravity update creating yaw**: H_g must stay rank 2 with null along e_z;
  `testGravityUpdateLevelsTiltAndPreservesYaw` requires yaw preserved to 1e-9.
- **Estimate-dependent quasi-static gate** (regression F.3): the gate reads the
  sensor-driven gravity reference, never `R̂ᵀe_z` — the old bug locked out
  leveling above 2.92° of tilt, exactly when it was needed.
- **Reseed double-fire**: the latch exists because of the mid-strike p:1→0→1
  pulse; sub-dwell dips must never re-arm.
- **Uniform noise caps** (I9). **float32 leak** (I8).

---

## 7. ContactNet interface slot (build the socket, not the plug)

```python
class ContactUncertaintyProvider(Protocol):
    def __call__(self, features: Features, contact_mask: Array) -> ContactNoise: ...
# ContactNoise: Sigma_C: (N,3,3)      — InEKF contact/slip noise
#               Sigma_eps: (K_max,3,3) — joint-KF stance-anchor slip covariance
```

Default = current heuristics (constant diagonal Σ_C + per-tick swing-foot
inflation restored in stance, `setContactSlipVariance` semantics; constant
Σ_ε = 4e-4·I₃). `Features` is a typed container of sensor-history quantities only
— no filter mean states. Note the Java contract: `update(..., learnedFlag=true)`
currently raises `NotImplementedException` and the ported test asserts
`NotImplementedError` — keep that behavior until Lucas lands the module. No
`stop_gradient` between provider output and either filter.

---

## 8. Definition of done

- G1–G10 green, meaning: **the ported Java suite passes** (with the map's
  documented skips/adaptations recorded), plus the port-specific oracles
  (armature equivalence, permutation restriction, jaxpr constancy, MJX gates).
- Both filters NIS/NEES-consistent on held-out MJX walking rollouts with the
  heuristic provider — the frozen pre-ContactNet baseline.
- ContactNet socket per §7 wired through both injection points.
- `PORT_NOTES.md`: every paper-vs-Java-vs-test reconciliation; every §2b TODO
  unfilled; every deliberate deviation (allocation tests skipped, NaN-export
  init replaced, direct-q̇ default-off, masked anchors, diagnostics-as-pytree);
  and per-test adaptation notes where `TEST_SUITE_MAP.md` flagged "Adapt".
