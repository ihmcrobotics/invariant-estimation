# PORT_NOTES.md

Running record of paper-vs-Java-vs-test reconciliations, deliberate deviations,
and unfilled config TODOs (CLAUDE.md §8). One section per ported test class.

---

## G2 — `InvariantStateTest` → `tests/inEKF/test_invariant_state.py`

**Status:** green (7 Java tests → 10 pytest functions; the parametrized
constructor test expands to 4).

### Structural reconciliation

The repo predates the current CLAUDE.md and follows the older
`src/invariant_estimation/inEKF/CLAUDE.md` design record. Two naming deltas
against the new module map, both cosmetic — **no math differs**:

| CLAUDE.md §3 map | This repo | Note |
|---|---|---|
| `inekf/` | `inEKF/` | package case only |
| `lie/se_k3.py` | `inEKF/group.py` | `exp_SEn3` / `log_SEn3` / `Adjoint` live here |
| `InvariantState` (Java-style object) | `InEKFState` (`NamedTuple` pytree) | see below |

The **tangent ordering already matches I4** — rotation 0, base velocity 3, base
position 6, contact *i* at 9+3*i* — and the group column layout matches the map
(`R` at 0:3, `v` col 3, `p` col 4, contact *i* col 5+*i*). No permutation
needed; `testTangentIndices` passes against the existing layout unchanged.

### Deliberate deviations from the Java class

1. **Mutating setters → functional returns.** Java `setRotation(R)` mutates in
   place; the port's `InEKFState` is an immutable `NamedTuple` (I10, and
   required for a clean `lax.scan` carry). Round-trip semantics under test are
   identical: `st = st._replace(R=...)` then read `st.R`. `set_contact_position`
   returns a new state rather than mutating.
2. **`IndexOutOfBoundsException` → `IndexError`**, per the map's port note.
   Negative indices are rejected rather than wrapping, matching Java — Python's
   usual `d[-1]` idiom is deliberately *not* honored on the contact accessors.
3. **Two constructors.** Java's `InvariantState(N)` gives `X = I, P = 0`; that
   is `InEKFState.identity(N)`, added for this port. The pre-existing
   `init_state(N, ...)` (diagonal prior on `P`) is kept for filter use — the
   Java class has no analogue because the Java EKF seeds `P` separately in
   `initialize`.
4. **`EuclidCoreRandomTools` oracles reimplemented** in `tests/inEKF/_oracles.py`:
   `next_rotation_matrix` (random unit axis × angle ~ U(-π, π) through the
   reference Rodrigues formula) and `next_vector3d` (components ~ U(-1, 1)). Seeds
   1234 / 2345 / 3456 / 4567 and `ITERATIONS = 500`, `EPSILON = 1e-12` are
   carried over verbatim; exact draw-matching is not required (the tests are
   round-trip properties, not statistical).

### Additions beyond the Java test

- **dtype assertion** on `X` and `P` in the constructor test (I8: float64 at the
  filter boundary — the Java suite has no analogue since Java is double-only).
- Round-trips are additionally checked *through* `as_matrix`, pinning the
  column convention (`v` = col 3, `p` = col 4, `d_i` = col 5+*i*) that the map
  flagged as "confirm which column holds velocity vs position".

### Open

- `TEST_SUITE_MAP.md` is not in the repo (read from `~/Documents/`). It is
  declared a companion file to `CLAUDE.md` — should be checked in.
- All Java-dependency oracles now live in `tests/inEKF/_oracles.py`, shared by
  the ported classes and deliberately NumPy-only (never calling the code under
  test).

---

## G2 — `SEK3UtilsTest` → `tests/inEKF/test_sek3_utils.py`

**Status:** green (5 Java tests → 13 pytest functions; the three `k = 1,2,3`
loops are parametrized, and the size guard splits into `log` and `exp` halves).

`tests/inEKF/test_group.py` (pre-existing) is kept — it covers what the Java
class does not: the `Γ_0/Γ_1/Γ_2` closed forms, the θ→0 branch, finite gradients
at θ = 0, and jit-vs-eager parity.

### Source change: general-`k` entry point

`group.log_SEn3` and `group.Adjoint` were already generic in the matrix size,
but `exp_SEn3(xi, N)` is parameterized by *contact count*, so `k = 1` (plain
SE(3)) would have required the nonsense `N = -1`. Added **`exp_SEk3(xi)`**,
which infers `k = (len(ξ) - 3)/3` — the direct analogue of Java
`SEK3_Utils.exp`. `exp_SEn3(xi, N)` is now a thin filter-facing alias at
`k = N + 2`; no call site changed and no math moved.

### Deliberate deviations

1. **`testLogRejectsWrongSizedOutput` adapted.** Java packs into a
   caller-supplied output array and throws `IllegalArgumentException` when its
   length ≠ `3 + 3k`. The port *returns* the tangent, so that failure mode does
   not exist; the observable ported instead is the size-consistency guard
   itself, split in two: `log_SEn3` raises `ValueError` on a non-square or
   sub-4×4 input, and `exp_SEk3`/`exp_SEn3` raise `ValueError` on a tangent
   length inconsistent with `k`/`N`.
2. **1000-trial loops → one `vmap` over a batch of 1000.** Trial counts are
   preserved exactly (`ITERATIONS = 1000` per `k`); the Java `for` loop over the
   sample index becomes a batch axis, per the repo's no-Python-loops-over-data
   convention. Every assertion is a property recomputed from the same draw, so
   matching Java's RNG stream is unnecessary (and impossible).
3. **`SE3LieGroupTools` replaced** by `_oracles.se3_exp_reference` /
   `se3_adjoint_reference`: NumPy Rodrigues + left-Jacobian `V`, written from
   the closed forms rather than calling `group.Gamma0/Gamma1`, so the k=1
   agreement test is a real cross-check and not a tautology. Adjoint reference
   block form under rotation-first ordering is `[[R, 0], [(t)_× R, R]]`.
4. **`EuclidCoreRandomTools.nextRotationVector`** reproduced as random unit axis
   × angle ~ U(-π, π) (bounded by the injectivity radius); translational blocks
   ~ U(-1, 1) per `nextVector3D`.

### Tolerances (SEK3Utils)

Verbatim from Java: `1e-10` on round-trip and both k=1 agreement tests, loosened
to `1e-9` on the adjoint homomorphism and `1e-8` on the conjugation identity.
The port is comfortably inside all three — observed worst-case error on the
conjugation identity is **2e-15** across all `k`, i.e. ~7 orders of margin.

One consideration checked and discarded: whether full-π ξ draws could push the
conjugated element `X exp(ξ) X⁻¹` outside the injectivity radius and break the
uniqueness of `log`. They cannot — conjugation is a similarity transform on the
rotation block, so the conjugated angle equals `‖φ_ξ‖ ≤ π`. Java's
full-magnitude draws are kept unmodified.

---

## G3 — `InvariantPropagatorTest` → `tests/inEKF/test_invariant_propagator.py`

**Status:** green (8 Java tests → 9 pytest functions; the extra one is
port-specific, below). No source change was needed — the existing
`propagate.py` mean integrator and `Φ` already satisfy every Java assertion,
including the log-linear property to 1.4e-14 against a 1e-10 tolerance.

### Signature mapping

Java `InvariantPropagator(N, gyroNoise, accelNoise, contactNoise)` +
`predict(state, omega, accel, dt)` → `propagate(state, omega, accel, sigma_c,
params)`. The constructor scalars are folded into `InEKFParams` and the
per-contact `sigma_c` stack by the `_propagator` shim in the test file.

**Resolved (2026-07-21, Lucas):** all noise parameters in the port are
**variances**, matching CLAUDE.md §2b ("variances 1e-4 / 1e-3 / 1e-6"). Only one
upstream source is quoted in standard deviations; those get converted at the
build boundary rather than propagated through the filter. `InEKFParams.sigma_gyro`
/`sigma_accel` were accordingly renamed to **`gyro_var`/`accel_var`** and are now
consumed directly (`Q_g = gyro_var · I₃`, no squaring). The Java constructor
scalars map straight through.

### Deliberate deviations

1. **Multi-step loops → `lax.scan`.** Identical arithmetic to a Python loop, and
   it exercises the scan path the filter actually uses. `testCovarianceStaysSymmetric`
   asserts after *every* step, as in Java, by scanning out the full `P` history.
2. **`buildErrorTransition` reimplemented in NumPy** inside the test file rather
   than calling `state.build_Phi` — it is the *oracle* for the exact Φ of I3, so
   delegating to the code under test would be circular.

### Coverage gap in the Java class (port-specific test added)

**Nothing in `InvariantPropagatorTest` constrains `Γ_1`/`Γ_2`.** A plain Euler
mean — `v⁺ = v + R a dt + g dt`, `p⁺ = p + v dt + ½R a dt² + ½g dt²` — passes
all eight tests. Verified by mutation, not by inspection:

- `testFreeFall`, `testStationaryWithGravityCompensation`: ω = 0, so `Γ_1 = I`
  and `Γ_2 = ½I` regardless. Euler and exact are the *same expression*.
- `testConstantAngularVelocityComposesRotation`: a = 0, so the accelerometer
  carrier never appears.
- `testLogLinearErrorPropagation` does drive both ω ≠ 0 and a ≠ 0, but is blind
  to integration accuracy: log-linearity follows from the propagation being
  **group-affine**, not from it being accurate, and both trajectories are
  propagated with the same integrator. Measured residual is ~1.1e-14 under the
  Euler mutant — indistinguishable from the 1.4e-14 of the exact integrator.

The test is not vacuous — a gravity-sign flip in the position update is caught
at 9.3e-4 — it simply does not constrain the rotating-accelerometer coupling.

Added `test_mean_integration_exact_under_simultaneous_rotation_and_acceleration`
(ω = (0.9,-0.7,1.3) and a = (2.0,-1.5,+9.81), 50 steps at dt = 1e-3). Oracle is
`_oracles.so3_step_integrals`: Gauss-Legendre quadrature (order 40) of
`∫₀^dt exp((ω)_× s) ds` and `∫₀^dt (dt-u) exp((ω)_× u) du` against the reference
Rodrigues formula — independent of `Γ_1`/`Γ_2` by construction. Discrimination
is ~10 orders: exact integrator 1e-14, Euler mutant 2e-4.

### Resolved — `Q_d` is now I3 / paper Eq. 38, verbatim

The old `src/invariant_estimation/inEKF/CLAUDE.md` specified an exact
closed-form integral `∫₀^dt e^{As} Q_c e^{Aᵀs} ds` with **no** `Ad`
conjugation, and argued the conjugation was trivial for isotropic `Q_g, Q_a`.
That argument is wrong: it holds for the rotation block (`R̂ σ²I R̂ᵀ = σ²I`), but
`Ad_X̂` also carries `(v)_× R̂` and `(p)_× R̂` in its first block-column, so
`Ad Q_c Adᵀ` generates genuine cross terms regardless of isotropy.

`InvariantPropagatorTest` does not adjudicate — every covariance assertion in it
(symmetry, positive trace, zero-noise-stays-zero) holds under both forms, and the
ported class stayed green across the switch.

**Decision (2026-07-21, Lucas):** implement I3 literally,
`Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt`, including the first-order `·Δt` discretisation —
the working Java estimator does not use the Van Loan / exact integral, so neither
does the port. The old design doc was deleted rather than amended.

Consequences:

- `inertial_Qd` (the closed-form integral) is **gone**, replaced by
  `continuous_Qc(sigma_c, params)` building the block-diagonal continuous density
  `blkdiag(gyro_var·I, accel_var·I, 0, Σ_{C_i})`.
- `build_Qd(sigma_c, Ad, params)` and `propagate_cov(P, sigma_c, Ad, params)` now
  take the adjoint. `propagate` evaluates it at the **prior** estimate, matching
  the Java predict ordering (the difference is O(dt)).
- **Frame change in the digest:** `Ad_X̂`'s contact diagonal blocks are `R̂`, so the
  conjugation performs the rotation to world itself. `contact.digest` therefore
  no longer calls `rotate_to_world` and returns **body-frame** `Σ_{C_i}`;
  digesting to world as well would apply `R̂` twice. `rotate_to_world` is kept as
  a standalone utility, explicitly out of the propagation path.
- `TODO(van-loan)` left at the `build_Qd` docstring: revisit the exact integral if
  NEES/NIS at G10 shows the cross terms are overstated.
- New regression `test_build_Qd_keeps_the_adjoint` fails if anyone "cleans up"
  the conjugation to match Hartley — the §6 trap, now guarded.

---

## G4a — `ContactUpdaterTest` → `tests/inEKF/test_contact_updater.py`

**Status:** green (9 Java tests → 13 pytest functions; 2 are port-specific
oracles, 1 parametrized over N). G4's other half (`GravityLevelingUpdaterTest`,
14 tests + a new `gravity_update.py`) is **not** done — see "Remaining" below.

### Source change: sign convention flipped to Java / I5

The port previously used `H_i = [0 0 −I … +I(d_i) …]` with the update
`X̂⁺ = exp(+Kν)X̂`. Java (and CLAUDE.md **I5**) use the opposite:
`H_i = [0 0 +I … −I(d_i) …]` with `X̂⁺ = exp(−(Kν)^∧)X̂`.

**The two are mathematically identical** — flipping `H` flips `K = P Hᵀ S⁻¹`, so
`KH` (hence the Joseph form) is unchanged and `exp(+K_old ν) = exp(−K_new ν)`
gives the same posterior. Confirmed empirically: after the flip every behavioral
test (residual reduction, covariance shrink, PSD, Joseph-vs-short form) passed
untouched; only the three tests that *assert the convention itself* needed
updating.

Flipped anyway, because:
- `ContactUpdaterTest.testJacobianStructureAndStateIndependence` asserts `H`'s
  signs **element-wise at tol 0.0** — tests outrank implementation (precedence
  rule), and this one is explicit.
- It makes I5 read literally rather than as an equivalent-but-mirrored
  convention, which matters for the §6 update-sign trap: with Java's `H` the
  residual linearises as `ν ≈ +Hξ`, so `ξ⁺ = Kν` estimates the error *itself*
  and must be subtracted.

Changed: `state.build_H` (p block +I, contact block −I), `correct.apply_correction`
(now `exp(−ξ)`), and the three convention tests
(`test_H_shape_and_pattern`, `test_innovation_linearises_to_plus_H` — renamed
from `_minus_H`, `test_apply_correction_left_multiply`).

### Source addition: ContactUpdater seams

`correct.py` had only the vectorised all-contacts hot path. Added the
single-contact seams the Java class exposes and the ported test drives directly
(I10 — the seam list is the required public surface):
`contact_jacobian(N, i)`, `contact_residual(state, i, y)`,
`rotate_measurement_covariance(state, body_cov)`, `map_encoder_noise(J, Σ)`,
and `contact_update(state, i, y, body_cov, learned=False)`.

Note `contact_jacobian` takes the contact **index**, not a state — state
independence is structural, not merely asserted.

### Deliberate deviations

1. **`testUpdateWithoutContactUpdaterThrows` adapted.** Java throws
   `IllegalStateException` when `InvariantUpdater` has no `ContactUpdater`
   installed. The port has no installable collaborator — `contact_update` is a
   free function, so the state is unreachable. Ported the analogous failure:
   an out-of-range contact index raises `IndexError`, consistent with the
   `InvariantStateTest` bounds contract.
2. **`NotImplementedException` → `NotImplementedError`** on the `learned=True`
   branch (§7 ContactNet socket), per the map's port note.
3. Java's `Random`/`EuclidCoreRandomTools` draws are not reproduced; every
   oracle (FK, residual, covariance rotation, `JΣJᵀ`) is recomputed from the
   same draw, so the assertions are tolerance-based as the map prescribes.

### Port-specific oracles added

- `test_single_contact_jacobians_stack_into_the_precomputed_H` — row-stacking the
  per-contact `H_i` must equal `state.build_H(N)` exactly. This is CLAUDE.md G4's
  "programmatic-H-from-b ≡ Table I closed form" check for the contact case, and
  it ties the seam to the vectorised hot path so a sign or block-offset drift in
  either is caught.
- `test_contact_update_matches_vectorised_correct_at_one_contact` — the two entry
  points must produce the same posterior at N = 1 (to 1e-12).

### Remaining for G4

`GravityLevelingUpdaterTest` (14 tests) and the `gravity_update.py` module it
tests. Note several of its tests (`testGravityUpdateLevelsTiltAndPreservesYaw`,
`testRollStillLevelsUnderAnisotropy`, `testPitchCorrectionAuthorityBelowRoll`)
drive the EKF orchestrator API (`assembleGravityLeveling` / `applyGravityLeveling`
/ `wasLastUpdateApplied` / `getLastConditionProxy`), which is G5's `ekf.py` — so
that half needs at least a minimal orchestrator seam alongside the new module.

---

## G4b — `GravityLevelingUpdaterTest` → `tests/inEKF/test_gravity_leveling_updater.py`

**Status:** green (14 Java tests → 17 pytest functions; the extra 3 are one
port-specific oracle parametrized over attitude). **G4 is now complete.**

New module: `src/invariant_estimation/inEKF/gravity_update.py`. All 14 tests are
deterministic (no RNG) — the map calls this the most portable behavioural spec in
the suite, and that held up: no tolerance needed loosening.

### Model as implemented

- **Residual** (body frame): `r = ĝ_ref − R̂ᵀe_z`, against the complementary
  reference rather than the raw specific force. Diagnostics read straight off it:
  `tilt_pitch = r[0]`, `tilt_roll = r[1]`,
  `tilt_angle = acos(clamp(ĝ_ref · R̂ᵀe_z))`.
- **Jacobian**: perturbing `R̂ = Γ_0(φ)R` gives `R̂ᵀe_z ≈ Rᵀe_z + Rᵀ(e_z)_×φ`, so
  `H = [−R̂ᵀ(e_z)_× | 0 …]`. Yaw column is `−R̂ᵀ(e_z)_×e_z = 0` exactly.
- **Reference**: `ġ_ref = −ω × g_ref + (ĝ_meas − g_ref)/τ`, τ = 5 s, renormalised.
  With ω = 0 it is a pure first-order low-pass (gain `1/√(1+(ωτ)²)` — the test's
  `PREDICTED_ARTIFACT_GAIN`); with real gyro the `−ω × g_ref` term tracks true
  tilt at unity gain and no lag. DC authority undiminished.
- **Gate**: norm ∧ rotation (raw gyro) ∧ horizontal, the last resolved against
  `ĝ_ref` — never `R̂ᵀe_z` (regression F.3).

### The one real bug, and what caught it

First implementation built `R`'s anisotropy triad about the **measured**
direction `ĝ_ref`. Every structural test passed; only
`testGravityUpdateLevelsTiltAndPreservesYaw` failed, and only just — tilt
converged to 1.87e-3 against a 1e-3 bound. Not a crash, not a sign error: a
convergence-*rate* shortfall, which is exactly the kind of thing that gets
"fixed" by loosening a tolerance.

Diagnosis: `H` satisfies `(R̂ᵀe_z)ᵀH = −e_zᵀ(e_z)_× = 0` **exactly** — the
measurement is a unit vector, so its variation is always orthogonal to itself and
the residual component along `R̂ᵀe_z` is structurally unobservable. Building `R`'s
triad about `ĝ_ref` instead left the two null directions misaligned by precisely
the tilt being corrected, so the unobservable channel stayed coupled into `S` and
got fitted against the tight `ROLL_VAR`. The update fought itself.

Fix: build the triad about the **predicted** direction `R̂ᵀe_z`. `S` block
diagonalises, the unobservable channel decouples, and its variance can no longer
perturb the roll/pitch correction. Measured: pitch tilt after 200 steps goes
1.87e-3 → 1.77e-4, against an analytic best-case of 1.90e-4 (the scalar-KF
telescoping bound `θ_0 σ²/(σ² + N)`) — i.e. the update now extracts essentially
all available information.

No test distinguishes the two choices structurally: at every point where the
covariance structure is asserted (`testAnisotropicMeasurementCovarianceStructure`,
`testPitchGateFreezesPitchButNotRoll`, `testPitchDistrustAxisIsBodyYAtNonZeroYaw`)
the predicted and measured directions coincide. Only the convergence-rate test
separates them.

### Deliberate deviations

1. **Mutable updater → explicit arguments + carried pytree.** Java's
   `setPitchObservable` becomes a `pitch_observable` argument; the internally-held
   gravity reference becomes a `GravityRef` pytree threaded through the caller.
2. **Lazy seeding kept, made branch-free.** Java seeds its reference on first use
   (the only behaviour consistent with all 14 tests: `testPitchTiltDiagnostic...`
   asserts a full residual with no prior settle, while the sway tests require the
   filtered reference). Reproduced with a float `initialized` mask and `jnp.where`
   rather than a Python branch, per I7.
3. **`InvariantEKF` stood up as a 3-line local driver** (`_level_once`) in the
   test file. Three tests drive the Java EKF's `assembleGravityLeveling` /
   `applyGravityLeveling` / `wasLastUpdateApplied` / `getLastConditionProxy`; the
   real orchestrator is G5, and the module under test here is the updater.
   `wasLastUpdateApplied` / `getLastConditionProxy` are ported as fields of the
   returned `GravityDiagnostics` pytree (§4: diagnostics are seam surface, not
   optional logging).
4. **`pitch_disabled_var = 1e4`** chosen to satisfy `R(0,0) > 1e3` while keeping
   `S` well conditioned. Java's exact value is not observable from the test.
5. **Settling and sway loops run through `lax.scan`** — identical arithmetic and
   tick counts (3000 / 12000), but eager Python loops took the full suite from
   60 s to 119 s; scanned it is 68 s.

### Note

`apply_gravity_leveling` implements the §4 conditioning gate (Cholesky-diagonal
proxy, masked `K`, so a gated update leaves `(X̂, P)` bit-for-bit unchanged) and
computes NIS on the **prior** `P` and prior residual. Neither is exercised by
this test class — both belong to G5's `InvariantUpdaterTest` — but they are
wired now so the orchestrator has nothing left to bolt on.

---

## Config — `config/filter_cfg.yaml`

Every tuning number in the estimator now lives in one file; no module hard-codes
one. `invariant_estimation.config` loads it, and each `default_*_params` factory
takes its defaults from there while still accepting explicit keyword overrides
(so a sweep or a test needn't touch the file).

Moved into the config: the InEKF noise variances / gravity / conditioning gate /
init priors, the whole gravity-leveling block (anisotropic variances, reference
τ, the three quasi-static gate thresholds), the joint-KF sigmas, and
`group._EPS` (the small-angle branch threshold, now `numerics.small_angle_eps`).

**Deliberately left in code:** tangent indices, block layouts, group sizes. Those
are structural — changing one changes the math, not the tuning — and I4 exists
precisely to stop them from moving.

Values the Java suite pins carry a `[test-locked]` comment, and
`tests/test_config.py::test_test_locked_values_match_the_java_suite` asserts them
against the §2b constants table, which CLAUDE.md §5 designates the acceptance
checksum for the config file.

### Trap found while writing it

PyYAML implements **YAML 1.1**, in which an exponent requires an explicit sign.
`cond_max: 1.0e9` parses as the *string* `"1.0e9"`, not a float — and the failure
surfaced ~2000 lines away as a `jnp.where` dtype error inside the gravity update,
with nothing pointing at the config. `1.0e-3` is fine (it has a sign), so most of
the file worked and only two keys were broken, which is the worst case.

`load_config` now walks the parsed tree and rejects any string that `float()`
accepts, naming the key and suggesting the signed form. `tests/test_config.py`
covers both the guard and the invariant that no configured scalar is a string.

---

## G5 (partial) — `InvariantUpdaterTest` + `InvariantEKFTest`

**Status:** green. `tests/inEKF/test_invariant_updater.py` (6 Java tests → 7
pytest functions) and `tests/inEKF/test_invariant_ekf.py` (7 → 9). New module
`inEKF/ekf.py`.

**Scope, per Lucas 2026-07-21:** reseed and contact-trust are out —
see "Deferred" below. `InvariantEKFReseedTest`, `TouchdownReseedLatchTest` and
`FootSwitchContactProbabilityProviderTest` are therefore unported, and
`reseed.py` / `contact_trust.py` do not exist.

### Source change: one shared update path

`correct.py` gained `linear_update(state, H, residual, R, cond_max, gate)` —
Java's generic `InvariantUpdater.update` — plus the `UpdateDiagnostics` pytree
and `no_update_diagnostics()` (NIS initialised to **NaN**, so a never-updated
value cannot read as in-band).

`contact_update`, `correct` and `apply_gravity_leveling` were all refactored to
go through it. That was not optional bookkeeping: `testUpdateDelegatesToUpdater`
asserts the orchestrator matches the standalone updater **bit-for-bit at 1e-12**,
and the first attempt failed exactly this way — `contact_update` had picked up
the conditioning gate while `correct` still had its own inline gain/Joseph, so
the two drifted. There is now one implementation of gain / Joseph / gating / NIS.

`contact_update` consequently returns `(state, residual, diagnostics)` rather
than `(state, residual)`; the G4a call sites were updated.

### `inEKF/ekf.py`

Java's mutable `InvariantEKF` splits in two: `InvariantEKF` (immutable wiring —
contact count, params, contact noise, gravity params; built by `create`) and the
`InEKFState` carry threaded explicitly through every call (I10). The
introspection getters (`wasLastUpdateApplied`, `getLastNormalizedInnovationSquared`,
`getLastConditionProxy`, `getLastCorrectionRotationNorm`) become the returned
`UpdateDiagnostics` pytree (§4).

### Deliberate deviations

1. **`testCreateWiresConsistentSizes`'s `IllegalStateException` has no analogue.**
   Java can construct an `InvariantUpdater` without installing a `ContactUpdater`
   and throws on use; the port wires the contact path by construction, so that
   state is unreachable. The test keeps the positive half (update must not
   raise); the negative half was already adapted in G4a.
2. **`IllegalArgumentException` → `ValueError`** on `initialize`'s two shape
   contracts (contact count, covariance size).
3. **`dt` is not a per-call `predict` argument.** It is baked into the
   precomputed `Φ`, so accepting it per call would mean rebuilding a constant
   inside the loop (I7). It lives in `ekf.params`.
4. **The 4000-sample NIS mean is `vmap`'d**, not looped — same 4000 draws.

### Port-specific tests added

- `test_gated_update_leaves_state_bit_for_bit_unchanged` — a gated-out update
  (`gate=0`) must leave `(X̂, P)` bit-identical and still report a finite NIS.
  This is the §4 masked-`K` contract that G8's
  `testSingularInnovationIsSkippedNotLatched` depends on; cheaper to pin here.
- `test_update_publishes_diagnostics` — the introspection surface, including NaN
  NIS before any update.
- `test_reseed_is_not_implemented` — asserts `reseed_contact` is absent and the
  TODO is present, so the deferral cannot rot into a half-implementation.

### Deferred (Lucas, 2026-07-21)

- **Touchdown reseed.** No measurable difference on the real robot. `TODO(reseed)`
  in `ekf.py` names the call site (between `predict` and `update`) and the exact
  contract (`P_dd = P_pp + R N Rᵀ`, `P_θd = P_θp`, fire-once latch at
  trigger 0.5 / rearm 0.1 / dwell 100). Parameters parked under `reseed:` in the
  config with `enabled: false`.
- **Contact trust** (`FootSwitchContactProbabilityProvider`). Will be validated
  by comparing the Python and Java implementations on logged data rather than by
  a unit-test port. Parameters parked under `contact_trust:`.

### Housekeeping

Deleted three empty placeholder modules (`inEKF/filter.py`, `routing.py`,
`update.py` — all 0 bytes, nothing imported them). Renamed the orchestrator's
gravity entry point to `gravity_leveling_update` so the package re-export stops
shadowing the `gravity_update` *module* — the same name collision that already
forced `importlib` imports for `correct` in two test files.

---

## Scan body — `inEKF/filter.py`

**Status:** green (`tests/inEKF/test_filter.py`, 15 tests). No Java analogue —
the Java `InvariantEKF` is driven by the controller's tick, so there is nothing
to port. This is the module the MJX integration (G9/G10) and BPTT both need, and
it did not exist: `inEKF/filter.py` was one of the 0-byte placeholders.

### The tick

    propagate (IMU)  ->  contact FK update  ->  gravity leveling (gated)

`make_step(ekf, kinematics)` builds the `lax.scan` body; `run(...)` scans it.
Carry is `InEKFCarry(state, gravity_ref)` — the gravity reference has to be in
the carry because the complementary filter is stateful across ticks.

### The joint-KF boundary

`JointFilterOutput(q, q_dot, sigma_q, sigma_q_dot)` — full covariance matrices,
not diagonals: the joint KF's covariance is genuinely coupled through the mass
matrix and `J Σ_q Jᵀ` needs the off-diagonals.

`b̂` is deliberately **not** in it: the InEKF consumes already-bias-corrected IMU
(I1), so that correction happens upstream.

Routing honours the §6 forbidden edges — joint-KF outputs enter only on the
correction side, always through a kinematic Jacobian:

- `N^p_i = J_{C_i} Σ_q J_{C_i}ᵀ` — **used** (`contact_position_noise`).
- `N^v_i = J_{Ċ_i} Σ_q̇ J_{Ċ_i}ᵀ` — implemented (`contact_velocity_noise`) but
  **not consumed**. It is the noise on the contact *zero-velocity constraint*,
  which is a separate measurement block with its own `H` rows stacked below the
  position block; that constraint is an open design decision (§10), and
  inventing it here would be unasked-for cleverness in the measurement model.
  The covariance is carried across the boundary so landing it later is a change
  in `step` only. `TODO(N^v / zero-velocity)` marks the call site.

`test_joint_outputs_do_not_reach_the_propagation` asserts the forbidden edge
structurally: inflating `Σ_q` by 7 orders leaves the *predicted* `(X, P)`
bit-identical.

### `ContactKinematics` — the `robot/` seam

`(q, q̇) -> ContactFrames(y, J, J_dot)`, closed over at build time so it is static
under `jit`. MJX implements it at G1; the tests fill it with a smooth analytic
fixture. `test_fixture_kinematics_are_self_consistent` checks by autodiff that
the fixture's `J` really is `∂y/∂q` — an inconsistent fixture would make the
`J Σ_q Jᵀ` routing tests plausible-looking fiction.

### I7 — the constant-graph proof

Three tests, and this is the payoff for doing the scan body before MJX:

- `test_jaxpr_is_identical_across_contact_masks` — 5 mask patterns
  (`[1,1] [1,0] [0,1] [0,0] [0.5,0.25]`) trace to a byte-identical jaxpr.
- `test_jaxpr_is_identical_across_gate_states` — open vs closed quasi-static gate.
- `test_step_does_not_recompile_across_masks` — `_cache_size() == 1` after
  cycling four mask patterns through the jitted step.

This is the port's analogue of the two skipped Java allocation tests: *no
recompilation IS no per-tick allocation*. G9 reuses it on the fused estimator.

Masking follows §4 exactly: an untrusted contact keeps its rows but gets
`R_LARGE = 1e12·I₃` **and** a zeroed residual — never a dropped row (shape
change), never a zeroed `R` row (singular `S`, §6 trap).
`test_masked_contact_matches_excluded_contact` checks the `R_LARGE → ∞` oracle:
the masked contact's covariance block moves < 1e-6 while the trusted one moves
> 1e-4.

### Differentiability

`test_scan_is_differentiable_through_contact_covariances` takes `jax.grad` of a
10-tick scan w.r.t. the ContactNet Cholesky factors and asserts the gradient is
finite **and non-zero** — a dead path would otherwise pass silently. Same for
`Σ_q` through the joint boundary. This is the BPTT prerequisite for ContactNet.

### Correction (2026-07-22): the contact mask was wrong and is gone

The first version of `filter.py` carried a per-foot `contact_mask` that blended
`R_eff = w·Np + (1−w)·R_LARGE·I`. Lucas questioned it ("isn't the whole point
that ContactNet gives us R via the Cholesky factor anyway?"), which surfaced two
defects:

1. **The blend was a step function.** With `Np ~ 1e-6` and `R_LARGE = 1e12`,
   `w = 0.999` contributed 1e-15 of the information; only `w` within ~1e-12 of 1
   meant anything. The `[0.5, 0.25]` pattern in the jaxpr test was therefore
   testing "off, off", not partial trust.
2. **It was imported from the wrong filter.** CLAUDE.md §4's `R_LARGE` masking
   rule governs the **joint KF's stance anchors** — §2's "trusted feet → anchors"
   row, with the oracle checked in the **G7** stacked-oracle port — not the InEKF
   contact update. The InEKF's governing invariants are I2 and §7, and neither
   mentions masking.

Removed `contact_mask` from `InEKFInputs`, and `mask_contact_noise` / `R_LARGE`
from the module; they belong to the joint KF at G7.

Rationale, measured: the FK measurement is not wrong in swing — the encoders
still locate the foot relative to the base. What breaks is the static-anchor
assumption, which lives in the **process** noise. With `Σ_C = 1.0` over 100 swing
ticks, an 8 cm displacement is absorbed 96% into the anchor and perturbs the base
by 3.7 mm — a 7.6× attenuation versus the planted control. The residual base
motion is correct Bayesian behaviour (`P_pp/(P_pp+P_dd) ≈ 8%`), not a leak.

Locked by `test_large_contact_covariance_isolates_a_swing_foot`, and documented
in **`DESIGN_DECISIONS.md` §1** (new file) plus a `DECISION` block in
`filter.py`'s module docstring.

### Correction: the jaxpr tests were oversold

`contact_mask`/`contact_chol` are *traced* arrays, so the jaxpr cannot depend on
their values — jaxpr equality across input patterns is close to automatic. What
those tests genuinely catch is a **data-dependent branch** (`if x > c`,
`jnp.nonzero`, boolean indexing), which raises at trace time. Renamed and
re-worded to say what they actually prove; the load-bearing one is
`test_step_does_not_recompile_across_contact_conditions` (`_cache_size() == 1`).

### Housekeeping

`inEKF/__init__.py` had picked up duplicated import and `__all__` blocks; deduped
and now asserted clean (74 names, no duplicates, all resolving).

New file **`DESIGN_DECISIONS.md`**, linked from the README: deliberate choices
whose *symptoms* look like bugs (no contact mask, first-order `Q_d`, no reseed,
no contact-trust port, unapplied `N^v`). Each entry records what / why / what it
costs / which test guards it.

---

## G6-G8 — joint KF port (`JOINTKF_PORT_PLAN.md`)

### Phase 0 — frozen contracts

`TEST_SUITE_MAP.md` vendored into the repo. `config/filter_cfg.yaml` `joint_kf:`
now carries the full Java tuning table (`JointLevelKFPreFilter.java:70-200`),
cross-checked against the map's constants table — that table is the acceptance
checksum for the config, so a mismatch fails the port, not the test
(`tests/test_config.py::test_test_locked_values_match_the_java_suite`).

**Breaking layout change: bias is per-IMU, not per-pair.** `m` = distinct IMUs,
`dim = 2n + 3m`. Invariant I6 requires the exact `L Sigma L^T` cross-covariance
on the shared-base-IMU star, and `testBiasColumnsOfHgAreExactlyL` asserts the
bias columns of `H_g` ARE `L`, bit-identically. Under a per-pair layout two pairs
sharing an IMU carry two independent copies of one physical bias, the shared-IMU
cross terms vanish, and the G7 stacked oracle cannot pass. The five Rev.1 jointKF
modules and their tests were deleted rather than adapted: they were built to the
locked-base design and lack the Schur complement, `Lambda_eff`, the Gram-form
`Qa`, stance anchors and the `cond(S)` gate.

### Model seam: MJX, not a hand-rolled CRB

CLAUDE.md §2 already specifies MJX for production, so a hand-rolled CRB would be
a throwaway — and worse, it would make the G3 armature-equivalence oracle a
tautology (the same hand writing both sides). Two conventions verified
empirically against mujoco 3.10 and asserted in `tests/model/`:

1. A floating base's free joint occupies DoF `0..5`; hinges follow in joint
   order. The Schur nuisance gather depends on this.
2. `dof_armature` folds into `qM` as an **exact diagonal add on the hinge DoFs
   alone** — `M(armature) - M(armature=0) == diag(armature)` exactly, touching
   neither `M_bb` nor `M_jb`.

(2) is why `Lambda_eff = Lambda + diag(rotor)` falls out of the Schur complement
for free, and therefore why adding rotor inertia *again* post-Schur would
double-count the drivetrain (CLAUDE.md §6). Production takes the armature path;
`JointKFBuild.rotor_inertia` is informational only.

API note: mujoco 3.10 changed the signature to `mj_fullM(model, data, dst)`.

### Contradiction inside `TEST_SUITE_MAP.md` — resolved in favour of the shapes

The map's prose says `n = child_index - parent_index - 1`, but its own shape
table says `singlePair(10, 1, 9) -> n = 8`, i.e. `n = child - parent`. The two
disagree for every shape. **The shape table wins**: it is what the Java fixtures
actually construct, so it is what the ported tests must reproduce, and the prose
formula would change every state dimension in the suite (8->7, 4->3, 3->2).
Verified by inspection: IMUs sit on `joints.get(i).getSuccessor()`, so the joints
strictly between the two IMU links are `parent+1 .. child`, which is
`child - parent` joints. Pinned by
`tests/jointKF/test_build.py::test_joints_between_matches_the_java_shape_table`.

### Bug found in `build.py` by its own test: anchor chains root at the base IMU

First implementation rooted the base->foot anchor chain at the **world**. It must
root at the **base IMU's body**: the anchor asserts a stance foot's absolute
angular rate is ~zero, and that rate is `omega_baseIMU + J(baseIMU->foot) q_dot`,
with the base IMU's own rate read back by the `+I3` bias column. Rooting at the
world drags every joint between world and base IMU into the unfiltered `U` split,
inflating `R_anchor` with velocities the anchor equation never referenced.

Java `singlePairFootBeyondIMUs(10, 1, 5, 9)` pins it exactly: `F` = joints 2..5,
`U` = joints 6..9, and joints 0..1 appear in **neither**. Caught by
`test_anchor_chain_splits_filtered_from_unfiltered_joints`; mutation-checked by
reverting the root and confirming two tests fail.

### Mutation checks (JOINTKF_PORT_PLAN §4 — mandatory at every phase gate)

- cycle check disabled -> `test_cycle_in_the_pair_graph_is_rejected` fails.
- anchor chain re-rooted at the world -> 2 tests fail.

Both confirm the assertions discriminate rather than merely pass.

### I6 is invisible on three of the four shapes — what actually constrains it

The decisive stacked oracle (`reference_marginalized` in `tests/jointKF/_oracles.py`)
was validated before any agent depended on it: for a single pair it reduces to
the relative-gyro update with `R_g = Sigma_child + R Sigma_parent R^T` to 7.7e-12
in the mean and 1.9e-10 in the covariance. That confirms the marginalisation is
the right oracle — differencing two IMUs *is* elimination of a shared unknown
`omega_base` held under an improper prior, so the information-form limit is
forced, not chosen.

Measuring the I6 trap (block-diagonal `R_g` instead of the exact `L Sigma L^T`)
then produced a result worth recording:

| configuration | max deviation, exact vs block-diagonal |
|---|---|
| single pair, **isotropic** `Sigma = 1e-4 I` | **2.6e-20** (machine noise) |
| single pair, anisotropic `Sigma` | 1.3e-04 |
| two pairs sharing the middle IMU | 1.7e-04 (a whole cross-block zeroed) |

The reason is algebraic: for isotropic `Sigma = sigma^2 I`,
`R Sigma R^T = sigma^2 R R^T = sigma^2 I` **exactly**, so the rotation cancels and
block-diagonal is not an approximation but an identity. The fixture's default IMU
covariance is `1e-4 I3`, and three of the four `SHAPES` are single-pair — so on
those three shapes the block-diagonal implementation is not merely hard to
distinguish, it is *numerically indistinguishable*.

Only two things in the suite genuinely constrain I6:

1. `testMeasurementNoiseUsesGyroMeasurementCovariance`, which sets deliberately
   ANISOTROPIC gyro covariances (parent `diag(4e-4, 1e-6, 2.5e-5)`, child
   `diag(9e-4, 1.6e-5, 4.9e-6)`) — that anisotropy is the whole point of the
   test, not incidental realism.
2. `SHAPES[3]`, the two-pair shared-middle-IMU star, where a block-diagonal `R_g`
   drops an entire off-diagonal block rather than perturbing one.

This is JOINTKF_PORT_PLAN §4 lesson 1 in its purest form: the other tests
*exercise* `R_g` without *constraining* its structure. Any future change to the
fixture that isotropises the noise, or that drops the two-pair shape, silently
removes all coverage of invariant I6.

### The armature double-add is undetectable by the Java suite as written

CLAUDE.md §6 names the double-add (rotor inertia in the MJCF `armature` *and*
added again post-Schur) as the trap `process.py` must retire. Mutation-checking
it produced the uncomfortable answer that the Java class cannot see it.

Mechanism. The Java random chain has no name matching the rotor table, so every
joint takes the `ROTOR_INERTIA_DEFAULT = 0.005` floor, and the fixture's `M` is
well-conditioned (`lambda_min(Lambda) ~ 14`). Doubling a 0.005 diagonal is then a
~0.036% perturbation of `Lambda`; since `Qa ~ Lambda^-2`, `Qa` moves ~0.07%
— i.e. **7e-4 relative, an order of magnitude below the map's own `relTol =
3e-3`**. The double-add is arithmetically invisible at the tolerance the test
ships with.

Closed two ways:

1. A second parametrised pass over a `near_singular_fixture` — a light
   articulated mode plus `rotor = 0.167`, the real Alex `KNEE` value rather than
   the unmatched-name default. This is the regime the rotor term exists *for*:
   distal joints whose link-side apparent inertia is ~8e-4 while their
   drivetrains reflect 0.05-0.07.
2. Tolerances tightened from the map's `3e-3` (CLAUDE.md §5 explicitly permits
   this — the loosening existed because Java compares LU against Cholesky, while
   this port has a single inversion path).

Verified: with both in place the true double-add fails the primary value oracle
at 6.28e-8 against a 3.1e-10 tolerance, a ~200x margin.

API consequence in `process.py`: `rotor=` defaults to the sentinel
`ROTOR_IN_MASS_MATRIX` ("`M` already carries it, add nothing"), and `None` is
*rejected* rather than aliased to it — so "M carries the rotor" and "I forgot to
pass the rotor" cannot be spelled the same way. Adding the term requires naming
it at the call site.

---

## G1 / Phase 0b — MJX model seam

`model/mjx_model.py` (`MjxModel`, implementing `robot.RobotModel`) plus the
seeded MJCF chain fixture in `tests/jointKF/_fixture.py`, which drives the SAME
adapter as production — so the armature oracle is a genuine two-model comparison
rather than one hand writing both sides. 65 tests, covering the six required
independent-route checks (FK vs hand-rolled, Jacobians vs `jax.jacobian`, `qM`
vs kinetic energy, armature folding, symmetry/PD, consistent-motion).

### CLAUDE.md §2's armature shorthand is WRONG as written

§2 justifies the MJCF-`armature` path with "armature never touches `M_bb`/`M_jb`,
so `(M_jj + diag(arm)) - M_jb M_bb^-1 M_bj = Lambda + diag(arm)` — algebraically
identical", and G3 asserts that equivalence to 1e-12.

That holds only when armature sits on **filtered joints alone**. A **gap joint is
a nuisance DoF that carries armature**, so its rotor inertia lands in `M_bb` and
is felt through the marginalisation. Verified independently:

| configuration | `max abs( Schur(M+diag(a)) - [Schur(M) + diag(a_j)] )` |
|---|---|
| armature on filtered joints only | **0.0** (exact) |
| gap joints also carry armature | **3.5e-3** (1.2e-4 relative) |

**This is not hypothetical for Alex: the ankles ARE gap joints** (unfiltered,
because there are no foot IMUs) **and they DO carry rotor inertia** (`ANKLE_Y
0.07`, `ANKLE_X 0.05` in the table). So the G3 oracle as specified would fail on
the real robot, not because the port is wrong but because the identity it asserts
is false in that configuration.

Note §2 is not self-inconsistent — its own parameter table already says "nuisance
rotor diag on gap joints, zero on base 6 DoF", i.e. Java *does* carry rotor on
gap joints. MJX and Java agree; it is the prose shorthand that overreaches.

**Consequence for `process.py`:** `Lambda_eff = Lambda_bare + diag(rotor)` must
not be written as a general identity. Both pinned as separate tests: (a) the
shift applies to both partitions; (b) with armature on filtered joints only it
*is* an exact post-Schur diagonal add.

### `mass_matrix` had two spellings, one tested and one used

Mutation M3 (`+1e-6*I` on `mass_matrix`) left the entire model suite green: every
oracle read `evaluate()`, while `mass_matrix` recomputed `qM` by an independent
route. Two spellings of one quantity — the tested one and the used one — which is
CONTRACT_CARD §8's failure mode exactly, and was found only because the mutation
was actually run. Fixed by routing both through a single `_mass_matrix(d)`, plus
`test_accessors_agree_with_the_single_pass`.

### MJX conventions and a G9/G10 risk

- `mjx.jac(...)` returns `(nv, 3)` — **transposed** relative to the MuJoCo C API.
- `d.site_xmat` is already `(nsite, 3, 3)` in MJX (a flat 9-vector in the C API).
- mujoco 3.10 changed the signature to `mj_fullM(model, data, dst)`.
- MuJoCo rejects `diaginertia` violating the triangle inequality; the fixture
  parameterises principal moments as pairwise sums.

**XLA CPU compile time explodes with kinematic depth**: jitting the position
pipeline costs ~1.3 s at 4 links, ~1.7 s at 6, and **~240 s at 10** (isolated to
`mjx.kinematics`, ~336 s). Eager `vmap` over 20 configs is ~7 s regardless, so
the fixture batch is eager-vmapped and jit-ability is asserted separately on the
shallowest shape. **This is a live risk for G9/G10**, where the jitted fused step
must run against full Alex; `lax.scan` over bodies rather than MJX's unrolled
per-level tracing is the likely lever. Flagged, not solved.

### Open question for B1

Which frame `b_omega` lives in is **not** constrained by anything ported so far.
The seam chose the **child site frame** for `relative_gyro_jacobian` (Java's
`GeometricJacobianCalculator` convention, and `state.py` stores bias per-IMU in
its own frame). The stacked oracle is what will actually decide it.

---

## G7 — stacked gyro measurement + stance anchors

`jointKF/measure.py` (encoder rows + stacked pair rows + `L`) and
`jointKF/anchors.py` (stance-anchor block, F/U split, masking). 171 tests green
in `tests/jointKF`.

### `b_omega` lives in each IMU's own measurement (site) frame — confirmed

This was flagged as an open question after Phase 0b, and it is now settled by
evidence rather than convention. `test_stacked_pair_rows_match_the_marginalized_raw_gyro_reference`
composes the built `(H, z, R)` through `reference_update` and compares against
`reference_marginalized`, which independently places `+I3` on each IMU's bias in
that IMU's own frame and carries `omega_base` on a rotation column. They agree on
both the single-pair shape and the two-pair shared-IMU star. Mutating the parent
block to `-I3` or to `R^T` moves the posterior by **1.99** and **1.45** — O(1)
signals, seven orders above tolerance.

Tolerance there is 1e-8 rather than 1e-12, and the mechanism was found before the
number was set (plan §4 lesson 2): the oracle's nuisance carries a *zero*
information block, so `cond(Lambda) ~ sigma^-2`. Sweeping the gyro STD gives
1e-2 -> 8e-11, 1e-3 -> 2e-8, 1e-4 -> 5e-7, 1e-5 -> 5e-4, 1e-6 -> 1e-2 — exactly
`eps * cond`, i.e. the oracle's own arithmetic, not the port's error.

### I6 coverage, corrected: anisotropy does NOT constrain the cross-block

The earlier note in this file said the anisotropic-`Sigma` test was one of two
things constraining invariant I6. **That was wrong**, and B1's per-shape mutation
disproved it. For a *single* pair, `L Sigma L^T` **is** `Sigma_c + R Sigma_p R^T`
— the per-pair block-diagonal form — at *any* anisotropy. So:

- anisotropy constrains the **rotation** inside the congruence (it catches a
  dropped or transposed `R_parent`, mutations c and d, at O(1));
- only a **shared IMU** constrains the **cross-block**.

Mutating `R_g` to block-diagonal fails exactly two tests, both on `SHAPES[3]`,
the two-pair shared-middle-IMU star. **I6's cross-covariance rests on that single
shape.** Delete it, or reduce the fixture to single pairs, and the invariant
loses all coverage while the suite stays green.
`test_isotropic_single_pair_cannot_constrain_i6` now asserts the blind spot
itself, so it fails loudly if the fixture is ever isotropised further.

### CONTRACT CONFLICT, found and fixed: `R_LARGE` vs `cond_s_max`

CLAUDE.md §4 sets `R_LARGE = 1e12` for an inactive anchor and `cond_s_max = 1e9`
for the conditioning gate. These are **mutually destructive**. An inactive
anchor's `R` block is structurally decoupled (its `H` rows are zeroed), so its
Cholesky diagonal is exactly `sqrt(R_LARGE)` and
`cond(S) >= 1e12 / lambda_min(pair block) ~ 4e11` — an order above the gate.

Consequence: **every tick with any foot in swing dropped the ENTIRE stacked
update, gyro rows included.** The filter would have stopped updating for the
whole of walking, reporting nothing worse than `was_applied = 0`. Java never
meets this because its stacked measurement literally has no anchor rows when no
foot is trusted; the fixed-shape port must say the same thing with a mask, and
the two constants as configured cannot both hold.

Fixed in `update.py` by computing the condition proxy over **informative rows
only** (`diag(R) < 0.5 * r_large`). This is the gate's own semantics rather than
a fudge: the gate exists to catch an `S` that inverts to a *huge* gain, and a row
we have deliberately declared uninformative contributes gain `~1/R_LARGE ~ 0` —
it is the safest row in the matrix, not the most dangerous. Deriving the mask
from `R` rather than an added argument keeps the property true for any caller
that follows the masking rule, with no plumbing to forget.

Found because B2 recorded it as a `strict` xfail rather than working around it,
so fixing it XPASSed and forced the marker's removal.

### Mutation findings

- **`sqrt(sigma)` instead of `sigma` in the anchor congruence** (a 10x inflation
  of `R_anchor`) passed every other test in `test_bias_observability.py`. The
  trace threshold and the tighter-vs-looser comparison constrain the
  congruence's *presence*; neither constrains its *magnitude*. Closed with an
  independent NumPy value oracle; now caught at 1.66e-1.
- **`+I3` on the wrong IMU's bias columns** is NOT caught by the map's own gauge
  tests, because `norm(R(imu <- W) beta) == norm(beta)` for *any* IMU. Only the
  beyond-the-map structural tests (which the plan explicitly required) catch it.

### Open, and the most likely Phase-3 mismatch source

`R_anchor` omits the base IMU's own gyro noise and the anchor<->pair
cross-covariance. After eliminating `omega_base` the anchor row *does* inherit
`Sigma_base` and *is* correlated with every pair row through it. Java models
neither, and CLAUDE.md §2 specifies `Sigma_eps + J_U diag(sigma^2) J_U^T`
exactly, so the port follows the spec — but `Sigma_base ~ 1e-4` against
`anchor_var = 4e-4` is not negligible. Expect this to show up when the Phase-3
stacked oracle runs with feet active.

Also open: anchor masking is applied in both `measure.build_stacked` and
`anchors.anchor_block` (idempotent, but ownership should collapse to one side),
and two `AnchorBlock` definitions now exist.

---

## Phase 3 — the tick (`jointKF/filter.py`)

`predict -> encoder update -> stacked gyro/anchor update`, plus a `lax.scan`.
7 tests. No single Java class corresponds: the Java suite exercises the
orchestration behaviourally (G8), so what is tested here is the wiring the port
must get right *because* it is fixed-shape and pure — the parts that have no Java
analogue because Java simply reshapes.

Three decisions live in this file rather than being distributed, because no
single module can see them:

1. **The trusted-feet mask is delayed by exactly one tick.** The contact signal
   derives from the same sensors the filter is about to consume, so using *this*
   tick's mask would correlate the gating decision with the measurement it gates
   through the shared noise — biasing the very bias estimate the anchor exists to
   make observable. Carrying it in the scan state is also what keeps it a value
   rather than a Python-level decision (I7).

2. **Encoders and the gyro/anchor stack are two sequential Joseph updates, not
   one concatenated block.** Algebraically equivalent when the noises are
   independent (they are), but completely different *under gating*: one stacked
   update means a single ill-conditioned gyro row throws the encoders away too. A
   foot in swing, or a NaN on one IMU, must cost only the channel that went bad.
   The `cond(S)` gate makes this behavioural, not numerical.

3. **Model quantities are arguments, not an internal `RobotModel` call.** Keeps
   the filter simulator-free (matching the InEKF's seam discipline), and leaves
   the caller free to evaluate once per tick, batch under `vmap`, or precompute —
   which matters because MJX tracing cost grows sharply with chain depth.

### Mutation checks

| mutation | caught by |
|---|---|
| drop the one-tick delay (use this tick's contact) | 2 tests |
| merge the two channels into one stacked update | `test_a_nan_gyro_does_not_cost_the_encoder_update` |

The second is the one worth noting: with a merged update, a NaN on a single IMU
silently costs the encoder update as well, and *nothing else in the suite
notices* — the state stays finite and PSD, it is simply less informed than it
should be.

### A constant-graph test that measured the test session, not the code

`test_step_does_not_recompile_across_contact_patterns` asserted
`jstep._cache_size() == 1`. It passed alone and inside its own file, and failed
only under the full suite — the shape of failure that usually gets "fixed" by
loosening something (plan §4 lesson 2), so the mechanism was found first.

`_cache_size()` read **0**, not 2. JAX's jit cache is a global LRU, and the other
177 tests in `tests/jointKF` evicted the entry. The assertion was measuring a
shared resource other tests pollute: it can fail with no retrace whatsoever, and
eviction means it says nothing about the property it claims to.

Replaced with a comparison of the **lowered program** (`lower(...).as_text()`)
across contact patterns, which is immune to eviction, plus `_cache_size() <= 1` —
a retrace can only push the cache *above* one entry, so that direction survives a
full-suite run while the equality did not. Mutation-checked: introducing a
data-dependent Python branch on `carry.trusted_feet` raises
`ConcretizationTypeError`, which is the failure this test exists to catch.

---

## G8 (part) — encoder NIS, the direct-velocity channel, singular-innovation attribution

`velocity.py` (the optional direct-q̇ channel, default OFF) and `diagnostics.py`
(`per_joint_nis`, `describe_singular_innovation`). 19 tests, 5.7 s — all pure
linear algebra on `stub_build`, no MJX.

Java dispatches the two channels on an exact-match label string
(`"encoder"` vs `"encoderVelocity"`). Strings cannot cross a jit boundary, so the
port carries the **observable** instead: separate fields in a `ChannelDiagnostics`
struct, which makes it *structurally* impossible for the velocity channel to
write into the position channel's NIS. The cross-talk guard tests that.

### "NIS on the posterior" — CLAUDE.md §6's named trap — survives the Java tests

Mutating `S` to be built from the *posterior* `P` rather than the prior left
**both** 4000-trial chi-square tests green. The mechanism is arithmetic, not
luck: the shift is `1.25/1.20 = +4.2%` of the mean, against a 4-sigma envelope of
`4*sqrt(2/4000) = 8.9%`. The statistical test cannot resolve a bias half the size
of its own envelope, and raising the trial count to fix that would need ~4x more
samples.

So the trap CLAUDE.md warns about is invisible to the class the suite provides
for it. Closed with a deterministic test asserting `UpdateInfo.S` equals the
prior innovation covariance to 1e-15 — which catches it immediately, because the
quantity is exactly specified even though its statistical consequence is not.

### The Java lag-inflation test exercises the smoother without constraining it

Mutating `dhat` to a **raw** finite difference (no 5 Hz low-pass) passed all
three phases of `lagInflationTracksMeasuredSlewExactly`. On a *noiseless*
constant/ramp/constant signal, raw and smoothed finite differences agree — and
the Java scenario is exactly that signal. But the smoother exists precisely
because the measurement is noisy: raw, the FD variance `2 sigma^2/dt^2` inflates
`R` by ~2 orders at quiet standing, which is the regime the channel exists for.
Caught only by an added noisy-standing test, where the mutation inflates `R` by
**519x**.

### Recorded, not a bug

Java's `testDiagnosticNamesTheDegenerateGyroPair` scenario (`R = 1e-6 I`, one
duplicated row) yields `cond(S) ~ 5.2e6` — *below* `cond_s_max = 1e9`, so that
measurement would be **accepted**, not gated. Java never claims otherwise (it
calls `describeSingularInnovation` directly rather than through the filter), so
the port is consistent with it; but it means the diagnostic's own scenario is not
a gating scenario. An added test tightens `R` to `1e-12 I` to make the
end-to-end statement the diagnostic is actually for.

### Follow-ups requested by the agent (parent-owned files)

- `UpdateInfo` should carry `nis_per_row`. The point of the encoder NIS
  diagnostic is to localise a bad encoder to a *joint*, which the aggregate
  scalar cannot do; the tests currently compute it themselves, which is a smell.
- `filter.TickDiagnostics` publishes `encoder_nis` as a scalar and does not use
  `ChannelDiagnostics`. The cross-talk observable only holds once the encoder
  channel writes the encoder half and nothing else does — today nothing writes it.

### G8 behaviour — mutation checks (completed after the agent was paused)

| mutation | caught by |
|---|---|
| (a) NaN sanitisation removed from `update.py` | **NOT** by either behaviour file — only by `test_update.py::test_gradients_are_finite_through_a_non_finite_measurement[H]` |
| (b) anchor block zeroed (base bias unobservable) | `test_whole_filter.py::test_stance_phase_bias_convergence` |
| (c) stacked-measurement velocity columns zeroed | `test_bias_stays_small_with_zero_true_bias`, `test_covariance_embeds_kinematic_coupling` |

Two of the three were caught by a *different* test than expected, and both
discrepancies are worth keeping:

**(a) The NaN sanitisation is load-bearing for gradients, not for values.** The
forward pass is already safe without it, because `jnp.where(applied, ...)` selects
the clean prior once the finiteness gate fires — so no NaN reaches `(x, P)` and
every behavioural NaN-hardening assertion passes. What breaks is reverse-mode:
the untaken branch of a `jnp.where` still propagates NaN into the cotangent
(`NaN * 0 = NaN`), so the *gradient* goes non-finite. That matters here rather
than being academic — CLAUDE.md §7 requires **no `stop_gradient`** between the
ContactNet provider and either filter, so this filter is differentiated through
during BPTT training. A behaviour-only test suite cannot constrain this.

**(c) Velocity convergence does not require the gyro velocity channel.** The
`test_velocity_converges` clause (including the `max|v_est| > 0.5 * peak` guard
that exists specifically to check velocity is *observed*) still passes with the
stacked measurement's `q_dot` columns zeroed: the double integrator plus encoder
positions lets the filter infer velocity by differencing, with no gyro
contribution at all. What actually fails is the bias estimate (which drifts to
|b| ~ 1.8) and the cross-joint velocity correlation (the shared Jacobian is the
only thing that could induce it). So the guard clause constrains observability of
velocity *somehow*, not observability *through the gyros*.

---

## Phase 3 — the decisive stacked oracle, and the bug it found

`tests/jointKF/test_stacked_oracle.py`: the feet-active half of
`JointLevelKFStackedOracleTest` (12 trials, tol 1e-5). The pairs-only half lives
in `test_measurement.py`. The plan reserves this one for the parent because it is
where `measure.py`, `anchors.py` and `update.py` must *compose*, and a wrong
answer is not locally detectable by any single one of them.

It earned its keep: **12/12 trials failed on first run**, off by 2e-4..9e-4
against a 1e-5 tolerance.

### `R_anchor` was missing the base IMU's own gyro noise, and every cross-term

The reference's base-IMU row has a zero joint Jacobian, so
`z_base = omega_base + b_base + v_base` with `v_base ~ N(0, Sigma_base)`, and its
anchor row asserts `J_leg qdot + omega_base = v_anchor` with
`v_anchor ~ N(0, Sigma_eps)`. Eliminating `omega_base` between the two gives the
port's anchor row exactly:

    z_base = -J_leg qdot + b_base + (v_anchor - v_base)

So the anchor row's noise is `Sigma_eps + Sigma_base`, **and** it is correlated
with every pair row that touches the base IMU — through that same shared
`v_base`. CLAUDE.md §2 specifies `R_anchor = Sigma_eps + J_U diag(sigma^2) J_U^T`,
which drops both terms; Java does the same, and B2 flagged the omission when
porting it (`Sigma_base ~ 1e-4` against `anchor_var = 4e-4` is not negligible).

**Fix**: run the `L Sigma L^T` congruence over the WHOLE stacked measurement
rather than over the pair block alone, and *add* the anchor's slip noise on top
instead of substituting it. Both missing terms then fall out for free, because
`L` already carries the anchor's `+I3` on the base-IMU bias columns — B1 had
built `L` that way so `H[:, 2n:2n+3m] == L` would hold for the whole stacked
Jacobian, which turned out to be exactly the structure the correct noise model
needs. This is invariant I6 extending to the anchors: the congruence is the
model, and anything assembled block-by-block loses the correlations it encodes.

Masking is unaffected: an untrusted anchor has its `H` rows zeroed, so its `L`
rows are zero, the congruence contributes nothing, and the block stays exactly
`r_large * I3` with zero cross-terms — which is also what keeps it structurally
decoupled for `update.py`'s informative-row condition proxy.

**This is a genuine correctness improvement over the Java implementation**, not a
port artefact. The Java filter under-states its anchor noise by `Sigma_base` and
treats the anchor as independent of the gyro rows it is derived from. The
consequence is over-trusting the anchor, which feeds the base gyro-bias estimate
the downstream InEKF integrates directly into orientation — the exact quantity
`SIGMA_QD_UNFILTERED`'s "erring large is safe, erring small is not" comment warns
about.

---

## Phase 5 — sign-off

`uv run pytest -q`: **592 passed**, 0 failed (~5.7 min). Baseline before this work
was 459. `tests/jointKF` + `tests/model` contribute 288 collected tests from 154
test functions (the rest is parametrisation over the four fixture shapes).

### Gate status

| gate | status |
|---|---|
| G1 model layer | green — 65 tests, six independent-route checks |
| G2-G5 InEKF | green (pre-existing, unchanged) |
| G6 joint-KF linear-algebra core | green |
| G7 measurement + anchors | green |
| G8 behaviour | green |
| Phase 3 decisive stacked oracle | green, both halves |
| G9 pipeline, G10 sim/eval | not started (plan §9) |

### Ported Java classes

`JointLevelKFStateTest`, `PredictTest`, `UpdateTest`, `MeasurementTest`,
`TransitionNoiseTest`, `MassMatrixNoiseTest`, `RotorAndGramTest`,
`StandingStabilityTest`, `BiasObservabilityTest`, `EncoderNISConsistencyTest`,
`DirectVelocityMeasurementTest`, `SingularInnovationDiagnosticTest`,
`StackedOracleTest`, `FilterTest`, `TrajectoryTest`, plus
`testHotPathStaysFinite`.

**Skipped as documented:** both `*AllocationTest` allocation tests (JVM
`ThreadMXBean`). Their analogue here is the constant-graph check — no
recompilation IS the port's "no per-tick allocation" — measured by comparing the
lowered program rather than a cache counter.

### The pattern worth carrying forward

Eleven tests in this port were found to pass against wrong implementations. They
are individually documented above; collectively they have a shape:

**This suite reliably verifies that a term is PRESENT and reliably fails to
verify that it is RIGHT.** The recurring mechanisms are

* **algebraic degeneracy in the fixture** — isotropic `Sigma` makes
  `R Sigma R^T = Sigma`, so a block-diagonal `R_g` is an identity, not an
  approximation; a well-conditioned `Lambda` with the 0.005 default rotor makes
  the armature double-add a 7e-4 perturbation;
* **statistical envelopes wider than the bias** — "NIS on the posterior" shifts
  the mean 4.2% against an 8.9% envelope, so 4000 trials cannot see it;
* **noiseless deterministic scenarios** — the lag-inflation test's
  constant/ramp/constant signal cannot distinguish a smoothed finite difference
  from a raw one, because the smoother only matters under noise;
* **oracles that share the implementation's assumption** — the Joseph reference
  KF also uses Joseph form, so the short form matches it exactly at the optimal
  gain;
* **two spellings of one quantity** — `mass_matrix` vs `evaluate()`, one tested
  and one used.

Three of these are traps the repo-root `CLAUDE.md` §6 names explicitly (the
armature double-add, block-diagonal `R_g`, NIS on the posterior) and **the Java
suite cannot detect any of them as written**. Naming a trap in prose is not the
same as testing for it; that gap is the single most useful thing this port found.

Practical consequence for future work: when adding a test here, state what it
*constrains*, not what it exercises, and mutate the source to confirm — the
`CONTRACT_CARD.md` §8 discipline. It caught ten of the eleven; the eleventh (the
jit-cache one) was caught only by running the full suite in a different order.

### Carried forward, unresolved

1. **XLA compile time vs kinematic depth** — ~1.3 s to jit a 4-link chain, ~240 s
   at 10 links, isolated to `mjx.kinematics`. A live risk for G9/G10 against full
   Alex. `lax.scan` over bodies rather than MJX's unrolled per-level tracing is
   the likely lever.
2. **`alpha_overrides` are calibrated for Alex's 9 filtered joints**; any other
   robot takes the 0.15 default and will surface through the `QA_MAX` tripwire,
   by design.
3. **`encoder_pos_std`, `encoder_vel_std`, `gyro_sigma` sidecars are empty** —
   every joint currently takes the loud fallback, which `build.py` warns about at
   construction. Wiring them is a config task, not a code one.
4. **The direct-velocity channel is default OFF** and has no config key for the
   drive corner frequency (it is a builder argument).

---

## Replay parity — `diag(Qa)` vs the 2026-07-17 Alex001 log

`jointKF_QaDiag_<joint>` is a pure function of `q`, so it can be recomputed tick
by tick with zero error accumulation. It was reading **14.4% high on the legs and
14.4% LOW on the spine** — a stable, opposite-signed ratio that ruled out every
scalar explanation. `alpha`, `tau_max` and the rotor table were byte-compared
against `JointLevelKFPreFilter.java`; tick offset and `q` source were swept
(<0.3%); `Lambda` was proved base-pose invariant to 9e-15. All of it lived in
`M(q)`.

### Root cause — Mecano freezes the lumped subtree inertia at construction

`CompositeRigidBodyMassMatrixCalculator(input, frame, considerIgnoredSubtreesInertia)`
defaults the flag to `true`, so ignored subtrees *are* composited rather than
dropped — but `rootCompositeInertia.updateIgnoredSubtreeInertia()` is called from
the **constructor and from nowhere else**. It stores
`MultiBodySystemTools.computeSubtreeInertia(childJoint)` as a plain
`SpatialInertia` in the parent's body-fixed frame, and `computeMassMatrix()` then
reuses that stored value every tick (`bodyInertia = rigidBody.getInertia() +
bodySubtreeInertia`). The ignored subtrees are therefore welded at whatever pose
the robot model held **when the estimator was built** — on hardware, the
freshly-constructed model, `q = 0`.

On this log the arms sit near `(shoulder 0.71, elbow -1.91)` rad and the ankles
near `-0.40` rad **from the first tick**, so `q = 0` is never the live pose.
Evaluating `M` with the off-path joints held at zero instead of live collapses
the error from 14.42% to 0.21%. Fitting a scale `s` on the off-path angles gives
a sharp unique minimum at `s = 0`:

| `q_ignored` | 0 | 0.05·q | 0.1·q | 0.25·q | 0.5·q | q |
|---|---|---|---|---|---|---|
| max rel err | **0.21%** | 1.59% | 3.09% | 7.06% | 11.68% | 14.42% |

Freezing only the arms leaves 9.8%; only the ankles, 14.9% — both groups matter,
and they push the spine and the legs in opposite directions, which is exactly the
sign structure the log showed.

`MjxModel.qpos` already reproduced this by construction (off-path joints keep
`qpos0`); the defect was in `tests/replay/test_java_parity.py`, whose oracle
filled all 29 hinges from the log. The behaviour is now documented as
load-bearing at `MjxModel.qpos` rather than left as an incidental default.

### Second, independent bug — the nuisance set was two different wrong sets

Java marginalises the base plus *gap* joints, where a gap joint is on a
`root -> filtered` path without being a filter state (`collectSpanningJoints`
minus the filtered set). On Alex there are **no** gap joints, so the nuisance
block is exactly the base 6 DoF. Two places disagreed:

* `MjxModel.dof_nuisance` was `setdiff1d(range(nv), joint_dof)` — all 26
  non-filtered DoFs. Off-path joints are *locked*, not marginalised;
  eliminating them models the ankles and arms as free to accelerate. Worth
  **58%** on `diag(Qa)` (at frozen `q`; 38% at live `q`, which is how it had been
  measured before and why it looked merely "wrong family").
* `build.py` appended the **anchor chain's** unfiltered joints (Alex's ankles) to
  `dof_nuisance`, conflating two genuinely different sets: the ankles are
  anchor-chain-unfiltered but off the root->filtered paths. Worth **1.7%**.

The conflation was invisible to every unit test because the route-1 serial-chain
fixture has *no* off-path joints — there, anchor-unfiltered and gap coincide.
`build.py` now computes gap joints properly and publishes the anchor set
separately as `JointKFBuild.dof_anchor_unfiltered`; `anchors.unfiltered_dof`
prefers it and keeps the old trailing-slice read only as a fallback for
hand-built fixtures.

### Residual, and whether Java is right

After both fixes, per joint over 1000 consecutive ticks: mean ratio within
**0.07%** of 1, spread ~0.05%, worst tick 0.21%. Not reducible by a tick shift
(the -1/0/+1 sweep moves the RMS between 0.039% and 0.086%), so it is sub-tick
sampling of `q` between the log decimation and the filter's own rate.
`QA_REL_TOL` in the replay suite dropped 0.15 -> 0.005.

A 0.2% agreement across 9 joints and 400 ticks also settles the open question of
whether `AlexRobotModel` and the log's `model.sdf` are the same description: at
this level of agreement on a quantity that depends on every link mass, inertia
tensor and joint origin in the legs and torso, they are.

**Java is stale here, and knowingly reproducing that is a decision.** The
physically correct lumped inertia would track the live off-path configuration;
Mecano's is fixed at `q = 0`, so the Java filter's `Qa` is off by up to 14%
whenever the arms are not at zero — which, on Alex, is always. This is a genuine
(mild) bug in `JointLevelKFPreFilter`, not in Mecano: the calculator's contract
is that ignored subtrees are static, and the estimator ignores subtrees that
move. The port matches Java because the replay suite measures parity; if the
Python filter is ever run as the primary estimator, feeding live off-path angles
is the better model and this note is the record of why the two differ.

---

## Hardware-log parity harness (`tests/replay/`) and the encoder-noise config gap

The `replay/` module reads a real SCS2 hardware log and compares this port
against what the Java estimator *actually published* on that run — the
acceptance test the whole port exists to pass. See `RUNNING.md` for how to point
it at a log. It is two tiers; only tier 1 (stateless, exact) is built so far.

### The encoder-noise gap this surfaced (a real fix, not just a test)

`config/filter_cfg.yaml` shipped with `encoder_pos_std: {}` and
`encoder_vel_std: {}` (both `TODO(Lucas)`), so **every filtered joint fell back
to the scalar `encoder_var = 5e-5`** — an implied std of 7.1e-3 rad, which is
15x–48x larger than the measured per-joint values. A filter under-trusting its
encoders by 2–3 orders of magnitude in variance leans far too hard on the
IMU/model side and exports a silently wrong `Sigma_q`.

Both tables are now filled from `AlexSensorNoiseParameters.java` (Lucas,
2026-07-15 — the authoritative source, not reverse-engineered from the log). The
2026-07-17 Alex001 log then *validates* them exactly:

* `jointKF_encR_<joint>` (the Java filter's per-joint position R) equals
  `encoder_pos_std**2` to rtol 1e-3, and is constant across the whole 630 s run.
* `jointKF_qdR_<joint>.min()` equals `encoder_vel_std**2` — the direct-velocity
  channel's lag-inflated R bottoms out at its `sigma**2` floor when the joint's
  measured slew passes through zero. (The channel was ON in this flight:
  `jointKFUseDirectVelocityMeasurement = 1`.)

`tests/replay/test_sensor_noise.py` locks both, and fails loudly if the config
ever drifts back onto the fallback. The lookups (`state.encoder_var_for_name`,
`velocity.velocity_var_for_name`) were made case-insensitive (`state._ci_get`) to
match Java's `getEncoderPositionNoiseStandardDeviation` (`.toLowerCase()`),
because the source table keys are lowercase and the MuJoCo joint names are upper.

### IMU-mount convention

`tests/replay/test_imu_mount.py` checks the one rotation most likely to be built
wrong: the pelvis IMU mount (yawed +90 deg on Alex). The log publishes the
applied gyro bias in both the IMU and pelvis frames, and `urdf2mjcf`'s
`rpy -> quat` must map one onto the other. Residual 8e-19 vs a ~5e-3 signal —
the fixed-axis convention is correct.

### Model source

The port converts the `model.sdf` that ships *inside each log directory*
(`model/urdf2mjcf.py`), not a vendored MJCF: it is by construction the exact
description that ran, and the vendored `alex_v1_full_body_mjx.xml` both is a
different build and currently fails to compile (zero-eigenvalue inertias on the
massless sensor frames). The 19 zero-mass IMU/ZED frames become inertia-less
placeholder bodies (mass 1e-12); their contribution to `M(q)` is below float64
noise (asserted).

### Not built, and why

* **Gravity-leveling gate vs log** — `invariantGravityUpdateActive` is
  reproducible in principle (it is `enabled && isQuasiStatic(a, omega, ...)`), but
  faithfully requires the *processed* body-frame specific force the sensor
  pipeline produces, which the port does not replicate. A raw-accel approximation
  reproduces it to only ~80%, which is not a clean oracle. The gravity path is
  already covered by the 14 ported `GravityLevelingUpdaterTest` unit tests; a
  log-based gate oracle waits on a specific-force replay (tier 2).
* **Tier 2 free-running replay** — seed from Java's state, integrate, compare
  trajectories. Deferred: the log carries only the covariance *diagonal* (via the
  `_upperBound`/`_lowerBound` pairs), so a clean per-tick re-seed of `P` is
  impossible; tier 2 is a divergence test, weaker than tier 1's per-module
  localisation, and best built once the `diag(Qa)` and sensor-noise oracles are
  green (they now are).

## G9 — the fused estimator step (`pipeline/main_estimator.py`)

G9 fuses the two already-validated filters into one constant-XLA-graph `lax.scan`
body: joint KF (at the carry's `q̂_prev`) → `(q̂, q̇̂, Σ_q, Σ_q̇, b̂)` → the boundary
→ InEKF → pelvis pose. The only genuinely new code is `_boundary`: bias-correct
the base gyro in the IMU frame (I1), rotate IMU→body (`R_mount`), route the full
`Σ_q`/`Σ_q̇` (never diagonalised). Gate: `tests/pipeline/test_main_estimator.py`
(9 green) — the jaxpr-constancy proof (I7) + a `_cache_size()==1` no-recompile
check + five scenarios (static equilibrium, no-contact yaw integration, free fall,
poisoned-encoder recovery, bias-correction wiring), all against a synthetic
floating-base biped MJX model.

### Deviations and decisions (DoD §8)

* **The build guide's premise was stale.** `~/Documents/g910_guide.md` says to
  fuse two Tier-2 replay drivers `replay/{jointkf,inekf}_driver.py`; those files
  do not exist (the harness is Tier-1 stateless, `tests/replay/`). The real
  assembly patterns are `tests/jointKF/_fixture.py::kinematic_tree` (the
  mj_model→`KinematicTree` adapter, generalised here as `kinematic_tree_from_mj`)
  and `tests/inEKF/test_filter.py` (the contact-FK closure + jaxpr-constancy
  pattern). The guide's `jkf.step(..., vel_ch)` 6th arg and `SensorInputs.velocity`
  field also do not exist — the direct-velocity channel is not wired into the
  current `jointKF.filter.step`, so the fused step does not use it.
* **G9 gate is synthetic, not the hardware replay.** The guide's stronger "diff
  the fused step against `jointKF_*`/`invariantFilter*` on the log" gate needs the
  9GB Alex001 log + `ihmc-log` skill (CI-excluded). Deferred to the same Tier-2
  replay above; the synthetic scenarios are the self-contained gate.
* **Landmine #1 (Q_bb=0) is wired.** `build_fused_estimator(imu_bias_process_var
  =0.0)` overrides the config's test-locked `1e-4` at the fusion boundary (flight
  value; config value would make the joint-KF bias ~200× too noisy).
* **Landmine #2 (contact measurement-noise floor) is a socket, default off.**
  `contact_meas_var` (default `0.0` = current port behaviour) adds an isotropic
  floor to `Σ_q` before the InEKF contact update, standing in for flight's
  `ConstantContactMeasurementNoiseProvider`. It affects velocity/position, not
  roll/pitch; wiring the flight value waits on the Tier-2 velocity check.
* **`R_mount` unverified for real Alex.** The synthetic model's base IMU is
  axis-aligned, so `R_mount=I`. Real Alex is a +90° pelvis-IMU yaw; the frame
  step (`_boundary`) MUST be cross-checked against `invariantRootAngularVelocityBody*`
  on the log before trusting fused velocity/position. Roll/pitch are `R_mount`-robust.
* **Benign one-time recompile fixed by `device_put`.** The init carry mixes device
  commitment (contacts `d0` come off an MJX-FK `einsum`, committed; `jnp.eye`/`zeros`
  leaves uncommitted), which forced a second `fused_step` compile with an identical
  jaxpr but a different `Argument mapping`. `init_fused_carry` now `device_put`s the
  whole carry to one device, so it is a single executable. This is not an I7
  violation (contact/gate flips do not recompile); it is sharding bookkeeping.
* **`J_dot = 0` in the contact-FK closure.** The InEKF velocity-noise term
  (`N^v = J_Ċ Σ_q̇ J_Ċᵀ`) is deferred, same as the standalone `inEKF/filter.py`
  TODO; `Σ_q̇` is carried through the boundary so adding it later is local.

### G9 on the real Alex model (2026-07-23) — wired and frame-verified

`fused_step` now runs on the actual 2026-07-17 Alex001 model (the log's own
`model.sdf`), not just the synthetic biped. `build_alex_fused_estimator` +
`ALEX_*` in `main_estimator.py` encode the topology; `tests/replay/
test_fused_real_model.py` is the gate (skips without the log). Three things fell
out, one of them a real bug fix:

* **Resolved the `imu_pairs: []` TODO (CLAUDE.md §2b).** The IMU set is *forced*
  by two facts from the log: `jointKFNumberOfIMUs = 8`, and the pair-chain union
  must equal the 9 logged `FILTERED_JOINTS` (spine + legs, no arms/head). The only
  set that satisfies both is a **star on `pelvis_imu`** paired with `torso_imu`
  and each leg's `hip_x / thigh / shin` IMUs. Verified: it reproduces exactly the
  9 filtered joints, `dof_nuisance` is base-6-only (no gap joints — matches the
  `diag(Qa)` parity harness), and the ankles fall out as the 4-joint unfiltered
  anchor split. The leg IMUs give overlapping chains (`hip_x ⊂ thigh ⊂ shin`) —
  the redundant shared-base-IMU measurement the `LΣLᵀ` star (I6) is for.

* **Fixed a frame bug in the fused step's body-frame model.** The first cut used
  the base *IMU site* as the InEKF body frame `B`. That is only correct when the
  IMU sits at the body origin with no rotation (true for the synthetic fixture,
  false for Alex: the pelvis IMU is offset AND yawed +90°). The InEKF's `B` is the
  *pelvis root body* (what `invariantRootAngularVelocityBody` reports), so the code
  now takes THREE distinct frames — base IMU site (joint-KF anchor + gyro source),
  body frame `B` = `base_body_site` (contact-FK origin+frame), and
  `R_mount = ᴮR_S` (auto-computed at `qpos0`). On real Alex `R_mount` is a clean
  +90° yaw.

* **`R_mount` verified against Java to 1e-18 — the frame is exactly right.** Java
  publishes the pelvis gyro bias in both frames (`jointKF_gyroBias_pelvis_imu_*`
  in the IMU frame, `invariantAppliedGyroBiasInPelvisFrame` in the body frame), so
  `R_mount @ bias_S == bias_B` is a pure-rotation check with no signal processing
  in the way. It holds to 1.3e-18 RMS over the [200,210] s window, and also
  confirms I1 end-to-end on hardware: the bias the InEKF applies IS the joint-KF's.

* **Finding for Tier-2: the real InEKF consumes a Mahony-prefiltered pelvis gyro,
  not the raw `gyroscope_pelvis_imu`.** `R_mount @ raw_gyro` misses
  `invariantRawAngularVelocityBody` by ~2.5e-2 rad/s RMS (33% of signal during
  walking) while the bias-frame check above is exact — so the gap is the *input
  signal*, not the frame. A per-IMU `*_imuMahony*` complementary filter sits
  upstream. A full free-running Tier-2 replay of the fused step must feed the same
  processed angular-velocity channel, not the raw gyroscope. Does not affect the
  synthetic G9 gate, the assembly, or `R_mount`.

### G10 remains

Not built: the MJX sim env, the ONNX→Flax policy port (≤1e-6 oracle), the
closed-loop scan + vmap, and NIS/NEES consistency bands (`eval/consistency.py`).
`FusedOutputs` already emits the joint-KF `TickDiagnostics` and InEKF
`InEKFOutputs` (with per-tick NIS) the consistency evaluation reads. Note the sim
model is a *different* MJCF from the estimator's: `urdf2mjcf` deliberately drops
collision/visual geoms (the estimator needs only FK/Jacobians/M), so the sim needs
its own build with geoms + actuators (from `resources.zip` meshes or a vendored
full-body MJCF).

## G10 (part) — the estimator in the loop with the RL policy (`sim/`, 2026-07-26)

`run_estimator.py` + `src/invariant_estimation/sim/` put the fused step inside the MuJoCo policy
sim: simulated IMUs/encoders in, the policy's `base_ang_vel` + `projected_gravity` out of the
filter. Full numbers in `.claude-reports/2026-07-26-estimator-in-mujoco-sim.md`; the two findings
that belong in the port record are below.

### Finding 1 (FIXED, behind a flag): the contact FK pinned the off-path ankles at `qpos0`

`_make_contact_kinematics` evaluates base→sole FK from the 9 filtered joints, and
`MjxModel.qpos` widens that by leaving every other joint at `qpos0`. Alex's ankles are off-path,
so the InEKF's contact FK always believed them to be at zero. Measured on a walking run: the
ankles travel **0.66 rad**, and the contact FK error is 3.4 cm mean with a **5.3 cm swing over a
gait cycle**. The constant part is harmless — contacts are seeded consistently — but the swing is
not: a planted foot appears to slide 5 cm every step, and a filter whose contacts are stationary
by construction can only read that as base motion.

Java anchors at the live sole frame (`referenceFrames.getSoleFrame`), so **this is a port gap, not
a modelling choice**. Invisible to the Tier-1 parity harness because that compares roll/pitch,
which gravity leveling holds; the error lands on velocity/position.

Fix: `build_fused_estimator(contact_fk_unfiltered=True)` feeds the measured off-path joints to the
contact FK and block-diagonally widens `Σ_q` with their encoder variance, so `N = J Σ_q Jᵀ` still
covers every joint the measurement depends on. `init_fused_carry` seeds from the same augmented
vector, or the whole standing FK offset arrives as a step at tick 1. **Default off** so recorded
gates keep their numbers; the sim CLI defaults it on. 30 s walk, tail-RMS: tilt error
1.40° → **0.81°**, attitude 1.61° → 0.83°, position drift 2.84 → 2.20 m.

The **mass matrix deliberately keeps seeing `qpos0`** — that pinning reproduces Mecano compositing
the ignored subtree's inertia once at construction (worth 14% on `diag(Qa)`, `MjxModel.qpos`) and
is a separate concern from kinematics. The fix does not touch it.

### Finding 2 (OPEN): the missing touchdown reseed costs ~2 m of height per 30 s of walking

Residual drift after Finding 1 is **almost entirely vertical**, linear at ~0.09 m/s; horizontal
odometry is fine (18.77 m estimated vs 19.42 m travelled, 3.3% stride scale). **Base and both
anchors sink together** (−1.87 m base, −1.82/−1.87 m anchors over 20 s) with a 0.4 mm contact
innovation — a common mode the relative contact constraint cannot see.

Eliminated by direct test, not by argument:
* **not the IMU lever arm** — r = (−0.087, 0.012, −0.081) m biases specific force by −0.023 m/s²
  in z, but substituting a body-origin accelerometer moves 20 s drift only −1.873 → −1.917 m;
* **not loose anchors** — tightening `contact_floor` 1e-4 → 1e-6 makes it −15 m with 18° of tilt
  error and **the robot falls**. That slack absorbs contact/FK inconsistency; it is load-bearing.

It is gait-driven: **standing 30 s drifts not at all** (0.012 m constant), and while walking
**63% of the vertical error accumulates in the 25% of ticks around a touchdown**, at 5x the
background rate. That is the mechanism `reseedContact` + `TouchdownReseedLatch` exist to prevent —
tested Java runtime behaviour per `CLAUDE.md` §2, **never implemented in this port** (no
`inEKF/reseed.py`; `reseed.enabled: false`, deferred 2026-07-21 as "no measurable difference on
the real robot", a judgement made where absolute height matters least). `InvariantEKFReseedTest`
already specifies the congruence and the zero-release property. Second, independent candidate: the
still-deferred contact zero-velocity constraint (`J_dot = 0`, `inEKF/filter.py`).

Neither reaches the policy — base position and velocity are not in the 98-term observation.

---

## ContactNet seam — two contact covariance sockets (2026-07-27)

> **SUPERSEDED IN PART, 2026-07-29.** ContactNet now drives `contact_chol`, the
> **process** socket, not `contact_meas_chol`. The table below is still the
> correct description of the two sockets; the sentence "`eps` is the right lever
> for the *measurement* socket, `contact_floor` for the *process* one" is no
> longer a division of labour, because both now act on the same object. See
> "ContactNet moves to the process socket" below for the argument, the
> measurement, and the reconciliation of the two floors.

`network_plan.md` §1 specifies ContactNet as supplying `Σ_C` into the contact
update's measurement noise, `N̄ = R̂(J_C Σ_q J_Cᵀ + Σ_C)R̂ᵀ`. That term **did not
exist in the port**: `correct.measurement_noise(Np)` block-diagonalised the
encoder term `J_Ci Σ_q J_Ciᵀ` and nothing else, and `InEKFParams` had no contact
measurement variance field.

What *did* exist — `InEKFInputs.contact_chol` → `contact.digest` → `sigma_c` — is
a different quantity. `propagate.continuous_Qc` places it at `Qc[9:, 9:]`, which
in the I4 tangent ordering `[R, v, p, d_1…d_N]` is the **contact anchor block**.
It is the random-walk density on `d_i`: the stance-anchor slip process noise.
The naming (`Σ_C` in both the plan and `contact.py`) hid this; they are not the
same object and must not be conflated.

| input | enters | answers |
|---|---|---|
| `contact_chol` | process, `Q_d` | *is this foot world-static?* |
| `contact_meas_chol` | measurement, `N` | *how well do we know where it is?* |

### Change

* `InEKFInputs` gains `contact_meas_chol: (N,3,3)`, lower-triangular Cholesky
  factors of the FK measurement noise. `filter.step` adds it per contact before
  block-diagonalising: `N_i = J_Ci Σ_q J_Ciᵀ + Σ_Ci`.
* **No floor on this path.** `S = H P Hᵀ + N` needs only `N` PSD (`H P Hᵀ` is
  already SPD — see `kalman_gain`), and ContactNet owns strict positivity of its
  factor diagonal. So `contact_floor` stays what it always was: the process
  knob. This resolves an ambiguity in `network_plan.md` §6.2, which instructs a
  sweep of the network's `eps` to satisfy `cond(S) < 1e9`; `eps` is the right
  lever for the *measurement* socket, `contact_floor` for the *process* one.
* `UpdateDiagnostics` gains `logdet_S` — `2 Σ log L_ii` from the Cholesky
  `linear_update` already computes, so it cannot drift from `nis`. Together
  `(nis, logdet_S)` are a complete Gaussian NLL, `0.5(nis + logdet_S)`, which is
  what ContactNet's β-NLL objective (`network_plan.md` §5.3) needs: `nis` alone
  constrains only ratios of `S`, and the `logdet` term is what fixes absolute
  scale. NaN before any update, matching `nis`.
* Zeros in `contact_meas_chol` reproduce the pre-change filter exactly.
  `pipeline/main_estimator.py` passes zeros today — that is the single line
  ContactNet replaces.

### This does not reopen the "no contact mask" DECISION

`inEKF/filter.py`'s DECISION block argues contact *condition* belongs in the
process noise and that masking the measurement "treats a true observation as
false". That argument stands and is untouched. The new socket is not a mask and
is not about condition: it is uncertainty in the FK measurement itself — sole
compliance, contact-point geometry, foot deformation — which is present in
**firm** stance and which `J Σ_q Jᵀ` structurally cannot express, since it maps
only *encoder* variance.

### Open: is the process knob compensating for the missing measurement term?

The 2026-07-21 vertical-drift entry above records that tightening
`contact_floor` 1e-4 → 1e-6 produces −15 m of drift, 18° of tilt error, and a
fall, and concludes the slack "absorbs contact/FK inconsistency; it is
load-bearing". Contact/FK inconsistency is *measurement* error. The hypothesis
this seam makes testable: some of what `contact_floor` is absorbing belongs in
`N`, and with the measurement socket fed, the process knob may tighten without
the fall.

Untested — stated as a hypothesis, not a result. `contact_chol` /
`contact_floor` are therefore flagged **keep-or-remove pending the trained
network**, not removed now.

---

## ContactNet training loop — two non-obvious behaviours (2026-07-27)

Measured on the `tests/inEKF/test_filter.py` kinematics fixture. Absolute values
are fixture artifacts (the fixture feeds physically inconsistent random inputs,
so NIS/dof starts at 1.6e4); only the *directions* below are meaningful.

### 1. Under β-NLL the loss value is not a progress metric

Over 60 steps the β-NLL loss went **7.7e-3 → 8.7e-3 (up)** while NIS/dof went
**15810 → 472** — a 33x improvement in calibration. The optimisation was working
correctly the entire time.

This is structural, not a tuning artifact. The loss is

    stop_grad(exp(β · logdet S)) · 0.5 (nis + logdet S)

When the filter starts overconfident, the correct move is to inflate `Σ_C`,
which raises `logdet S`, which raises the **detached** weight. The product can
rise while calibration improves by orders of magnitude, because the weight is
not part of what is being minimised.

Consequence: watch **`nis_over_dof → 1`**, never the loss, when the objective is
β-NLL. `train.Metrics` exists to make that the visible number. Under
`l2_velocity` the loss *is* monotone and usable (measured 661 → 3.33 over the
same 60 steps) — a second reason to run the §5.3 L2 baseline first: it is the
only one of the two runs whose loss curve can be read naively.

NIS is on the stacked contact measurement, so it is χ²(3N) — dof **6** for two
contacts, not 3. `nis_over_dof` divides by `3 * N_contacts`.

### 2. `init_value=0.0` makes the first optimiser step a literal no-op

`optax.warmup_cosine_decay_schedule(init_value=0.0, ...)` returns exactly `0.0`
at step 0, so the first update is scaled to zero and *nothing* moves — head
included. Combined with the §4 zero-initialised head (which makes the trunk
gradient exactly zero until the head becomes nonzero), the real sequence is:

| step | what moves |
|---|---|
| 1 | nothing (lr = 0) |
| 2 | head only (trunk gradient still exactly 0) |
| 3+ | everything |

Correct behaviour, but it will read as a broken training loop if unexpected.

### 3. Testing note — `lr=0` cannot validate the weight-decay mask

`optax.adamw` applies decay *inside* the update and then scales the whole thing
by the learning rate, so `lr=0` zeroes the decay term too. A mask test built on
`lr=0, weight_decay=large` passes vacuously against **any** mask, including a
wrong one. The valid form is two steps at the same nonzero lr with
`weight_decay=0` vs large, asserting the trunk differs and the head is bitwise
identical. Verified in that form: trunk max|Δ| 4.9e-2 / 9.6e-2, head bit-equal.

---

## ContactNet feature channels + the torque seam (2026-07-27)

### Feature vector (CoCo's, per contact `i`)

    o_i := (ᴮω, ᴮa, q, τ, ᴮp_{B→C_i}, ᴮv_{B→C_i})

Everything body-frame, so no filter state is required — which is forced, not
chosen: expressing any of it in world needs `R̂`, and §1 forbids that.

| term | source | note |
|---|---|---|
| `ᴮω` | `FusedSensors.gyros[base_imu]` | **raw**, never bias-corrected — the correction is the joint KF's `b̂_ω` (I1), i.e. a filter output |
| `ᴮa` | `FusedSensors.accel_base` | raw |
| `q` | `encoders` + `q_unfiltered` | the base→foot chain spans both arrays |
| `τ` | `FusedSensors.torques` | new field, see below |
| `ᴮp_{B→C_i}` | `ContactFrames.y` | already computed every tick |
| `ᴮv_{B→C_i}` | finite difference of `ᴮp` | **not** `J q̇` — see below |

`ᴮp` is the highest-value channel and the only nonlinear one: `FK(q)` cannot be
recovered from `q` history by the first dense layer at any `H`, so it is real
information gain rather than reconditioning. `kinematics(q_encoders, zeros).y`
yields it from raw encoders — `y` does not depend on `q̇`, only `J_dot` does — so
the whole channel is joint-KF-free.

`ᴮv = J_{C_i}(q) q̇` would reintroduce the joint-KF dependency the feature set
was chosen to avoid, and is redundant anyway: given `H` ticks of `ᴮp`, a
difference is a linear function of inputs already present. Computed instead as
`(ᴮp[k] − ᴮp[k−1])/dt` — same first-order content, sensor-only, and it fixes the
conditioning problem (`ᴮp` carries a ~0.9 m DC leg-length offset, against which
the velocity signal is a small residual after frozen standardization).

Do **not** feed `FusedSensors.contact`: it comes from `trust.update(foot_loads(d))`,
the same foot-load oracle deliberately frozen out of `contact_chol` for training.

### Count — this resolves `F`

    per subchain joint (J_sub = 6):  q, τ        -> 12
    per contact:                     ᴮp, ᴮv      ->  6
    shared per tick:                 ᴮω, ᴮa      ->  6
                                             F   = 24

`D_in = H·F = 20 × 24 = 480`; ~190K params at `480→256→256→6`, inside §8's ~240K
budget. `B` is now the only §0 number still open.

Base→foot chain is `HIP_X, HIP_Z, HIP_Y, KNEE_Y, ANKLE_Y, ANKLE_X`. Only the
first four are filter states — the ankles are the unfiltered off-path joints.
Rather than carry two index spaces, `build_subchain_indices` resolves names into
the single `concat(filtered, unfiltered)` space that `FusedSensors.torques`
already uses, so `q` and `τ` share one convention.

**Hard dependency: ContactNet requires `contact_fk_unfiltered=True`.**
`SimSensorReader` only populates `q_unfiltered` under that flag; without it the
ankle angles never reach `FusedSensors` and two of the six subchain joints have
no `q` channel. (The flag is wanted anyway — the 2026-07-2x contact-FK entry
records that pinning the ankles at `qpos0` roughly doubled attitude error.)
`contact_channels` raises with that instruction rather than failing on a width
mismatch, which is how it first surfaced.

Verified end to end on the real Alex model, 80 policy-driven ticks: windows
`(80, 2, 20, 24)`, all finite, float64; `p_bc` mirror-symmetric at
`y = ±0.121 m`, `z = −0.917 m`; `base_accel_z ≈ 9.86` at rest; `q_knee_y ≈ 0.84`
rad against `tau_knee_y ≈ −50.5 ± 8.2 N·m`. Straight into the network at init,
`Σ_C − σ₀²I` is `1.3e-23`.

### `FusedSensors.torques` — new field

Torque existed only in `replay/logsource.py` (which parses `q|qd|tau` from
hardware logs); nothing in the sim path read it. Added:

* `FusedSensors.torques: (n + n_u,)`, optional, defaulting to `()` — the same
  "field absent" encoding `q_unfiltered` uses, so every existing construction
  site keeps working. **The estimator never reads it**; it exists purely as a
  ContactNet channel.
* `SimSensorReader` gains `enc_dofadr` and reads
  `d.qfrc_actuator[concat(enc_dofadr, unf_dofadr)]`. `qfrc_actuator` is the
  actuator contribution in *generalised* coordinates, so it indexes by DOF and
  lines up with the encoder ordering; `actuator_force` would be per-actuator and
  need the transmission map.
* Order is `concat(filtered, unfiltered)` — the same concatenation
  `fused_inputs` already uses to widen `q̂` for the contact FK.

Verified: 13 entries (9 filtered + 4 ankles); standing values are physical
(both knees −50.9 N·m, ankles ≈ +29.5, `SPINE_Z` ≈ 0.03, near mirror-symmetric).
Ordering is locked by `test_torques_are_gathered_in_concat_filtered_unfiltered_order`,
which drives one actuator at a time and checks against MuJoCo's own
`actuator_trnid`; swapping the concat order fails it with
`driving LEFT_HIP_X (index 0) landed on index 4 (RIGHT_HIP_X)`.

### Torque noise — AWGN placeholder, deliberately not grounded

`IMUNoise` gains `torque_std = 0.5 N·m` and `corrupt_torques` (plain additive
white Gaussian, no bias term unlike the gyro). Chosen for one reason: leaving
`torques` uncorrupted while gyro, accel, encoders and velocities are all noised
would teach ContactNet that torque is the one trustworthy channel — a lie that
does not survive hardware.

Scale sanity: 0.5 N·m is ~1% of Alex's measured standing knee torque (−50.9),
against 0.02–0.5% relative noise on the other channels. Same order, slightly
noisier, which is the right direction *if* Alex's torque is current-derived.

Verified: empirical std 0.5033 over 400 seeds (target 0.5), zero-mean,
reproducible per seed, and `noise=None` leaves the channel bit-identical to
`qfrc_actuator`.

**Still open, and it matters before any absolute calibration claim:** whether
the logged `tau` is *measured* (a sensor — this model is then the right shape,
only the number needs fitting) or *commanded* (a controller output, which
carries no sensor noise at all and would want a different treatment entirely).
The Java/SCS2 side was not consulted; this is a deliberate placeholder chosen
for speed, to be revisited if run 1 shows torque-driven pathologies.

---

## H is not 20 — the window span, measured against the 2026-07-17 log (2026-07-27)

`network_plan.md` §3.1 carries `H = 20` from CoCo. At Alex's 1 kHz that is a
**20 ms** window, and the log says that is degenerate. Analysis on
`20260717_160126_Alex001UnifiedControlProcess` (walking window t = 200–220 s,
read at stride 1), cross-checked through two independent readers — `ihmclog`
and this repo's own `replay/logsource` — which agreed to within 0.1%.

### Finding 1 — the log ticks at 1 kHz but the sensors update at 500 Hz

Every channel shows *exactly* 50% consecutive-identical samples in a strict
2-tick zero-order hold, and the phases are opposite: IMU latches on even ticks,
joints on odd. A duplication bug cannot produce opposite phases.

    gyroscope_pelvis_imuX      dup=0.5000   change-parity even=1.00
    raw_q_LEFT_KNEE_Y          dup=0.5000   change-parity odd =1.00

**True sensor Nyquist is 250 Hz, not 500.** And `stride = 1` literally repeats
every second sample, so half of an `H = 20` window was duplicated values.

### Finding 2 — the information is far below what a 20 ms window can see

Power below 50 Hz, and the frequency containing 99% of it:

| channel | below 50 Hz | f99 |
|---|---|---|
| joint position `q` | **100.000%** | **1.10 Hz** |
| joint torque `tau` | 99.974% | 4.25 Hz |
| gyro | 98.703% | 53.9 Hz |
| **accel** | 76.279% | **155.6 Hz** |

Stride fundamental is 0.183 Hz (5.47 s stride — slow treadmill walking, not the
1–2 Hz one might assume). A direct shape analysis agrees: at `H = 20`, 1–2
principal components explain 99% of an encoder window's variance, i.e. the
window is one value plus a slope.

The accelerometer is the sole exception and its content is real, not noise:
band-limited above 50 Hz its envelope peaks at 5x the mean in the 800 ms after
each double-support transition, and walking carries 43x more >50 Hz power than
standing. That is foot-strike impact ringing.

### The fix: same H, wider spacing — `H = 50, stride = 8`

`H` was never the problem; the window's **span** was. 20 samples is ample for a
1.1 Hz signal — they just have to be spread over a stride rather than crammed
into 20 ms. `window_indices` gains a `stride`; the window now spans 393 ticks.

Full-rate `H = 400` was measured and rejected. Benchmarked batch-1, 2 contacts,
jitted and warmed:

| H | stride | span | D_in | params | float64 | float32 |
|---|---|---|---|---|---|---|
| 20 | 20 | 380 ms | 480 | 190,470 | 0.116 ms | 0.072 ms |
| **50** | **8** | **392 ms** | **1,200** | **374,790** | **0.132 ms** | **0.079 ms** |
| 100 | 4 | 396 ms | 2,400 | 681,990 | 0.175 ms | 0.101 ms |
| 400 | 1 | 399 ms | 9,600 | 2,525,190 | **1.779 ms** | 0.361 ms |

`H = 400` exceeds the entire 1 kHz loop budget on its own. `H = 50` costs
+0.016 ms over `H = 20` — 14% more wall clock for 97% more parameters, because
at this size the forward pass is dominated by dispatch and the fixed 256x256
hidden layer, not the input layer. A MAC-proportional estimate predicted ~2x and
was wrong; §8's ~240K parameter budget is exceeded (375K) but §8 says outright
it is latency-bound, and the latency is fine.

Caveat: this is XLA on CPU, not Java/EJML. The ratio is the transferable part,
and even that is not guaranteed — EJML with preallocated buffers has less
dispatch overhead and may be more MAC-bound. Re-measure at the §7 cross-language
oracle before locking.

### `boxcar` — the anti-alias filter, and why the obvious metric misleads

Subsampling every 8th tick folds everything above the new 62.5 Hz Nyquist back
into band. `window` therefore boxcar-averages over `stride` ticks first;
`boxcar` and the strided gather are composed inside `window` so the two cannot
be separated by accident.

Measured on the log, with the error decomposed into the fold (out-of-band energy
landing in band) and the droop (in-band attenuation):

| channel | fold, naive | fold, boxcar | reduction | droop |
|---|---|---|---|---|
| accel X | 0.394 | 0.129 | **3.07x** | 0.366 |
| accel Y | 0.277 | 0.107 | 2.58x | 0.375 |
| accel Z | 0.249 | 0.096 | 2.61x | 0.455 |
| gyro X | 0.0034 | 0.0015 | 2.34x | 0.011 |
| tau KNEE_Y | 1.211 | 0.853 | 1.42x | 1.053 |

**The decomposition is the point.** A first pass scored both methods by RMS
distance from ideally-anti-alias-filtered decimation, and by that metric the
boxcar looked *worse* (0.26–1.02x) — because it penalises passband droop equally
with aliasing. They are not equivalent: droop is a deterministic linear
distortion, identical in sim and on hardware, seen the same way at train and
deploy. Folded energy is sampling-phase dependent, does not reproduce, and can
mimic real low-frequency signal. The first metric answered the wrong question.

A boxcar is crude, but its first null sits at `f_s / s` — exactly the new sample
rate, and exactly where the most damaging folding originates. It was chosen over
a real low-pass because it carries **no state**: nothing crosses the Java
boundary (§7) but an `s`-tap average, where an IIR's state would have to be
reproduced bit-for-bit in EJML. Free side effects: sensor noise down by `sqrt(s)`
on channels with nothing above Nyquist to lose, and the 500 Hz 2-tick hold
becomes irrelevant.

What it does **not** do is preserve the >62.5 Hz impact energy — it removes it.
What survives is the impact's low-frequency envelope, the deceleration bump
carrying the momentum transfer (~76% of accel power). The structural ring is lost
either way; naive subsampling does not keep it, it scrambles it.

`stride = 1` is byte-identical to the pre-change behaviour (index matrix equal,
`boxcar(x, 1) is x`), so this change is purely additive.

### Deviation bookkeeping — forced vs optional

Two deviations from `network_plan.md`, and they are not the same kind:

* **Decimation is forced.** Copying CoCo's `H = 20` at Alex's rate would not
  reproduce CoCo's experiment; it would run a different, degenerate one.
* **An HF-energy channel is optional** and is therefore **deferred**, not
  adopted. See below.

### Deferred experiment 1 — high-frequency accel energy channels

The energy the boxcar discards is recoverable as a slow channel, with no new
machinery and no filter state:

    hf     = x - boxcar(x, s)          # what the anti-alias filter removed
    energy = boxcar(hf**2, s)          # its power, as a slow channel

Applied to the 3 base accel axes: `F` 24 -> 27, `D_in` 1200 -> 1350, ~413K
params, wall clock ~+0.003 ms. The uniform `(N_c, H, F)` layout is preserved, so
`window`, the H-major flatten, `export.py` and the Java forward pass are all
unchanged; only `channel_names()` grows.

**Not in run 1, deliberately.** §9 step 8 is a reproduction whose job is to prove
the BPTT plumbing, and a channel CoCo never had would make a disappointing result
unattributable. Run it as a clean A/B after beta-NLL: same everything, +/-3
channels, scored on NEES and velocity RMSE against the run-1 baseline.

Limitation to state when it is run: an energy channel is rectified, so it carries
how much HF content arrived and when, but not its spectral character. If slip and
firm contact ring at different frequencies, per-band energies (3 bands x 3 axes)
would be needed to separate them — same mechanism, more channels, still no state.

---

## Measured sensor noise floors — the table `normalize.py` cites (2026-07-28)

`normalize.NOISE_FLOOR` cited this file for its values and this file did not have
them. Recording them, from the 2026-07-17 Alex001 log analysis (walking window
t = 200–220 s, read at stride 1; cross-checked through both `ihmclog` and this
repo's `replay/logsource`, which agreed to within 0.1%).

| channel | noise std | how obtained |
|---|---|---|
| base gyro | 1.5–4e-3 rad/s | quiet-window 2–60 Hz PSD plateau |
| base accel | 0.04–0.05 m/s² | ditto; PSD flat to Nyquist, so partly aliased |
| joint `q` | ~4e-6 rad | quiet-window plateau — **a LOWER BOUND**: the log's `raw_q` is already low-passed (−18.6 dB at 50 Hz), so intrinsic encoder noise is probably 1–2e-5 |
| joint `tau` | 0.20 N·m median (0.10–0.28 per joint) | quiet-window plateau; SNR 37–196 while walking |
| `p_bc` | ~5e-6 m | **propagated**, `J σ_q`, not measured |
| `v_bc` | 7.1e-3 m/s | **propagated**, `sqrt(2)·σ_p/dt` — see below |

Method note worth keeping: the PSD-plateau and direct-quiet-window methods agree
within 2–4x on gyro/accel/torque and disagree by **10–40x on the encoders**. The
walking-window HF "plateau" sits ~1000x above the quiet-window floor, i.e. that
content is real mechanical vibration rather than sensor noise, so the walking
method over-reads badly. The quiet 2–60 Hz plateau is the defensible number.

### Bug: the `v_bc` floor was 7071x too small, and therefore inert

`v_bc` is a first difference at 1 kHz, so it amplifies position noise by
`sqrt(2)/dt` — from `σ_p = 5e-6 m` that is **7.1e-3 m/s**. The table shipped
`1e-6`.

The error came from applying a reduction that does not apply. `features.window`
boxcar-averages before subsampling, and the two operations telescope:

    boxcar_s(diff(p)/dt)[k] = (p[k] - p[k-s]) / (s*dt)

so it is tempting to divide by `stride`. But **`normalize.fit`/`apply` run on the
`(T, N_c, F)` channels BEFORE windowing** — that ordering is the whole point of
the split — so at the moment the floor is compared against `raw_std`, the boxcar
has not happened. The amplification is the full `sqrt(2)/dt`.

Consequence of the old value: the floor could never fire for `v_bc`. It would
have gone unnoticed while walking and failed exactly the case the floor exists
for — a standing calibration set, where `v_bc` is almost entirely noise.
Verified after the fix: `v_bc_x/y/z` now appear in `floored` on a noise-only set.

---

## β-NLL's β is dimension-dependent, and 0.5 at k=6 is not the paper's 0.5

Flagged during the theory write-up; **not yet acted on**, because run 1 is the L2
baseline (`config.objective` defaults to `l2_velocity`) so it does not block.

The β-NLL reweight in the source formulation is **per dimension**: each term is
weighted by that dimension's `σ^{2β}`. `losses.beta_nll_from_diagnostics` instead
weights the whole joint NLL by `det(S)^β`. For isotropic `S = s·I_k`:

    det(S)^β = s^{k·β}

so at `k = 3N = 6` contacts-stacked and `β = 0.5` the weight goes as **`s³`**,
where the per-dimension intent is `s^0.5`. Matching the paper's semantics would
need `β_ours = β_paper / k ≈ 0.083`.

Two consequences worth checking before run 2:

1. The weight is `det(S)^β`, so **small `S` ⇒ small weight**. An overconfident
   filter has small `S` and large NIS, so the reweight *down-weights the most
   overconfident ticks* — the wrong direction for an estimator, and at `s³` the
   effect is severe rather than marginal.
2. `losses.py` has two entry points at different `k` — `beta_nll` (k=3, single
   contact) and `beta_nll_from_diagnostics` (k=6, stacked). **The same `β` means
   different things in each**, which is a trap for whoever tunes it.

Not a bug in the L2 path. Decide before switching objectives.

---

## BUG FIXED — the contact measurement noise was never rotated to world (2026-07-28)

Found while writing the theory document; verified from source and quantified
before fixing.

`correct.innovation` returns a **world-frame** residual — its own docstring says
so:

    nu = y @ state.R.T - rel          # R̄ y_i − (d̄_i − p̄), measurement − model, WORLD frame

but `Np = J_C Σ_q J_Cᵀ` and ContactNet's `Σ_C` are both **body-frame**, and
`filter.step` passed them straight into `linear_update`. So `S = H P Hᵀ + N`
mixed frames.

The port already contained the fix and never called it.
`correct.rotate_measurement_covariance`, docstring verbatim:

> Java `ContactUpdater.computeMeasurementCovariance`. The residual lives in the
> world frame (`contact_residual`), so the body-frame FK noise must be
> conjugated by the estimated attitude before it enters `S`.

and `map_encoder_noise` says its output "is the **body-frame** FK covariance,
which `rotate_measurement_covariance` then takes to world". `network_plan.md` §1
specifies `N̄ = R̂(J_C Σ_q J_Cᵀ + Σ_C)R̂ᵀ`. Three independent sources agree; only
the wiring was missing.

### Why nothing caught it, and why it matters *here*

Measured on the test fixture with a genuinely non-identity `R̂` (‖R̂−I‖ = 2.82):

| `Σ_C` | max\|N_body − N_world\| | relative Kalman-gain error |
|---|---|---|
| isotropic `1e-6·I` | **6.4e-22** | 1.2e-21 |
| anisotropic slip `diag(1e-3, 1e-3, 1e-8)` | 3.3e-4 | **2.7e-3** |

`R̂(σ²I)R̂ᵀ = σ²I` exactly, so for the shipped isotropic default the bug is
*machine-zero invisible* — which is why 653 tests passed over it.

It is not a no-op for anisotropy, and **anisotropy is ContactNet's entire
justification**. `inEKF/filter.py`'s own DECISION note argues `Σ_C` is "strictly
more expressive than a scalar trust weight: a full covariance can say 'this foot
slides along the surface but not through it'". That statement was false in the
scan body: the sideways-vs-through distinction was being partially discarded
before it reached `S`. And the network cannot compensate, because §1 forbids it
from seeing `R̂`.

Training against the unfixed path would have produced a network optimised
through a filter that throws away the thing it is learning.

### Fix

`filter.step` now conjugates on the **prior** state, matching `innovation`:

    N_world = rotate_measurement_covariance(state, Np + Nc)
    state, diag = linear_update(state, ekf.params.H, nu, measurement_noise(N_world))

Full suite **667 passed**, no failures — the ported Java tests are silent on this
because they use isotropic noise, so they neither caught the bug nor object to
the fix. Worth a dedicated anisotropic regression test when the ContactNet suite
is written.

---

## ContactNet dataset + first real training runs (2026-07-28)

`contactnet/dataset.py` (segment loader) and `train_contactnet.py` (entry point)
close the gap between `sim/collect.py` and `contactnet/train.py`. Data collected:
**12/12 rollouts**, 4 terrains x 3 seeds, 62 s each (2 s settle + 60 s walk at
vx = 0.4), 1.5 GB. Every rollout stayed upright (tilt max 2.3-5.2 deg against the
15 deg bound) and on the field; none were skipped. Cost 3.5-6.0 wall-s per
simulated second, ~50 min total, matching the collector's own measurement.

### Segment construction — the three decisions that were open

**1. Windows are computed by a global boxcar plus a gather, not per segment.**
`features.window` is `boxcar(channels, stride)` then a strided gather. The boxcar
is a cumulative sum, so running it on a 128-tick slice gives a *different* (and
equally valid) float64 answer from running it on the full 62 000. Doing it once
globally, at prepare time, makes a segment's windows **bit-identical** to
`features.window(...)[t0:t0+L]` and turns the whole per-segment path into one
`np.take`. That equality is the loader's strongest oracle
(`test_segment_windows_are_bit_identical_to_features_window`) and it is only
available because of this ordering.

**2. `state0`'s contact anchors come from the FK at the FILTERED `q̂`, not truth
`q`.** `R, v, p` are ground truth at the start tick; the anchors are
`d_i = R_true · y_i(q̂) + p_true` with `y_i` evaluated at `inputs.joint.q` — the
same vector the filter measures against on tick 1 — so the first contact residual
is exactly zero. Seeding from truth `q` instead injects the filter-vs-truth joint
offset as a step at tick 1, which is the failure `init_fused_carry` already
documents for the ankles. Note `truth["q"]` does not even carry the off-path
ankles, so the truth-`q` option is not fully available.

The joint KF is **not** reseeded. It ran continuously through collection and its
outputs are frozen into `inputs`; there is nothing to reseed.

**3. `P0` is measured, not chosen.** Seeding a mid-trajectory segment with the
diffuse `initial_covariance = 1.0` prior would put `NIS/dof` on a ramp from ~0
that has nothing to do with the network. `dataset.measure_p0` instead runs the
InEKF alone over 3 000 ticks of recorded input under *training* conventions
(`contact_chol` frozen at the constant, `contact_meas_chol = 0`) and takes the
converged `P`. On `flat/seed0`:

    diag(R) = [6.90e-5, 6.91e-5, 1.000]     yaw at the prior — unobservable, correct
    diag(v) = [7.6e-4, 7.6e-4, 2.2e-4]      ~2.8 cm/s
    diag(p) = diag(d) = 0.34 (all six)      absolute position unobservable; only p − d is
    eigenvalues in [4.2e-10, 1.03]

The `diag(p) == diag(d)` coincidence is the structure, not a bug: the InEKF sees
only `R̂ᵀ(d − p)`, so the common mode is untouched by the contact update. A
converged Joseph-form `P` also lands a hair below zero in its smallest direction
(measured -1.4e-16 against a spectral radius of 1.0); `measure_p0` nudges that to
strictly PD and raises only if the violation exceeds `1e-8 · λ_max`.

A per-gait-phase `P0` would be more faithful (store `P` at every tick, 111 MB per
rollout) and is the obvious refinement if the seeding transient shows up.

### Normalization — the `floored` gate passes clean

Fit over the pooled post-warm-up region of all 12 rollouts: **1 104 000 samples**
(12 x 46 000 ticks x 2 contacts). **`floored` is empty.** The failure this gate
exists to catch — a standing-only calibration set flooring `base_gyro_y/z` and
`base_accel_x` — does not occur here, and by a wide margin: the closest channel to
its floor is `base_gyro_x` at std 9.6e-2 against a 3.0e-3 floor (32x), and
`base_accel_x` sits at 0.87 against 4.5e-2 (19x). Physical spot-checks all land:
`base_accel_z` mean **9.807** (specific force at rest), `p_bc_z` mean
**-0.866 m** (foot below the pelvis), `q_knee_y` **0.99 rad** against
`tau_knee_y` **-57.6 N·m**.

### `B` — the last §0 open number

Measured with `train_contactnet.py measure-b`: one subprocess per point (because
`ru_maxrss` is a high-water mark that cannot be reset), and an **ahead-of-time
compile** so XLA's own peak is a separate column from the gradient's. 20 CPU
cores, L = 128, 4 rollouts resident.

| B | remat | compile Δ [MB] | exec Δ [MB] | peak [MB] | compile [s] | step [s] |
|---:|---|---:|---:|---:|---:|---:|
| 8 | on | 732 | -13 | 2521 | 24.4 | 0.20 |
| 8 | off | 464 | 24 | 2232 | 16.5 | 0.13 |
| 16 | on | 717 | 35 | 2545 | 24.7 | 0.38 |
| 16 | off | 468 | 101 | 2386 | 17.3 | 0.24 |
| 32 | on | 753 | 135 | 2729 | 24.6 | 0.63 |
| 32 | off | 471 | 242 | 2541 | 16.8 | 0.44 |
| 64 | on | 718 | 305 | 3037 | 24.8 | 1.23 |
| 64 | off | 472 | 533 | 2933 | 16.9 | 0.84 |

**`B` is not memory-bound at this `L`.** Forward+backward costs ~8.5 MB per unit
`B` without remat, ~5.7 with; B = 64 peaks at 2.9 GB of a 94 GB machine, and the
largest single allocation in the whole process is the XLA *compiler*, not the
gradient. That was not the expected answer — the prior was that BPTT through 128
InEKF ticks would be the constraint. It is not, because the scan carry is a
15x15 covariance and the network runs *outside* the scan, batched over the whole
time axis at once (`rollout.contact_factors`).

`remat` does exactly what it advertises (43% less execution memory at B = 64) but
costs +250 MB of compile-time peak and **+47% step time**, so at L = 128 on CPU it
is a net loss. Gradients are identical with it on or off (max abs difference
4.4e-11 against a gradient norm of 0.915, i.e. 4.8e-11 relative) — it is purely a
space/time trade, as it should be.

**Recommendation: `B = 32`, `remat = False`.** 0.44 s/step -> 10 000 steps in
~75 min. 32 is ~2.7 segments per rollout; beyond that the extra segments come
from trajectories already in the batch, and with only 12 independent trajectories
the effective sample size grows far slower than `B`. Revisit remat when `B·L`
passes ~10 000 tick-segments, or on GPU where device memory binds.

`config.py` ships `remat = True` and `B = 32`. The `B` default is confirmed by
measurement; the `remat` default is worth flipping for the CPU run — not changed
here, since `config.py` is committed.

### §4 init parity holds end to end on real feature windows

`train_contactnet.py check-init`, B = 32 real segments:

    Σ_C − σ₀²I           1.3e-23 absolute, 1.3e-15 relative
    trunk gradient       exactly 0.0 (bitwise, every leaf)
    head gradient        max 0.949
    windows finite       True

So the zero-head initialisation survives normalization, windowing, the vmapped
network, the 128-tick scan and the reverse pass. `d_in = H·F = 50 · 24 = 1200`
and the network is **374 790 params** — worth noting that `network_plan.md` §8's
"~240K budget" was written against `H = 20` (`d_in = 480`, ~190K) and is stale
since the "H is not 20" entry raised `H` to 50. The budget question is a Java
inference-cost question, so it should be re-decided rather than silently exceeded.

### First real training runs — two findings the fixture could not show

**Run A — `l2_velocity`, 200 steps, B = 32, remat on, lr 1e-4, warmup 100.**
140 s. Gradients finite and nonzero throughout, no NaN, `applied_frac = 1.000`
every step, params moved (trunk max |Δ| 9.9e-3, head.b 1.3e-2).

| step | loss | ‖g‖ | NIS/dof | cond proxy |
|---:|---:|---:|---:|---:|
| 0 | 4.82e-2 | 4.30 | 1.456 | 1.9 |
| 32 | 3.34e-2 | 1.04e-1 | 1.004 | 4.5e4 |
| 96 | 3.76e-2 | 2.48e-1 | 0.955 | 2.4e5 |
| 128 | 3.92e-3 | 1.03 | 0.224 | 1.1e7 |
| 160 | 8.40e-4 | 9.47e-2 | 0.0436 | 1.8e8 |
| 199 | 5.90e-4 | 1.21e-1 | 0.0422 | 1.8e8 |

The loss falls **82x** and is monotone in trend, as expected for L2. But
**`NIS/dof` moves away from 1, not toward it** — 1.46 down through 1.0 to 0.034,
i.e. the trained filter is ~30x under-confident. This is not a bug: it is the
pathology `losses.l2_velocity`'s own docstring names. Σ reaches an L2 loss only
through the Kalman gain, so only *ratios* of Σ are constrained and the absolute
scale is free; the optimiser buys velocity accuracy by inflating Σ_C. So under L2
the success criterion is the loss, and `nis_over_dof` is a diagnostic, not a
target. Two consequences worth acting on before a 10 000-step run:

* the conditioning proxy climbs 1.9 -> 1.8e8 over 200 steps against
  `cond_max = 1e9`. It has not gated an update yet (`applied_frac` never left
  1.000), but the trend is toward the gate, and a gated update means the loss is
  scoring innovations that never corrected anything. Watch `applied_frac`.
* the big move lands right after the warmup ends at step 100, which is the
  expected schedule shape, not instability.

**Run B — `beta_nll`, 150 steps, B = 32, warmup 20. It does not train at all.**
Loss stuck at -2.2e-19, ‖g‖ at 5e-18, `NIS/dof` flat at 1.3-1.6 for 150 steps.
Diagnosed:

    nis        mean 11.80  (dof 6 -> NIS/dof 1.97)
    logdet_S   mean -93.42   (per-axis innovation std ~ 4.2e-4 m)
    exp(0.5 · logdet_S)      5.2e-21     <- the detached beta weight
    nll = 0.5(nis + logdet)  -40.8
    loss = weight · nll      -2.1e-19

**This is a units problem, not an optimisation problem.** `S` is in m², the
contact block is 6-dimensional, and `det(S)` is therefore ~`σ¹²` — at
`σ ≈ 4e-4 m` that is `e^-93`. `beta_nll_from_diagnostics`'s detached weight
`exp(β · logdet S)` underflows to 5e-21, and while Adam is scale-invariant in
principle, `optax.adamw`'s default `eps = 1e-8` is nine orders above `sqrt(v)`
here, so `m/(sqrt(v) + eps)` collapses and the updates are ~1e-13.

The fixture did not show this because its `S` was ~1e5 times larger
(`P = 0.1·I`), giving `logdet ≈ -14` and a workable weight of ~1e-3 — which is
why the earlier "15810 -> 472" fixture result cannot be carried over.

Two fixes, neither applied here because `losses.py`/`train.py` are committed:

1. **Offset the detached weight by a constant**:
   `weight = stop_grad(exp(β·(logdet_S − c)))` with `c` a fixed reference (e.g.
   -93.4, the value at init). Because the weight is detached and `c` is constant,
   this multiplies the whole loss by `exp(-βc)` and is therefore a **pure loss
   rescale** — mathematically the same optimisation, lifted off the floor. One
   line, provably harmless, and it is the recommended fix.
2. Shrink `optax.adamw(eps=...)` to ~1e-30. Fixes the symptom, leaves the loss
   denormal-adjacent, and does nothing about `float64` headroom if `L` grows.

A third option worth considering separately: Seitzer's β-NLL is defined
per-dimension (`stop_grad(σ_i^{2β})` per output), where the exponent does not
scale with the measurement dimension. The multivariate `det^β` generalisation
used here makes the weight scale as `σ^{2kβ}`, which is what makes `k = 6` fatal.

### Deliberate deviations, recorded

* **The stance/swing `contact_chol` freeze cuts both ways.** Freezing it at the
  *stance* value (trap: it must be frozen, or the network gets a free ground-truth
  contact flag) also means the process model insists a swinging foot is
  world-static, so ContactNet must reject swing through the *measurement*
  covariance — the lever `inEKF/filter.py`'s DECISION note argues is the wrong one
  for contact condition. This is coherent only if the deployed filter also drops
  the heuristic swing inflation, which PORT_NOTES already lists as "a candidate
  for removal once the learned path is trained". If it is kept at deploy, the two
  mechanisms double-count. Decide this before run 2.
* **The gravity reference cold-starts every segment.** `make_segment_loss` calls
  `inEKF.filter.init_carry`, which returns an *unseeded* `GravityRef`, so each
  128-tick segment re-seeds the complementary filter from its first accelerometer
  sample rather than inheriting the converged direction a continuously-running
  filter would have. With `reference_tau = 5 s` against a 0.128 s segment, the
  reference barely moves within a segment, so the effect is a constant offset in
  the leveling residual rather than a transient — but it is a train/deploy
  difference and it lives in a committed module.
* **Segment starts skip `warmup + (H-1)·stride` ticks.** The lead-in term is
  strictly unnecessary as implemented — channels are windowed over the *full*
  stream, so windows at `warmup + 0` are real rather than boxcar-clamped — but it
  costs 0.9% of the usable starts and removes the question. 545 772 legal starts
  remain over 12 rollouts.
* **`rollout_paths` is structural, not name-based** (regression): the first
  training run wrote its checkpoint into `data/`, and a glob-based loader promoted
  it to a rollout. It now checks for the `sensors.encoders` key in the archive
  directory. Training artifacts go to `artifacts/` (gitignored).

### Verification

`tests/contactnet/test_dataset.py`: 17 tests, no MJX (rollouts are fabricated in
`collect.save_rollout`'s own format; `measure_p0` runs against the analytic
kinematics fixture from `tests/inEKF/test_filter.py`). **Mutation-tested: 16
mutants applied to `dataset.py`, 16 killed** — dropped `swapaxes`, stride ignored
in the window indices, segment slice off by one, `contact_chol` not overwritten,
`contact_chol` frozen at the swing value instead, lead-in dropped, warm-up
dropped, iid rollout draw instead of a permutation, tiled starts instead of
random, warm-up ignored in the normalization fit, `normalize.apply` skipped,
contacts not rotated into world, `v` seeded from the wrong tick, `measure_p0`
keeping the sim `contact_chol`, boxcar dropped, and the name-based
`rollout_paths`. Full suite **704 passed**.

---

## β-NLL does not train on the real model — one root cause, found twice (2026-07-28)

**Not fixed.** Run 1 is the L2 baseline (`config.objective` defaults to
`l2_velocity`), so this does not block. It must be decided before run 2, and the
decision is about *semantics*, not just about a magnitude.

Two independent analyses tonight landed on the same place from opposite ends.

### Symptom (measured on real data)

A 150-step β-NLL run moved nothing: ‖g‖ = 5e-18, `nis_over_dof` flat at 1.3–1.6.

`S` is in m² and the stacked contact block is 6-D, so on the real model
`logdet S ≈ −93.4`, and the detached weight is

    exp(β · logdet S) = exp(0.5 × −93.4) = 5.2e-21

which AdamW's default `eps = 1e-8` then swallows whole. **A units problem, not an
optimisation one.**

Note this did *not* reproduce on the test fixture, whose `S` is ~1e5 larger — so
the earlier fixture result recorded above ("15810 → 472") does **not** carry over
to the real model. That is worth remembering generally: the fixture's scale is
not Alex's.

### Root cause (derived independently, while writing the theory document)

Seitzer's β-NLL reweight is **per dimension** — each term carries its own
`σ^{2β}`. `beta_nll_from_diagnostics` instead weights the whole joint NLL by
`det(S)^β`, and for `S = s·I_k`:

    det(S)^β = s^{k·β}

So at `k = 3N = 6` and `β = 0.5` the weight scales as **`s³`**, where the source
formulation intends `s^0.5`. Matching the paper's semantics needs
`β_ours = β_paper / k ≈ 0.083`.

**These are the same fact.** `σ^{2kβ}` is what makes the weight collapse to 1e-21
at k = 6, *and* what makes `β = 0.5` mean something six times more aggressive
than the number suggests. Fixing the magnitude without fixing the semantics
leaves a knob whose label lies.

### Consequences, in order of how much they should worry you

1. `losses.py` has **two entry points at different k** — `beta_nll` (k = 3, one
   contact) and `beta_nll_from_diagnostics` (k = 6, stacked). The same `β` means
   different things in each. That is a trap for whoever tunes it.
2. The weight is `det(S)^β`, so **small `S` ⇒ small weight**: the reweight
   *down-weights the most overconfident ticks*, which is the wrong direction for
   an estimator. At `s³` the effect is severe rather than marginal.

### Options (decide, do not patch blindly)

* **Offset the detached weight**: `stop_grad(exp(β·(logdet_S − c)))`. Since the
  weight is detached and `c` constant, this is exactly a uniform loss rescale, so
  the optimisation is mathematically identical and only the magnitude moves above
  `eps`. Fixes the symptom; leaves the semantics wrong.
* **`adamw(eps=1e-30)`** — same, cruder.
* **Re-derive β per-dimension**, or set `β = β_paper / k`. Fixes the semantics.
  Preferred, but changes what previously-recorded β numbers mean.

Whichever is chosen, make the two entry points agree, and re-record any β result
measured before the change.

---

## Run 1's dataset is narrow — read the result accordingly (2026-07-28)

Recording this because a good run-1 number is easy to over-read.

Run 1 trains on **12 rollouts**: 4 terrains x 3 seeds, 62 s each, all at a single
straight-line command `vx = 0.4`, one policy, one gait. After the 16 s warm-up
and the lead-in that is ~552 s of usable trajectory, 1.104M pooled samples.

**Count contact events, not samples.** Samples within a stride are near
duplicates. At roughly a 1 s stride that is ~550 strides x 2 feet ≈ **1 100
independent contact events** against a 374 790-parameter network. It is less
alarming than it sounds — the output is 6 numbers with a strong prior (§4 init at
the shipped filter) — but 1 100 is the number to reason with.

### The specific gap is friction, not volume

`Sigma_C` is contact *measurement* uncertainty: sole compliance, contact
geometry, and slip. The dataset varies terrain and (after the planned sweep)
speed and yaw. It does **not vary friction**, and `TERRAIN.md` §5 records that
`randomize.py` already implements the Brax/MJX domain-randomisation pattern for
Alex (`geom_friction`, `dof_frictionloss`, `dof_armature`, `body_ipos`,
`body_mass`) — none of which is in use.

If the sim never slips, ContactNet can only learn "`Sigma_C` ≈ constant,
modulated slightly by load and geometry". That is a real result but a small one,
and **a good L2 number would be exactly what you would expect for the wrong
reason**.

### Do this before collecting a large dataset

Instrument the collector for slip — tangential foot-sole velocity while loaded,
per contact — and report event counts. If slip is near-zero across all rollouts,
add `geom_friction` randomisation in the same collection pass rather than
collecting twice. That measurement is ~20 minutes; a 40-rollout sweep is hours.

Run 1 remains worth running as the §9 step 8 plumbing proof. Its number means
"the machinery works", not "ContactNet helps".

---

## Run 1 learned to switch the contact update OFF (2026-07-28)

**Read this before using `artifacts/contactnet_run1.npz` for anything.**

Run 1 completed: 10 000 steps, `l2_velocity`, B = 32, 31 min on GPU. It is
healthy by every process metric — `applied_frac` held **1.000** to step 9999
(the conditioning gate flagged as the likely silent failure never fired; the
proxy peaked at 7.7e7 against `cond_max = 1e9`), gradients finite throughout,
loss down to ~1e-4, output SPD and finite everywhere.

It is also **degenerate**. Evaluated on real cached feature windows from
`flat_seed000`, 400 ticks x 2 contacts:

| per-axis contact std | min | median | max | init `sigma_0` |
|---|---|---|---|---|
| x | 5.3e-4 | **6.755e-1 m** | 4.79 | 1e-4 |
| y | 2.7e-4 | **1.718e-1 m** | 4.30e-1 | 1e-4 |
| z | 8.7e-3 | **2.745e-1 m** | 1.53 | 1e-4 |

A contact-position measurement uncertainty of **0.68 m**. For scale, the other
term in the same innovation, `N = J Sigma_q J^T`, is 1.26e-5 m^2 (3.5e-3 m std,
`config.py`). `Sigma_C` now exceeds it by ~4 orders of magnitude in variance and
therefore *is* `S`.

### Measured, not argued

Contact-update Kalman gain on the measured `P0` (`artifacts/p0.npz`), one
contact block, `S = H P H^T + N + Sigma_C`:

| `Sigma_C` | tr(S) | ‖K‖_F |
|---|---|---|
| init, `sigma_0^2 I` | 3.785e-05 | 2.3424e-03 |
| trained, median | 5.612e-01 | **8.3895e-07** |

**2793x gain suppression.** The contact update is off. The argument is
rotation-invariant in the part that matters (`rotate_measurement_covariance`
preserves the trace), so the frame conjugation does not rescue it.

The network did move off init — head.W delta 0.235, trunk.W delta 0.185, output
anisotropy 10.7x median, off-diag/diag mass 0.30, temporal CoV 37%. It learned a
rich, time-varying, anisotropic function. It just learned the wrong one.

### Root cause: our segment seeding is a force-teacher scheme

`dataset.make_segment` seeds `state0.R, v, p` from ground truth at **every**
segment start, and a segment is `L = 128` ticks = **128 ms**. The filter
therefore begins each training sample perfect and only has to survive 0.128 s —
a horizon over which IMU dead-reckoning beats any contact correction. The
globally optimal `Sigma_C` under that objective is **infinity**, and 10 000
steps of Adam found it. `nis_over_dof` = 0.03 is the same fact seen from the
innovation side, and it was never going to recover with more steps.

CoCo-InEKF (arXiv 2605.15122) reports exactly this, Sec. III-B:

> "We experimented with re-initializing the InEKF state to the ground-truth
> state at the start of each rollout in a force-teacher fashion, but observed a
> degradation in the filter's learning performance."

They instead **carry (X, P) across consecutive buffers** up to an episode length
T (100 s for dancing, 6 s for ground motions), letting the filter drift away
from truth. With accumulated drift, the contact update is the only thing that
can correct it, so `Sigma_C` acquires a finite optimum. Same `L = 128`, same L2
body-frame-velocity loss, same 6-element lower-triangular `Sigma = L L^T`
parameterisation as ours — the seeding is the difference.

**Fix before run 2:** carry the filter carry across consecutive segments within
a rollout rather than re-seeding from truth. This is a `dataset.py` /
`train.py` change, not a filter change. Note the interaction with (d) in
`dataset.py`'s docstring: batches are currently composed *across* rollouts with
uniform random starts, which is incompatible with a carried state; segments
within a rollout will have to advance in order, and B independent chains will
have to be maintained in parallel.

Do not treat this as a hyperparameter to tune around. Inflating `Sigma_C` is not
a local minimum the optimiser fell into — it is the correct answer to the
question we asked.

---

## CoCo-InEKF comparison — why 31 min vs their 5 days (2026-07-28)

arXiv **2605.15122**, Baumgartner, Mueller, Serifi, Grandia, Knoop, Gross,
Baecher (Disney Research / ETH). PDF at `~/Downloads/2605.15122v1.pdf`.

Recorded because "our training is 200x faster" is the kind of number that reads
as an advantage and is mostly a difference in what is being counted.

### The gap decomposes exactly

Their Table VII (BPTT unroll ablation) reports iterations reached inside the
5-day cap, at **L = 128 — identical to `ContactNetConfig.L`**:

| | iters in 5 days | s/iter |
|---|---|---|
| L = 64 | 89 600 | 4.8 |
| **L = 128 (theirs and ours)** | **64 600** | **6.7** |
| L = 256 | 18 800 | 23.0 |

Their iteration: 6.7 s. Ours: 0.196 s. **34x.** Their iteration count: 64 600 to
our 10 000, **6.5x**. Product **220x**; 5 days / 31 min = 225x. No residual.

The 34x is two things: physics regenerated *every iteration* ("Per learning
iteration, we collect a training dataset ... by forward-simulating a pretrained
policy in E environments"), and **E = 1280 parallel envs vs our B = 32** — a 40x
batch. They use a **pretrained policy and do not train it**, same as us; the 5
days contains no RL.

Segments seen: theirs 64 600 x 1280 = **82.7M**, freshly simulated. Ours
10 000 x 32 = **320k**, drawn from ~1 100 independent contact events. 258x on
count, and the independence gap is larger than the count gap.

### Architecture is near-identical — we under-fed, not under-built

| | CoCo-InEKF | ours |
|---|---|---|
| params | 240 344 | 374 790 |
| BPTT unroll L | 128 | 128 |
| history H | 20 | 50 |
| loss | L2, body velocity in body frame | `l2_velocity` — same |
| output | 6 lower-tri elements, `Sigma = L L^T`, body frame | same |
| inputs | `omega, a, q, qd, tau, p_B->Ci, v_B->Ci` | our 24 channels, same families |
| filter state as NN input | **excluded, deliberately** | excluded (CLAUDE.md §7) |
| MLP not CNN | yes, for onboard real-time | yes |

Convergent design, arrived at independently. Their H = 150 variant reached 1.74M
params and got *worse* (RMSE 0.052 vs 0.046), which retires the §8 parameter-
budget worry as a first-order concern.

### The four gaps, ranked

1. **Fresh physics per iteration vs 300x replay of 552 s.** Structural.
2. **Friction and disturbance-force randomization.** They randomize both; their
   *test* set adds periodic disturbances specifically "to induce slippage". We
   randomize terrain only. Already flagged above under dataset narrowness — the
   paper confirms it is the right thing to have flagged.
3. **Force-teacher seeding** — see the run-1 entry above. This one is a defect,
   not a scale gap.
4. **Difficulty.** Their headroom on dancing is 15x (heuristic-contact InEKF
   2.675 velocity RMSE vs 0.176 with GT contacts). On a straight-line
   `vx = 0.4` walk the heuristic baseline is likely already near the GT-contact
   bound, so **our dataset may contain almost no headroom for ContactNet to
   capture**. Also `N = 2` contact points against their 4/10/18, where accuracy
   improved monotonically (0.134 -> 0.099 -> 0.069).

We are **not** compute-bound and they are: their L = 256 lost to L = 128 only
because it cost 3.4x the iterations. Our 34x-cheaper iteration is a real asset
and should be spent on more and harder data, not more steps over the same data.

### Bearing on the beta-NLL decision

Their consistency result comes from **pure L2, no NLL term**. Heuristic and
learned-binary-contact baselines sat inside the 95% chi^2 band 18-20% of the
time (37.7% for GT contacts) and were *underconfident*; CoCo-InEKF improved on
that. Caveat: their NEES is on the core state (dof 9, band [2.7, 19]) and our
`nis_over_dof` is contact-*innovation* NIS at dof 6 — different quantities, do
not map one onto the other. But it weakens the premise that L2 must leave the
filter miscalibrated, and combined with the run-1 root cause above it suggests
**fixing the seeding before reaching for beta-NLL**.

---

## T* measured — and the frozen `contact_chol` is the primary cause (2026-07-28)

`experiments/measure_tstar.py`. From a truth seed, run the InEKF forward over
the same recorded inputs under three configurations, and record per-tick
body-frame velocity error (`losses.l2_velocity`'s exact integrand):

* **A** — training config: `contact_chol` frozen at `contact_chol_const` (stance),
  contact update ON (`Sigma_C = sigma_0^2 I`).
* **B** — training config, contact update OFF (`Sigma_C = 1e6 m^2`, gain ~0).
* **C** — deployed config: `contact_chol` as recorded (`ContactTrust` Schmitt
  trigger off `f_n/(0.5 m g)`, *not* ground truth), contact update ON.

1 rollout x 4 seeds, 5 s each. RMS body-frame velocity error [m/s]:

| horizon | A (train, on) | B (train, off) | C (deploy, on) | A/B | C/B |
|---|---|---|---|---|---|
| 10 ms | 4.39e-2 | 3.66e-3 | 9.51e-3 | 11.99 | 2.60 |
| 50 ms | 1.90e-1 | 1.74e-2 | 2.43e-2 | 10.92 | 1.40 |
| **128 ms (= L·dt)** | **3.69e-1** | **3.61e-2** | **3.32e-2** | **10.20** | **0.92** |
| 250 ms | 5.46e-1 | 3.74e-2 | 5.63e-2 | 14.61 | 1.51 |
| 500 ms | 5.81e-1 | 2.80e-2 | 7.85e-2 | 20.75 | 2.80 |
| 1 s | 6.26e-1 | 3.53e-2 | 9.40e-2 | 17.74 | 2.66 |
| 2 s | 6.29e-1 | 1.35e-1 | 9.70e-2 | 4.66 | 0.72 |
| 4 s | 6.47e-1 | 5.33e-1 | 8.73e-2 | 1.22 | **0.16** |

`T*` (A vs B) = **4.24 s**, 33x the segment horizon.

### The contact update is not the problem — the training config is

**C shows the contact update works.** In the deployed configuration it beats
dead reckoning from ~2 s and is **6.1x better at 4 s**. The filter is fine.

**A shows the training configuration breaks it.** Freezing `contact_chol` at the
stance value pins *swing* feet as world-static, so the contact update fights a
process model that is wrong for half the gait. The result is 10-20x **worse**
than not using contacts at all, at every horizon out to 4 s.

So run 1's `Sigma_C -> infinity` was not merely the consequence of a short
horizon. It was **the correct response to a broken process model**: with swing
feet pinned, the only way the network can stop the contact update from injecting
0.4 m/s of velocity error is to switch it off globally. It did exactly that.

This revises the diagnosis in "Run 1 learned to switch the contact update OFF".
The force-teacher seeding is real but **secondary**. Ranked by measured effect:

1. **Frozen `contact_chol`** — 10.2x penalty at the segment horizon. Primary.
2. **Truth-seeded 128 ms horizon** — even with the process socket correct, C is
   only 8% better than doing nothing at 128 ms (C/B = 0.92), rising to 6.1x at
   4 s. The gradient signal for `Sigma_C` at 128 ms is marginal and probably
   noise-dominated. Real, but it is a weak-signal problem, not a wrong-sign one.

### This reopens a documented decision

`dataset.py` docstring (b) freezes `contact_chol` deliberately: *"A training
segment that kept it would hand the network a free ground-truth contact flag and
void the experiment ... the swing/slip signal has to come out of ContactNet's
measurement covariance, which is the only channel it owns."*

The leak concern does not survive inspection. **The network never sees
`contact_chol`.** Its inputs are the 24 feature channels (gyro, accel, q, tau,
p_bc, v_bc); `contact_chol` enters the *filter's process model* only. Feeding
the real value changes the filter ContactNet is differentiated through, not the
information ContactNet receives. There is no path from `contact_chol` to the
network's input.

The freeze also fights theory-doc S3.2 / the `inEKF/filter.py` DECISION, which
argues that contact condition belongs in the **process** noise and that the FK
measurement is not wrong during swing. Freezing the process socket removes the
correct lever and then asks the measurement socket to compensate — which is the
thing that DECISION says not to do.

Note the recorded signal is deployable: `sim/sensors.py` drives `contact_chol`
from `ContactTrust` (the Schmitt-trigger port of
`FootSwitchContactProbabilityProvider`), off normal force `f_n/(0.5 m g)`. It is
a sensor-derived contact estimate, the same one the deployed filter uses — not
the sim's binary contact truth.

---

## The run-2 fix: chained segments + unfrozen process socket (2026-07-28)

Both causes from the two entries above, addressed. Commits `09c2d39`, `f778e60`.

### 1. `contact_chol` passes through (primary, 10.2x)

`ContactNetConfig.freeze_contact_chol` defaults to `False`; `make_segment` no
longer overwrites the field. `True` reproduces run 1.

The leak argument that motivated the freeze does not hold — the network's input
is the 24 feature channels and `contact_chol` reaches only the *filter's process
model*, so the value changes the filter ContactNet is differentiated through,
not the information it receives. `sim/sensors.py` drives it from `ContactTrust`
(Schmitt trigger off `f_n/(0.5 m g)`), a sensor-derived estimate, so the deployed
filter and the trained one now see the same signal.

### 2. `dataset.ChainedBatcher` (secondary, weak-signal)

`B` chains walk the rollouts in order, carrying `(X̂, P)` **and the gravity
reference** between steps. Lifecycle: seed from truth → warm in `warm_in_s`
untrained → walk `L` ticks per step → re-seed on episode end, rollout end, or a
non-finite carry.

* **`warm_in_s = 2.0`**, from the `T*` table: with the process socket correct the
  contact update is 0.92x of dead reckoning at 128 ms, 0.72x at 2 s, 0.16x at
  4 s. Two seconds is where it is clearly earning without spending the episode
  warming up.
* **`episode_s = 20.0`** bounds drift. CoCo-InEKF uses 100 s / 6 s for the same
  purpose.
* **The carry is `stop_gradient`'d.** Truncated BPTT is the only reason `L`
  bounds anything. Note this is a *different* edge from CLAUDE.md §7's "no
  `stop_gradient` between provider output and either filter" — that forbids
  detaching `Σ_C`, not detaching the carry across a truncation boundary.
* **Carrying the gravity reference retires open item 6** from the 2026-07-27
  report (`make_segment_loss` cold-started it every segment, τ = 5 s against
  0.128 s).

**Phase stagger — found live, worth recording.** The first run-2 launch seeded
every chain at `ticks = warm_in_ticks`, so all `B` reached `episode_ticks` on the
same step: they re-seeded in a synchronised wave and then marched in lockstep,
leaving the batch permanently at one common time-since-seed. Any phase-dependent
effect is then perfectly correlated across the batch, costing most of the `B`
independent samples the batch is sized by. Visible as the reseed counter jumping
12 → 37 between steps 100 and 150. Initial phase is now uniform over
`[warm_in_ticks, episode_ticks)`; the counter climbs at a steady ~0.29/step.

**Known cost, not fixed:** `_seed` calls `make_segment` purely to obtain
`state0`, which gathers a full window tensor and discards it, and each re-seed
runs a 2000-tick warm-in scan. Run 2 costs **0.53 s/step against run 1's
0.196** — ~2.7x. Worth reclaiming before a larger dataset makes it matter.

### The gate: `experiments/alpha_sweep.py`

Sweep a global scale `alpha` on `Sigma_C`, evaluate the real training loss, and
require an interior `argmin`. This is theory-doc §7.2.1 Claim 1 pointed at the
objective actually in use rather than at the β-NLL quadratic term, and it is the
check that would have caught run 1 in thirty seconds.

| config | argmin | verdict |
|---|---|---|
| run 1 (frozen chol, truth-seeded) | `alpha = 1e4` (largest) | **FAIL** — monotone; loss 4.44e-2 → 3.25e-4, a 136x improvement bought purely by disabling contacts |
| run 2 (chained, pass-through) | `alpha = 1e2`, i.e. `Sigma_C` ≈ 1 cm | **PASS** — interior, and the loss *rises* beyond it |

### `experiments/check_sigma.py`

Reports what a checkpoint does to the contact-update Kalman gain on the measured
`P0`. Loss curves cannot see the run-1 failure; this can. On a 60-step chained
smoke train:

| | run 1 (10k steps) | run-2 smoke (60 steps) |
|---|---|---|
| velocity-gain suppression | **3835x** | **1.2x** |
| `Sigma_C` median std x/y/z [m] | 0.68 / 0.17 / 0.27 | 1.1e-4 / 3.3e-3 / 1.4e-2 |
| conditioning proxy | 7.7e7 | 8.2e5 |

The learned anisotropy is physically readable: tightest along x, loosest in z at
~1.4 cm, which is where sole compliance and ground penetration live — and it
agrees with the alpha sweep's independent interior optimum of ~1 cm.

### Tests

29 in `tests/contactnet` (full suite **715 passed**). The 8 `ChainedBatcher`
tests were mutation-checked against 8 mutants, **8/8 caught**. One initially
survived: `test_cursor_starts_after_the_warm_in` derived the seed index *from*
the cursor it was meant to verify — a circular assertion, the exact failure mode
that file's docstring warns about. It now locates the warm-in slice's last
`omega` row back in the source rollout instead.

---

## Run 2 works — and the gain-suppression gate misled me twice (2026-07-28)

**`experiments/replay_eval.py`**, 2 rollouts x 2 seeds, 20 s each from a truth
seed, identical inputs, differing only in `contact_meas_chol`:

| metric | heuristic `σ₀²I` | run-2 `Σ_C` | ratio |
|---|---|---|---|
| body-frame velocity RMS | 0.0844 m/s | **0.0254** | 0.301 |
| position RMS | 0.790 m | **0.245** | 0.310 |
| **height RMS** | 0.780 m | **0.0918** | **0.118** |
| **height final** | 1.347 m | **0.125** | **0.093** |
| mean tilt | 0.689° | **0.353°** | 0.513 |

**The learned `Σ_C` improves every axis measured**: 3.3x in velocity, 3.2x in
position, 8.5x in height, 10.8x in terminal height, 2x in tilt. This is the
first genuine ContactNet result in the project.

Note the height number against the known "vertical drift at touchdowns" issue —
the heuristic's 1.35 m of terminal height error over 20 s *is* that problem, and
a trained `Σ_C` cuts it to 0.125 m with no reseed change at all.

### The methodological lesson

`check_sigma`'s gain table gave the wrong verdict **twice, in opposite
directions**, on the same checkpoint:

1. As `‖K‖_F` it said **1.6x, healthy**. The Frobenius norm is dominated by its
   largest column, so a live x axis masked y at 178x and z at 3115x.
2. Rewritten per-axis it said **DEGENERATE, z suppressed 3115x**. Also wrong —
   the filter is 8.5x *better* in height with that suppression than without it.

**A gain ratio cannot distinguish "switched off because the objective was
degenerate" from "switched off because that residual direction is
uninformative."** Run 1 was the first; run 2 is the second. The vertical contact
residual during walking is dominated by sole compliance, ground penetration and
terrain error — none of which is base velocity error — so down-weighting it
removes a bias source rather than discarding information.

Only running the filter separates the two cases. `check_sigma` now prints the
gain table as a **diagnostic** and defers the verdict to `replay_eval`.

### So is L2 "dead in the water"? No — it is incomplete in one measured way

L2 gets the **mean** right: the numbers above. What it does not touch is the
**covariance**. `nis_over_dof` finished run 2 at 7.5e-3, i.e. the filter is
~130x underconfident, and that is exactly theory doc §7.1's stated limitation
(L2 constrains the gain sequence and says nothing about the covariance) plus
§7.2.1 Claim 1 (the quadratic term fixes shape, not scale).

That split is the whole case for β-NLL, and it is now a measured split rather
than an argued one:

* **Consumers of the estimate** (the policy, which reads `v` and projected
  gravity) get a 3.3x better signal today.
* **Consumers of `P`** — NEES health monitoring, any MPC reasoning about
  uncertainty, the G10 consistency gate — get a covariance that is wrong by two
  orders of magnitude.

### Caveats on the number

* **All in-sample.** 12 rollouts collected, 12 trained on. This is "does it help
  the filter", not a generalisation test.
* The dataset narrowness entry still stands: ~1 100 contact events, one command
  velocity, no friction randomisation.
* 20 s horizons from a truth seed, not a closed-loop deployment.

### Timing, after the `episode_s` / `warm_in_s` change

`warm_in_s` 2.0 → 1.0 and `episode_s` 20 → 43 (the ceiling this dataset allows —
45.5 s usable per rollout, so a true 100 s episode needs re-collection at
`--seconds 120`+). Re-seeds 0.305 → 0.18/step, chain construction 48 s → 29 s,
and **0.598 → ~0.36 s/step steady state, 1.66x**. 100k steps would be ~10 h on
this box (an RTX 4070 SUPER, 12 GB — not a 4090).

Batch size does **not** buy throughput here: B=32 450 ms, B=64 1333 ms (2.96x),
B=128 2227 ms. Superlinear, so the GPU is not under-occupied at B=32.

## Run 3 — a clean-machine reproduction, and what the second box taught (2026-07-28)

A full from-scratch pipeline (collect → cache → norm → gate → train → gates) on a
**second machine**: WSL2, RTX 4090, **6 cores**, 19 GB RAM. Everything above was
measured on the 4070 SUPER / 20-core / native-Linux box. Run 3 used the current
defaults with no ablation flags: `--steps 10000 --objective l2_velocity --B 32
--no-remat`, chaining on, process socket passing through, `warm_in_s=1.0`,
`episode_s=43.0`.

### It reproduces run 2

`replay_eval`, 20 s horizons, all in-sample:

| metric | heuristic | run 3 | ratio | run 2 |
|---|---|---|---|---|
| body-frame velocity RMS | 0.0843 m/s | 0.0257 | **0.304** | 0.301 |
| position RMS | 0.789 m | 0.237 | **0.301** | 0.310 |
| height RMS | 0.779 m | 0.0767 | **0.098** | 0.118 |
| height final | 1.346 m | 0.0983 | **0.073** | 0.093 |
| mean tilt | 0.688° | 0.356° | **0.517** | 0.513 |

Better on both height metrics, matching elsewhere. `alpha_sweep` PASSED before
training (interior argmin at α=3.16e+01, Σ_C ≈ 3.2 mm). `check_sigma` showed the
run-2 signature — per-axis differentiation (x 1.2×, y 135.6×, z 3552.5×), not
run 1's uniform 2339–36157× on all three. `NIS/dof` finished at 7.3e-3 against
run 2's 7.5e-3, expected under L2.

**Two independent checks that the data pipeline is right, not just the outcome:**
measured P0 matched the documented `diag(R)=[6.9e-5, 6.9e-5, 1.0]`,
`diag(v)=7.6e-4`, `diag(p)=0.34` to three figures, and `prepare` found **exactly
545 772** legal segment starts — the documented count, from a freshly collected
dataset with different terrain seeds.

### The gate could never have run on a clean machine

`alpha_sweep` died with `FileNotFoundError` on `artifacts/p0.npz`. P0 is
*measured*, not configured, and `train_contactnet.py` was the **only** code path
that minted it; `alpha_sweep` merely `np.load`ed it. Every previous run had an
`artifacts/` left over from an earlier training run, so the documented order
("gate before training") had never actually been executed against an empty tree.
`--steps 1` is not a workaround either: `warmup_steps` clamps to `total_steps`
and then trips its own `warmup_steps < total_steps` guard, so a throwaway P0 run
must be `--steps 2 --warmup-steps 1`. Fixed: `alpha_sweep` now falls back to
`dataset.measure_p0` and caches, and both scripts `mkdir` the artifacts dir.

### The bottleneck is the host, not the GPU — and not the batch size

The 4090 trains at **0.632 s/step marginal** against the 4070 SUPER's 0.36 at
identical settings. A *better* GPU running 1.76× slower is the whole finding:
during training the card sits at **24% utilisation, 73 W of 337 W, P-state P2**,
with sustained ~112 MiB/s host→device traffic.

* **Any GPU upgrade is capped at ~1.3×.** Util is the fraction of wall time a
  kernel is running; 24% busy means deleting *all* GPU time is `1/0.76`. The
  workload is a `lax.scan` of L=128 **sequentially dependent** filter steps on
  15×15 float64 matrices — many tiny serial kernels, not big GEMMs. This is the
  same fact the superlinear `B` scaling reports from the other direction: batch
  size widens each kernel but cannot remove the 128 round trips.
* **Cores scale sublinearly.** 20c → 6c costs 1.76× on `train`, not 3.3×, so part
  of the host path (Python/JAX dispatch, NumPy segment gathering) is
  single-threaded. Expect ~2× from a 24-core box, not more.
* **`collect` inverts it.** It is ~88% `run_fused`, so the GPU absorbs it: 2.63
  s/sim-s here vs 3.2 on the 20-core box. `cache` is host-bound and loses badly
  (4.4 min/rollout vs ~2).
* WSL2 vs native Linux is confounded with core count in every number here and was
  **not** profiled. Both point the same way; the split between them is unknown.

### WSL2: the startup CUDA OOM spam is benign

XLA preallocates 75% of VRAM as *one contiguous block* (17.99 GiB), fails, and
walks down ~10% at a time. Nothing is lost — `bytes_limit` stays 17.99 GiB and
the allocator grows on demand. With 19 GB free: a single 8 GiB block **fails**,
8 × 1 GiB **succeeds**, largest single block bisects to **~3.6 GiB**. It is a
**contiguity limit, not a capacity limit**, from the paravirtualised WDDM
(`dxgkrnl`) path. Only bites if one tensor exceeds ~3.6 GiB; B=32 peaks near
242 MB. `XLA_PYTHON_CLIENT_PREALLOCATE=false` silences it. Note `nvidia-smi`'s
free-VRAM is not a valid health check on this box — these came from actual
allocation attempts.

### `measure-b` on a 24 GB card overturns the batch-size guidance

Re-measured on the 4090, GPU, L=128, AOT compile (the earlier 12 GB sweep asked
for exactly this):

| B | remat | peak [MB] | compile [s] | step [s] |
|---:|---|---:|---:|---:|
| 8 | off | 2866 | 25.7 | 0.25 |
| 32 | off | 3052 | 26.1 | 0.30 |
| 64 | off | 3206 | 25.7 | 0.30 |
| 128 | off | 3605 | 25.7 | **0.35** |

**16× the batch for 1.4× the step.** On the 4070 the same sweep was superlinear
(B=32 450 ms → B=64 1333 ms, 2.96×). So "larger `B` does not help" is a **12 GB**
statement, not a general one — at B=32 the 4070 was already saturated and the
4090 is not. `remat` remains a net loss on both (+0.02–0.09 s/step here).

Raising `B` still buys sample **throughput**, not sample **diversity**: at 12
rollouts, B=128 is ~10.7 segments per rollout, so the extra draws increasingly
replay trajectories already in the batch. The "more iterations is the wrong
purchase" argument applies unchanged — friction randomisation and a wider
`vx`/yaw sweep are still the better spend.

### The host overhead is now measured, not inferred

`measure_one` times a single compiled `grad_fn` on a **pre-built** batch
(`train_contactnet.py:226-228`): no `ChainedBatcher`, no re-seed warm-in, no
segment gathering. At B=32/no-remat that step is **0.30 s**, against the real
training loop's **0.632 s/step marginal**.

So **~0.33 s/step — 52% of wall time — is host-side work outside the jitted
step**, decomposing roughly into the documented ~0.18 s/step of re-seed warm-in
plus ~0.15 s of batch construction and dispatch. This supersedes the 24%-GPU-util
*inference* with a direct measurement, and it localises the only optimisation
worth making: neither the GPU nor `B`, but the host path between steps.

---

## Run 3 is a reproducibility result, not a generalisation one (2026-07-28)

Run 3 was trained from scratch on a second machine (WSL2 / RTX 4090 / 6 cores)
with its own re-collected dataset. Reproduced locally, its `replay_eval` numbers
match the 4090 box's to the digit.

### What it establishes

**1. The pipeline is deterministic and portable.** The 4090's rollouts and this
machine's are the *same* simulation: identical seed, spawn pose, injected
`true_gyro_bias`, `travelled_m` (23.572) and `tilt_max_deg` (2.504). Measured
`P0` agrees to **1.3e-12**. The ~300-byte file-size differences are float
rounding under compression, nothing more.

**2. The learned function is stable across trainings.** Run 2 and run 3 are
independent trainings under *different chain schedules* (`warm_in_s`/`episode_s`
2.0/20 vs 1.0/43). Their `Sigma_C` on the same 600 ticks:

| axis | run 2 median [m] | run 3 median [m] | per-tick correlation |
|---|---|---|---|
| x | 1.668e-3 | 1.729e-3 | **0.995** |
| y | 4.634e-2 | 4.064e-2 | **0.948** |
| z | 1.962e-1 | 2.135e-1 | **0.967** |

Median relative difference of `Sigma_C` is 0.122. The network is learning a
reproducible function of the features, not landing somewhere arbitrary in a flat
valley — which was a live worry given L2 says nothing about absolute scale.

And the filter outcome tracks:

| metric (ratio vs heuristic) | run 2 | run 3 |
|---|---|---|
| velocity RMS | 0.301 | 0.304 |
| position RMS | 0.310 | 0.301 |
| height RMS | 0.118 | 0.098 |
| height final | 0.093 | 0.073 |
| tilt | 0.513 | 0.517 |

### What it does NOT establish

**`data/data/` is not a held-out set.** It is the same deterministic run
re-executed, so run 3 tests *reproducibility*, not generalisation. Zero new
evidence about unseen conditions.

**The replay numbers are still in-sample**, on the same trajectories the network
trained on. Run 3 being "better on both height metrics" is two samples of the
same estimator differing by ~2% — the `Sigma_C` correlation above says these are
the same solution, so that gap is noise and should not be read as an
improvement.

**Nothing here moves the dataset gap.** Still one command velocity, 4 terrains,
3 seeds, no friction randomisation, ~1 100 independent contact events.

So confidence in *reproducibility* is now high and confidence in *generalisation*
is unchanged at zero evidence. That is the reason not to wire run 3 into the
deployed filter yet.

### Two findings from that run worth keeping

**The 12 GB "larger B does not help" was card-specific.** On 24 GB, `measure-b`
gives 16x the batch for 1.4x the step — so the superlinear scaling measured here
(B=32 450 ms, B=64 1333 ms) was memory pressure, as that entry suspected.
Re-measure per card. Note what it does and does not buy: lower gradient variance
and better rollout decorrelation (B=512 is ~43 chains per rollout), but the
segments still come from the same ~1 100 contact events.

**The loop is host-bound, not GPU-bound.** A compiled `grad_fn` on a pre-built
batch is 0.30 s/step at B=32 against the real loop's 0.632 s/step marginal — so
**52% of wall time is host-side work outside the jitted step**. That is
`ChainedBatcher`'s per-step Python: 32 `make_segment` gathers off 130 MB-backed
arrays, the stack, and the host-to-device transfer. It explains why the 4090 box
was *slower* per step (0.683) than this 4070 SUPER (0.598): 6 cores against 20.
It also means larger `B` is close to free, since the host cost is per *step*.
This is now the top performance item, ahead of anything GPU-side.

### Housekeeping

`data/data/` holds a 1.8 GB duplicate of the dataset from the rsync.
`dataset.rollout_paths` globs non-recursively, so it is not silently picked up —
but it is wasted disk and a trap for anyone pointing `--data` at it.

---

## What the training set actually contains, measured and rendered (2026-07-28)

`experiments/render_rollout.py` re-simulates a saved rollout with a camera on it
(videos in `artifacts/video/`), and `experiments/dataset_stats.py` counts what is
in it. The renderer re-simulates rather than replays — the `.npz` stores
`truth.R/v/p` and sensors but no `qpos` — which is sound because
`collect._RecordingLoop` builds its observation from `MjData`, so the estimator
is not in the control loop and physics + policy + spawn reproduce the trajectory.
**Checked, not assumed:** on `flat_seed000` the re-simulation lands within
**0.4 mm** of the recorded base position after the full 60 s walk.

### The four terrains are contact-wise indistinguishable

| rollout | events | duty | relief [m] | travel [m] | tilt_max [deg] |
|---|---|---|---|---|---|
| flat/s0-2 | 94, 94, 94 | 0.639 | 0.000 | 23.57-23.66 | 2.30-2.51 |
| stepping_stones/s0-2 | 93, 94, 95 | 0.632-0.635 | 0.030 | 23.70-23.86 | 3.33-3.60 |
| hard_stepping/s0-2 | 94, 95, 96 | 0.630-0.634 | 0.070 | 23.49-23.76 | 3.63-5.24 |
| waves/s0-2 | 94, 95, 96 | 0.637-0.642 | 0.100 | 23.37-23.89 | 2.66-3.49 |

Across all 12: contact events **93-96**, stance duty **0.630-0.642**, distance
**23.37-23.89 m**. `tilt_max` (2.30-5.24 deg) is the *only* column the terrain
moves. Terrain randomisation is changing the ground under the robot and almost
nothing about the contact process the network sees.

**Totals: 1 134 contact events over 552 s** — confirming the ~1 100 estimate the
dataset-narrowness entry reasoned from. ~47 steps per foot per rollout, ~1 s
stride, ~27% double support.

### Slip cannot be recovered from the saved rollouts

The reconstruction `W v_C = v_B + R(omega x p_bc) + R v_bc` is exactly zero for a
planted foot. It is not zero: in deep mid-stance (trust > 0.95, eroded 60 ms per
edge) it reads **0.29 m/s p50 against a 0.406 m/s base speed**, and an
independent finite difference of the world contact point agrees to three digits,
so it is not an algebra error.

It is also not 0.29 m/s of sliding. `p_bc` is FK to the **sole site**, not the
contact patch, so a foot rolling heel-to-toe carries that site through the world
without sliding at all — a few tens of cm over a 0.64 s stance is 0.2-0.3 m/s by
itself. Differentiating a noisy FK at 1 kHz adds more. **Nothing in the recorded
signals separates roll from slip:** the rollouts store no contact-patch position
and no contact forces.

So the slip measurement this file has been asking for since the narrowness entry
**must go into the collector** — per-contact tangential sole velocity while
loaded, taken where MuJoCo still knows the contact geometry — and cannot be
recovered afterwards. Note the first attempt at it reported the *body-frame*
`v_bc` (p50 0.466 m/s) as slip; that is just the base walking at 0.4 m/s, since
`v_bc` is a body-frame derivative and a planted foot reads ~ the base speed by
construction.

---

## The learned gate is a stride-phase clock — and friction randomisation is feasible (2026-07-28)

### Correction: runs 2/3 do not "switch off y and z"

That reading was a **median artifact**. Evaluating the learned `Sigma_C` densely
and taking the velocity-row gain at percentiles *of the learned distribution*:

| pct of learned std | std_x | std_y | std_z | supp_x | supp_y | supp_z |
|---|---|---|---|---|---|---|
| 1 | 1.0e-4 | 1.0e-3 | 4.5e-3 | 1.0x | 1.1x | **2.6x** |
| 10 | 1.3e-4 | 1.1e-2 | 1.9e-2 | 1.0x | 9.9x | 30.8x |
| 50 | 1.8e-3 | 4.7e-2 | 2.0e-1 | 1.2x | 174x | 3105x |
| 75 | 2.7e-1 | 1.1e-1 | 4.6e-1 | **5715x** | 1015x | 16552x |
| 99 | 7.5e-1 | 3.4e-1 | 8.6e-1 | 44654x | 9063x | 58015x |

The network **gates in time, not by axis**. At ~10% of ticks it trusts `z` to
within 2 cm; at ~22% it turns `z` off entirely; and at the 75th percentile it
turns **x** off too. Every axis is trusted sometimes and rejected sometimes.
That is why `replay_eval` improves every metric — the network is choosing *when*
to believe the feet, not *which axis*.

### What the gate is keyed on: gait phase, not contact condition

On `flat_seed000`, regressing `log10 std_z` on time-since-last-touchdown:

* **R² = 0.790** from **gait phase alone**
* R² = 0.020 from the `ContactTrust` signal
* correlation with trust: −0.143

So ~79% of the learned covariance is a **stride-phase clock**. With one gait,
"contact quality" and "stride phase" are the same variable and the network cannot
tell them apart — a clock is the most it can learn, and more of the same gait
teaches it nothing new. Runs 2 and 3 correlating at 0.97 is consistent: both
learned the same clock.

This reframes the dataset problem. It is not primarily terrain (measured
indistinguishable across our four) and not "no information in z" (`std_z` spans
192x p99/p1 and is genuinely trusted 10% of the time). It is that **the only
thing that varies is phase**.

### Friction randomisation: feasible, with a wide window

`experiments/friction_feasibility.py` sweeps floor friction and measures slip
from the **friction cone** — `|f_t| >= 0.99 mu f_n` via `mj_contactForce`, which
is Coulomb's law rather than a proxy, and is the collector-side instrumentation
the narrowness entry asked for. 15 s walks, flat/seed0:

| mu | walks | travel [m] | tilt_max | contacts | slip % | cone p50 |
|---|---|---|---|---|---|---|
| 1.40 | yes | 5.74 | 2.46 | 14173 | 0.62 | 0.098 |
| **1.00 (ours)** | yes | 5.75 | 2.50 | 13767 | **1.23** | 0.130 |
| 0.80 | yes | 5.76 | 2.49 | 13050 | 1.51 | 0.153 |
| 0.60 | yes | 5.76 | 2.48 | 11602 | 2.51 | 0.180 |
| 0.45 | yes | 5.76 | 2.44 | 10405 | 3.00 | 0.237 |
| 0.35 | yes | 5.73 | 2.35 | 9375 | 4.84 | 0.298 |
| 0.25 | yes | 5.82 | 2.45 | 8594 | 9.38 | 0.402 |
| 0.15 | yes | 5.56 | 2.12 | 7310 | **30.15** | 0.658 |

**The policy walks at every friction tested, down to mu = 0.15** — travel
5.56-5.82 m and tilt_max 2.12-2.50 deg throughout — despite its own DR range
being only [0.8, 1.4] (`alexander-mujoco/.../randomize.py`). Slip rises
monotonically and is 24x more frequent at mu = 0.25 than at our current 1.0.

Two corrections to earlier assumptions:

* **The sim is not slip-free today.** At mu = 1.0 the cone saturates on 1.23% of
  loaded contact-ticks. The narrowness entry's "possibly no slip at all" was too
  strong; the problem is that slip is *rare and probably phase-locked* (push-off),
  not that it is absent.
* Contact count *falls* with mu (14173 -> 7310), so lower friction shortens or
  lightens contacts as well as sliding them.

### The gate that decides whether any of this worked

Per-*rollout* friction randomisation varies the slip **rate between** rollouts,
but within a rollout slip may stay phase-locked — in which case the clock
survives and nothing is gained. That is measurable before committing to a large
collection: **collect a small friction-randomised batch and re-run the
phase-R² diagnostic.** R² dropping well below 0.79 is the evidence the
intervention worked; R² staying near 0.79 is the evidence that motion diversity
(dancing / mimic motions), not friction, is the necessary change.

---

## ContactNet closed the loop: z drift cut 8x in sim (2026-07-29)

The first closed-loop evidence in this project. Everything before this was
open-loop replay, where the estimate never affects the motion.

`run_estimator.py` gains `--contactnet CKPT` / `--contactnet-norm NORM`.
`estimator_loop.py` needed **no change** — it tree-maps over whatever
`init_fused_carry` returns, so `with_contactnet`'s 3-tuple carry flows through.

**Checkpoint/norm pairing is load-bearing:** run 4 <-> `data/dr/norm_constants.npz`,
run 2 <-> `data/norm_constants.npz`. A mismatch shifts the input distribution and
fails silently.

30 s, vx=0.6, `--imu-noise`, `--contact-fk measured`, 3 noise seeds. Headline is
signed vertical drift `est_z - true_z` (negative = estimate sinking into ground):

| mean of 3 seeds | no ContactNet | run 4 | run 2 |
|---|---|---|---|
| **final dz [m]** | **−3.183** | **−0.399** | −0.769 |
| dz RMS, last half | 2.432 | 0.303 | 0.583 |
| sink rate, last 20 s [m/s] | −0.107 | −0.0135 | −0.026 |
| tilt error, tail RMS [deg] | 1.201 | 0.252 | 0.658 |
| 3D drift, final [m] | 3.260 | 0.511 | 2.220 |
| velocity err, tail RMS [m/s] | 0.1327 | 0.0243 | 0.0437 |

No metric regresses. The robot itself walks fine in every arm (true height
0.88-0.91 m) — **the sinking is entirely in the estimate.**

**The wiring is proved live, not assumed.** Matched runs differing only in
`--contactnet` are **bit-identical through control tick 18** (max|diff| = 0.000e+00
on every leaf) and first diverge at tick 19 — exactly the 400-filter-tick
ring-buffer warm-up over 20 substeps. Identity-then-divergence at the predicted
boundary is the cheapest proof the network is not being silently dropped, and it
should be re-run after any change to the seam.

### The z drift has a specced fix that is deliberately absent

`inEKF/ekf.py` carries a TODO for `reseedContact` — the touchdown re-anchor
(`P_dd = P_pp + R N Rᵀ`, `P_θd = P_θp`, fire-once latch). CLAUDE.md §2/§3 require
it, G5 lists `InvariantEKFReseedTest`, and `tests/inEKF/test_invariant_ekf.py:271`
**asserts its absence**. The stated reason: *"Lucas measured no meaningful
difference on the real robot (2026-07-21)."*

That rejection may not transfer. The sim failure is **−3.18 m in 30 s**, far
larger than anything a hardware run would have shown, so "no meaningful
difference" was measured against a much smaller drift than the one ContactNet is
fixing here. ContactNet leaves ~1.4 cm/s of residual sink (~−0.8 m at 60 s); if
that residual matters, re-testing the reseed **in sim** is the open lever.

Note CoCo-InEKF has no reseed either — their Eq. (5) models a zero-mean contact
*velocity* in the process model, doing the same job continuously. Our port has
that socket too (`contact_chol`), unfrozen as of run 2.

## Structural safety: the network cannot make the filter over-trust contacts

Over 22 080 samples across all 12 DR rollouts: **zero non-finite, zero
non-positive**; SPD by construction from `L Lᵀ` plus softplus. `Sigma_C` does dip
below `sigma_0` (min 9.8e-6 m, on 3.6% of x samples), which looks like an
over-trust risk and is not one:

* `S = H P Hᵀ + N + Sigma_C` with **N = J Sigma_q Jᵀ = 1.26e-5 m²**. A `Sigma_C`
  of 1e-5 m contributes 1e-10 m² — five orders below N, hence invisible.
* The shipped analytic filter passes `contact_meas_chol = 0`, so "below sigma_0"
  is *closer* to shipped behaviour, not more aggressive than it.

**The encoder term floors the gain.** The network can only ever make the filter
more cautious, never less. That is a structural guarantee rather than a
statistical one, and it is the single most important property for live use.

## Acceptance criterion for deploying Sigma_C live

1. **Closed-loop A/B on unseen conditions** — tilt error and position drift no
   worse than the heuristic over >=30 s, on terrain and friction not in training.
   Gate 1 is now *partly* satisfied: passed on flat, 3 noise seeds. Not yet on
   `waves` / `stepping_stones`, and not at an untrained friction.
2. **No per-window regression** — the *worst* window must not be worse than the
   heuristic's worst. Mean ratios hide tail events; not yet measured.
3. **Structural check** — the floor result above. Passes; belongs in the test
   suite rather than a one-off script.
4. **NIS/dof in the chi² band** — only if anything downstream consumes `P` (NEES
   monitoring, an MPC reasoning about uncertainty, the G10 gate). Run 4 fails by
   ~100x (`nis_over_dof` 9.5e-3). This is beta-NLL's job, not more data's. The
   policy reads only the mean, so it does not block a policy-only deployment.

## Run 2 transfers out of sample — the first such number in the project

Run 2 was trained only on the original single-gait set and had never seen
randomised friction, disturbances or commands. Evaluated on the DR data it still
beats the analytic heuristic: **velocity 25%, position 50%, height 62%**. Every
previous ContactNet number in this project — including the 3.3x/8.5x quoted since
run 2 — was in-sample. ContactNet transfers; it just transfers less well than a
network trained on the conditions.

### Head-to-head, same rollouts / same P0 / each net with its own norm

The per-dataset `replay_eval` rows are NOT comparable across runs: DR rollouts
random-walk ~3.8 m while the originals march ~23.6 m, which moves the heuristic
denominator before any network is involved.

| metric | heuristic | run 2 | run 4 | run 4 vs run 2 |
|---|---|---|---|---|
| velocity RMS | 0.07369 | 0.05526 | **0.04046** | 27% better |
| position RMS | 0.37037 | 0.18425 | **0.06093** | 67% better |
| height RMS | 0.36367 | 0.13875 | **0.04930** | 64% better |
| height final | 0.61858 | 0.26306 | **0.09098** | 65% better |
| mean tilt | 0.56588 | 0.44297 | **0.35386** | 20% better |

### Suppression across runs — percentile form, never a median

| run | axis | @p10 | @p50 | @p90 | % ticks trusted (<2x) |
|---|---|---|---|---|---|
| run 2 | x | 1.0 | 1.2 | 15793 | 52.3% |
| run 3 | x | 1.0 | 1.3 | 15928 | 52.5% |
| **run 4** | x | 1.0 | **1.0** | **5.9** | **83.9%** |
| run 2 | y | 9.9 | 174.0 | 4251 | 2.5% |
| run 4 | y | 5.7 | 128.4 | 2589 | 3.4% |
| run 2 | z | 30.8 | 3105 | 34582 | 0.8% |
| run 4 | z | 183.6 | 2153 | 12277 | 1.1% |

The **medians barely move** (y 174->128, z 3105->2153, both inside the
run-2-vs-run-3 spread, and those two learned the same function at 0.97
correlation). The change is in the tail: runs 2/3 shut the forward axis
completely off at 10%+ of ticks (~16 000x); run 4's worst case is **5.9x** and it
keeps x live 83.9% of the time against 52.3%. That is the phase clock breaking —
a sharp phase-locked on/off switch replaced by continuous modulation.

## Two tooling defects found while doing the above

* **`alpha_sweep --B 8` is unsafe on any dataset containing standing.** It FAILED
  on the DR set — the documented hard stop — and is a false negative: at B=32 and
  B=64 the DR set lands on the same interior optimum the original set does. The
  DR set has **29.2% of ticks below 0.2 m/s** (against 3.7% originally), almost
  all of it the 17.3%-duty standing command, and those slow segments individually
  prefer `Sigma_C -> 0`. At B=8 two or three of them flip the batch argmin to the
  boundary. **Default `--B` must match the training batch size.**
* **`run_policy.cycloid_forearm_urdf` writes its hands-free URDF to a fixed
  `tempfile.gettempdir()` path.** N concurrent sim processes race on that one file
  and some read it mid-write, dying with `xml.etree.ElementTree.ParseError`. It
  cost 4 of 9 runs once and 1 of 3 sweeps earlier the same night. Workaround: a
  per-run `TMPDIR`. **Not fixed** — it is a real latent bug for anyone running the
  sim in parallel.

---

## Run 4 converged, and the residual z drift is a BIAS (2026-07-29)

Two measurements that together rule out the two obvious next steps.

### The loss converged — more steps buy nothing

| steps | mean loss | | steps | mean loss |
|---|---|---|---|---|
| 0-999 | 1.5922e-03 | | 5000-5999 | 1.2866e-03 |
| 1000-1999 | 1.3793e-03 | | 6000-6999 | 1.2718e-03 |
| 2000-2999 | 1.3067e-03 | | 7000-7999 | 1.2054e-03 |
| 3000-3999 | 1.3637e-03 | | 8000-8999 | 1.1973e-03 |
| 4000-4999 | 1.2699e-03 | | 9000-9999 | 1.2374e-03 |

Last 2000 vs previous 2000: **ratio 0.983** — 1.7% over 2000 steps, i.e. flat
from ~step 3000. `nis_over_dof` is likewise static (9.05e-3 -> 9.52e-3).
**A longer run is not the lever.** More *friction conditions* still might be;
more *steps* on this data are not.

### The sink is systematic, not accumulated noise

Fitting the closed-loop vertical drift against a line and against sqrt(t),
t > 2 s, seed 0:

| arm | slope | linear-fit resid | sqrt-fit resid | verdict |
|---|---|---|---|---|
| no ContactNet | −0.1079 m/s | **0.0076** | 0.1255 | **bias** |
| run 4 | −0.0135 m/s | **0.0023** | 0.0161 | **bias** |

`dz/t` is constant to three significant figures in both arms (run 4: −0.0122,
−0.0131, −0.0130 at 5/10/20 s). The linear fit beats the random-walk fit by 16x
and 7x respectively.

### Why this matters more than any hyperparameter

**A covariance cannot remove a bias.** `Sigma_C` sets *how much* the filter
listens to the contact measurement; it says nothing about that measurement being
*offset*. ContactNet cut the accumulation rate 8x by listening less — it did not
and structurally cannot drive it to zero, because `l2_velocity` contains no term
that could. No amount of additional data or training changes this.

It also rules out the two obvious next moves:

* **Harder disturbances / falls** add motion diversity but still only move a
  covariance. Worse, a fall creates contacts at knees, torso and hands while the
  filter has **N = 2 slots, both soles** — CoCo-InEKF runs that scenario with
  N = 10 body-wide points. The data would contain contact events the filter
  cannot represent.
* **Longer training** is converged (above) and could not fix a bias regardless.

### Three candidate sources, and the diagnostic that separates them

1. **The FK contact point is the sole SITE, not the contact patch.** Already
   observed independently: the reconstructed contact point moves **0.29 m/s in
   deep mid-stance** against a 0.406 m/s base speed, which is foot roll carrying
   the site through the world (see "What the training set actually contains").
   If that site also sits systematically above or below the true patch, the
   filter infers a wrong base height on every stance, in the same direction,
   every step — exactly a constant-rate sink.
2. **Ground penetration** in MuJoCo's soft contact: the foot settles into the
   floor, so the world-static anchor sits below nominal.
3. **No re-anchoring** — the deferred `reseedContact` would not remove the bias
   but would bound its accumulation.

**The diagnostic:** on a recorded rollout, compare the FK sole-site height at
mid-stance against the terrain height under that foot. A persistent offset
separates a **model bug** (fix the contact point — cheap, and it would also have
been quietly corrupting every `p_bc` feature ContactNet trained on) from a
**filter gap** (needs the reseed, previously rejected on hardware against a far
smaller drift). Run this before spending another GPU hour.

---

## The diagnostic ran: it is not the contact point, it is a velocity bias (2026-07-29)

`experiments/z_bias_diag.py`, read-only over the 12 DR + 3 control rollouts. No
filter run, no JAX — every number is already in the `.npz`.

### Candidate 1/2 (contact point, ground penetration) are RULED OUT

Reconstructing the world sole position from the **true** base pose and the cached
FK, `sole_w = p_true + R_true · y_fk`, a planted foot must be world-static.

| quantity | result |
|---|---|
| vertical drift within deep stance, pooled over 24 foot-rollouts | **+0.00213 ± 0.00034 m/s** |
| sole height above `z = 0`, flat terrain, 6 foot-rollouts | **+0.0055 m**, never below +0.0005 |

The drift is **positive** — the FK sole point rises ~2 mm/s during stance, the
opposite sign from the sink. The site sits a constant 5.5 mm above the ground and
**never penetrates**. Neither a wrong contact point nor MuJoCo soft-contact
settling is producing a per-step downward ramp.

**Methodology warning — the naive version of this measurement gives the wrong
answer with the wrong sign.** A tick-wise `np.gradient` over a stance mask returns
**−0.0096 m/s**, which looks like a confirmation of candidate 1. It is an
artifact: at each end of a stance block the centred difference straddles the
swing transition, and those few samples carry ~100x the interior magnitude.
Eroding 50 ticks per block and taking a *secant* across the interior flips the
sign. Per-**phase** statistics are also mandatory — 1 kHz samples inside one
stance are nearly perfectly correlated, so a per-tick standard error understates
by ~25x and would have made the artifact look significant.

### The sink is an integrated velocity bias

| relation | across 15 rollouts |
|---|---|
| `mean(est_v_z − v_z)` vs `slope(est_p_z − p_z)` | **r = +0.958**, mean ratio **1.068** |

The position error is the time-integral of a persistent negative velocity error,
to within 7%. The control set is the cleanest case: three seeds of steady walking
give `e_vz` = −0.0624 / −0.0629 / −0.0626 and a ratio of **1.059 on all three**.
A perfectly periodic gait produces a perfectly repeatable velocity bias.

So the question is not "what pushes the base down" but **"why does `v̂_z` sit
low"** — and `v` is observed by *nothing* in this filter (no `H` has a `ξ_v`
block), so it moves only by propagation or by `P_vp` coupling out of the contact
update.

### It is not the propagation

Measuring the world-z specific-force error the filter actually integrates,
`(R̂ a)_z − (R a)_z`:

* DR set: **−0.0018 to −0.0033 m/s²**, consistently negative;
* control set: **+0.0004 m/s²**, *positive* — while sinking the **worst**
  (−0.063 m/s, 2-3x the DR arms).

The sign flips with no corresponding flip in the sink, and `corr(a_err_z, e_vz)`
is **−0.726** — the wrong sign for a causal story. The propagation is exonerated.

Second-order tilt rectification was checked explicitly because it is a genuine
bias mechanism (`Δv̇_z = −½ g |δ|²` is negative for *any* tilt-error direction,
so a zero-mean attitude error rectifies into a downward push). It is the right
order of magnitude (−0.0006 to −0.0049 m/s²) but `corr(|δ|², a_err_z) = +0.278`.
Contributory, not causal. Worth remembering on hardware, where tilt error is
larger.

### What that leaves

By elimination, the velocity bias enters through the **contact correction**. The
`v`-row of `P Hᵀ` is `P_vp − P_vd`, which is a pure prior quantity: the
measurement noise `N` cannot change it, only scale the whole correction. That is
the same conclusion the apportionment argument reaches for `p`, and it lands on
the same lever — **the process socket** (`contact_chol`), which is not what
ContactNet is currently wired to.

Note this also re-reads the run-4 result. ContactNet cut the sink 8x by driving
`N_z → ∞` (median z suppression 2153x, "Suppression across runs" above). Under
the analysis here that works by shrinking the *whole* correction, including the
`v_z` pull — i.e. it bought the improvement by partially disconnecting the
contact update, which is exactly what `inEKF/filter.py`'s DECISION block argues
against. The gain is real; the mechanism is not the one the design intended.

### Not yet separated

* **No accelerometer-bias state exists anywhere in this port.** The joint KF
  carries `b_omega` only (`jointKF/state.py:12`); I1 keeps the InEKF pure
  `SE_{N+2}(3)`. CoCo-InEKF has both Eq. (6) and Eq. (7). `sim/sensors.py`
  injects no accel bias, so this is invisible in sim — it is a **hardware**
  exposure, and on hardware it would produce exactly this signature.
* The per-tick contact innovation is not recorded in the rollouts, so the
  `P_vp − P_vd` claim is inferred by elimination rather than measured directly.
  Logging `contact_innovation` in the collector would close that gap.

---
## ContactNet moves to the process socket (2026-07-29)

Runs 1–4 trained ContactNet into `contact_meas_chol`, the FK **measurement**
noise. It now drives `contact_chol`, the stance-anchor **process** noise;
`contact_meas_chol` returns to zeros, which is the shipped analytic filter
exactly. `network_plan.md` §1 and the "two contact covariance sockets" note above
are superseded on this point. Deferred work is in `TODO.md`.

### The structural argument

For one contact `H = [0 0 I −I]` over `(R, v, p, d)`, so `K = P Hᵀ (H P Hᵀ + N)⁻¹`
and `N` appears **only inside the inverted factor**. It scales the correction and
reweights residual axes; it cannot change how a residual is *apportioned* between
the base and the anchor. That apportionment is pure prior, hence pure process
noise. The residual sink is an integrated velocity bias (`r = +0.958`, ratio
1.068 over 15 rollouts — see "the diagnostic ran" above), so it lives on exactly
that split. CoCo-InEKF Eq. (5) puts the learned covariance in the same place (the
network is called inside Prediction, Alg. 1 line 1).

The argument bounds what `N` can do to the *split*. It does **not** say `N` is
powerless against the sink — see the arm-D result below.

### The ablation that gated it — `experiments/process_socket_ablation.py`

Recorded rollouts, no network, no training; every arm replays the same rollout
from the same truth seed and differs in one field. Arm B loosens the anchor `N`
ticks **before** liftoff (non-causal, a mechanism test, not a deployable filter),
arm C sweeps the stance value up, arm D is run 4 on the measurement socket as a
control. `data/dr`, 3 rollouts × 2 truth seeds × 20 s, `P0 = p0_dr.npz`:

| arm | slope(e_pz) [m/s] | vel_rms | height_rms | tilt [deg] |
|---|---|---|---|---|
| A heuristic | −0.03750 | 0.0771 | 0.3980 | 0.590 |
| B −0 ticks | −0.03750 | 0.0771 | 0.3980 | 0.590 |
| B −10 | −0.03110 | 0.0680 | 0.3303 | 0.534 |
| B −25 | −0.02290 | 0.0571 | 0.2440 | 0.463 |
| B −50 | −0.01253 | 0.0461 | 0.1358 | 0.381 |
| B −100 | **−0.00200** | 0.0416 | 0.0338 | 0.323 |
| C stance 1e−4 → 1e−2 | −0.03750 → −0.03767 | | | flat |
| D run 4 on `N` | −0.00490 | 0.0390 | 0.0548 | 0.337 |

Arm B at 0 ticks is bit-identical to A, which is the harness's own self-check.
The response is **monotone in the shift and 19x at 100 ticks**, with `height_rms`
down 12x and tilt down 1.8x.

### Arm C as specified was a floor artifact — corrected

`branch_out.md` §1's arm C sweeps the stance value `1e-4 → 1e-3 → 1e-2` and the
table above shows it flat. That "flat" is **not** a statement about tightness:
`contact.apply_floor` adds `contact_floor = 1e-4` to the digested covariance, so

| chol | `Σ = chol²` | digested | floor's share |
|---|---|---|---|
| 1e−4 | 1.0e−8 | 1.000e−4 | **100%** |
| 1e−3 | 1.0e−6 | 1.010e−4 | 99% |
| 1e−2 | 1.0e−4 | 2.000e−4 | 50% |
| 1e−1 | 1.0e−2 | 1.010e−2 | 1% |
| 1e0 | 1.0e0 | 1.000e0 | 0% |

The whole specified sweep lives inside the floor's saturation region — it moved
`Σ_C` from 1.000e−4 to 2.000e−4, a factor of two. Flatness was guaranteed.

Swept where the value actually varies (same 3×2×20 s protocol):

| stance chol | slope(e_pz) | vel_rms | height_rms | tilt [deg] |
|---|---|---|---|---|
| A (1e−4) | −0.03750 | 0.0771 | 0.3980 | 0.590 |
| 1e−2 | −0.03767 | 0.0718 | 0.3973 | 0.452 |
| 1e−1 | −0.03435 | 0.1057 | 0.3268 | 0.326 |
| 1e0 | **+0.00642** | **0.4539** | 0.1551 | 0.557 |

So tightness **is** a lever above the floor — and a **harmful** one. At stance
1e0 the sink flips sign, but `vel_rms` degrades **5.9x**: uniformly loosening the
anchor in stance removes the sink by removing the contact update's velocity
information altogether. Compare arm B at −100 ticks, which reaches a comparable
sink (−0.0020) while *improving* `vel_rms` to 0.0416, better than baseline.

**Two distinct levers, one good and one bad**, both in the process socket:

* **Release timing** (arm B) improves the sink *and* velocity. This is the one to
  learn.
* **Uniform magnitude** (arm C) trades velocity for sink. This is the one run 1
  found when it drove `Σ_C → ∞`.

That is a useful property of the objective rather than a hazard: `l2_velocity`
scores exactly the quantity arm C degrades, so the loss rewards the timing lever
and penalises the magnitude one. A `Σ_C → ∞` collapse should be self-limiting on
this socket in a way it was not on the measurement socket.

### The floor costs the network four decades

At the heuristic's stance value the floor supplies **100%** of the digested
covariance, so a network emitting anything below chol ≈ 1e-2 has **no effect on
the filter and therefore no gradient**. The usable output range is chol ∈
[1e-2, 1e1] — three decades, not the nominal seven. Two consequences:

* `ContactNetConfig.sigma_0 = 1e-4` initialises the network *inside the dead
  zone*. That is worse than the run-1 configuration, which at least had an
  effect; see `TODO.md` item 1.
* A network cannot express a stance tighter than the floor no matter what it
  emits.

### The floor is load-bearing — swept, not assumed (`--arms F`)

Same 3×2×20 s protocol, heuristic `contact_chol`, only `InEKFParams.contact_floor`
varying:

| `contact_floor` | slope(e_pz) | vel_rms | height_rms | tilt [deg] |
|---|---|---|---|---|
| 1e−4 (shipped) | −0.03750 | 0.0771 | 0.3980 | 0.590 |
| 1e−5 | −0.03388 | 0.2072 | 0.3616 | 2.745 |
| 1e−6 | −0.02103 | **0.6536** | 0.2419 | **10.403** |

Lowering it buys a marginally smaller sink and costs **8.5x in velocity and 17.6x
in attitude**. This reproduces the 2026-07-21 closed-loop finding (−15 m of drift,
18° of tilt, a fall) from a truth seed in 20 s, so that entry's conclusion — the
slack "absorbs contact/FK inconsistency; it is load-bearing" — is confirmed rather
than inferred. **Keep 1e-4.** The cost is that the learned `Σ_C` has three usable
decades instead of seven; that is the right trade.

### Where a run actually starts (`--arms I`)

`network.init` zeroes the output head, so iteration 0 emits a **constant**
`sigma_0 · I` for every foot at every gait phase — stance and swing alike. Arm C
only moves the stance value, so it does not answer this:

| init constant | slope(e_pz) | vel_rms | height_rms | tilt [deg] |
|---|---|---|---|---|
| A heuristic | −0.03750 | 0.0771 | 0.3980 | 0.590 |
| 1e−1 | **+0.00008** | 0.5136 | 0.0596 | 1.259 |
| 1e0 | +0.01162 | 0.5418 | 0.1825 | 0.688 |

A uniform constant is the arm-C trade in its extreme form: the sink and the height
error essentially vanish, and velocity accuracy degrades 6.7x. The filter is
degraded but **stable** — no fall, tilt ~1.3° — which is what makes it a usable
starting point.

`sigma_0 = 1e-1` was chosen for run 5 on that basis. Three properties, all of
which the old 1e-4 lacked:

* **It is inside the floor's live range** (1% floor contribution), so the
  network's output actually reaches the filter and therefore has a gradient. At
  1e-4 the floor supplies 100% and the output is inert.
* **The gradient is available where it starts.** `d softplus/dx = σ(x)`, which at
  the tight end equals the emitted value — so learning is starved *at* 1e-4 and
  healthy at 1e-1.
* **The loss starts high on the quantity it scores** (vel_rms 0.51 against the
  heuristic's 0.077), and the descent direction is "tighten in stance to recover
  the contact update's velocity information". A `Σ_C → ∞` collapse is where the
  run *begins*, and L2 penalises it — the run-1 escape route is closed by
  construction on this socket.

### Arm D falsifies the strong form of the claim

`branch_out.md` §0 concludes "ContactNet has been holding the one knob that
provably cannot reach it". Arm D — run 4, unchanged, on the measurement socket —
is **7.7x better than baseline** (−0.00490 against −0.03750). So the measurement
socket is not powerless on this metric.

The structural argument still holds as far as it goes: `N` cannot change the
base/anchor *split*. What it can do is scale every correction, and the sink is an
*accumulated* quantity — shrinking each dose shrinks the integral even with the
split untouched. The defensible claim is the weaker one: **the process socket is
the more direct lever, not the only one.** Practically, a process-socket retrain
has to beat −0.0049, not run 4's −0.0135.

### The mechanism is late stance, not early swing

`branch_out.md` §1 argues the sink is a rectified liftoff dose because `P_dd` is
"at its tightest exactly then, after a whole stance of being told it was
world-static". Measured on the **prior** covariance (`replay_eval.run_arm(
want_traces=True)`, reconstructed by replaying `propagate` — the posterior is the
wrong object, since `Σ_C` enters through `Q_d` and the update immediately shrinks
most of it back), over 234 liftoffs:

|  | `P_pp` | `P_pd` | `P_dd` | base share `f` | vel. gain `g_v` [1/s] |
|---|---|---|---|---|---|
| late stance | 3.368e−1 | 3.368e−1 | 3.368e−1 | 7.85e−1 | **2.673** |
| early swing | 3.369e−1 | 3.369e−1 | 4.369e−1 | 8.1e−7 | 2.78e−6 |
| late swing | 3.370e−1 | 3.370e−1 | 4.370e−1 | 9.0e−7 | 3.01e−6 |

1. **The asymmetry is stance-vs-swing, not early-vs-late swing** — and it is
   ~**10⁶** in `g_v`, not the modest effect §1 imagines. The Schmitt trigger
   releases *at* liftoff, so the anchor is already at its swing value on the first
   tick of swing (`P_dd` picks up `Σ_C Δt = 0.1` the moment it releases).
   Comparing two swing windows finds nothing, which is exactly what the first
   version of this diagnostic reported.
2. **The dose is injected in the last ~100 ms of stance**, while the foot is
   unloading and beginning to move but the trigger still says "planted": the
   contact residual reaches base velocity at `g_v ≈ 2.7 /s` there. That is why
   arm B works and improves monotonically out to 100 ticks, and why arm C does
   nothing — the stance *value* barely moves `g_v`, since in stance `P_dd` is not
   what limits the coupling.
3. **Base position is unobservable**, so `P_pp`, `P_pd` and `P_dd` agree to four
   digits and the naive `P_pp/(P_pp+P_dd)` reads exactly 0.5 forever. The
   informative row is `v`, which §0's argument names and §1's does not.

Consequence for the retrain: what the network has to learn is a **slightly early
release** — a leading indicator of unloading. The torque channels carry that. But
100 ticks is 100 ms, and `experiments/phase_lock.py` already showed this dataset
lets a net learn a stride-phase *clock* instead of a load signal; whether it
learns unloading or merely phase is the thing to check.

### Consequences of the move that are not optional

* **The warm-up fallback inverts.** `online.make_provider` emitted **zeros**
  before its ring buffer filled, because zeros in the measurement socket reproduce
  the shipped filter. Zeros in the process socket are `Σ_C = 0`: every anchor,
  swing feet included, asserted perfectly world-static — run 1's failure mode
  applied to every contact for ~400 ticks. It now falls back to
  `sensors.contact_chol`, the heuristic the caller already holds.
  `test_warmup_fallback_is_the_heuristic_not_zeros` is the regression.
* **Two floors now act on one quantity.** `contact_floor` (1e-4, a variance floor
  on the reconstructed `Σ`) and the network's `eps` (1e-6, a factor floor on
  `diag(L)`, contributing `eps² = 1e-12`). Not redundant and not interchangeable:
  `eps` keeps the softplus output strictly positive so the Cholesky
  parameterisation stays valid and differentiable; `contact_floor` is the physical
  bound on how world-static an anchor may be asserted to be. The effective floor
  is `contact_floor` alone at any sane setting. It is now **safety-critical** — the
  only thing between a mis-prediction and a pinned swing foot — and lowering it to
  1e-6 was already measured at −15 m of drift, 18° of tilt, and a fall. This
  corrects the note above, which had `eps` as the measurement-socket lever and
  `contact_floor` as the process one; both now act on the process socket.
* **`measure_p0` and `make_warm_in` change conventions.** Both burned in / warmed
  in at a *constant* `Σ_C`, described as "exactly the conventions a training
  segment runs under". False after the move: they now use the recorded heuristic,
  and `freeze_contact_chol` is the only remaining route to the old behaviour.
  `make_warm_in`'s `sigma_0` argument is optional and defaults to pass-through —
  the process socket has no "reproduces the shipped filter" constant to hold.
  **`artifacts/p0_dr.npz` is stale**; see `TODO.md` item 2.

### One correction found on the way

`filter.contact_velocity_noise` paired `J̇_C` with `Σ_q̇`. The world velocity of
contact `i` is `v + R(ω × h_i + J_{C_i} q̇)`, so the sensitivity of a measured
contact velocity to `q̇` is `J_{C_i}`, the *position* Jacobian. Two independent
checks: `J̇ Σ_q̇ J̇ᵀ` has units m²/s⁴ (not a velocity covariance), and
`pipeline/main_estimator`'s MJX kinematics returns `J_dot = 0`, so the old pairing
would have been identically zero on the deployment path while looking wired. The
function still has **zero call sites** — see `TODO.md` item 3 for why.

### An unrelated pre-existing test bug, fixed here

`test_step_does_not_recompile_across_contact_conditions` asserted
`step._cache_size() == 1`. It passes in isolation and fails with `0 == 1` under a
full-directory run: JAX's jit cache is a process-wide LRU, so an unrelated suite
evicts the entry and the assertion reads a *miss* as a recompile. Now compares the
lowered HLO across contact conditions, which says the thing under test directly
and cannot be evicted. (It was failing on `main` too — verified by stashing.)

---

## Run 5 — the first process-socket network (2026-07-29)

`sigma_0 = 1e-1`, `contact_floor` 1e-4, `l2_velocity`, `artifacts/p0_process_dr.npz`,
10 000 steps in 63 min on the 4070 SUPER. Checkpoint `artifacts/contactnet_run5.npz`.

Loss 3.96e-3 → **1.06e-3** (median of the last 1000), plateaued over the final
2000 steps. `applied_frac = 1.000` for the entire run — not one gated update, so
the loss scored real corrections throughout — and `cond_proxy_max` peaked at
1.8e4 against the 1e9 gate. The step-50 spike to 1.1e-1 is the chains
equilibrating away from their truth seed under the deliberately-loose init, not
divergence.

### Same weights, two sockets — the head-to-head

The only apples-to-apples statement about the move: one network, one `P0`, same
seeds and rollouts, differing *only* in which field it drives.

**In sample** (`data/dr`, 3×2×20 s, `p0_process_dr.npz`):

| arm | slope(e_pz) | ratio | vel_rms | height_rms | tilt [deg] |
|---|---|---|---|---|---|
| heuristic | −0.03751 | 1.162 | 0.0771 | 0.3993 | 0.590 |
| run 5 → `N` (meas) | −0.00696 | 1.572 | 0.0448 | 0.0806 | 0.328 |
| run 5 → `Q_d` (proc) | **+0.00356** | **1.063** | **0.0378** | **0.0636** | 0.365 |

**Held out** (`data/control`, never trained on; DR norm constants, which travel
with the checkpoint; `p0_process_control.npz`):

| arm | slope(e_pz) | ratio | vel_rms | height_rms | tilt [deg] |
|---|---|---|---|---|---|
| heuristic | −0.06694 | 1.081 | 0.0846 | 0.7825 | 0.690 |
| run 5 → `N` (meas) | −0.00606 | 1.562 | **0.0293** | 0.0855 | **0.303** |
| run 5 → `Q_d` (proc) | **+0.00517** | **1.044** | 0.0335 | **0.0483** | 0.390 |

Every acceptance criterion in `branch_out.md` §5 passes: the sink is 10.5x (in
sample) / 12.9x (held out) smaller than the heuristic, `ratio` stays ≈1, and
nothing regresses against the heuristic.

**What replicates and what does not.** The process socket is robustly better on
the **vertical** axis in both datasets — sink 2.0x / 1.2x and `height_rms` 1.3x /
1.8x better than the same weights on the measurement socket. On **velocity and
tilt** the two sockets are within ~15% and *the ordering flips between datasets*
(process wins in sample, measurement wins held out). So the defensible claim is
narrow and exactly on target: **the socket move buys vertical drift**, which is
what it was aimed at. It does not buy a uniform improvement, and anyone quoting
the in-sample velocity win as evidence for the move is over-reading it.

One mechanistic detail worth keeping: `ratio = slope(e_pz)/mean(e_vz)` is **1.04–1.06
on the process socket and 1.56 on the measurement socket**, in both datasets. On
the process socket the residual drift is still a clean integrated velocity bias;
on the measurement socket it is not, so part of it is position injection. That is
the §5 acceptance check doing real work rather than passing vacuously.

### Closed loop — the policy driven by the estimate

30 s at `vx = 0.6`, `--imu-noise`, `--contact-fk measured`, seed 0. Run 4's column
is the recorded table above.

| | no ContactNet | run 4 | **run 5** |
|---|---|---|---|
| final signed dz [m] | −3.194 | −0.399 | **−0.180** |
| dz RMS, last half [m] | 2.435 | 0.303 | **0.143** |
| sink rate, last 20 s [m/s] | −0.1079 | −0.0135 | **−0.0064** |
| 3D pos drift, final [m] | 3.279 | 0.511 | 0.501 |
| base velocity error RMS [m/s] | 0.1343 | — | 0.0304 |
| tilt err, tail RMS [deg] | 1.210 | **0.252** | 0.370 |

**2.1–2.2x better than run 4 on all three vertical metrics**, 17.8x better than no
ContactNet. The robot walks normally in both arms (18.9 m vs 20.1 m travelled,
true height held at 0.89 m).

### The cost is rotational, and it is yaw

`att_deg` (full 3D attitude error) *regressed* 1.13x rms / 1.34x tail against the
no-ContactNet baseline while `tilt_deg` improved 2.3x / 3.3x. Decomposing:

| | none | run 5 |
|---|---|---|
| tilt (gravity direction — observable) rms / tail | 1.276 / 1.210 | **0.546 / 0.370** |
| yaw component rms / tail | 0.884 / 1.168 | 1.669 / 2.225 |

The regression is **entirely yaw**, and it is structural rather than a bug. Yaw is
unobservable in this filter (`enableYawSeeding=false`; `H_g` is rank 2 with null
along `e_z` by construction), so the *only* thing constraining it is the contact
update — planted feet at distinct world locations. The network bought its vertical
win by loosening anchors, and loose anchors carry less yaw information. `trusted
feet avg` is 1.211 vs 1.208, so this is genuinely `Σ_C` and not the Schmitt
trigger behaving differently.

Two consequences:

* **`l2_velocity` has no attitude term.** Nothing in the objective was defending
  tilt or yaw; that tilt improved 3.3x anyway is incidental. If the rotational
  cost matters, the fix is an attitude term in the loss or `beta_nll`, not
  reverting the socket.
* Run 5 is decisively better than run 4 vertically and slightly worse
  rotationally (tilt tail 0.370 vs 0.252). It is a **different point on the same
  trade**, not a strict improvement.

### Still not calibrated

`nis_over_dof` sat at 2.2e-2 for the whole run (calibrated = 1.0) and the
closed-loop NIS tail mean went 0.69 → 0.20. The filter is ~5–45x
**over-conservative in absolute scale**, which is the known `l2_velocity` gap: the
quadratic term constrains only *ratios* of `S`, and the `logdet` term that fixes
absolute scale is exactly what `beta_nll` adds. Safe direction, but **G10's
NIS/NEES consistency bands will not pass on this checkpoint.** That is the next
run's job, not a defect in the socket move.

Notably, `nis_over_dof` stayed *flat* while the loss fell 4x. That is the
predicted signature of learning the right lever: `NIS` tracks the overall
magnitude of `S`, which is the lever that trades velocity for sink, and the
network improved velocity through timing and anisotropy instead — exactly the
split the two-lever sweep said `l2_velocity` would enforce.

---

## Toe/heel contact points — two per foot (2026-07-29)

CoCo-InEKF anchors **two** contact points per foot; this port anchored one, at the
sole centre. `build_fused_estimator(contact_sites=...)` /
`build_alex_fused_estimator(toe_heel=True)` / `run_estimator.py --toe-heel` now
give the InEKF four (`ALEX_CONTACT_SITES`), default off.

### Why, and it is not fidelity

A single contact point carries **no information about foot orientation**. Two
points 0.197 m apart do — which is exactly what run 5 gave up when it learned to
loosen the anchors (see "the cost is rotational, and it is yaw" above). So this
was tested as a candidate *fix for that specific regression*, not as a general
refinement.

### `K` and `N` are decoupled, deliberately

`foot_sites` used to set both the joint KF's `K` stance anchors and the InEKF's
`N` contacts. They are now separate: **the joint KF keeps one anchor per foot.**
Two anchors on one *rigid* foot are kinematically locked, so stacking them as
independent rows in the anchor block would double-count the bias information they
carry with nothing modelling the lock — the same class of error as the
block-diagonal `R_g` trap (I6). The InEKF has no such problem: its contacts are
independent *states*, not stacked measurement rows.

`FusedEstimator` gains `contact_site_ords` and an `n_anchors` property; `n_contacts`
is now `N`, which is no longer the same number.

### Geometry — the Java sole plate, not the collision box

`ACTUAL_FOOT_LENGTH = 0.197`, `FOOT_BACK = 0.052`, so in the `*_FOOT` frame the
plate spans x ∈ [−0.052, +0.145] and `ALEX_SOLE_OFFSET` is its midpoint. Heel and
toe go at the ends, both on the sole plane z = −0.072. The **collision box** is
more generous (x ∈ [−0.085, +0.175], half-extents 0.13 × 0.07 × 0.0275 at
(0.045, 0, −0.05)); anchoring at *its* corners would place the contact points 3.3 cm
beyond the physical plate. The sole sites remain the exact midpoints of their
toe/heel pairs, so the two builds are directly comparable.

The toe/heel sites are emitted in **every** build, used or not, so one
`alex_site_names()` ordering is valid for both and an `N = 2` rollout stays
readable against an `N = 4` model. Verified that adding the unused sites perturbs
nothing the estimator consumes: `site_poses`, `M`, `J_rel`, `J_ang` and `site_rot`
all agree to **exactly 0.0**.

### The load split needs no change to the collision geometry

There is one collision geom per foot, so per-point normal force is recovered from
contact *positions*: MuJoCo reports each contact's world position, and a contact is
attributed toe/heel by the sign of its x in the foot frame relative to the sole
centre (`sensors.point_loads`). The `(foot, fore) → slot` map is resolved at build
time **from the estimator's own contact-site names**, so the sim cannot disagree
with the filter about which slot is which foot's toe.

Measured over 15 s of walking, it tracks the roll of the foot correctly — heel-only
at spawn, flat mid-stance, and **toe-only during push-off** (`[1.0, 0.99, 0.0, 0.66]`
= left flat, right heel lifted). That last state is precisely what a single centre
point cannot represent.

| | load mean | trusted |
|---|---|---|
| left heel / toe | 0.453 / 0.491 | 47.5% / 60.3% |
| right heel / toe | 0.447 / 0.507 | 47.9% / 60.4% |
| joint-KF anchors (per foot) | — | 63.3% |

Toe is trusted more than heel, as toe-off implies. **A known cost:** splitting the
load means each point sees ~half of it, so each clears the `enter = 0.35` Schmitt
threshold less often than a whole foot does — per-point trust averages 54% against
the per-foot 63.3%, i.e. slightly fewer live anchors. The per-point normaliser is
still `0.5·m·g` (a whole foot's share); dividing by `0.25·m·g` instead would
restore the margin and is the obvious knob if anchor availability ever binds.

### Measured: it fixes the yaw regression, with no training at all

Closed loop, 30 s at `vx = 0.6`, `--imu-noise`, seed 0. **The toe/heel column has
no ContactNet attached** — this is the analytic filter.

| | N=2 analytic | **N=4 toe/heel** | N=2 + run 5 |
|---|---|---|---|
| final signed dz [m] | −3.194 | −0.567 | **−0.180** |
| dz RMS last half [m] | 2.435 | 0.432 | **0.143** |
| tilt err rms / tail [deg] | 1.276 / 1.210 | **0.430 / 0.240** | 0.546 / 0.370 |
| attitude err rms / tail [deg] | 1.552 / 1.682 | **0.806 / 0.929** | 1.756 / 2.255 |
| yaw component rms / tail [deg] | 0.884 / 1.168 | **0.682 / 0.897** | 1.669 / 2.225 |
| base vel err rms [m/s] | 0.1343 | 0.0347 | **0.0304** |
| base pos err rms [m] | 1.885 | 0.338 | **0.239** |
| base gyro err rms | 0.0116 | **0.0094** | 0.0101 |

Two contact points per foot, on the **heuristic** filter, cut the sink 5.6x, the
tilt error 5.0x in the tail, and the velocity error 3.9x — and they take yaw
*below* the baseline (1.168 → 0.897 tail) where run 5 had pushed it 1.9x above.
The hypothesis holds: the yaw cost was missing orientation observability, and two
points restore it.

**The two changes are complementary, not competing.** Run 5 owns the vertical axis
(dz −0.180 against −0.567); toe/heel owns everything rotational (yaw tail 0.897
against 2.225). Nothing yet combines them: a learned `Σ_C` over four contact
points is the obvious next run, and it needs a **full re-collection** — a rollout
stores `(T, N, 3, 3)` arrays, so every recorded rollout, the feature cache, the
norm constants, `P0` and the checkpoint are all `N = 2` artifacts.

### The closed loop is not bit-reproducible, so do not A/B it that way

Two **identical** `run_estimator.py` invocations differ by `1.4e-6` in final `dz`
(−3.1936414 vs −3.1936428). The policy↔estimator↔plant loop is chaotic and GPU
kernel selection is not fixed across processes, so rounding at 1e-16 amplifies to
1e-6 over 30 s. An earlier attempt to prove the `N = 2` path unchanged by
comparing two runs "bit-for-bit" therefore measured nothing — the real evidence is
the exact-0.0 agreement of every estimator input above. Effects below ~1e-5 in this
harness are not measurable; the toe/heel results are 1.3–5.6x, far above it.
