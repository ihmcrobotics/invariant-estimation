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

## G10 (part) — viewer tooling: ORT pinning, the ghost, threading, GPU (2026-07-27)

Four changes aimed at "the estimator loop is too slow to watch, and the comparison is
numbers-only". Two of the four landed roughly as designed; **two produced the opposite of the
predicted result**, which is the part worth reading.

### 1. The ONNX session was the speed problem (`run_policy._ort_session`)

`onnxruntime` defaults to one intra-op thread per physical core *and spins* after each `Run`.
Measured: a default session spawns **9 threads** on this 20-thread box and keeps them hot, so they
fight XLA's own pool in the gap between control ticks. The policy is a tiny MLP; one thread
computes it faster than nine can be synchronised.

Pinning it (`intra_op_num_threads=1`, `inter_op_num_threads=1`,
`session.intra_op.allow_spinning="0"`) takes the median control tick from ~27 ms to ~17 ms and
carries the loop across the real-time line: **0.67–0.83x → 1.12–1.24x**. Interleaved A/B, one
session per process (swapping sessions mid-process leaves the first pool alive and measures *more*
threads, not fewer — an earlier attempt at this measurement got the sign wrong that way).

The planning note had expected this to pay only in the **tail** ("the median is noisy"). On the
full loop the median moved robustly too, and it alone retired the "viewer runs at 0.6x" problem
that the other three features were designed around. Everything downstream had to be re-baselined
against it — which is why it landed first.

### 2. The ghost (`sim/ghost.py`)

A translucent second robot drawn at the estimated state: a second `MjData` on the same `MjModel`,
`mj_kinematics` only, never `mj_step`ped, appended to the scene with `mjv_addGeoms(mjCAT_DYNAMIC)`.

* `est.p`/`est.R` drop into the free joint with **no frame conversion** — verified: the ghost's
  pelvis lands on `est.p` to 0.0 and on `est.R` to 8e-16, because `qpos[0:3]` ≡
  `d.xpos[PELVIS_LINK]` and the free joint's quaternion is world-from-body like `est.R`.
* The dynamic pass draws **sites too** (32 meshes + 20 sites = 52 geoms), so the ghost carries a
  dedicated `MjvOption` with `sitegroup[:] = 0`. Collision geoms are group 3 and already hidden;
  the floor is static and excluded for free (the static pass adds exactly 1 geom).
* The 9 filtered joints come from `est.q`; the other 20 are copied from the real `qpos`, which is
  exactly what the estimator knows (on hardware those are raw encoders).
* Cost **0.083 ms/frame** (update 0.027 + draw 0.057). The planning note said 0.004 ms — 20x
  optimistic, still irrelevant against a 20 ms budget.
* Bound to keypad `*` (GLFW `KP_MULTIPLY` 332). MuJoCo reserves every letter A–Z for render
  toggles and `run_policy.py` raises on a sub-128 binding, so the keypad is not a style choice.

`run_free_viewer` grew an optional pre-built `loop` argument so `run_estimator.py --wasd` can drive
it; without that the ghost hook there would have been dead code, since `run_policy`'s own loop has
no estimator.

### 3. Threading works, and is no longer needed (`sim/estimator_thread.py`)

`ThreadedEstimator` wraps `EstimatorRuntime`; `estimator_loop.py` is untouched, because it is what
`test_sources_truth_bypasses_the_estimate` runs through at `atol=0`. Never drops a sample (plain
`deque`, no `maxlen` — a `maxlen` deque discards the *oldest*, the worst possible choice for a
sequential recursion); consumes **fixed `substeps` chunks** so XLA never retraces; scores against
the truth **paired** with each chunk rather than the current `MjData`.

Two corrections to the design:

* **Default `max_backlog_ticks` is 2, not 5.** Staleness tracks the allowed backlog almost exactly
  (age ≈ backlog + 1 ticks), and tilt error against the synchronous run degrades sharply past ~3:
  `1 → +0.099°, 2 → +0.112°, 3 → +0.456°, 5 → +1.598°`. The planned default of 5 failed the
  plan's own 0.3° acceptance gate by 5x.
* **It buys almost nothing now.** 0.97x headless, 1.03x with a render per tick. The 1.90x
  thread-overlap figure that motivated it was a micro-benchmark of JAX against `mj_step`; in the
  real loop, once ORT stopped stealing cores, the sim thread has too little work left to overlap.
  It stays opt-in, viewer-only, off by default.

**A liveness bug that only mutation testing found.** Back-pressure originally waited while
`self._error is None`. A worker that dies *without* recording an error (swallowed exception,
library `sys.exit`) then leaves the producer blocked forever — and the test that should have caught
it **hung instead of failing**, which is the failure mode that hides in CI. The wait is now gated
on `self._alive()`, and `_reraise` also raises when the thread stopped silently. With the fix, the
same mutant fails in 2 s instead of hanging.

### 4. The GPU wins — the prediction was backwards

`pyproject.toml` gained its first `[project.optional-dependencies]`: `gpu = ["jax[cuda13]"]`.
Every reason to expect a **loss** is still true — batch size 1, no `vmap` anywhere in the estimator
path, hundreds of tiny kernels, float64 at 1/64 rate on a consumer card, a blocking host
round-trip every tick. Measured anyway, RTX 4070 SUPER, 250 ticks at vx=0.6, three interleaved
repeats:

| device | build | p50 | xRT | tilt tail-RMS | drift |
|---|---|---|---|---|---|
| cpu | 54 s | 13.59–13.84 ms | 1.45–1.47x | 0.856° | 0.353 m |
| gpu | 71 s | **8.93–9.05 ms** | **2.21–2.24x** | 0.856° | 0.353 m |

**1.5x faster with the error columns identical to three decimals** — a real win, not a
speed/accuracy trade. Build+compile is ~17 s slower, paid once. The reasoning above was sound and
the conclusion was still wrong; `experiments/bench_estimator_device.py` exists so the question is
re-measured rather than re-argued.

Backend selection is the `JAX_PLATFORMS` env var, deliberately **not** a `--device` flag: the
backend must be chosen before `jax` is imported, and `run_estimator.py` imports the whole
`invariant_estimation` chain (which runs `jax.config.update` at `__init__.py:15`) at module scope,
before `argparse`. A flag would need an `sys.argv` scan above the imports — the same import-order
landmine the file already carries once for `MUJOCO_GL`.

The repo-root `conftest.py` pins the suite to `JAX_PLATFORMS=cpu` via `setdefault`, so installing
the extra cannot silently move MJX kinematics onto the GPU and shift tolerances measured on CPU.
Checked afterwards: `tests/sim` + `tests/pipeline` (61 tests) pass on CUDA as well, so the pin is
precaution rather than a workaround, and `JAX_PLATFORMS=cuda uv run pytest` remains available.
`uv lock` resolves all extras, so `uv.lock` carries a large `nvidia-*` block that a default
`uv sync` never downloads.

### Deliberate deviations from the plan

* `max_backlog_ticks` default 5 → **2** (the planned default failed the planned accuracy gate).
* `run_free_viewer` takes an optional pre-built loop, and `run_estimator.py` gained `--wasd`;
  without it the planned ghost hook in that viewer would have been unreachable.
* The threaded-vs-synchronous acceptance test runs **250 ticks, not 120**: `summarise` scores the
  tail half, and at 120 ticks that window is still inside the gait transient, where the two
  trajectories differ by more than the estimator does (a 120-tick version read 1.157 vs 0.738° and
  failed on startup noise alone).
* `--ghost` with `--headless`/`--video` is a hard error rather than a silent no-op, as is
  `--realtime` with `--headless` — the latter because a threaded run is not reproducible and must
  not be able to write an `.npz` or a video that looks authoritative.
* The plan's note about re-recording `experiments/sim_runs/*.npz` after the ORT change was moot:
  that directory has never existed in the repo (it was an ad-hoc scratch path). The stale
  reference in `RUNNING.md` was removed instead.
