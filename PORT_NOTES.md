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
