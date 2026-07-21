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
4. **`EuclidCoreRandomTools` oracles reimplemented** in the test file:
   `_next_rotation_matrix` (random unit axis × angle ~ U(-π, π), mapped through
   the port's own `Γ_0`) and `_next_vector3d` (components ~ U(-1, 1)). Seeds
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
3. **`SE3LieGroupTools` replaced** by `_se3_exp_reference` /
   `_se3_adjoint_reference`: NumPy Rodrigues + left-Jacobian `V`, written from
   the closed forms rather than calling `group.Gamma0/Gamma1`, so the k=1
   agreement test is a real cross-check and not a tautology. Adjoint reference
   block form under rotation-first ordering is `[[R, 0], [(t)_× R, R]]`.
4. **`EuclidCoreRandomTools.nextRotationVector`** reproduced as random unit axis
   × angle ~ U(-π, π) (bounded by the injectivity radius); translational blocks
   ~ U(-1, 1) per `nextVector3D`.

### Tolerances

Verbatim from Java: `1e-10` on round-trip and both k=1 agreement tests, loosened
to `1e-9` on the adjoint homomorphism and `1e-8` on the conjugation identity.
The port is comfortably inside all three — observed worst-case error on the
conjugation identity is **2e-15** across all `k`, i.e. ~7 orders of margin.

One consideration checked and discarded: whether full-π ξ draws could push the
conjugated element `X exp(ξ) X⁻¹` outside the injectivity radius and break the
uniqueness of `log`. They cannot — conjugation is a similarity transform on the
rotation block, so the conjugated angle equals `‖φ_ξ‖ ≤ π`. Java's
full-magnitude draws are kept unmodified.
