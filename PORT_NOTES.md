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
- `SEK3UtilsTest` (the other half of gate G2) is not yet ported;
  `tests/inEKF/test_group.py` covers `group.py` but is not the 1:1 port and
  lacks the SE(3)-reference agreement at k=1, the adjoint homomorphism /
  conjugation identities at 1000 trials, and the wrong-sized-output `ValueError`
  guard on `log`.
