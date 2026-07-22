"""Port of `JointLevelKFRotorAndGramTest.java` (3 tests).

Machine-precision oracle for the two algebraic halves of the Part-B process-noise
fix, on a **well-conditioned hand-built** `Lambda`:

1. the reflected-rotor-inertia diagonal add and the Weyl spectral floor it buys,
2. the Gram form of `Qa` against the dense sandwich,
3. the rotor-inertia name table.

Fully self-contained -- no fixture, no mass matrix, no kinematics -- which is why
`TEST_SUITE_MAP.md` flags it as the most portable class in the joint-level suite.
Every matrix below is transcribed verbatim from the Java.
"""
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  -- enables float64
from invariant_estimation.jointKF import process
from invariant_estimation.jointKF.state import rotor_inertia_for_name

from ._oracles import reference_qa, symmetric_eigenvalues


# ---------------------------------------------------------------------------
# testReflectedRotorInertiaAddAndWeylFloor
# ---------------------------------------------------------------------------

def test_reflected_rotor_inertia_add_and_weyl_floor():
    """Diagonal add is exact (tol 0.0); off-diagonals untouched; Weyl floor holds.

    `Lambda` has a deliberately light mode at index 1 (`0.03` on the diagonal)
    -- the near-singular articulated mode that produced the Alex002 blow-up. The
    Weyl assertions are the *reason* the rotor term is a principled regulariser
    and not a fudge: `lambda_min(Lambda + diag(a)) >= lambda_min(Lambda) + min(a)`
    is a theorem, so the floor is guaranteed, not observed.
    """
    lam = np.array([
        [1.20, 0.10, 0.02],
        [0.10, 0.03, 0.01],
        [0.02, 0.01, 0.90],
    ])
    rotor = np.array([0.062, 0.167, 0.070])

    lam_eff = np.asarray(process.lambda_eff(lam, rotor))

    # -- diagonal add exact, off-diagonals bit-identical (Java tol 0.0) --------
    for i in range(3):
        for j in range(3):
            expected = lam[i, j] + (rotor[i] if i == j else 0.0)
            assert lam_eff[i, j] == expected, f"({i},{j}): {lam_eff[i, j]!r} != {expected!r}"

    # -- Weyl: the drivetrain floors the spectrum -----------------------------
    min_lam = float(symmetric_eigenvalues(lam).min())
    min_eff = float(symmetric_eigenvalues(lam_eff).min())
    assert min_eff >= min_lam + rotor.min() - 1e-12
    assert min_eff >= rotor.min() - 1e-12


# ---------------------------------------------------------------------------
# testGramFormQaEqualsDenseReferenceAndIsSymmetricPSD
# ---------------------------------------------------------------------------

def test_gram_form_qa_equals_dense_reference_and_is_symmetric_psd():
    """Gram `Y Y^T` == dense `L^-1 diag(s^2) L^-T`, and is EXACTLY symmetric.

    The `tol = 0.0` symmetry assertion is the load-bearing one and the reason
    `process.qa_from_lambda_eff` must never apply `0.5(A + A^T)`: the dense
    sandwich agrees to 1e-17 but is *not* bit-symmetric, so a symmetrisation
    would make this test blind to the very substitution it exists to catch.

    `sigma_tau` here is `alpha * tau_max` at `alpha = 0.15` for three real Alex
    effort limits -- the per-joint form of invariant I9, not a uniform scalar.
    """
    lam_eff = np.array([
        [1.262, 0.10, 0.02],
        [0.10, 0.197, 0.01],
        [0.02, 0.01, 0.970],
    ])
    sigma_tau = 0.15 * np.array([160.7, 217.2, 193.6])

    qa_gram = np.asarray(process.qa_from_lambda_eff(lam_eff, sigma_tau))
    qa_ref = reference_qa(lam_eff, sigma_tau)

    tol = 1e-10 * max(1.0, float(np.max(np.abs(qa_ref))))
    assert np.max(np.abs(qa_gram - qa_ref)) <= tol

    # exact symmetry -- tol 0.0
    assert np.array_equal(qa_gram, qa_gram.T), (
        "Qa is not bit-symmetric; the Gram form guarantees it, the dense form does not"
    )

    assert float(symmetric_eigenvalues(qa_gram).min()) >= -1e-12


# ---------------------------------------------------------------------------
# testRotorInertiaTableLookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("LEFT_HIP_X", 0.062),
        ("RIGHT_HIP_Y", 0.167),
        ("left_knee_y", 0.167),   # case-insensitive, matches KNEE
        ("LEFT_ANKLE_Y", 0.070),
        ("LEFT_ANKLE_X", 0.050),
        ("SPINE_Z", 0.062),
        ("SOME_UNKNOWN_JOINT", 0.005),  # default floor
    ],
)
def test_rotor_inertia_table_lookup(name, expected):
    """Case-insensitive substring lookup, tol 0.0 -- a filter constant, not a tuning.

    These same numbers are written into the MJCF as `armature`. The lookup is
    for *populating* that file and for the G3 equivalence oracle -- never as a
    second additive term (the double-add trap).
    """
    assert rotor_inertia_for_name(name) == expected
