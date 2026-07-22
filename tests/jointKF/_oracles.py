"""Independent oracles shared by the ported `jointLevel` test classes.

Everything here stands in for a Java dependency the port cannot use (EJML
`CommonOps_DDRM`, `MatrixFeatures_DDRM`, `EigenDecomposition_F64`) or reproduces
a deterministic generator the Java suite defines inline.

**Deliberately written in NumPy from the closed forms, never by calling
`invariant_estimation.jointKF`** — an oracle that delegates to the code under
test proves nothing (`CONTRACT_CARD.md` §7).

The generators in the first section are ported **bit-for-bit** from
`JointLevelKFTestFixture.java` / `JointLevelKFUpdateTest.java`: they are pure
`sin`-based fills, so they are language-agnostic and must match Java exactly.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Deterministic matrix generators — bit-for-bit ports (CLAUDE.md §5)
# ---------------------------------------------------------------------------


def spd(size: int, seed: float) -> np.ndarray:
    """Deterministic SPD matrix — Java `JointLevelKFTestFixture.spd(size, seed)`.

    Fill `m.data[i] = sin(i + 1.0 + seed)` row-major over `size**2` entries, take
    `a = m @ m.T` (PSD), then add `size` to each diagonal, which makes it
    strictly PD. This is the oracle behind every seeded prior and noise matrix in
    the suite, so it must match Java elementwise.
    """
    idx = np.arange(size * size, dtype=float)
    m = np.sin(idx + 1.0 + seed).reshape(size, size)
    return m @ m.T + size * np.eye(size)


def generic_h(k: int, dim: int, seed: float) -> np.ndarray:
    """Deterministic non-geometric Jacobian — Java `genericH(k, dim, seed)`.

    `H[r, c] = sin(0.37 * (r * dim + c + 1) + seed)`. Non-geometric on purpose:
    the Joseph-update tests must exercise the linear algebra without any
    kinematic structure making a wrong implementation accidentally right.
    """
    r = np.arange(k, dtype=float)[:, None]
    c = np.arange(dim, dtype=float)[None, :]
    return np.sin(0.37 * (r * dim + c + 1.0) + seed)


def scaled_identity(size: int, value: float) -> np.ndarray:
    """Java `scaledIdentity(size, value)`."""
    return value * np.eye(size)


# ---------------------------------------------------------------------------
# Seeded priors — the exact mean patterns the Java tests use
# ---------------------------------------------------------------------------


def seeded_prior_predict(n: int, m: int, seed: float) -> tuple[np.ndarray, np.ndarray]:
    """`JointLevelKFPredictTest.seededPrior`: mean `i+1 / i+1+100 / i+1+1000`.

    Position segment `x[i] = i+1`, velocity `x[n+i] = i+1+100`, bias
    `x[2n+i] = i+1+1000`; covariance `spd(dim, seed)`. The three offsets keep the
    segments separated by orders of magnitude so a mis-ordered state layout shows
    up immediately rather than as a small numeric error.
    """
    dim = 2 * n + 3 * m
    x = np.empty(dim)
    x[:n] = np.arange(1, n + 1)
    x[n:2 * n] = np.arange(1, n + 1) + 100.0
    x[2 * n:] = np.arange(1, 3 * m + 1) + 1000.0
    return x, spd(dim, seed)


def seeded_prior_update(dim: int, seed: float) -> tuple[np.ndarray, np.ndarray]:
    """`JointLevelKFUpdateTest.seededPrior`: mean `0.1*(i+1)`, cov `spd(dim, seed)`."""
    return 0.1 * np.arange(1, dim + 1), spd(dim, seed)


# ---------------------------------------------------------------------------
# Reference filters — explicit-inverse, never the code under test
# ---------------------------------------------------------------------------


def reference_update(x, P, H, z, R) -> tuple[np.ndarray, np.ndarray]:
    """Explicit-inverse reference KF — Java `JointLevelKFUpdateTest.referenceUpdate`.

    Uses `np.linalg.inv` deliberately: the filter under test solves via Cholesky,
    so an independent inversion path is what makes this an oracle rather than a
    restatement. Joseph form on the covariance.
    """
    x, P, H, z, R = (np.asarray(a, dtype=float) for a in (x, P, H, z, R))
    PHt = P @ H.T
    S = H @ PHt + R
    K = PHt @ np.linalg.inv(S)
    nu = z - H @ x
    x_new = x + K @ nu
    IKH = np.eye(P.shape[0]) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    return x_new, P_new


def reference_schur(M: np.ndarray, filtered: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """`Lambda = M_ff - M_Nf^T M_NN^-1 M_Nf` via explicit LU inverse.

    The filter uses a Cholesky `cho_solve`; this uses `np.linalg.inv`, so the two
    agree only if the algebra is right — which is the point. Java's
    `referenceSchur` builds the same quantity with EJML LU.
    """
    M = np.asarray(M, dtype=float)
    M_ff = M[np.ix_(filtered, filtered)]
    M_Nf = M[np.ix_(nuisance, filtered)]
    M_NN = M[np.ix_(nuisance, nuisance)]
    return M_ff - M_Nf.T @ np.linalg.inv(M_NN) @ M_Nf


def reference_qa(lambda_eff: np.ndarray, sigma_tau: np.ndarray) -> np.ndarray:
    """Dense reference `Qa = Lambda_eff^-1 diag(sigma_tau^2) Lambda_eff^-T`.

    The filter builds the **Gram** form `Y[i,j] = Lambda_eff^-1[i,j]*sigma_tau[j]`,
    `Qa = Y Y^T`, which is exactly symmetric-PSD by construction. This dense form
    is algebraically identical but numerically distinct — the agreement is the
    test (`JointLevelKFRotorAndGramTest`).
    """
    inv = np.linalg.inv(np.asarray(lambda_eff, dtype=float))
    sig = np.asarray(sigma_tau, dtype=float)
    return inv @ np.diag(sig ** 2) @ inv.T


def van_loan_blocks(qa: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Van-Loan discretisation of a double integrator driven by `Qa`.

    Exact (not truncated): `A` is nilpotent on the `(q, q_dot)` block, so the
    integral `int_0^dt e^{As} G Qa G^T e^{A^T s} ds` closes in three terms.
    """
    return {
        "qq": (dt ** 3 / 3.0) * qa,
        "qqd": (dt ** 2 / 2.0) * qa,
        "qdq": (dt ** 2 / 2.0) * qa,
        "qdqd": dt * qa,
    }


def reference_marginalized(
    mu_x: np.ndarray,
    P_xx: np.ndarray,
    *,
    n: int,
    imu_bias_col,
    raw_gyro: np.ndarray,
    imu_omega_base_rot: np.ndarray,
    imu_joint_jacobian: np.ndarray,
    imu_sigma: np.ndarray,
    foot_joint_jacobian: np.ndarray | None = None,
    anchor_var: float = 4.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    r"""THE decisive oracle — `JointLevelKFStackedOracleTest.referenceMarginalized`.

    The stacked *relative*-gyro update must equal a reference KF that measures
    **raw** per-IMU gyros with block-diagonal (independent) noise, over a state
    augmented with a nuisance base angular velocity, and then marginalises that
    nuisance out.  This is the one place a wrong answer is not locally detectable
    by any single component: `measure.py`, `anchors.py` and `update.py` can each
    be individually plausible and still not compose.

    Why the two agree at all
    ------------------------
    Differencing two IMUs to form `omega_child - R omega_parent` is exactly what
    you get by writing both raw measurements in terms of a shared unknown
    `omega_base` and eliminating it.  Eliminating a variable you hold **no**
    prior on (the improper `gamma -> infinity` limit) is precisely marginalisation
    in the information form, which is why the nuisance block enters as a zero
    information block rather than a large-covariance one.  That also means the
    correlations the differencing induces between pairs sharing an IMU are not a
    modelling choice — they are forced, and reproducing them is what invariant I6
    (`R_g = L Sigma L^T`, never block-diagonal) is about.

    Parameters
    ----------
    mu_x, P_xx
        Prior over `x = [q ; q_dot ; b_omega]`, dimension `dim`.
    imu_bias_col : callable
        `imu -> ` first bias column of that IMU.
    raw_gyro : (n_imus, 3)
        Raw per-IMU gyro readings.
    imu_omega_base_rot : (n_imus, 3, 3)
        `R(base measurement frame -> IMU measurement frame)` — the nuisance
        columns. The base IMU's own block is the identity.
    imu_joint_jacobian : (n_imus, 3, n)
        Absolute angular Jacobian base->IMU link, expressed in the IMU frame
        (all-zero for the base IMU itself).
    imu_sigma : (n_imus, 3, 3)
        Independent per-IMU gyro measurement covariance — deliberately
        **block-diagonal** here, because the correlation must emerge from the
        marginalisation rather than be assumed.
    foot_joint_jacobian : (n_active_feet, 3, n) or None
        `J_leg` base->foot in the base frame, for each ACTIVE stance foot. Each
        contributes a near-zero absolute-rate constraint with `+I3` on the
        nuisance and `R = anchor_var * I3`.

    Returns
    -------
    (mu_post, P_post)
        The posterior over `x` alone, with the nuisance integrated out.
    """
    mu_x = np.asarray(mu_x, dtype=float)
    P_xx = np.asarray(P_xx, dtype=float)
    dim = mu_x.shape[0]
    D = dim + 3                                   # augmented with omega_base
    n_imus = raw_gyro.shape[0]
    feet = np.zeros((0, 3, n)) if foot_joint_jacobian is None else np.asarray(foot_joint_jacobian, float)

    M = 3 * n_imus + 3 * len(feet)
    H = np.zeros((M, D))
    z = np.zeros(M)
    R = np.zeros((M, M))

    for k in range(n_imus):
        r = 3 * k
        z[r:r + 3] = raw_gyro[k]
        H[r:r + 3, n:2 * n] = imu_joint_jacobian[k]          # q_dot columns
        col = imu_bias_col(k)
        H[r:r + 3, col:col + 3] = np.eye(3)                  # bias enters as +I
        H[r:r + 3, dim:] = imu_omega_base_rot[k]             # nuisance columns
        R[r:r + 3, r:r + 3] = imu_sigma[k]

    for f in range(len(feet)):
        r = 3 * n_imus + 3 * f
        H[r:r + 3, n:2 * n] = feet[f]
        H[r:r + 3, dim:] = np.eye(3)
        R[r:r + 3, r:r + 3] = anchor_var * np.eye(3)
        # z stays 0: a trusted stance foot has ~zero absolute angular rate.

    # Information form. The nuisance gets a ZERO information block -- the
    # improper prior -- which is what makes this the gamma -> infinity limit
    # rather than merely a very diffuse one.
    P_inv = np.linalg.inv(P_xx)
    Lam = np.zeros((D, D))
    Lam[:dim, :dim] = P_inv
    R_inv = np.linalg.inv(R)
    Lam += H.T @ R_inv @ H

    eta = np.zeros(D)
    eta[:dim] = P_inv @ mu_x
    eta += H.T @ R_inv @ z

    Sigma = np.linalg.inv(Lam)
    mu = Sigma @ eta
    return mu[:dim], Sigma[:dim, :dim]


def nis_quadratic_form(nu, S) -> float:
    """NIS as the quadratic form `nu^T S^-1 nu`, on the PRIOR residual and `S`.

    CLAUDE.md §6 trap: computing NIS on the *posterior* passes easy tests and
    fails this one.
    """
    nu = np.asarray(nu, dtype=float)
    return float(nu @ np.linalg.solve(np.asarray(S, dtype=float), nu))


# ---------------------------------------------------------------------------
# Assertion helpers — Java `JointLevelKFTestFixture`
# ---------------------------------------------------------------------------


def assert_all_close(actual, expected, tol: float, msg: str = "") -> None:
    """Java `assertAllClose`: shape check, then elementwise `|a - e| <= tol`.

    Note Java's is an absolute tolerance, not `np.allclose`'s mixed abs/rel — the
    suite's tolerances were chosen against that meaning, so keep it absolute.
    """
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"{msg}: shape {a.shape} != {e.shape}"
    worst = np.max(np.abs(a - e)) if a.size else 0.0
    assert worst <= tol, f"{msg}: max |diff| = {worst:.3e} > {tol:.3e}"


def assert_symmetric(A, tol: float, msg: str = "") -> None:
    """Java `assertSymmetric`: `|A[r,c] - A[c,r]| <= tol` over the upper triangle."""
    A = np.asarray(A, dtype=float)
    worst = np.max(np.abs(A - A.T)) if A.size else 0.0
    assert worst <= tol, f"{msg}: max asymmetry = {worst:.3e} > {tol:.3e}"


def symmetric_eigenvalues(A) -> np.ndarray:
    """Real eigenvalues of a symmetric matrix — Java `symmetricEigenvalues`."""
    return np.linalg.eigvalsh(np.asarray(A, dtype=float))


def assert_positive_semidefinite(A, msg: str = "") -> None:
    """Java `assertPositiveSemiDefinite`: `min_eig >= -1e-6 * max(max_eig, 1)`.

    The tolerance is relative to the matrix scale, because a covariance with
    entries of order 1e6 legitimately carries eigenvalue noise far above any
    fixed absolute floor.
    """
    eig = symmetric_eigenvalues(0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T))
    lo, hi = float(eig.min()), float(eig.max())
    bound = -1.0e-6 * max(hi, 1.0)
    assert lo >= bound, f"{msg}: min eig {lo:.6e} < {bound:.6e} (max eig {hi:.6e})"


def assert_all_finite(A, msg: str = "") -> None:
    """Java `assertAllFinite` — used by the NaN-hardening regression tests."""
    A = np.asarray(A, dtype=float)
    assert np.all(np.isfinite(A)), f"{msg}: {np.count_nonzero(~np.isfinite(A))} non-finite entries"


def cond_spd(A) -> float:
    """`maxEig / minEig` of a symmetric matrix — Java `condSPD`."""
    eig = symmetric_eigenvalues(A)
    return float(eig.max() / eig.min())


# ---------------------------------------------------------------------------
# Java `String.hashCode` — for the per-joint noise maps in the NIS tests
# ---------------------------------------------------------------------------


def java_hash_code(s: str) -> int:
    """Java `String.hashCode`: `h = 31*h + c`, wrapped to signed 32-bit.

    The NIS tests key per-joint noise on it (`sigmaFor(name) = 1e-4*(1 +
    floorMod(hashCode, 9))`). Any deterministic per-joint map would satisfy the
    tests, but reproducing Java's keeps the ported values identical, which makes
    a cross-language discrepancy attributable.
    """
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - 0x100000000 if h >= 0x80000000 else h


def sigma_for(name: str) -> float:
    """`JointLevelKFEncoderNISConsistencyTest.sigmaFor` — STD in [1e-4, 9e-4]."""
    return 1.0e-4 * (1 + java_hash_code(name) % 9)


def vel_sigma_for(name: str) -> float:
    """`JointLevelKFDirectVelocityMeasurementTest.velSigmaFor` — STD in [5e-3, 3.5e-2]."""
    return 5.0e-3 * (1 + java_hash_code(name) % 7)


# ---------------------------------------------------------------------------
# The four fixture shapes (Java `JointLevelKFTestFixture.shapes`)
# ---------------------------------------------------------------------------

#: `(chain_joints, parent_imu_after_joint, child_imu_after_joint)` per shape,
#: yielding `(n, m)` = (8,2), (4,2), (3,2), (8,3). `n` = joints strictly between
#: the parent and child links = `child_index - parent_index - 1`. Shape 4 is two
#: pairs sharing the middle IMU — the shared-base-IMU star that invariant I6 is
#: about, and the only shape that can catch a per-pair bias layout.
SHAPES: tuple[dict, ...] = (
    {"name": "n8_m2", "chain": 10, "imus": (1, 9), "pairs": ((0, 1),), "n": 8, "m": 2},
    {"name": "n4_m2", "chain": 6, "imus": (1, 5), "pairs": ((0, 1),), "n": 4, "m": 2},
    {"name": "n3_m2", "chain": 4, "imus": (0, 3), "pairs": ((0, 1),), "n": 3, "m": 2},
    {"name": "n8_m3", "chain": 10, "imus": (1, 5, 9), "pairs": ((0, 1), (1, 2)), "n": 8, "m": 3},
)


def shape_dims(shape: dict) -> tuple[int, int, int]:
    """`(n, m, dim)` for a `SHAPES` entry."""
    n, m = shape["n"], shape["m"]
    return n, m, 2 * n + 3 * m


def stub_build(shape: dict, **overrides):
    """A `JointKFBuild` carrying **dimensions only** — no real geometry.

    The state / predict / Joseph-update tests are pure linear algebra over
    `(n, m)`: the Java fixtures differ only in dimension there, and the seeded
    priors are synthetic, so the chain's actual link geometry never enters. This
    stub lets those classes be ported without the MJX model seam.

    Anything that touches kinematics — the stacked gyro measurement, the stance
    anchors, `M(q)`, the process noise on the mass-matrix path — must use the
    real fixture in `_fixture.py` instead. The placeholder arrays here are
    deliberately obvious (identity-ish masks, zero Jacobians) so that a test
    which *should* have used the real fixture fails loudly rather than quietly
    computing with fake geometry.
    """
    import numpy as _np

    from invariant_estimation.jointKF.state import JointKFBuild

    n, m = shape["n"], shape["m"]
    pairs = shape["pairs"]
    fields = dict(
        n_joints=n,
        n_imus=m,
        n_pairs=len(pairs),
        n_anchors=0,
        joint_names=tuple(f"joint{i}" for i in range(n)),
        imu_names=tuple(f"imu{k}" for k in range(m)),
        pair_parent=_np.array([p for p, _ in pairs], dtype=int),
        pair_child=_np.array([c for _, c in pairs], dtype=int),
        pair_velocity_mask=_np.ones((len(pairs), n)),
        base_imu=0,
        anchor_filtered_mask=_np.zeros((0, n)),
        anchor_unfiltered_mask=_np.zeros((0, 0)),
        anchor_imu=_np.zeros(0, dtype=int),
        alpha=_np.full(n, 0.15),
        tau_max=_np.full(n, _np.nan),          # absent => sigma_tau falls back
        sigma_tau=_np.full(n, 5.0),
        rotor_inertia=_np.full(n, 0.005),
        encoder_var=_np.full(n, 5.0e-5),
        encoder_wired=tuple(False for _ in range(n)),
        gyro_sigma=_np.tile(1.0e-4 * _np.eye(3), (m, 1, 1)),
        dof_joint=_np.arange(n),
        dof_nuisance=_np.arange(n, n + 6),
        use_mass_matrix=False,
    )
    fields.update(overrides)
    return JointKFBuild(**fields)
