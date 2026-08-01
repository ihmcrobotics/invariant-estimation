r"""Process-noise core of the joint-space KF.

    M(q)  --Schur-->  Lambda  --rotor-->  Lambda_eff  --Gram-->  Qa  --Van Loan-->  Q

Java reference: ``JointLevelKFPreFilter.updateProcessNoiseFromMassMatrix``
(L1287-1425) and ``buildProcessNoise`` (L1227).  Ported tests:
`JointLevelKFTransitionNoiseTest`, `JointLevelKFMassMatrixNoiseTest`,
`JointLevelKFRotorAndGramTest`, `JointLevelKFStandingStabilityTest`.

**Schur, not ``M_jj``.**  The modelled disturbance is an unmodeled joint torque
``w_tau``.  On a floating base the base recoils, so eliminating ``qdd_b`` from

    [ M_jj  M_jb ] [ qdd_j ]   [ w_tau ]
    [ M_bj  M_bb ] [ qdd_b ] = [   0   ]

gives ``Lambda qdd_j = w_tau`` with the articulated inertia

    Lambda = M_jj - M_jb M_bb^-1 M_bj                                     (1)

``M_jb M_bb^-1 M_bj`` is PSD, so ``Lambda <= M_jj`` and ``Lambda^-2 >= M_jj^-2``:
a free base accelerates *more* per unit torque, so the honest process noise is
strictly larger than the locked-base one — the ordering
`JointLevelKFStandingStabilityTest.testCandidateFixesBoundQa` asserts.  `Lambda`
is dense where `M` is sparse, which is the point: a diagonal torque uncertainty
comes out as *correlated* acceleration uncertainty, which is what makes the
exported ``Sigma_q`` honest for the InEKF contact update.

**Rotor inertia and the double-add trap.**  ``Lambda_eff = Lambda + diag(rotor)``
(`lambda_eff`).  The reflected rotor inertia ``n^2 J_rotor`` sits on the joint DoF
and couples to nothing else — in particular not to the floating base — so it is a
pure diagonal add on ``M_jj`` and commutes with the Schur complement:

      (M_jj + diag(a)) - M_jb M_bb^-1 M_bj = Lambda + diag(a)             (2)

By Weyl it also floors the spectrum: on Alex the distal link-side inertias fall to
~8e-4 while their drivetrains reflect 0.05-0.07, and without this term
``Lambda^-2`` carries diagonal outliers up to ~1.6e6 — the Alex002 velocity
covariance blow-up.

THE TRAP (CLAUDE.md §6): identity (2) cuts both ways.  In production the rotor
inertia arrives as the MJCF ``armature``, folded into ``qM`` *before* we gather
blocks, so ``Lambda`` from that ``qM`` is *already* ``Lambda_eff``.  Adding
``diag(rotor)`` again post-Schur counts the drivetrain **twice**, starving ``Qa``
by ~4x on the distal joints.  Nothing errors; the filter quietly over-trusts its
own prediction.  Hence ``rotor=`` defaults to the sentinel
`ROTOR_IN_MASS_MATRIX`: the caller must opt *in* to the post-Schur add by naming
an array at the call site.

**Per-joint sigma_tau (I9).**  ``sigma_tau_i = alpha_i * tau_max_i``, falling back
to scalar ``params.sigma_tau``.  A *uniform* sigma is the I9 failure mode:
``Lambda_eff^-1`` varies by orders of magnitude across joints, so one uniform
torque fraction trips the QA_MAX tripwire on one joint after another.
`equalized_sigma_tau` is the offline fix (CLAUDE.md §2).

**Gram form.**  ``Qa = Lambda_eff^-1 diag(sigma_tau^2) Lambda_eff^-T`` is computed
as ``Y = cho_solve(Lambda_eff, diag(sigma_tau))``, ``Qa = Y Y^T`` (3): symmetric
PSD by construction and, on XLA, **bit-exactly** symmetric, because ``Qa[i,j]``
and ``Qa[j,i]`` are the same reduction over the same summands in the same order.
The dense sandwich is algebraically identical but fails exact symmetry at 1e-17 —
which is why `JointLevelKFMassMatrixNoiseTest.testVanLoanBlocksAreExactlySymmetric`
uses ``tol = 0.0``, and why **this module never applies ``0.5(A + A^T)`` to `Qa`
or `Q`**: doing so would restore the symmetry a wrong (dense) implementation
loses, turning a load-bearing assertion into a tautology.  `Lambda` *is*
symmetrised — there the tolerance is 1e-9 and the Cholesky residue has no
structural meaning.

**QA_MAX is a tripwire, never a clamp.**  Over ``params.qa_max`` we surface it (a
per-joint float counter, plus a host-side warning naming the argmax joint) and
change nothing.  The superseded behaviour rescaled *all* joints by a uniform
factor, which coupled one joint's outlier into global ``Q`` starvation (hips down
~6 orders), collapsed ``P`` onto the measurement floors, and *caused* the min-side
``S`` singularity it was meant to prevent.

**Van Loan.**  ``A`` is nilpotent on ``(q, q_dot)``, so the integral closes exactly
in three terms — no truncation, no ``expm``:

    Q_qq = dt^3/3 Qa,  Q_q_qd = Q_qd_q = dt^2/2 Qa,  Q_qdqd = dt Qa       (4)

The bias block is an independent random walk ``dt * imu_bias_process_var * I``,
with **exactly zero** joint<->bias cross terms: gyro bias is not driven by joint
torque.

Everything is fixed-shape, branch-free on traced values (I7) and float64 (I8).
The only Python branch is ``M is None`` — a structural build-time choice, not
data.
"""
from typing import Any, NamedTuple
import warnings

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_factor, cho_solve

from .state import JointKFBuild, JointKFParams


#: Sentinel for ``rotor=``: "the mass matrix already carries the reflected rotor
#: inertia (MJCF ``armature``, folded into ``qM`` pre-Schur) -- add nothing".
#: This is the DEFAULT, so the double-add trap (module docstring, CLAUDE.md §6)
#: cannot be entered by omission; only by explicitly passing an array.
ROTOR_IN_MASS_MATRIX = "in_mass_matrix"

_QA_WARNED: set[str] = set()


class ProcessDiagnostics(NamedTuple):
    """Per-tick process-noise diagnostics.  All float64 -- pytree-safe under scan.

    Part of the seam surface (CLAUDE.md §4: the tests read these), not optional
    logging.  ``sqrt(qa_diag)`` is the per-joint unmodeled-acceleration STD that
    `equalized_sigma_tau` equalises at ``target_qdd_std``, and the quantity to
    read off a live run when recalibrating ``alpha_overrides``.
    """

    qa_diag: Array          # (n,) diag(Qa) [(rad/s^2)^2]
    qa_max_diag: Array      # () max_i diag(Qa)_i
    qa_argmax: Array        # () int, worst joint -- named in the tripwire warning
    qa_tripped: Array       # (n,) 1.0 where diag(Qa)_i > qa_max on THIS tick
    qa_trip_count: Array    # (n,) running counter; float, so it rides a scan carry


def qa_tripwire(qa: Array, params: JointKFParams, count: Array | None = None) -> ProcessDiagnostics:
    """Evaluate the QA_MAX tripwire.  **Surfaces, never rescales** (module docstring).

    `count` is the previous tick's `ProcessDiagnostics.qa_trip_count`; `None`
    starts from zero.  The advance is a `jnp.where`, so the graph is constant
    whether or not the wire trips (I7).
    """
    qa = jnp.asarray(qa, dtype=jnp.float64)
    diag = jnp.diagonal(qa)
    n = diag.shape[-1]
    tripped = jnp.where(diag > params.qa_max, 1.0, 0.0)
    prev = jnp.zeros(n, dtype=jnp.float64) if count is None else jnp.asarray(count, dtype=jnp.float64)
    return ProcessDiagnostics(
        qa_diag=diag,
        qa_max_diag=jnp.max(diag),
        qa_argmax=jnp.argmax(diag),
        qa_tripped=tripped,
        qa_trip_count=prev + tripped,
    )


def warn_qa_tripwire(diag: ProcessDiagnostics, build: JointKFBuild, params: JointKFParams) -> None:
    """Host-side, warn-**once**-per-joint report of a QA_MAX trip.

    Off the jitted path: a name lookup and a warning are host effects (I7).  Pure
    reporting -- it does not and must not alter ``Qa``.
    """
    worst = int(diag.qa_argmax)
    value = float(diag.qa_max_diag)
    if value <= params.qa_max:
        return
    name = build.joint_names[worst] if worst < len(build.joint_names) else f"joint{worst}"
    if name in _QA_WARNED:
        return
    _QA_WARNED.add(name)
    warnings.warn(
        f"QA_MAX tripwire: diag(Qa)[{name}] = {value:.4e} > qa_max = {params.qa_max:.4e} "
        f"[(rad/s^2)^2]. NOT rescaled (by design -- CLAUDE.md §6). Check that joint's "
        f"alpha_override / effort limit / rotor inertia; a near-singular Lambda_eff mode "
        f"is the usual cause.",
        RuntimeWarning,
        stacklevel=2,
    )


def schur_complement(M: Array, filtered_idx: Array, nuisance_idx: Array) -> Array:
    r"""Articulated joint inertia ``Lambda = M_jj - M_jb M_bb^-1 M_bj``  (eq. 1).

    ``M`` `(D, D)` is the FULL configuration-space inertia (MJX's dense ``qM``, or
    a synthetic SPD matrix in the algebra tests) — a plain array, never a model
    object, so this module imports no simulator.  `filtered_idx` `(n,)` is in
    filter state order (`build.dof_joint`); `nuisance_idx` is the floating base's
    6 plus any "gap" joint on a chain but not a filter state
    (`build.dof_nuisance`) — the Java "considered subsystem" trick, since gap
    joints are equally recoil paths from the filter's view.

    Solved with a Cholesky of ``M_bb``, never an explicit inverse; the oracle in
    `tests/jointKF/_oracles.py` deliberately uses an LU inverse so agreement is
    evidence rather than a restatement.  Symmetrising is safe here
    (`testSchurComplementIsSymmetricPDAndDominatedByLockedInertia` allows 1e-9):
    the Cholesky residue has no structural meaning.  Contrast
    `qa_from_lambda_eff`, which must NOT be symmetrised.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    f = jnp.asarray(filtered_idx)
    b = jnp.asarray(nuisance_idx)

    M_jj = M[jnp.ix_(f, f)]
    M_jb = M[jnp.ix_(f, b)]
    M_bj = M[jnp.ix_(b, f)]
    M_bb = M[jnp.ix_(b, b)]

    recoil = M_jb @ cho_solve(cho_factor(M_bb, lower=True), M_bj)
    lam = M_jj - recoil
    return 0.5 * (lam + lam.T)


def lambda_eff(lam: Array, rotor: Array) -> Array:
    r"""``Lambda_eff = Lambda + diag(rotor)``  (eq. 2) -- exact diagonal add.

    Off-diagonals stay bit-identical: the drivetrain does not couple through the
    floating base.  The Weyl floor is what
    `JointLevelKFRotorAndGramTest.testReflectedRotorInertiaAddAndWeylFloor`
    asserts.

    **Call this at most once per mass matrix** -- see the double-add trap in the
    module docstring.  If ``M`` came from an MJCF carrying ``armature``, the
    Schur complement already returned ``Lambda_eff``.
    """
    lam = jnp.asarray(lam, dtype=jnp.float64)
    return lam + jnp.diag(jnp.asarray(rotor, dtype=jnp.float64))


def _apply_rotor(lam: Array, rotor: Any) -> Array:
    """Resolve the ``rotor=`` argument: sentinel -> identity, array -> add."""
    if rotor is None or isinstance(rotor, str):
        # `None` is rejected rather than aliased to "add nothing": the two
        # meanings ("M carries it" vs "I forgot") must not be spellable the same
        # way, or the double-add trap becomes reachable by accident.
        if rotor != ROTOR_IN_MASS_MATRIX:
            raise ValueError(
                f"rotor must be an array of shape (n,) or the sentinel "
                f"ROTOR_IN_MASS_MATRIX, got {rotor!r}"
            )
        return lam
    return lambda_eff(lam, rotor)


def sigma_tau_per_joint(build: JointKFBuild, params: JointKFParams) -> Array:
    r"""``sigma_tau_i = alpha_i * tau_max_i``, else the scalar fallback (I9).

    Java ``referenceSigmaTau``: ``alphaForName(name) * effortLimitUpper`` when the
    effort limit is finite and positive, otherwise ``SIGMA_TAU = 5.0``.  The
    random chain fixtures have no effort limits, so they exercise the fallback.

    Recomputed here rather than read from ``build.sigma_tau`` so this module owns
    the closed form; `test_massmatrix_noise` asserts the two agree.
    """
    alpha = jnp.asarray(build.alpha, dtype=jnp.float64)
    tau_max = jnp.asarray(build.tau_max, dtype=jnp.float64)
    usable = jnp.isfinite(tau_max) & (tau_max > 0.0)
    # jnp.where over a sanitised tau_max: NaN * 0 is NaN, so the non-finite
    # branch must be neutralised BEFORE the multiply, not after.
    safe_tau = jnp.where(usable, tau_max, 0.0)
    return jnp.where(usable, alpha * safe_tau, params.sigma_tau)


def equalized_sigma_tau(lam_eff: Array, target_qdd_std: float) -> Array:
    r"""Offline calibration: ``sigma_i = target / |Lambda_eff^-1[i,i]|``  (CLAUDE.md §2).

    Equalises the DOMINANT term of each joint's unmodeled-acceleration STD:
    ``sqrt(Qa_ii) = sqrt(sum_j (inv[i,j] sigma_j)^2) >= |inv[i,i]| sigma_i``.  The
    inequality is why the achieved STD is a *floor* at the target rather than
    equal to it — off-diagonal inertial coupling can only add — which is the pair
    of assertions in
    `testAccelerationEqualizedSigmaTauFloorsAtTargetAndEqualizesDominantTerm` and
    the reason CLAUDE.md prescribes 2-3 iterations rather than a one-shot solve.

    Feed the result back as ``alpha_i = sigma_i / tau_max_i``.
    """
    lam_eff = jnp.asarray(lam_eff, dtype=jnp.float64)
    n = lam_eff.shape[-1]
    inv = cho_solve(cho_factor(lam_eff, lower=True), jnp.eye(n, dtype=jnp.float64))
    return target_qdd_std / jnp.abs(jnp.diagonal(inv))


def qa_from_lambda_eff(lam_eff: Array, sigma_tau: Array) -> Array:
    r"""``Qa = Y Y^T`` with ``Y = cho_solve(Lambda_eff, diag(sigma_tau))``  (eq. 3).

    Deliberately NOT symmetrised afterwards.  See the module docstring: the
    ``tol = 0.0`` symmetry assertions are the detector for someone "simplifying"
    this into the dense sandwich, and a ``0.5(A + A^T)`` would blind them.
    """
    lam_eff = jnp.asarray(lam_eff, dtype=jnp.float64)
    sigma_tau = jnp.asarray(sigma_tau, dtype=jnp.float64)
    Y = cho_solve(cho_factor(lam_eff, lower=True), jnp.diag(sigma_tau))
    return Y @ Y.T


def acceleration_covariance(
    build: JointKFBuild,
    params: JointKFParams,
    M: Array | None = None,
    *,
    rotor: Any = ROTOR_IN_MASS_MATRIX,
) -> Array:
    r"""Unmodeled-acceleration covariance ``Qa`` `(n, n)`.  Java seam:
    ``updateProcessNoiseFromMassMatrixForTest``.

    ``M = None`` selects the scalar-CWNA fallback ``Qa = sigma_accel^2 I_n``
    (``sigma_accel = 50 rad/s^2``), diagonal by construction — the *point* of
    `testProcessNoiseCouplesJointsThroughInertia`: the mass-matrix path couples
    joints through ``Lambda_eff^-1`` and the fallback provably cannot.

    ``rotor`` defaults to `ROTOR_IN_MASS_MATRIX` -- read the module docstring
    before changing it at any call site.
    """
    n = build.n_joints
    if M is None:
        if build.use_mass_matrix:
            raise ValueError(
                "build.use_mass_matrix is True but M is None -- the mass-matrix "
                "path was declared at build time and then not wired. Pass M, or "
                "build with use_mass_matrix=False to take the scalar-CWNA fallback."
            )
        return (params.sigma_accel ** 2) * jnp.eye(n, dtype=jnp.float64)

    lam = schur_complement(M, build.dof_joint, build.dof_nuisance)
    lam_eff = _apply_rotor(lam, rotor)
    return qa_from_lambda_eff(lam_eff, sigma_tau_per_joint(build, params))


def van_loan(qa: Array, params: JointKFParams, n_imus: int) -> Array:
    r"""Assemble the discrete process noise ``Q`` from ``Qa``  (eq. 4).

    ::

        Q = [ dt^3/3 Qa   dt^2/2 Qa       0      ]
            [ dt^2/2 Qa     dt Qa         0      ]
            [    0            0     dt s_b I_3m  ]

    The zero joint<->bias blocks are structural: joint torque does not drive the
    gyro bias random walk, and coupling them would let the bias estimator absorb
    joint modelling error (the mechanism I1 exists to prevent).

    No ``0.5(A + A^T)`` is applied -- ``Qa`` is already bit-symmetric out of the
    Gram form and each block is a positive scalar multiple of it, so exact
    symmetry propagates.  See `qa_from_lambda_eff`.
    """
    qa = jnp.asarray(qa, dtype=jnp.float64)
    n = qa.shape[-1]
    dt = params.dt
    m3 = 3 * n_imus

    zero_jb = jnp.zeros((n, m3), dtype=jnp.float64)
    bias = (dt * params.imu_bias_process_var) * jnp.eye(m3, dtype=jnp.float64)

    return jnp.block([
        [(dt ** 3 / 3.0) * qa, (dt ** 2 / 2.0) * qa, zero_jb],
        [(dt ** 2 / 2.0) * qa, dt * qa, zero_jb],
        [zero_jb.T, zero_jb.T, bias],
    ])


def build_process_noise(
    build: JointKFBuild,
    params: JointKFParams,
    M: Array | None = None,
    *,
    rotor: Any = ROTOR_IN_MASS_MATRIX,
) -> Array:
    """Discrete process noise ``Q`` `(dim, dim)`.  Java ``getProcessNoise``.

    Kept separate from `acceleration_covariance` so a test (and the Java
    ``updateProcessNoiseFromMassMatrixForTest`` seam) can look at ``Qa`` alone.
    """
    qa = acceleration_covariance(build, params, M, rotor=rotor)
    return van_loan(qa, params, build.n_imus)


def build_process_noise_with_diagnostics(
    build: JointKFBuild,
    params: JointKFParams,
    M: Array | None = None,
    *,
    rotor: Any = ROTOR_IN_MASS_MATRIX,
    qa_trip_count: Array | None = None,
) -> tuple[Array, ProcessDiagnostics]:
    """`build_process_noise` plus `ProcessDiagnostics` — the form the scan body wants.

    ``qa_trip_count`` threads through the carry so the QA_MAX tripwire accumulates
    over a run without a host effect.
    """
    qa = acceleration_covariance(build, params, M, rotor=rotor)
    return van_loan(qa, params, build.n_imus), qa_tripwire(qa, params, qa_trip_count)
