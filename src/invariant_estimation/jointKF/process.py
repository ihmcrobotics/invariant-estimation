r"""
jointKF/process.py
==================
The process-noise core of the joint-space KF: the chain

    M(q)  --Schur-->  Lambda  --rotor-->  Lambda_eff  --Gram-->  Qa  --Van Loan-->  Q

Java reference: ``JointLevelKFPreFilter.updateProcessNoiseFromMassMatrix``
(L1287-1425) and ``buildProcessNoise`` (L1227).  Ported tests:
`JointLevelKFTransitionNoiseTest`, `JointLevelKFMassMatrixNoiseTest`,
`JointLevelKFRotorAndGramTest`, `JointLevelKFStandingStabilityTest`.

Why a Schur complement and not ``M_jj``
---------------------------------------
The disturbance the filter models is an unmodeled **joint torque** ``w_tau``, not
an unmodeled acceleration.  On a *floating* base, applying ``w_tau`` at the
joints does not simply give ``M_jj^-1 w_tau``: the base recoils.  Eliminating the
base's (and any gap joint's) acceleration from the constrained Newton-Euler
system

    [ M_jj  M_jb ] [ qdd_j ]   [ w_tau ]
    [ M_bj  M_bb ] [ qdd_b ] = [   0   ]

by solving the second row for ``qdd_b = -M_bb^-1 M_bj qdd_j`` and substituting
gives ``Lambda qdd_j = w_tau`` with

    Lambda = M_jj - M_jb M_bb^-1 M_bj                                     (1)

the **articulated / Schur-complement** inertia.  Since ``M_jb M_bb^-1 M_bj`` is
PSD (it is a congruence of ``M_bb^-1 >= 0``), ``Lambda <= M_jj`` in the Loewner
order, hence ``Lambda^-2 >= M_jj^-2``: a free base makes the joints accelerate
*more* per unit torque, so the honest process noise is strictly larger than the
locked-base one.  `JointLevelKFStandingStabilityTest.testCandidateFixesBoundQa`
asserts exactly that ordering.

`Lambda` is dense even where `M` is sparse, and its off-diagonal structure is the
whole reason for using an inertia at all: a diagonal torque uncertainty comes out
as *correlated* acceleration uncertainty across the kinematic tree, which is what
makes the exported ``Sigma_q`` honest for the downstream InEKF contact update.

Rotor inertia -- and the double-add trap
----------------------------------------
``Lambda_eff = Lambda + diag(rotor)`` (`lambda_eff`).  This is not a hack:

* **Exactness.** The reflected rotor inertia ``n^2 J_rotor`` is the inertia of a
  body spinning about *its own* axis behind the gearbox.  It appears on the joint
  DoF and couples to nothing else -- in particular not to the floating base -- so
  it is a pure diagonal add on ``M_jj`` that leaves ``M_bb`` and ``M_jb``
  untouched.  Consequently it commutes with the Schur complement:

      (M_jj + diag(a)) - M_jb M_bb^-1 M_bj = Lambda + diag(a)             (2)

* **Regularisation, for free.** By Weyl's inequality
  ``lambda_min(Lambda + diag(a)) >= lambda_min(Lambda) + min(a)``, so the
  drivetrain term *floors* the spectrum.  On Alex the distal link-side inertias
  fall to ~8e-4 while their drivetrains reflect 0.05-0.07, and without this term
  ``Lambda^-2`` carries diagonal outliers up to ~1.6e6 -- the Alex002 velocity
  covariance blow-up.

**THE TRAP (CLAUDE.md §6).** Identity (2) cuts both ways.  In production the
rotor inertia arrives as the MJCF ``armature``, which MuJoCo folds into ``qM``
*before* we ever gather blocks -- so ``Lambda`` computed from that ``qM`` is
*already* ``Lambda_eff``.  Adding ``diag(rotor)`` again post-Schur counts the
drivetrain **twice**, silently doubling the joints' apparent inertia and
starving ``Qa`` by ~4x on the distal joints.  Nothing downstream errors; the
filter just quietly over-trusts its own prediction.

Therefore `acceleration_covariance` and `build_process_noise` take an explicit
``rotor=`` argument whose **default is the sentinel `ROTOR_IN_MASS_MATRIX`**,
meaning "``M`` already carries it, add nothing".  The caller must opt *in* to the
post-Schur add by passing an array.  There is deliberately no way to express
"add rotor" without naming it at the call site.

Per-joint sigma_tau (invariant I9)
----------------------------------
``sigma_tau_i = alpha_i * tau_max_i``, falling back to the scalar
``params.sigma_tau`` where the effort limit is absent or non-finite.  A *uniform*
sigma is the failure mode I9 forbids: ``Lambda_eff^-1`` varies by orders of
magnitude across joints, so one uniform torque fraction trips the QA_MAX
tripwire on one joint after another.  `equalized_sigma_tau` implements the
offline calibration that fixes this by equalising the unmodeled *acceleration*
STD instead (CLAUDE.md §2).

Gram form
---------
``Qa = Lambda_eff^-1 diag(sigma_tau^2) Lambda_eff^-T`` is computed as

    Y = Lambda_eff^-1 diag(sigma_tau)      (a Cholesky solve, never an inverse)
    Qa = Y Y^T                                                            (3)

which is symmetric-PSD *by construction* -- and, on XLA, **bit-exactly**
symmetric, because ``Qa[i,j]`` and ``Qa[j,i]`` are the same reduction over the
same summands in the same order.  The dense sandwich is algebraically identical
but numerically is not: it fails exact symmetry at the 1e-17 level.  That is
precisely why `JointLevelKFMassMatrixNoiseTest.testVanLoanBlocksAreExactlySymmetric`
uses ``tol = 0.0``, and why **this module never applies a ``0.5(A + A^T)``
symmetrisation to `Qa` or `Q`** -- doing so would restore the symmetry a wrong
(dense) implementation loses, turning a load-bearing assertion into a tautology.
`Lambda` *is* symmetrised, because there the test tolerance is 1e-9 and the
Cholesky solve genuinely leaves an asymmetric residue with no structural meaning.

QA_MAX is a tripwire, never a clamp
-----------------------------------
When ``max diag(Qa)`` exceeds ``params.qa_max`` we **surface** it -- a per-joint
float counter in `ProcessDiagnostics`, plus a host-side warning naming the argmax
joint -- and change nothing.  The superseded behaviour rescaled *all* joints by a
uniform factor to bring the outlier under the cap; that coupled one joint's
outlier into global ``Q`` starvation (hips down ~6 orders), collapsing ``P`` onto
the measurement floors and *causing* the min-side ``S`` singularity it was meant
to prevent.  A cap that silently rescales the whole robot is worse than the
disease.

Van Loan
--------
``A`` is nilpotent on the ``(q, q_dot)`` block (``A^2 = 0`` there), so the
Van Loan integral ``int_0^dt e^{As} G Qa G^T e^{A^T s} ds`` closes exactly in
three terms -- no truncation, no matrix exponential:

    Q_qq = dt^3/3 Qa,  Q_q_qd = Q_qd_q = dt^2/2 Qa,  Q_qdqd = dt Qa       (4)

The bias block is an independent random walk, ``dt * imu_bias_process_var * I``,
with **exactly zero** joint<->bias cross terms: the gyro bias is not driven by
joint torque.

Constant-XLA-graph (I7) and precision (I8)
------------------------------------------
Everything below is fixed-shape and branch-free on traced values.  The only
Python-level branch is ``M is None``, which selects the scalar-CWNA fallback --
a *structural* choice fixed at build time, not data.  The QA_MAX tripwire is a
float counter advanced with `jnp.where`, never an ``if``.  All arrays are
float64.
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


# ---------------------------------------------------------------------------
# Diagnostics pytree (CONTRACT_CARD §6: the tests read these, so they are part
# of the seam surface, not optional logging)
# ---------------------------------------------------------------------------

class ProcessDiagnostics(NamedTuple):
    """Per-tick process-noise diagnostics.  All float64 -- pytree-safe under scan.

    Attributes
    ----------
    qa_diag : Array, shape (n,)
        ``diag(Qa)`` -- the per-joint unmodeled-acceleration VARIANCE
        [(rad/s^2)^2].  Its square root is the quantity `equalized_sigma_tau`
        equalises at ``target_qdd_std``, and the quantity to read off a live run
        when recalibrating ``alpha_overrides``.
    qa_max_diag : Array, shape ()
        ``max_i diag(Qa)_i``.
    qa_argmax : Array, shape (), int
        Index of the worst joint -- the one to name in the tripwire warning.
    qa_tripped : Array, shape (n,)
        1.0 where ``diag(Qa)_i > qa_max`` on THIS tick, else 0.0.
    qa_trip_count : Array, shape (n,)
        Running per-joint tripwire counter (carry-in + `qa_tripped`).  A float,
        not an int, so it rides in a scan carry without dtype gymnastics.
    """

    qa_diag: Array
    qa_max_diag: Array
    qa_argmax: Array
    qa_tripped: Array
    qa_trip_count: Array


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

    Kept out of the jitted path on purpose: a name lookup and a warning are both
    host effects (I7 forbids strings inside jit).  Call this from the eager
    orchestrator, or after a scan on the collected `ProcessDiagnostics`.  It is a
    pure reporting function -- it does not and must not alter ``Qa``.
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


# ---------------------------------------------------------------------------
# Step 1: Schur complement
# ---------------------------------------------------------------------------

def schur_complement(M: Array, filtered_idx: Array, nuisance_idx: Array) -> Array:
    r"""Articulated joint inertia ``Lambda = M_jj - M_jb M_bb^-1 M_bj``  (eq. 1).

    Parameters
    ----------
    M : Array, shape (D, D)
        The FULL configuration-space inertia over all DoF -- MJX's dense ``qM``
        in production, a synthetic SPD matrix in the algebra tests.  Passed as a
        plain array, never a model object, so this module imports no simulator.
    filtered_idx : Array, shape (n,)
        DoF indices of the filtered joints, **in filter state order** (`build.dof_joint`).
    nuisance_idx : Array, shape (n_nuisance,)
        DoF indices eliminated: the floating base's 6 plus any "gap" joint that
        lies on a chain but is not a filter state (`build.dof_nuisance`).
        Eliminating gap joints alongside the base is the Java "considered
        subsystem" trick -- they are equally unactuated-from-the-filter's-view
        recoil paths.

    Returns
    -------
    Array, shape (n, n) -- symmetric PD, symmetrised as ``0.5(L + L^T)``.

    Notes
    -----
    Solved with a Cholesky factorisation of ``M_bb`` (SPD by construction);
    never an explicit inverse.  The oracle in `tests/jointKF/_oracles.py`
    deliberately uses an LU inverse instead, so agreement is evidence rather
    than a restatement.

    Symmetrising is safe here (`testSchurComplementIsSymmetricPDAndDominatedByLockedInertia`
    allows 1e-9): the Cholesky solve leaves an asymmetric residue with no
    structural meaning.  Contrast `qa_from_lambda_eff`, which must NOT be
    symmetrised.
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


# ---------------------------------------------------------------------------
# Step 2: rotor inertia
# ---------------------------------------------------------------------------

def lambda_eff(lam: Array, rotor: Array) -> Array:
    r"""``Lambda_eff = Lambda + diag(rotor)``  (eq. 2) -- exact diagonal add.

    Off-diagonals are left bit-identical: the drivetrain does not couple through
    the floating base (module docstring).  By Weyl,
    ``lambda_min(Lambda_eff) >= lambda_min(Lambda) + min(rotor)``, which is the
    spectral floor `JointLevelKFRotorAndGramTest.testReflectedRotorInertiaAddAndWeylFloor`
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


# ---------------------------------------------------------------------------
# Step 3: per-joint sigma_tau (I9)
# ---------------------------------------------------------------------------

def sigma_tau_per_joint(build: JointKFBuild, params: JointKFParams) -> Array:
    r"""``sigma_tau_i = alpha_i * tau_max_i``, else the scalar fallback.

    Java ``referenceSigmaTau``: use ``alphaForName(name) * effortLimitUpper`` when
    the effort limit is finite and positive, otherwise ``SIGMA_TAU = 5.0``.  The
    randomly generated chain fixtures have no effort limits, so they exercise the
    fallback -- which is exactly why the fallback has to be right.

    Recomputed here rather than read from ``build.sigma_tau`` so this module owns
    the closed form; the two must agree, and `test_massmatrix_noise` asserts they
    do on the stub build.
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

    Equalises the DOMINANT term of each joint's unmodeled-acceleration STD --
    ``sqrt(Qa_ii) = sqrt(sum_j (inv[i,j] sigma_j)^2) >= |inv[i,i]| sigma_i`` --
    at the common target.  The inequality is why the achieved STD is a *floor*
    at the target rather than equal to it: off-diagonal inertial coupling can
    only add.  That is precisely the pair of assertions in
    `testAccelerationEqualizedSigmaTauFloorsAtTargetAndEqualizesDominantTerm`,
    and the reason CLAUDE.md prescribes 2-3 calibration iterations rather than a
    one-shot solve.

    Feed the result back as ``alpha_i = sigma_i / tau_max_i``.
    """
    lam_eff = jnp.asarray(lam_eff, dtype=jnp.float64)
    n = lam_eff.shape[-1]
    inv = cho_solve(cho_factor(lam_eff, lower=True), jnp.eye(n, dtype=jnp.float64))
    return target_qdd_std / jnp.abs(jnp.diagonal(inv))


# ---------------------------------------------------------------------------
# Step 4: Gram-form Qa
# ---------------------------------------------------------------------------

def qa_from_lambda_eff(lam_eff: Array, sigma_tau: Array) -> Array:
    r"""``Qa = Y Y^T`` with ``Y[i,j] = Lambda_eff^-1[i,j] sigma_tau_j``  (eq. 3).

    ``Y`` is obtained as ``cho_solve(Lambda_eff, diag(sigma_tau))`` -- one
    triangular solve, no inverse.  The result is symmetric-PSD by construction
    and, on XLA, **bit-exactly** symmetric.

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
    r"""Unmodeled-acceleration covariance ``Qa``, shape ``(n, n)``.

    Java seam: ``updateProcessNoiseFromMassMatrixForTest`` (`SEAM_MAP`).

    ``M = None`` selects the **scalar-CWNA fallback** ``Qa = sigma_accel^2 I_n``
    -- the no-robot-model path, ``sigma_accel = 50 rad/s^2``.  It is diagonal by
    construction, which is the *point* of
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


# ---------------------------------------------------------------------------
# Step 5: Van Loan discretisation
# ---------------------------------------------------------------------------

def van_loan(qa: Array, params: JointKFParams, n_imus: int) -> Array:
    r"""Assemble the discrete process noise ``Q`` from ``Qa``  (eq. 4).

    ::

        Q = [ dt^3/3 Qa   dt^2/2 Qa       0      ]
            [ dt^2/2 Qa     dt Qa         0      ]
            [    0            0     dt s_b I_3m  ]

    The zero joint<->bias blocks are structural, not a tuning choice: joint
    torque does not drive the gyro bias random walk, and coupling them would let
    the bias estimator absorb joint modelling error (the mechanism I1 exists to
    prevent).

    No ``0.5(A + A^T)`` is applied -- ``Qa`` is already bit-symmetric out of the
    Gram form, and each block is a positive scalar multiple of it, so exact
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
    """Discrete process noise ``Q``, shape ``(dim, dim)``.  Java ``getProcessNoise``.

    Thin composition of `acceleration_covariance` and `van_loan`; kept separate
    so a test (and the Java suite's ``updateProcessNoiseFromMassMatrixForTest``
    seam) can look at ``Qa`` alone.
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
    """`build_process_noise` plus the `ProcessDiagnostics` pytree.

    This is the form the scan body wants: ``qa_trip_count`` threads through the
    carry so the QA_MAX tripwire accumulates over a run without a host effect.
    """
    qa = acceleration_covariance(build, params, M, rotor=rotor)
    return van_loan(qa, params, build.n_imus), qa_tripwire(qa, params, qa_trip_count)
