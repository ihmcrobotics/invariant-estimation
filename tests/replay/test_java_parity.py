r"""
Java parity, tier 1: **stateless** oracles against a real hardware log.

Why tier 1 exists
-----------------
The obvious way to compare two estimators is to run both and diff the
trajectories.  That is tier 2 (a free-running replay), and on its own it is a
poor test: the logged covariance is only ever its *diagonal* (via the
``_upperBound``/``_lowerBound`` pairs), so the Python filter cannot be re-seeded
with Java's ``P``, so the two integrate away from each other and every
disagreement looks the same -- "it drifts".

Tier 1 sidesteps that entirely.  Some of what the Java filter publishes is a
**pure function of quantities the log already contains**, with no dependence on
filter state.  Those can be recomputed exactly, tick by tick, with zero error
accumulation, and a mismatch points at one module instead of at the trajectory.
``jointKF_QaDiag_<joint>`` is the prize: it is a function of ``q`` alone and it
exercises the entire process-noise chain --

    M(q) --Schur--> Lambda --rotor--> Lambda_eff --Gram--> Qa

-- which is the hardest and most trap-laden part of the joint KF (`CLAUDE.md`
§6: the armature double-add, the uniform-noise cap, per-joint sigma_tau).

The considered subsystem (the thing this suite discovered)
----------------------------------------------------------
Java builds ``M`` over a **considered subsystem**: the floating base plus the
joints spanning base->filtered, with every off-path subtree *locked* and its
inertia composited into its parent (``considerIgnoredSubtreesInertia``).  On
Alex there are no gap joints, so the nuisance block is exactly the base 6 DoF.

Locking a coordinate restricts the kinetic-energy form ``T = 1/2 qd' M qd`` to a
subspace, so the locked system's mass matrix is the **principal submatrix** of
the full ``M`` on the retained DoFs.  That makes the port's job concrete:

    Lambda = M_ff - M_fb M_bb^-1 M_bf,   f = the 9 filtered joints,
                                          b = the base 6 DoF *only*

and NOT an elimination of all 26 non-filtered DoFs, which would model the ankles
and arms as free to accelerate (58% max relative error on this log).

...and locked *at which configuration*
--------------------------------------
At ``q = 0``, not at the live one.  Mecano composites an ignored subtree's
inertia into its parent exactly once, inside
``CompositeRigidBodyMassMatrixCalculator``'s **constructor**
(``updateIgnoredSubtreeInertia``, called from the ctor and from nowhere else),
and the result is a plain ``SpatialInertia`` in the parent's body-fixed frame
that is never refreshed.  So Java's ``M(q)`` sees Alex's ankles, arms and head
welded at whatever pose the robot model held when the estimator was built --
which on hardware is the freshly-constructed model, ``q = 0``.

That is not a subtlety worth 1%: on this log the arms sit at roughly
``(shoulder 0.71, elbow -1.91)`` rad and the ankles near ``-0.40`` rad from the
first tick onward, and feeding those live instead of zero moves ``diag(Qa)`` by
up to **14.4%** -- with the spine moving the *opposite* way to the legs, because
freezing the arms straightens the torso's composited inertia while freezing the
ankles shortens the legs'.  Fitting a scale ``s`` on the off-path angles
(``q_ignored = s * q_live``) gives a sharp unique minimum at ``s = 0``:
0.21% at ``s = 0``, 1.6% at ``s = 0.05``, 3.1% at ``s = 0.1``, 14.4% at
``s = 1``.  ``MjxModel.qpos`` reproduces this by construction -- off-path joints
keep ``qpos0`` -- so the oracle below must do the same or it is testing a
different quantity than the estimator computes.

Whether Java is *right* to freeze them is a separate question (it is not; the
lumped inertia is stale by construction).  This file measures parity.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from invariant_estimation.config import load_config
from invariant_estimation.replay.logsource import joint_channel, read_window

from .conftest import FILTERED_JOINTS, WINDOW

# Current agreement on diag(Qa) against the 2026-07-17 Alex001 run, with the
# off-path joints frozen at qpos0 the way Mecano freezes them (module docstring).
# The remaining ~0.2% is sub-tick sampling: per joint the mean ratio sits within
# 0.07% of 1 with ~0.05% spread, and it is not reducible by a tick shift (the
# -1/0/+1 sweep moves the RMS between 0.039% and 0.086%).
QA_REL_TOL = 0.005


@pytest.fixture(scope="module")
def mj(model_spec):
    m = mujoco.MjModel.from_xml_string(model_spec.mjcf)
    return m, mujoco.MjData(m)


@pytest.fixture(scope="module")
def hinge_names(mj):
    m, _ = mj
    return [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(m.njnt)
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]


@pytest.fixture(scope="module")
def qa_window(log_dir, ihmclog_module, hinge_names):
    """Java's diag(Qa) and everything needed to recompute it, tick-aligned."""
    reader = ihmclog_module.LogReader(str(log_dir))
    measured = {j: joint_channel(reader, j, "q") for j in hinge_names}
    names = (
        sorted(set(measured.values()))
        + [f"jointKF_q_{j}" for j in FILTERED_JOINTS]
        + [f"jointKF_QaDiag_{j}" for j in FILTERED_JOINTS]
        + ["jointKFInitialized"]
    )
    window = read_window(log_dir, names, start=WINDOW[0], end=WINDOW[1], stride=100)
    return window, measured


def _lambda_and_qa(mj, hinge_names, q_by_joint, sigma_tau, base_quat=(1.0, 0.0, 0.0, 0.0)):
    """The port's `Lambda` and `diag(Qa)` at one configuration.

    Written out here rather than called through `jointKF.process` on purpose:
    this file is the *oracle*, and an oracle that shares code with the thing it
    checks proves nothing. `test_matches_the_production_process_module` below is
    what ties the two together.
    """
    m, d = mj
    d.qpos[:] = 0.0
    d.qpos[3:7] = base_quat
    # Only the CONSIDERED joints get a live angle. Everything else stays at zero,
    # because that is the configuration Mecano froze their lumped inertia at --
    # see "...and locked at which configuration" in the module docstring.
    for j in hinge_names:
        if j in FILTERED_JOINTS:
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q_by_joint[j]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_crb(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)

    f = np.array(
        [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FILTERED_JOINTS]
    )
    b = np.arange(6)  # base only: every other joint is LOCKED, not marginalised
    lam = M[np.ix_(f, f)] - M[np.ix_(f, b)] @ np.linalg.solve(M[np.ix_(b, b)], M[np.ix_(b, f)])
    lam = 0.5 * (lam + lam.T)
    Y = np.linalg.solve(lam, np.diag(sigma_tau))
    return lam, np.diag(Y @ Y.T)


@pytest.fixture(scope="module")
def sigma_tau(model_spec):
    jk = load_config()["joint_kf"]
    return np.array(
        [
            jk["alpha_overrides"].get(j, jk["alpha_default"]) * model_spec.effort_limits[j]
            for j in FILTERED_JOINTS
        ]
    )


# --------------------------------------------------------------------------
# properties that must hold exactly -- no tolerance negotiation
# --------------------------------------------------------------------------


def test_lambda_is_invariant_to_base_orientation(mj, hinge_names, qa_window, sigma_tau):
    r"""`Lambda` must not depend on where the robot is standing.

    MuJoCo expresses the free joint's DoFs in the world frame, so ``M_bb`` and
    ``M_fb`` both change with base orientation -- but the Schur complement is
    invariant under any invertible change of the *nuisance* coordinates:

        M_fb T (T' M_bb T)^-1 T' M_bf  =  M_fb M_bb^-1 M_bf

    If this fails, the block gather is wrong (almost certainly the base DoFs are
    not 0..5, or filtered columns are being mixed into the nuisance set). It also
    licenses every other test here to use an arbitrary base pose, which the log
    does not need to supply.
    """
    window, measured = qa_window
    q = {j: window[measured[j]][0] for j in hinge_names}
    upright, _ = _lambda_and_qa(mj, hinge_names, q, sigma_tau)
    tilted, _ = _lambda_and_qa(
        mj, hinge_names, q, sigma_tau, base_quat=(0.9238795, 0.0, 0.3826834, 0.0)
    )
    assert np.abs(upright / tilted - 1).max() < 1e-12


def test_lambda_is_dominated_by_the_rotor_floor_where_it_should_be(mj, hinge_names, qa_window, sigma_tau):
    """Weyl's floor is doing its job: ``lambda_min(Lambda_eff) >= min rotor``.

    This is the property that retired the Alex002 velocity-covariance blow-up
    (`process.py`), so it is worth asserting on real configurations rather than
    trusting the algebra.
    """
    window, measured = qa_window
    jk = load_config()["joint_kf"]
    floor = min(
        jk["rotor_inertia"].get(
            next((k for k in jk["rotor_inertia"] if k in j.upper()), ""),
            jk["rotor_inertia_default"],
        )
        for j in FILTERED_JOINTS
    )
    for t in range(0, len(window.time), 10):
        q = {j: window[measured[j]][t] for j in hinge_names}
        lam, _ = _lambda_and_qa(mj, hinge_names, q, sigma_tau)
        assert np.linalg.eigvalsh(lam).min() >= floor - 1e-9


def test_qa_stays_under_the_tripwire(mj, hinge_names, qa_window, sigma_tau):
    """QA_MAX is a tripwire, not a clamp -- but on this run it must not trip.

    The log agrees: ``jointKFQaCapWouldBindCount`` stays at 0 for the whole run.
    """
    window, measured = qa_window
    qa_max = load_config()["joint_kf"]["qa_max"]
    for t in range(0, len(window.time), 10):
        q = {j: window[measured[j]][t] for j in hinge_names}
        _, qa = _lambda_and_qa(mj, hinge_names, q, sigma_tau)
        assert qa.max() < qa_max


# --------------------------------------------------------------------------
# the parity comparison itself
# --------------------------------------------------------------------------


def test_qa_diagonal_matches_the_java_estimator(mj, hinge_names, qa_window, sigma_tau):
    """diag(Qa) recomputed here vs what the Java filter published, tick by tick.

    Reported per joint on failure -- an aggregate max would hide the structure
    that makes these numbers diagnosable (the legs and the spine disagree with
    opposite sign, which is what says "model", not "constant").
    """
    window, measured = qa_window
    assert window["jointKFInitialized"].min() == 1.0, "window includes uninitialised ticks"

    errors = np.zeros((len(window.time), len(FILTERED_JOINTS)))
    for t in range(len(window.time)):
        q = {
            j: window[f"jointKF_q_{j}"][t] if j in FILTERED_JOINTS else window[measured[j]][t]
            for j in hinge_names
        }
        _, qa = _lambda_and_qa(mj, hinge_names, q, sigma_tau)
        java = np.array([window[f"jointKF_QaDiag_{j}"][t] for j in FILTERED_JOINTS])
        errors[t] = qa / java - 1.0

    worst = np.abs(errors).max(axis=0)
    report = "\n".join(
        f"    {j:<14s} max rel err {100 * e:6.2f}%  (mean ratio {1 + errors[:, i].mean():.4f})"
        for i, (j, e) in enumerate(zip(FILTERED_JOINTS, worst))
    )
    assert worst.max() < QA_REL_TOL, f"diag(Qa) parity regressed:\n{report}"


def test_matches_the_production_process_module(mj, hinge_names, qa_window, sigma_tau):
    """The oracle above and `jointKF.process` must agree.

    Without this the parity test only validates a private reimplementation. With
    it, the parity number transfers to the module the estimator actually runs.
    """
    from invariant_estimation.jointKF.process import qa_from_lambda_eff

    window, measured = qa_window
    q = {j: window[measured[j]][0] for j in hinge_names}
    lam, qa_ref = _lambda_and_qa(mj, hinge_names, q, sigma_tau)
    qa_mod = np.asarray(qa_from_lambda_eff(lam, sigma_tau))
    assert np.abs(np.diag(qa_mod) / qa_ref - 1).max() < 1e-12


def test_the_whole_lambda_chain_matches_the_production_model(mj, hinge_names, qa_window, sigma_tau):
    """`MjxModel` + `process.schur_complement` must reproduce the oracle's `Lambda`.

    The test above only ties the *Gram* step (`Lambda -> Qa`). This one ties the
    two decisions the log actually caught:

    * `MjxModel.qpos` leaves off-path joints at `qpos0`, matching Mecano's
      construct-time freeze -- if it ever started widening a full `qpos` from
      live sensor angles, `Lambda` would move by up to 14%;
    * `MjxModel.dof_nuisance` is base + *gap* joints. On Alex there are no gap
      joints, so it must be exactly `range(6)`; marginalising the off-path DoFs
      instead is worth 58%.
    """
    from invariant_estimation.jointKF.process import schur_complement
    from invariant_estimation.model.mjx_model import MjxModel

    m, _ = mj
    window, measured = qa_window
    model = MjxModel.from_mj_model(m, joint_names=tuple(FILTERED_JOINTS))
    assert list(model.dof_nuisance) == list(range(6)), (
        "Alex has no gap joints: every unfiltered hinge is off the root->filtered "
        "paths and must stay LOCKED inside the composited inertia"
    )

    q = {j: window[f"jointKF_q_{j}"][0] for j in FILTERED_JOINTS}
    lam_ref, _ = _lambda_and_qa(mj, hinge_names, {**{j: 0.0 for j in hinge_names}, **q}, sigma_tau)
    lam_mod = np.asarray(
        schur_complement(
            model.mass_matrix(np.array([q[j] for j in FILTERED_JOINTS])),
            model.joint_dof,
            model.dof_nuisance,
        )
    )
    assert np.abs(lam_mod / lam_ref - 1).max() < 1e-9
