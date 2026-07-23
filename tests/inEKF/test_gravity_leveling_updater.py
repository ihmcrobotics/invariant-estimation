"""1:1 port of ``GravityLevelingUpdaterTest.java`` (TEST_SUITE_MAP.md
§invariant_estimator contact / support tests) onto `inEKF.gravity_update`.

Java class constants preserved verbatim::

    G = 9.81                      UP = (0,0,1)
    ROLL_VAR  = 2.5e-3            (~(2.9°)²)
    PITCH_VAR = 1.9e-1            (~(25°)²)
    DT = 1.0e-3
    BALANCE_OMEGA = 2π·0.49       (0.49 Hz hardware balance mode)
    PREDICTED_ARTIFACT_GAIN = 1/√(1+(BALANCE_OMEGA·5.0)²) ≈ 0.065

The whole class is deterministic — no RNG anywhere — which the map calls the
most portable behavioural spec in the suite.

Java's mutable updater (`setPitchObservable`, an internally-held gravity
reference) becomes explicit arguments and a carried `GravityRef` pytree; the
Java `InvariantEKF` used by the filter-level tests is stood up here as a
three-line local driver (`_level_once`), since the real orchestrator is G5.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import state as s

gu = importlib.import_module("invariant_estimation.inEKF.gravity_update")

from ._oracles import (
    assert_symmetric_psd,
    matrix_to_yaw_pitch_roll,
    yaw_pitch_roll_to_matrix,
)

G = 9.81
UP = np.array([0.0, 0.0, 1.0])
ROLL_VAR = 2.5e-3
PITCH_VAR = 1.9e-1
DT = 1.0e-3
BALANCE_OMEGA = 2.0 * np.pi * 0.49
PREDICTED_ARTIFACT_GAIN = 1.0 / np.sqrt(1.0 + (BALANCE_OMEGA * 5.0) ** 2)

ZERO_OMEGA = jnp.zeros(3)


# ---------------------------------------------------------------------------
# Java helpers / drivers
# ---------------------------------------------------------------------------

def _state(N=0, R=None, P=None):
    """`InvariantState(N)` with an optional rotation and covariance."""
    st = s.InEKFState.identity(N)
    if R is not None:
        st = st._replace(R=jnp.asarray(R))
    if P is not None:
        st = st._replace(P=jnp.asarray(P))
    return st


def _settle_gravity_reference(ref, specific_force, params, ticks=3000):
    """Java ``settleGravityReference``: 3000x updateGravityReference at zero omega.

    Run through `lax.scan` — identical arithmetic and tick count, but it keeps
    the suite fast (12000-tick sway tests below are eager-loop prohibitive).
    """
    def body(r, _):
        return gu.update_gravity_reference(r, specific_force, ZERO_OMEGA, DT, params), None

    ref, _ = jax.lax.scan(body, ref, None, length=ticks)
    return ref


def _level_once(state, ref, specific_force, params):
    """One assemble + apply — the Java EKF's assembleGravityLeveling/apply pair.

    Stands in for `InvariantEKF` (G5); the gravity module itself is what is
    under test, so the driver is deliberately thin.
    """
    meas = gu.assemble_gravity_leveling(ref, state, specific_force, params)
    state, diag = gu.apply_gravity_leveling(state, meas, params)
    return state, meas, diag


def _tilt(state):
    return float(gu.tilt_angle(state))


def _one_step_tilt_after_correction(yaw, pitch, roll, p0, params):
    """Java ``oneStepTiltAfterCorrection``: one leveling step from a P = p0·I prior."""
    state = _state(0, R=yaw_pitch_roll_to_matrix(yaw, pitch, roll),
                   P=p0 * jnp.eye(9))
    state, _, _ = _level_once(state, gu.init_gravity_ref(),
                              jnp.array([0.0, 0.0, G]), params)
    assert_symmetric_psd(state.P)
    return _tilt(state)


def _quadratic_form(R, u):
    u = np.asarray(u)
    return float(u @ np.asarray(R) @ u)


# ---------------------------------------------------------------------------
# testUprightGivesZeroResidualAndTilt
# ---------------------------------------------------------------------------

def test_upright_gives_zero_residual_and_tilt():
    params = gu.isotropic_gravity_params(2.5e-3, G)
    meas = gu.assemble_gravity_leveling(
        gu.init_gravity_ref(), _state(0), jnp.array([0.0, 0.0, G]), params
    )
    assert np.max(np.abs(np.asarray(meas.residual))) < 1.0e-12
    assert abs(float(meas.tilt_angle)) < 1.0e-12


# ---------------------------------------------------------------------------
# testPitchTiltDiagnosticAndResidual
# ---------------------------------------------------------------------------

def test_pitch_tilt_diagnostic_and_residual():
    """Body pitched +θ ⇒ residual [−sinθ, 0, cosθ−1], and H's yaw column is 0."""
    theta = 0.20
    params = gu.isotropic_gravity_params(2.5e-3, G)
    specific_force = jnp.array([-G * np.sin(theta), 0.0, G * np.cos(theta)])

    meas = gu.assemble_gravity_leveling(
        gu.init_gravity_ref(), _state(0), specific_force, params
    )

    assert float(meas.tilt_angle) == pytest.approx(abs(theta), abs=1.0e-9)
    assert float(meas.tilt_pitch) == pytest.approx(-np.sin(theta), abs=1.0e-9)
    assert float(meas.tilt_roll) == pytest.approx(0.0, abs=1.0e-9)

    expected = np.array([-np.sin(theta), 0.0, np.cos(theta) - 1.0])
    assert np.max(np.abs(np.asarray(meas.residual) - expected)) < 1.0e-9

    # Yaw unobservable: the δφ_z column is exactly zero (Java asserts tol 0.0).
    assert jnp.array_equal(meas.H[:, 2], jnp.zeros(3))


# ---------------------------------------------------------------------------
# testGravityUpdateLevelsTiltAndPreservesYaw
# ---------------------------------------------------------------------------

def test_gravity_update_levels_tilt_and_preserves_yaw():
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    true_gravity = jnp.array([0.0, 0.0, G])

    # (1) pitched 0.20, 200 iterations.
    state = _state(0, R=yaw_pitch_roll_to_matrix(0.0, 0.20, 0.0), P=jnp.eye(9))
    ref = gu.init_gravity_ref()
    for _ in range(200):
        state, meas, diag = _level_once(state, ref, true_gravity, params)
        ref = meas.ref
        assert float(diag.applied) == 1.0
        assert np.isfinite(float(diag.condition_proxy))
        assert_symmetric_psd(state.P)

    assert _tilt(state) < 1.0e-3
    yaw, _, roll = matrix_to_yaw_pitch_roll(state.R)
    assert yaw == pytest.approx(0.0, abs=1.0e-9)
    assert roll == pytest.approx(0.0, abs=1.0e-9)

    # (2) yawed 0.7 and upright: yaw must survive untouched.
    state = _state(0, R=yaw_pitch_roll_to_matrix(0.7, 0.0, 0.0), P=jnp.eye(9))
    ref = gu.init_gravity_ref()
    for _ in range(50):
        state, meas, _ = _level_once(state, ref, true_gravity, params)
        ref = meas.ref

    yaw, _, _ = matrix_to_yaw_pitch_roll(state.R)
    assert yaw == pytest.approx(0.7, abs=1.0e-9)
    assert _tilt(state) < 1.0e-9


# ---------------------------------------------------------------------------
# testAnisotropicMeasurementCovarianceStructure
# ---------------------------------------------------------------------------

def test_anisotropic_measurement_covariance_structure():
    """R = diag(PITCH_VAR, ROLL_VAR, ROLL_VAR) at upright — pitch trusted least."""
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    meas = gu.assemble_gravity_leveling(
        gu.init_gravity_ref(), _state(0), jnp.array([0.0, 0.0, G]), params
    )
    R = np.asarray(meas.R)

    assert R[0, 0] == pytest.approx(PITCH_VAR, abs=1.0e-12)
    assert R[1, 1] == pytest.approx(ROLL_VAR, abs=1.0e-12)
    assert R[2, 2] == pytest.approx(ROLL_VAR, abs=1.0e-12)
    off = R - np.diag(np.diag(R))
    assert np.max(np.abs(off)) < 1.0e-12
    assert np.max(np.abs(R - R.T)) < 1.0e-15


# ---------------------------------------------------------------------------
# testPitchCorrectionAuthorityBelowRoll
# ---------------------------------------------------------------------------

def test_pitch_correction_authority_below_roll():
    """Anisotropy must actually bite: roll corrects >5× harder than pitch."""
    theta, p0 = 0.10, 1.0e-2
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)

    roll_reduction = theta - _one_step_tilt_after_correction(0.0, 0.0, theta, p0, params)
    pitch_reduction = theta - _one_step_tilt_after_correction(0.0, theta, 0.0, p0, params)

    assert pitch_reduction > 0.0
    assert roll_reduction > 5.0 * pitch_reduction


# ---------------------------------------------------------------------------
# testRollStillLevelsUnderAnisotropy
# ---------------------------------------------------------------------------

def test_roll_still_levels_under_anisotropy():
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    state = _state(0, R=yaw_pitch_roll_to_matrix(0.0, 0.0, 0.20), P=jnp.eye(9))
    ref = gu.init_gravity_ref()

    for _ in range(200):
        state, meas, _ = _level_once(state, ref, jnp.array([0.0, 0.0, G]), params)
        ref = meas.ref

    assert _tilt(state) < 1.0e-3


# ---------------------------------------------------------------------------
# testPitchGateFreezesPitchButNotRoll
# ---------------------------------------------------------------------------

def test_pitch_gate_freezes_pitch_but_not_roll():
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    meas = gu.assemble_gravity_leveling(
        gu.init_gravity_ref(), _state(0), jnp.array([0.0, 0.0, G]), params,
        pitch_observable=False,
    )
    R = np.asarray(meas.R)

    assert R[0, 0] > 1.0e3
    assert R[1, 1] == pytest.approx(ROLL_VAR, abs=1.0e-12)


# ---------------------------------------------------------------------------
# testHorizontalAccelGateRejectsForeAftButPassesGravity
# ---------------------------------------------------------------------------

def test_horizontal_accel_gate_rejects_fore_aft_but_passes_gravity():
    """The norm gate alone would pass a 3 m/s² fore-aft push — the horizontal one won't."""
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    gravity = jnp.array([0.0, 0.0, G])
    ref = _settle_gravity_reference(gu.init_gravity_ref(), gravity, params)

    assert bool(gu.is_quasi_static(ref, gravity, ZERO_OMEGA, params))

    fore_aft = jnp.array([3.0, 0.0, np.sqrt(G**2 - 9.0)])
    # The norm gate alone would let this through: ‖foreAft‖ is within 5% of g.
    assert abs(float(jnp.linalg.norm(fore_aft)) - G) <= 0.05 * G

    ref = gu.update_gravity_reference(ref, fore_aft, ZERO_OMEGA, DT, params)
    assert not bool(gu.is_quasi_static(ref, fore_aft, ZERO_OMEGA, params))


# ---------------------------------------------------------------------------
# testQuasiStaticGateIsIndependentOfEstimatorAttitude  (regression, FINDINGS §F.3)
# ---------------------------------------------------------------------------

def test_quasi_static_gate_is_independent_of_estimator_attitude():
    """The gate reads the sensor-driven reference, never R̂ᵀe_z (§6 trap).

    The old gate resolved horizontal force against ĝ = R̂ᵀe_z, reading g·sinθ at
    estimator tilt θ — which locked leveling out above 2.92°, exactly when it was
    needed. Robot is static throughout; only the *estimate* tilts.
    """
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    true_gravity = jnp.array([0.0, 0.0, G])
    ref = _settle_gravity_reference(gu.init_gravity_ref(), true_gravity, params)

    for deg in range(0, 16):
        state = _state(0, R=yaw_pitch_roll_to_matrix(0.0, np.radians(deg), 0.0))
        gu.assemble_gravity_leveling(ref, state, true_gravity, params)
        assert bool(gu.is_quasi_static(ref, true_gravity, ZERO_OMEGA, params)), (
            f"gate closed at estimator tilt {deg}° — F.3 regression"
        )


# ---------------------------------------------------------------------------
# testRotationGateUsesRawGyroNotBiasCorrupted
# ---------------------------------------------------------------------------

def test_rotation_gate_uses_raw_gyro():
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    gravity = jnp.array([0.0, 0.0, G])
    ref = _settle_gravity_reference(gu.init_gravity_ref(), gravity, params)

    assert bool(gu.is_quasi_static(ref, gravity, jnp.zeros(3), params))
    # 0.3 rad/s is past the 0.15 gate.
    assert not bool(gu.is_quasi_static(ref, gravity, jnp.array([0.0, 0.3, 0.0]), params))


# ---------------------------------------------------------------------------
# testPitchDistrustAxisIsBodyYAtNonZeroYaw
# ---------------------------------------------------------------------------

def test_pitch_distrust_axis_is_body_y_at_non_zero_yaw():
    """Pitch distrust follows the BODY axis, not the world one, even at 90° yaw."""
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    yawed = yaw_pitch_roll_to_matrix(np.radians(90.0), 0.0, 0.0)
    gravity_body = jnp.asarray(yawed.T @ np.array([0.0, 0.0, G]))

    meas = gu.assemble_gravity_leveling(
        gu.init_gravity_ref(), _state(0, R=yawed), gravity_body, params
    )
    R = meas.R

    g_hat = np.asarray(gravity_body) / np.linalg.norm(np.asarray(gravity_body))
    pitch_dir = np.cross(np.array([0.0, 1.0, 0.0]), g_hat)
    pitch_dir /= np.linalg.norm(pitch_dir)
    roll_dir = np.cross(np.array([1.0, 0.0, 0.0]), g_hat)
    roll_dir /= np.linalg.norm(roll_dir)

    assert _quadratic_form(R, pitch_dir) == pytest.approx(PITCH_VAR, abs=1.0e-9)
    assert _quadratic_form(R, roll_dir) == pytest.approx(ROLL_VAR, abs=1.0e-9)


# ---------------------------------------------------------------------------
# testLateralAccelArtifactIsRejectedAtTheBalanceFrequency   (roll-sway 1)
# ---------------------------------------------------------------------------

def test_lateral_accel_artifact_is_rejected_at_the_balance_frequency():
    """Sway artifact at 0.49 Hz is attenuated ~15×, matching 1/√(1+(ωτ)²)."""
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    amplitude = 0.21                        # m/s² lateral
    raw_artifact = amplitude / G            # the un-filtered residual it would cause

    state = _state(0)                       # never rotates: true tilt is zero
    ref = _settle_gravity_reference(gu.init_gravity_ref(), jnp.array([0.0, 0.0, G]), params)

    ticks = jnp.arange(12000)
    forces = jnp.stack([
        jnp.zeros(12000),
        amplitude * jnp.sin(BALANCE_OMEGA * ticks * DT),
        jnp.full(12000, G),
    ], axis=1)

    def body(r, sf):
        r = gu.update_gravity_reference(r, sf, ZERO_OMEGA, DT, params)
        meas = gu.assemble_gravity_leveling(r, state, sf, params)
        return r, meas.residual[1]

    _, residual_y = jax.lax.scan(body, ref, forces)

    max_residual_y = float(jnp.max(jnp.abs(residual_y[8000:])))
    achieved_gain = max_residual_y / raw_artifact
    assert achieved_gain < 0.1
    assert achieved_gain == pytest.approx(PREDICTED_ARTIFACT_GAIN, abs=0.03)


# ---------------------------------------------------------------------------
# testTrueTiltStillPassesAtUnityGainAtTheBalanceFrequency   (roll-sway 2)
# ---------------------------------------------------------------------------

def test_true_tilt_still_passes_at_unity_gain_at_the_balance_frequency():
    """Real tilt at the same frequency tracks at unity — a naive low-pass fails this."""
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    roll_amplitude = 0.03

    ref = _settle_gravity_reference(gu.init_gravity_ref(), jnp.array([0.0, 0.0, G]), params)

    # Trajectory precomputed in NumPy from the reference YPR construction.
    times = np.arange(12000) * DT
    rolls = roll_amplitude * np.sin(BALANCE_OMEGA * times)
    roll_rates = roll_amplitude * BALANCE_OMEGA * np.cos(BALANCE_OMEGA * times)
    true_rotations = np.stack([yaw_pitch_roll_to_matrix(0.0, 0.0, r) for r in rolls])
    forces = jnp.asarray(np.einsum("tji,j->ti", true_rotations, np.array([0.0, 0.0, G])))
    omegas = jnp.asarray(np.stack([roll_rates, np.zeros(12000), np.zeros(12000)], axis=1))
    true_directions = np.einsum("tji,j->ti", true_rotations, UP)

    def body(r, xs):
        r = gu.update_gravity_reference(r, xs[0], xs[1], DT, params)
        return r, r.direction

    _, directions = jax.lax.scan(body, ref, (forces, omegas))

    errors = np.linalg.norm(np.asarray(directions)[8000:] - true_directions[8000:], axis=1)
    max_tracking_error = float(errors.max())

    assert max_tracking_error < 0.1 * roll_amplitude


# ---------------------------------------------------------------------------
# testStaticTiltStillProducesFullResidual   (roll-sway 3 — DC authority)
# ---------------------------------------------------------------------------

def test_static_tilt_still_produces_full_residual():
    """DC tilt is undiminished by the reference filter — full sin(tilt) residual."""
    params = gu.default_gravity_params(gravity=G, roll_var=ROLL_VAR, pitch_var=PITCH_VAR)
    roll = 0.05
    true_rotation = yaw_pitch_roll_to_matrix(0.0, 0.0, roll)
    sf = jnp.asarray(true_rotation.T @ np.array([0.0, 0.0, G]))

    ref = _settle_gravity_reference(gu.init_gravity_ref(), sf, params)
    meas = gu.assemble_gravity_leveling(ref, _state(0), sf, params)

    assert float(meas.residual[0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(meas.residual[1]) == pytest.approx(np.sin(roll), abs=1.0e-4)
    assert float(meas.residual[2]) == pytest.approx(np.cos(roll) - 1.0, abs=1.0e-4)


# ---------------------------------------------------------------------------
# Port-specific oracle (CLAUDE.md G4: "gravity H rank 2, null along e_z")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ypr", [(0.0, 0.0, 0.0), (0.7, 0.2, -0.1), (-2.0, 0.4, 1.1)])
def test_gravity_jacobian_is_rank_two_with_null_along_ez(ypr):
    """H must never be able to produce yaw, at any attitude (§6 trap)."""
    state = _state(2, R=yaw_pitch_roll_to_matrix(*ypr))
    H = gu.gravity_jacobian(state)

    assert H.shape == (3, state.dim)
    assert np.linalg.matrix_rank(np.asarray(H), tol=1e-12) == 2
    # Null direction is exactly e_z in the rotation block …
    assert np.max(np.abs(np.asarray(H[:, 0:3] @ jnp.asarray(UP)))) < 1.0e-15
    # … and gravity says nothing about velocity, position or contacts.
    assert jnp.array_equal(H[:, 3:], jnp.zeros((3, state.dim - 3)))
