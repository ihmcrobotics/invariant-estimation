r"""
G9 gate — the fused estimator step (`pipeline/main_estimator.py`).

No direct Java analogue: Java's `InvariantMainStateEstimator` is driven by the
controller tick, so there is nothing to port line-for-line. CLAUDE.md §3 G9 asks
for the *analogue* — five scenarios against the MJX model plus the constant-graph
(I7) proof. The strongest available G9 gate is the hardware replay (diff against
`jointKF_*` / `invariantFilter*`), but that needs the 9GB log and is CI-excluded;
these synthetic scenarios are the self-contained gate.

The model here is a minimal floating-base biped: a pelvis carrying the base IMU,
two legs of hip(X)+knee(Y) each ending in a foot IMU and a sole site. That is the
smallest model that exercises everything the fusion touches — two IMU pairs on a
shared base IMU (the `LΣLᵀ` star), a 4-joint filtered state, two stance anchors,
and two InEKF contacts.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.model.mjx_model import MjxModel
from invariant_estimation.pipeline.main_estimator import (
    FusedSensors,
    build_fused_estimator,
    init_fused_carry,
    make_fused_step,
    run_fused,
)

DT = 1.0e-3
G = 9.81

IMU_SITES = ("imu_base", "imu_L", "imu_R")
FOOT_SITES = ("sole_L", "sole_R")
PAIRS = ((0, 1), (0, 2))          # base IMU vs each foot IMU — a star
N = 4                             # filtered joints: L_hip, L_knee, R_hip, R_knee
M = 3                             # IMUs
K = 2                             # feet / contacts


# ---------------------------------------------------------------------------
# Synthetic biped
# ---------------------------------------------------------------------------

def _biped_mjcf() -> str:
    """Minimal floating-base biped MJCF with explicit inertials and armature.

    Explicit `<inertial>` (no geoms) so every mass property is a chosen number and
    nothing is inferred; `armature` on every hinge so the joint-KF rotor inertia
    reaches `qM` pre-Schur (the production path, CLAUDE.md §6).
    """
    inertial = '<inertial pos="0 0 -0.15" mass="1.5" diaginertia="0.02 0.02 0.01"/>'

    def leg(side: str, y: float) -> str:
        return (
            f'<body name="{side}_thigh" pos="0 {y} 0">'
            f'<joint name="{side}_HIP_X" type="hinge" axis="1 0 0" armature="0.01" limited="false"/>'
            f'{inertial}'
            f'<body name="{side}_shank" pos="0 0 -0.4">'
            f'<joint name="{side}_KNEE_Y" type="hinge" axis="0 1 0" armature="0.01" limited="false"/>'
            f'{inertial}'
            f'<site name="imu_{side}" pos="0 0 -0.2"/>'
            f'<site name="sole_{side}" pos="0 0 -0.4"/>'
            f'</body></body>'
        )

    return (
        '<mujoco model="g9_biped">'
        '<compiler angle="radian"/>'
        '<option gravity="0 0 -9.81"/>'
        '<worldbody>'
        '<body name="pelvis" pos="0 0 1">'
        '<freejoint name="floating_base"/>'
        '<inertial pos="0 0 0" mass="7.5" diaginertia="0.12 0.19 0.23"/>'
        '<site name="imu_base" pos="0 0 0"/>'
        f'{leg("L", 0.1)}{leg("R", -0.1)}'
        '</body></worldbody></mujoco>'
    )


@pytest.fixture(scope="module")
def fused():
    model = MjxModel.from_xml_string(
        _biped_mjcf(),
        site_names=IMU_SITES + FOOT_SITES,
        pairs=PAIRS,
    )
    assert model.n_joints == N, f"expected {N} filtered joints, got {model.n_joints}"
    est = build_fused_estimator(
        model, imu_sites=IMU_SITES, pairs=PAIRS, foot_sites=FOOT_SITES,
        base_imu=0, dt=DT,
    )
    assert est.n_joints == N and est.n_contacts == K
    return est


# ---------------------------------------------------------------------------
# Sensor helpers
# ---------------------------------------------------------------------------

def _sensors(
    *,
    q=None, gyro=None, accel=None, contact=None, contact_chol=None, n_u=0,
) -> FusedSensors:
    """One tick of `FusedSensors` (all float64), with sensible level-rest defaults.

    `gyro` is the common per-IMU rate (same in every IMU frame — a rigid-body
    motion with the legs at `q0=0`, so the relative-gyro measurement is zero and no
    joint velocity is implied).
    """
    q = np.zeros(N) if q is None else np.asarray(q, float)
    gyro = np.zeros(3) if gyro is None else np.asarray(gyro, float)
    accel = np.array([0.0, 0.0, G]) if accel is None else np.asarray(accel, float)
    contact = np.ones(K) if contact is None else np.asarray(contact, float)
    if contact_chol is None:
        contact_chol = np.tile(np.eye(3) * 1.0e-4, (K, 1, 1))
    return FusedSensors(
        encoders=jnp.asarray(q, dtype=jnp.float64),
        gyros=jnp.asarray(np.tile(gyro, (M, 1)), dtype=jnp.float64),
        accel_base=jnp.asarray(accel, dtype=jnp.float64),
        qd_unfiltered=jnp.zeros(n_u, dtype=jnp.float64),
        contact=jnp.asarray(contact, dtype=jnp.float64),
        contact_chol=jnp.asarray(contact_chol, dtype=jnp.float64),
    )


def _stack(sensors_list) -> FusedSensors:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *sensors_list)


def _rotation_angle(R) -> float:
    """Geodesic angle of a rotation matrix, radians."""
    c = (np.trace(np.asarray(R)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _tilt(R) -> float:
    """Angle between the body z-axis and world up — roll/pitch magnitude."""
    z = np.asarray(R)[:, 2]
    return float(np.arccos(np.clip(z[2], -1.0, 1.0)))


def _is_psd(P, tol=1e-9) -> bool:
    P = np.asarray(P)
    return bool(np.allclose(P, P.T, atol=1e-8) and np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -tol)


# ---------------------------------------------------------------------------
# Mechanics — assembles, runs, stays finite/PSD
# ---------------------------------------------------------------------------

def test_assembles_and_runs(fused):
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    step = make_fused_step(fused)
    (jkf_c, inekf_c), out = step(carry, _sensors())

    assert out.R.shape == (3, 3) and out.p.shape == (3,)
    assert out.q.shape == (N,) and out.bias.shape == (3 * M,)
    assert np.all(np.isfinite(np.asarray(out.p)))
    assert _is_psd(jkf_c.state.P) and _is_psd(inekf_c.state.P)
    # R stays a rotation.
    assert np.allclose(np.asarray(out.R) @ np.asarray(out.R).T, np.eye(3), atol=1e-9)
    assert np.isfinite(float(out.inekf.contact_diagnostics.nis))


def test_run_scans_a_trajectory_psd_throughout(fused):
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    T = 100
    sensors = _stack([_sensors() for _ in range(T)])
    (jkf_c, inekf_c), out = run_fused(fused, carry, sensors)

    assert out.p.shape == (T, 3) and out.R.shape == (T, 3, 3)
    assert np.all(np.isfinite(np.asarray(out.p)))
    for k in range(0, T, 20):
        assert _is_psd(out.inekf.state.P[k])


# ---------------------------------------------------------------------------
# I7 — the constant-graph proof (the load-bearing G9 gate)
# ---------------------------------------------------------------------------

def _jaxpr(fused, carry, sensors) -> str:
    return str(jax.make_jaxpr(make_fused_step(fused))(carry, sensors))


def test_jaxpr_constant_across_contact_and_gate(fused):
    """I7: flipping a contact AND the quasi-static gate must not change the graph.

    This is CLAUDE.md §3 G9's "jaxpr-hash constancy of the fused step across
    differing contact masks / gate states" — the port's proof of no per-tick
    allocation. Every gate inside both filters is a `jnp.where` mask, so the traced
    graph is identical; a Python `if`/`nonzero` on a traced value would raise at
    trace time instead.
    """
    carry = init_fused_carry(fused, q0=jnp.zeros(N))

    planted_quiet = _sensors()                                    # feet down, gate open
    swing_shaken = _sensors(
        contact=np.zeros(K),                                      # both feet swing
        contact_chol=np.tile(np.eye(3) * 1.0, (K, 1, 1)),
        accel=np.array([3.0, 0.0, 9.0]),                          # gate closed
        gyro=np.array([0.0, 0.9, 0.0]),
    )
    mixed = _sensors(
        contact=np.array([1.0, 0.0]),
        contact_chol=np.stack([np.eye(3) * 1e-6, np.eye(3) * 10.0]),
    )

    ref = _jaxpr(fused, carry, planted_quiet)
    assert _jaxpr(fused, carry, swing_shaken) == ref, "contact/gate changed the graph — I7"
    assert _jaxpr(fused, carry, mixed) == ref, "mixed contact changed the graph — I7"


def test_step_does_not_recompile_across_conditions(fused):
    """Operational I7: one XLA trace serves every contact/gate condition."""
    step = jax.jit(make_fused_step(fused))
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    for contact, scale, accel in [
        (np.ones(K), 1e-4, [0.0, 0.0, G]),
        (np.zeros(K), 1.0, [3.0, 0.0, 9.0]),
        (np.array([1.0, 0.0]), 1e-6, [0.0, 0.0, G]),
        (np.ones(K), 1e-2, [0.0, 0.5, G]),
    ]:
        s = _sensors(contact=contact,
                     contact_chol=np.tile(np.eye(3) * scale, (K, 1, 1)),
                     accel=np.array(accel))
        carry, _ = step(carry, s)
    assert step._cache_size() == 1


# ---------------------------------------------------------------------------
# Scenario 1 — static equilibrium: the pose holds, everything stays bounded
# ---------------------------------------------------------------------------

def test_static_equilibrium_holds(fused):
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    T = 300
    sensors = _stack([_sensors() for _ in range(T)])
    (jkf_c, inekf_c), out = run_fused(fused, carry, sensors)

    # Level, at rest, feet planted, sensors perfectly consistent ⇒ the estimate
    # must not wander: tilt stays tiny, base stays put, bias stays bounded.
    assert _tilt(out.R[-1]) < np.deg2rad(1.0)
    assert float(jnp.linalg.norm(out.v[-1])) < 1e-2
    assert float(jnp.linalg.norm(out.p[-1])) < 1e-2
    assert float(jnp.linalg.norm(out.bias[-1])) < 1e-2
    assert _is_psd(inekf_c.state.P) and _is_psd(jkf_c.state.P)
    assert np.all(np.isfinite(np.asarray(out.inekf.contact_diagnostics.nis)))


# ---------------------------------------------------------------------------
# Scenario 2 — no-contact rotation: pure gyro integration, no contact fighting it
# ---------------------------------------------------------------------------

def test_no_contact_rotation_integration(fused):
    """Feet in swing (untrusted + large Σ_C), constant yaw rate ⇒ R integrates it.

    Yaw is unobservable (gravity leveling is rank-2, null along e_z; no trusted
    anchor), so the base gyro integrates freely and the estimate should track
    `∫ω dt` while roll/pitch stay level.
    """
    omega_z = 0.5
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    T = 100
    sensors = _stack([
        _sensors(
            gyro=np.array([0.0, 0.0, omega_z]),
            contact=np.zeros(K),
            contact_chol=np.tile(np.eye(3) * 1.0, (K, 1, 1)),
        )
        for _ in range(T)
    ])
    _, out = run_fused(fused, carry, sensors)

    expected = omega_z * T * DT
    assert _rotation_angle(out.R[-1]) == pytest.approx(expected, abs=5e-3)
    assert _tilt(out.R[-1]) < np.deg2rad(0.5)          # roll/pitch stay level
    # It is a yaw rotation: the body z-axis is unchanged.
    assert np.allclose(np.asarray(out.R[-1])[:, 2], [0, 0, 1], atol=1e-3)


# ---------------------------------------------------------------------------
# Scenario 3 — free fall: no runaway, the base accelerates downward under gravity
# ---------------------------------------------------------------------------

def test_free_fall_no_runaway(fused):
    """Zero specific force (accelerometer in free fall) ⇒ world accel = g, finite.

    The InEKF adds gravity internally, so a zero reading integrates to a clean
    downward parabola rather than diverging. Feet are in swing so no contact update
    fights the fall.
    """
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    T = 100
    sensors = _stack([
        _sensors(
            accel=np.zeros(3),
            contact=np.zeros(K),
            contact_chol=np.tile(np.eye(3) * 1.0, (K, 1, 1)),
        )
        for _ in range(T)
    ])
    _, out = run_fused(fused, carry, sensors)

    assert np.all(np.isfinite(np.asarray(out.p)))
    t = T * DT
    # Free-fall drop ≈ ½ g t²; sign and rough magnitude, not a tight fit.
    assert float(out.p[-1][2]) < 0.0
    assert float(out.p[-1][2]) == pytest.approx(-0.5 * G * t * t, rel=0.3)


# ---------------------------------------------------------------------------
# Scenario 4 — poisoned sensor: a bad tick is gated, the filter recovers
# ---------------------------------------------------------------------------

def test_poisoned_encoder_tick_is_gated_and_recovers(fused):
    """A single NaN encoder tick must be skipped (never propagated) — NaN hardening.

    The joint KF gates a non-finite measurement (masked K, not a latch); the fused
    pose must stay finite through the bad tick and be indistinguishable afterward
    from a clean run.
    """
    carry = init_fused_carry(fused, q0=jnp.zeros(N))
    step = make_fused_step(fused)

    clean = _sensors()
    poisoned = clean._replace(encoders=clean.encoders.at[0].set(jnp.nan))

    c = carry
    for k in range(50):
        c, out = step(c, poisoned if k == 25 else clean)
        assert np.all(np.isfinite(np.asarray(out.p))), f"non-finite pose at tick {k}"
        assert np.all(np.isfinite(np.asarray(out.q)))

    # After recovery the estimate is still level and bounded.
    assert _tilt(out.R) < np.deg2rad(1.0)
    assert float(jnp.linalg.norm(out.bias)) < 1e-2


# ---------------------------------------------------------------------------
# Boundary contract — the joint-KF bias actually corrects the InEKF gyro (I1)
# ---------------------------------------------------------------------------

def test_bias_correction_reaches_the_inekf(fused):
    """A nonzero base-IMU bias in the carry must change the propagated attitude.

    The boundary bias-corrects the base gyro before the InEKF integrates it (I1),
    so injecting a bias into the joint-KF carry and holding the raw gyro fixed must
    move the InEKF rotation relative to a zero-bias run — proof the correction is
    wired, not dropped.
    """
    step = make_fused_step(fused)
    base = init_fused_carry(fused, q0=jnp.zeros(N))
    jkf_c, inekf_c = base

    # Same everything, but seed a base-IMU bias of 0.2 rad/s about z.
    biased_x = jkf_c.state.x.at[2 * N].set(0.0).at[2 * N + 2].set(0.2)
    biased = (jkf_c._replace(state=jkf_c.state._replace(x=biased_x)), inekf_c)

    # Raw gyro reads the true rate (0.2 about z); with a matched bias the corrected
    # rate is ~0, so the two runs must diverge in yaw.
    s = _sensors(gyro=np.array([0.0, 0.0, 0.2]),
                 contact=np.zeros(K),
                 contact_chol=np.tile(np.eye(3) * 1.0, (K, 1, 1)))
    (_, ic0), _ = step(base, s)       # zero bias  ⇒ integrates the full 0.2 rad/s
    (_, ic1), _ = step(biased, s)     # matched bias ⇒ corrected rate is smaller

    a0 = _rotation_angle(ic0.state.R)
    a1 = _rotation_angle(ic1.state.R)
    assert a0 > 1e-4                  # the unbiased run integrated a real yaw
    assert a1 < 0.9 * a0             # the bias correction subtracted from it (I1 wired)
