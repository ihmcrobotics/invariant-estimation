"""Tests for `inEKF/filter.py` — the scan body and trajectory driver.

No Java analogue: `InvariantEKF` in Java is driven by the controller's tick, so
there is nothing to port. These are port-specific and cover the three things the
scan body must guarantee:

* the joint-KF boundary is routed correctly (`Σ_q` reaches `N^p` through `J`,
  and nothing from the joint KF reaches `Φ` or the inertial `Q`);
* **I7 — no data-dependent branch survives tracing**, and the jitted step never
  recompiles as contact condition or gate state changes (the constant-graph
  property G9 reuses);
* contact condition rides in `Σ_C` alone — see the DECISION note in
  `inEKF/filter.py`, and `test_large_contact_covariance_isolates_a_swing_foot`
  below, which is the regression guarding it;
* the whole scan is differentiable, so BPTT into ContactNet works.

The `ContactKinematics` seam is filled by a deterministic analytic fixture; MJX
implements it for real at G1.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import ekf as ekf_mod
from invariant_estimation.inEKF import state as s
from invariant_estimation.inEKF.contact import digest, reconstruct_cov
from invariant_estimation.inEKF.correct import innovation, linear_update, measurement_noise
from invariant_estimation.inEKF.propagate import propagate
from invariant_estimation.inEKF.filter import (
    ContactFrames,
    InEKFInputs,
    JointFilterOutput,
    contact_position_noise,
    contact_velocity_noise,
    init_carry,
    make_step,
    run,
)

fl = importlib.import_module("invariant_estimation.inEKF.filter")

from ._oracles import assert_symmetric_psd, next_rotation_matrix, next_vector3d

N_CONTACTS = 2
N_JOINTS = 6
DT = 1.0e-3


# ---------------------------------------------------------------------------
# Fixture kinematics — the `robot/` seam, filled analytically
# ---------------------------------------------------------------------------

def _make_kinematics(n_contacts=N_CONTACTS, n_joints=N_JOINTS):
    """A smooth, deterministic stand-in for FK + contact Jacobians.

    Not a robot: it only has to be differentiable, jit-able, correctly shaped,
    and consistent between `y` and `J` so the noise routing is exercised. The
    real one arrives with MJX at G1.
    """
    offsets = jnp.asarray(
        [[0.0, 0.1 * (-1) ** i, -0.9] for i in range(n_contacts)]
    )
    weights = jnp.asarray(
        [[0.1 * ((i + j) % 3 + 1) for j in range(n_joints)] for i in range(n_contacts)]
    )

    def kinematics(q, q_dot):
        # y_i = offset_i + (sum_j w_ij sin q_j) * 1₃  ⇒  ∂y_i/∂q_j = w_ij cos q_j
        y = offsets + (weights * jnp.sin(q)[None, :]) @ jnp.ones((n_joints, 3))
        J = jnp.einsum("ij,j->ij", weights, jnp.cos(q))[:, None, :] * jnp.ones((1, 3, 1))
        J_dot = -jnp.einsum("ij,j->ij", weights, jnp.sin(q) * q_dot)[:, None, :] \
            * jnp.ones((1, 3, 1))
        return ContactFrames(y=y, J=J, J_dot=J_dot)

    return kinematics


def _inputs(rng, contact_chol=None, accel=None, omega=None, contact_meas_chol=None):
    """One tick of inputs."""
    if contact_chol is None:
        contact_chol = jnp.tile(jnp.eye(3) * 1.0e-3, (N_CONTACTS, 1, 1))
    if contact_meas_chol is None:
        # Zeros = no learned FK measurement noise = the pre-ContactNet filter.
        contact_meas_chol = jnp.zeros((N_CONTACTS, 3, 3))
    joint = JointFilterOutput(
        q=jnp.asarray(rng.uniform(-0.5, 0.5, N_JOINTS)),
        q_dot=jnp.asarray(rng.uniform(-0.5, 0.5, N_JOINTS)),
        sigma_q=jnp.eye(N_JOINTS) * 5.0e-5,
        sigma_q_dot=jnp.eye(N_JOINTS) * 1.0e-2,
    )
    return InEKFInputs(
        omega=jnp.zeros(3) if omega is None else omega,
        accel=jnp.array([0.0, 0.0, 9.81]) if accel is None else accel,
        raw_omega=jnp.zeros(3) if omega is None else omega,
        joint=joint,
        contact_chol=jnp.asarray(contact_chol, dtype=float),
        contact_meas_chol=jnp.asarray(contact_meas_chol, dtype=float),
    )


def _stack(inputs_list):
    """Stack a list of `InEKFInputs` along a leading time axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *inputs_list)


def _setup(rng):
    ekf = ekf_mod.create(N_CONTACTS, dt=DT)
    state = ekf_mod.initialize(
        ekf,
        jnp.asarray(next_rotation_matrix(rng)),
        jnp.asarray(next_vector3d(rng)[0]),
        jnp.asarray(next_vector3d(rng)[0]),
        jnp.asarray(next_vector3d(rng, N_CONTACTS)),
        jnp.eye(9 + 3 * N_CONTACTS) * 0.1,
    )
    return ekf, state, _make_kinematics()


def test_fixture_kinematics_are_self_consistent():
    """The fixture's J must actually be dy/dq — otherwise it is not an oracle.

    Cheap to assert, and it keeps the noise-routing tests meaningful: a J that
    does not match y would make `J Σ_q Jᵀ` a plausible-looking fiction.
    """
    kinematics = _make_kinematics()
    q = jnp.asarray(np.random.default_rng(0).uniform(-0.5, 0.5, N_JOINTS))
    q_dot = jnp.zeros(N_JOINTS)

    frames = kinematics(q, q_dot)
    autodiff = jax.jacobian(lambda qq: kinematics(qq, q_dot).y)(q)   # (N,3,n)

    assert frames.J.shape == autodiff.shape
    assert np.max(np.abs(np.asarray(frames.J) - np.asarray(autodiff))) < 1.0e-12


# ---------------------------------------------------------------------------
# Noise routing — the joint-KF boundary (§4.2)
# ---------------------------------------------------------------------------

def test_contact_position_noise_is_J_sigma_Jt():
    rng = np.random.default_rng(1)
    J = jnp.asarray(rng.normal(size=(N_CONTACTS, 3, N_JOINTS)))
    A = rng.normal(size=(N_JOINTS, N_JOINTS))
    sigma_q = jnp.asarray(A @ A.T)

    Np = contact_position_noise(J, sigma_q)

    assert Np.shape == (N_CONTACTS, 3, 3)
    for i in range(N_CONTACTS):
        expected = np.asarray(J[i]) @ np.asarray(sigma_q) @ np.asarray(J[i]).T
        assert np.max(np.abs(np.asarray(Np[i]) - expected)) < 1.0e-12
        assert_symmetric_psd(Np[i], 1.0e-9)


def test_velocity_noise_is_routed_separately():
    """`Σ_q̇` goes through `J_Ċ`, never folded into `Σ_q` (design decision §4.2)."""
    rng = np.random.default_rng(2)
    J_dot = jnp.asarray(rng.normal(size=(N_CONTACTS, 3, N_JOINTS)))
    sigma_q_dot = jnp.eye(N_JOINTS) * 0.01

    Nv = contact_velocity_noise(J_dot, sigma_q_dot)

    assert Nv.shape == (N_CONTACTS, 3, 3)
    for i in range(N_CONTACTS):
        expected = np.asarray(J_dot[i]) @ np.asarray(sigma_q_dot) @ np.asarray(J_dot[i]).T
        assert np.max(np.abs(np.asarray(Nv[i]) - expected)) < 1.0e-12


def test_large_contact_covariance_isolates_a_swing_foot():
    """Contact condition rides in Σ_C alone — this is the measurement that decision rests on.

    See the DECISION note in `inEKF/filter.py`: there is no contact mask. A swing
    foot is expressed as a large Σ_C, which inflates P_dd so the FK residual is
    absorbed by the *anchor* rather than the base.

    If someone reintroduces a mask, or Σ_C stops reaching Q_d, this is the test
    that should fail. The thresholds are loose around the measured values
    (96% absorbed, 7.6x attenuation) so ordinary retuning does not trip it.
    """
    from invariant_estimation.inEKF.correct import (
        innovation, linear_update, measurement_noise,
    )
    from invariant_estimation.inEKF.propagate import propagate

    ekf = ekf_mod.create(N_CONTACTS, dt=DT)
    state = ekf_mod.initialize(
        ekf, jnp.eye(3), jnp.zeros(3), jnp.array([0.0, 0.0, 0.9]),
        jnp.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        jnp.eye(9 + 3 * N_CONTACTS) * 1.0e-2,
    )
    Np = jnp.tile(jnp.eye(3) * 1.0e-6, (N_CONTACTS, 1, 1))
    planted = jnp.tile(jnp.eye(3) * 1.0e-6, (N_CONTACTS, 1, 1))
    swinging = planted.at[1].set(jnp.eye(3) * 1.0)      # foot 1 in swing

    def evolve(sigma_c, ticks=100):
        st = state
        for _ in range(ticks):
            st = propagate(st, jnp.zeros(3), jnp.array([0.0, 0.0, 9.81]),
                           sigma_c, ekf.params)
        return st

    def apply_displaced_measurement(st):
        """Foot 1 has physically moved 8 cm; foot 0's measurement is consistent."""
        y = jax.vmap(lambda d: st.R.T @ (d - st.p))(st.d)
        y = y.at[1].add(jnp.array([0.08, 0.0, 0.0]))
        out, _ = linear_update(st, ekf.params.H, innovation(st, y), measurement_noise(Np))
        return out

    swing_state = evolve(swinging)
    planted_state = evolve(planted)

    # The swing foot's anchor covariance has grown; the planted one's has not.
    assert float(swing_state.P[12, 12]) > 10.0 * float(swing_state.P[9, 9])

    after_swing = apply_displaced_measurement(swing_state)
    after_planted = apply_displaced_measurement(planted_state)

    base_swing = float(jnp.linalg.norm(after_swing.p - swing_state.p))
    base_planted = float(jnp.linalg.norm(after_planted.p - planted_state.p))
    absorbed = float(jnp.linalg.norm(after_swing.d[1] - swing_state.d[1]))

    # Most of the 8 cm discrepancy is taken by the anchor, not the base.
    assert absorbed > 0.9 * 0.08
    # …and the base is far less perturbed than it would be for a planted foot.
    assert base_swing < 0.25 * base_planted


# ---------------------------------------------------------------------------
# Step / run mechanics
# ---------------------------------------------------------------------------

def test_step_shapes_and_psd():
    rng = np.random.default_rng(4)
    ekf, state, kinematics = _setup(rng)
    step = make_step(ekf, kinematics)

    carry, outputs = step(init_carry(state), _inputs(rng))

    m = 9 + 3 * N_CONTACTS
    assert carry.state.P.shape == (m, m)
    assert carry.state.d.shape == (N_CONTACTS, 3)
    assert outputs.contact_innovation.shape == (3 * N_CONTACTS,)
    assert_symmetric_psd(carry.state.P, 1.0e-9)
    assert np.isfinite(float(outputs.contact_diagnostics.nis))
    assert jnp.allclose(carry.state.R @ carry.state.R.T, jnp.eye(3), atol=1e-10)


def test_run_scans_a_trajectory():
    rng = np.random.default_rng(5)
    ekf, state, kinematics = _setup(rng)
    T = 50
    inputs = _stack([_inputs(rng) for _ in range(T)])

    carry, outputs = run(ekf, kinematics, state, inputs)

    m = 9 + 3 * N_CONTACTS
    assert outputs.state.P.shape == (T, m, m)
    assert outputs.contact_innovation.shape == (T, 3 * N_CONTACTS)
    assert outputs.tilt_angle.shape == (T,)
    assert np.all(np.isfinite(np.asarray(outputs.state.p)))
    assert_symmetric_psd(carry.state.P, 1.0e-9)
    # Covariance stays PSD at every tick, not just the last.
    for k in range(0, T, 10):
        assert_symmetric_psd(outputs.state.P[k], 1.0e-9)


def test_run_matches_manual_stepping():
    """`run` is exactly the scan of `step` — no hidden per-trajectory logic."""
    rng = np.random.default_rng(6)
    ekf, state, kinematics = _setup(rng)
    inputs_list = [_inputs(rng) for _ in range(5)]
    inputs = _stack(inputs_list)

    carry_scan, _ = run(ekf, kinematics, state, inputs)

    step = make_step(ekf, kinematics)
    carry = init_carry(state)
    for one in inputs_list:
        carry, _ = step(carry, one)

    assert jnp.allclose(carry_scan.state.as_matrix, carry.state.as_matrix, atol=1e-12)
    assert jnp.allclose(carry_scan.state.P, carry.state.P, atol=1e-12)


# ---------------------------------------------------------------------------
# I7 — the constant-graph proof
# ---------------------------------------------------------------------------

def _jaxpr_of(ekf, kinematics, state, inputs):
    step = make_step(ekf, kinematics)
    return jax.make_jaxpr(step)(init_carry(state), inputs)


def test_no_data_dependent_branch_on_contact_covariance():
    """I7: contact condition must not change the traced graph.

    Note what this does and does not prove. `contact_chol` is a *traced* array,
    so the jaxpr cannot depend on its value — equality here is close to
    automatic. What it genuinely catches is a data-dependent branch
    (`if sigma[i] > x`, `jnp.nonzero(...)`, boolean indexing), which raises at
    trace time rather than producing a different graph. That is the failure mode
    worth guarding, and it is why the assertion is phrased as "tracing succeeds
    and agrees" rather than as a strong claim about coverage.
    """
    rng = np.random.default_rng(7)
    ekf, state, kinematics = _setup(rng)

    # Firm, slipping (anisotropic), swinging, and mixed.
    cases = [
        jnp.tile(jnp.eye(3) * 1e-4, (N_CONTACTS, 1, 1)),
        jnp.tile(jnp.diag(jnp.array([1.0, 1.0, 1e-6])), (N_CONTACTS, 1, 1)),
        jnp.tile(jnp.eye(3) * 1.0, (N_CONTACTS, 1, 1)),
        jnp.stack([jnp.eye(3) * 1e-6, jnp.eye(3) * 10.0]),
    ]
    jaxprs = [
        str(_jaxpr_of(ekf, kinematics, state, _inputs(np.random.default_rng(7), c)))
        for c in cases
    ]

    for other in jaxprs[1:]:
        assert other == jaxprs[0], "contact covariance changed the traced graph — I7"


def test_jaxpr_is_identical_across_gate_states():
    """I7: an open vs closed quasi-static gate must trace identically."""
    rng = np.random.default_rng(8)
    ekf, state, kinematics = _setup(rng)

    quiet = _inputs(np.random.default_rng(8))                       # gate open
    shaken = _inputs(
        np.random.default_rng(8),
        accel=jnp.array([3.0, 0.0, 9.0]),                           # gate closed
        omega=jnp.array([0.0, 0.9, 0.0]),
    )

    assert str(_jaxpr_of(ekf, kinematics, state, quiet)) == \
        str(_jaxpr_of(ekf, kinematics, state, shaken))


def test_step_does_not_recompile_across_contact_conditions():
    """The operational form of I7: one trace, every contact condition.

    This is the port's analogue of the Java allocation tests — *no recompilation
    IS no per-tick allocation*. G9 reuses it on the fused estimator.
    """
    rng = np.random.default_rng(9)
    ekf, state, kinematics = _setup(rng)
    step = jax.jit(make_step(ekf, kinematics))

    carry = init_carry(state)
    for scale in (1e-6, 1.0, 1e3, 1e-2):
        chol = jnp.stack([jnp.eye(3) * scale, jnp.eye(3) * 1e-4])
        carry, _ = step(carry, _inputs(np.random.default_rng(9), chol))

    assert step._cache_size() == 1


def test_closed_gate_leaves_state_unchanged_by_the_gravity_update():
    """A closed quasi-static gate is exact: masked K, not skipped work (§4)."""
    rng = np.random.default_rng(10)
    ekf, state, kinematics = _setup(rng)
    step = make_step(ekf, kinematics)

    shaken = _inputs(
        np.random.default_rng(10),
        accel=jnp.array([3.0, 0.0, 9.0]),
        omega=jnp.array([0.0, 0.9, 0.0]),
    )
    _, outputs = step(init_carry(state), shaken)

    assert float(outputs.quasi_static) == 0.0
    assert float(outputs.gravity_diagnostics.applied) == 0.0


# ---------------------------------------------------------------------------
# Differentiability — BPTT into ContactNet
# ---------------------------------------------------------------------------

def test_scan_is_differentiable_through_contact_covariances():
    """BPTT must flow through the whole trajectory into the Cholesky factors.

    This is why the filter holds no trainable parameters: ContactNet trains by
    differentiating *through* a fixed, differentiable function.
    """
    rng = np.random.default_rng(11)
    ekf, state, kinematics = _setup(rng)
    inputs = _stack([_inputs(rng) for _ in range(10)])

    def loss(contact_chol):
        xs = inputs._replace(contact_chol=contact_chol)
        carry, _ = run(ekf, kinematics, state, xs)
        return jnp.sum(carry.state.p ** 2) + jnp.trace(carry.state.P)

    grad = jax.grad(loss)(inputs.contact_chol)

    assert grad.shape == inputs.contact_chol.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0      # not a dead path


def test_scan_is_differentiable_through_joint_covariance():
    """The joint-KF boundary is differentiable too — gradients reach `Σ_q`."""
    rng = np.random.default_rng(12)
    ekf, state, kinematics = _setup(rng)
    inputs = _stack([_inputs(rng) for _ in range(10)])

    def loss(sigma_q):
        xs = inputs._replace(
            joint=inputs.joint._replace(
                sigma_q=jnp.broadcast_to(sigma_q, inputs.joint.sigma_q.shape)
            )
        )
        carry, _ = run(ekf, kinematics, state, xs)
        return jnp.trace(carry.state.P)

    grad = jax.grad(loss)(jnp.eye(N_JOINTS) * 5.0e-5)

    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# Boundary contract (§6 forbidden edges)
# ---------------------------------------------------------------------------

def test_joint_outputs_do_not_reach_the_propagation():
    """Forbidden edge: nothing from the joint KF may touch Φ or the inertial Q.

    Structural check — changing `Σ_q` must leave the *predicted* covariance
    identical, since it only enters through the measurement noise.
    """
    rng = np.random.default_rng(13)
    ekf, state, kinematics = _setup(rng)

    from invariant_estimation.inEKF.propagate import propagate
    from invariant_estimation.inEKF.contact import digest

    inputs = _inputs(rng)
    sigma_c = digest(inputs.contact_chol, ekf.params)
    predicted = propagate(state, inputs.omega, inputs.accel, sigma_c, ekf.params)

    loud = inputs._replace(
        joint=inputs.joint._replace(sigma_q=jnp.eye(N_JOINTS) * 1.0e3)
    )
    sigma_c_loud = digest(loud.contact_chol, ekf.params)
    predicted_loud = propagate(state, loud.omega, loud.accel, sigma_c_loud, ekf.params)

    assert jnp.array_equal(predicted.P, predicted_loud.P)
    assert jnp.array_equal(predicted.as_matrix, predicted_loud.as_matrix)


def test_contact_measurement_noise_is_rotated_to_world():
    """`S` must use `R̂ (J Σ_q Jᵀ + Σ_C) R̂ᵀ`, not the body-frame noise.

    `innovation` returns a WORLD-frame residual (`R̄y − (d̄−p̄)`), while `Np` and
    `Σ_C` are body-frame, so the conjugation is required for `S = H P Hᵀ + N` to
    be frame-consistent. It was missing until 2026-07-28.

    This test exists because **no other test in the suite can see the bug**: every
    contact-update test uses isotropic noise, and `R̂(σ²I)R̂ᵀ = σ²I` exactly, so
    the buggy and correct paths agree to ~1e-21 there. 653 tests passed over it.
    The noise here is deliberately ANISOTROPIC -- diag(1e-3, 1e-3, 1e-8), the
    "slides along the surface but not through it" case that is ContactNet's whole
    justification (see filter.py's DECISION note).

    Checked on the POSTERIOR STATE, not on `logdet S`. `logdet S` is the obvious
    observable and it is useless here: with the fixture's isotropic `P = 0.1·I`,
    `H P Hᵀ` is `M ⊗ I₃`, which commutes with the per-contact rotation `I_N ⊗ R̂`,
    so `det(S)` is invariant under the conjugation and a test built on it passes
    against the bug. The correction itself is not invariant.

    The unconjugated form is kept as a NEGATIVE control -- a test asserting only
    the positive would also pass on a filter that conjugated twice, or by the
    wrong rotation.
    """
    rng = np.random.default_rng(11)
    ekf, state, kinematics = _setup(rng)
    assert jnp.linalg.norm(state.R - jnp.eye(3)) > 0.5, "need a non-identity R̂ to see it"

    slip = jnp.diag(jnp.array([1.0e-3, 1.0e-3, 1.0e-8]))
    # Large raw gyro closes the quasi-static gate, so the gravity update that
    # follows the contact update in `step` is masked out and leaves (X, P)
    # bit-unchanged. Without this the gravity correction dominates the base
    # position and the comparison below is noise -- an earlier version of this
    # test passed against a mutant with the conjugation removed for exactly that
    # reason.
    spin = jnp.array([0.0, 0.0, 5.0])
    inputs = _inputs(rng, omega=spin,
                     contact_meas_chol=jnp.tile(jnp.linalg.cholesky(slip),
                                                (N_CONTACTS, 1, 1)))

    step = make_step(ekf, kinematics)
    carry, out = step(init_carry(state), inputs)
    assert float(out.quasi_static) == 0.0, "gravity gate must be shut for this test"

    # Reproduce the contact update both ways and see which one `step` matches.
    prior = propagate(state, inputs.omega, inputs.accel,
                      digest(inputs.contact_chol, ekf.params), ekf.params)
    frames = kinematics(inputs.joint.q, inputs.joint.q_dot)
    body = (contact_position_noise(frames.J, inputs.joint.sigma_q)
            + reconstruct_cov(inputs.contact_meas_chol))
    world = jnp.einsum("ij,njk,lk->nil", prior.R, body, prior.R)
    nu = innovation(prior, frames.y)

    def posterior(N):
        st, _ = linear_update(prior, ekf.params.H, nu, measurement_noise(N))
        return np.asarray(st.p)

    p_world, p_body = posterior(world), posterior(body)
    sep = np.max(np.abs(p_world - p_body))
    assert sep > 1e-9, "the two paths are indistinguishable here -- degenerate test"

    got = np.asarray(carry.state.p)
    d_world = np.max(np.abs(got - p_world))
    d_body = np.max(np.abs(got - p_body))
    assert d_world < 1e-12, (
        f"step's posterior does not match the world-frame update ({d_world:.3e}); "
        f"separation between the two candidate paths is {sep:.3e}")
    assert d_body > 1e-9, (
        "step's posterior also matches the UNCONJUGATED update -- "
        "the frame conjugation is missing or the test is degenerate")
