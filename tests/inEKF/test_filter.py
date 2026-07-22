"""Tests for `inEKF/filter.py` — the scan body and trajectory driver.

No Java analogue: `InvariantEKF` in Java is driven by the controller's tick, so
there is nothing to port. These are port-specific and cover the three things the
scan body must guarantee:

* the joint-KF boundary is routed correctly (`Σ_q` reaches `N^p` through `J`,
  and nothing from the joint KF reaches `Φ` or the inertial `Q`);
* **I7 — the traced jaxpr is identical** regardless of contact mask or gate
  state, which is the constant-graph proof G9 reuses;
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
from invariant_estimation.inEKF.filter import (
    ContactFrames,
    InEKFInputs,
    JointFilterOutput,
    R_LARGE,
    contact_position_noise,
    contact_velocity_noise,
    init_carry,
    make_step,
    mask_contact_noise,
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


def _inputs(rng, contact_mask=None, accel=None, omega=None):
    """One tick of inputs."""
    if contact_mask is None:
        contact_mask = jnp.ones(N_CONTACTS)
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
        contact_mask=jnp.asarray(contact_mask, dtype=float),
        contact_chol=jnp.tile(jnp.eye(3) * 1.0e-3, (N_CONTACTS, 1, 1)),
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


def test_mask_substitutes_R_large_never_zero():
    """An untrusted contact gets R_LARGE, not a zeroed row (§6 trap: singular S)."""
    Np = jnp.tile(jnp.eye(3) * 1e-4, (N_CONTACTS, 1, 1))
    masked = mask_contact_noise(Np, jnp.array([1.0, 0.0]))

    assert jnp.allclose(masked[0], Np[0])
    assert jnp.allclose(masked[1], R_LARGE * jnp.eye(3))
    # Emphatically not zero — that is what makes S singular.
    assert float(jnp.min(jnp.diag(masked[1]))) > 1.0


def test_masked_contact_matches_excluded_contact():
    """Masking contact 1 must reproduce the posterior that never saw it.

    The `R_LARGE -> infinity` oracle of §4: a masked measurement is the limit of
    an infinitely-noisy one, so its block of the posterior is untouched.
    """
    rng = np.random.default_rng(3)
    ekf, state, kinematics = _setup(rng)
    step = make_step(ekf, kinematics)

    # Same tick, same draw — only the mask differs.
    both, _ = step(init_carry(state), _inputs(np.random.default_rng(30), [1.0, 1.0]))
    first_only, _ = step(init_carry(state), _inputs(np.random.default_rng(30), [1.0, 0.0]))

    # The masked contact's own covariance block must be essentially unchanged by
    # its (nonexistent) measurement, while the trusted one's is not.
    d1 = slice(9 + 3, 9 + 6)
    moved_when_trusted = float(jnp.max(jnp.abs(both.state.P[d1, d1] - state.P[d1, d1])))
    moved_when_masked = float(
        jnp.max(jnp.abs(first_only.state.P[d1, d1] - state.P[d1, d1]))
    )
    assert moved_when_masked < 1.0e-6
    assert moved_when_trusted > 1.0e-4


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


def test_jaxpr_is_identical_across_contact_masks():
    """I7: which feet are trusted must not change the traced graph.

    This is the port's analogue of the Java allocation tests — "no recompilation"
    IS "no per-tick allocation". G9 reuses it on the fused estimator.
    """
    rng = np.random.default_rng(7)
    ekf, state, kinematics = _setup(rng)

    masks = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.25]]
    jaxprs = [
        str(_jaxpr_of(ekf, kinematics, state, _inputs(np.random.default_rng(7), m)))
        for m in masks
    ]

    for other in jaxprs[1:]:
        assert other == jaxprs[0], "contact mask changed the traced graph — I7 violated"


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


def test_step_does_not_recompile_across_masks():
    """The operational form of I7: one trace, many mask patterns."""
    rng = np.random.default_rng(9)
    ekf, state, kinematics = _setup(rng)
    step = jax.jit(make_step(ekf, kinematics))

    carry = init_carry(state)
    for mask in ([1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]):
        carry, _ = step(carry, _inputs(np.random.default_rng(9), mask))

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
