r"""Is the learned ``Σ_C`` actually reaching the *process* model? — the wiring proof.

`test_contactnet_seam.py` proves the seam writes `InEKFInputs.contact_chol` and
that the filter's trajectory changes. That is necessary and not sufficient: it
would still pass if `Σ_C` were reaching the filter through some other route, or if
the training gradient were flowing through a path that no longer exists. These are
the two properties that make the socket move real rather than nominal.

**1. `Σ_C` lands in the contact block of ``Q_d``, rotated by ``R̂`` and nothing
else.** The anchor dynamics the port implements are

    ḋ_i = R̂ w_{C_i},        w_{C_i} ~ N(0, Σ_{C_i})

i.e. CoCo-InEKF Eq. (5) (``Wṗ_Ci = −WR_B · Bw_Ci``; the sign is irrelevant for a
zero-mean Gaussian). `continuous_Qc` places ``Σ_{C_i}`` at ``Q_c[9+3i:12+3i]`` in
the **body/contact** frame, and the contact diagonal block of ``Ad_X̂`` is ``R̂``,
so the adjoint conjugation in ``Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt`` (I3) performs the
rotation. Asserted numerically below, because "it's in the process noise" is the
kind of claim that reads as true from either side of a sign or a frame error.

Note this is ``R̂``, **not** a contact Jacobian ``ᴮJ_{C_i}``. The distinction is
physical, not cosmetic: ``R̂ w`` makes ``w`` a body-frame contact-point velocity —
slip at the foot/ground interface, which is what a stance/slip covariance means —
whereas ``J_{C_i} w_q`` would make ``w`` a *joint-rate* noise mapped through
kinematics, which is FK/encoder-rate error and belongs to the term this branch
moved ContactNet *away* from.

**2. The BPTT gradient reaches the network through ``Q_d``.** Training through the
process socket is a different differentiation path than training through ``N``:
``Σ_C`` now influences the loss only via the covariance propagation and hence via
the Kalman gain on *later* ticks, never via the residual on the current one. A
segment of length 1 therefore has (almost) no gradient, and a longer one must.
That asymmetry is the signature of a process-noise path, and it is what a
measurement-socket wiring would fail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect, must precede arrays)
from invariant_estimation.inEKF import ekf as ekf_mod
from invariant_estimation.inEKF.contact import digest
from invariant_estimation.inEKF.group import Adjoint
from invariant_estimation.inEKF.propagate import build_Qd, continuous_Qc

N_C = 2


def _state(rng, ekf):
    from tests.inEKF._oracles import next_rotation_matrix, next_vector3d
    return ekf_mod.initialize(
        ekf,
        jnp.asarray(next_rotation_matrix(rng)),
        jnp.asarray(next_vector3d(rng)[0]),
        jnp.asarray(next_vector3d(rng)[0]),
        jnp.asarray(np.stack([next_vector3d(rng)[0] for _ in range(N_C)])),
    )


# ---------------------------------------------------------------------------
# 1. Sigma_C -> the contact block of Q_d, rotated by R-hat
# ---------------------------------------------------------------------------

def test_sigma_c_enters_the_contact_block_of_Qc_in_the_body_frame():
    """`Q_c[9+3i:12+3i]` is the digested `Σ_{C_i}` verbatim — no frame change yet."""
    ekf = ekf_mod.create(N_C, dt=1.0e-3)
    chol = jnp.stack([jnp.diag(jnp.array([2.0e-2, 3.0e-2, 1.0e-3])),
                      jnp.eye(3) * 5.0e-1])
    sigma_c = digest(chol, ekf.params)
    Qc = np.asarray(continuous_Qc(sigma_c, ekf.params))

    for i in range(N_C):
        s = slice(9 + 3 * i, 12 + 3 * i)
        assert np.allclose(Qc[s, s], np.asarray(sigma_c[i]), rtol=0, atol=0)
    # The IMU blocks are untouched and the position block is zero (noise reaches
    # position only through the A-coupling, which Phi supplies).
    assert np.allclose(Qc[0:3, 0:3], ekf.params.gyro_var * np.eye(3))
    assert np.allclose(Qc[3:6, 3:6], ekf.params.accel_var * np.eye(3))
    assert not Qc[6:9, 6:9].any()
    # No leakage between contacts: the anchors are independent.
    assert not Qc[9:12, 12:15].any()


def test_anchor_process_noise_is_R_sigma_R_dt_not_a_jacobian():
    r"""``ḋ_i = R̂ w_{C_i}`` — CoCo Eq. (5), and the frame claim checked directly.

    Isolates the contact block by zeroing the IMU densities, so the only thing
    left in ``Q_d`` is what ``Σ_C`` put there.  With ``Φ``'s contact rows being
    identity (contacts are world-static in the nominal model), the contact block
    of ``Q_d`` must be exactly ``R̂ Σ_{C_i} R̂ᵀ Δt``.

    An **anisotropic** ``Σ_C`` is essential: for isotropic noise
    ``R̂(σ²I)R̂ᵀ = σ²I`` and the rotation is invisible, so an isotropic test would
    pass against a wrong frame, a missing conjugation, or a Jacobian in place of
    ``R̂``.  Anisotropy is also the whole point of a learned ``Σ_C`` ("slides along
    the surface but not through it").
    """
    ekf = ekf_mod.create(N_C, dt=1.0e-3, gyro_var=0.0, accel_var=0.0)
    dt = ekf.params.dt
    rng = np.random.default_rng(4)
    state = _state(rng, ekf)

    # Strongly anisotropic, and well above `contact_floor` so the floor does not
    # wash the anisotropy out.
    chol = jnp.stack([jnp.diag(jnp.array([1.0e-1, 5.0e-2, 1.0e-3])),
                      jnp.diag(jnp.array([2.0e-3, 7.0e-1, 4.0e-2]))])
    sigma_c = digest(chol, ekf.params)
    Qd = np.asarray(build_Qd(sigma_c, Adjoint(state.as_matrix), ekf.params))

    R = np.asarray(state.R)
    for i in range(N_C):
        s = slice(9 + 3 * i, 12 + 3 * i)
        want = R @ np.asarray(sigma_c[i]) @ R.T * dt
        assert np.allclose(Qd[s, s], want, rtol=1e-12, atol=1e-18), (
            f"contact {i}'s process block is not R Sigma_C R^T dt")

    # And the rotation is genuinely doing something: the same Sigma_C without the
    # conjugation would be a different matrix. This is what an isotropic Sigma_C
    # could not detect.
    unrotated = np.asarray(sigma_c[0]) * dt
    assert not np.allclose(Qd[9:12, 9:12], unrotated, atol=1e-12), (
        "R Sigma_C R^T == Sigma_C, so this fixture cannot tell a frame error from "
        "a correct one -- make Sigma_C more anisotropic or R less trivial")


def test_a_tighter_sigma_c_tightens_the_propagated_anchor_covariance():
    """Monotone, and in the direction the physics requires.

    The sign that matters: a *smaller* learned `Σ_C` must make the anchor's
    propagated covariance *smaller*, i.e. assert the foot is more firmly
    world-static. A sign inversion anywhere in the digest would leave every other
    test here passing.
    """
    ekf = ekf_mod.create(N_C, dt=1.0e-3, gyro_var=0.0, accel_var=0.0)
    rng = np.random.default_rng(5)
    Ad = Adjoint(_state(rng, ekf).as_matrix)

    def anchor_trace(scale):
        chol = jnp.stack([jnp.eye(3) * scale] * N_C)
        Qd = build_Qd(digest(chol, ekf.params), Ad, ekf.params)
        return float(jnp.trace(Qd[9:12, 9:12]))

    stance, swing = anchor_trace(1.0e-4), anchor_trace(1.0e1)
    assert stance < swing
    # The heuristic's own two values. The usable dynamic range is ~1e6, and it is
    # `contact_floor` that sets the bottom of it, not `sigma_0`: at the stance end
    # `Sigma_C = 1e-8` is swamped by the 1e-4 floor, so the ratio measures
    # `swing / floor` and saturates there. That is the safety bound working as
    # designed (see `contact.apply_floor`) -- and it means a network cannot express
    # a stance tighter than the floor no matter what it emits.
    assert 1.0e5 < swing / stance < 1.0e7
    assert np.isclose(stance, 3.0 * 1.0001e-4 * ekf.params.dt, rtol=1e-6), (
        "the stance end is not floor-dominated; contact_floor may have moved")


# ---------------------------------------------------------------------------
# 2. The BPTT gradient reaches the network through Q_d
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _seam():
    """A tiny end-to-end training seam: network -> contact_chol -> filter -> loss."""
    from invariant_estimation.contactnet import network
    from invariant_estimation.contactnet.config import ContactNetConfig
    from invariant_estimation.contactnet.rollout import Segment, make_segment_loss
    from tests.inEKF.test_filter import (DT, N_CONTACTS, N_JOINTS, _inputs,
                                         _make_kinematics, _stack)

    assert N_CONTACTS == N_C
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # short window aliases the torques
        cfg = ContactNetConfig(F=6, sigma_0=1.0e-2, H=3, window_span_s=0.003,
                               dt=DT, widths=(8,), L=24)
    ekf = ekf_mod.create(N_C, dt=DT)
    kin = _make_kinematics()

    def build(L: int):
        rng = np.random.default_rng(31)
        xs = _stack([_inputs(np.random.default_rng(31 + k)) for k in range(L)])
        state = _state(np.random.default_rng(2), ekf)
        # Non-degenerate windows: a zero window gives a zero trunk gradient for a
        # reason that has nothing to do with the socket.
        windows = jnp.asarray(rng.normal(size=(L, N_C, cfg.H, cfg.F)))
        return Segment(inputs=xs, windows=windows, state0=state,
                       v_true=jnp.zeros((L, 3)),
                       R_true=jnp.broadcast_to(jnp.eye(3), (L, 3, 3)))

    params = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    # Move the head off its zero init, or Sigma_C is a constant and the gradient
    # w.r.t. the trunk is exactly zero by construction (`check_init` asserts that
    # property deliberately; here it would mask the thing under test).
    params = params._replace(head=params.head._replace(
        W=params.head.W + 0.05, b=params.head.b + 0.02))

    loss_fn = make_segment_loss(ekf, kin, cfg.eps, objective="l2_velocity",
                               remat=False)
    return cfg, params, build, loss_fn


def test_gradient_reaches_the_network_through_the_process_socket(_seam):
    """Non-zero d(loss)/d(params) with `Σ_C` entering only via `Q_d`."""
    cfg, params, build, loss_fn = _seam
    grads = jax.grad(loss_fn, has_aux=True)(params, build(cfg.L))[0]
    for name, sub in (("head", grads.head), ("trunk", grads.trunk)):
        g = max(float(np.abs(np.asarray(x)).max()) for x in jax.tree.leaves(sub))
        assert np.isfinite(g), f"{name} gradient is not finite"
        assert g > 0.0, (
            f"{name} gradient is exactly zero -- Sigma_C is not reaching the loss "
            f"through Q_d")


def test_the_gradient_path_is_the_process_model_not_the_measurement(_seam):
    r"""The asymmetry that distinguishes the two sockets.

    On the **measurement** socket ``Σ_C`` enters ``S = H P Hᵀ + N`` on the *same*
    tick it is emitted, so even a one-tick segment has a full-strength gradient.
    On the **process** socket it enters ``Q_d``, so it can only affect the loss
    through the covariance carried into *later* ticks — a one-tick segment has
    essentially none, and the gradient must grow with the horizon.

    Asserted as a ratio rather than an absolute, and with a wide envelope: the
    point is the qualitative shape (≈0 at L=1, real at L=24), not a number.
    """
    cfg, params, build, loss_fn = _seam

    def gnorm(L):
        g = jax.grad(loss_fn, has_aux=True)(params, build(L))[0]
        return float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g))))

    short, long = gnorm(1), gnorm(cfg.L)
    assert long > 0.0
    assert short < 1.0e-3 * long, (
        f"a ONE-TICK segment already carries {short / long:.2e} of the "
        f"{cfg.L}-tick gradient. Sigma_C is influencing the residual on the tick "
        f"it is emitted, which is the MEASUREMENT socket's signature -- the "
        f"process socket can only act through the covariance it hands forward")


def test_the_measurement_socket_stays_zero_through_training(_seam):
    """Nothing in the training path writes `contact_meas_chol` any more."""
    cfg, params, build, loss_fn = _seam
    outputs, _ = jax.jit(loss_fn)(params, build(cfg.L))[1]
    del outputs
    seg = build(cfg.L)
    assert not np.asarray(seg.inputs.contact_meas_chol).any(), (
        "the fixture's measurement socket is non-zero, so this test is vacuous")
