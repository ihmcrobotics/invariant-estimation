r"""Port of `JointLevelKFUpdateTest.java` (6 tests) — gate G6 — plus the two gate
regressions that constrain `joseph_update`'s masking
(`JointLevelKFTrajectoryTest.testSingularInnovationIsSkippedNotLatched` and the
NaN-hardening half of `testTransientNonFiniteInputRecovers`).

The Java class checks the update against an **explicit-inverse reference KF**
(`_oracles.reference_update`): the filter solves through a Cholesky factorisation,
so an independent inversion path is what makes the reference an oracle rather than
a restatement.  ``H`` is `generic_h` — a `sin`-fill with no kinematic structure —
precisely so no geometric accident can make a wrong implementation look right.

The gate tests are the reason the two regressions live here rather than waiting
for the trajectory class: a masked gain that "nearly" leaves the state alone
passes every algebraic test above and still ruins a 20 k-tick run.  Both demand
tolerance **exactly 0.0**.
"""
import jax
import numpy as np
import pytest

from invariant_estimation.jointKF.predict import build_transition, predict
from invariant_estimation.jointKF.state import JointKFState, default_params, init_state
from invariant_estimation.jointKF.update import joseph_update

from ._oracles import (
    SHAPES,
    assert_all_close,
    assert_all_finite,
    assert_positive_semidefinite,
    assert_symmetric,
    generic_h,
    nis_quadratic_form,
    reference_update,
    seeded_prior_update,
    shape_dims,
    spd,
    stub_build,
)
from .test_predict import process_noise
from .test_state import single_pair

PARAMS = default_params()


def _prior(shape, seed):
    """Java `seededPrior(f, seed)`: mean `0.1*(i+1)`, covariance `spd(dim, seed)`."""
    _, _, dim = shape_dims(shape)
    x, P = seeded_prior_update(dim, seed)
    return JointKFState(x=x, P=P)


def _rotation(angle: float, axis: int) -> np.ndarray:
    """Elementary rotation — a stand-in for a real inter-IMU frame rotation."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


# ---------------------------------------------------------------------------
# Against the explicit-inverse reference KF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_shapes_and_reference_kf(shape):
    """`testShapesAndReferenceKF`: `(x⁺, P⁺)` match the reference KF, tol 1e-6."""
    _, _, dim = shape_dims(shape)
    prior = _prior(shape, 1)
    H = generic_h(3, dim, 2)
    z = 0.05 * np.arange(1, 4)
    R = spd(3, 7)
    ref_x, ref_P = reference_update(prior.x, prior.P, H, z, R)

    post, info = joseph_update(prior, H, z, R, PARAMS)
    assert post.x.shape == (dim,)
    assert post.P.shape == (dim, dim)
    assert_all_close(post.x, ref_x, 1.0e-6, "x posterior")
    assert_all_close(post.P, ref_P, 1.0e-6, "P posterior")
    assert float(info.was_applied) == 1.0


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_covariance_symmetric_psd(shape):
    """`testCovarianceSymmetricPSD`."""
    _, _, dim = shape_dims(shape)
    post, _ = joseph_update(_prior(shape, 3), generic_h(3, dim, 4), np.zeros(3), spd(3, 9), PARAMS)
    assert_symmetric(post.P, 1.0e-6, "P posterior")
    assert_positive_semidefinite(post.P, "P posterior")


@pytest.mark.parametrize("shape", SHAPES, ids=[s["name"] for s in SHAPES])
def test_covariance_shrinks(shape):
    """`testCovarianceShrinks`: `prior − post` is PSD and the trace does not grow.

    A measurement cannot increase uncertainty.  Under the Joseph form this holds
    for the *optimal* gain, which is what is used here; it is the sharpest cheap
    check that the gain and the covariance update are consistent with each other
    (a mismatched pair typically breaks the PSD ordering long before it breaks
    the trace).
    """
    _, _, dim = shape_dims(shape)
    prior = _prior(shape, 5)
    post, _ = joseph_update(prior, generic_h(3, dim, 6), np.zeros(3), spd(3, 11), PARAMS)
    diff = np.asarray(prior.P) - np.asarray(post.P)
    assert_positive_semidefinite(diff, "prior - posterior")
    assert np.trace(np.asarray(post.P)) <= np.trace(np.asarray(prior.P)) + 1.0e-6


def test_encoder_pull_tiny_r():
    """`testEncoderPullTinyR`: an all-but-noiseless encoder pulls the posterior onto it."""
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _prior(shape, 13)
    target = 0.3 + float(np.asarray(prior.x)[0])
    H = np.zeros((1, dim))
    H[0, 0] = 1.0
    post, _ = joseph_update(prior, H, np.array([target]), np.array([[1.0e-16]]), PARAMS)
    assert abs(float(np.asarray(post.x)[0]) - target) <= 1.0e-3


def test_bias_observability():
    """`testBiasObservability`: a nonzero gyro innovation must move the per-IMU bias.

    **Adapted** (`CONTRACT_CARD.md` §7): Java reads the stacked measurement off
    the filter (`buildStackedMeasurementForTest`), which lives in `measure.py`
    and is another agent's module.  The structure that matters for *this*
    assertion is reproduced directly instead — the pair row block is
    ``[0 | J q̇-columns | −R_parent at the parent bias | +I₃ at the child bias]``
    with ``z = ω_child − R ω_parent`` — because what is under test here is that
    the update *couples the residual into the bias states at all*, not the
    provenance of the rows.  A layout that put bias somewhere else, or an ``H``
    whose bias columns were dropped, fails exactly as in Java.
    """
    shape = single_pair(8, 1, 7)                      # n=5, m=2
    n, _, dim = shape_dims(shape)
    build = stub_build(shape)
    parent_col, child_col = build.pair_parent_bias_col(0), build.pair_child_bias_col(0)

    rot = _rotation(0.23, 1)                           # parent frame -> child frame
    omega_parent = np.array([0.05, -0.03, 0.02])
    omega_child = np.array([-0.04, 0.06, -0.01])

    H = np.zeros((3, dim))
    H[:, n:2 * n] = 0.1 * generic_h(3, n, 31)          # some q̇ sensitivity
    H[:, parent_col:parent_col + 3] = -rot
    H[:, child_col:child_col + 3] = np.eye(3)
    z = omega_child - rot @ omega_parent
    R = rot @ (1.0e-4 * np.eye(3)) @ rot.T + 1.0e-4 * np.eye(3)

    prior = init_state(build, PARAMS)
    post, info = joseph_update(prior, H, z, R, PARAMS)

    delta = np.abs(np.asarray(post.x) - np.asarray(prior.x))
    moved = delta[parent_col:parent_col + 3].sum() + delta[child_col:child_col + 3].sum()
    assert float(info.was_applied) == 1.0
    assert moved > 1.0e-9


def test_deterministic():
    """`testDeterministic`: two updates from the same carry agree bit-for-bit (tol 0.0)."""
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    state = JointKFState(x=0.02 * np.arange(1, dim + 1), P=spd(dim, 21))
    H, z, R = generic_h(3, dim, 22), np.zeros(3), spd(3, 23)
    a, _ = joseph_update(state, H, z, R, PARAMS)
    b, _ = joseph_update(state, H, z, R, PARAMS)
    assert_all_close(a.x, b.x, 0.0, "x determinism")
    assert_all_close(a.P, b.P, 0.0, "P determinism")


# ---------------------------------------------------------------------------
# The Joseph form itself — only constrainable at a SUBOPTIMAL gain
# ---------------------------------------------------------------------------

def test_joseph_form_is_correct_at_a_suboptimal_gain():
    """The covariance update is the honest `LΣLᵀ` pushforward for *any* gain.

    Port-specific, and it is the only test in this file that constrains the
    Joseph form at all: at the optimal ``K`` the short form ``(I−KH)P`` is
    algebraically identical, so every test driven through `joseph_update` —
    including the Java reference-KF comparison — passes against the short form
    too (`JOINTKF_PORT_PLAN` §4 lesson 1: exercising a term is not constraining
    it).

    Here ``K`` is deliberately wrong by a factor 0.4, the posterior error is
    ``e⁺ = (I−KH)e⁻ − Kv`` with ``e⁻ ⟂ v``, and its covariance is recomputed
    independently.  The final assertion records the discrimination margin: the
    short form is off by ~1e-2, seven orders above the tolerance, so this test
    cannot be satisfied by accident.
    """
    from invariant_estimation.jointKF.update import joseph_covariance

    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    P = spd(dim, 3)
    H = generic_h(3, dim, 4)
    R = spd(3, 5)

    K_opt = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    K = 0.4 * K_opt                                    # deliberately suboptimal

    IKH = np.eye(dim) - K @ H
    expected = IKH @ P @ IKH.T + K @ R @ K.T           # Cov[(I−KH)e⁻ − Kv]
    actual = np.asarray(joseph_covariance(P, K, H, R))
    assert_all_close(actual, expected, 1.0e-9, "Joseph covariance at suboptimal K")
    assert_positive_semidefinite(actual, "Joseph covariance at suboptimal K")

    short_form = IKH @ P
    assert np.max(np.abs(actual - 0.5 * (short_form + short_form.T))) > 1.0e-6


# ---------------------------------------------------------------------------
# Diagnostics (seam surface — CLAUDE.md §4)
# ---------------------------------------------------------------------------

def test_nis_is_the_prior_quadratic_form():
    """NIS is `νᵀ S⁻¹ ν` on the **prior** `P` and the **prior** residual.

    CLAUDE.md §6 names computing it on the posterior as a trap: the posterior
    residual is smaller by construction, so the statistic looks healthy exactly
    when the filter is over-trusting the measurement.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _prior(shape, 17)
    H, z, R = generic_h(3, dim, 18), 0.05 * np.arange(1, 4), spd(3, 19)
    _, info = joseph_update(prior, H, z, R, PARAMS)

    nu = z - H @ np.asarray(prior.x)
    S = H @ np.asarray(prior.P) @ H.T + R
    assert_all_close(info.nu, nu, 1.0e-12, "innovation")
    assert_all_close(info.S, S, 1.0e-9, "innovation covariance")
    assert abs(float(info.nis) - nis_quadratic_form(nu, S)) <= 1.0e-9


# ---------------------------------------------------------------------------
# Gates — skip, never latch
# ---------------------------------------------------------------------------

def _predicted_prior(shape):
    """A finite PD prior: `initialize()` then one `predict()`, as the Java test does."""
    build = stub_build(shape)
    state = init_state(build, PARAMS)
    return predict(state, build_transition(build, PARAMS), process_noise(shape))


def test_singular_innovation_is_skipped_not_latched():
    """`testSingularInnovationIsSkippedNotLatched`: rank-deficient `H`, `R = 0`.

    ``S = H P Hᵀ`` is then exactly singular.  The update must be skipped and
    ``(x, P)`` must come back **bit-identical** (tolerance 0.0): "nearly
    unchanged" is not good enough, because the alternative to skipping is a gain
    of order ``1/λ_min(S)`` whose ``K R Kᵀ`` term squares itself every tick.
    """
    shape = single_pair(8, 1, 7)
    n, _, dim = shape_dims(shape)
    prior = _predicted_prior(shape)

    H = np.zeros((3, dim))
    H[0, n] = 1.0
    H[1, n] = 1.0                                     # duplicate row => rank deficient
    z = np.zeros(3)
    R = np.zeros((3, 3))                              # no regularisation at all

    post, info = joseph_update(prior, H, z, R, PARAMS)
    assert_all_finite(post.x, "x after singular update")
    assert_all_finite(post.P, "P after singular update")
    assert_all_close(post.x, prior.x, 0.0, "x unchanged")
    assert_all_close(post.P, prior.P, 0.0, "P unchanged")
    assert float(info.was_applied) == 0.0
    assert np.isnan(float(info.nis))


def test_gated_update_is_bit_identical_on_a_non_symmetric_carry():
    """The skip returns the carry itself, not a re-derivation of it.

    With ``K = 0`` the Joseph form *algebraically* reduces to ``P``, but its
    floating-point evaluation only returns ``P`` bit-for-bit if ``P`` is already
    exactly symmetric (the update symmetrises).  A carry carrying a 1e-17
    asymmetry — which any external `setStateForTest`-style seam can produce —
    would come back changed, and "changed by 1e-17 per skipped tick" is a slow
    latch rather than a skip.  So the gated branch selects the *original*
    ``(x, P)``.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    P = spd(dim, 29)
    P[0, 1] = np.nextafter(P[0, 1], np.inf)            # break exact symmetry
    prior = JointKFState(x=0.02 * np.arange(1, dim + 1), P=P)

    H = np.zeros((3, dim))
    H[0, 0] = H[1, 0] = 1.0
    post, info = joseph_update(prior, H, np.zeros(3), np.zeros((3, 3)), PARAMS)
    assert float(info.was_applied) == 0.0
    assert_all_close(post.P, prior.P, 0.0, "P unchanged on a non-symmetric carry")
    assert_all_close(post.x, prior.x, 0.0, "x unchanged on a non-symmetric carry")


def test_well_conditioned_update_is_applied():
    """Companion to the gate test: the gate does not fire on a healthy measurement.

    Without this, a gate hard-wired to 0 would pass the skip test and silently
    disable every update in the filter.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _predicted_prior(shape)
    H = generic_h(3, dim, 41)
    post, info = joseph_update(prior, H, 0.05 * np.arange(1, 4), spd(3, 43), PARAMS)
    assert float(info.was_applied) == 1.0
    assert float(info.condition_proxy) < PARAMS.cond_s_max
    assert np.max(np.abs(np.asarray(post.x) - np.asarray(prior.x))) > 1.0e-9


@pytest.mark.parametrize("poison", ["H", "z", "R"])
def test_non_finite_measurement_is_skipped(poison):
    """NaN hardening: a non-finite `H`, `z` or `R` leaves `(x, P)` bit-identical.

    The trap this guards (`update.py` docstring): masking the gain alone is not
    enough, because ``0.0 * NaN = NaN`` — the inputs must be sanitised *before*
    they touch `P`.  One NaN reaching the covariance is permanent.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _predicted_prior(shape)
    H, z, R = generic_h(3, dim, 51), 0.05 * np.arange(1, 4), spd(3, 53)
    if poison == "H":
        H = H.copy()
        H[1, 2] = np.nan
    elif poison == "z":
        z = z.copy()
        z[0] = np.nan
    else:
        R = R.copy()
        R[0, 0] = np.inf

    post, info = joseph_update(prior, H, z, R, PARAMS)
    assert_all_finite(post.x, "x after poisoned update")
    assert_all_finite(post.P, "P after poisoned update")
    assert_all_close(post.x, prior.x, 0.0, "x unchanged")
    assert_all_close(post.P, prior.P, 0.0, "P unchanged")
    assert float(info.was_applied) == 0.0


def test_non_finite_input_does_not_latch():
    """Recovery is automatic: a clean update right after a poisoned one applies.

    `testTransientNonFiniteInputRecovers` restores good sensors and expects the
    filter to track again with no intervention, so the skip must be stateless.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    state = _predicted_prior(shape)
    H, z, R = generic_h(3, dim, 61), 0.05 * np.arange(1, 4), spd(3, 63)

    bad = z.copy()
    bad[2] = np.nan
    state, info_bad = joseph_update(state, H, bad, R, PARAMS)
    state, info_good = joseph_update(state, H, z, R, PARAMS)

    assert float(info_bad.was_applied) == 0.0
    assert float(info_good.was_applied) == 1.0
    assert_all_finite(state.P, "P after recovery")


def test_jit_graph_is_constant_across_gate_states():
    """I7: the jitted update compiles **once** for gated and ungated measurements.

    Same jaxpr, same executable — the gate is a `jnp.where` mask, not a Python
    branch.  Per `JOINTKF_PORT_PLAN` §4 lesson 3, what this really proves is the
    absence of a data-dependent branch (which would raise a `TracerBoolConversion`
    error), not that the arithmetic is right.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _predicted_prior(shape)
    fn = jax.jit(joseph_update, static_argnums=())

    H_ok, z_ok, R_ok = generic_h(3, dim, 71), 0.05 * np.arange(1, 4), spd(3, 73)
    H_bad = np.zeros((3, dim))
    H_bad[0, 0] = H_bad[1, 0] = 1.0

    _, info_good = fn(prior, H_ok, z_ok, R_ok, PARAMS)
    gated, info_gated = fn(prior, H_bad, np.zeros(3), np.zeros((3, 3)), PARAMS)

    assert float(info_good.was_applied) == 1.0
    assert float(info_gated.was_applied) == 0.0
    assert_all_close(gated.x, prior.x, 0.0, "gated x under jit")
    assert_all_close(gated.P, prior.P, 0.0, "gated P under jit")

    a = jax.make_jaxpr(joseph_update)(prior, H_ok, z_ok, R_ok, PARAMS)
    b = jax.make_jaxpr(joseph_update)(prior, H_bad, np.zeros(3), np.zeros((3, 3)), PARAMS)
    assert str(a) == str(b)


@pytest.mark.parametrize("poison", ["H", "z", "R"])
def test_gradients_are_finite_through_a_non_finite_measurement(poison):
    """A non-finite *sensor sample* must not poison the gradient w.r.t. the carry.

    This is what makes the input sanitisation load-bearing rather than
    decorative: selecting the prior carry at the end already keeps the returned
    ``(x, P)`` finite, so only the reverse pass can tell whether ``H, z, R`` were
    sanitised **before** entering the Cholesky.  A bad ``H`` or ``R`` puts the
    non-finite value inside ``S``, so ``K`` itself is NaN and back-propagates NaN
    through the discarded branch even though its cotangent is zero.  ContactNet
    trains through this update over long rollouts, where one bad tick would
    otherwise NaN the whole trajectory's gradient.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _predicted_prior(shape)
    H, z, R = generic_h(3, dim, 81), np.array([0.05, 0.10, 0.15]), spd(3, 83)
    if poison == "H":
        H = H.copy()
        H[2, 3] = np.nan
    elif poison == "z":
        z = z.copy()
        z[1] = np.nan
    else:
        R = R.copy()
        R[1, 1] = np.inf

    def loss(x):
        post, _ = joseph_update(prior._replace(x=x), H, z, R, PARAMS)
        return post.x.sum() + post.P.sum()

    g = jax.grad(loss)(prior.x)
    assert_all_finite(g, "gradient with a NaN measurement")


def test_gradients_are_finite_through_a_gated_update():
    """BPTT must survive the gate: no NaN gradient from the skipped branch.

    `jnp.where` on a NaN-carrying branch still back-propagates NaN, which is why
    the gain is *sanitised* and not merely selected.  ContactNet trains through
    this update, so a NaN here would be silent and fatal.
    """
    shape = single_pair(8, 1, 7)
    _, _, dim = shape_dims(shape)
    prior = _predicted_prior(shape)
    H_bad = np.zeros((3, dim))
    H_bad[0, 0] = H_bad[1, 0] = 1.0

    def loss(z):
        post, _ = joseph_update(prior, H_bad, z, np.zeros((3, 3)), PARAMS)
        return post.x.sum() + post.P.sum()

    g = jax.grad(loss)(np.zeros(3))
    assert_all_finite(g, "gradient through a gated update")
