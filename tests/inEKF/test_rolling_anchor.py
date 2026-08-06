r"""The rolling-anchor contact density ``κ(‖ω‖²I − ωωᵀ)``.

No Java analogue — this term does not exist in the reference filter. Every test
below is therefore an oracle against the *derivation*, not against a port:

    ḋ_i = ċ + ω × (d_i − c)      rigid-body kinematics
        = ω × r_i                no-slip contact, ċ = 0
    Cov(ḋ_i) = [ω]_× Cov(r_i) [ω]_×ᵀ = σ_r²(‖ω‖²I − ωωᵀ)

The Monte-Carlo test is the one that would actually catch an algebra error: it
samples ``r`` and forms the sample covariance of ``ω × r`` rather than
re-deriving the closed form, so it cannot agree with a wrong closed form.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from invariant_estimation.inEKF import ekf as ekf_mod
from invariant_estimation.inEKF.contact import (
    RollingAnchorParams,
    default_rolling_anchor_params,
    rolling_anchor_density,
)

PARAMS = RollingAnchorParams(enabled=True, tau=0.25, sigma_r=0.0985)
KAPPA = PARAMS.tau * PARAMS.sigma_r ** 2


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------

def test_matches_a_monte_carlo_over_the_unknown_lever_arm():
    r"""Sample ``r``, form ``Cov(ω × r)`` empirically, compare to the closed form.

    This is the load-bearing test: it never writes ``‖ω‖²I − ωωᵀ`` down, so an
    error in that identity (a sign, a transpose, ``[ω]_×²`` vs ``[ω]_×[ω]_×ᵀ``)
    shows up here and nowhere else.
    """
    rng = np.random.default_rng(11)
    omega = np.array([0.3, -1.7, 0.9])

    r = PARAMS.sigma_r * rng.standard_normal((400_000, 3))
    ddot = r @ skew(omega).T                                 # (ω × r)ᵢ per sample
    empirical = np.cov(ddot, rowvar=False)

    closed = np.asarray(rolling_anchor_density(jnp.asarray(omega)[None, :], PARAMS))[0]

    # `tau` scales the closed form but not the sample; compare the shape. The
    # tolerance is Monte-Carlo error, not modelling slack: at 4e5 draws the
    # off-diagonals converge ~1/sqrt(n), which is a few 1e-4 against a peak of
    # 3.6e-2. An algebra error would be O(1) here, not O(1e-3).
    np.testing.assert_allclose(closed / PARAMS.tau, empirical,
                               rtol=0.05, atol=1e-4 * np.trace(empirical))


def test_is_rank_two_and_null_along_the_rotation_axis():
    """A point rotating about an axis moves ⊥ to it and nowhere else."""
    omega = np.array([0.4, 1.1, -0.6])
    Sigma = np.asarray(rolling_anchor_density(jnp.asarray(omega)[None, :], PARAMS))[0]

    np.testing.assert_allclose(Sigma @ omega, np.zeros(3), atol=1e-12)

    w = np.linalg.eigvalsh(Sigma)
    np.testing.assert_allclose(w[0], 0.0, atol=1e-12)                  # the null one
    np.testing.assert_allclose(w[1:], KAPPA * omega @ omega, rtol=1e-12)  # x2, ⊥ plane


def test_is_exactly_zero_when_the_foot_is_not_rotating():
    """Flat stance must be bit-for-bit untouched — this is the self-gating claim."""
    Sigma = rolling_anchor_density(jnp.zeros((3, 3)), PARAMS)
    np.testing.assert_array_equal(np.asarray(Sigma), np.zeros((3, 3, 3)))


def test_is_frame_equivariant():
    r"""``R Σ(ω) Rᵀ = Σ(Rω)`` — so building it in body frame and letting ``Ad_X̂``
    rotate it (as `build_Qd` does) is correct, not an approximation."""
    rng = np.random.default_rng(5)
    omega = rng.standard_normal(3)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    R *= np.sign(np.linalg.det(R))

    lhs = R @ np.asarray(rolling_anchor_density(jnp.asarray(omega)[None], PARAMS))[0] @ R.T
    rhs = np.asarray(rolling_anchor_density(jnp.asarray(R @ omega)[None], PARAMS))[0]
    np.testing.assert_allclose(lhs, rhs, atol=1e-14)


def test_is_symmetric_psd_and_scales_quadratically():
    rng = np.random.default_rng(7)
    omega = rng.standard_normal((6, 3))
    Sigma = np.asarray(rolling_anchor_density(jnp.asarray(omega), PARAMS))

    for S in Sigma:
        np.testing.assert_allclose(S, S.T, atol=1e-15)
        assert np.linalg.eigvalsh(S).min() > -1e-14

    doubled = np.asarray(rolling_anchor_density(jnp.asarray(2 * omega), PARAMS))
    np.testing.assert_allclose(doubled, 4.0 * Sigma, rtol=1e-12)


def test_is_vectorised_over_contacts_independently():
    """Contact i's density depends on contact i's ω only."""
    rng = np.random.default_rng(13)
    omega = rng.standard_normal((4, 3))
    stacked = np.asarray(rolling_anchor_density(jnp.asarray(omega), PARAMS))
    for i in range(4):
        one = np.asarray(rolling_anchor_density(jnp.asarray(omega[i])[None], PARAMS))[0]
        np.testing.assert_allclose(stacked[i], one, atol=1e-15)


# ---------------------------------------------------------------------------
# Magnitude — the term has to be the size the measurement says it is
# ---------------------------------------------------------------------------

def test_magnitude_covers_the_measured_toe_off_motion():
    r"""At the measured toe-off rate the density must cover the measured motion.

    `experiments/anchor_static_check.py`: the N=2 sole-centre anchor rises
    17.7-28.8 mm over a ~0.15 s toe-off at ω ≈ 1.2-2.0 rad/s. For a random walk
    to cover ``σ_total`` over ``T`` needs density ``σ_total²/T``, i.e.
    2.1e-3 to 5.5e-3 m²/s. The shipped `contact_floor` is 1.0e-4 — 20-55x short,
    which IS the drift. This pins the fix to the right order of magnitude; if it
    fails, the term is cosmetic.
    """
    T = 0.15
    need_lo, need_hi = 0.0177 ** 2 / T, 0.0288 ** 2 / T

    for omega_mag in (1.2, 2.0):
        omega = jnp.array([[0.0, omega_mag, 0.0]])           # pitch about the y axis
        Sigma = np.asarray(rolling_anchor_density(omega, PARAMS))[0]
        vertical = Sigma[2, 2]        # the direction the anchor actually rises
        assert vertical > 1.0e-4, "must dominate the contact_floor it has to beat"

    # At the top of the measured band it should reach the requirement.
    Sigma = np.asarray(rolling_anchor_density(jnp.array([[0.0, 2.0, 0.0]]), PARAMS))[0]
    assert need_lo * 0.3 < Sigma[2, 2] < need_hi * 3.0, (
        f"vertical density {Sigma[2, 2]:.2e} outside the measured requirement "
        f"[{need_lo:.2e}, {need_hi:.2e}] by more than 3x")


def test_flat_stance_stays_below_the_contact_floor():
    """At a flat-stance rate the term must be negligible against the 1e-4 floor."""
    Sigma = np.asarray(rolling_anchor_density(jnp.array([[0.0, 0.05, 0.0]]), PARAMS))[0]
    assert Sigma.max() < 1.0e-4


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_off_by_default_with_the_shipped_config():
    """Recorded gate numbers were measured without it; the default must not move."""
    ekf = ekf_mod.create(2)
    assert ekf.rolling.enabled is False
    assert ekf.rolling.tau == 0.25
    assert ekf.rolling.sigma_r == 0.0985


@pytest.mark.parametrize("bad", [dict(tau=0.0), dict(tau=-1.0), dict(sigma_r=-0.1)])
def test_rejects_unphysical_parameters(bad):
    with pytest.raises(ValueError):
        default_rolling_anchor_params(enabled=True, **bad)
