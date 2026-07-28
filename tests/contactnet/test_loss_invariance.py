r"""`losses.l2_velocity` under the InEKF's unobservable direction.

The property this file exists for is the one that makes a long training episode
safe.  CoCo-InEKF (arXiv 2605.15122, §IV-A) justifies its ``T = 100 s`` episodes
by asserting that body-frame velocity is

    "an observable quantity of the InEKF ... also not affected by the InEKF's
     drifting behavior during an episode."

That is exact rather than approximate, and the reason is group structure.  The
right-invariant InEKF on ``SE_{N+2}(3)`` has global yaw as its one unobservable
direction, and it is unobservable as a **left action on the whole state**: an
accumulated yaw error rotates ``R``, ``v``, ``p`` and every contact anchor
together, ``X -> exp(psi e_z^) X``.  Under that action

    R_hat^T v_hat  ->  (R_z R)^T (R_z v)  =  R^T R_z^T R_z v  =  R^T v

so the body-frame velocity — and hence `l2_velocity` — is **invariant**.  Yaw
drift, however large, contributes exactly zero loss, and `Sigma_C` is never asked
to correct something it has no lever over.

This is a different perturbation from the one theory doc §7.1 Eq. 23 analyses.
That one holds ``v`` fixed and rotates only ``R``, giving
``L ~ ||delta_perp||^2 ||v||^2`` — sensitivity to attitude error perpendicular to
the body velocity.  Both statements are true; only the whole-state action is the
InEKF's unobservable direction, and only it bears on episode length.

If these tests ever fail, `ContactNetConfig.episode_s` is load-bearing in a way
it is not designed to be, and long episodes will silently inject an
unfixable-by-`Sigma_C` term into the objective.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect)
from invariant_estimation.contactnet.losses import l2_velocity


def _rot_z(psi: float) -> jnp.ndarray:
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)


def _traj(seed: int, L: int = 64):
    """A trajectory with a nonzero, non-degenerate attitude and velocity."""
    r = np.random.default_rng(seed)
    ang = np.cumsum(r.standard_normal((L, 3)) * 0.02, axis=0)

    def rpy(a):
        cr, sr, cp, sp, cy, sy = (np.cos(a[0]), np.sin(a[0]), np.cos(a[1]),
                                  np.sin(a[1]), np.cos(a[2]), np.sin(a[2]))
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
        Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    R = jnp.asarray(np.stack([rpy(a) for a in ang]))
    v = jnp.asarray(r.standard_normal((L, 3)) * 0.4 + np.array([0.4, 0.0, 0.0]))
    return R, v


@pytest.mark.parametrize("psi", [1e-3, 0.1, 1.0, 3.0, -2.5])
def test_l2_velocity_is_invariant_to_global_yaw_on_the_whole_state(psi):
    r"""Rotating ``(R_hat, v_hat)`` together by any yaw leaves the loss unchanged.

    This is the property CoCo-InEKF's ``T = 100 s`` rests on, and the reason our
    ``episode_s`` can be raised without poisoning the objective.

    Kills: a world-frame loss (``||v_hat - v||^2``), which is *not* invariant and
    would grow without bound as yaw drifts; and any formulation that rotates the
    two sides by a shared matrix, which `l2_velocity`'s own docstring warns is a
    no-op in the other direction.
    """
    R_true, v_true = _traj(seed=0)
    R_est, v_est = _traj(seed=1)

    base = l2_velocity(v_est, R_est, v_true, R_true)

    Rz = _rot_z(psi)
    yawed = l2_velocity(v_est @ Rz.T, Rz @ R_est, v_true, R_true)

    assert float(jnp.abs(yawed - base)) < 1e-12 * max(1.0, float(base)), (
        f"yaw {psi} changed the loss by {float(yawed - base):.3e}")

    # Non-vacuity: the WORLD-frame form of the same comparison is not invariant,
    # so the equality above is a property of the body-frame formulation and not
    # of the trajectory being degenerate.
    w_base = float(jnp.mean(jnp.sum((v_est - v_true) ** 2, axis=-1)))
    w_yaw = float(jnp.mean(jnp.sum((v_est @ Rz.T - v_true) ** 2, axis=-1)))
    if abs(psi) > 1e-2:
        assert abs(w_yaw - w_base) > 1e-6, "world-frame control is degenerate here"


def test_yaw_drift_growing_over_an_episode_adds_no_loss():
    r"""A yaw error ramping to 0.5 rad over the episode still costs exactly zero.

    The realistic failure this guards: over a long ``episode_s`` the InEKF's
    unobservable yaw walks away, and if the loss saw it, that term would grow
    without bound and swamp the contact signal ``Sigma_C`` actually controls.
    """
    R_true, v_true = _traj(seed=2, L=128)
    R_est, v_est = _traj(seed=3, L=128)
    base = l2_velocity(v_est, R_est, v_true, R_true)

    psi = jnp.linspace(0.0, 0.5, R_est.shape[0])
    Rz = jax.vmap(_rot_z)(psi)
    drifted = l2_velocity(jnp.einsum("kij,kj->ki", Rz, v_est),
                          jnp.einsum("kij,kjl->kil", Rz, R_est),
                          v_true, R_true)

    assert float(jnp.abs(drifted - base)) < 1e-12 * max(1.0, float(base))


def test_attitude_error_alone_does_reach_the_loss():
    r"""Rotating ``R_hat`` WITHOUT ``v_hat`` must change the loss (doc §7.1 Eq. 23).

    The counterpart to the invariance above, and what stops these tests from
    passing against a loss that ignores attitude entirely: a real attitude error
    — one not along the group's unobservable direction — has to be visible, or
    `l2_velocity` would be the world-frame form in disguise.
    """
    R_true, v_true = _traj(seed=4)
    base = l2_velocity(v_true, R_true, v_true, R_true)
    assert float(base) < 1e-24, "identical estimates should score ~0"

    # Perturb attitude only, about an axis perpendicular to the mean body
    # velocity, which Eq. 23 predicts is the sensitive direction.
    tilt = _rot_z(0.0).at[1, 1].set(jnp.cos(0.05)).at[1, 2].set(-jnp.sin(0.05))
    tilt = tilt.at[2, 1].set(jnp.sin(0.05)).at[2, 2].set(jnp.cos(0.05))
    perturbed = l2_velocity(v_true, jnp.einsum("ij,kjl->kil", tilt, R_true),
                            v_true, R_true)

    u = jnp.mean(jnp.einsum("kji,kj->ki", R_true, v_true), axis=0)
    predicted = 0.05 ** 2 * float(jnp.sum(u[:2] ** 2))   # ||delta_perp||^2 ||v||^2
    assert float(perturbed) > 1e-6, "attitude-only error vanished from the loss"
    assert 0.2 < float(perturbed) / predicted < 5.0, (
        f"loss {float(perturbed):.3e} is far from Eq. 23's {predicted:.3e}")
