r"""
inEKF/contact.py
================
Contact-covariance **digest** for the world-centric, right-invariant InEKF
(CLAUDE.md §5).  This module is a pure *consumer*: it takes the lower-triangular
Cholesky factors ``L_{C_i}`` emitted by the external ContactNet module and turns
them into the world-frame contact noise densities ``R̄ Σ_{C_i} R̄ᵀ`` that
`propagate.build_Qd` injects into the ``Q̄_d`` contact blocks (§3.3).

> **It contains no learned components and does not compute ``Σ_{C_i}``.**
> ContactNet (the MLP, its features, weights and the SPD-safe parameterisation
> that *produces* a valid Cholesky factor with positive diagonal) is a separate
> module, out of scope here (invariant 4).  This file imports nothing from it; it
> only consumes its output — the lower-triangular Cholesky argument, nothing more.

The digest is three pure, branch-free, vectorised-over-contacts steps:

    1. reconstruct   Σ_{C_i} = L_{C_i} L_{C_i}ᵀ          (`reconstruct_cov`)
    2. noise floor   Σ_{C_i} ← Σ_{C_i} + floor · I       (`apply_floor`)
    3. rotate        Σ^W_{C_i} = R̄ Σ_{C_i} R̄ᵀ            (`rotate_to_world`)

`digest` composes the three and is the single entry point the filter calls.

Contact condition is expressed *only* through the magnitude / anisotropy of
``Σ_{C_i}`` — small (firm) / anisotropic (slip) / large (no contact).  There is
**no add/remove**: all ``N`` candidates live in the state permanently (CoCo), so
the computation graph is constant — jit/scan/BPTT-safe (§1.1, invariant 5).

Everything here is ``jax.jit``-able and differentiable end-to-end (§8): BPTT
during training flows *through* this digest into ContactNet, so the map
``L → Σ^W`` must be smooth.  The additive floor keeps it smooth and PSD even when
a factor is (near-)singular, where an eigenvalue clamp would not be.
"""
from jax import Array
import jax.numpy as jnp

from .state import InEKFParams


# ---------------------------------------------------------------------------
# Digest steps (§5)
# ---------------------------------------------------------------------------

def reconstruct_cov(L: Array) -> Array:
    r"""Reconstruct ``Σ_{C_i} = L_{C_i} L_{C_i}ᵀ`` from Cholesky factors.

    ``jnp.tril`` enforces the lower-triangular contract structurally (it consumes
    only the lower triangle of the incoming factor; the upper triangle, if any,
    is ignored) — this is input conditioning the filter owns, not a value
    transform.  The result is symmetric PSD by construction.

    Parameters
    ----------
    L : Array, shape (N, 3, 3)
        Per-contact lower-triangular Cholesky factors from ContactNet.

    Returns
    -------
    Array, shape (N, 3, 3)
        Per-contact covariances ``Σ_{C_i}`` (body / contact frame).
    """
    Ltri = jnp.tril(L)                                  # enforce lower-triangular
    return jnp.matmul(Ltri, jnp.swapaxes(Ltri, -1, -2))  # L Lᵀ, vectorised over i


def apply_floor(Sigma: Array, floor: float) -> Array:
    r"""Apply the variance noise floor ``Σ ← Σ + floor · I`` (§5).

    Additive floor: raises every eigenvalue by ``floor`` so the conditioned
    covariance has minimum eigenvalue ``≥ floor`` (``Σ`` is PSD ⇒ eigenvalues
    ``≥ 0``).  Branch-free and smooth — unlike an eigenvalue clamp — which keeps
    the BPTT gradient finite even at a singular ``Σ``.

    Parameters
    ----------
    Sigma : Array, shape (N, 3, 3)
        Per-contact covariances.
    floor : float
        Variance floor [m²] (`InEKFParams.contact_floor`).

    Returns
    -------
    Array, shape (N, 3, 3)
    """
    return Sigma + floor * jnp.eye(3)                   # broadcast over contacts


def rotate_to_world(Sigma: Array, R: Array) -> Array:
    r"""Rotate per-contact covariances to world, ``Σ^W_{C_i} = R̄ Σ_{C_i} R̄ᵀ``.

    This is the only place ``R̄`` enters the contact noise (the honest frame
    caveat of §3.3); the rotation preserves symmetry, eigenvalues and the floor.

    Parameters
    ----------
    Sigma : Array, shape (N, 3, 3)
        Per-contact covariances in the body / contact frame.
    R : Array, shape (3, 3)
        Base orientation ``R̄ = {}^{W}R_{B}`` (from the current state mean).

    Returns
    -------
    Array, shape (N, 3, 3)
        World-frame densities, ready for `propagate.build_Qd`.
    """
    return jnp.einsum("ij,njk,lk->nil", R, Sigma, R)    # R Σ Rᵀ per contact


# ---------------------------------------------------------------------------
# Full digest (§5) — the single entry point the filter calls
# ---------------------------------------------------------------------------

def digest(L: Array, R: Array, params: InEKFParams) -> Array:
    r"""Digest ContactNet Cholesky factors into world-frame noise densities (§5).

    Composes reconstruct → floor → rotate.  The output ``sigma_c`` (shape
    ``(N, 3, 3)``) is exactly what `propagate.propagate` / `propagate.build_Qd`
    consume for the ``Q̄_d`` contact blocks (``Σ^W_{C_i} dt``).

    Parameters
    ----------
    L : Array, shape (N, 3, 3)
        Per-contact lower-triangular Cholesky factors ``L_{C_i}`` from ContactNet.
    R : Array, shape (3, 3)
        Base orientation ``R̄`` (current state mean).
    params : InEKFParams
        Carries the variance floor (`InEKFParams.contact_floor`).

    Returns
    -------
    Array, shape (N, 3, 3)
        World-frame per-contact noise densities ``R̄ Σ_{C_i} R̄ᵀ``.
    """
    Sigma = reconstruct_cov(L)
    Sigma = apply_floor(Sigma, params.contact_floor)

    # TODO(re-anchor): see §5.  On a candidate's transition to firm contact, snap
    # its mean d̄_i ← p̄ + R̄ h_{p,i}(q̂) and reset its covariance block via the
    # linear augmentation map P ← F P Fᵀ + G Cov Gᵀ (Hartley eq. 38), gated by a
    # *soft* contact indicator through jnp.where (branch-free).  Deferred for v1;
    # this is the call site — it consumes the same per-contact Σ digested above.

    return rotate_to_world(Sigma, R)
