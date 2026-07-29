r"""
inEKF/contact.py
================
Contact-covariance **digest** for the world-centric, right-invariant InEKF
(CLAUDE.md §5).  This module is a pure *consumer*: it takes the lower-triangular
Cholesky factors ``L_{C_i}`` emitted by the external ContactNet module and turns
them into the per-contact covariances ``Σ_{C_i}`` that `propagate.build_Qd`
injects into the contact blocks of the continuous density ``Q_c``.

> **It contains no learned components and does not compute ``Σ_{C_i}``.**
> ContactNet (the MLP, its features, weights and the SPD-safe parameterisation
> that *produces* a valid Cholesky factor with positive diagonal) is a separate
> module, out of scope here (invariant 4).  This file imports nothing from it; it
> only consumes its output — the lower-triangular Cholesky argument, nothing more.

The digest is two pure, branch-free, vectorised-over-contacts steps:

    1. reconstruct   Σ_{C_i} = L_{C_i} L_{C_i}ᵀ          (`reconstruct_cov`)
    2. noise floor   Σ_{C_i} ← Σ_{C_i} + floor · I       (`apply_floor`)

`digest` composes the two and is the single entry point the filter calls.  The
output stays in the **body / contact frame**: under CLAUDE.md I3 the process
noise is ``Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt``, and the diagonal contact blocks of
``Ad_X̂`` are ``R̂`` — so the adjoint conjugation already performs the rotation to
world (and adds the ``(d_i)_× R̂`` cross terms with rotation, which a bare
rotate-to-world cannot).  Rotating here as well would apply ``R̂`` twice.
`rotate_to_world` is kept as a standalone utility but is deliberately **not**
part of the propagation path.

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

    **This floor is safety-critical since 2026-07-29.**  While ``L`` came from
    the heuristic it was a conditioning nicety: the supplier never emitted
    anything near singular.  ContactNet now supplies it, and the floor is the
    only thing between a mis-prediction and a *pinned swing foot* — an anchor
    asserted world-static while the foot is in flight, which is the run-1 failure
    mode (measured 10.2x worse in body-frame velocity than not using contacts at
    all).  Do not lower it to "let the network express confident stance": at
    1e-6 the closed-loop filter was measured at −15 m of drift and 18° of tilt,
    and the robot falls.

    **Two floors act on one quantity — reconcile them, do not stack them.**  The
    network's own ``eps`` (`ContactNetConfig.eps`, 1e-6) floors ``diag(L)``, so
    ``L Lᵀ`` already has eigenvalues ``≥ eps²`` before this adds ``floor``.  They
    are not redundant and not interchangeable:

    * ``eps`` is a *factor* floor, applies before the outer product, and exists
      to keep ``softplus`` output strictly positive so the Cholesky
      parameterisation stays valid and differentiable.  It contributes
      ``eps² = 1e-12`` of variance, which is negligible here.
    * ``floor`` is a *variance* floor applied to the reconstructed ``Σ`` and is
      the actual physical bound on how world-static any anchor may be asserted
      to be.  It is the number to change if that bound is wrong.

    The effective floor is therefore ``floor`` alone at any sane setting, and
    `PORT_NOTES.md` §"ContactNet seam" was corrected on the same date: it had
    ``eps`` as the measurement-socket lever and ``floor`` as the process one,
    which stopped being true when the network moved to the process socket.

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

def digest(L: Array, params: InEKFParams) -> Array:
    r"""Digest ContactNet Cholesky factors into per-contact covariances (§5).

    Composes reconstruct → floor.  The output ``sigma_c`` (shape ``(N, 3, 3)``)
    is exactly what `propagate.propagate` / `propagate.build_Qd` consume for the
    contact blocks of ``Q_c``.

    Frame: **body / contact**, not world.  ``Ad_X̂`` in `build_Qd` does the
    rotation (see the module docstring) — digesting to world here would double-
    rotate.

    Parameters
    ----------
    L : Array, shape (N, 3, 3)
        Per-contact lower-triangular Cholesky factors ``L_{C_i}`` from ContactNet.
    params : InEKFParams
        Carries the variance floor (`InEKFParams.contact_floor`).

    Returns
    -------
    Array, shape (N, 3, 3)
        Per-contact body-frame covariances ``Σ_{C_i}``.
    """
    Sigma = reconstruct_cov(L)
    Sigma = apply_floor(Sigma, params.contact_floor)


    # TODO(re-anchor): see §5.  On a candidate's transition to firm contact, snap
    # its mean d̄_i ← p̄ + R̄ h_{p,i}(q̂) and reset its covariance block via the
    # linear augmentation map P ← F P Fᵀ + G Cov Gᵀ (Hartley eq. 38), gated by a
    # *soft* contact indicator through jnp.where (branch-free).  Deferred for v1;
    # this is the call site — it consumes the same per-contact Σ digested above.

    return Sigma
