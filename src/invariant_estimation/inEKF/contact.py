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
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp

from ..config import section
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
    # Touchdown re-anchoring lives in `reseed.py` and is applied in `filter.step`,
    # not here: it is a covariance *congruence* on the whole state, not a
    # per-contact density.  The rolling-anchor term below IS a density and is
    # added by `filter.step` immediately after this digest.
    return Sigma


# ---------------------------------------------------------------------------
# Rolling-anchor density — the rigid-foot term (derivation in the class docstring)
# ---------------------------------------------------------------------------

class RollingAnchorParams(NamedTuple):
    r"""Configuration for the rolling-anchor contact density (``rolling_anchor:``).

    Why this exists
    ---------------
    The shipped contact process model is ``ḋ_i = 0``: "this material point is
    world-stationary".  For a *rigid foot* that is only true while the anchor
    coincides with the contact patch.  Rigid-body kinematics gives, for any two
    material points ``d_i`` and ``c`` of the same foot,

        ḋ_i = ċ + ω × (d_i − c),

    and no-slip contact says the material point at the patch is instantaneously
    stationary, ``ċ = 0``.  So

        ḋ_i = ω × r_i,        r_i = d_i − c.

    The anchor's *translation* is generated by the foot's *rotation*; the model is
    exact iff the anchor sits on the patch.  Measured on Alex
    (`experiments/anchor_static_check.py`): the N=2 sole-centre anchor rises
    **16–29 mm per stance**, essentially all of it in the final third, as the foot
    pitches up about its toe edge at toe-off.  The filter charges that to the base
    and sinks.  This term is the fix.

    ``ω`` is *measured* — base gyro plus the leg encoders' angular Jacobian, no
    contact inference.  ``r_i`` is not: locating the patch is exactly the hard
    problem.  Modelling ``r_i`` as zero-mean with ``Cov(r_i) = σ_r² I`` and
    pushing it through the (known) linear map ``r ↦ [ω]_× r`` gives

        Cov(ḋ_i) = σ_r² [ω]_× [ω]_×ᵀ = σ_r² (‖ω‖² I₃ − ω ωᵀ)

    using ``[ω]_×ᵀ = −[ω]_×`` and ``[ω]_×² = ω ωᵀ − ‖ω‖² I``.

    Three properties, all of which are the point:

    * **rank 2, null along ω** — a point rotating about an axis moves in the plane
      perpendicular to it and nowhere else.  The anisotropy is derived, not
      imposed, and the anchor keeps full stiffness along the axis.
    * **identically zero at ω = 0** — flat stance is untouched.  Self-gating on a
      measured quantity: no threshold, no schedule, no latch, no contact flag.
    * **frame-equivariant** — ``R(‖ω_B‖²I − ω_Bω_Bᵀ)Rᵀ = ‖ω_W‖²I − ω_Wω_Wᵀ``, so
      building it in the body frame and letting ``Ad_X̂`` rotate it (as
      `propagate.build_Qd` does for every contact block) is exactly correct.

    Attributes
    ----------
    enabled : bool
        Build-time flag.  Static; a disabled build adds nothing to the graph.
    tau : float
        Correlation time [s].  ``ω × r_i`` is a *coherent* disturbance over the
        toe-off window, not white noise: it accumulates ``∝ T`` while a Wiener
        process accumulates ``∝ √T``.  Approximating one by the other costs a
        factor of the disturbance's correlation time, which is the toe-off
        duration (~0.15–0.3 s measured).  This is the one honest fudge in the
        derivation; everything else is an identity.
    sigma_r : float
        Prior std [m] on ``‖r_i‖`` — how far the contact patch might be from the
        anchor.  For the N=2 sole-centre anchor this is about half the foot
        length (0.0985 m on the URDF foot).  With per-corner anchors the *prior*
        is not smaller — the far edge is a whole foot away — what N=8 buys is
        that differences of corner innovations, ``ν_i − ν_j = ξ_{d_j} − ξ_{d_i}``,
        cancel ``ξ_p`` and so measure the foot's internal geometry independently
        of the base, making ``r_i`` observable rather than merely bounded.
    """
    enabled: bool
    tau: float
    sigma_r: float


def default_rolling_anchor_params(
    enabled: bool | None = None,
    tau: float | None = None,
    sigma_r: float | None = None,
) -> RollingAnchorParams:
    """Build `RollingAnchorParams` from the ``rolling_anchor`` config section."""
    cfg = section("rolling_anchor")
    params = RollingAnchorParams(
        enabled=bool(cfg["enabled"] if enabled is None else enabled),
        tau=float(cfg["tau"] if tau is None else tau),
        sigma_r=float(cfg["sigma_r"] if sigma_r is None else sigma_r),
    )
    if params.tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {params.tau}")
    if params.sigma_r < 0.0:
        raise ValueError(f"sigma_r must be >= 0, got {params.sigma_r}")
    return params


def rolling_anchor_density(omega: Array, params: RollingAnchorParams) -> Array:
    r"""``κ (‖ω‖² I₃ − ω ωᵀ)`` per contact, ``κ = τ σ_r²`` (see `RollingAnchorParams`).

    Parameters
    ----------
    omega : Array, shape (N, 3)
        Each contact's **foot** angular velocity, in the same frame the rest of
        ``Σ_C`` lives in (body frame ``B`` — `contact.digest`'s output frame).
    params : RollingAnchorParams

    Returns
    -------
    Array, shape (N, 3, 3)
        Symmetric PSD density [m²/s], to be **added** to the digested ``Σ_C``.
    """
    kappa = params.tau * params.sigma_r ** 2
    w2 = jnp.sum(omega ** 2, axis=-1)                       # (N,)
    outer = jnp.einsum("ni,nj->nij", omega, omega)          # (N,3,3)
    return kappa * (w2[:, None, None] * jnp.eye(3) - outer)
