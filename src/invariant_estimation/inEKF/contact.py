r"""Contact-covariance digest: ContactNet Cholesky factors → per-contact ``Σ_{C_i}``.

A pure consumer of the external ContactNet module — it imports nothing from it and
computes no learned quantity. `digest` is the single entry point the filter calls;
`propagate.build_Qd` consumes its output for the contact blocks of ``Q_c``.

**Frame: body/contact, not world.** Under I3 the process noise is
``Q_d = Φ Ad_X̂ Q_c Ad_X̂ᵀ Φᵀ Δt``, and the diagonal contact blocks of ``Ad_X̂`` are
``R̂`` — the adjoint conjugation already performs the rotation to world, and adds
the ``(d_i)_× R̂`` cross terms that a bare rotate-to-world cannot. Rotating here
too would apply ``R̂`` twice. `rotate_to_world` is kept as a standalone utility and
is deliberately **not** on the propagation path.

Everything here is jit-able and differentiable: BPTT flows through this digest into
ContactNet, so ``L → Σ`` must stay smooth. That is why the floor is additive and
not an eigenvalue clamp.
"""
from jax import Array
import jax.numpy as jnp

from .state import InEKFParams


def reconstruct_cov(L: Array) -> Array:
    r"""``(N,3,3)`` Cholesky factors → ``Σ_{C_i} = L L ᵀ``, symmetric PSD by construction.

    `jnp.tril` enforces the lower-triangular contract structurally: any upper
    triangle in the incoming factor is ignored rather than trusted.
    """
    Ltri = jnp.tril(L)
    return jnp.matmul(Ltri, jnp.swapaxes(Ltri, -1, -2))


def apply_floor(Sigma: Array, floor: float) -> Array:
    r"""``Σ ← Σ + floor·I`` [m²] — raises every eigenvalue by ``floor``.

    **Safety-critical since 2026-07-29.** While ``L`` came from the heuristic this
    was a conditioning nicety; ContactNet now supplies it, and the floor is the
    only thing between a mis-prediction and a *pinned swing foot* — an anchor
    asserted world-static while the foot is in flight. Measured: a pinned swing
    foot is 10.2x worse in body-frame velocity than not using contacts at all, and
    at ``floor = 1e-6`` the closed-loop filter drifts −15 m with 18° of tilt and
    the robot falls. Do not lower it to "let the network express confident stance".

    **Two floors act on this quantity — reconcile, do not stack.** The network's
    own ``eps`` (`ContactNetConfig.eps`, 1e-6) floors ``diag(L)``, contributing
    ``eps² = 1e-12`` of variance, which is negligible; it exists to keep the
    Cholesky parameterisation valid and differentiable. ``floor`` is the physical
    bound on how world-static an anchor may be asserted to be, and is the number to
    change if that bound is wrong.
    """
    return Sigma + floor * jnp.eye(3)


def rotate_to_world(Sigma: Array, R: Array) -> Array:
    r"""``Σ^W_{C_i} = R̄ Σ_{C_i} R̄ᵀ`` for ``R̄ = {}^{W}R_{B}``. NOT on the propagation path.

    See the module docstring: `build_Qd`'s ``Ad_X̂`` already rotates. Preserves
    symmetry, eigenvalues and the floor.
    """
    return jnp.einsum("ij,njk,lk->nil", R, Sigma, R)


def digest(L: Array, params: InEKFParams) -> Array:
    r"""``(N,3,3)`` ContactNet factors → body-frame ``Σ_{C_i}``: reconstruct, then floor."""
    return apply_floor(reconstruct_cov(L), params.contact_floor)
