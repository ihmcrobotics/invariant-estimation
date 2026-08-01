r"""Time update of the joint-space KF: ``x⁻ = F x``, ``P⁻ = F P Fᵀ + Q``.

Java `buildConstantTransition` / `predict`, gate G6.

**``F = I + A Δt`` is exact, not a truncation.**  The continuous model is a double
integrator on ``(q, q̇)`` plus a bias random walk, so ``A`` has a single nonzero
block ``A[q, q̇] = I`` and ``A² = 0``; the matrix exponential closes after two
terms for *every* Δt.  That is why `testMeanPropagationExact` demands
``q⁺ᵢ = qᵢ + Δt q̇ᵢ`` to 1e-9 with no discretisation allowance, and why
`build_transition` needs no series, no `expm`, and no Δt-smallness assumption.

Bias is *not* coupled to the joints by the dynamics — it becomes observable only
through the **measurement** (the stacked ``L Σ Lᵀ`` gyro rows and the stance
anchors, I6).  Hence the zero off-diagonal blocks
`testBuildFStructureAndExactness` pins: a nonzero bias↔joint coupling here would
manufacture observability the physics does not provide, and
`testBiasBlockGrowthDiagonal` would see the bias marginal grow by something other
than exactly ``Δt σ_b² I``.

`predict` takes ``F`` and ``Q`` as **arguments** rather than rebuilding them
(I10): it keeps this module a pure map over the ``(x, P)`` carry, makes the Java
`getTransitionMatrix` / `getProcessNoise` seams literally the sub-functions the
tests call, and lets the orchestrator hoist the constant ``F`` out of the scan.
"""
import jax.numpy as jnp
from jax import Array

from .state import JointKFBuild, JointKFParams, JointKFState


def build_transition(build: JointKFBuild, params: JointKFParams) -> Array:
    r"""Constant transition ``F = I + A Δt``, shape ``(dim, dim)``.

    Java `buildConstantTransition` / seam `getTransitionMatrix`.  Structure
    (locked by `testBuildFStructureAndExactness`)::

        F = [ I_n   Δt I_n   0     ]
            [ 0     I_n      0     ]
            [ 0     0        I_3m  ]

    Constant for the filter's lifetime: nothing here depends on the state, so the
    orchestrator builds it once outside the scan (I7).
    """
    n, dim = build.n_joints, build.dim
    rows = jnp.arange(n)
    return jnp.eye(dim, dtype=jnp.float64).at[rows, n + rows].set(params.dt)


def predict(state: JointKFState, F: Array, Q: Array) -> JointKFState:
    r"""One time update: ``x⁻ = F x``, ``P⁻ = F P Fᵀ + Q`` (Java `predict`).

    ``P`` is symmetrised as ``½(P + Pᵀ)`` on the way out: ``F P Fᵀ`` is a
    congruence and ``Q`` is symmetric, so this only removes the ~1e-16 asymmetry
    the matrix products accumulate, which otherwise compounds tick over tick into
    a tiny negative eigenvalue in the PSD checks.  It is a projection onto the
    symmetric matrices, not a correction: it cannot mask a genuinely wrong ``Q``.
    """
    x = F @ state.x
    P = F @ state.P @ F.T + Q
    return JointKFState(x=x, P=0.5 * (P + P.T))
