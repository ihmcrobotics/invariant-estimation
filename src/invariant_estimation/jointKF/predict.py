r"""
jointKF/predict.py
==================
Time update of the joint-space KF: the **exact** constant-velocity transition and
the covariance propagation ``P⁻ = F P Fᵀ + Q`` (Java `buildConstantTransition` /
`predict`, gate G6).

Why ``F = I + A Δt`` is exact, not a truncation
----------------------------------------------
The continuous model is a double integrator on ``(q, q̇)`` plus a random walk on
the per-IMU gyro bias::

    q̇  = q̇         q̈ = w_a         ḃ_ω = w_b

so ``A`` has a single nonzero block, ``A[q, q̇] = I``, and ``A² = 0`` — ``A`` is
nilpotent on the joint block and identically zero on the bias block.  The matrix
exponential therefore closes after two terms::

    F = exp(A Δt) = I + A Δt          (exactly, for every Δt)

That is why `testMeanPropagationExact` can demand ``q⁺ᵢ = qᵢ + Δt q̇ᵢ`` to 1e-9
with no discretisation allowance: there is no truncation error to allow for.  It
also means `build_transition` needs no series, no `expm`, and no Δt-smallness
assumption — and that a change of Δt is a pure rescale of one off-diagonal block.

Bias is *not* coupled to the joints by the dynamics.  The gyro bias only ever
becomes observable through the **measurement** (the stacked ``L Σ Lᵀ`` gyro rows
and the stance anchors, I6), never through ``F``.  Hence the zero off-diagonal
blocks that `testBuildFStructureAndExactness` pins: a nonzero bias↔joint coupling
here would manufacture observability the physics does not provide, and
`testBiasBlockGrowthDiagonal` would see the bias marginal grow by something other
than exactly ``Δt σ_b² I``.

Decoupling from the process noise (I10)
---------------------------------------
`predict` takes ``F`` and ``Q`` as **arguments** rather than rebuilding them.
Both are pure functions of the build/params (``F``) and of ``q̂`` through the mass
matrix (``Q``, `jointKF.process`), so passing them in keeps this module a pure
map over the ``(x, P)`` carry, makes the Java `getTransitionMatrix` /
`getProcessNoise` seams literally the sub-functions the tests call, and lets the
orchestrator hoist the constant ``F`` out of the scan.
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

    Parameters
    ----------
    build : JointKFBuild
        Supplies ``n_joints`` and ``dim`` (both static).
    params : JointKFParams
        Supplies ``dt``.

    Returns
    -------
    Array, shape (dim, dim)
    """
    n, dim = build.n_joints, build.dim
    rows = jnp.arange(n)
    return jnp.eye(dim, dtype=jnp.float64).at[rows, n + rows].set(params.dt)


def predict(state: JointKFState, F: Array, Q: Array) -> JointKFState:
    r"""One time update: ``x⁻ = F x``, ``P⁻ = F P Fᵀ + Q`` (Java `predict`).

    The covariance is symmetrised as ``½(P + Pᵀ)`` on the way out.  ``F P Fᵀ`` is
    a congruence and ``Q`` is symmetric, so the result is symmetric in exact
    arithmetic; the symmetrisation removes the ~1e-16 asymmetry that the matrix
    products accumulate, which otherwise compounds tick over tick and eventually
    shows up as a (tiny, negative) eigenvalue in the PSD checks.  It is a
    projection onto the symmetric matrices, not a correction: it cannot mask a
    genuinely wrong ``Q``.

    Parameters
    ----------
    state : JointKFState
        Prior carry ``(x, P)``.
    F : Array, shape (dim, dim)
        Transition, from `build_transition`.
    Q : Array, shape (dim, dim)
        Discrete process noise for one step, from `jointKF.process`.  Passed in
        rather than rebuilt: it depends on ``q̂`` through ``M(q)`` and that
        dependency belongs to the caller (I10).

    Returns
    -------
    JointKFState
        Predicted carry ``(x⁻, P⁻)``.
    """
    x = F @ state.x
    P = F @ state.P @ F.T + Q
    return JointKFState(x=x, P=0.5 * (P + P.T))
