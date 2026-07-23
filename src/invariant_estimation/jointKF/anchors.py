r"""
jointKF/anchors.py
==================
Stance anchors — the **only absolute observation of gyro bias in the filter**
(Java `JointLevelKFPreFilter` anchor loop L1826-1875, gate G7,
`JointLevelKFBiasObservabilityTest`).

Why this module exists at all
-----------------------------
The IMU-pair rows measure a *relative* gyro,
``omega_child - {}^{c}R_{p} omega_parent``.  Rotate every IMU's bias by that
IMU's own attitude and the differences cancel identically: for
``delta b_i = {}^{i}R_{W} beta`` (one common bias in the *world*, seen by each
IMU in its own frame),

.. math::
    (+I_3)\,\delta b_c \;-\; {}^{c}R_{p}\,\delta b_p
      = {}^{c}R_{W}\beta - {}^{c}R_{p}\,{}^{p}R_{W}\beta = 0 .

So the pair block has an exact **3-dimensional nullspace** — the common-mode
gauge.  No amount of relative-gyro data ever shrinks it.  Left unfixed, the
gauge direction of the bias covariance grows without bound under the bias random
walk, the joint KF hands the InEKF a bias-corrected ``omega_bar`` that is wrong
by a slowly-wandering constant, and the InEKF integrates that straight into
attitude.  This is the documented root cause of Alex's pelvis pitch drift.

A trusted stance foot fixes the gauge, because it is the one thing in the system
with a *known absolute* angular rate: ~zero.

The anchor equation
-------------------
Expressed in the **base IMU's** measurement frame (which is why the ``+I_3``
lands on the base IMU's bias columns and nowhere else):

.. math::
    0 \;\approx\; \omega_{\text{foot}}
      \;=\; \omega_{\text{baseIMU}} + J(\text{baseIMU}\to\text{foot})\,\dot q ,
    \qquad
    \omega_{\text{baseIMU}} = \tilde\omega_{\text{base}} - b_{\text{base}} .

Splitting ``J qdot = J_F qdot_F + J_U qdot_U`` (see the F/U split below) and
moving everything measured to the left gives the row this module builds::

    z = omega_tilde_base + J_U qdot_U_measured
    H = [ 0_q | -J_F | +I3 at bias_col(base_imu) ]

The ``+I_3`` is the gauge fixer.  Applying it to the gauge direction returns
``{}^{b}R_{W}\beta``, whose norm is ``|beta|`` — the anchor reads the common-mode
bias back *exactly*, which is precisely what
`testStanceAnchorFixesTheGauge` asserts.

Sign note: the overall row sign is a free choice (both tests are on norms).  The
convention above is the one forced by `tests/jointKF/_oracles.reference_marginalized`
— eliminate the nuisance ``omega_base`` from that oracle's base-IMU row
(``z = J qdot + b + I omega_base``, with ``J = 0`` for the base IMU itself) and
substitute into its foot row (``0 = J_leg qdot + omega_base``), and this row is
what drops out.  Keeping the two consistent is what lets the Phase-3 stacked
oracle compose.

The F/U split and why ``R_anchor`` is not just ``Sigma_eps``
-----------------------------------------------------------
The base->foot chain generally contains joints that are **not filter states** —
on Alex the ankles, because there are no foot IMUs, so no IMU pair brackets them.
Their velocity is not estimated; it enters the anchor row as a **known input**
read straight off the encoders.  By the standard input-noise congruence its
covariance must therefore propagate into the measurement covariance:

.. math::
    R_{\text{anchor}} = \Sigma_\varepsilon + J_U \operatorname{diag}(\sigma_{\dot q,U}^2) J_U^\mathsf{T}

with ``Sigma_eps = anchor_var * I3`` (4e-4; the ContactNet injection point,
CLAUDE.md §7) and ``sigma_qd_unfiltered = 0.1 rad/s``.  Dropping the congruence
term is not conservative: with four ankle joints at 0.1 rad/s the congruence is
an order of magnitude *above* ``Sigma_eps``, so omitting it over-trusts a noisy
input and feeds that noise directly into the base gyro-bias estimate the whole
InEKF attitude solution rests on.  Erring large merely weakens the anchor — the
gauge still gets fixed, just with more averaging.

Constant-graph masking (CLAUDE.md §4 — this rule lives HERE, not in the InEKF)
-----------------------------------------------------------------------------
``K_max = build.n_anchors`` is fixed for the filter's lifetime (invariant I2), so
the block is **always** ``3*K_max`` rows and a foot landing changes a *mask*,
never a shape.  For an inactive anchor:

* the residual is zeroed,
* the ``H`` rows are zeroed,
* and the ``R`` block becomes ``r_large * I3`` — **never zero**.  A zeroed ``R``
  block on zeroed ``H`` rows makes ``S = H P Hᵀ + R`` exactly singular, which is
  a named trap (CLAUDE.md §6): the Cholesky in `update.joseph_update` goes
  non-finite and the *whole* stacked update — pair rows included — is gated out.

Zeroing ``H`` as well as the residual goes one step beyond CLAUDE.md §4's
minimum, deliberately.  Java's stacked ``H`` literally *has no anchor rows* when
no foot is trusted, and `testCommonModeBiasIsUnobservableWithoutAnchors` reads
``H`` directly: an unmasked-but-large-``R`` anchor row would leave the gauge
looking observable *in H* while contributing nothing to the estimate, so the
ported assertion would have to be weakened to a slice.  Masking ``H`` makes the
fixed-shape block behave exactly like Java's dynamic one — the gain contribution
is identically zero rather than merely ``O(1/r_large)``.

Phase ordering
--------------
`trusted_feet` is an explicit argument and is **never** computed here: the
previous tick's trusted set drives this tick's anchors (CLAUDE.md §4, mask
written at the end of step k and read at the start of k+1).  Deciding trust
inside this module would silently make it same-tick.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .state import JointKFBuild, JointKFParams

__all__ = [
    "AnchorJacobians",
    "AnchorBlock",
    "unfiltered_dof",
    "anchor_jacobians",
    "anchor_noise",
    "anchor_block",
]


class AnchorJacobians(NamedTuple):
    r"""The F/U split of the base->foot angular Jacobian, in the base IMU frame.

    Attributes
    ----------
    filtered : Array, shape (K, 3, n)
        ``J_F`` — columns of the filter's own joints.  These multiply *state*
        (``q_dot``), so they land in ``H``.
    unfiltered : Array, shape (K, 3, n_u)
        ``J_U`` — columns of chain joints that are not filter states.  These
        multiply a *measured input*, so they land in ``z`` (times the reading)
        and in ``R`` (times the reading's covariance).
    """

    filtered: Array
    unfiltered: Array


class AnchorBlock(NamedTuple):
    r"""The anchor rows of the stacked measurement, always ``3*K_max`` of them.

    Concatenates onto the pair rows built by `measure.py` at
    ``build.anchor_row0 == 3 * build.n_pairs``; together they fill
    ``build.n_stacked_rows``.

    Attributes
    ----------
    H : Array, shape (3K, dim)
    z : Array, shape (3K,)
    R : Array, shape (3K, 3K)
        Block-diagonal by construction: each foot's slip is modelled as
        independent of every other foot's.  (The two feet *are* correlated
        through the shared base-IMU gyro noise — see the module note in
        PORT_NOTES; Java models neither that nor the base gyro's own
        contribution to ``R_anchor``, and this port follows Java.)
    active : Array, shape (K,)
        The float mask actually applied, echoed back so a caller need not
        re-derive it.
    n_active : Array, scalar
        Java `getActiveAnchorCountForTest` — a diagnostic, part of the seam
        surface (invariant I10), not optional logging.
    """

    H: Array
    z: Array
    R: Array
    active: Array
    n_active: Array


def unfiltered_dof(build: JointKFBuild) -> np.ndarray:
    """DoF indices of the unfiltered anchor-chain joints, in mask-column order.

    `build.py` builds ``dof_nuisance = [base DoFs] ++ [unfiltered chain joints]``
    with the second group in the same (sorted) order as
    ``anchor_unfiltered_mask``'s columns, so the trailing ``n_u`` entries are
    exactly what ``J_U``'s column gather needs.

    Kept as a named helper rather than inlined at the call site because the
    coupling to `build.py`'s concatenation order is invisible otherwise — and a
    silently mis-ordered gather would put the ankle's Jacobian column under the
    hip's velocity noise, inflating ``R_anchor`` by the wrong amount in a way no
    shape check would catch.
    """
    n_u = int(np.asarray(build.anchor_unfiltered_mask).shape[1])
    dof = np.asarray(build.dof_nuisance, dtype=int)
    return dof[len(dof) - n_u:] if n_u else dof[:0]


def anchor_jacobians(
    build: JointKFBuild,
    J_ang_world: Array,
    site_rot: Array,
    *,
    base_site: int,
    foot_sites: Array,
    dof_unfiltered: Array | None = None,
) -> AnchorJacobians:
    r"""Split the base->foot angular Jacobian into its F and U parts.

    The anchor constrains the foot's rate *relative to the base IMU's*, with the
    base IMU's own rate supplied by its gyro (and its bias by the state).  So the
    Jacobian is the **difference** of the two sites' absolute angular Jacobians,
    rotated into the base IMU frame:

    .. math::
        J = {}^{W}R_{b}^{\mathsf{T}} \left( J^{W}_{\text{foot}} - J^{W}_{\text{base}} \right).

    Differencing is also what removes the floating base: its three rotational DoFs
    enter both site Jacobians as the same identity block and cancel exactly, while
    its translational DoFs generate no angular velocity at all.  Gathering only
    joint columns therefore loses nothing — the same argument
    `MjxModel.relative_gyro_jacobian` rests on.

    Parameters
    ----------
    build : JointKFBuild
        Supplies ``dof_joint`` and the two anchor masks.
    J_ang_world : Array, shape (n_sites, 3, nv)
        World-frame site angular Jacobians (`MjxModel.site_angular_jacobians`).
    site_rot : Array, shape (n_sites, 3, 3)
        World rotations of the same sites.
    base_site : int
        Site ordinal of the base IMU.  Static (a Python int) — it selects the
        frame the whole row is written in, not data.
    foot_sites : Array, shape (K,) int
        Site ordinals of the sole sites, in `build`'s anchor order.
    dof_unfiltered : Array, shape (n_u,) int, optional
        Defaults to `unfiltered_dof(build)`.

    Returns
    -------
    AnchorJacobians

    Notes
    -----
    Both masks are applied.  For a clean tree they are redundant — a joint off
    the base->foot path either moves both sites identically (and cancels in the
    difference) or moves neither — which is exactly what makes applying them a
    cheap structural assertion rather than a correction.
    """
    J_ang_world = jnp.asarray(J_ang_world, dtype=jnp.float64)
    site_rot = jnp.asarray(site_rot, dtype=jnp.float64)
    feet = jnp.asarray(foot_sites, dtype=int)
    dof_u = unfiltered_dof(build) if dof_unfiltered is None else np.asarray(dof_unfiltered, dtype=int)

    diff = J_ang_world[feet] - J_ang_world[base_site]            # (K, 3, nv), world
    R_base = site_rot[base_site]                                 # (3, 3)
    J = jnp.einsum("ji,kjc->kic", R_base, diff)                  # ^W R_b^T @ diff

    mask_f = jnp.asarray(build.anchor_filtered_mask, dtype=jnp.float64)
    mask_u = jnp.asarray(build.anchor_unfiltered_mask, dtype=jnp.float64)
    return AnchorJacobians(
        filtered=J[:, :, jnp.asarray(build.dof_joint)] * mask_f[:, None, :],
        unfiltered=J[:, :, jnp.asarray(dof_u)] * mask_u[:, None, :],
    )


def anchor_noise(
    build: JointKFBuild,
    params: JointKFParams,
    jac: AnchorJacobians,
    trusted_feet: Array,
    *,
    sigma_eps: Array | None = None,
) -> Array:
    r"""Per-anchor ``3x3`` measurement covariance, masked.

    Active::

        R_k = Sigma_eps + J_U,k diag(sigma_qd_unfiltered^2) J_U,k^T

    Inactive::

        R_k = r_large * I3          # NEVER zero -- singular S (CLAUDE.md §6)

    Parameters
    ----------
    sigma_eps : Array, shape (K, 3, 3), optional
        The ContactNet socket (CLAUDE.md §7): per-foot slip covariance.  Defaults
        to the heuristic ``anchor_var * I3`` on every foot.  Kept as an argument
        rather than read from `params` so the learned provider can be dropped in
        without touching this module — and so nothing here ever needs a
        `stop_gradient`.
    """
    K = build.n_anchors
    eye3 = jnp.eye(3, dtype=jnp.float64)
    J_U = jnp.asarray(jac.unfiltered, dtype=jnp.float64)
    active = jnp.asarray(trusted_feet, dtype=jnp.float64) > 0.0        # (K,)

    eps = (
        jnp.broadcast_to(params.anchor_var * eye3, (K, 3, 3))
        if sigma_eps is None
        else jnp.asarray(sigma_eps, dtype=jnp.float64)
    )
    # Input-noise congruence for the unfiltered chain velocities.  Written as a
    # Gram product (scale the columns, then J J^T) so it is exactly symmetric
    # PSD by construction rather than merely to round-off -- the same reason
    # `process.qa_from_lambda_eff` uses the Gram form.
    Y = J_U * params.sigma_qd_unfiltered
    R_on = eps + jnp.einsum("kic,kjc->kij", Y, Y)
    R_off = jnp.broadcast_to(params.r_large * eye3, (K, 3, 3))
    return jnp.where(active[:, None, None], R_on, R_off)


def anchor_block(
    build: JointKFBuild,
    params: JointKFParams,
    jac: AnchorJacobians,
    *,
    gyro_base: Array,
    qd_unfiltered: Array,
    trusted_feet: Array,
    sigma_eps: Array | None = None,
) -> AnchorBlock:
    r"""Build the ``(H, z, R)`` anchor block — Java's anchor loop, fixed-shape.

    Parameters
    ----------
    gyro_base : Array, shape (3,)
        **Raw** (bias-uncorrected) gyro of the base IMU, in its own measurement
        frame.  Uncorrected on purpose: the bias is what the row observes, so
        subtracting an estimate here would close the loop on the filter's own
        guess and destroy the very observability the anchor provides.
    qd_unfiltered : Array, shape (n_u,)
        Measured velocities of the unfiltered chain joints, in
        ``anchor_unfiltered_mask`` column order (see `unfiltered_dof`).
    trusted_feet : Array, shape (K,)
        **Previous tick's** trusted-stance mask, one entry per anchor slot.
        Non-zero means trusted; the value itself is not used as a weight, because
        the Java trusted set is boolean and partial trust is expressed upstream
        (the Schmitt trigger) rather than by softening the anchor.
    sigma_eps : Array, shape (K, 3, 3), optional
        See `anchor_noise`.

    Returns
    -------
    AnchorBlock
    """
    n, K = build.n_joints, build.n_anchors
    dim, m = build.dim, build.n_imus

    gyro_base = jnp.asarray(gyro_base, dtype=jnp.float64)
    qd_u = jnp.asarray(qd_unfiltered, dtype=jnp.float64)
    J_F = jnp.asarray(jac.filtered, dtype=jnp.float64)
    J_U = jnp.asarray(jac.unfiltered, dtype=jnp.float64)
    active = (jnp.asarray(trusted_feet, dtype=jnp.float64) > 0.0).astype(jnp.float64)

    # -- bias columns: +I3 on THIS anchor's IMU, zero everywhere else ---------
    # One-hot times I3, not a scatter: the graph must not depend on which IMU an
    # anchor pins (I7), and this also keeps the block correct if a future robot
    # anchors different feet to different IMUs.
    one_hot = (jnp.arange(m)[None, :] == jnp.asarray(build.anchor_imu)[:, None]).astype(jnp.float64)
    bias_block = jnp.einsum("km,ij->kimj", one_hot, jnp.eye(3, dtype=jnp.float64)).reshape(K, 3, 3 * m)

    H = jnp.concatenate([jnp.zeros((K, 3, n), dtype=jnp.float64), -J_F, bias_block], axis=2)
    z = gyro_base[None, :] + jnp.einsum("kic,c->ki", J_U, qd_u)

    # Inactive: residual AND rows zeroed, R -> r_large * I3 (never zeroed).
    H = H * active[:, None, None]
    z = z * active[:, None]
    R_blocks = anchor_noise(build, params, jac, active, sigma_eps=sigma_eps)

    # Block-diagonalise (K,3,3) -> (3K,3K) without a Python loop: the Kronecker
    # delta on the anchor index IS the block-diagonal structure.
    R = jnp.einsum("kij,kl->kilj", R_blocks, jnp.eye(K, dtype=jnp.float64)).reshape(3 * K, 3 * K)

    return AnchorBlock(
        H=H.reshape(3 * K, dim),
        z=z.reshape(3 * K),
        R=R,
        active=active,
        n_active=jnp.sum(active),
    )
