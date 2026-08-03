"""
jointKF/measure.py
==================
The measurement side of the joint-space KF: the encoder rows, and the **stacked
relative-gyro rows** over the IMU-pair graph (paper §II, Java
`JointLevelKFPreFilter.buildStackedMeasurementForTest`).

The one identity this module exists to get right (invariant I6)
---------------------------------------------------------------
For an IMU pair `(a, b)` the filter does not measure two gyros; it measures their
*difference*::

    z_e = omega_b^b - {}^{b}R_{a} omega_a^a
        = J_ang(q) S_ab qdot  +  b_b  -  {}^{b}R_{a} b_a  +  ( v_b - {}^{b}R_{a} v_a )

Two facts follow, and they are the whole module:

1. **The bias columns of `H_g` are a linear operator on the per-IMU bias vector**
   -- `+I3` on the child (the child's measurement frame *is* the Jacobian frame,
   so no rotation), `-{}^{b}R_{a}` on the parent.  Call that operator `L`
   ((3*n_pairs) x (3m)).  It is built ONCE here and *scattered into* `H_g`, so
   `H_g[:, 2n : 2n+3m] == L` holds bit-identically by construction rather than by
   coincidence (`testBiasColumnsOfHgAreExactlyL`, tol 0.0).

2. **The same `L` mixes the noise**: the stacked measurement noise is the exact
   congruence `R_g = L Sigma L^T` with `Sigma = blkdiag(Sigma_imu)`.  It is
   written literally that way below.  A block-diagonal per-pair `R_g` is WRONG on
   a shared-IMU star -- two pairs that share an IMU inherit that IMU's noise with
   opposite signs, and `L Sigma L^T` produces exactly the resulting off-diagonal
   block.  Assembling `R_g` per pair and hoping it agrees is how that block gets
   silently dropped.

   The trap is close to invisible: for isotropic `Sigma = sigma^2 I`,
   `R Sigma R^T = sigma^2 R R^T = sigma^2 I` **exactly**, so on a single pair with
   isotropic noise the block-diagonal form is not an approximation but an
   identity (measured deviation 2.6e-20).  Only anisotropic `Sigma`, or a
   shared-IMU star, can tell them apart -- see `PORT_NOTES.md`, "I6 is invisible
   on three of the four shapes".

`Sigma` is the gyro **MEASUREMENT** covariance, never the bias random-walk
covariance (`testMeasurementNoiseUsesGyroMeasurementCovariance`, and
`testMeasurementNoiseIndependentOfBiasProcessCovariance` at tol 0.0).  The bias
random walk is a *process* noise and lives in `process.py`; the two are numerically
far apart (1e-4 vs 1e-9 in the Java fixture) so mixing them up is a quiet 5-order
mis-weighting rather than a crash.  `build.gyro_sigma` is already floored at build
time -- do not floor again per tick (invariant I7: the decision is not data
dependent).

Which frame does `b_omega` live in?
-----------------------------------
**Each IMU's own measurement (site) frame.**  That is what makes the child block
exactly `+I3` and the parent block exactly `-{}^{b}R_{a}`, and it is the
convention the model seam already chose for `relative_gyro_jacobian` (child
frame, Java `GeometricJacobianCalculator`) and that `state.py`'s
`b_omega_imus` documents.  It is *verified*, not merely assumed, by
`tests/jointKF/test_measurement.py::test_stacked_pair_rows_match_the_marginalized_
raw_gyro_reference`, which puts `+I3` on each IMU's bias in that IMU's own frame,
carries the shared unknown `omega_base` on a rotation column, and marginalises it
out -- an independently-derived route to the same posterior.

Shapes are constant (invariant I7)
----------------------------------
`build_stacked` always returns `3*(n_pairs + n_anchors)` rows.  An untrusted
stance anchor keeps its rows: its residual is zeroed and its `R` block is set to
`params.r_large * I3`, never to zero (a zero `R` block makes `S` singular --
CLAUDE.md §6).  The anchor block itself is built by `anchors.py` and passed in;
this module only stacks and masks it, so the two can be developed against the
row-layout contract (`build.anchor_row0`, `build.n_stacked_rows`) rather than
against each other's code.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from .state import JointKFBuild, JointKFParams

__all__ = ["StackedMeasurement", "AnchorBlock", "encoder_jacobian", "encoder_noise",
           "pair_frames", "mixing_operator", "build_stacked"]


class AnchorBlock(NamedTuple):
    """The stance-anchor rows, as produced by `anchors.py` (owner: B2).

    Fixed shape `3*n_anchors` rows always -- a foot landing flips `trusted_feet`,
    never a dimension (invariant I2).  `R` is the *trusted* anchor noise
    `Sigma_eps + J_U diag(sigma_qd_U^2) J_U^T`; `build_stacked` is what swaps in
    `r_large * I3` for the untrusted rows, so `anchors.py` never has to know about
    the masking convention.

    This is a **structural fallback**, not the only accepted type: `build_stacked`
    reads `.H`, `.z`, `.R` and nothing else, so `anchors.AnchorBlock` (which also
    carries `active` / `n_active` diagnostics) drops straight in.  Note that
    `anchors.py` applies the same mask on its side; the two compose because the
    masking is idempotent -- `r_large` selected twice is still `r_large`, and a
    zeroed residual zeroed again is still zero.  Who *owns* that masking is a
    genuine duplication for the parent to collapse, not a numerical discrepancy.
    """

    H: Array   # (3*n_anchors, dim)
    z: Array   # (3*n_anchors,)
    R: Array   # (3*n_anchors, 3*n_anchors)


class StackedMeasurement(NamedTuple):
    """One tick's stacked gyro + anchor measurement.

    Java seams: `getStackedMeasurementJacobian/Residual/Noise` and
    `getMixingOperator` (`state.SEAM_MAP`).

    Attributes
    ----------
    H : (n_stacked_rows, dim)
        Rows `[3e, 3e+3)` belong to pair `e` (`build.stacked_row_for_pair`);
        rows from `build.anchor_row0` on are the anchors.
    z : (n_stacked_rows,)
        The measurement itself, not the innovation: `z_e = omega_child -
        R_{child<-parent} omega_parent`.  The filter forms `nu = z - H x`.
    R : (n_stacked_rows, n_stacked_rows)
        `L Sigma L^T` on the pair block (exactly -- invariant I6), the masked
        anchor noise on the anchor block, and zero between them.
    L : (n_stacked_rows, 3m)
        The mixing operator.  Bit-identical to `H[:, 2n:2n+3m]`.
    """

    H: Array
    z: Array
    R: Array
    L: Array


# ---------------------------------------------------------------------------
# Encoder rows -- linear and time-invariant
# ---------------------------------------------------------------------------

def encoder_jacobian(build: JointKFBuild, params: JointKFParams | None = None) -> Array:
    """`H_enc = [I_n | 0]`, shape `(n, dim)` -- Java `getEncoderJacobian`.

    Encoders observe `q` and nothing else: not `q_dot` (which is *inferred* from
    the encoder history through the process model, never measured directly on
    this channel) and not bias.  Locked by `testEncoderJacobianStructure` and
    `testEncoderPredictsPosition`.

    `params` is accepted and unused so the two call styles in the ported suite
    (`(build)` here, `(build, params)` in `test_transition_noise.py`) both work --
    nothing about the encoder Jacobian is tunable.
    """
    n = build.n_joints
    return jnp.eye(n, build.dim, dtype=jnp.float64)


def encoder_noise(build: JointKFBuild, params: JointKFParams | None = None) -> Array:
    """`R_enc = diag(build.encoder_var)`, shape `(n, n)` -- Java `getEncoderNoise`.

    Strictly diagonal: encoder errors on separate joints share no mechanism, and
    the per-joint variances come from `build` (name-resolved, with the loud 5e-5
    fallback) rather than from a single scalar -- invariant I9, and the wiring
    `JointLevelKFEncoderNISConsistencyTest` locks in.

    `params` is accepted for call-style compatibility (see `encoder_jacobian`)
    and deliberately NOT read: `params.encoder_var` is only the scalar fallback,
    which `build.py` has already applied per joint. Reading it here would silently
    re-uniformise the variances -- exactly the invariant I9 failure.
    """
    return jnp.diag(jnp.asarray(build.encoder_var, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Pair geometry
# ---------------------------------------------------------------------------

def pair_frames(model, q: Array) -> tuple[Array, Array]:
    r"""`(J_rel, R_rel)` for every pair, from ONE position-level model pass.

    * `J_rel` : `(n_pairs, 3, n)` -- `J_ang(q) S_ab` in the **child** frame.
    * `R_rel` : `(n_pairs, 3, 3)` -- `{}^{child}R_{parent} = {}^{W}R_c^T {}^{W}R_p`,
      the rotation that carries the parent's gyro (and its bias, and its noise)
      into the frame the measurement is written in.

    Split out from `build_stacked` because the two are separable concerns: this
    one is the only part that touches the robot model, and the tests need to
    drive `build_stacked` from injected transforms (`TEST_SUITE_MAP.md`'s
    "restructure the geometry-dependent tests around known transforms").
    """
    ev = model.evaluate(q)
    sites = jnp.asarray(model.pair_sites)
    R = ev.site_rot[sites]                                  # (n_pairs, 2, 3, 3)
    R_rel = jnp.einsum("eji,ejk->eik", R[:, 1], R[:, 0])     # R_c^T R_p
    return ev.J_rel, R_rel


# ---------------------------------------------------------------------------
# The mixing operator L -- the object invariant I6 is about
# ---------------------------------------------------------------------------

def mixing_operator(build: JointKFBuild, R_rel: Array) -> Array:
    r"""`L`, shape `(3*n_pairs, 3m)`: how per-IMU bias/noise enters the pair rows.

    Row block `e`, column block `k` is

    * `+I3`             if IMU `k` is pair `e`'s child,
    * `-{}^{c}R_{p}`    if IMU `k` is pair `e`'s parent,
    * `0`               otherwise.

    Written as a one-hot contraction rather than a scatter loop so the shape is
    static and the whole thing is one XLA op (invariant I7).  A self-pair would
    make the two terms collide; `build.py` rejects those, which is why the sum
    below needs no special case.

    This is deliberately the ONLY place the bias-column structure is written
    down.  `build_stacked` scatters this array into `H_g` and squeezes
    `L Sigma L^T` out of it, so the Java identity "the bias columns of H_g ARE
    the mixing operator" is enforced by construction: there is no second spelling
    that could drift.
    """
    m = build.n_imus
    imu = jnp.arange(m)
    child_hot = (imu[None, :] == jnp.asarray(build.pair_child)[:, None]).astype(jnp.float64)
    parent_hot = (imu[None, :] == jnp.asarray(build.pair_parent)[:, None]).astype(jnp.float64)

    eye = jnp.eye(3, dtype=jnp.float64)
    blocks = (child_hot[:, :, None, None] * eye
              - parent_hot[:, :, None, None] * R_rel[:, None, :, :])   # (P, m, 3, 3)
    return blocks.transpose(0, 2, 1, 3).reshape(3 * build.n_pairs, 3 * m)


def _block_diag_sigma(gyro_sigma: Array) -> Array:
    """`blkdiag(Sigma_0, ..., Sigma_{m-1})`, shape `(3m, 3m)`.

    Per-IMU gyro noise is genuinely independent -- separate silicon, separate
    clocks -- so the *input* is block diagonal.  Every correlation in `R_g` is
    then produced by `L`, which is the point: the correlations are forced by the
    differencing, not assumed.
    """
    m = gyro_sigma.shape[0]
    dense = jnp.einsum("kl,kij->kilj", jnp.eye(m, dtype=jnp.float64), gyro_sigma)
    return dense.reshape(3 * m, 3 * m)


# ---------------------------------------------------------------------------
# The stacked measurement
# ---------------------------------------------------------------------------

def build_stacked(
    build: JointKFBuild,
    params: JointKFParams,
    q: Array | None = None,
    gyros: Array | None = None,
    trusted_feet: Array | None = None,
    *,
    model=None,
    J_rel: Array | None = None,
    R_rel: Array | None = None,
    anchor: AnchorBlock | None = None,
) -> StackedMeasurement:
    r"""Java `buildStackedMeasurementForTest` -- the pair rows plus the anchors.

    Parameters
    ----------
    build, params
        Static structure and scalars.  `params.r_large` is the only scalar used.
    q : (n,) or (nq,), optional
        Configuration.  Only needed when `model` is given.
    gyros : (m, 3)
        Per-IMU gyro readings, each **in its own measurement frame** -- the same
        frame `b_omega` is stored in, which is what makes the child bias block
        `+I3`.
    trusted_feet : (n_anchors,) float, optional
        Previous tick's trust mask (CLAUDE.md §4 phase ordering).  Default: all
        zero, i.e. no anchors -- the configuration every ported measurement test
        uses, and the one that makes a single-pair stacked build reduce to the
        plain `3 x dim` pair measurement.
    model : RobotModel, optional
        Source of `(J_rel, R_rel)`.  Alternatively pass those two directly (the
        test seam), which keeps this function free of any model dependency.
    anchor : AnchorBlock, optional
        The `3*n_anchors` anchor rows from `anchors.py`.  Omitted => zero rows
        with `r_large` noise, i.e. structurally present and numerically inert.

    Returns
    -------
    StackedMeasurement

    Notes
    -----
    The residual is `z`, not `z - Hx`: the Java seam returns the measurement and
    the Joseph update forms the innovation, and keeping it that way is what lets
    `update.py` own the `cond(S)` / finite-mask gating for *all* channels
    uniformly.
    """
    if (J_rel is None or R_rel is None):
        if model is None:
            raise ValueError("build_stacked needs either `model` (with `q`) or both `J_rel` and `R_rel`")
        J_model, R_model = pair_frames(model, q)
        J_rel = J_model if J_rel is None else J_rel
        R_rel = R_model if R_rel is None else R_rel

    _check_float64(J_rel=J_rel, R_rel=R_rel, gyros=gyros)     # invariant I8, BEFORE any cast
    J_rel = jnp.asarray(J_rel, dtype=jnp.float64)
    R_rel = jnp.asarray(R_rel, dtype=jnp.float64)
    gyros = jnp.asarray(gyros, dtype=jnp.float64)

    n, P, K = build.n_joints, build.n_pairs, build.n_anchors
    dim = build.dim

    # -- pair rows ----------------------------------------------------------
    # S_ab is applied again here even though the model already masks: it is a
    # structural assertion (an off-path joint moves both sites identically and
    # cancels in the difference), so re-applying it costs one multiply and turns
    # a silent geometry/graph disagreement into an exact zero.
    J = J_rel * jnp.asarray(build.pair_velocity_mask, dtype=jnp.float64)[:, None, :]

    L_pair = mixing_operator(build, R_rel)                    # (3P, 3m)

    H_pair = jnp.zeros((3 * P, dim), dtype=jnp.float64)
    H_pair = H_pair.at[:, n:2 * n].set(J.reshape(3 * P, n))   # q_dot columns
    H_pair = H_pair.at[:, 2 * n:].set(L_pair)                 # THE identity (I6)

    parent, child = jnp.asarray(build.pair_parent), jnp.asarray(build.pair_child)
    z_pair = (gyros[child] - jnp.einsum("eij,ej->ei", R_rel, gyros[parent])).reshape(3 * P)

    # The per-IMU gyro noise the congruence runs on. There is deliberately NO
    # per-pair `L_pair Sigma L_pairᵀ` here: the congruence is taken once over the
    # whole stacked `L` at the end of this function, because assembling it per pair
    # loses the shared-IMU cross-covariance -- and on isotropic noise the two agree
    # to 1e-20, so no test on a single pair would ever notice the difference.
    Sigma = _block_diag_sigma(jnp.asarray(build.gyro_sigma, dtype=jnp.float64))

    # -- anchor rows: fixed shape, masked, never reshaped -------------------
    if anchor is None:
        anchor = AnchorBlock(
            H=jnp.zeros((3 * K, dim), dtype=jnp.float64),
            z=jnp.zeros(3 * K, dtype=jnp.float64),
            R=jnp.zeros((3 * K, 3 * K), dtype=jnp.float64),
        )
    trust = (jnp.zeros(K, dtype=jnp.float64) if trusted_feet is None
             else jnp.asarray(trusted_feet, dtype=jnp.float64))
    per_row = jnp.repeat(trust, 3)                            # (3K,)

    # Untrusted => residual zeroed AND R -> r_large * I3. Zeroing the rows
    # instead would make S singular (CLAUDE.md §6); r_large drives K -> 0 while
    # leaving S strictly positive definite.
    keep = per_row[:, None] * per_row[None, :]
    R_anchor = jnp.where(
        keep > 0.0,
        jnp.asarray(anchor.R, dtype=jnp.float64),
        jnp.eye(3 * K, dtype=jnp.float64) * params.r_large,
    )
    z_anchor = per_row * jnp.asarray(anchor.z, dtype=jnp.float64)
    # Mask `H` here rather than trusting the caller to have done it. `anchors.py`
    # masks too (idempotent), but the masking rule has to hold for ANY caller
    # following the seam contract -- and it is load-bearing twice over: it is what
    # zeroes the untrusted anchor's `L` rows, so the congruence below leaves that
    # block exactly `r_large * I3` with no cross-terms, which in turn is what
    # makes it structurally decoupled for `update.py`'s condition proxy.
    H_anchor = per_row[:, None] * jnp.asarray(anchor.H, dtype=jnp.float64)

    # -- stack --------------------------------------------------------------
    H = jnp.concatenate([H_pair, H_anchor], axis=0)
    z = jnp.concatenate([z_pair, z_anchor], axis=0)

    # `L` spans all rows, so `H[:, 2n:2n+3m] == L` holds for the WHOLE stacked
    # Jacobian (`testBiasColumnsOfHgAreExactlyL`).
    L = jnp.concatenate([L_pair, H_anchor[:, 2 * n:]], axis=0)

    # The congruence runs over the WHOLE stacked measurement, anchors included.
    #
    # An anchor row is written in terms of the base IMU's own *measured* rate --
    # eliminating `omega_base` between the reference's base-gyro row and its
    # absolute-rate constraint leaves `z_base = -J_leg qdot + b_base +
    # (v_anchor - v_base)`. So the anchor row inherits `Sigma_base`, AND it is
    # correlated with every pair row that touches the base IMU, through exactly
    # that shared `v_base`. Both terms fall out of `L Sigma L^T` because `L`
    # already carries the anchor's `+I3` on the base-IMU bias columns.
    #
    # Building the anchor block as a separate diagonal entry -- which is what
    # CLAUDE.md §2's `R_anchor = Sigma_eps + J_U diag(sigma^2) J_U^T` says, and
    # what Java does -- drops both terms. The stacked oracle sees it: 12/12
    # trials off by 2e-4..9e-4 against a 1e-5 tolerance. See PORT_NOTES.md.
    #
    # Masking still works: an untrusted anchor has its `H` rows zeroed, so its
    # `L` rows are zero, the congruence contributes nothing there, and the block
    # stays exactly `r_large * I3` with zero cross-terms -- which is also what
    # keeps it structurally decoupled for `update.py`'s condition proxy.
    R = L @ Sigma @ L.T
    R = R.at[build.anchor_row0:, build.anchor_row0:].add(R_anchor)

    return StackedMeasurement(H=H, z=z, R=R, L=L)


def _check_float64(**arrays) -> None:
    """Invariant I8: float64 at every filter entry point.

    Checked on the RAW argument, before any `asarray(..., float64)` -- that cast
    would happily upcast a float32 input and make the check vacuous, which is the
    "two spellings of one quantity" failure mode in miniature.  Untyped inputs
    (lists, Python floats) are left alone; only an array that already carries a
    narrower float dtype is a genuine leak.

    A float32 leak is not loud: it degrades `S`'s conditioning and surfaces
    thousands of ticks later as a covariance that will not settle.
    """
    for name, a in arrays.items():
        dtype = getattr(a, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating) and dtype != jnp.float64:
            raise TypeError(f"joint-KF measurement input `{name}` must be float64, got {dtype}")
