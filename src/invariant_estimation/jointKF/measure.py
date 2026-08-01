"""Measurement side of the joint-space KF: encoder rows + the **stacked
relative-gyro rows** over the IMU-pair graph.

Paper §II, Java `JointLevelKFPreFilter.buildStackedMeasurementForTest`.

For an IMU pair `(a, b)` the filter measures a *difference*::

    z_e = omega_b^b - {}^{b}R_{a} omega_a^a
        = J_ang(q) S_ab qdot  +  b_b  -  {}^{b}R_{a} b_a  +  ( v_b - {}^{b}R_{a} v_a )

Two facts follow, and they are the whole module (invariant I6):

1. The bias columns of `H_g` are a linear operator `L` `((3*n_pairs), 3m)` on the
   per-IMU bias vector -- `+I3` on the child, `-{}^{b}R_{a}` on the parent.  It is
   built ONCE, by `mixing_operator`, and *scattered into* `H_g`, so
   `H_g[:, 2n:2n+3m] == L` holds bit-identically by construction rather than by
   coincidence (`testBiasColumnsOfHgAreExactlyL`, tol 0.0).

2. The same `L` mixes the noise: `R_g = L Sigma L^T` exactly, with
   `Sigma = blkdiag(Sigma_imu)`.  A block-diagonal per-pair `R_g` is WRONG on a
   shared-IMU star -- two pairs sharing an IMU inherit that IMU's noise with
   opposite signs, and `L Sigma L^T` produces exactly the resulting off-diagonal
   block.  The trap is close to invisible: for isotropic `Sigma = sigma^2 I`,
   `R Sigma R^T = sigma^2 I` **exactly**, so on a single pair the block-diagonal
   form is an identity, not an approximation (measured deviation 2.6e-20).  Only
   anisotropic `Sigma` or a shared-IMU star can tell them apart -- see
   `PORT_NOTES.md`, "I6 is invisible on three of the four shapes".

`Sigma` is the gyro **MEASUREMENT** covariance, never the bias random-walk
covariance (`testMeasurementNoiseUsesGyroMeasurementCovariance`, and
`testMeasurementNoiseIndependentOfBiasProcessCovariance` at tol 0.0).  The bias
random walk is a *process* noise living in `process.py`; the two are numerically
far apart (1e-4 vs 1e-9 in the Java fixture) so mixing them up is a quiet 5-order
mis-weighting rather than a crash.  `build.gyro_sigma` is already floored at
build time -- do not floor again per tick (I7: the decision is not data
dependent).

`b_omega` lives in **each IMU's own measurement (site) frame**, which is what
makes the child block exactly `+I3` and the parent block exactly `-{}^{b}R_{a}`.
Verified, not assumed, by `tests/jointKF/test_measurement.py::
test_stacked_pair_rows_match_the_marginalized_raw_gyro_reference`, which puts
`+I3` on each IMU's bias in that IMU's own frame, carries the shared unknown
`omega_base` on a rotation column, and marginalises it out.

Shapes are constant (I7): `build_stacked` always returns
`3*(n_pairs + n_anchors)` rows.  An untrusted anchor keeps its rows, with the
residual zeroed and the `R` block set to `params.r_large * I3`, never to zero (a
zero `R` block makes `S` singular -- CLAUDE.md §4).  `anchors.py` builds the
anchor block and passes it in, so the two are developed against the row-layout
contract (`build.anchor_row0`, `build.n_stacked_rows`), not each other's code.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from .state import JointKFBuild, JointKFParams

__all__ = ["StackedMeasurement", "AnchorBlock", "encoder_jacobian", "encoder_noise",
           "pair_frames", "mixing_operator", "build_stacked"]


class AnchorBlock(NamedTuple):
    """Minimal structural stand-in for the anchor rows `anchors.py` produces.

    Always `3*n_anchors` rows -- a foot landing flips `trusted_feet`, never a
    dimension (invariant I2).  `R` is the *trusted* anchor noise; `build_stacked`
    swaps in `r_large * I3` for the untrusted rows, so `anchors.py` never has to
    know the masking convention.

    Not the only accepted type: `build_stacked` reads `.H`, `.z`, `.R` and nothing
    else, so `anchors.AnchorBlock` (which also carries `active` / `n_active`)
    drops straight in.  `anchors.py` masks on its side too; the two compose
    because the masking is idempotent.
    """

    H: Array   # (3*n_anchors, dim)
    z: Array   # (3*n_anchors,)
    R: Array   # (3*n_anchors, 3*n_anchors)


class StackedMeasurement(NamedTuple):
    """One tick's stacked gyro + anchor measurement.

    Java seams `getStackedMeasurementJacobian/Residual/Noise` and
    `getMixingOperator`.  `z` is the measurement itself, not the innovation
    (`z_e = omega_child - R_{child<-parent} omega_parent`); the filter forms
    `nu = z - H x`.
    """

    H: Array   # (n_stacked_rows, dim)  rows [3e,3e+3) are pair e; anchor_row0 on
    z: Array   # (n_stacked_rows,)      are the anchors
    R: Array   # (n_stacked_rows, n_stacked_rows)  L Sigma L^T over the WHOLE stack
    L: Array   # (n_stacked_rows, 3m)   bit-identical to H[:, 2n:2n+3m]


def encoder_jacobian(build: JointKFBuild, params: JointKFParams | None = None) -> Array:
    """`H_enc = [I_n | 0]`, shape `(n, dim)` -- Java `getEncoderJacobian`.

    Encoders observe `q` and nothing else: not `q_dot` (inferred through the
    process model, never measured on this channel) and not bias.  Locked by
    `testEncoderJacobianStructure` and `testEncoderPredictsPosition`.

    `params` is accepted and unused so both call styles in the ported suite work;
    nothing about the encoder Jacobian is tunable.
    """
    n = build.n_joints
    return jnp.eye(n, build.dim, dtype=jnp.float64)


def encoder_noise(build: JointKFBuild, params: JointKFParams | None = None) -> Array:
    """`R_enc = diag(build.encoder_var)`, shape `(n, n)` -- Java `getEncoderNoise`.

    Strictly diagonal: encoder errors on separate joints share no mechanism, and
    the per-joint variances come from `build` (name-resolved, with the loud 5e-5
    fallback) rather than a single scalar -- invariant I9, and the wiring
    `JointLevelKFEncoderNISConsistencyTest` locks in.

    `params` is accepted for call-style compatibility and deliberately NOT read:
    `params.encoder_var` is only the scalar fallback, which `build.py` has already
    applied per joint.  Reading it here would silently re-uniformise the
    variances -- exactly the invariant I9 failure.
    """
    return jnp.diag(jnp.asarray(build.encoder_var, dtype=jnp.float64))


def pair_frames(model, q: Array) -> tuple[Array, Array]:
    r"""`(J_rel, R_rel)` for every pair, from ONE position-level model pass.

    `J_rel` `(n_pairs, 3, n)` is `J_ang(q) S_ab` in the **child** frame; `R_rel`
    `(n_pairs, 3, 3)` is `{}^{child}R_{parent} = {}^{W}R_c^T {}^{W}R_p`, the
    rotation carrying the parent's gyro (and its bias, and its noise) into the
    frame the measurement is written in.

    Split out from `build_stacked` because this is the only part that touches the
    robot model, and the tests drive `build_stacked` from injected transforms.
    """
    ev = model.evaluate(q)
    sites = jnp.asarray(model.pair_sites)
    R = ev.site_rot[sites]                                  # (n_pairs, 2, 3, 3)
    R_rel = jnp.einsum("eji,ejk->eik", R[:, 1], R[:, 0])     # R_c^T R_p
    return ev.J_rel, R_rel


def mixing_operator(build: JointKFBuild, R_rel: Array) -> Array:
    r"""`L`, shape `(3*n_pairs, 3m)`: how per-IMU bias/noise enters the pair rows.

    Row block `e`, column block `k` is `+I3` if IMU `k` is pair `e`'s child,
    `-{}^{c}R_{p}` if it is the parent, `0` otherwise.  Written as a one-hot
    contraction rather than a scatter loop so the shape is static and it is one
    XLA op (I7).  A self-pair would make the two terms collide; `build.py` rejects
    those, so the sum needs no special case.

    Deliberately the ONLY place the bias-column structure is written down:
    `build_stacked` scatters this array into `H_g` and squeezes `L Sigma L^T` out
    of it, so "the bias columns of H_g ARE the mixing operator" is enforced by
    construction, with no second spelling that could drift.
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
    then produced by `L`: the correlations are forced by the differencing, not
    assumed.
    """
    m = gyro_sigma.shape[0]
    dense = jnp.einsum("kl,kij->kilj", jnp.eye(m, dtype=jnp.float64), gyro_sigma)
    return dense.reshape(3 * m, 3 * m)


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

    `gyros` `(m, 3)` are per-IMU readings each **in its own measurement frame**,
    the frame `b_omega` is stored in, which is what makes the child bias block
    `+I3`.  `trusted_feet` `(n_anchors,)` is the PREVIOUS tick's trust mask
    (CLAUDE.md §4 phase ordering); the default of all-zero is what every ported
    measurement test uses, and makes a single-pair stacked build reduce to the
    plain `3 x dim` pair measurement.  Pass `model` (with `q`) or `J_rel`/`R_rel`
    directly (the test seam, keeping this free of any model dependency).  An
    omitted `anchor` gives zero rows with `r_large` noise: structurally present,
    numerically inert.

    Returns `z`, not `z - Hx`: the Java seam returns the measurement and the
    Joseph update forms the innovation, which is what lets `update.py` own the
    `cond(S)` / finite-mask gating for *all* channels uniformly.
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

    n, m, P, K = build.n_joints, build.n_imus, build.n_pairs, build.n_anchors
    dim, rows = build.dim, build.n_stacked_rows

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

    # The exact congruence. Written as one triple product on purpose: assembling
    # it per pair loses the shared-IMU cross-covariance, and on isotropic noise
    # the two agree to 1e-20, so no test on a single pair would ever notice.
    Sigma = _block_diag_sigma(jnp.asarray(build.gyro_sigma, dtype=jnp.float64))
    R_pair = L_pair @ Sigma @ L_pair.T

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
    # instead would make S singular (CLAUDE.md §4); r_large drives K -> 0 while
    # leaving S strictly positive definite.
    keep = per_row[:, None] * per_row[None, :]
    R_anchor = jnp.where(
        keep > 0.0,
        jnp.asarray(anchor.R, dtype=jnp.float64),
        jnp.eye(3 * K, dtype=jnp.float64) * params.r_large,
    )
    z_anchor = per_row * jnp.asarray(anchor.z, dtype=jnp.float64)
    # Masked here rather than trusting the caller (`anchors.py` masks too, and it
    # is idempotent): this is what zeroes the untrusted anchor's `L` rows, so the
    # congruence below leaves that block exactly `r_large * I3` with no
    # cross-terms -- which is what makes it structurally decoupled for
    # `update.py`'s condition proxy.
    H_anchor = per_row[:, None] * jnp.asarray(anchor.H, dtype=jnp.float64)

    # -- stack --------------------------------------------------------------
    H = jnp.concatenate([H_pair, H_anchor], axis=0)
    z = jnp.concatenate([z_pair, z_anchor], axis=0)

    # `L` spans all rows, so `H[:, 2n:2n+3m] == L` holds for the WHOLE stacked
    # Jacobian (`testBiasColumnsOfHgAreExactlyL`).
    L = jnp.concatenate([L_pair, H_anchor[:, 2 * n:]], axis=0)

    # The congruence runs over the WHOLE stacked measurement, anchors included --
    # this port deliberately does NOT follow Java here.
    #
    # An anchor row is written in terms of the base IMU's own *measured* rate:
    # eliminating `omega_base` between the reference's base-gyro row and its
    # absolute-rate constraint leaves `z_base = -J_leg qdot + b_base +
    # (v_anchor - v_base)`. So the anchor row inherits `Sigma_base`, AND it is
    # correlated with every pair row touching the base IMU through that shared
    # `v_base`. Both terms fall out of `L Sigma L^T` because `L` already carries
    # the anchor's `+I3` on the base-IMU bias columns.
    #
    # Building the anchor block as a separate diagonal entry -- what CLAUDE.md
    # §2's `R_anchor = Sigma_eps + J_U diag(sigma^2) J_U^T` says, and what Java
    # does -- drops both terms. The stacked oracle sees it: 12/12 trials off by
    # 2e-4..9e-4 against a 1e-5 tolerance. See PORT_NOTES.md.
    #
    # Masking still works: an untrusted anchor's `H` rows are zeroed, so its `L`
    # rows are zero, the congruence contributes nothing, and the block stays
    # exactly `r_large * I3` with zero cross-terms.
    R = L @ Sigma @ L.T
    R = R.at[build.anchor_row0:, build.anchor_row0:].add(R_anchor)

    return StackedMeasurement(H=H, z=z, R=R, L=L)


def _check_float64(**arrays) -> None:
    """Invariant I8: float64 at every filter entry point.

    Checked on the RAW argument, before any `asarray(..., float64)` -- that cast
    would upcast a float32 input and make the check vacuous.  Untyped inputs
    (lists, Python floats) are left alone; only an array already carrying a
    narrower float dtype is a genuine leak.  A float32 leak is not loud: it
    degrades `S`'s conditioning and surfaces thousands of ticks later as a
    covariance that will not settle.
    """
    for name, a in arrays.items():
        dtype = getattr(a, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating) and dtype != jnp.float64:
            raise TypeError(f"joint-KF measurement input `{name}` must be float64, got {dtype}")
