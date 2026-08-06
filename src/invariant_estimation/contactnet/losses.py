import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from ..inEKF.group import so3_log

def l2_velocity(
    v_est: Array,
    R_est: Array,
    v_true: Array,
    R_true: Array, #TODO: should this be `jaxlie.SO3`?
) -> Array:
    """CoCo's objective (run 1): mean_k || R_est[k]^T v_est[k] - R_true[k]^T v_true[k] ||^2.

    Both velocities arrive in WORLD frame; each is rotated into its OWN body frame,
    so a correct velocity paired with a WRONG attitude still produces loss (attitude
    error reaches ContactNet's gradient). `rollout.make_segment_loss` calls this on
    (L, ...) segment arrays, so the einsum/reduction MUST be batched -- '...ji,...j'
    and mean-over-time of the per-tick squared error, matching the validated
    reference losses.py. (take-two shipped a non-batched 'ji,j' + linalg.norm that
    raised on the batched call.)
    """
    body_estimated = jnp.einsum("...ji,...j->...i", R_est, v_est)
    body_true = jnp.einsum("...ji,...j->...i", R_true, v_true)
    return jnp.mean(jnp.sum((body_estimated - body_true) ** 2, axis=-1))

def beta_nll_from_diagnostics(
    nis,
    logdet_S,
    applied,
    beta,
    dof
):
    """
    Per-tick Gaussian innovation NLL, beta-weighted (Seitzer et al. 2022, "On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks"). 
    This is averaged over applied+finite ticks, with NIS and the logdet_S defined as:
        NIS = nu^T S^{-1} nu, logdet_S = log(det(S)), where nu is the innovation and S is the innovation covariance.
    These are both from the same cholesky in linear_update, with DOF = 3 * N_c
    """
    finite = (applied > 0) & jnp.isfinite(nis) & jnp.isfinite(logdet_S)
    nis_c = jnp.where(finite, nis, 0.0)
    logdet_c = jnp.where(finite, logdet_S, 0.0)

    per_tick = 0.5 * (nis_c + logdet_c)
    weight = jax.lax.stop_gradient(
        jnp.exp((beta / dof * logdet_c))
    )
    term = jnp.where(finite, weight * per_tick, 0.0)

    denom = jnp.maximum(jnp.sum(finite), 1.0)
    return jnp.sum(term) / denom


def l2_position(
    p_est: Array,
    p_true: Array,
    relative: bool = True,
) -> Array:
    """Position (displacement) L2, the analog of `l2_velocity` for base position.

    ``relative=True`` (default) compares the displacement FROM THE SEGMENT START,

        Δp_k = p_k − p_0,   L_pos = mean_k ‖Δp_est,k − Δp_true,k‖²,

    which removes the accumulated offset the filter carries into a *chained* segment
    (the batcher starts a segment from the drifted carry, not from truth). Base
    world position is UNOBSERVABLE in this InEKF — contact FK constrains only
    relative motion — so the *absolute* error grows without bound over an episode
    and would swamp the velocity term; the segment-relative displacement is bounded
    and equals ≈ dt·Σ(v_est − v_true), an integral-of-velocity-error signal that
    weights sustained/low-frequency velocity bias. ``relative=False`` returns the
    raw absolute error.

    Both positions are WORLD frame, ``(L, 3)`` per segment with axis 0 = time (the
    leading ``...`` keeps it correct under the ``make_batch_loss`` vmap). Reduction
    is mean-over-time of the per-tick squared error, matching `l2_velocity`.
    """
    if relative:
        p_est = p_est - p_est[..., 0:1, :]
        p_true = p_true - p_true[..., 0:1, :]
    return jnp.mean(jnp.sum((p_est - p_true) ** 2, axis=-1))


def so3_log_orientation(
    R_est: Array,
    R_true: Array,
    relative: bool = True,
) -> Array:
    r"""Orientation error as the SO(3) log, ``mean_k ‖Log(·)^∨‖²`` (rad²).

    The proper log-map (rotation-vector) parameterisation Lucas asked for, not a
    chordal/Frobenius surrogate.

    ``relative=True`` (default) measures the INCREMENTAL rotation over the segment,

        ΔR_k = R_0ᵀ R_k,   err_k = Log(ΔR_est,kᵀ ΔR_true,k)^∨,

    removing the accumulated attitude offset at the segment start. Yaw is
    unobservable here (``enableYawSeeding=false``), so absolute attitude error
    drifts over an episode; the incremental form is bounded. ``relative=False``
    returns the absolute ``Log(R_est,kᵀ R_true,k)^∨`` — exactly ``log(R_estᵀ R_true)``.

    ``R_est, R_true`` are ``(L, 3, 3)`` per segment, axis 0 = time (leading ``...``
    broadcasts under the vmap). ``einsum`` ``...ji,...jk->...ik`` is ``AᵀB``.
    """
    if relative:
        R0e = R_est[..., 0:1, :, :]
        R0t = R_true[..., 0:1, :, :]
        dR_est = jnp.einsum("...ji,...jk->...ik", R0e, R_est)    # R_est,0ᵀ R_est,k
        dR_true = jnp.einsum("...ji,...jk->...ik", R0t, R_true)  # R_true,0ᵀ R_true,k
        rel = jnp.einsum("...ji,...jk->...ik", dR_est, dR_true)  # ΔR_est,kᵀ ΔR_true,k
    else:
        rel = jnp.einsum("...ji,...jk->...ik", R_est, R_true)    # R_est,kᵀ R_true,k
    xi = so3_log(rel)                                            # (..., 3)
    return jnp.mean(jnp.sum(xi ** 2, axis=-1))


def pose_l2(
    v_est: Array,
    R_est: Array,
    p_est: Array,
    v_true: Array,
    R_true: Array,
    p_true: Array,
    w_pos: float = 0.0,
    w_ori: float = 0.0,
    use_pos: bool = False,
    use_ori: bool = False,
    relative: bool = True,
) -> Array:
    """Composite objective ``L_vel + w_pos·L_pos + w_ori·L_ori``.

    ``L_vel`` is the shipped body-frame velocity MSE, unchanged — so
    ``use_pos=use_ori=False`` returns `l2_velocity` bit-for-bit (arm 1 ≡ R3b).
    ``L_pos`` / ``L_ori`` are the segment-relative (`relative=True`) terms above,
    added only when their flag is set. ``use_pos``/``use_ori``/``relative`` are
    static (closed over at build time — no data-dependent branch in the graph);
    ``w_pos``/``w_ori`` are the run-frozen scalar weights (sized once upstream).
    Inactive terms are not computed, so each arm pays only for the terms it uses.
    """
    loss = l2_velocity(v_est, R_est, v_true, R_true)
    if use_pos:
        loss = loss + w_pos * l2_position(p_est, p_true, relative=relative)
    if use_ori:
        loss = loss + w_ori * so3_log_orientation(R_est, R_true, relative=relative)
    return loss
