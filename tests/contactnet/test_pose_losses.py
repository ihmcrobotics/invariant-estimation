"""Property tests for the composite pose losses (l2_position, so3_log_orientation,
pose_l2) added for the L2-options ablation.

The load-bearing guarantees:
  * arm 1 (`use_pos=use_ori=False`) reproduces `l2_velocity` bit-for-bit;
  * each term is zero at truth, non-negative, and batched-shape safe (the take-two
    regression where a non-batched einsum raised on the (L, ...) segment call);
  * the segment-relative terms have the closed forms the plan claims
    (orientation = geodesic θ²; position = displacement, offset-invariant);
  * gradients through the SO(3) log are finite near identity (small-angle path).
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.inEKF.group import Gamma0
from invariant_estimation.contactnet.losses import (
    l2_velocity, l2_position, so3_log_orientation, pose_l2)

L = 8
DT = 1.0e-3


def _rot(phi):
    """(..., 3) rotation vectors -> (..., 3, 3) rotation matrices (repo Rodrigues)."""
    flat = phi.reshape(-1, 3)
    R = jax.vmap(Gamma0)(flat)
    return R.reshape(phi.shape[:-1] + (3, 3))


def _rand_scene(seed, batch=None):
    """Random (v, R, p) est/true tensors, shape (L, 3)/(L, 3, 3) or (B, L, ...)."""
    rng = np.random.default_rng(seed)
    shp = (L,) if batch is None else (batch, L)
    mk = lambda: jnp.asarray(rng.standard_normal(shp + (3,)))
    mkR = lambda: _rot(jnp.asarray(0.3 * rng.standard_normal(shp + (3,))))
    return (mk(), mkR(), mk(), mk(), mkR(), mk())


def test_zero_at_truth():
    v, R, p, *_ = _rand_scene(0)
    assert float(l2_velocity(v, R, v, R)) == pytest.approx(0.0, abs=1e-12)
    assert float(l2_position(p, p)) == pytest.approx(0.0, abs=1e-12)
    assert float(so3_log_orientation(R, R)) == pytest.approx(0.0, abs=1e-12)
    assert float(pose_l2(v, R, p, v, R, p, w_pos=1.0, w_ori=1.0,
                         use_pos=True, use_ori=True)) == pytest.approx(0.0, abs=1e-12)


def test_non_negative():
    for s in range(5):
        ve, Re, pe, vt, Rt, pt = _rand_scene(s)
        assert float(l2_position(pe, pt)) >= 0.0
        assert float(so3_log_orientation(Re, Rt)) >= 0.0
        assert float(pose_l2(ve, Re, pe, vt, Rt, pt, 1.0, 1.0, True, True)) >= 0.0


def test_reduces_to_l2_velocity():
    """use_pos=use_ori=False MUST equal l2_velocity regardless of weights (arm 1)."""
    ve, Re, pe, vt, Rt, pt = _rand_scene(1)
    base = l2_velocity(ve, Re, vt, Rt)
    got = pose_l2(ve, Re, pe, vt, Rt, pt, w_pos=1e6, w_ori=1e6,
                  use_pos=False, use_ori=False)
    assert float(got) == float(base)  # bit-for-bit


def test_batched_shape_contract():
    """Terms run on (L, ...) and (B, L, ...); a batch of identical scenes equals
    the single-scene value (the reduction means over batch too)."""
    ve, Re, pe, vt, Rt, pt = _rand_scene(2)
    single_p = float(l2_position(pe, pt))
    single_o = float(so3_log_orientation(Re, Rt))
    stack = lambda a: jnp.broadcast_to(a, (3,) + a.shape)
    assert float(l2_position(stack(pe), stack(pt))) == pytest.approx(single_p, rel=1e-9)
    assert float(so3_log_orientation(stack(Re), stack(Rt))) == pytest.approx(single_o, rel=1e-9)
    # genuinely batched (distinct scenes) must not raise and returns a scalar
    veb, Reb, peb, vtb, Rtb, ptb = _rand_scene(3, batch=4)
    assert jnp.ndim(pose_l2(veb, Reb, peb, vtb, Rtb, ptb, 1.0, 1.0, True, True)) == 0


def test_orientation_absolute_oracle():
    """relative=False: R_true = R_est @ exp(θ a) constant  =>  L_ori = θ²."""
    rng = np.random.default_rng(4)
    Re = _rot(jnp.asarray(0.4 * rng.standard_normal((L, 3))))
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    theta = 0.37
    C = _rot(jnp.asarray(theta * axis)[None])[0]           # (3,3)
    Rt = jnp.einsum("kij,jl->kil", Re, C)                  # R_est @ C
    got = float(so3_log_orientation(Re, Rt, relative=False))
    assert got == pytest.approx(theta ** 2, rel=1e-6)


def test_orientation_relative_oracle():
    """relative=True with R_est,0 = R_true,0 = I and R_true,k = R_est,k @ exp(θ a):
    L_ori = θ² (L-1)/L (the k=0 tick contributes exactly zero)."""
    rng = np.random.default_rng(5)
    Re = _rot(jnp.asarray(0.4 * rng.standard_normal((L, 3))))
    Re = Re.at[0].set(jnp.eye(3))
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    theta = 0.29
    C = _rot(jnp.asarray(theta * axis)[None])[0]
    Rt = jnp.einsum("kij,jl->kil", Re, C).at[0].set(jnp.eye(3))
    got = float(so3_log_orientation(Re, Rt, relative=True))
    assert got == pytest.approx(theta ** 2 * (L - 1) / L, rel=1e-6)


def test_position_relative_oracle_and_offset_invariance():
    """relative=True: p_est - p_true = c + δ·k  =>  Δ-error = δ·k, so
    L_pos = ‖δ‖² · mean_k k², independent of the constant offset c."""
    rng = np.random.default_rng(6)
    pt = jnp.asarray(rng.standard_normal((L, 3)))
    delta = np.array([0.02, -0.01, 0.03])
    k = np.arange(L)[:, None]
    c1 = np.array([0.5, -0.2, 0.1])
    pe1 = pt + jnp.asarray(c1 + delta * k)
    expect = float(np.sum(delta ** 2) * np.mean((np.arange(L)) ** 2))
    assert float(l2_position(pe1, pt, relative=True)) == pytest.approx(expect, rel=1e-9)
    # a different constant offset leaves the segment-relative loss unchanged...
    pe2 = pt + jnp.asarray(np.array([-3.0, 4.0, 5.0]) + delta * k)
    assert float(l2_position(pe2, pt, relative=True)) == pytest.approx(expect, rel=1e-9)
    # ...but the absolute loss does depend on the offset
    assert float(l2_position(pe1, pt, relative=False)) != pytest.approx(
        float(l2_position(pe2, pt, relative=False)), rel=1e-6)


def test_gradients_finite():
    """grad of each term w.r.t. its est input is finite, incl. the SO(3) log's
    small-angle branch (near-identity relative rotations)."""
    ve, Re, pe, vt, Rt, pt = _rand_scene(7)
    gp = jax.grad(lambda p: l2_position(p, pt))(pe)
    assert bool(jnp.all(jnp.isfinite(gp)))
    # near-identity: R_est ~ R_true so ΔR ~ I (small-angle path of so3_log)
    Rt_near = _rot(jnp.asarray(1e-5 * np.random.default_rng(8).standard_normal((L, 3))))
    go = jax.grad(lambda R: so3_log_orientation(R, Rt_near))(Rt_near)
    assert bool(jnp.all(jnp.isfinite(go)))
    gc = jax.grad(lambda v: pose_l2(v, Re, pe, vt, Rt, pt, 0.5, 0.5, True, True))(ve)
    assert bool(jnp.all(jnp.isfinite(gc)))
