import time
import numpy as np
import jax, jax.numpy as jnp

from invariant_estimation.jointKF import anchors as anchors_mod, measure
from invariant_estimation.jointKF.build import build_joint_kf
from invariant_estimation.jointKF.filter import ModelInputs, SensorInputs, init_carry, run, step
from invariant_estimation.jointKF.state import default_params

import sys
sys.path.insert(0, "tests")
from jointKF import _fixture as fx
from jointKF._fixture import kinematic_tree

SHAPE = {"name": "traj_n6_m2", "chain": 8, "imus": (1, 7), "pairs": ((0, 1),), "n": 6, "m": 2}

t0 = time.time()
f = fx.build_fixture(SHAPE)
print("fixture", time.time() - t0, f.n, f.m)

tree = kinematic_tree(f)
build = build_joint_kf(tree, imu_sites=list(f.imu_names), pairs=[tuple(p) for p in f.pairs],
                       foot_sites=[f.foot_site], use_mass_matrix=False,
                       gyro_sigma=lambda nm: 1.0e-4*np.eye(3))
params = default_params()
print("dim", build.dim, "K", build.n_anchors, "n_u", build.anchor_unfiltered_mask.shape)

names = list(f.model.site_names)
base_site = names.index(f.imu_names[build.base_imu])
foot_sites = np.array([names.index(f.foot_site)])


def model_inputs(q):
    q = jnp.asarray(q, dtype=jnp.float64)
    ev = f.model.evaluate(q)
    sites = jnp.asarray(f.model.pair_sites)
    R = ev.site_rot[sites]
    R_rel = jnp.einsum("eji,ejk->eik", R[:, 1], R[:, 0])
    jac = anchors_mod.anchor_jacobians(build, ev.J_ang, ev.site_rot,
                                       base_site=base_site, foot_sites=foot_sites)
    return ModelInputs(J_rel=ev.J_rel, R_rel=R_rel, anchor_jac=jac, M=None)


t0 = time.time()
m0 = model_inputs(np.zeros(f.n))
print("one eval", time.time() - t0)

T = 3000
AMP, OMEGA, DT = 0.10, 2 * np.pi * 0.5, 1.0e-3
tick = np.arange(T)[:, None]
i = np.arange(f.n)[None, :]
phase = OMEGA * tick * DT + i * np.pi / f.n
q_traj = AMP * np.sin(phase)
qd_traj = AMP * OMEGA * np.cos(phase)

t0 = time.time()
models = jax.vmap(model_inputs)(jnp.asarray(q_traj))
jax.block_until_ready(models.J_rel)
print("vmap eval T=3000", time.time() - t0)

motions = [f.apply_consistent_motion(q_traj[t], qd_traj[t]) for t in range(T)]
gyros = np.stack([m.gyro for m in motions])
n_u = build.anchor_unfiltered_mask.shape[1]
sensors = SensorInputs(
    encoders=jnp.asarray(q_traj),
    gyros=jnp.asarray(gyros),
    qd_unfiltered=jnp.zeros((T, n_u)),
    contact=jnp.zeros((T, build.n_anchors)),
)

carry = init_carry(build, params, jnp.asarray(q_traj[0]))


def scan_x(carry, sensors, models):
    def body(c, xs):
        s, mdl = xs
        c, d = step(c, s, mdl, build, params)
        return c, (c.state.x, jnp.trace(c.state.P))
    return jax.lax.scan(body, carry, (sensors, models))


t0 = time.time()
final, (xs, tr) = jax.jit(scan_x)(carry, sensors, models)
jax.block_until_ready(xs)
print("scan 3000", time.time() - t0)
xs = np.asarray(xs)
n = f.n
print("pos err final", np.abs(xs[-1, :n] - q_traj[-1]))
print("vel err final", np.abs(xs[-1, n:2 * n] - qd_traj[-1]))
maxv = np.max(np.abs(xs[500:, n:2 * n]), axis=0)
print("maxv", maxv, "0.5*peak", 0.5 * AMP * OMEGA)
print("bias norms", np.linalg.norm(xs[-1, 2 * n:].reshape(-1, 3), axis=1))
print("trace warm/final", tr[200], tr[-1])
