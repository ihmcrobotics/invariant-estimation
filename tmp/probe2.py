import time
import numpy as np
import jax, jax.numpy as jnp

from invariant_estimation.jointKF import anchors as anchors_mod
from invariant_estimation.jointKF.build import build_joint_kf
from invariant_estimation.jointKF.filter import ModelInputs, SensorInputs, init_carry, step
from invariant_estimation.jointKF.state import default_params

import sys
sys.path.insert(0, "tests")
from jointKF import _fixture as fx
from jointKF._fixture import kinematic_tree

SHAPE = {"name": "traj_n6_m2", "chain": 8, "imus": (1, 7), "pairs": ((0, 1),), "n": 6, "m": 2}
f = fx.build_fixture(SHAPE)
tree = kinematic_tree(f)
build = build_joint_kf(tree, imu_sites=list(f.imu_names), pairs=[tuple(p) for p in f.pairs],
                       foot_sites=[f.foot_site], use_mass_matrix=False, gyro_sigma=lambda nm: 1.0e-4*np.eye(3))
params = default_params()
names = list(f.model.site_names)
base_site = names.index(f.imu_names[build.base_imu])
foot_sites = np.array([names.index(f.foot_site)])
n = f.n


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
mdl = model_inputs(np.zeros(n))
print("eval", time.time() - t0)

OFFSET = np.array([0.01, -0.02, 0.03])
n_u = build.anchor_unfiltered_mask.shape[1]
sens = SensorInputs(
    encoders=jnp.zeros(n),
    gyros=jnp.tile(jnp.asarray(OFFSET), (f.m, 1)),
    qd_unfiltered=jnp.zeros(n_u),
    contact=jnp.ones(build.n_anchors),
)


def roll(carry, sensors, T):
    def body(c, _):
        c, d = step(c, sensors, mdl, build, params)
        return c, None
    return jax.lax.scan(body, carry, None, length=T)[0]


carry = init_carry(build, params, jnp.zeros(n))
t0 = time.time()
final = jax.jit(roll, static_argnums=2)(carry, sens, 20000)
jax.block_until_ready(final.state.x)
print("20k scan", time.time() - t0)
x = np.asarray(final.state.x)
b = x[2 * n:].reshape(-1, 3)
print("bias base", b[build.base_imu], "err", np.abs(b[build.base_imu] - OFFSET))
print("bias other", b)
print("q", x[:n], "qd", x[n:2 * n])
print("trace", float(jnp.trace(final.state.P)))

# encoder tracking, 100 ticks, targets
target = 0.2 * (np.arange(n) + 1) - 0.5
sens2 = sens._replace(encoders=jnp.asarray(target), gyros=jnp.zeros((f.m, 3)))
carry = init_carry(build, params, jnp.asarray(target))
final2 = jax.jit(roll, static_argnums=2)(carry, sens2, 100)
print("enc err", np.abs(np.asarray(final2.state.x)[:n] - target))

# covariance bounded, 2000 ticks
carry = init_carry(build, params, jnp.zeros(n))
tr0 = float(jnp.trace(carry.state.P))
final3 = jax.jit(roll, static_argnums=2)(carry, sens, 2000)
tr1 = float(jnp.trace(final3.state.P))
print("trace", tr0, tr1, tr0 * 10 + 1)

# --- from-zero encoder convergence curve ---
def roll_hist(carry, sensors, T):
    def body(c, _):
        c, d = step(c, sensors, mdl, build, params)
        return c, c.state.x
    return jax.lax.scan(body, carry, None, length=T)

carry = init_carry(build, params, jnp.zeros(n))
_, xs = jax.jit(roll_hist, static_argnums=2)(carry, sens2, 400)
xs = np.asarray(xs)
err = np.max(np.abs(xs[:, :n] - target), axis=1)
print("from-zero max err at ticks", [(t, float(err[t])) for t in [0,9,49,99,199,299,399]])
print("start err", np.max(np.abs(target)))
