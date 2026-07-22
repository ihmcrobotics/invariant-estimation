import numpy as np, jax, jax.numpy as jnp, sys, time
from invariant_estimation.jointKF import anchors as anchors_mod
from invariant_estimation.jointKF.build import build_joint_kf
from invariant_estimation.jointKF.filter import ModelInputs, SensorInputs, init_carry, step
from invariant_estimation.jointKF.state import default_params
sys.path.insert(0, "tests")
from jointKF import _fixture as fx
from jointKF._fixture import kinematic_tree

SHAPE = {"name": "traj_n6_m2", "chain": 8, "imus": (1, 7), "pairs": ((0, 1),), "n": 6, "m": 2}
f = fx.build_fixture(SHAPE)
tree = kinematic_tree(f)
build = build_joint_kf(tree, imu_sites=list(f.imu_names), pairs=[tuple(p) for p in f.pairs],
                       foot_sites=[f.foot_site], use_mass_matrix=False,
                       gyro_sigma=lambda nm: 1.0e-4 * np.eye(3))
params = default_params()
names = list(f.model.site_names)
base_site = names.index(f.imu_names[build.base_imu])
foot_sites = np.array([names.index(f.foot_site)])
n, m = f.n, f.m
AMP, OMEGA, DT = 0.10, 2 * np.pi * 0.5, 1.0e-3


def model_inputs(q):
    q = jnp.asarray(q, dtype=jnp.float64)
    ev = f.model.evaluate(q)
    R = ev.site_rot[jnp.asarray(f.model.pair_sites)]
    R_rel = jnp.einsum("eji,ejk->eik", R[:, 1], R[:, 0])
    jac = anchors_mod.anchor_jacobians(build, ev.J_ang, ev.site_rot,
                                       base_site=base_site, foot_sites=foot_sites)
    return ModelInputs(J_rel=ev.J_rel, R_rel=R_rel, anchor_jac=jac, M=None)


def traj(ticks):
    t = np.asarray(ticks, float)[:, None]
    i = np.arange(n)[None, :]
    ph = OMEGA * t * DT + i * np.pi / n
    return AMP * np.sin(ph), AMP * OMEGA * np.cos(ph)


def sensors_for(q, qd):
    T = len(q)
    gy = np.stack([f.apply_consistent_motion(q[t], qd[t]).gyro for t in range(T)])
    return SensorInputs(encoders=jnp.asarray(q), gyros=jnp.asarray(gy),
                        qd_unfiltered=jnp.zeros((T, 0)), contact=jnp.zeros((T, 1)))


def scan(carry, sens, mdls):
    def body(c, xs):
        s, mm = xs
        c, d = step(c, s, mm, build, params)
        return c, (c.state.x, c.state.P)
    return jax.lax.scan(body, carry, (sens, mdls))


# ---- coupling: one predict from rest
q0, _ = traj([0])
zero = np.zeros((1, n))
mdl0 = jax.vmap(model_inputs)(jnp.asarray(q0))
s0 = sensors_for(q0, zero)
carry = init_carry(build, params, jnp.asarray(q0[0]))
# one *predict only* -> use the internal pieces: run one step but with gated updates?
from invariant_estimation.jointKF import predict as predict_mod, process
F = predict_mod.build_transition(build, params)
Q = process.build_process_noise(build, params, None, rotor=process.ROTOR_IN_MASS_MATRIX)
st = predict_mod.predict(carry.state, F, Q)
P0 = np.asarray(st.P)
print("within joint P[i,n+i]", np.abs(np.diag(P0[:n, n:2 * n])))
off = np.abs(P0[n:2 * n, n:2 * n] - np.diag(np.diag(P0[n:2 * n, n:2 * n])))
print("cross vel max", off.max())

# ---- 200 ticks of motion
T = 200
qq, qd = traj(np.arange(T))
mdls = jax.vmap(model_inputs)(jnp.asarray(qq))
sens = sensors_for(qq, qd)
final, (xs, Ps) = jax.jit(scan)(carry, sens, mdls)
P1 = np.asarray(final.state.P)
V = P1[n:2 * n, n:2 * n]
d = np.sqrt(np.diag(V))
C = np.abs(V / np.outer(d, d))
np.fill_diagonal(C, 0)
print("max cross corr", C.max())
print("within-joint coupling", np.abs(np.diag(P1[:n, n:2 * n])))

# ---- NaN window
T1, T2, T3 = 100, 5, 1000
tot = T1 + T2 + T3
qq, qd = traj(np.arange(tot))
mdls = jax.vmap(model_inputs)(jnp.asarray(qq))
sens = sensors_for(qq, qd)
gy = np.asarray(sens.gyros).copy()
gy[T1:T1 + T2, 0, :] = np.nan
enc = np.asarray(sens.encoders).copy()
enc[T1:T1 + T2, 0] = np.nan
sens = sens._replace(gyros=jnp.asarray(gy), encoders=jnp.asarray(enc))
carry = init_carry(build, params, jnp.asarray(qq[0]))
t0 = time.time()
final, (xs, Ps) = jax.jit(scan)(carry, sens, mdls)
xs, Ps = np.asarray(xs), np.asarray(Ps)
print("nan scan", time.time() - t0)
print("bad window finite", np.all(np.isfinite(xs[T1:T1 + T2])), np.all(np.isfinite(Ps[T1:T1 + T2])))
print("all finite", np.all(np.isfinite(xs)))
print("pos err end", np.abs(xs[-1, :n] - qq[-1]).max())
print("vel err end", np.abs(xs[-1, n:2 * n] - qd[-1]).max())
