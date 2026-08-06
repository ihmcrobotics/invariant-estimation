"""Same replay as `reseed_drift.py`, written against the PRE-reseed API.

Run this from the base checkout to prove the disabled path is unchanged:

    cd <base>  && python3 experiments/_base_arm.py            # writes /tmp/base_arm.npz
    cd <reseed> && python3 experiments/reseed_drift.py --arm off
    # then diff /tmp/base_arm.npz against /tmp/reseed_arms/off_16000_20000.npz

It imports no reseed symbol and calls `init_carry(state0)` with the old
one-argument signature, so it runs unmodified on either checkout.
"""
import numpy as np
import jax
import jax.numpy as jnp

from invariant_estimation.inEKF import ekf as inekf_mod
from invariant_estimation.inEKF.filter import (
    InEKFInputs, JointFilterOutput, init_carry, make_step,
)
from invariant_estimation.pipeline import main_estimator as me
import run_policy as rp

T0, TICKS = 16000, 20000

z = np.load("data/flat_seed005.npz", allow_pickle=True)
sl = slice(T0, T0 + TICKS)
joint = JointFilterOutput(
    q=z["inputs.joint.q"][sl], q_dot=z["inputs.joint.q_dot"][sl],
    sigma_q=z["inputs.joint.sigma_q"][sl], sigma_q_dot=z["inputs.joint.sigma_q_dot"][sl],
)
inputs = InEKFInputs(
    omega=z["inputs.omega"][sl], accel=z["inputs.accel"][sl],
    raw_omega=z["inputs.raw_omega"][sl], joint=joint,
    contact_chol=z["inputs.contact_chol"][sl],
)

fused = me.build_alex_fused_estimator_from_urdf(
    rp.cycloid_forearm_urdf(rp.URDF), dt=1.0e-3, contact_fk_unfiltered=True)

R0, p0, v0 = z["truth.R"][T0], z["truth.p"][T0], z["truth.v"][T0]
y0 = np.asarray(fused.kinematics(jnp.asarray(joint.q[0]), jnp.asarray(joint.q_dot[0])).y)
d0 = np.einsum("ij,kj->ki", R0, y0) + p0[None, :]
state0 = inekf_mod.initialize(
    fused.ekf, rotation=jnp.asarray(R0), velocity=jnp.asarray(v0),
    position=jnp.asarray(p0), contacts=jnp.asarray(d0))

_, out = jax.lax.scan(make_step(fused.ekf, fused.kinematics),
                      init_carry(state0), jax.tree.map(jnp.asarray, inputs))
np.savez("/tmp/base_arm.npz", est_p=np.asarray(out.state.p))
print("wrote /tmp/base_arm.npz")
