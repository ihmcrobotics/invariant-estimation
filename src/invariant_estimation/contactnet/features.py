r"""
Per-contact feature extraction for the ContactNet MLP.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Define global Alex variables (might want to parameterize to be per structure)
ALEX_FOOT_CHAINS: tuple[tuple[str, ...], ...] = (
    ("LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
     "LEFT_ANKLE_Y", "LEFT_ANKLE_X"),
    ("RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y",
     "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X"),
)

"""
Above, these are the joints IN ORDER for Alex, so they need to stay the same. 
Do not touch unless you're sure!
"""

# extra helpers for lower body, as the structure is the same per side.
JOINT_LABELS: tuple[str, ...] = ("hip_x","hip_z","hip_y","knee_y","ankle_y","ankle_x")

def window_indices(T: int, H: int, stride: int = 1) -> Array:
    if stride < 1:
        raise ValueError(f"Stride must be >= 1, but got {stride}")
    k = jnp.arange(T)[:,None]
    h = jnp.arange(H)[:,None]
    return jnp.maximum(k - (H - 1 - h) * stride, 0)

def boxcar(x: Array, s: int) -> Array:
    if s < 1:
        raise ValueError(f"Boxcar size must be >= 1, but got {s}")
    if s == 1:
        return x
    pad = jnp.repeat(x[:1], s - 1, axis=0)
    c = jnp.cumsum(jnp.concatenate([pad, x], axis=0), axis=0)
    c = jnp.concatenate([jnp.zeros_like(c[:1]), c], axis=0)
    return (c[s:] - c[:-s]) / s

def window(channels: Array, H: int, stride: int = 1) -> Array:
    if channels.ndim != 3:
        raise ValueError(f"Expected (T, N_c, F), but got {channels.shape}")
    smoothed = boxcar(channels, stride)
    idx = window_indices(smoothed.shape[0], H, stride)
    gathered = smoothed[idx]
    return jnp.swapaxes(gathered, 1, 2) # (T, F, H)

def build_subchain_indices(joint_names, unfiltered_names, foot_chains=ALEX_FOOT_CHAINS, contacts_per_foot: int = 1):
    if contacts_per_foot < 1:
        raise ValueError(f"contacts_per_foot must be >= 1, got {contacts_per_foot}")
    order = list(joint_names) + list(unfiltered_names)
    pos = {n: i for i, n in enumerate(order)}
    if len(pos) != len(order):
        raise ValueError("duplicate joint name across the filtered and unfiltered sets")

    widths = {len(c) for c in foot_chains}
    if len(widths) != 1:
        raise ValueError(f"every foot chain must be the same length, got {widths}")

    idx = []
    for chain in foot_chains:
        missing = [n for n in chain if n not in pos]
        if missing:
            raise KeyError(f"joints not present in the model: {missing}")
        row = [pos[n] for n in chain]
        idx.extend([row] * contacts_per_foot)     # foot-major: L,L,R,R for 2/foot
    return np.asarray(idx, dtype=int)

def subchain_for(fused, unfiltered_names, foot_chains=ALEX_FOOT_CHAINS):
    n_feet = len(foot_chains)
    n_c = int(fused.n_contacts)
    per, rem = divmod(n_c, n_feet)
    if rem:
        raise ValueError(
            f"Estimator has N={n_c} contacts, but {n_feet} feet; must be a multiple of {n_feet}"
        )
    return build_subchain_indices(fused.build.joint_names, unfiltered_names, foot_chains=foot_chains, contacts_per_foot=per)

def channel_names(joint_labels: tuple[str, ...] = JOINT_LABELS) -> tuple[str, ...]:
    """
    The frozen channel ordering - must match `make_contact_channels` exactly.
    """
    return (
        "base_gyro_x", "base_gyro_y", "base_gyro_z",
        "base_accel_x", "base_accel_y", "base_accel_z",
        *(f"q_{s}" for s in joint_labels),
        *(f"qd_{s}" for s in joint_labels), #NOTE: added 8/2 for CoCo implementation parity.
        *(f"tau_{s}" for s in joint_labels),
        "p_bc_x", "p_bc_y", "p_bc_z",
        "v_bc_x", "v_bc_y", "v_bc_z",
    )

def make_contact_channels(subchain, base_imu: int, kinematics, dt: float):
    subchain = jnp.asarray(subchain)
    n_c, j_sub = subchain.shape
    def contact_channels(sensors) -> Array:
        if sensors.q_unfiltered.shape[-1] == 0:
            raise ValueError("No unfiltered joints in the model; cannot compute contact channels")
        q_all = jnp.concatenate([sensors.encoders, sensors.q_unfiltered], axis=-1)
        qd_all = jnp.concatenate([sensors.encoders_vel, sensors.qd_unfiltered], axis=-1)
        tau_all = sensors.torques
        T = q_all.shape[0]

        if qd_all.shape[-1] != q_all.shape[-1]:
            raise ValueError(f"Expected qd_all.shape[-1] == q_all.shape[-1], but got {qd_all.shape[-1]} != {q_all.shape[-1]}")

        q_sub = q_all[:, subchain]
        qd_sub = qd_all[:, subchain]
        tau_sub = tau_all[:, subchain]

        omega = jnp.broadcast_to(
            sensors.gyros[:, base_imu, :][:, None, :], (T, n_c, 3)
        )
        accel = jnp.broadcast_to(
            sensors.accel_base[:, None, :], (T, n_c, 3)
        )

        #WARNING: why do it this way? Don't we want this to be based on the encoder positions and the FK there, which is estimator independent if called with just sensor values?
        zero_qd = jnp.zeros_like(q_all[0])
        p = jax.vmap(lambda q: kinematics(q, zero_qd).y)(q_all)

        v = jnp.concatenate(
            [jnp.zeros((1, n_c, 3), dtype=p.dtype), jnp.diff(p, axis=0) / dt], axis=0
        )

        return jnp.concatenate([omega, accel, q_sub, qd_sub, tau_sub, p, v], axis=-1)
    return contact_channels



def make_feature_windows(subchain, base_imu: int, kinematics, dt: float, H: int, stride: int = 1):
    channels = make_contact_channels(subchain, base_imu, kinematics, dt)
    def feature_windows(sensors) -> Array:
        return window(channels(sensors), H, stride)
    return feature_windows
