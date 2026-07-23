r"""
pipeline/main_estimator.py
==========================
G9 -- the **fused estimator step**: one constant-XLA-graph `lax.scan` body that
runs the joint-space KF and the world-centric InEKF back to back, threading the
joint KF's live `(q̂, q̇̂, Σ_q, Σ_q̇, b̂)` into the InEKF the way the log fed each
filter independently at Tier-2 (CLAUDE.md §0 deliverable 3, §3 gate G9).

What is new here, and what is not
---------------------------------
Both filters are already validated at the sensor→state level (`PORT_NOTES.md`;
memory `invariant-estimation-port-status`). G9 is a *composition* job, not an
estimator job. The only genuinely new code is the **boundary** (`_boundary`
below): the joint KF's per-IMU bias corrects the base gyro, which is then rotated
IMU-frame→body-frame and handed to the InEKF as the bias-corrected `ω̄` (I1). The
rest is wiring two `step` functions and one carry.

The single invariant that makes this non-trivial is **I7 (constant XLA graph)**:
no data-dependent shapes or Python branches inside the jitted step. Each filter
already obeys it individually (every gate is a `jnp.where` mask). G9 keeps it true
across the fusion by (a) resolving every name→index at build time in plain Python,
and (b) evaluating the MJX model *inside* the scan at the carry's estimate, with
no Python `if` on any traced value. `tests/pipeline/test_main_estimator.py` proves
it: the `fused_step` jaxpr hashes identically across differing contact / gate
states.

The two G9 landmines (memory `invariant-estimation-g9-landmines`)
-----------------------------------------------------------------
1. **Gyro-bias process noise must be 0 at flight.** `config` holds the Java
   *unit-test* value `imu_bias_process_var = 1e-4` (test-locked); flight is 0.0.
   `build_fused_estimator` overrides it at this boundary by default
   (`imu_bias_process_var=0.0`), so the fused joint-KF bias is not ~200× too noisy.
2. **The InEKF contact *measurement*-noise floor has no port analogue.** Flight
   wires `contactMeasurementVariance = 1e-4` (+ swing inflation); the port's
   contact R is purely `J Σ_q Jᵀ`. This is a structural gap that affects
   velocity/position, not roll/pitch. Exposed here as the `contact_meas_var`
   argument (default 0.0 = current port behaviour); when non-zero it is added as an
   isotropic floor to the InEKF's contact-position noise. See `_boundary`.

Frames (the one place a bug can hide -- guide §G9.3)
----------------------------------------------------
The pelvis IMU is mounted rotated relative to the pelvis *body* frame (`R_mount`,
IMU→body). The base gyro/accel arrive in the IMU measurement frame and must be
rotated into the body frame the InEKF's `R = ᵂR_B` refers to. The contact FK is
written in the *same* body frame, so `R_mount` also enters the kinematics closure.
For a model whose base IMU site is axis-aligned with the base body (the synthetic
G9 fixture), `R_mount = I`. On real Alex it is a +90° yaw and MUST be verified
against `InvariantMainStateEstimator` / `invariantRootAngularVelocityBody*` on the
hardware log before trusting velocity/position (roll/pitch are `R_mount`-robust).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..inEKF import ekf as inekf_mod
from ..inEKF import filter as inf
from ..inEKF.gravity_update import UP, GravityRef
from ..inEKF.state import InEKFState
from ..jointKF import anchors as anch
from ..jointKF import filter as jkf
from ..jointKF.build import KinematicTree, build_joint_kf
from ..jointKF.state import JointKFBuild, JointKFParams, default_params, split_x
from ..model.mjx_model import MjxModel

__all__ = [
    "kinematic_tree_from_mj",
    "FusedEstimator",
    "FusedSensors",
    "FusedOutputs",
    "build_fused_estimator",
    "make_fused_step",
    "init_fused_carry",
    "run_fused",
]


# ---------------------------------------------------------------------------
# mj_model -> KinematicTree adapter (generalised from tests/jointKF/_fixture.py)
# ---------------------------------------------------------------------------

def kinematic_tree_from_mj(
    mj_model,
    *,
    effort_limits: dict[str, float] | None = None,
) -> KinematicTree:
    """Describe a MuJoCo model as the model-agnostic `build.KinematicTree`.

    `build_joint_kf` deliberately takes a plain tree rather than an `mjx.Model`
    (its graph logic is pure and unit-testable without a physics engine). This is
    the adapter that lets the real build run on a real MuJoCo model — the
    production analogue of the test fixture's `kinematic_tree`.

    Parameters
    ----------
    mj_model : mujoco.MjModel
    effort_limits : dict, optional
        Per-joint URDF effort limit (`AlexModelSpec.effort_limits`). Joints absent
        from the dict get `tau_max = NaN`, which makes `sigma_tau` fall back to the
        config scalar — acceptable for a synthetic fixture, wrong for flight, so
        pass the real limits in production.
    """
    import mujoco

    hinge = [j for j in range(mj_model.njnt) if int(mj_model.jnt_type[j]) == 3]
    free = [j for j in range(mj_model.njnt) if int(mj_model.jnt_type[j]) == 0]
    base_dofs = (
        np.concatenate([np.arange(mj_model.jnt_dofadr[j], mj_model.jnt_dofadr[j] + 6) for j in free])
        if free
        else np.zeros(0, dtype=int)
    )
    names = tuple(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge)
    limits = effort_limits or {}
    tau_max = np.array([limits.get(nm, np.nan) for nm in names], dtype=float)
    site_body = {
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, s): int(mj_model.site_bodyid[s])
        for s in range(mj_model.nsite)
    }
    return KinematicTree(
        joint_names=names,
        joint_body=np.array([int(mj_model.jnt_bodyid[j]) for j in hinge]),
        body_parent=np.array(mj_model.body_parentid, dtype=int),
        joint_dof=np.array([int(mj_model.jnt_dofadr[j]) for j in hinge]),
        base_dofs=base_dofs.astype(int),
        site_body=site_body,
        tau_max=tau_max,
    )


# ---------------------------------------------------------------------------
# Per-tick I/O
# ---------------------------------------------------------------------------

class FusedSensors(NamedTuple):
    """One tick of raw proprioception feeding the fused estimator.

    Attributes
    ----------
    encoders : (n,)
        Measured filtered-joint positions, in state order.
    gyros : (m, 3)
        Per-IMU angular rate, each **in its own measurement frame**. The base
        IMU's row (`base_imu`) is the one the InEKF propagation and the joint-KF
        anchor both read.
    accel_base : (3,)
        Base-IMU specific force, in the base-IMU measurement frame. Gravity is
        added inside the InEKF propagation (feed it as read).
    qd_unfiltered : (n_u,)
        Measured velocities of the anchor-chain joints that are not filter states
        (Alex's ankles), in `anchor_unfiltered_mask` column order. Empty when the
        model has no such joints.
    contact : (K,)
        This tick's contact/trust signal per stance-anchor slot (joint KF). It is
        consumed on the NEXT tick — the one-tick-delayed phase ordering is handled
        inside `jkf.step`.
    contact_chol : (N, 3, 3)
        ContactNet Cholesky factors for the InEKF (the ONLY contact-condition
        input to the InEKF; firm ⇒ small, swing ⇒ large). `N == K`.
    """

    encoders: Array
    gyros: Array
    accel_base: Array
    qd_unfiltered: Array
    contact: Array
    contact_chol: Array


class FusedOutputs(NamedTuple):
    """Per-tick emitted estimate + diagnostics (stacked over time by `run_fused`).

    The InEKF pose is the headline deliverable; `q`/`q_dot`/`bias` are the joint-KF
    estimate; the two diagnostics pytrees carry the NIS/gate observables the G10
    consistency evaluation and ContactNet trust features read.
    """

    R: Array                     # (3,3)  ᵂR_B base orientation
    v: Array                     # (3,)   base velocity, world
    p: Array                     # (3,)   base position, world
    q: Array                     # (n,)   filtered joint positions
    q_dot: Array                 # (n,)   filtered joint velocities
    bias: Array                  # (3m,)  per-IMU gyro bias
    jkf: jkf.TickDiagnostics
    inekf: inf.InEKFOutputs


# ---------------------------------------------------------------------------
# The assembled estimator (built once, plain Python — I7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusedEstimator:
    """Everything the jitted `fused_step` closes over. Built by `build_fused_estimator`.

    Nothing here is traced or changes shape during a run (I2, I7).
    """

    model: MjxModel
    build: JointKFBuild
    params: JointKFParams
    ekf: inekf_mod.InvariantEKF
    kinematics: inf.ContactKinematics
    inekf_step: Callable
    base_imu: int                 # IMU ordinal of the base IMU (star centre)
    base_site: int                # site ordinal of the base IMU (== base_imu here)
    foot_site_ords: np.ndarray    # (K,) site ordinals of the sole sites
    pair_sites: np.ndarray        # (n_pairs, 2) site ordinals per IMU pair
    R_mount: Array                # (3,3) IMU-frame -> body-frame
    contact_meas_var: float       # isotropic floor on InEKF contact-position noise

    @property
    def n_joints(self) -> int:
        return self.build.n_joints

    @property
    def n_contacts(self) -> int:
        return self.ekf.N


def build_fused_estimator(
    model: MjxModel,
    imu_sites: Sequence[str],
    pairs: Sequence[tuple[int, int]],
    foot_sites: Sequence[str],
    *,
    base_imu: int = 0,
    R_mount: Array | None = None,
    effort_limits: dict[str, float] | None = None,
    dt: float = 1.0e-3,
    imu_bias_process_var: float = 0.0,       # landmine #1: flight value, not config's 1e-4
    contact_meas_var: float = 0.0,           # landmine #2: 0.0 == current port behaviour
    gyro_var: float | None = None,
    accel_var: float | None = None,
    contact_var: float | None = None,
) -> FusedEstimator:
    """Assemble the joint KF + InEKF into one fused estimator (plain Python, I7).

    `imu_sites`, `pairs`, `foot_sites` follow the same conventions as
    `build_joint_kf` / `MjxModel`: `imu_sites` fixes each IMU's ordinal; `pairs`
    are `(parent_ordinal, child_ordinal)` over that ordering; `foot_sites` host the
    stance anchors (`K = len(foot_sites)`) AND become the InEKF's `N` contacts.

    The two landmine arguments default to their FLIGHT values (`imu_bias_process_var
    = 0`) or to the current-port value (`contact_meas_var = 0`); see the module
    docstring. Both are surfaced here rather than buried so the G9 gate can flip
    them and measure the effect.
    """
    site_names = model.site_names
    imu_sites = tuple(imu_sites)
    foot_sites = tuple(foot_sites)

    tree = kinematic_tree_from_mj(model.mj_model, effort_limits=effort_limits)
    build = build_joint_kf(
        tree, imu_sites, pairs, foot_sites,
        base_imu=base_imu, use_mass_matrix=True, use_armature_for_rotor=True,
    )
    params = default_params(dt=dt, imu_bias_process_var=imu_bias_process_var)

    K = len(foot_sites)
    ekf = inekf_mod.create(
        number_of_contacts=K, gyro_var=gyro_var, accel_var=accel_var,
        contact_var=contact_var, dt=dt,
    )

    base_site = site_names.index(imu_sites[base_imu])
    foot_site_ords = np.array([site_names.index(s) for s in foot_sites], dtype=int)
    R_mount = jnp.eye(3, dtype=jnp.float64) if R_mount is None \
        else jnp.asarray(R_mount, dtype=jnp.float64)

    kinematics = _make_contact_kinematics(model, base_site, foot_site_ords, R_mount)
    inekf_step = inf.make_step(ekf, kinematics)

    return FusedEstimator(
        model=model,
        build=build,
        params=params,
        ekf=ekf,
        kinematics=kinematics,
        inekf_step=inekf_step,
        base_imu=base_imu,
        base_site=base_site,
        foot_site_ords=foot_site_ords,
        pair_sites=np.asarray(model.pair_sites, dtype=int),
        R_mount=R_mount,
        contact_meas_var=float(contact_meas_var),
    )


# ---------------------------------------------------------------------------
# Contact FK closure — base->foot vectors in the InEKF body frame
# ---------------------------------------------------------------------------

def _make_contact_kinematics(
    model: MjxModel, base_site: int, foot_site_ords: np.ndarray, R_mount: Array
) -> inf.ContactKinematics:
    r"""The `robot/` seam: `q ↦ ContactFrames(y, J)` in the InEKF body frame.

    `y_i = ᵂR_B^T (p_{foot_i} − p_B)` with the FK evaluated at the model's `qpos0`
    base pose (the base cancels — every quantity is base-relative), and
    `ᵂR_B = ᵂR_{baseIMU} R_mount^T` so the FK body frame is exactly the frame the
    InEKF's `R` refers to. `J = ∂y/∂q` by forward-mode autodiff; `J_dot = 0` (a
    port TODO shared with the Tier-2 replay — the velocity-noise term is deferred,
    `inEKF/filter.py`).
    """
    feet = jnp.asarray(foot_site_ords, dtype=int)
    Rm_T = R_mount.T

    def _foot_y(q: Array) -> Array:
        pos, rot = model.site_poses(q)
        p_base = pos[base_site]
        R_bw = rot[base_site] @ Rm_T                     # ᵂR_B
        return jnp.einsum("ij,kj->ki", R_bw.T, pos[feet] - p_base)   # (K,3)

    def kinematics(q: Array, q_dot: Array) -> inf.ContactFrames:
        y = _foot_y(q)
        J = jax.jacfwd(_foot_y)(q)                       # (K,3,n)
        return inf.ContactFrames(y=y, J=J, J_dot=jnp.zeros_like(J))

    return kinematics


# ---------------------------------------------------------------------------
# The fused scan body
# ---------------------------------------------------------------------------

def make_fused_step(fused: FusedEstimator) -> Callable:
    """Build the jitted `lax.scan` body `fused_step(carry, sensors)`.

    Carry is `(jkf_carry, inekf_carry)`; `sensors` is a `FusedSensors`. Returns
    `((jkf_carry', inekf_carry'), FusedOutputs)`. Traces to one constant graph
    regardless of contact/gate state (I7).
    """
    model = fused.model
    build = fused.build
    params = fused.params
    base_imu = fused.base_imu
    base_site = fused.base_site
    foot_ords = jnp.asarray(fused.foot_site_ords, dtype=int)
    pair_sites = jnp.asarray(fused.pair_sites, dtype=int)
    R_mount = fused.R_mount
    inekf_step = fused.inekf_step
    n = build.n_joints
    contact_meas_var = fused.contact_meas_var

    def fused_step(carry, sensors: FusedSensors):
        jkf_carry, inekf_carry = carry

        # -- (a) MJX eval at the PREVIOUS q̂ (the EKF linearisation point) -------
        # One position-level FK pass; R_rel is derived from site rotations rather
        # than re-running `measure.pair_frames` (guide §G9.2: one FK per tick).
        q_prev = jkf_carry.state.x[:n]
        ev = model.evaluate(q_prev)
        R_pair = ev.site_rot[pair_sites]                          # (n_pairs,2,3,3)
        R_rel = jnp.einsum("eji,ejk->eik", R_pair[:, 1], R_pair[:, 0])   # ᶜR_p
        jac = anch.anchor_jacobians(
            build, ev.J_ang, ev.site_rot,
            base_site=base_site, foot_sites=foot_ords,
        )
        model_in = jkf.ModelInputs(J_rel=ev.J_rel, R_rel=R_rel, anchor_jac=jac, M=ev.M)

        # -- (b) joint-KF step -------------------------------------------------
        jkf_sensors = jkf.SensorInputs(
            encoders=sensors.encoders,
            gyros=sensors.gyros,
            qd_unfiltered=sensors.qd_unfiltered,
            contact=sensors.contact,
        )
        jkf_carry, jkf_diag = jkf.step(jkf_carry, jkf_sensors, model_in, build, params)
        q_hat, qd_hat, bias = split_x(jkf_carry.state.x, n)
        sigma_q = jkf_carry.state.P[:n, :n]
        sigma_qd = jkf_carry.state.P[n:2 * n, n:2 * n]

        # -- (c) the boundary: bias-correct + frame the base IMU (I1, G9.3) ----
        inekf_inputs = _boundary(
            sensors, bias, base_imu, R_mount, q_hat, qd_hat, sigma_q, sigma_qd,
            contact_meas_var,
        )

        # -- (d/e) InEKF step --------------------------------------------------
        inekf_carry, inekf_out = inekf_step(inekf_carry, inekf_inputs)

        outputs = FusedOutputs(
            R=inekf_out.state.R, v=inekf_out.state.v, p=inekf_out.state.p,
            q=q_hat, q_dot=qd_hat, bias=bias,
            jkf=jkf_diag, inekf=inekf_out,
        )
        return (jkf_carry, inekf_carry), outputs

    return fused_step


def _boundary(
    sensors, bias, base_imu, R_mount, q_hat, qd_hat, sigma_q, sigma_qd,
    contact_meas_var,
) -> inf.InEKFInputs:
    r"""Joint-KF output → InEKF input. The one genuinely new piece of G9.

    * Bias-correct the base gyro in the IMU frame (`b_base` lives in the base
      IMU's own measurement frame, I1), then rotate IMU→body (`R_mount`).
    * Rotate the specific force IMU→body (gravity is added inside the InEKF).
    * Route `Σ_q, Σ_q̇` through the `JointFilterOutput` unchanged — the coupling in
      the full matrix is exactly what the InEKF contact update `N = J Σ_q Jᵀ`
      needs; never diagonalise them.

    `contact_meas_var` (landmine #2) is folded into `Σ_q` as an isotropic floor
    `contact_meas_var · I` before it reaches the contact update, which is the
    port's stand-in for flight's `ConstantContactMeasurementNoiseProvider`. With
    the default 0.0 this is the identity and the contact R is purely `J Σ_q Jᵀ`.
    """
    b_base = jax.lax.dynamic_slice(bias, (3 * base_imu,), (3,))
    gyro_base = sensors.gyros[base_imu]
    omega_body = R_mount @ (gyro_base - b_base)
    accel_body = R_mount @ sensors.accel_base
    raw_omega_body = R_mount @ gyro_base

    n = q_hat.shape[0]
    sigma_q_eff = sigma_q + contact_meas_var * jnp.eye(n, dtype=jnp.float64)

    joint = inf.JointFilterOutput(
        q=q_hat, q_dot=qd_hat, sigma_q=sigma_q_eff, sigma_q_dot=sigma_qd,
    )
    return inf.InEKFInputs(
        omega=omega_body, accel=accel_body, raw_omega=raw_omega_body,
        joint=joint, contact_chol=sensors.contact_chol,
    )


# ---------------------------------------------------------------------------
# Initialisation and the trajectory driver
# ---------------------------------------------------------------------------

def init_fused_carry(
    fused: FusedEstimator,
    q0: Array,
    *,
    rotation: Array | None = None,
    velocity: Array | None = None,
    position: Array | None = None,
    covariance: Array | None = None,
    seed_gravity: bool = True,
):
    """Seed `(jkf_carry, inekf_carry)` for a level, planted start.

    The InEKF contacts are seeded at the FK foot positions consistent with the
    initial base pose, so the first contact residual is exactly zero. The gravity
    reference is seeded *converged* to `Rᵀ·UP` (`seed_gravity=True`) so leveling is
    active from tick 1 — matching a continuously-running filter; pass
    `seed_gravity=False` to cold-start it from the first accelerometer reading.
    """
    q0 = jnp.asarray(q0, dtype=jnp.float64)
    R0 = jnp.eye(3, dtype=jnp.float64) if rotation is None else jnp.asarray(rotation, float)
    v0 = jnp.zeros(3, dtype=jnp.float64) if velocity is None else jnp.asarray(velocity, float)
    p0 = jnp.zeros(3, dtype=jnp.float64) if position is None else jnp.asarray(position, float)

    # Contacts consistent with the initial pose: d_i = R0 · y_i(q0) + p0, so
    # y_i = R0ᵀ(d_i − p0) holds and the first FK innovation is zero.
    frames = fused.kinematics(q0, jnp.zeros_like(q0))
    d0 = jnp.einsum("ij,kj->ki", R0, frames.y) + p0[None, :]

    state0 = inekf_mod.initialize(
        fused.ekf, rotation=R0, velocity=v0, position=p0, contacts=d0,
        covariance=covariance,
    )
    inekf_carry = inf.init_carry(state0)
    if seed_gravity:
        inekf_carry = inf.InEKFCarry(
            state=state0,
            gravity_ref=GravityRef(direction=R0.T @ UP, initialized=jnp.array(1.0)),
        )

    jkf_carry = jkf.init_carry(fused.build, fused.params, q0=q0)
    # Commit the whole carry to one device so every leaf shares a sharding. The
    # contacts `d0` come off an MJX-FK einsum (device-committed) while the
    # `jnp.eye`/`jnp.zeros` leaves are uncommitted; that *mixed* commitment is what
    # makes the first jitted step recompile once to the uniform-committed sharding
    # (identical jaxpr, different `Argument mapping`). Uniform commitment keeps the
    # jitted `fused_step` at a single compiled executable — the operational I7
    # property the G9 gate asserts.
    return jax.device_put((jkf_carry, inekf_carry), jax.devices()[0])


def run_fused(fused: FusedEstimator, carry, sensors: FusedSensors):
    """Scan `fused_step` over a trajectory.

    `sensors` is a `FusedSensors` whose every field carries a leading time axis of
    length `T`. Returns `(final_carry, FusedOutputs)` with outputs stacked over
    time. The compiled graph is one tick regardless of `T` (I7).
    """
    return jax.lax.scan(make_fused_step(fused), carry, sensors)
