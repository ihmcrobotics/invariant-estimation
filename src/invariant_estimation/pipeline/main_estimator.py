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

Frames — THREE of them, kept distinct (guide §G9.3; the real-Alex trap)
-----------------------------------------------------------------------
1. **Base IMU site `S`** (`imu_sites[base_imu]`): where the base gyro/accel are
   measured, and the joint-KF stance-anchor frame.
2. **Body frame `B`** (`base_body_site`, the pelvis *root* body): the frame the
   InEKF's `R = ᵂR_B` refers to and the contact-FK origin. On real Alex the IMU is
   both offset from and yawed +90° relative to `B`, so `B ≠ S` — using the IMU site
   as the body frame (as an early cut did) puts that offset+yaw straight into the
   pose. `build_fused_estimator(base_body_site=...)` selects `B`; it defaults to
   the base IMU site, which is correct only when the two coincide (the synthetic
   fixture, where `R_mount = I`).
3. **`R_mount = ᴮR_S`**: rotates the base IMU measurement into `B`. Auto-computed
   from FK at `qpos0`. Enters only the boundary (`_boundary`); the contact FK uses
   `B` directly. On real Alex it is a clean +90° yaw, verified against Java to
   1e-18 (`R_mount @ jointKF_bias_S == invariantAppliedGyroBiasInPelvisFrame`;
   `tests/replay/test_fused_real_model.py`). NB: the real InEKF consumes a
   Mahony-prefiltered pelvis gyro, so a full trajectory replay must feed that
   processed channel, not the raw `gyroscope_pelvis_imu` — see PORT_NOTES "G9 —
   real model". Roll/pitch are `R_mount`-robust; velocity/position are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..config import load_config
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
    "ALEX_IMU_SITES",
    "ALEX_PAIRS",
    "ALEX_FOOT_SITES",
    "ALEX_EXTRA_SITES",
    "ALEX_SOLE_OFFSET",
    "ALEX_ANKLE_HEIGHT",
    "alex_site_names",
    "build_alex_fused_estimator",
    "alex_spec_from_urdf",
    "build_alex_fused_estimator_from_urdf",
]


# ---------------------------------------------------------------------------
# Alex topology — the resolved `imu_pairs` TODO (CLAUDE.md §2b)
# ---------------------------------------------------------------------------
# Derived from the 2026-07-17 Alex001 log's model.sdf and cross-checked three ways
# (scratch verification, recorded in PORT_NOTES "G9 — real model"):
#   * this IMU set + star reproduces EXACTLY the 9 logged FILTERED_JOINTS
#     (SPINE_Z + both legs' HIP_X/Z/Y + KNEE_Y) and jointKFNumberOfIMUs = 8;
#   * `R_mount` auto-computed from these sites matches the Java InEKF's
#     `invariantAppliedGyroBiasInPelvisFrame` to 1e-18 (the +90° pelvis-IMU yaw);
#   * `dof_nuisance` is base-6-only (no gap joints), matching the parity harness.
# It is a STAR on the pelvis IMU (CLAUDE.md §2 "star on the base IMU"): every other
# IMU is paired against the pelvis, so the shared-base-IMU `LΣLᵀ` cross-covariance
# (I6) is exercised. The leg IMUs give progressively longer overlapping chains
# (hip_x ⊂ thigh ⊂ shin), which is the redundant multi-measurement the star buys.
ALEX_IMU_SITES: tuple[str, ...] = (
    "pelvis_imu",                                   # ordinal 0 == base IMU (star centre)
    "torso_imu",                                    # -> SPINE_Z
    "left_hip_x_imu", "left_thigh_imu", "left_shin_imu",     # -> LEFT hip X/Z/Y + KNEE_Y
    "right_hip_x_imu", "right_thigh_imu", "right_shin_imu",  # -> RIGHT hip X/Z/Y + KNEE_Y
)
ALEX_PAIRS: tuple[tuple[int, int], ...] = tuple((0, k) for k in range(1, len(ALEX_IMU_SITES)))
ALEX_FOOT_SITES: tuple[str, ...] = ("left_sole", "right_sole")
# The sole plane, in the `*_FOOT` link frame == the ankle-roll frame. Java:
#   AlexV1PhysicalProperties.soleToAnkleFrameTransforms
#     translation = (ACTUAL_FOOT_LENGTH / 2 - FOOT_BACK, 0, -ANKLE_HEIGHT)
#                 = (0.197 / 2 - 0.052, 0, -0.072)
# (the transform's rotation is commented out in Java, so this is a pure translation)
# and `InvariantMainStateEstimator` anchors contacts at `referenceFrames.getSoleFrame(side)`,
# i.e. HERE and not at the ankle. Emitting the soles at the link origin -- as this table did
# until 2026-07-26 -- put the InEKF's contact points and the joint-KF's stance anchors 7.2 cm
# above the ground and 4.65 cm behind the sole centre.
ALEX_ANKLE_HEIGHT: float = 0.072
ALEX_SOLE_OFFSET: tuple[float, float, float] = (0.197 / 2.0 - 0.052, 0.0, -ALEX_ANKLE_HEIGHT)

# extra_sites for `urdf2mjcf`: the InEKF body frame (pelvis root body) and the two foot soles
# the stance anchors / contacts sit on.
ALEX_EXTRA_SITES: dict[str, str | tuple[str, tuple[float, float, float]]] = {
    "base_body": "PELVIS_LINK",
    "left_sole": ("LEFT_FOOT", ALEX_SOLE_OFFSET),
    "right_sole": ("RIGHT_FOOT", ALEX_SOLE_OFFSET),
}


def alex_site_names() -> tuple[str, ...]:
    """The full site-name tuple for the Alex `MjxModel` (IMUs, body frame, soles)."""
    return ALEX_IMU_SITES + ("base_body",) + ALEX_FOOT_SITES


def build_alex_fused_estimator(spec, **overrides) -> "FusedEstimator":
    """Build the fused estimator for real Alex from an `AlexModelSpec`.

    `spec` must come from `urdf2mjcf.convert_log_model(log_dir,
    extra_sites=ALEX_EXTRA_SITES, ...)` so the body-frame and sole sites exist.
    Encapsulates the resolved Alex topology (`ALEX_*` above) so production and the
    replay test share one definition; `**overrides` pass straight through to
    `build_fused_estimator` (e.g. `contact_meas_var=1e-4`, `dt=...`).
    """
    model = MjxModel.from_xml_string(spec.mjcf, site_names=alex_site_names(), pairs=ALEX_PAIRS)
    return build_fused_estimator(
        model, imu_sites=ALEX_IMU_SITES, pairs=ALEX_PAIRS, foot_sites=ALEX_FOOT_SITES,
        base_imu=0, base_body_site="base_body", effort_limits=spec.effort_limits,
        **overrides,
    )


def alex_spec_from_urdf(urdf_path):
    """`AlexModelSpec` from a standalone `.urdf` (the config rotor table + Alex sites).

    The URDF analogue of `convert_log_model`: reads the file, writes the rotor
    inertia into `armature`, and adds the `base_body`/sole `extra_sites`.
    """
    import pathlib

    from ..config import load_config
    from ..model.urdf2mjcf import urdf_to_mjcf

    jk = load_config()["joint_kf"]
    return urdf_to_mjcf(
        pathlib.Path(urdf_path).read_text(),
        rotor_inertia=jk["rotor_inertia"],
        rotor_inertia_default=jk["rotor_inertia_default"],
        extra_sites=ALEX_EXTRA_SITES,
    )


def build_alex_fused_estimator_from_urdf(urdf_path, **overrides) -> "FusedEstimator":
    """Build the Alex fused estimator directly from a standalone URDF file.

    The production path when the model comes from a `.urdf` (the RL training body
    `alex_with_imus.urdf`) rather than a log's `model.sdf`. Verified in
    `tests/replay/test_fused_real_model.py` to reproduce the Java-model `R_mount`
    and site FK bit-for-bit — the permanent lock on the training↔hardware
    cross-check. `**overrides` pass through to `build_fused_estimator`.
    """
    return build_alex_fused_estimator(alex_spec_from_urdf(urdf_path), **overrides)


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
    q_unfiltered : (n_u,), optional
        Measured POSITIONS of the same off-path anchor joints. Only read when the
        estimator was built with `contact_fk_unfiltered=True`, which lets the
        contact FK stand on the live ankle angles instead of `qpos0`; ignored
        otherwise, so the field is optional and defaults to empty.
    """

    #TODO: the encoders are not just the unfiltered joints, we use the entire robot as an input. This is pure sensors, so it shouldn't matter which ones we use.
    encoders: Array
    gyros: Array
    accel_base: Array
    qd_unfiltered: Array
    contact: Array
    contact_chol: Array
    q_unfiltered: Array = ()
    encoders_vel: Array = ()
    torques: Array = ()             # (n + n_u,) ContactNet feature channel only; concat(filtered, unfiltered)


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
    base_site: int                # site ordinal of the base IMU (joint-KF anchor frame + gyro source)
    base_body_site: int           # site ordinal of the InEKF body frame B (root/pelvis body origin)
    foot_site_ords: np.ndarray    # (K,) site ordinals of the sole sites
    pair_sites: np.ndarray        # (n_pairs, 2) site ordinals per IMU pair
    R_mount: Array                # (3,3) ᴮR_S: base-IMU measurement frame -> InEKF body frame
    contact_meas_var: float       # isotropic floor on InEKF contact-position noise
    aux_qpos: np.ndarray          # (n_u,) qpos indices of the off-path anchor joints (may be empty)
    aux_encoder_var: np.ndarray   # (n_u,) their encoder position variance
    aux_qd_var: float             # their velocity variance (config `sigma_qd_unfiltered`)

    @property
    def n_joints(self) -> int:
        return self.build.n_joints

    @property
    def n_contacts(self) -> int:
        return self.ekf.N

    @property
    def n_aux(self) -> int:
        """Off-path joints fed to the contact FK (0 when the feature is off)."""
        return int(len(self.aux_qpos))


def build_fused_estimator(
    model: MjxModel,
    imu_sites: Sequence[str],
    pairs: Sequence[tuple[int, int]],
    foot_sites: Sequence[str],
    *,
    base_imu: int = 0,
    base_body_site: str | None = None,
    R_mount: Array | None = None,
    effort_limits: dict[str, float] | None = None,
    dt: float = 1.0e-3,
    imu_bias_process_var: float = 0.0,       # landmine #1: flight value, not config's 1e-4
    contact_meas_var: float = 0.0,           # landmine #2: 0.0 == current port behaviour
    gyro_var: float | None = None,
    accel_var: float | None = None,
    contact_var: float | None = None,
    contact_fk_unfiltered: bool = False,
) -> FusedEstimator:
    """Assemble the joint KF + InEKF into one fused estimator (plain Python, I7).

    `imu_sites`, `pairs`, `foot_sites` follow the same conventions as
    `build_joint_kf` / `MjxModel`: `imu_sites` fixes each IMU's ordinal; `pairs`
    are `(parent_ordinal, child_ordinal)` over that ordering; `foot_sites` host the
    stance anchors (`K = len(foot_sites)`) AND become the InEKF's `N` contacts.

    Three frames, kept distinct (the real-Alex frame trap)
    ------------------------------------------------------
    * **Base IMU site** = `imu_sites[base_imu]`: the joint-KF anchor frame and the
      source of the base gyro/accel. The measurement lives here.
    * **Body frame `B`** = `base_body_site`: the InEKF's `R = ᵂR_B` frame and the
      contact-FK origin — the *root/pelvis body*, which is what
      `invariantRootAngularVelocityBody` reports. Defaults to the base IMU site
      (correct only when the IMU is mounted at the body origin with no rotation,
      e.g. the synthetic fixture). On real Alex, pass the pelvis-body site: the
      pelvis IMU is offset AND yawed +90° from the body, and using the IMU site as
      the body frame puts that offset+rotation straight into the pose.
    * **`R_mount = ᴮR_S`**: rotates the base IMU measurement into `B`. Auto-computed
      from the model at `qpos0` (`base_body_rotᵀ · base_imu_rot`) unless overridden.

    The two landmine arguments default to their FLIGHT values (`imu_bias_process_var
    = 0`) or to the current-port value (`contact_meas_var = 0`); see the module
    docstring. Both are surfaced here so the gate can flip them and measure the effect.

    `contact_fk_unfiltered` feeds the MEASURED off-path anchor joints (Alex's
    ankles) to the contact FK instead of pinning them at `qpos0`, and widens
    `Σ_q` with their encoder variance so `N = J Σ_q Jᵀ` still accounts for every
    joint the measurement depends on. Java's InEKF anchors at the live sole frame,
    so this is the faithful behaviour; it is off by default only because it
    changes numbers the existing gates were recorded against. See
    `_make_contact_kinematics` for what it is worth in metres.
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
    base_body_ord = base_site if base_body_site is None else site_names.index(base_body_site)
    foot_site_ords = np.array([site_names.index(s) for s in foot_sites], dtype=int)

    # R_mount = ᴮR_S at qpos0. When base_body_site is the IMU site (synthetic case)
    # this is exactly I. When it is the pelvis body (real Alex) it carries the
    # +90° mount yaw. Auto-computed from FK unless the caller pins it.
    #
    # Computed with plain MuJoCo, NOT `model.site_poses` (MJX): it is a build-time
    # constant, and running it through MJX would trace `mjx.kinematics` over the
    # whole model — minutes for the deep-hand training URDF (PORT_NOTES G1). Plain
    # `mj_kinematics` at qpos0 is instant and gives the identical rotations.
    if R_mount is None:
        import mujoco

        d = mujoco.MjData(model.mj_model)
        mujoco.mj_kinematics(model.mj_model, d)
        sid = np.asarray(model.site_ids)
        Rb = d.site_xmat[sid[base_body_ord]].reshape(3, 3)
        Rs = d.site_xmat[sid[base_site]].reshape(3, 3)
        R_mount = jnp.asarray(Rb.T @ Rs, dtype=jnp.float64)
    else:
        R_mount = jnp.asarray(R_mount, dtype=jnp.float64)

    # -- off-path anchor joints for the contact FK (Alex: the four ankles) ----
    aux_qpos, aux_var = _aux_joint_tables(model, build) if contact_fk_unfiltered else (
        np.zeros(0, dtype=int), np.zeros(0))
    kinematics = _make_contact_kinematics(
        model, base_body_ord, foot_site_ords,
        aux_qpos=aux_qpos, n_filtered=build.n_joints,
    )
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
        base_body_site=base_body_ord,
        foot_site_ords=foot_site_ords,
        pair_sites=np.asarray(model.pair_sites, dtype=int),
        R_mount=R_mount,
        contact_meas_var=float(contact_meas_var),
        aux_qpos=aux_qpos,
        aux_encoder_var=aux_var,
        aux_qd_var=float(load_config()["joint_kf"]["sigma_qd_unfiltered"]) ** 2,
    )


def _aux_joint_tables(model: MjxModel, build: JointKFBuild) -> tuple[np.ndarray, np.ndarray]:
    """`qpos` indices and encoder variances of the anchor chain's off-path joints.

    Resolved by NAME through the same per-joint table the filtered encoders use
    (`encoder_var_for_name`), so an ankle with no measured value falls back loudly
    exactly as a filtered joint would.
    """
    import mujoco

    from ..jointKF.state import encoder_var_for_name

    mj = model.mj_model
    cfg = load_config()["joint_kf"]
    qpos, var = [], []
    for dof in np.asarray(build.dof_anchor_unfiltered, dtype=int):
        j = int(np.flatnonzero(mj.jnt_dofadr == dof)[0])
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos.append(int(mj.jnt_qposadr[j]))
        var.append(encoder_var_for_name(name, cfg)[0])
    return np.array(qpos, dtype=int), np.array(var, dtype=float)


# ---------------------------------------------------------------------------
# Contact FK closure — base->foot vectors in the InEKF body frame
# ---------------------------------------------------------------------------

def _make_contact_kinematics(
    model: MjxModel, base_body_site: int, foot_site_ords: np.ndarray,
    aux_qpos: np.ndarray | None = None, n_filtered: int | None = None,
) -> inf.ContactKinematics:
    r"""The `robot/` seam: `q ↦ ContactFrames(y, J)` in the InEKF body frame `B`.

    `y_i = ᵂR_B^T (p_{foot_i} − p_B)` with the FK evaluated at the model's `qpos0`
    base pose (the base cancels — every quantity is base-relative). `B` is the
    `base_body_site` frame directly (the pelvis/root body), so its origin is `p_B`
    and its rotation is `ᵂR_B` — no `R_mount` here: the mount rotation is a
    *sensor* concern (the boundary), not a kinematics one. `J = ∂y/∂q` by
    forward-mode autodiff; `J_dot = 0` (a port TODO shared with the Tier-2 replay —
    the velocity-noise term is deferred, `inEKF/filter.py`).

    Off-path joints (`aux_qpos`)
    ----------------------------
    With `aux_qpos` given, `q` arrives as `concat(q_filtered, q_offpath)` and the
    off-path joints are scattered into `qpos` at their MEASURED values instead of
    staying at `qpos0`. On Alex those are the four ankles, and pinning them is not
    a small effect: they travel 0.66 rad while walking, which swings the base→sole
    vector by **5.3 cm over a gait cycle** (measured). A planted foot then appears
    to slide by that much every step, and a filter whose contacts are stationary by
    construction can only explain it as base motion — which is exactly the odometry
    drift it produces.

    This deliberately does NOT extend to `MjxModel.evaluate`: the mass matrix must
    keep seeing off-path joints at `qpos0`, because Mecano composites the ignored
    subtree's inertia once at construction and the joint-KF `Qa` parity depends on
    matching that (`MjxModel.qpos`, worth 14% on `diag(Qa)`). Live angles are right
    for kinematics and wrong for this model's inertia; the two uses are separate.
    """
    feet = jnp.asarray(foot_site_ords, dtype=int)
    use_aux = aux_qpos is not None and len(aux_qpos) > 0
    if use_aux:
        qpos0 = jnp.asarray(model.mj_model.qpos0, dtype=jnp.float64)
        idx_filtered = jnp.asarray(model.joint_qpos, dtype=int)
        idx_aux = jnp.asarray(aux_qpos, dtype=int)
        n_f = int(n_filtered if n_filtered is not None else model.n_joints)

    def _foot_y(q: Array) -> Array:
        if use_aux:
            q = qpos0.at[idx_filtered].set(q[:n_f]).at[idx_aux].set(q[n_f:])
        pos, rot = model.site_poses(q)
        p_base = pos[base_body_site]
        R_bw = rot[base_body_site]                       # ᵂR_B (body frame == site frame)
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
    aux_encoder_var = (jnp.asarray(fused.aux_encoder_var, dtype=jnp.float64)
                       if fused.n_aux else None)
    aux_qd_var = fused.aux_qd_var

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
            contact_meas_var, aux_encoder_var, aux_qd_var,
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
    contact_meas_var, aux_encoder_var=None, aux_qd_var=0.0,
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

    if aux_encoder_var is not None:
        # The contact FK also stands on the measured off-path joints, so the joint
        # vector it is handed is `concat(filtered, off-path)` and Σ_q grows to
        # match: the off-path block is their ENCODER variance (they are measured,
        # not estimated, so there is no cross-covariance with the filter states).
        # Widening Σ_q is not optional bookkeeping — `N = J Σ_q Jᵀ` would otherwise
        # claim the ankle contribution to the contact position is noise-free.
        q_hat = jnp.concatenate([q_hat, jnp.asarray(sensors.q_unfiltered, dtype=jnp.float64)])
        qd_hat = jnp.concatenate([qd_hat, jnp.asarray(sensors.qd_unfiltered, dtype=jnp.float64)])
        sigma_q_eff = jax.scipy.linalg.block_diag(sigma_q_eff, jnp.diag(aux_encoder_var))
        sigma_qd = jax.scipy.linalg.block_diag(
            sigma_qd, jnp.eye(aux_encoder_var.shape[0], dtype=jnp.float64) * aux_qd_var)

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
    q0_unfiltered: Array | None = None,
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
    # y_i = R0ᵀ(d_i − p0) holds and the first FK innovation is zero. With the
    # off-path joints wired in, the seed must use the SAME augmented vector the
    # running filter will — seeding at `qpos0` ankles and then updating against
    # measured ones injects the whole standing FK offset as a step at tick 1.
    q_seed = q0
    if fused.n_aux:
        aux0 = (jnp.zeros(fused.n_aux, dtype=jnp.float64) if q0_unfiltered is None
                else jnp.asarray(q0_unfiltered, dtype=jnp.float64))
        q_seed = jnp.concatenate([q0, aux0])
    frames = fused.kinematics(q_seed, jnp.zeros_like(q_seed))
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
