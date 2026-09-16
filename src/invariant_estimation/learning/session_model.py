"""Connect the existing MJX model to log-adapter model/kinematics seams."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from ..inEKF.filter import ContactFrames
from ..jointKF.anchors import anchor_jacobians, unfiltered_dof
from ..jointKF.filter import ModelInputs


class MjxSessionModel:
    """Name-resolved model binding; sites must be actual measurement frames.

    `body_site` is the pelvis estimation origin/orientation, not necessarily the
    base IMU site. `contact_sites` gives anchor order, which v1 also uses as the
    InEKF contact order. Filtered q is dynamic; remaining hinges use logged q.
    Mass uses the Java considered-subsystem convention: off-path hinges remain
    at construction qpos0, while filtered and nuisance gap joints remain live.
    """

    def __init__(self, model, build, *, body_site, contact_sites):
        self.model, self.build = model, build
        self.contact_names = tuple(contact_sites)
        if (
            tuple(model.joint_names) != tuple(build.joint_names)
            or not np.array_equal(model.joint_dof, build.dof_joint)
            or not np.array_equal(model.dof_nuisance, build.dof_nuisance)
        ):
            raise ValueError("MJX/build joint order or DoF mapping mismatch")
        if (
            len(self.contact_names) != build.n_anchors
            or len(set(self.contact_names)) != build.n_anchors
        ):
            raise ValueError("contact sites must match anchor count with unique names")
        self.imu_sites = np.array(
            [model.site_names.index(n) for n in build.imu_names], dtype=int
        )
        self.feet = np.array(
            [model.site_names.index(n) for n in self.contact_names], dtype=int
        )
        self.base = int(self.imu_sites[build.base_imu])
        self.body = model.site_names.index(body_site)
        pairs = np.column_stack(
            (
                self.imu_sites[np.asarray(build.pair_parent)],
                self.imu_sites[np.asarray(build.pair_child)],
            )
        )
        if not np.array_equal(pairs, model.pair_sites):
            raise ValueError("MJX/build IMU pair order mismatch")
        mj = model.mj_model
        # Rigidly attached, by MuJoCo's own weld groups rather than by body identity. imu_to_body is
        # computed once below at qpos0, so it is only valid for all time if the two sites cannot move
        # relative to each other -- which is what a shared weld id means: bodies joined by fixed
        # joints share one, and any joint between them breaks it.
        #
        # Same-body-id was too strict and ruled out the configuration the study actually wants. On
        # Alex the pelvis IMU lives on its own PELVIS_IMU_LINK, fixed-jointed to PELVIS_LINK and
        # yawed 90 degrees from it, so requiring one body forced body_site onto the IMU itself --
        # making the filter estimate the IMU frame rather than the pelvis that mocap registers and
        # the Java estimator reports.
        base_weld = int(mj.body_weldid[mj.site_bodyid[model.site_ids[self.base]]])
        body_weld = int(mj.body_weldid[mj.site_bodyid[model.site_ids[self.body]]])
        if base_weld != body_weld:
            raise ValueError(
                "base IMU and pelvis body site must be rigidly attached; "
                f"{body_site!r} and the base IMU are separated by a joint"
            )
        if any(int(t) not in (0, 3) for t in mj.jnt_type):
            raise ValueError("v1 supports free-base and hinge joints only")
        hinge = [i for i, t in enumerate(mj.jnt_type) if int(t) == 3]
        self.joint_names = tuple(
            mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinge
        )
        self.qpos_indices = np.array([mj.jnt_qposadr[i] for i in hinge], dtype=int)
        dofs = np.array([mj.jnt_dofadr[i] for i in hinge], dtype=int)
        by_dof = dict(zip(dofs, self.joint_names))
        self.unfiltered_names = tuple(by_dof[int(i)] for i in unfiltered_dof(build))
        self.filtered_indices = np.array(
            [self.joint_names.index(n) for n in build.joint_names], dtype=int
        )
        live = set(map(int, build.dof_joint)) | set(map(int, build.dof_nuisance))
        self.mass_live = np.array(
            [i for i, dof in enumerate(dofs) if int(dof) in live], dtype=int
        )
        _, rotations = model.site_poses(jnp.asarray(mj.qpos0))
        self.imu_to_body = np.asarray(rotations[self.body].T @ rotations[self.base])

    def _qpos(self, filtered_q, all_q, *, mass=False):
        q = jnp.asarray(all_q).at[self.filtered_indices].set(filtered_q)
        indices = self.mass_live if mass else np.arange(len(self.joint_names))
        return (
            jnp.asarray(self.model.mj_model.qpos0)
            .at[self.qpos_indices[indices]]
            .set(q[indices])
        )

    def model_inputs(self, filtered_q, all_q):
        ev = self.model.evaluate(self._qpos(filtered_q, all_q))
        parent, child = self.model.pair_sites.T
        relative = jnp.einsum("eji,ejk->eik", ev.site_rot[child], ev.site_rot[parent])
        anchors = anchor_jacobians(
            self.build, ev.J_ang, ev.site_rot, base_site=self.base, foot_sites=self.feet
        )
        mass = (
            self.model.mass_matrix(self._qpos(filtered_q, all_q, mass=True))
            if self.build.use_mass_matrix
            else None
        )
        return ModelInputs(ev.J_rel, relative, anchors, mass)

    def _positions(self, filtered_q, all_q):
        positions, rotations = self.model.site_poses(self._qpos(filtered_q, all_q))
        return jnp.einsum(
            "ji,kj->ki",
            rotations[self.body],
            positions[self.feet] - positions[self.body],
        )

    def contact_frames(self, filtered_q, filtered_qd, all_q, all_qd):
        jacobian = jax.jacfwd(self._positions, argnums=0)
        J, J_dot = jax.jvp(jacobian, (filtered_q, all_q), (filtered_qd, all_qd))
        return ContactFrames(self._positions(filtered_q, all_q), J, J_dot)
