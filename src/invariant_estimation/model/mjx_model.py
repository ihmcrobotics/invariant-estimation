"""
model/mjx_model.py
==================
The **MJX adapter**: the one place where rigid-body kinematics and inertia enter
the estimator (`robot.RobotModel`, `CONTRACT_CARD.md` §5).

Why MJX rather than a hand-rolled CRB
-------------------------------------
`CLAUDE.md` §2 already routes production through MJX, so a hand-rolled composite
rigid-body algorithm would be throwaway code.  Worse, it would make the G3
armature-equivalence oracle a **tautology** -- the same hand writing `Lambda` and
its supposedly independent reference.  Because MuJoCo folds `dof_armature` into
`qM` *before* anything else sees it, that oracle instead becomes a genuine
two-model comparison (armature set vs armature zeroed), which is the whole reason
the synthetic chain fixture drives *this* adapter rather than a test-local
kinematics stub.

Two MuJoCo conventions this module depends on
--------------------------------------------
Both are pinned by explicit assertions in `tests/model/test_mjx_model.py`, so a
MuJoCo upgrade that changes either fails there rather than silently at G9 against
full Alex:

1. **A floating base's free joint owns DoF indices 0..5** (3 translational then 3
   rotational, both expressed in the *world* frame), and hinge DoFs follow in
   joint order.  `jnt_dofadr` / `jnt_type` are read at build time rather than
   assumed, but the nuisance gather (`base 6 DoF + gap joints`, `CLAUDE.md` §2)
   is only meaningful if this holds.
2. **`dof_armature` folds into `qM` as an exact diagonal add on its own DoF** and
   touches nothing else.  This is what makes `Lambda_eff = Lambda + diag(rotor)`
   fall out of the Schur complement for free -- and it is exactly why rotor
   inertia must **never** be added a second time post-Schur (`CLAUDE.md` §6, the
   double-add trap).

Constant-XLA-graph discipline (invariant I7)
--------------------------------------------
Every name -> index resolution happens **once, in plain Python, at construction**.
The jit-able methods close over integer index arrays only; no strings, no
data-dependent shapes, no Python branches on traced values.  `MjxModel` is a
frozen dataclass of static metadata plus the `mjx.Model` pytree, so a jitted step
can simply capture `self`.

Precision (invariant I8)
------------------------
`invariant_estimation/__init__.py` sets `jax_enable_x64=True` at import, so
`mjx.put_model` produces float64 arrays and every return value below is float64.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array
from mujoco import mjx

__all__ = ["MjxModel", "MassMatrixBlocks", "ModelEval"]

_MJ_JNT_FREE = int(mujoco.mjtJoint.mjJNT_FREE)      # 0
_MJ_JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)    # 3


class MassMatrixBlocks(NamedTuple):
    """The four gathers of `M(q)` the Schur complement needs.

    `j` = filtered joints (the estimator's `q`), `b` = nuisance DoFs (the floating
    base's 6 plus every "gap" joint that lies off the IMU chains).  Java takes the
    same four blocks off its *considered subsystem*; here they are pure index
    gathers off the full `qM`, which is equivalent and keeps the graph constant.

    `Lambda = M_jj - M_jb M_bb^-1 M_bj` (`jointKF/process.py`).  `M_bj` is
    `M_jb.T` up to round-off; both are returned so the consumer never has to
    decide which transpose MuJoCo happened to produce.
    """

    jj: Array   # (n, n)
    jb: Array   # (n, n_nuisance)
    bb: Array   # (n_nuisance, n_nuisance)
    bj: Array   # (n_nuisance, n)


class ModelEval(NamedTuple):
    """Everything the filter needs from the model at one configuration.

    The individual accessors below each run their own position-level pass, which
    is convenient but wasteful: the estimator wants FK, the site Jacobians and
    `M(q)` at the *same* `q`, once per tick.  `MjxModel.evaluate` is that single
    pass, and it is also what keeps the test gate affordable -- one MJX trace per
    shape instead of one per quantity.
    """

    site_pos: Array     # (n_sites, 3)
    site_rot: Array     # (n_sites, 3, 3)
    J_ang: Array        # (n_sites, 3, nv)   world frame
    J_rel: Array        # (n_pairs, 3, n)    child frame, S_ab applied
    M: Array            # (nv, nv)


@dataclass(frozen=True)
class MjxModel:
    """`robot.RobotModel` backed by MuJoCo/MJX.

    Construct with :meth:`from_xml_path` (the Gitman-vendored Alex MJCF drops in
    here unchanged), :meth:`from_xml_string` (the synthetic chain fixture), or
    :meth:`from_mj_model`.

    Parameters resolved at construction
    -----------------------------------
    site_names
        IMU / sole sites, in the order the estimator indexes them.  A "site
        ordinal" anywhere in this class means an index into this tuple, never a
        MuJoCo site id.
    pairs
        `(parent_site_ordinal, child_site_ordinal)` per IMU pair.
    joint_names
        The filtered joints, in state order.  `None` means "the union of hinge
        joints on the tree path of every pair", i.e. `CLAUDE.md` §2's rule.

    Attributes
    ----------
    dof_joint : (n,) int
        MuJoCo DoF index of each filtered joint -- the `j` gather.
    dof_nuisance : (n_nuisance,) int
        Base 6 DoF + gap-joint DoFs -- the `b` gather.  Complement of
        `dof_joint`, so `j` and `b` together tile all `nv` DoFs exactly once.
    pair_joint_mask : (n_pairs, n) float
        The selection `S_ab`: 1.0 on joints of that pair's tree path.  Applying it
        is redundant for a clean chain (an off-path joint either moves both sites
        identically and cancels in the difference, or moves neither), which is
        precisely what makes it a cheap structural assertion rather than a
        correction.
    """

    mj_model: mujoco.MjModel
    mjx_model: mjx.Model
    site_names: tuple[str, ...]
    site_ids: np.ndarray
    joint_names: tuple[str, ...]
    joint_dof: np.ndarray
    joint_qpos: np.ndarray
    dof_nuisance: np.ndarray
    pair_sites: np.ndarray          # (n_pairs, 2) site ordinals
    pair_joint_mask: np.ndarray     # (n_pairs, n) float

    # -- construction -------------------------------------------------------

    @classmethod
    def from_xml_path(cls, path, **kwargs) -> "MjxModel":
        """Build from an MJCF file -- the production entry point."""
        return cls.from_mj_model(mujoco.MjModel.from_xml_path(str(path)), **kwargs)

    @classmethod
    def from_xml_string(cls, xml: str, **kwargs) -> "MjxModel":
        """Build from an MJCF string -- used by the synthetic chain fixture."""
        return cls.from_mj_model(mujoco.MjModel.from_xml_string(xml), **kwargs)

    @classmethod
    def from_mj_model(
        cls,
        mj_model: mujoco.MjModel,
        *,
        site_names: Sequence[str] = (),
        pairs: Sequence[tuple[int, int]] = (),
        joint_names: Sequence[str] | None = None,
    ) -> "MjxModel":
        """Resolve every name to an index and hand the model to MJX.

        Structural rejections happen here, loudly, at build time -- `CLAUDE.md`
        §2 requires self-pairs and same-link pairs to be refused because both make
        the pair's `S` singular (a pair that brackets no joint measures nothing
        but bias).
        """
        site_names = tuple(site_names)
        site_ids = np.array(
            [_require_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, s) for s in site_names],
            dtype=int,
        ).reshape(len(site_names))

        pair_sites = np.array(pairs, dtype=int).reshape(len(pairs), 2)
        for e, (a, b) in enumerate(pair_sites):
            if a == b:
                raise ValueError(f"pair {e} is a self-pair on site '{site_names[a]}'")
            if mj_model.site_bodyid[site_ids[a]] == mj_model.site_bodyid[site_ids[b]]:
                raise ValueError(
                    f"pair {e} ('{site_names[a]}', '{site_names[b]}') sits on one body: "
                    "it brackets no joint, so its measurement Jacobian is identically zero"
                )

        # Tree path of every pair, in plain Python -- I7: never inside jit.
        path_joints = [
            _path_joints(mj_model, mj_model.site_bodyid[site_ids[a]], mj_model.site_bodyid[site_ids[b]])
            for a, b in pair_sites
        ]

        if joint_names is None:
            union = sorted({j for p in path_joints for j in p})
            joint_names = tuple(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in union)
        else:
            joint_names = tuple(joint_names)
        joint_ids = np.array(
            [_require_id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_names], dtype=int
        ).reshape(len(joint_names))
        bad = [n for n, i in zip(joint_names, joint_ids) if mj_model.jnt_type[i] != _MJ_JNT_HINGE]
        if bad:
            raise ValueError(f"filtered joints must be 1-DoF hinges; got non-hinge {bad}")

        joint_dof = mj_model.jnt_dofadr[joint_ids].astype(int)
        joint_qpos = mj_model.jnt_qposadr[joint_ids].astype(int)
        # Nuisance = the *considered but unfiltered* DoFs: the floating base, plus
        # the "gap" hinges that lie on a root->filtered-joint path without being
        # filter states.  NOT "everything that is not filtered": off-path joints
        # (Alex's ankles, arms, neck) are LOCKED by Java's considered-subsystem,
        # not marginalised, and eliminating them models them as free to
        # accelerate -- worth 58% on diag(Qa) against the hardware log.
        dof_nuisance = np.concatenate([
            _base_dofs(mj_model),
            np.array(sorted(_gap_dofs(mj_model, joint_ids)), dtype=int),
        ]).astype(int)

        n = len(joint_names)
        mask = np.zeros((len(pair_sites), n), dtype=float)
        for e, p in enumerate(path_joints):
            mask[e] = np.isin(joint_ids, np.array(sorted(p), dtype=int)).astype(float)

        return cls(
            mj_model=mj_model,
            mjx_model=mjx.put_model(mj_model),
            site_names=site_names,
            site_ids=site_ids,
            joint_names=joint_names,
            joint_dof=joint_dof,
            joint_qpos=joint_qpos,
            dof_nuisance=dof_nuisance,
            pair_sites=pair_sites,
            pair_joint_mask=mask,
        )

    # -- static dimensions --------------------------------------------------

    @property
    def n_joints(self) -> int:
        """Number of filtered joints -- the `n` of the state layout."""
        return len(self.joint_names)

    @property
    def n_sites(self) -> int:
        """Number of registered sites (IMUs + soles)."""
        return len(self.site_names)

    @property
    def n_pairs(self) -> int:
        """Number of IMU pairs."""
        return int(self.pair_sites.shape[0])

    @property
    def nv(self) -> int:
        """MuJoCo DoF count (`6 + n_hinges` for a floating base)."""
        return int(self.mj_model.nv)

    @property
    def nq(self) -> int:
        """MuJoCo generalised-coordinate count (`7 + n_hinges` for a free base)."""
        return int(self.mj_model.nq)

    # -- configuration plumbing ---------------------------------------------

    def qpos(self, q: Array) -> Array:
        """Widen a filtered-joint vector `(n,)` to a full `qpos` `(nq,)`.

        Accepts a full `qpos` unchanged, so callers may pass either.  The choice
        is made on `q.shape`, which is static under jit, so this stays trace-safe.
        Unspecified coordinates take the model's `qpos0` (identity base pose,
        gap joints at zero) -- the estimator never needs a base pose, since every
        quantity it consumes is either base-invariant (`M`, the joint blocks of
        `qM`) or a *relative* site quantity where the base cancels.

        **Off-path joints stay at `qpos0`, and that is load-bearing, not a
        shortcut.**  Mecano composites an ignored subtree's inertia into its
        parent exactly once, in `CompositeRigidBodyMassMatrixCalculator`'s
        *constructor* (`updateIgnoredSubtreeInertia`), and never refreshes it --
        so the Java filter's `M(q)` sees the ankles, arms and head welded at the
        configuration the robot model was **constructed** in, which is `q = 0`.
        Feeding live off-path angles here would be more physical but would stop
        matching Java: on the 2026-07-17 Alex001 log it moves `diag(Qa)` by up to
        14% (`tests/replay/test_java_parity.py`).
        """
        q = jnp.asarray(q, dtype=jnp.float64)
        if q.shape[-1] == self.nq:
            return q
        if q.shape[-1] != self.n_joints:
            raise ValueError(f"q must have {self.n_joints} (joints) or {self.nq} (qpos) entries, got {q.shape}")
        q0 = jnp.asarray(self.mj_model.qpos0, dtype=jnp.float64)
        return q0.at[jnp.asarray(self.joint_qpos)].set(q)

    def _data(self, q: Array) -> mjx.Data:
        """Position-level pipeline only: FK, COM frames, CRB.

        `mjx.forward` would additionally run collision detection and the
        constraint solver -- data-dependent work the estimator neither needs nor
        wants inside its jitted step (I7).
        """
        d = mjx.make_data(self.mjx_model).replace(qpos=self.qpos(q))
        d = mjx.kinematics(self.mjx_model, d)
        d = mjx.com_pos(self.mjx_model, d)
        return mjx.crb(self.mjx_model, d)

    # -- forward kinematics -------------------------------------------------

    def site_positions(self, q: Array) -> Array:
        """World positions of the registered sites, `(n_sites, 3)`."""
        return self._data(q).site_xpos[jnp.asarray(self.site_ids)]

    def site_rotations(self, q: Array) -> Array:
        """World rotations of the registered sites, `(n_sites, 3, 3)`.

        Row `k` is `^W R_{site_k}`; the IMU measurement frame *is* the site frame,
        so this is the rotation that carries a gyro reading into world.
        """
        return self._data(q).site_xmat[jnp.asarray(self.site_ids)]

    def site_poses(self, q: Array) -> tuple[Array, Array]:
        """`(positions, rotations)` in one FK pass."""
        d = self._data(q)
        ids = jnp.asarray(self.site_ids)
        return d.site_xpos[ids], d.site_xmat[ids]

    # -- Jacobians ----------------------------------------------------------

    def site_angular_jacobians(self, q: Array) -> Array:
        """World-frame angular Jacobians of the sites, `(n_sites, 3, nv)`.

        MJX's `mjx.jac` returns `(nv, 3)` -- transposed relative to the usual
        `(3, nv)` convention and relative to MuJoCo's own C API.  Transposing here
        keeps that surprise inside this module.
        """
        return self._site_angular_jacobians(self._data(q))

    def _site_angular_jacobians(self, d: mjx.Data) -> Array:
        """`site_angular_jacobians` on an already-computed `Data`."""
        ids = jnp.asarray(self.site_ids)
        bodies = jnp.asarray(self.mj_model.site_bodyid[self.site_ids])

        def one(point, bid):
            _, jr = mjx.jac(self.mjx_model, d, point, bid)
            return jr.T

        return jax.vmap(one)(d.site_xpos[ids], bodies)

    def relative_gyro_jacobian(self, q: Array) -> Array:
        r"""Stacked pair Jacobians `J_ang(q) S_ab`, `(n_pairs, 3, n)`, child frame.

        For pair `(a, b)` the gyro difference is

        .. math::
            \omega_b^{b} - {}^{b}R_{a}\,\omega_a^{a}
              = {}^{W}R_{b}^{T}\,(\omega_b^{W} - \omega_a^{W})
              = {}^{W}R_{b}^{T}\,(J^{W}_{ang,b} - J^{W}_{ang,a})\,\dot{q}_{full}

        so the *difference* is what makes this a relative measurement, and it is
        also what removes the base: the free joint's three rotational DoFs enter
        both site Jacobians as the same identity block and cancel exactly, while
        its translational DoFs contribute no angular velocity at all.  The
        base-DoF columns are therefore structurally zero, which is why gathering
        only the filtered-joint columns loses nothing -- see
        `tests/model/test_mjx_model.py::test_base_columns_cancel_in_the_pair_difference`.

        Expressed in the **child** frame, matching the Java convention
        (`GeometricJacobianCalculator`, angular block, child frame) and the state
        contract's `b_omega` being stored in each IMU's own frame.
        """
        d = self._data(q)
        return self._relative_gyro_jacobian(d, self._site_angular_jacobians(d))

    def _relative_gyro_jacobian(self, d: mjx.Data, J_world: Array) -> Array:
        """`relative_gyro_jacobian` given the world-frame site Jacobians."""
        ids = jnp.asarray(self.site_ids)
        cols = jnp.asarray(self.joint_dof)
        J_joint = J_world[:, :, cols]                     # (n_sites, 3, n)

        parent, child = jnp.asarray(self.pair_sites[:, 0]), jnp.asarray(self.pair_sites[:, 1])
        R_child = d.site_xmat[ids][child]                 # (n_pairs, 3, 3)
        diff = J_joint[child] - J_joint[parent]           # (n_pairs, 3, n), world
        J_rel = jnp.einsum("eji,ejk->eik", R_child, diff)  # ^W R_b^T @ diff
        return J_rel * jnp.asarray(self.pair_joint_mask)[:, None, :]

    # -- inertia ------------------------------------------------------------

    def mass_matrix(self, q: Array) -> Array:
        """Dense composite-rigid-body inertia `M(q)`, `(nv, nv)`, symmetric PD.

        Includes `dof_armature` on the diagonal -- MuJoCo folds it in during CRB.
        Consumers must not add reflected rotor inertia again (`CLAUDE.md` §6).
        """
        return self._mass_matrix(self._data(q))

    def _mass_matrix(self, d: mjx.Data) -> Array:
        """`mass_matrix` on an already-computed `Data`.

        The public accessor and `evaluate` must go through *this*, not each call
        `mjx.full_m` for itself: two spellings of the same quantity is exactly the
        arrangement where a test constrains one path and the estimator uses the
        other (that failure showed up in the Phase 0b mutation check).
        """
        return mjx.full_m(self.mjx_model, d)

    def evaluate(self, q: Array) -> ModelEval:
        """FK, site Jacobians and `M(q)` from a **single** position-level pass.

        This is the entry point a jitted filter step should call: `_data` (FK ->
        COM frames -> CRB) is by far the expensive part and there is no reason to
        run it three times for one tick.
        """
        d = self._data(q)
        ids = jnp.asarray(self.site_ids)
        J_ang = self._site_angular_jacobians(d)
        return ModelEval(
            site_pos=d.site_xpos[ids],
            site_rot=d.site_xmat[ids],
            J_ang=J_ang,
            J_rel=self._relative_gyro_jacobian(d, J_ang),
            M=self._mass_matrix(d),
        )

    def mass_matrix_blocks(self, q: Array) -> MassMatrixBlocks:
        """`(M_jj, M_jb, M_bb, M_bj)` gathered by the build-time DoF index arrays."""
        M = self.mass_matrix(q)
        j = jnp.asarray(self.joint_dof)
        b = jnp.asarray(self.dof_nuisance)
        return MassMatrixBlocks(
            jj=M[j][:, j],
            jb=M[j][:, b],
            bb=M[b][:, b],
            bj=M[b][:, j],
        )

    @property
    def armature(self) -> np.ndarray:
        """`dof_armature`, `(nv,)` -- the reflected rotor inertia MuJoCo already applied."""
        return np.asarray(self.mj_model.dof_armature, dtype=float)


# ---------------------------------------------------------------------------
# Build-time helpers -- plain Python, never traced
# ---------------------------------------------------------------------------

def _require_id(mj_model: mujoco.MjModel, obj: "mujoco.mjtObj", name: str) -> int:
    """`mj_name2id` with a useful error instead of a silent -1."""
    i = mujoco.mj_name2id(mj_model, obj, name)
    if i < 0:
        raise KeyError(f"no {obj} named '{name}' in the model")
    return i


def _ancestors(mj_model: mujoco.MjModel, body: int) -> list[int]:
    """Bodies from `body` up to (and including) the world body."""
    chain, b = [], int(body)
    while True:
        chain.append(b)
        if b == 0:
            return chain
        b = int(mj_model.body_parentid[b])


def _base_dofs(mj_model: mujoco.MjModel) -> np.ndarray:
    """DoF indices of the floating base -- the non-hinge joints at the tree root."""
    return np.array(
        [
            d
            for j in range(mj_model.njnt)
            if int(mj_model.jnt_type[j]) != _MJ_JNT_HINGE
            for d in range(
                int(mj_model.jnt_dofadr[j]),
                int(mj_model.jnt_dofadr[j]) + (6 if int(mj_model.jnt_type[j]) == 0 else 1),
            )
        ],
        dtype=int,
    )


def _gap_dofs(mj_model: mujoco.MjModel, joint_ids: np.ndarray) -> set[int]:
    """DoFs of hinges on a root->filtered path that are not themselves filtered.

    Java's `collectSpanningJoints`: walk from each filtered joint up to the
    floating base, collecting every joint passed.  What is collected but not
    filtered is a *gap* joint and gets marginalised with the base; what is never
    collected is off-path and stays locked inside the composited inertia.
    """
    filtered = set(int(i) for i in joint_ids)
    spanning: set[int] = set()
    joint_of_body: dict[int, list[int]] = {}
    for j in range(mj_model.njnt):
        joint_of_body.setdefault(int(mj_model.jnt_bodyid[j]), []).append(j)
    for i in filtered:
        for body in _ancestors(mj_model, int(mj_model.jnt_bodyid[i])):
            for j in joint_of_body.get(body, ()):
                if int(mj_model.jnt_type[j]) == _MJ_JNT_HINGE:
                    spanning.add(j)
    return {int(mj_model.jnt_dofadr[j]) for j in spanning - filtered}


def _path_joints(mj_model: mujoco.MjModel, body_a: int, body_b: int) -> set[int]:
    """Hinge joints strictly between two bodies in the kinematic tree.

    The path is `a -> lca -> b`; a joint belongs to it iff its child body lies on
    either branch below the least common ancestor.  Only hinges are returned: a
    free or slide joint on the path is *not* a filtered joint (the free base's
    contribution cancels in the pair difference anyway), and the state layout is
    1-DoF-per-joint by construction.
    """
    anc_a, anc_b = _ancestors(mj_model, body_a), _ancestors(mj_model, body_b)
    lca = next(b for b in anc_a if b in anc_b)
    branch = set(anc_a[:anc_a.index(lca)]) | set(anc_b[:anc_b.index(lca)])
    return {
        j
        for j in range(mj_model.njnt)
        if int(mj_model.jnt_bodyid[j]) in branch and int(mj_model.jnt_type[j]) == _MJ_JNT_HINGE
    }
