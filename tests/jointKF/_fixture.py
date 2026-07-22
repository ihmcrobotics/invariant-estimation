"""Synthetic serial-revolute-chain fixture -- the port of Java's
`RandomFloatingRevoluteJointChain` + `JointLevelKFTestFixture` kinematics.

What this is for
----------------
`TEST_SUITE_MAP.md`'s "route 1": a floating-base serial revolute chain with joint
axes cycling X/Y/Z, randomised (but seeded) link offsets, and IMU sites on named
links.  Every kinematics-dependent ported class -- the stacked gyro measurement,
the stance anchors, the mass-matrix process noise -- drives *this* chain rather
than the real robot, exactly as the Java suite does.  The Java tests depend only
on self-consistency between the fixture's kinematics and the filter's, never on
Mecano's particular random geometry, so the numbers here need not match Java's.

**The chain drives the production adapter** (`model.mjx_model.MjxModel`), not a
test-local kinematics stub (`CONTRACT_CARD.md` §5).  That is what keeps the G3
armature-equivalence oracle honest: with a hand-rolled CRB on one side it would
be the same hand writing both sides of the comparison.

Indexing convention
-------------------
`C = shape["chain"]` hinge joints, `joint_i` connecting `link_{i-1}` to `link_i`
(`link_0` hangs off the floating base).  An entry of `shape["imus"]` is a **link
index**, so the joints strictly between IMU links `p < c` are `joint_{p+1} ..
joint_c`, i.e. `n = c - p`.  That reproduces all four `_oracles.SHAPES` entries
exactly: (1,9)->8, (1,5)->4, (0,3)->3, and (1,5,9) with two pairs -> 8 as a union.

`apply_consistent_motion` -- the oracle everything else leans on
----------------------------------------------------------------
Zero base twist, and every IMU's gyro is its link's body-frame angular velocity
recomputed from the commanded `(q, qd)` by an explicit recursion that never calls
MJX.  Zero base twist is what makes the pair measurement exact rather than
approximate: `omega_child - R_cp omega_parent` cancels the base term identically,
so the relative gyro equals `J_ang @ qd` to round-off.  Every tracking tolerance
downstream assumes that exactness -- if this recursion and the adapter's Jacobian
ever disagree, the measurement tests are measuring the disagreement, not the
filter (`tests/model/test_mjx_model.py` pins it).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from invariant_estimation.jointKF.build import KinematicTree
from invariant_estimation.model.mjx_model import MjxModel

from ._oracles import SHAPES, java_hash_code

#: One seed for the whole file: the geometry must be identical across runs and
#: across agents, or two ported test classes silently disagree about the model.
CHAIN_SEED = 20260722

#: Hinge axes cycle X/Y/Z by `i % 3` -- Java `RandomFloatingRevoluteJointChain`.
#: Cycling (rather than randomising) axes guarantees the chain is never
#: degenerate: three consecutive joints always span R^3.
AXES = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))

#: Per-joint reflected rotor inertia, cycled over the chain.  Values are taken
#: from the real `rotor_inertia` table (`config/filter_cfg.yaml`) so the armature
#: oracle runs on a physically plausible spread rather than one constant, which
#: would not distinguish `diag(rotor)` from `rotor * I`.
ROTOR_CYCLE = (0.062, 0.020, 0.167, 0.167, 0.070, 0.050, 0.067, 0.022)


def _quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion in MuJoCo `(w, x, y, z)` order."""
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """`(w, x, y, z)` -> rotation matrix.  NumPy, so the fixture never needs MJX."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit `axis`."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class ChainGeometry(NamedTuple):
    """The randomised numbers the MJCF was written from.

    Kept alongside the model so an oracle can rebuild the chain's kinematics
    without parsing MJCF back or querying MuJoCo -- that independence is the
    point.
    """

    link_pos: np.ndarray        # (C, 3)   body offset from parent, parent frame
    link_quat: np.ndarray       # (C, 4)   body orientation offset, parent frame
    axis: np.ndarray            # (C, 3)   hinge axis, child body frame
    mass: np.ndarray            # (C,)
    com: np.ndarray             # (C, 3)   inertial position, body frame
    inertia_quat: np.ndarray    # (C, 4)   principal-axis orientation, body frame
    inertia_diag: np.ndarray    # (C, 3)
    armature: np.ndarray        # (C,)
    site_link: np.ndarray       # (S,)     link index carrying each site
    site_pos: np.ndarray        # (S, 3)
    site_quat: np.ndarray       # (S, 4)
    base_mass: float
    base_inertia: np.ndarray    # (3,)


def chain_geometry(shape: dict, *, armature: bool = True, seed: int = CHAIN_SEED) -> ChainGeometry:
    """Draw the chain's geometry deterministically from `seed`.

    `armature=False` zeroes `dof_armature` while leaving every other number
    untouched -- the two-model comparison the G3 oracle needs.
    """
    C = shape["chain"]
    # Per-shape stream, keyed on the shape *name* rather than its position in
    # SHAPES: two shapes must not share geometry (a bug that only shows up as
    # suspiciously identical numbers), and the keying must survive the table
    # being reordered or a one-off shape being built ad hoc.
    rng = np.random.default_rng(seed + (java_hash_code(shape["name"]) % 10_000))

    sites = list(shape["imus"]) + [C - 1]            # IMU sites, then the foot site
    n_sites = len(sites)
    # Principal moments must obey the triangle inequality or MuJoCo refuses the
    # model; parameterising them as pairwise sums of positive numbers (the
    # "inertia of a point-mass triple" form) satisfies it by construction.
    half = rng.uniform(0.005, 0.03, size=(C, 3))
    diag = np.stack([half[:, 1] + half[:, 2], half[:, 2] + half[:, 0], half[:, 0] + half[:, 1]], axis=1)
    return ChainGeometry(
        link_pos=rng.uniform(-0.25, 0.25, size=(C, 3)),
        link_quat=np.stack([_quat(rng) for _ in range(C)]),
        axis=np.stack([AXES[i % 3] for i in range(C)]),
        mass=rng.uniform(0.5, 3.0, size=C),
        com=rng.uniform(-0.08, 0.08, size=(C, 3)),
        inertia_quat=np.stack([_quat(rng) for _ in range(C)]),
        inertia_diag=diag,
        armature=np.array([ROTOR_CYCLE[i % len(ROTOR_CYCLE)] for i in range(C)]) if armature
        else np.zeros(C),
        site_link=np.array(sites, dtype=int),
        site_pos=rng.uniform(-0.05, 0.05, size=(n_sites, 3)),
        site_quat=np.stack([_quat(rng) for _ in range(n_sites)]),
        base_mass=7.5,
        base_inertia=np.array([0.12, 0.19, 0.23]),
    )


def _fmt(v) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.ravel(v))


def chain_mjcf(shape: dict, geom: ChainGeometry) -> str:
    """Emit MJCF for the chain.

    Inertias are given explicitly (`<inertial>`), so the bodies need no geoms and
    MuJoCo infers nothing: every mass property in the model is a number this
    fixture chose and the kinetic-energy oracle can reuse.
    """
    C = len(geom.mass)
    site_by_link: dict[int, list[int]] = {}
    for s, link in enumerate(geom.site_link):
        site_by_link.setdefault(int(link), []).append(s)

    def site_xml(link: int) -> str:
        out = []
        for s in site_by_link.get(link, []):
            name = "foot" if s == len(geom.site_link) - 1 else f"imu{s}"
            out.append(f'<site name="{name}" pos="{_fmt(geom.site_pos[s])}" '
                       f'quat="{_fmt(geom.site_quat[s])}"/>')
        return "".join(out)

    body = ""
    for i in reversed(range(C)):
        body = (
            f'<body name="link{i}" pos="{_fmt(geom.link_pos[i])}" quat="{_fmt(geom.link_quat[i])}">'
            f'<joint name="joint{i}" type="hinge" axis="{_fmt(geom.axis[i])}" '
            f'armature="{geom.armature[i]:.17g}" limited="false"/>'
            f'<inertial pos="{_fmt(geom.com[i])}" quat="{_fmt(geom.inertia_quat[i])}" '
            f'mass="{geom.mass[i]:.17g}" diaginertia="{_fmt(geom.inertia_diag[i])}"/>'
            f'{site_xml(i)}{body}</body>'
        )
    return (
        '<mujoco model="revolute_chain">'
        '<compiler angle="radian"/>'
        '<option gravity="0 0 -9.81"/>'
        '<worldbody>'
        '<body name="base" pos="0 0 1">'
        '<freejoint name="floating_base"/>'
        f'<inertial pos="0 0 0" mass="{geom.base_mass:.17g}" '
        f'diaginertia="{_fmt(geom.base_inertia)}"/>'
        f'{site_xml(-1)}{body}'
        '</body></worldbody></mujoco>'
    )


class ConsistentMotion(NamedTuple):
    """Output of :meth:`ChainFixture.apply_consistent_motion`.

    `gyro` is what the IMUs would report with a perfect sensor: the link's angular
    velocity in the *site* frame, base held still.
    """

    qpos: np.ndarray       # (nq,)  full MuJoCo configuration
    qvel: np.ndarray       # (nv,)  full MuJoCo velocity, base DoFs exactly zero
    q: np.ndarray          # (n,)   filtered-joint positions
    qd: np.ndarray         # (n,)   filtered-joint velocities
    gyro: np.ndarray       # (m, 3) per-IMU body-frame angular velocity
    site_rot: np.ndarray   # (S, 3, 3) world rotation of every site
    link_rot: np.ndarray   # (C, 3, 3) world rotation of every link
    link_omega: np.ndarray # (C, 3) world angular velocity of every link


@dataclass(frozen=True)
class ChainFixture:
    """One `_oracles.SHAPES` entry, realised as a model plus its index arrays."""

    name: str
    model: MjxModel
    geometry: ChainGeometry
    n: int                      # filtered joints
    m: int                      # distinct IMUs
    n_chain: int                # total hinge joints in the chain
    joint_names: tuple[str, ...]
    imu_names: tuple[str, ...]
    imu_site_ids: np.ndarray    # MuJoCo site ids of the IMU sites
    imu_links: np.ndarray       # link index carrying each IMU
    pairs: np.ndarray           # (n_pairs, 2) IMU ordinals
    dof_joint: np.ndarray       # (n,) MuJoCo DoF of each filtered joint
    dof_nuisance: np.ndarray    # (6 + n_gap,) base DoFs + gap joints
    chain_joint_of_filtered: np.ndarray   # (n,) chain index of each filtered joint
    foot_link: int
    foot_site: str

    # -- kinematics recomputed WITHOUT MJX ---------------------------------

    def forward(self, q_chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Independent FK: world `(positions, rotations)` of every link.

        Plain homogeneous-transform composition down the chain, base at the
        model's `qpos0` pose (identity orientation).  This is the reference route
        for `tests/model/test_mjx_model.py`'s FK check and the substrate of
        :meth:`apply_consistent_motion`.
        """
        g = self.geometry
        p, R = np.zeros((self.n_chain, 3)), np.zeros((self.n_chain, 3, 3))
        p_par, R_par = np.array([0.0, 0.0, 1.0]), np.eye(3)      # base body pose
        for i in range(self.n_chain):
            R_off = quat_to_mat(g.link_quat[i])
            p[i] = p_par + R_par @ g.link_pos[i]
            R[i] = R_par @ R_off @ axis_angle_to_mat(g.axis[i], float(q_chain[i]))
            p_par, R_par = p[i], R[i]
        return p, R

    def apply_consistent_motion(self, q, qd, *, q_gap=None) -> ConsistentMotion:
        """Commanded `(q, qd)` -> the sensor set a perfect robot would produce.

        Base twist is exactly zero, so `omega_link_i = sum_{j<=i} R_j a_j qd_j`
        with no base term -- and therefore
        `gyro_child - R_child^T R_parent gyro_parent = J_ang(q) qd` **exactly**,
        not to first order.  Gap joints (off every IMU chain) are held at
        `q_gap` with zero velocity: they change the geometry, which is the point
        of having them, but they contribute nothing to any gyro.
        """
        q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
        assert q.shape == (self.n,) and qd.shape == (self.n,)

        q_chain = np.zeros(self.n_chain) if q_gap is None else np.array(q_gap, dtype=float)
        qd_chain = np.zeros(self.n_chain)
        q_chain[self.chain_joint_of_filtered] = q
        qd_chain[self.chain_joint_of_filtered] = qd

        _, R = self.forward(q_chain)
        omega = np.cumsum(R @ self.geometry.axis[..., None] * qd_chain[:, None, None], axis=0)[..., 0]

        g = self.geometry
        site_rot = np.stack([
            (np.eye(3) if link < 0 else R[link]) @ quat_to_mat(g.site_quat[s])
            for s, link in enumerate(g.site_link)
        ])
        site_omega = np.stack([
            np.zeros(3) if link < 0 else omega[link] for link in g.site_link
        ])
        gyro = np.einsum("sji,sj->si", site_rot[:self.m], site_omega[:self.m])

        nq, nv = self.model.nq, self.model.nv
        qpos = np.array(self.model.mj_model.qpos0, dtype=float).reshape(nq)
        qpos[self.model.mj_model.jnt_qposadr[1:]] = q_chain
        qvel = np.zeros(nv)
        qvel[self.model.mj_model.jnt_dofadr[1:]] = qd_chain
        return ConsistentMotion(qpos=qpos, qvel=qvel, q=q, qd=qd, gyro=gyro,
                                site_rot=site_rot, link_rot=R, link_omega=omega)

    def random_q(self, rng: np.random.Generator) -> np.ndarray:
        """A filtered-joint configuration, drawn wide enough to be non-degenerate."""
        return rng.uniform(-1.2, 1.2, size=self.n)


def build_fixture(shape: dict, *, armature: bool = True, seed: int = CHAIN_SEED) -> ChainFixture:
    """Realise one `SHAPES` entry.

    `armature=False` reuses identical geometry with `dof_armature` zeroed, which
    is the second model in the armature-equivalence oracle.
    """
    C = shape["chain"]
    geom = chain_geometry(shape, armature=armature, seed=seed)
    imu_names = tuple(f"imu{k}" for k in range(len(shape["imus"])))
    site_names = imu_names + ("foot",)

    model = MjxModel.from_xml_string(
        chain_mjcf(shape, geom), site_names=site_names, pairs=tuple(shape["pairs"]),
    )

    n, m = shape["n"], shape["m"]
    assert model.n_joints == n, (
        f"{shape['name']}: chain resolved {model.n_joints} filtered joints, expected {n} -- "
        "the SHAPES table and the link-index convention disagree"
    )
    assert len(imu_names) == m

    chain_of_filtered = np.array([int(nm.removeprefix("joint")) for nm in model.joint_names])
    return ChainFixture(
        name=shape["name"],
        model=model,
        geometry=geom,
        n=n,
        m=m,
        n_chain=C,
        joint_names=model.joint_names,
        imu_names=imu_names,
        imu_site_ids=model.site_ids[:m],
        imu_links=np.array(shape["imus"], dtype=int),
        pairs=np.array(shape["pairs"], dtype=int),
        dof_joint=model.joint_dof,
        dof_nuisance=model.dof_nuisance,
        chain_joint_of_filtered=chain_of_filtered,
        foot_link=C - 1,
        foot_site="foot",
    )


@lru_cache(maxsize=None)
def _cached(name: str, armature: bool) -> ChainFixture:
    shape = next(s for s in SHAPES if s["name"] == name)
    return build_fixture(shape, armature=armature)


def fixture(name: str, *, armature: bool = True) -> ChainFixture:
    """Cached lookup by `SHAPES` name -- building an `mjx.Model` is not free."""
    return _cached(name, armature)


def all_fixtures(*, armature: bool = True) -> tuple[ChainFixture, ...]:
    """All four shapes, in `SHAPES` order."""
    return tuple(fixture(s["name"], armature=armature) for s in SHAPES)


def kinematic_tree(fixture: ChainFixture) -> KinematicTree:
    """Describe a `ChainFixture`'s MuJoCo model as a `build.KinematicTree`.

    `build.py` is deliberately model-agnostic (it takes a plain tree, not an
    `mjx.Model`), so this adapter is what lets the *real* build run on the *real*
    fixture geometry.  Without it the anchor tests would have to hand-write both
    the F/U split and the Jacobians, and the F/U split is precisely the thing
    `build.py` got wrong once already (PORT_NOTES: "anchor chains root at the
    base IMU").
    """
    mj = fixture.model.mj_model
    hinge = [j for j in range(mj.njnt) if int(mj.jnt_type[j]) == 3]
    free = [j for j in range(mj.njnt) if int(mj.jnt_type[j]) == 0]
    base_dofs = np.concatenate(
        [np.arange(mj.jnt_dofadr[j], mj.jnt_dofadr[j] + 6) for j in free]
    ) if free else np.zeros(0, dtype=int)
    import mujoco
    site_body = {
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, s): int(mj.site_bodyid[s])
        for s in range(mj.nsite)
    }
    return KinematicTree(
        joint_names=tuple(
            mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge
        ),
        joint_body=np.array([int(mj.jnt_bodyid[j]) for j in hinge]),
        body_parent=np.array(mj.body_parentid, dtype=int),
        joint_dof=np.array([int(mj.jnt_dofadr[j]) for j in hinge]),
        base_dofs=base_dofs.astype(int),
        site_body=site_body,
        tau_max=np.full(len(hinge), np.nan),
    )
