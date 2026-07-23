"""Phase 0b verification gate for the MJX model seam -- and G1's model-layer sign-off.

This file has no Java analogue: it is the port-specific oracle set that decides
whether `MjxModel` may be trusted as *the* source of `M(q)` and `J_ang(q)` for the
joint KF.  Everything downstream (the Schur complement, the stacked gyro
measurement, the stance anchors) is validated only *relative* to this seam, so an
error here is invisible everywhere else -- the four ported test classes would
agree perfectly with each other while all being wrong together.

Rule for every check below: **reach `M` or `J` by a route independent of the MJX
call under test.**

    FK           -> hand-rolled homogeneous transforms (`ChainFixture.forward`)
    J_ang        -> `jax.jvp` of a hand-rolled FK  (autodiff, not MJX's analytic
                    Jacobian path)
    M            -> kinetic energy accumulated from link twists and each body's
                    spatial inertia (Newton-Euler bookkeeping, not CRB)
    armature     -> two-model difference, armature set vs zeroed
    J_ang @ qd   -> the fixture's own gyro recursion

Two MuJoCo conventions are pinned here on purpose (`test_convention_*`): a
floating base's DoF layout, and armature folding into `qM`.  Both are load-bearing
for the nuisance gather and for `Lambda_eff = Lambda + diag(rotor)`.  Pinning them
means a MuJoCo upgrade that changes either fails in this file, against a 4-joint
chain, rather than at G9 against full Alex.

Performance note: an MJX position pass costs seconds, so every model call goes
through `_evaluated()`, which vmaps `MjxModel.evaluate` over **one shared batch**
of `N_Q` random configurations and caches the result -- the whole file pays one
MJX trace per shape, not one per assertion.  That is also the repo convention
(`CONTRACT_CARD.md` §7: vmap the trial loop rather than Python-looping it).
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jointKF._fixture import AXES, ChainFixture, all_fixtures, fixture, quat_to_mat
from jointKF._oracles import SHAPES, assert_symmetric

SHAPE_NAMES = [s["name"] for s in SHAPES]
SEED = 20260722
N_Q = 20        # the "20 random q" the Phase 0b gate calls for


@pytest.fixture(params=SHAPE_NAMES)
def chain(request) -> ChainFixture:
    """Each of the four `SHAPES`, so no check passes on one topology by accident."""
    return fixture(request.param)


@lru_cache(maxsize=None)
def _configs(name: str) -> np.ndarray:
    """The shared `(N_Q, n)` batch of filtered-joint configurations.

    Wide enough (+-1.2 rad) that the chain is nowhere near a linearisation of its
    home pose, and identical across tests so the jitted batch is compiled once.
    """
    ch = fixture(name)
    return np.random.default_rng(SEED).uniform(-1.2, 1.2, size=(N_Q, ch.n))


@lru_cache(maxsize=None)
def _evaluated(name: str, armature: bool = True) -> dict:
    """Every model quantity, vmapped over `_configs(name)`, computed once.

    Deliberately **not** wrapped in `jax.jit`: XLA's CPU compile time for MJX's
    position pipeline grows explosively with kinematic depth on this machine
    (~1.3 s for the 4-link chain, ~240 s for the 10-link one), while eager vmap
    over all 20 configurations costs a few seconds regardless.  `jit`-ability is a
    real requirement (I7) and is asserted separately, on the shallowest shape.
    """
    ch = fixture(name, armature=armature)
    qs = jnp.asarray(_configs(name))
    ev = jax.vmap(ch.model.evaluate)(qs)
    return {"q": np.asarray(qs)} | {k: np.asarray(v) for k, v in ev._asdict().items()}


def _hat(w):
    return jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _vee(W):
    return 0.5 * jnp.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])


def _chain_q(chain: ChainFixture, q: np.ndarray) -> np.ndarray:
    """Scatter filtered-joint values into the full chain (gap joints at zero)."""
    full = np.zeros(chain.n_chain)
    full[chain.chain_joint_of_filtered] = q
    return full


# ---------------------------------------------------------------------------
# Convention pins -- fail HERE on a MuJoCo upgrade, not at G9
# ---------------------------------------------------------------------------

def test_convention_floating_base_owns_the_first_six_dofs(chain):
    """Free joint = joint 0, DoFs 0..5; hinges follow in joint order, one DoF each.

    `dof_nuisance` is "base 6 + gap joints" (`CLAUDE.md` §2), where a gap joint
    lies on a `root -> filtered` path without being a filter state.  Every
    `SHAPES` entry puts its IMUs so that the only unfiltered hinges are *above*
    the filtered span, so here -- and only here -- "base + gap" coincides with
    "everything that is not filtered", which is why the last assertion holds.  It
    does not hold on Alex, whose ankles hang below the filtered span and are
    locked rather than marginalised; `tests/jointKF/test_build.py` covers that
    case with a foot site beyond the IMUs.
    """
    m = chain.model.mj_model
    assert m.jnt_type[0] == 0, "joint 0 must be the free joint (mjJNT_FREE == 0)"
    assert m.jnt_dofadr[0] == 0
    assert list(m.jnt_type[1:]) == [3] * (m.njnt - 1), "chain joints are hinges (mjJNT_HINGE == 3)"
    assert list(m.jnt_dofadr[1:]) == list(range(6, m.nv)), "hinge DoFs follow the base's six"
    assert m.nv == 6 + chain.n_chain and m.nq == 7 + chain.n_chain

    gap = [int(m.jnt_dofadr[j + 1]) for j in range(chain.n_chain)
           if f"joint{j}" not in chain.joint_names]
    assert list(chain.dof_nuisance) == list(range(6)) + gap, "nuisance = base 6, then gap joints"
    assert sorted(list(chain.dof_joint) + list(chain.dof_nuisance)) == list(range(m.nv))


def test_convention_free_joint_dofs_are_world_linear_then_world_angular(chain):
    """DoFs 0..2 are world linear velocity of the base origin, 3..5 world angular.

    Read off the site angular Jacobian: a purely translational DoF induces no
    angular velocity anywhere, and the three rotational DoFs give the identity in
    *world* axes at every site (a body-frame convention would give `R^T`).
    """
    J = _evaluated(chain.name)["J_ang"]
    assert np.all(J[:, :, :, 0:3] == 0.0), "translational base DoFs must not induce rotation"
    eye = np.broadcast_to(np.eye(3), J[:, :, :, 3:6].shape)
    np.testing.assert_allclose(J[:, :, :, 3:6], eye, rtol=0.0, atol=0.0)


def test_convention_armature_is_an_exact_diagonal_add_to_qM(chain):
    """`M(armature) - M(0) == diag(dof_armature)`, exactly, at 20 configurations.

    The foundation of the G3 armature-equivalence claim: the add is diagonal and
    touches `M_jb` not at all, so `jointKF/process.py` gets its rotor inertia for
    free out of `qM` and must never add it a second time post-Schur (`CLAUDE.md`
    §6, the double-add trap).  How that diagonal propagates through the Schur
    complement is a separate statement -- see the two tests below, which are more
    careful about it than the §2 shorthand.
    """
    arm = np.asarray(chain.model.mj_model.dof_armature, dtype=float)
    assert np.all(arm[6:] > 0.0) and np.all(arm[:6] == 0.0), "fixture arms the hinges only"
    assert np.all(fixture(chain.name, armature=False).model.mj_model.dof_armature == 0.0)

    M = _evaluated(chain.name)["M"]
    M0 = _evaluated(chain.name, armature=False)["M"]
    np.testing.assert_allclose(M - M0, np.broadcast_to(np.diag(arm), M.shape),
                               rtol=0.0, atol=1e-14)


def _schur(M, j, b):
    """`Lambda = M_jj - M_jb M_bb^-1 M_bj` -- the reference form, explicit solve."""
    return M[np.ix_(j, j)] - M[np.ix_(j, b)] @ np.linalg.solve(M[np.ix_(b, b)], M[np.ix_(b, j)])


def test_armature_enters_the_schur_complement_on_both_partitions(chain):
    """`Schur(M + diag(a)) == Schur(M_0 shifted by a on **both** j and b)`.

    Careful with `CLAUDE.md` §2's shorthand "armature never touches `M_bb`".  It
    is true of the *base* six DoFs, which carry no armature -- but a **gap joint**
    is a nuisance DoF that does, so its rotor inertia legitimately lands inside
    `M_bb` and is felt through the marginalisation.  Java does the same thing
    (§2: "nuisance rotor diag on gap joints, zero on the base 6 DoF"), so the two
    agree; what does *not* hold in general is the naive
    `Lambda_eff = Lambda + diag(rotor_j)`.  See the next test for the exact
    condition under which it does.
    """
    j, b = chain.dof_joint, chain.dof_nuisance
    arm = np.asarray(chain.model.mj_model.dof_armature, dtype=float)
    assert np.any(arm[b] > 0.0), "this shape must have armed gap joints, or the test is vacuous"

    M, M0 = _evaluated(chain.name)["M"], _evaluated(chain.name, armature=False)["M"]
    for k in range(5):
        shifted = M0[k] + np.diag(arm)
        np.testing.assert_allclose(_schur(M[k], j, b), _schur(shifted, j, b), rtol=0.0, atol=1e-12)


def test_armature_on_filtered_joints_alone_is_a_post_schur_diagonal_add(chain):
    """`Schur(M_0 + diag(a_j)) == Schur(M_0) + diag(a_j)`, exactly -- the G3 claim.

    With armature **only** on the filtered joints, the add lands entirely inside
    `M_jj` and the marginalisation cannot see it, so MuJoCo's pre-Schur armature
    and Java's post-Schur rotor term are the same number.  That equivalence is what
    retires the double-add trap (`CLAUDE.md` §6): `jointKF/process.py` must take
    `Lambda_eff` straight from the MJX `qM` and never add rotor inertia again.

    A third model is built here (armature on the filtered joints, zero on the gap
    joints) rather than reusing a fixture, because the fixture arms every joint --
    and on that model the identity above is *false*, which is precisely the
    distinction worth pinning.
    """
    from invariant_estimation.model.mjx_model import MjxModel
    from jointKF._fixture import chain_geometry, chain_mjcf

    shape = next(s for s in SHAPES if s["name"] == chain.name)
    g = chain_geometry(shape)
    filtered_only = np.zeros_like(g.armature)
    filtered_only[chain.chain_joint_of_filtered] = g.armature[chain.chain_joint_of_filtered]
    model = MjxModel.from_xml_string(
        chain_mjcf(shape, g._replace(armature=filtered_only)),
        site_names=chain.model.site_names, pairs=tuple(shape["pairs"]),
    )

    j, b = chain.dof_joint, chain.dof_nuisance
    rotor = filtered_only[chain.chain_joint_of_filtered]
    M0 = _evaluated(chain.name, armature=False)["M"]
    for k in range(3):
        M = np.asarray(model.mass_matrix(_configs(chain.name)[k]))
        np.testing.assert_allclose(_schur(M, j, b), _schur(M0[k], j, b) + np.diag(rotor),
                                   rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 1. Forward kinematics vs hand-rolled homogeneous transforms
# ---------------------------------------------------------------------------

def test_site_fk_matches_hand_rolled_chain_fk(chain):
    """MJX site poses vs plain transform composition, 20 random `q`, <= 1e-10.

    The reference walks `T_i = T_{i-1} T_off(link_i) T_rot(axis_i, q_i)` and then
    applies the site offset -- no MuJoCo call anywhere in it.
    """
    g, ev = chain.geometry, _evaluated(chain.name)
    for k, q in enumerate(ev["q"]):
        p_link, R_link = chain.forward(_chain_q(chain, q))
        expect_p = np.stack([p_link[b] + R_link[b] @ g.site_pos[s] for s, b in enumerate(g.site_link)])
        expect_R = np.stack([R_link[b] @ quat_to_mat(g.site_quat[s]) for s, b in enumerate(g.site_link)])
        np.testing.assert_allclose(ev["site_pos"][k], expect_p, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(ev["site_rot"][k], expect_R, rtol=0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Angular Jacobians vs autodiff of the FK rotation
# ---------------------------------------------------------------------------

def _site_rotations_jax(chain: ChainFixture, q_chain, u):
    """Hand-rolled FK rotations as a function of a DoF perturbation `u` (nv,).

    `u` is parameterised to match MuJoCo's DoF ordering exactly: `u[0:3]` shifts
    the base (no effect on rotation), `u[3:6]` **left**-multiplies the base by
    `exp(hat(.))` -- a world-frame rotation about the base origin -- and `u[6:]`
    perturbs the hinge angles.  Differentiating at `u = 0` therefore yields
    exactly the columns of the world-frame angular Jacobian, by a route that
    shares no code with `mjx.jac`.
    """
    g = chain.geometry
    R = jax.scipy.linalg.expm(_hat(u[3:6]))
    R_link = []
    for i in range(chain.n_chain):
        angle = q_chain[i] + u[6 + i]
        K = _hat(jnp.asarray(g.axis[i]))
        R_joint = jnp.eye(3) + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)
        R = R @ jnp.asarray(quat_to_mat(g.link_quat[i])) @ R_joint
        R_link.append(R)
    return jnp.stack([R_link[b] @ jnp.asarray(quat_to_mat(g.site_quat[s]))
                      for s, b in enumerate(g.site_link)])


def test_site_angular_jacobian_matches_autodiff_of_fk(chain):
    """`J_world(q) @ qdot` vs `vee(dR R^T)` from a JVP of the hand-rolled FK.

    Autodiff of an independently written FK is the only reference here genuinely
    disjoint from MJX's analytic Jacobian implementation.  `qdot` spans *all* `nv`
    DoFs, so the base columns are exercised as well as the hinges.
    """
    ev = _evaluated(chain.name)
    rng = np.random.default_rng(SEED + 1)
    qdots = jnp.asarray(rng.normal(size=(6, chain.model.nv)))
    q_chain = jnp.asarray(np.stack([_chain_q(chain, q) for q in ev["q"][:6]]))

    def omega_ad(qc, qdot):
        R, dR = jax.jvp(lambda u: _site_rotations_jax(chain, qc, u),
                        (jnp.zeros(chain.model.nv),), (qdot,))
        return jax.vmap(lambda dr, r: _vee(dr @ r.T))(dR, R)

    want = np.asarray(jax.jit(jax.vmap(omega_ad))(q_chain, qdots))
    got = np.einsum("ksdv,kv->ksd", ev["J_ang"][:6], np.asarray(qdots))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-10)


def test_base_columns_cancel_in_the_pair_difference(chain):
    """The base's six DoFs contribute nothing to any *relative* gyro.

    This is what licenses `relative_gyro_jacobian` gathering only the
    filtered-joint columns: the free joint's rotational block is the same identity
    at both sites and cancels, its translational block is zero.  If it ever stops
    holding, the pair measurement acquires an unmodelled base-twist term and the
    G7 stacked oracle fails for a reason no single test would explain.
    """
    J = _evaluated(chain.name)["J_ang"]
    for parent, child in chain.pairs:
        np.testing.assert_allclose(J[:, child, :, :6], J[:, parent, :, :6], rtol=0.0, atol=1e-14)


def test_relative_gyro_jacobian_is_the_child_frame_difference(chain):
    """`J_rel = R_child^T (J_child - J_parent)` on the filtered columns, masked by `S_ab`.

    Restating the definition against the world-frame Jacobians.  The frame
    conversion is the single most error-prone line in the adapter -- a transposed
    `R_child`, or the parent's rotation instead of the child's, is silently
    plausible -- so it gets its own assertion rather than only being covered
    through `apply_consistent_motion`.
    """
    ev = _evaluated(chain.name)
    Jw, R, got = ev["J_ang"][:, :, :, chain.dof_joint], ev["site_rot"], ev["J_rel"]
    for e, (parent, child) in enumerate(chain.pairs):
        diff = Jw[:, child] - Jw[:, parent]
        want = np.einsum("kji,kjn->kin", R[:, child], diff)
        np.testing.assert_allclose(got[:, e], want * chain.model.pair_joint_mask[e],
                                   rtol=0.0, atol=1e-14)
        # The path mask is redundant on a serial chain: off-path joints move both
        # sites identically (common ancestors) or neither (descendants). Asserting
        # that makes S_ab a structural check rather than a silent correction.
        np.testing.assert_allclose(got[:, e], want, rtol=0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 3. Mass matrix vs kinetic energy
# ---------------------------------------------------------------------------

def _kinetic_energy(chain: ChainFixture, q_chain: np.ndarray, qvel: np.ndarray) -> float:
    """`T` from link twists and spatial inertias -- Newton-Euler, never CRB.

    Base twist is `qvel[0:3]` (world linear velocity of the base origin) and
    `qvel[3:6]` (world angular velocity), matching the convention pinned above.
    Twists propagate as `omega_i = omega_{i-1} + R_i a_i qd_i` and
    `v_i = v_{i-1} + omega_{i-1} x (p_i - p_{i-1})`; each body contributes
    `1/2 m |v_com|^2 + 1/2 omega^T (R I R^T) omega`, and armature contributes
    `1/2 armature_i qd_i^2` -- a rotor spinning in its own coordinate, invisible
    to the rigid-body recursion, which is exactly why MuJoCo can add it as a pure
    diagonal.
    """
    g = chain.geometry
    p_link, R_link = chain.forward(q_chain)

    v, w = np.asarray(qvel[0:3], float), np.asarray(qvel[3:6], float)
    p_par = np.array([0.0, 0.0, 1.0])                       # base body origin
    T = 0.5 * g.base_mass * v @ v + 0.5 * w @ (np.diag(g.base_inertia) @ w)

    qd_chain = np.asarray(qvel[6:], float)
    for i in range(chain.n_chain):
        v = v + np.cross(w, p_link[i] - p_par)
        w = w + R_link[i] @ g.axis[i] * qd_chain[i]
        p_par = p_link[i]

        r_com = R_link[i] @ g.com[i]
        v_com = v + np.cross(w, r_com)
        Rw = R_link[i] @ quat_to_mat(g.inertia_quat[i])
        T += (0.5 * g.mass[i] * v_com @ v_com
              + 0.5 * w @ (Rw @ np.diag(g.inertia_diag[i]) @ Rw.T @ w))

    return float(T + 0.5 * np.sum(np.asarray(g.armature) * qd_chain ** 2))


def test_mass_matrix_reproduces_kinetic_energy(chain):
    """`0.5 qd^T M qd == T`, at 10 random `(q, qd)` including base motion.

    A quadratic form is determined by its values on enough directions, so this
    constrains every entry of `M` -- including the `M_jb` coupling the Schur
    complement lives on, which a diagonal-only check would leave completely free.
    """
    ev = _evaluated(chain.name)
    rng = np.random.default_rng(SEED + 5)
    for k in range(10):
        qvel = rng.normal(size=chain.model.nv)
        got = 0.5 * qvel @ ev["M"][k] @ qvel
        want = _kinetic_energy(chain, _chain_q(chain, ev["q"][k]), qvel)
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-12)


def test_mass_matrix_is_symmetric_positive_definite(chain):
    """Symmetry to 1e-12 and `min eig > 0` at 20 random configurations.

    `M_bb` must be invertible for the Schur complement to exist at all; PD of the
    whole matrix is the strongest cheap statement of that.
    """
    M = _evaluated(chain.name)["M"]
    assert M.dtype == np.float64, "invariant I8: the model seam is float64"
    for k in range(N_Q):
        assert_symmetric(M[k], 1e-12, f"{chain.name} M(q) [{k}]")
        assert float(np.linalg.eigvalsh(0.5 * (M[k] + M[k].T)).min()) > 0.0


def test_mass_matrix_blocks_are_the_documented_gathers(chain):
    """`(jj, jb, bb, bj)` are exactly `M` indexed by the build-time DoF arrays."""
    q = _configs(chain.name)[0]
    M = _evaluated(chain.name)["M"][0]
    j, b = chain.dof_joint, chain.dof_nuisance
    blocks = chain.model.mass_matrix_blocks(q)
    np.testing.assert_allclose(np.asarray(blocks.jj), M[np.ix_(j, j)], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(blocks.jb), M[np.ix_(j, b)], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(blocks.bb), M[np.ix_(b, b)], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(blocks.bj), M[np.ix_(b, j)], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(blocks.bj), np.asarray(blocks.jb).T, atol=1e-14)


# ---------------------------------------------------------------------------
# 6. The consistent-motion oracle
# ---------------------------------------------------------------------------

def test_apply_consistent_motion_relative_gyro_equals_J_qdot(chain):
    """`gyro_child - R_cp gyro_parent == J_ang(q) qd` to ~1e-12, at 10 draws.

    THE oracle the whole kinematic half of the suite rests on.  It holds
    *exactly*, not to first order, because the fixture commands zero base twist:
    both gyros then carry the same (absent) base term, so the difference is purely
    the chain's relative motion.  Every downstream tracking tolerance that assumes
    encoder/gyro consistency is assuming this identity.
    """
    ev = _evaluated(chain.name)
    rng = np.random.default_rng(SEED + 10)
    for k in range(10):
        q, qd = ev["q"][k], rng.normal(size=chain.n)
        cm = chain.apply_consistent_motion(q, qd)
        for e, (parent, child) in enumerate(chain.pairs):
            R_cp = cm.site_rot[child].T @ cm.site_rot[parent]
            rel = cm.gyro[child] - R_cp @ cm.gyro[parent]
            np.testing.assert_allclose(rel, ev["J_rel"][k, e] @ qd, rtol=0.0, atol=1e-12)


def test_apply_consistent_motion_agrees_with_mjx_kinematics(chain):
    """The fixture's own recursion vs MJX, on site rotations and gyros.

    `apply_consistent_motion` is written without MJX on purpose (otherwise the
    check above would be circular); this test is what stops the two from drifting
    apart, and it also pins the `(qpos, qvel)` packing the fixture hands to
    anything that wants to drive MuJoCo directly.
    """
    rng = np.random.default_rng(SEED + 11)
    q, qd = _configs(chain.name)[0], rng.normal(size=chain.n)
    cm = chain.apply_consistent_motion(q, qd)
    assert np.all(cm.qvel[:6] == 0.0), "consistent motion must command zero base twist"

    ev = chain.model.evaluate(cm.qpos)
    R, J = np.asarray(ev.site_rot), np.asarray(ev.J_ang)
    np.testing.assert_allclose(R, cm.site_rot, rtol=0.0, atol=1e-12)
    omega_world = J @ cm.qvel
    for k in range(chain.m):
        np.testing.assert_allclose(cm.site_rot[k].T @ omega_world[k], cm.gyro[k],
                                   rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Adapter hygiene
# ---------------------------------------------------------------------------

def test_accessors_agree_with_the_single_pass(chain):
    """`evaluate(q)` and the individual accessors must be the same numbers.

    Every oracle above reads `evaluate`; the estimator may reach for
    `mass_matrix` / `relative_gyro_jacobian`.  Without this test a mutation to one
    path is invisible to the other -- which is not hypothetical: the Phase 0b
    mutation check found exactly that hole (a perturbed `mass_matrix` passed the
    whole file because `evaluate` recomputed `qM` independently).
    """
    q = _configs(chain.name)[0]
    ev = _evaluated(chain.name)
    np.testing.assert_array_equal(np.asarray(chain.model.mass_matrix(q)), ev["M"][0])
    np.testing.assert_array_equal(np.asarray(chain.model.relative_gyro_jacobian(q)), ev["J_rel"][0])
    np.testing.assert_array_equal(np.asarray(chain.model.site_angular_jacobians(q)), ev["J_ang"][0])
    pos, rot = chain.model.site_poses(q)
    np.testing.assert_array_equal(np.asarray(pos), ev["site_pos"][0])
    np.testing.assert_array_equal(np.asarray(rot), ev["site_rot"][0])
    np.testing.assert_array_equal(np.asarray(chain.model.site_positions(q)), ev["site_pos"][0])
    np.testing.assert_array_equal(np.asarray(chain.model.site_rotations(q)), ev["site_rot"][0])


def test_implements_the_robot_model_protocol():
    """`MjxModel` satisfies `robot.RobotModel` -- the seam the estimator depends on."""
    from invariant_estimation.robot import RobotModel
    assert isinstance(fixture("n3_m2").model, RobotModel)


def test_is_jittable_and_vmappable():
    """Invariant I7: `q` in, arrays out, with a graph that does not depend on `q`.

    `mass_matrix` and `relative_gyro_jacobian` are the two the jitted filter step
    calls per tick; both must trace without a Python branch on a traced value and
    must batch under `vmap` for the MJX rollouts of G10.
    """
    ch = fixture("n3_m2")
    qs = jnp.asarray(_configs("n3_m2")[:4])
    M = jax.jit(ch.model.mass_matrix)(qs[0])
    J = jax.jit(ch.model.relative_gyro_jacobian)(qs[0])
    assert M.dtype == jnp.float64 and J.dtype == jnp.float64
    assert J.shape == (ch.model.n_pairs, 3, ch.n)

    Ms = jax.jit(jax.vmap(ch.model.mass_matrix))(qs)
    assert Ms.shape == (4, ch.model.nv, ch.model.nv)
    np.testing.assert_allclose(np.asarray(Ms[0]), np.asarray(M), rtol=0.0, atol=0.0)


def test_qpos_accepts_joint_or_full_configuration():
    """`qpos(q)` widens `(n,)` and passes `(nq,)` through -- both jit-safe.

    The choice is made on a *static* shape, so it never becomes a traced branch
    (I7); gap joints and the base take `qpos0`.
    """
    ch = fixture("n8_m2")
    q = _configs("n8_m2")[0]
    full = np.asarray(ch.model.qpos(q))
    assert full.shape == (ch.model.nq,)
    np.testing.assert_allclose(full[ch.model.joint_qpos], q, atol=0.0)
    np.testing.assert_allclose(np.asarray(ch.model.qpos(full)), full, atol=0.0)
    with pytest.raises(ValueError, match="qpos"):
        ch.model.qpos(np.zeros(ch.n + 1))


def test_build_time_structural_rejections():
    """Self-pairs and same-body pairs are refused at build (`CLAUDE.md` §2).

    Both make the pair's measurement Jacobian identically zero, hence `S`
    singular; catching them at build is the only place the failure is legible.
    """
    from invariant_estimation.model.mjx_model import MjxModel

    ch = fixture("n8_m2")
    mj = ch.model.mj_model
    with pytest.raises(ValueError, match="self-pair"):
        MjxModel.from_mj_model(mj, site_names=ch.model.site_names, pairs=((0, 0),))

    same = [(a, b) for a in range(ch.model.n_sites) for b in range(ch.model.n_sites)
            if a != b and mj.site_bodyid[ch.model.site_ids[a]] == mj.site_bodyid[ch.model.site_ids[b]]]
    assert same, "n8_m2 puts the foot site on the same link as imu1 -- the same-body case"
    with pytest.raises(ValueError, match="one body"):
        MjxModel.from_mj_model(mj, site_names=ch.model.site_names, pairs=(same[0],))


def test_all_four_shapes_have_the_expected_dimensions():
    """`(n, m)` from `_oracles.SHAPES` must come out of the *model*, not the table.

    The link-index convention (`n = child_link - parent_link`) is the only thing
    tying the two together; if it is off by one, three of the four shapes still
    look plausible and only this assertion catches it.  Shape `n8_m3` is the
    shared-middle-IMU star that invariant I6 is about: its two pairs must overlap
    in IMU 1 and partition the eight joints four/four.
    """
    for shape, ch in zip(SHAPES, all_fixtures()):
        assert (ch.n, ch.m) == (shape["n"], shape["m"]) == (ch.model.n_joints, len(ch.imu_names))
        assert ch.model.n_pairs == len(shape["pairs"])
        assert all(np.array_equal(ch.geometry.axis[i], AXES[i % 3]) for i in range(ch.n_chain)), \
            "axes must cycle X/Y/Z by i % 3"

    star = fixture("n8_m3")
    assert set(star.pairs[0]) & set(star.pairs[1]) == {1}, "the two pairs share IMU 1"
    np.testing.assert_array_equal(star.model.pair_joint_mask.sum(axis=1), [4.0, 4.0])
    np.testing.assert_array_equal(star.model.pair_joint_mask.sum(axis=0), np.ones(star.n))
