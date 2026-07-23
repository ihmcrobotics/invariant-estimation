"""
jointKF/build.py
================
Graph resolution: turn a robot description plus an IMU-pair list into the static
`JointKFBuild` the jitted filter step closes over.

This module is where **every name becomes an index** (invariant I7).  Substring
tables, site names, joint names, sensor-map keys — all resolved here, once, in
plain Python.  Nothing downstream ever sees a string, and no shape downstream
ever depends on data.  The Java filter does the equivalent work in its
constructor; the difference is that we must also fix `K_max` (the anchor count)
for the filter's lifetime, because a `jnp` graph cannot grow rows when a foot
lands (invariant I2 / CLAUDE.md §4).

Structural rejection at build time
----------------------------------
Two IMU-pair configurations produce a singular measurement and must be rejected
loudly here rather than debugged later as an ill-conditioned `S`:

* **self-pair** (`parent is child`) — the relative gyro is identically zero, so
  the pair's three rows are all-zero: `S` loses rank.
* **same-link pair** — both IMUs rigidly attached to the same body.  No joint
  lies between them, the selection `S_ab` is empty, and the rows again carry no
  joint information while still claiming three measurement dimensions.

Acyclicity (paper §II-B3) is checked by union-find over the pair graph.  A cycle
means two IMU paths share joints in a way that makes the same `q_dot` observable
twice through different rotations; the stacked measurement is then rank-deficient
in a way `LSigmaL^T` cannot express, because the shared-bias bookkeeping assumes
a *tree* of relative measurements over the IMU set.  The star topology Alex
actually uses (every pair against the base IMU) is a tree, so this passes; it is
a guard against a mis-specified config, not a limitation.

Loud fallbacks
--------------
Java logs a boot warning for every joint that falls back to a default noise
value.  We do the same, via `logging`, because the fallback encoder variance
(5e-5) is two to four orders of magnitude *above* the hardware-measured
per-joint values: a joint silently on the fallback badly under-trusts its
encoder, and the only symptom is a slightly-too-smooth estimate.  The build log
is the one place that is cheap to notice.
"""
import logging
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from ..config import section
from .state import (
    JointKFBuild,
    alpha_for_name,
    encoder_var_for_name,
    rotor_inertia_for_name,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot description — the minimal tree the build needs
# ---------------------------------------------------------------------------

class KinematicTree(NamedTuple):
    """The subset of a robot description `build_joint_kf` actually needs.

    Deliberately *not* an MJX object: the build logic is pure graph work, and
    keeping it model-agnostic means it can be exercised against a hand-written
    tree in a unit test without standing up a physics engine.  The MJX adapter
    (`model/mjx_model.py`) produces one of these; so can a URDF reader.

    Attributes
    ----------
    joint_names : tuple of str
        All 1-DoF (hinge) joints, in model order.
    joint_body : ndarray, shape (n_all,)
        Body index each joint drives (its child body).
    body_parent : ndarray, shape (n_bodies,)
        Parent body of each body; the root's parent is itself or -1.
    joint_dof : ndarray, shape (n_all,)
        DoF index of each hinge joint in the full velocity vector.
    base_dofs : ndarray
        The floating base's DoF indices (6 for a free joint, empty if fixed).
    site_body : dict
        Site (IMU / sole) name -> body index it is attached to.
    tau_max : ndarray, shape (n_all,)
        Effort limit per joint; NaN where absent (then `sigma_tau` falls back).
    """

    joint_names: tuple[str, ...]
    joint_body: np.ndarray
    body_parent: np.ndarray
    joint_dof: np.ndarray
    base_dofs: np.ndarray
    site_body: dict[str, int]
    tau_max: np.ndarray


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def _ancestors(tree: KinematicTree, body: int) -> list[int]:
    """Bodies from `body` up to the root, inclusive."""
    chain, seen = [], set()
    while body >= 0 and body not in seen:
        seen.add(body)
        chain.append(body)
        parent = int(tree.body_parent[body])
        if parent == body:
            break
        body = parent
    return chain


def joints_between(tree: KinematicTree, body_a: int, body_b: int) -> list[int]:
    """Indices of the hinge joints strictly on the path between two bodies.

    Walks both bodies to the root, finds the lowest common ancestor, and takes
    the joints driving every body on either branch below it.  This is the Java
    "union of 1-DoF joints on the pair chain", and it is what makes `n` — and
    therefore the whole state dimension — a build-time constant.
    """
    up_a, up_b = _ancestors(tree, body_a), _ancestors(tree, body_b)
    set_b = set(up_b)
    lca = next((b for b in up_a if b in set_b), -1)

    branch = [b for b in up_a[:up_a.index(lca)] if lca >= 0] if lca >= 0 else list(up_a)
    branch += [b for b in up_b[:up_b.index(lca)]] if lca >= 0 else list(up_b)
    on_branch = set(branch)
    return [j for j in range(len(tree.joint_names)) if int(tree.joint_body[j]) in on_branch]


class _UnionFind:
    """Textbook union-find with path compression — used only for the cycle check."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        """Merge; return False if `a` and `b` were already connected (a cycle)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[ra] = rb
        return True


def check_pair_graph(pairs: Sequence[tuple[int, int]], n_imus: int, imu_body: Sequence[int]) -> None:
    """Reject self-pairs, same-link pairs, and cycles. Raises `ValueError`.

    Called before anything else in the build, because each of these produces a
    *singular* measurement rather than a merely inaccurate one — and a singular
    `S` surfaces as an inscrutable conditioning failure thousands of ticks later.
    """
    uf = _UnionFind(n_imus)
    for e, (parent, child) in enumerate(pairs):
        if parent == child:
            raise ValueError(
                f"IMU pair {e} is a self-pair (IMU {parent} against itself): its "
                f"relative gyro is identically zero, so its three rows are rank-0."
            )
        if imu_body[parent] == imu_body[child]:
            raise ValueError(
                f"IMU pair {e} has both IMUs on body {imu_body[parent]}: no joint "
                f"lies between them, so the pair claims three measurement "
                f"dimensions while carrying no joint information (singular S)."
            )
        if not uf.union(parent, child):
            raise ValueError(
                f"IMU pair {e} ({parent}, {child}) closes a cycle in the pair "
                f"graph. The stacked measurement assumes a TREE of relative "
                f"measurements over the IMU set (paper §II-B3); a cycle makes the "
                f"same q_dot observable twice through different rotations, which "
                f"the shared-bias L*Sigma*L^T bookkeeping cannot express."
            )


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def build_joint_kf(
    tree: KinematicTree,
    imu_sites: Sequence[str],
    pairs: Sequence[tuple[int, int]],
    foot_sites: Sequence[str] = (),
    *,
    base_imu: int = 0,
    use_mass_matrix: bool = True,
    use_armature_for_rotor: bool = True,
    cfg: dict[str, Any] | None = None,
    gyro_sigma: Callable[[str], np.ndarray] | None = None,
) -> JointKFBuild:
    """Resolve a robot + IMU-pair spec into the static `JointKFBuild`.

    Parameters
    ----------
    tree : KinematicTree
        The robot description (see that class).
    imu_sites : sequence of str
        Site names of the IMUs, in the order that fixes each IMU's *ordinal* —
        and therefore its bias columns `2n + 3k`. Stable ordering matters: the
        ordinal is baked into the state layout.
    pairs : sequence of (parent_ordinal, child_ordinal)
        The IMU graph. On Alex this is a star on the base IMU.
    foot_sites : sequence of str
        Sole sites that can host a stance anchor. `K_max = len(foot_sites)` and
        is FIXED for the filter's lifetime (invariant I2): a foot landing changes
        a *mask*, never a shape.
    base_imu : int
        Ordinal of the IMU whose bias the stance anchor pins. This is the gauge
        fixer — without an anchor the common-mode bias direction is unobservable.
    use_armature_for_rotor : bool
        If True (the production path) rotor inertia is expected to reach the
        filter through the MJCF `armature`, folded into `qM` pre-Schur, and the
        returned `rotor_inertia` array is INFORMATIONAL ONLY — adding it again
        post-Schur would double-count the drivetrain (CLAUDE.md §6).

    Returns
    -------
    JointKFBuild
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    imu_body = [tree.site_body[s] for s in imu_sites]
    check_pair_graph(pairs, len(imu_sites), imu_body)

    # -- filtered joint set: the union over all pair chains -----------------
    filtered: list[int] = []
    for parent, child in pairs:
        for j in joints_between(tree, imu_body[parent], imu_body[child]):
            if j not in filtered:
                filtered.append(j)
    filtered.sort()                       # state order == model order, stable
    if not filtered:
        raise ValueError("no joints lie on any IMU-pair chain; the filter would have no state")

    n, m = len(filtered), len(imu_sites)
    names = tuple(tree.joint_names[j] for j in filtered)
    index_of = {j: i for i, j in enumerate(filtered)}

    # -- per-pair velocity mask (fixed shape, never a gather) ---------------
    pair_velocity_mask = np.zeros((len(pairs), n))
    for e, (parent, child) in enumerate(pairs):
        for j in joints_between(tree, imu_body[parent], imu_body[child]):
            pair_velocity_mask[e, index_of[j]] = 1.0

    # -- stance anchors, F/U split ------------------------------------------
    # A base->foot chain generally contains joints that are NOT filter states
    # (on Alex: the ankles, because there are no foot IMUs). Their measured
    # velocity enters the anchor row as a known INPUT, so by the input-noise
    # congruence their covariance must be pushed into R_anchor. Splitting the
    # chain here is what lets measure/anchors build that congruence.
    # The chain starts at the BASE IMU's body, NOT the world root. The anchor
    # asserts that a stance foot's ABSOLUTE angular rate is ~zero, and that rate
    # is `omega_baseIMU + J(baseIMU->foot) q_dot`; the base IMU's own rate is
    # what the `+I3` bias column reads back. Rooting the chain at the world
    # instead would drag every joint between the world and the base IMU into the
    # unfiltered set, inflating `R_anchor` with velocities the anchor equation
    # never referenced. (Java `singlePairFootBeyondIMUs(10, 1, 5, 9)` pins this:
    # F = joints 2..5, U = joints 6..9 -- joints 0..1 appear in NEITHER.)
    anchor_root = imu_body[base_imu]
    anchor_filtered = np.zeros((len(foot_sites), n))
    unfiltered_cols: list[int] = []
    anchor_unfiltered_sets: list[list[int]] = []
    for k, site in enumerate(foot_sites):
        chain = joints_between(tree, anchor_root, tree.site_body[site])
        u_here = []
        for j in chain:
            if j in index_of:
                anchor_filtered[k, index_of[j]] = 1.0
            else:
                if j not in unfiltered_cols:
                    unfiltered_cols.append(j)
                u_here.append(j)
        anchor_unfiltered_sets.append(u_here)
    unfiltered_cols.sort()
    u_index = {j: i for i, j in enumerate(unfiltered_cols)}
    anchor_unfiltered = np.zeros((len(foot_sites), len(unfiltered_cols)))
    for k, u_here in enumerate(anchor_unfiltered_sets):
        for j in u_here:
            anchor_unfiltered[k, u_index[j]] = 1.0

    # -- per-joint parameter vectors ----------------------------------------
    alpha = np.array([alpha_for_name(nm, cfg) for nm in names])
    tau_max = np.array([tree.tau_max[j] for j in filtered], dtype=float)
    good_tau = np.isfinite(tau_max) & (tau_max > 0.0)
    sigma_tau = np.where(good_tau, alpha * np.nan_to_num(tau_max), cfg["sigma_tau"])
    rotor = np.array([rotor_inertia_for_name(nm, cfg) for nm in names])

    enc = [encoder_var_for_name(nm, cfg) for nm in names]
    encoder_var = np.array([v for v, _ in enc])
    encoder_wired = tuple(w for _, w in enc)

    # -- per-IMU gyro noise, floored at build (never per tick) --------------
    floor, floor_trace = cfg["sigma_gyro_floor"], cfg["sigma_gyro_floor_trace"]
    sigmas = []
    for name in imu_sites:
        S = np.eye(3) * floor if gyro_sigma is None else np.asarray(gyro_sigma(name), float)
        # A zero Sigma removes the innovation-covariance floor on the pure-bias
        # rows, collapsing lambda_min(S) and diverging P through the Joseph
        # K R K^T loop. Alex historically ran with unset SensorNoiseParameters,
        # i.e. all covariances exactly 0 -- hence this safety net.
        if not np.all(np.isfinite(S)) or np.trace(S) < floor_trace:
            log.warning("IMU %r gyro Sigma trace %.3e below floor %.3e; flooring to %.3e*I3",
                        name, np.trace(S) if np.all(np.isfinite(S)) else float("nan"), floor_trace, floor)
            S = np.eye(3) * floor
        sigmas.append(S)

    # -- loud build-time fallback reporting (Java parity) -------------------
    unwired = [nm for nm, w in zip(names, encoder_wired) if not w]
    if unwired:
        log.warning(
            "%d/%d filtered joints have no per-joint encoder variance and fall back to "
            "%.1e rad^2 (sigma ~ %.1e rad). Hardware-measured values run 2-4 ORDERS "
            "lower, so these joints badly UNDER-trust their encoders: %s",
            len(unwired), n, cfg["encoder_var"], np.sqrt(cfg["encoder_var"]), ", ".join(unwired),
        )
    default_alpha = [nm for nm in names if alpha_for_name(nm, cfg) == cfg["alpha_default"]]
    if default_alpha:
        log.warning(
            "%d filtered joints are on the alpha default (%.2f) rather than a calibrated "
            "value; watch the QA_MAX tripwire for them: %s",
            len(default_alpha), cfg["alpha_default"], ", ".join(default_alpha),
        )

    # -- mass-matrix nuisance set: base + GAP joints ------------------------
    # A gap joint lies on a root->filtered path without being a filter state
    # (Java `collectSpanningJoints` minus the filtered set). It must be
    # marginalised, because it genuinely accelerates between the base and a
    # filtered joint.
    #
    # This is NOT the same set as the anchor chain's unfiltered joints, and
    # conflating them was a real bug: on Alex the ankles are unfiltered members
    # of the base->foot anchor chain but are OFF the root->filtered paths, so
    # Java locks them into the composited inertia while the old code eliminated
    # them. Worth 1.7% on diag(Qa) against the hardware log. The two sets
    # coincide on a serial chain, which is why the unit fixtures never saw it.
    spanning: set[int] = set()
    for j in filtered:
        for body in _ancestors(tree, int(tree.joint_body[j])):
            spanning.update(k for k in range(len(tree.joint_names))
                            if int(tree.joint_body[k]) == body)
    gap_cols = sorted(spanning - set(filtered))
    nuisance = np.concatenate([
        np.asarray(tree.base_dofs, dtype=int),
        np.array([tree.joint_dof[j] for j in gap_cols], dtype=int),
    ]) if len(gap_cols) or len(tree.base_dofs) else np.zeros(0, dtype=int)
    anchor_unfiltered_dof = np.array(
        [tree.joint_dof[j] for j in unfiltered_cols], dtype=int
    )

    return JointKFBuild(
        n_joints=n,
        n_imus=m,
        n_pairs=len(pairs),
        n_anchors=len(foot_sites),
        joint_names=names,
        imu_names=tuple(imu_sites),
        pair_parent=np.array([p for p, _ in pairs], dtype=int),
        pair_child=np.array([c for _, c in pairs], dtype=int),
        pair_velocity_mask=pair_velocity_mask,
        base_imu=base_imu,
        anchor_filtered_mask=anchor_filtered,
        anchor_unfiltered_mask=anchor_unfiltered,
        anchor_imu=np.full(len(foot_sites), base_imu, dtype=int),
        alpha=alpha,
        tau_max=tau_max,
        sigma_tau=sigma_tau,
        rotor_inertia=rotor if not use_armature_for_rotor else rotor,
        encoder_var=encoder_var,
        encoder_wired=encoder_wired,
        gyro_sigma=np.stack(sigmas) if sigmas else np.zeros((0, 3, 3)),
        dof_joint=np.array([tree.joint_dof[j] for j in filtered], dtype=int),
        dof_nuisance=nuisance,
        use_mass_matrix=use_mass_matrix,
        dof_anchor_unfiltered=anchor_unfiltered_dof,
    )
