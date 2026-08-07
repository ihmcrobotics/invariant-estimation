"""
jointKF/state.py
================
The **frozen contract** for the joint-space KF (CLAUDE.md §1 deliverable 1, gates
G6-G8).  Every other module in this package -- and every ported test -- builds
against the layout, parameter names, and index helpers defined here.

State layout (locked by `JointLevelKFStateTest.testXOrdering`)
-------------------------------------------------------------
::

    x = [ q (n) ; q_dot (n) ; b_omega (3m) ]  in R^{2n + 3m}

    n = number of FILTERED joints (the union of 1-DoF joints on the IMU-pair
        chains -- fixed at build time, so `dim` is static)
    m = number of DISTINCT IMUs

**`m` is per-IMU, not per-pair.**  This is the breaking change from the
superseded Rev.1 design and it is not cosmetic: invariant I6 requires the exact
`L Sigma L^T` cross-covariance on the stacked gyro measurement over a
shared-base-IMU star, and the bias columns of `H_g` must *be* the mixing operator
`L` (`testBiasColumnsOfHgAreExactlyL` asserts this bit-identically).  With
per-pair bias, two pairs sharing an IMU carry two independent copies of one
physical bias, the shared-IMU cross terms vanish, and the G7 stacked oracle
cannot pass.

Bias lives here and **only** here -- invariant I1.  The InEKF state stays pure
SE_{N+2}(3) and consumes bias-corrected `omega_bar, a_bar`.

Design notes
------------
* `JointKFState` is a NamedTuple, hence a JAX pytree with no registration, so
  `jax.lax.scan` carries it directly.

* `n` and `m` are NOT stored in the state -- array shapes encode them.  Storing
  them would make the struct non-pytree-safe under jit unless marked static
  everywhere.

* `b_omega` is stored FLAT `(3m,)` so the stacked vector `x` is a plain
  concatenation and the `P` block layout is contiguous.  `b_omega_imus` gives the
  `(m, 3)` view.

* `JointKFBuild` holds everything resolved from **names** -- index arrays, masks,
  per-joint parameter vectors.  Name-table resolution happens once, in plain
  Python, at build time (invariant I7: no strings and no data-dependent shapes
  inside jit).  The jitted step closes over a `JointKFBuild`.

* Every "skip"/"gate"/"anchor active" decision is a fixed-shape float mask, never
  a Python branch or a reshape (CLAUDE.md §4).

Pure-function discipline (invariant I10)
----------------------------------------
Both filters are pure functions over an explicit `(x, P)` carry.  The Java
suite's `*ForTest` seams then cost nothing -- they are just these sub-functions
called directly.  `TEST_SUITE_MAP.md` §"Test seams" is the required public
surface; the mapping is recorded in `SEAM_MAP` below so a ported test can be read
against the Java one without guessing.
"""
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..config import section

# ---------------------------------------------------------------------------
# Seam map: Java test hook  ->  Python callable. Part of the public surface
# (invariant I10), kept here so a ported test reads 1:1 against the Java one.
# ---------------------------------------------------------------------------
SEAM_MAP: dict[str, str] = {
    "initialize":                        "jointKF.filter.initialize",
    "predict":                           "jointKF.predict.predict",
    "josephUpdate":                      "jointKF.update.joseph_update",
    "setStateForTest":                   "JointKFState(x=..., P=...)  (pure carry)",
    "getStateVector":                    "JointKFState.x",
    "getCovariance":                     "JointKFState.P",
    "getStateDimension":                 "JointKFBuild.dim",
    "getTransitionMatrix":               "jointKF.predict.build_transition",
    "getProcessNoise":                   "jointKF.process.build_process_noise",
    "getEncoderJacobian":                "jointKF.measure.encoder_jacobian",
    "getEncoderNoise":                   "jointKF.measure.encoder_noise",
    "buildStackedMeasurementForTest":    "jointKF.measure.build_stacked",
    "getStackedMeasurementJacobian":     "StackedMeasurement.H",
    "getStackedMeasurementResidual":     "StackedMeasurement.z",
    "getStackedMeasurementNoise":        "StackedMeasurement.R",
    "getStackedRowForPair":              "JointKFBuild.stacked_row_for_pair",
    "getMixingOperator":                 "StackedMeasurement.L",
    "getPairParentBiasColumn":           "JointKFBuild.pair_parent_bias_col",
    "getPairChildBiasColumn":            "JointKFBuild.pair_child_bias_col",
    "getPairVelocityColumns":            "JointKFBuild.pair_velocity_cols",
    "getBiasBlockColumn":                "JointKFBuild.bias_col",
    "getJointStateIndex":                "JointKFBuild.joint_index",
    "getNumberOfPairs":                  "JointKFBuild.n_pairs",
    "getNumberOfFilteredJoints":         "JointKFBuild.n_joints",
    "getNumberOfIMUs":                   "JointKFBuild.n_imus",
    "getActiveAnchorCountForTest":       "JointKFBuild-shaped mask sum in Diagnostics",
    "setTrustedFeetForTest":             "trusted_feet mask argument to build_stacked",
    "isUsingMassMatrixProcessNoise":     "JointKFBuild.use_mass_matrix",
    "updateProcessNoiseFromMassMatrixForTest": "jointKF.process.acceleration_covariance",
    "reflectedRotorInertiaForNameOrDefault":   "jointKF.state.rotor_inertia_for_name",
    "getAngularVelocityBiasInIMUFrame":  "JointKFState.b_omega_imus[i]",
    "describeSingularInnovation":        "Diagnostics.degenerate_row_attribution",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class JointKFState(NamedTuple):
    """Sufficient statistic for the bias-augmented joint KF: the `(x, P)` carry.

    Attributes
    ----------
    x : Array, shape (2n + 3m,)
        Stacked mean `[q ; q_dot ; b_omega]`.  Stored stacked rather than as
        three fields because every seam the Java suite exposes
        (`getStateVector`, `setStateForTest`, `josephUpdate`) operates on the
        stacked vector, and `P`'s blocks are indexed against it.
    P : Array, shape (2n + 3m, 2n + 3m)
        Full error covariance::

            P = [[ P_qq    P_q_qd   P_q_b  ],
                 [ P_qd_q  P_qdqd   P_qd_b ],
                 [ P_b_q   P_b_qd   P_bb   ]]

        `P_qq` is `Sigma_q`, the marginal the InEKF contact update pushes forward
        as `N = J_C Sigma_q J_C^T`.
    """

    x: Array
    P: Array

    # -- segment views ------------------------------------------------------
    def q(self, n: int) -> Array:
        """Joint positions `q`, shape (n,)."""
        return self.x[..., :n]

    def q_dot(self, n: int) -> Array:
        """Joint velocities `q_dot`, shape (n,)."""
        return self.x[..., n:2 * n]

    def b_omega(self, n: int) -> Array:
        """Per-IMU gyro bias, flat, shape (3m,)."""
        return self.x[..., 2 * n:]

    def b_omega_imus(self, n: int) -> Array:
        """Per-IMU view of the gyro bias, shape (m, 3).

        The Java seam `getAngularVelocityBiasInIMUFrame(imu)` is row `imu` of
        this: the bias is *stored* in each IMU's own measurement frame, so no
        rotation is applied on read.
        """
        return self.b_omega(n).reshape(-1, 3)

    # -- covariance marginals ----------------------------------------------
    def sigma_q(self, n: int) -> Array:
        r"""Marginal position covariance $\Sigma_q$, shape (n, n) -- InEKF FK noise."""
        return self.P[:n, :n]

    def sigma_q_dot(self, n: int) -> Array:
        """Marginal velocity covariance, shape (n, n).

        Note the explicit `2n` upper bound: `P` carries the bias block after the
        velocity block, so an open-ended slice would fold bias rows/cols in.
        """
        return self.P[n:2 * n, n:2 * n]

    def sigma_b(self, n: int) -> Array:
        """Marginal bias covariance, shape (3m, 3m)."""
        return self.P[2 * n:, 2 * n:]


def split_x(x: Array, n_joints: int) -> tuple[Array, Array, Array]:
    """Split `[q ; q_dot ; b_omega]` into its three segments.

    `b_omega` is the remainder, so `m` is not needed.
    """
    n = n_joints
    return x[..., :n], x[..., n:2 * n], x[..., 2 * n:]


# ---------------------------------------------------------------------------
# Parameters -- scalars, straight from config/filter_cfg.yaml
# ---------------------------------------------------------------------------

class JointKFParams(NamedTuple):
    """Scalar tunables, constant for a filter run.  See `config/filter_cfg.yaml`.

    Everything name-resolved (per-joint alpha, rotor inertia, encoder variance,
    per-IMU gyro Sigma) lives in `JointKFBuild`, not here -- those are arrays
    produced by build-time name matching, and keeping them out of this struct is
    what lets the jitted step treat `JointKFParams` as a plain pytree of scalars.
    """

    dt: float                       # [s]
    # measurement
    encoder_var: float              # [rad^2] fallback per-joint encoder variance
    sigma_gyro_floor: float         # [(rad/s)^2] per-axis floor
    sigma_gyro_floor_trace: float   # floor engages below this trace
    # process
    sigma_accel: float              # [rad/s^2] scalar-CWNA fallback
    sigma_tau: float                # [N.m] fallback torque STD
    target_qdd_std: float           # [rad/s^2] alpha-equalization target
    alpha_default: float
    qa_max: float                   # [(rad/s^2)^2] TRIPWIRE, never a scaler
    rotor_inertia_default: float
    imu_bias_process_var: float     # [(rad/s)^2/s]
    # conditioning
    cond_s_max: float
    # initial covariance
    init_pos_var: float
    init_vel_var: float
    init_bias_var: float
    # stance anchors
    anchor_var: float               # Sigma_eps -- ContactNet injection point
    sigma_qd_unfiltered: float      # [rad/s]
    anchor_rate_gain: float         # dimensionless; 0 == shipped constant Sigma_eps
    # direct velocity channel
    direct_velocity_enabled: bool
    lag_slew_smoothing_hz: float    # [Hz]
    # masking
    r_large: float                  # inactive-anchor R (never zero the rows)


def default_params(**overrides: Any) -> JointKFParams:
    """Build `JointKFParams` from the `joint_kf` config section.

    Any field may be overridden by keyword so a test or a sweep needn't touch the
    file::

        default_params(dt=2.0e-3, qa_max=1e9)
    """
    cfg = section("joint_kf")
    values = dict(
        dt=cfg["dt"],
        encoder_var=cfg["encoder_var"],
        sigma_gyro_floor=cfg["sigma_gyro_floor"],
        sigma_gyro_floor_trace=cfg["sigma_gyro_floor_trace"],
        sigma_accel=cfg["sigma_accel"],
        sigma_tau=cfg["sigma_tau"],
        target_qdd_std=cfg["target_qdd_std"],
        alpha_default=cfg["alpha_default"],
        qa_max=cfg["qa_max"],
        rotor_inertia_default=cfg["rotor_inertia_default"],
        imu_bias_process_var=cfg["imu_bias_process_var"],
        cond_s_max=cfg["cond_s_max"],
        init_pos_var=cfg["init"]["pos_var"],
        init_vel_var=cfg["init"]["vel_var"],
        init_bias_var=cfg["init"]["bias_var"],
        anchor_var=cfg["anchor_var"],
        sigma_qd_unfiltered=cfg["sigma_qd_unfiltered"],
        anchor_rate_gain=cfg["anchor_rate_gain"],
        direct_velocity_enabled=cfg["direct_velocity_enabled"],
        lag_slew_smoothing_hz=cfg["lag_slew_smoothing_hz"],
        r_large=cfg["r_large"],
    )
    unknown = set(overrides) - set(values)
    if unknown:
        raise TypeError(f"unknown JointKFParams field(s): {sorted(unknown)}")
    values.update(overrides)
    return JointKFParams(**values)


# ---------------------------------------------------------------------------
# Build-time name tables (plain Python -- invariant I7, no strings in jit)
# ---------------------------------------------------------------------------

def _ci_get(table: dict[str, float], name: str) -> float | None:
    """Case-insensitive **exact** lookup, mirroring Java's per-joint sensor tables.

    ``AlexSensorNoiseParameters`` keys its encoder-noise maps in lowercase and
    looks up ``jointName.toLowerCase()``; the MuJoCo joint names arrive uppercase
    (``LEFT_HIP_X``).  This folds both sides so the config author can write either
    case.  Exact, not substring: these are per-joint measured values, so
    ``LEFT_HIP_X`` must never inherit ``HIP_X``'s entry the way the rotor table
    (`_substring_lookup`) deliberately does.
    """
    if not table:
        return None
    lowered = {str(k).lower(): v for k, v in table.items()}
    return lowered.get(name.lower())


def _substring_lookup(name: str, table: dict[str, float], default: float) -> float:
    """Case-insensitive **substring** match, Java `reflectedRotorInertiaForNameOrDefault`.

    The Java table is keyed on fragments like ``"HIP_X"`` matched against a full
    joint name like ``"LEFT_HIP_X"``.  Longest key first, so ``ANKLE_Y`` is not
    shadowed by a hypothetical ``ANKLE``, and the match is order-independent
    rather than dict-insertion-order dependent.
    """
    upper = name.upper()
    for key in sorted(table, key=len, reverse=True):
        if key.upper() in upper:
            return table[key]
    return default


def rotor_inertia_for_name(name: str, cfg: dict[str, Any] | None = None) -> float:
    """Reflected rotor inertia `n^2 J_rotor` for a joint, by name substring.

    Locked by `JointLevelKFRotorAndGramTest.testRotorInertiaTableLookup`:
    ``LEFT_HIP_X -> 0.062``, ``RIGHT_HIP_Y -> 0.167``, ``left_knee_y -> 0.167``
    (case-insensitive), ``LEFT_ANKLE_Y -> 0.070``, ``LEFT_ANKLE_X -> 0.050``,
    ``SPINE_Z -> 0.062``, ``SOME_UNKNOWN_JOINT -> 0.005``.

    NOTE (CLAUDE.md §6, the armature double-add trap): production takes these
    values from the MJCF `armature`, which MuJoCo folds into `qM` *pre*-Schur.
    That is algebraically identical to adding them post-Schur, so doing BOTH
    counts the drivetrain twice.  Use this lookup to *populate* the MJCF, or to
    check the equivalence oracle -- never as a second additive term.
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    return _substring_lookup(name, cfg["rotor_inertia"], cfg["rotor_inertia_default"])


def alpha_for_name(name: str, cfg: dict[str, Any] | None = None) -> float:
    """Per-joint unmodeled-torque fraction `alpha_i`, by name substring.

    Falls back to `alpha_default` (0.15) for an unlisted joint -- deliberately,
    so an unlisted filtered joint surfaces via the `QA_MAX` tripwire rather than
    silently taking a calibrated neighbour's value.
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    return _substring_lookup(name, cfg["alpha_overrides"], cfg["alpha_default"])


def encoder_var_for_name(name: str, cfg: dict[str, Any] | None = None) -> tuple[float, bool]:
    """Per-joint encoder position VARIANCE, and whether the lookup was wired.

    Returns `(variance, wired)`.  `wired=False` means the joint fell back to
    `encoder_var` (5e-5), which is 2-4 orders ABOVE the hardware-measured values
    -- the joint will badly under-trust its encoder.  `build.py` logs every
    unwired joint loudly at build time (Java parity: "watch jointKF_encR_<joint>
    at boot").
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    std = _ci_get(cfg.get("encoder_pos_std", {}), name)
    if std is None or not np.isfinite(std) or std <= 0.0:
        return cfg["encoder_var"], False
    return float(std) ** 2, True


# ---------------------------------------------------------------------------
# Build -- everything resolved from names, once, before jit
# ---------------------------------------------------------------------------

class JointKFBuild(NamedTuple):
    """Static structure + per-joint/per-IMU parameter arrays.

    Produced by `jointKF.build.build_joint_kf(...)` and closed over by the jitted
    step.  Nothing here is traced and nothing here changes shape during a run
    (invariants I2, I7).

    Index conventions
    -----------------
    ``q``      of joint i  -> state index ``i``
    ``q_dot``  of joint i  -> state index ``n + i``
    ``b_omega`` of IMU k   -> state indices ``2n + 3k .. 2n + 3k + 3``

    Masks, not branches
    -------------------
    `pair_velocity_mask` and `anchor_*` are fixed-shape float masks.  An inactive
    anchor keeps its rows but gets `R_LARGE * I3`; its rows are never zeroed,
    which would make `S` singular (CLAUDE.md §6).
    """

    # -- static dimensions (plain ints; static under jit) -------------------
    n_joints: int
    n_imus: int
    n_pairs: int
    n_anchors: int                  # K_max, fixed for the filter lifetime

    # -- names, for diagnostics and build-time reporting only ---------------
    joint_names: tuple[str, ...]
    imu_names: tuple[str, ...]

    # -- IMU pair topology --------------------------------------------------
    pair_parent: Array              # (n_pairs,) int   IMU ordinal
    pair_child: Array               # (n_pairs,) int   IMU ordinal
    pair_velocity_mask: Array       # (n_pairs, n) float  1.0 on chain joints
    base_imu: int                   # ordinal of the base IMU (the star centre)

    # -- stance anchors -----------------------------------------------------
    anchor_filtered_mask: Array     # (n_anchors, n) float  F split: chain & state
    anchor_unfiltered_mask: Array   # (n_anchors, n_unfiltered) float  U split
    anchor_imu: Array               # (n_anchors,) int  IMU whose bias the anchor pins

    # -- per-joint parameter vectors (name-resolved at build) ---------------
    alpha: Array                    # (n,)
    tau_max: Array                  # (n,)  effort limits from the URDF
    sigma_tau: Array                # (n,)  alpha_i * tau_max_i, fallback SIGMA_TAU
    rotor_inertia: Array            # (n,)  informational; MJCF armature is the live path
    encoder_var: Array              # (n,)
    encoder_wired: tuple[bool, ...] # which joints got a real per-joint value

    # -- per-IMU noise ------------------------------------------------------
    gyro_sigma: Array               # (m, 3, 3)  floored at build

    # -- mass-matrix gather indices (I7: resolved here, not in jit) ---------
    dof_joint: Array                # (n,)  MJX DoF index of each filtered joint
    dof_nuisance: Array             # (n_nuisance,)  base 6 DoF + gap joints
    use_mass_matrix: bool           # False => scalar-CWNA fallback path

    # -- anchor-chain gather indices ----------------------------------------
    # DoFs of the unfiltered joints on the base->foot anchor chains, in
    # `anchor_unfiltered_mask` column order. A SEPARATE set from `dof_nuisance`:
    # Alex's ankles belong here but are off the root->filtered paths, so they are
    # locked (not marginalised) in `M`. Empty tuple => fall back to the trailing
    # slice of `dof_nuisance` (the pre-decoupling layout the fixtures build).
    dof_anchor_unfiltered: Array | tuple[int, ...] = ()

    # -- derived ------------------------------------------------------------
    @property
    def dim(self) -> int:
        """State dimension `2n + 3m`."""
        return 2 * self.n_joints + 3 * self.n_imus

    def joint_index(self, i: int) -> int:
        """State index of joint `i`'s position (Java `getJointStateIndex`)."""
        return i

    def velocity_index(self, i: int) -> int:
        """State index of joint `i`'s velocity."""
        return self.n_joints + i

    def bias_col(self, imu: int) -> int:
        """First state index of IMU `imu`'s bias (Java `getBiasBlockColumn`)."""
        return 2 * self.n_joints + 3 * imu

    def pair_parent_bias_col(self, pair: int) -> int:
        """Java `getPairParentBiasColumn`."""
        return self.bias_col(int(self.pair_parent[pair]))

    def pair_child_bias_col(self, pair: int) -> int:
        """Java `getPairChildBiasColumn`."""
        return self.bias_col(int(self.pair_child[pair]))

    def pair_velocity_cols(self, pair: int) -> tuple[int, ...]:
        """Velocity state columns pair `pair` observes (Java `getPairVelocityColumns`)."""
        on = np.nonzero(np.asarray(self.pair_velocity_mask[pair]) > 0.0)[0]
        return tuple(self.n_joints + int(i) for i in on)

    def stacked_row_for_pair(self, pair: int) -> int:
        """First stacked-measurement row of pair `pair` (Java `getStackedRowForPair`)."""
        return 3 * pair

    @property
    def anchor_row0(self) -> int:
        """First stacked row of the anchor block: anchors follow all pair rows."""
        return 3 * self.n_pairs

    @property
    def n_stacked_rows(self) -> int:
        """Total stacked-measurement rows: `3*(n_pairs + n_anchors)`, always."""
        return 3 * (self.n_pairs + self.n_anchors)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_state(build: JointKFBuild, params: JointKFParams, q0: Array | None = None) -> JointKFState:
    """Seed `(x, P)` -- Java `initialize()`.

    Locked by `JointLevelKFStateTest`:

    * `q` is seeded from the encoders (`testQ0Seed`), `q_dot` and `b_omega` from
      zero (`testXOrdering`, exact 0.0).
    * `P` is diagonal with `pos_var=1e-6`, `vel_var=1.0`, `bias_var=2.5e-3`
      (`testMarginalBlocksMatchP`, tol 1e-12).
    * The prior-confidence ordering `pos < bias < vel` (`testPriorConfidenceOrdering`)
      is intent, not coincidence: encoders are trusted at init, velocity is
      genuinely unknown, bias sits between.
    """
    n, m = build.n_joints, build.n_imus
    q = jnp.zeros(n, dtype=jnp.float64) if q0 is None else jnp.asarray(q0, dtype=jnp.float64)
    if q.shape != (n,):
        raise ValueError(f"q0 must have shape ({n},), got {q.shape}")

    x = jnp.concatenate([q, jnp.zeros(n, dtype=jnp.float64), jnp.zeros(3 * m, dtype=jnp.float64)])
    P = jnp.diag(jnp.concatenate([
        jnp.full(n, params.init_pos_var, dtype=jnp.float64),
        jnp.full(n, params.init_vel_var, dtype=jnp.float64),
        jnp.full(3 * m, params.init_bias_var, dtype=jnp.float64),
    ]))
    return JointKFState(x=x, P=P)
