r"""Accelerometer **gravity-leveling** (roll/pitch) — the attitude observation the
contact update lacks.  Behaviour is locked by the ported `GravityLevelingUpdaterTest`
(14 deterministic tests, no RNG).

The contact FK update observes ``p`` and ``d_i`` but leaves the *absolute* tilt of the
base weakly constrained: gravity is the only sensor that sees it.  This module turns
the accelerometer into a roll/pitch measurement while leaving yaw untouched — ``H``
is rank 2 with its null direction along ``e_z``, so a gravity update can never invent
heading.

Residual in the **body** frame, against the complementary gravity reference::

    r = ĝ_ref − R̂ᵀ e_z

Perturbing the right-invariant error, ``R̂ = Γ_0(φ) R`` ⟹
``R̂ᵀ e_z ≈ Rᵀ e_z + Rᵀ (e_z)_× φ``, so::

    H = [ −R̂ᵀ (e_z)_×   0₃  0₃  (0₃ per contact) ]        (3, 3N+9)

whose yaw column is ``−R̂ᵀ (e_z)_× e_z = 0`` **exactly**.  Diagnostics read straight
off the residual: a pitch error tips the measured gravity along body ``x``, a roll
error along body ``y``::

    tilt_pitch = r[0]      tilt_roll = r[1]
    tilt_angle = acos(clamp(ĝ_ref · R̂ᵀ e_z))

**Complementary gravity reference (τ = 5 s).**  The reference is the body-frame
gravity **direction**, propagated by the raw gyro and corrected toward the
accelerometer with a first-order lag, ``ġ_ref = −ω × g_ref + (ĝ_meas − g_ref) / τ``.
Three properties the tests pin, which a plain low-pass cannot have together:

* **Lateral-accel artifacts are rejected.**  With ``ω = 0`` this is a pure
  first-order low-pass, so a sway artifact at ``ω_b`` is attenuated by
  ``1/√(1 + (ω_b τ)²)`` ≈ 0.065 at the 0.49 Hz hardware balance mode (~15×).
* **True tilt passes at unity gain.**  Real tilt comes with real gyro rate, and the
  ``−ω × g_ref`` term tracks it with no lag at all.
* **DC authority is undiminished** — a static tilt drives the reference all the way
  there, so the residual is the full ``sin(tilt)``.

**Quasi-static gate.**  Leveling is only trustworthy when the specific force *is*
gravity: ``‖f‖`` within ``norm_tol·g``, ``‖ω‖ ≤ rot_tol``, and horizontal ``f ≤
horiz_tol``.  Regression F.3: the horizontal component is resolved against the
sensor-driven ``ĝ_ref``, **never** against the estimate's own attitude ``R̂ᵀ e_z``.
The old gate did the latter, which reads ``g·sinθ`` of horizontal force at estimator
tilt θ and so locked leveling out above 2.92° — exactly when it was needed most.

Everything here is jit-able and branch-free (I7): the lazy seeding of the reference,
the pitch-observability gate and the conditioning gate are all `jnp.where` on float
masks.
"""
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp

from ..config import section
from .correct import UpdateDiagnostics, linear_update
from .group import skew
from .state import InEKFState

UP = jnp.array([0.0, 0.0, 1.0])


class GravityParams(NamedTuple):
    """Run-fixed gravity-leveling configuration (all values test-locked).

    Attributes
    ----------
    gravity : float
        Gravity magnitude ``g`` [m/s²], positive.
    roll_var, pitch_var : float
        Anisotropic measurement variances.  Pitch is trusted far less than roll
        (2.5e-3 ≈ (2.9°)² vs 1.9e-1 ≈ (25°)²): fore-aft specific force is
        contaminated by walking, lateral much less so.
    pitch_disabled_var : float
        Variance substituted for `pitch_var` when pitch is not observable — large
        enough to freeze pitch without making ``S`` singular.
    tau : float
        Complementary gravity-reference time constant [s].
    norm_tol, rot_tol, horiz_tol : float
        Quasi-static gate thresholds (fraction of g, rad/s, m/s²).
    cond_max : float
        Conditioning gate on the innovation covariance (§4 masked-K rule).
    """
    gravity: float
    roll_var: float
    pitch_var: float
    pitch_disabled_var: float
    tau: float
    norm_tol: float
    rot_tol: float
    horiz_tol: float
    cond_max: float


def default_gravity_params(
    gravity: float | None = None,
    roll_var: float | None = None,
    pitch_var: float | None = None,
    pitch_disabled_var: float | None = None,
    tau: float | None = None,
    norm_tol: float | None = None,
    rot_tol: float | None = None,
    horiz_tol: float | None = None,
    cond_max: float | None = None,
) -> GravityParams:
    """`GravityParams`, defaulting to ``gravity_leveling`` in filter_cfg.yaml.

    Every argument is an override: pass ``None`` to take the configured value.
    """
    cfg = section("gravity_leveling")
    gates = cfg["gates"]
    return GravityParams(
        gravity=cfg["gravity"] if gravity is None else gravity,
        roll_var=cfg["roll_var"] if roll_var is None else roll_var,
        pitch_var=cfg["pitch_var"] if pitch_var is None else pitch_var,
        pitch_disabled_var=(cfg["pitch_disabled_var"] if pitch_disabled_var is None
                            else pitch_disabled_var),
        tau=cfg["reference_tau"] if tau is None else tau,
        norm_tol=gates["norm_tol"] if norm_tol is None else norm_tol,
        rot_tol=gates["rot_tol"] if rot_tol is None else rot_tol,
        horiz_tol=gates["horiz_tol"] if horiz_tol is None else horiz_tol,
        cond_max=section("inekf")["cond_max"] if cond_max is None else cond_max,
    )


def isotropic_gravity_params(variance: float, gravity: float | None = None, **kw) -> GravityParams:
    """Java `GravityLevelingUpdater(tangentSize, isotropicVar, G)` — roll and pitch share one variance."""
    return default_gravity_params(
        gravity=gravity, roll_var=variance, pitch_var=variance, **kw
    )


class GravityRef(NamedTuple):
    """Complementary gravity reference carried in the scan state.

    ``direction`` ``(3,)`` is the unit body-frame gravity direction.
    ``initialized`` is 0.0 until the first measurement seeds it, 1.0 after — a float,
    not a bool branch, so seeding is a `jnp.where` and the graph stays constant (I7).
    """
    direction: Array
    initialized: Array


def init_gravity_ref() -> GravityRef:
    """Unseeded reference — the first measurement supplies the direction."""
    return GravityRef(direction=UP, initialized=jnp.array(0.0))


class GravityMeasurement(NamedTuple):
    """Assembled gravity-leveling measurement plus its diagnostics."""
    residual: Array          # (3,)
    H: Array                 # (3, 3N+9)
    R: Array                 # (3, 3)
    tilt_angle: Array        # scalar
    tilt_pitch: Array        # scalar
    tilt_roll: Array         # scalar
    ref: GravityRef          # possibly-seeded reference


def _normalize(v: Array) -> Array:
    """Unit vector, safe at zero (returns the input direction unchanged there)."""
    n = jnp.linalg.norm(v)
    return jnp.where(n > 1e-12, v / jnp.where(n > 1e-12, n, 1.0), v)


def seed_reference(ref: GravityRef, specific_force: Array) -> GravityRef:
    """Seed an unseeded reference from the first measurement, branch-free.

    Java seeds lazily on first use; reproduced here with a float mask so the
    behaviour is identical without a data-dependent Python branch.
    """
    measured = _normalize(specific_force)
    direction = jnp.where(ref.initialized > 0.5, ref.direction, measured)
    return GravityRef(direction=direction, initialized=jnp.array(1.0))


def update_gravity_reference(
    ref: GravityRef,
    specific_force: Array,
    omega: Array,
    dt: float,
    params: GravityParams,
) -> GravityRef:
    r"""Advance the complementary reference one tick: ``ġ_ref = −ω × g_ref + (ĝ_meas − g_ref)/τ``, renormalised.

    The gyro term is the high-pass path (true tilt, unity gain, no lag); the
    accelerometer term is the low-pass path (``τ = 5 s``, which is what rejects sway
    artifacts).  ``specific_force`` ``(3,)`` is the body-frame IMU specific force;
    ``omega`` ``(3,)`` is the **raw** body-frame gyro, not bias-corrected — the gate
    and the reference both want the raw signal, cf.
    `testRotationGateUsesRawGyroNotBiasCorrupted`.
    """
    ref = seed_reference(ref, specific_force)
    measured = _normalize(specific_force)
    rate = -jnp.cross(omega, ref.direction) + (measured - ref.direction) / params.tau
    return GravityRef(
        direction=_normalize(ref.direction + rate * dt),
        initialized=jnp.array(1.0),
    )


def is_quasi_static(
    ref: GravityRef,
    specific_force: Array,
    omega: Array,
    params: GravityParams,
) -> Array:
    r"""Quasi-static gate — is this specific force actually just gravity?

    Three conditions, all of which must hold:

    1. **norm** — ``|‖f‖ − g| ≤ norm_tol·g``
    2. **rotation** — ``‖ω‖ ≤ rot_tol`` (raw gyro)
    3. **horizontal** — ``‖f − (f·ĝ_ref) ĝ_ref‖ ≤ horiz_tol``

    (3) is resolved against the **sensor-driven reference**, never against ``R̂ᵀ e_z``
    (regression F.3).  Returns a float-castable boolean array so it composes as a
    mask.
    """
    ref = seed_reference(ref, specific_force)

    norm = jnp.linalg.norm(specific_force)
    norm_ok = jnp.abs(norm - params.gravity) <= params.norm_tol * params.gravity
    rotation_ok = jnp.linalg.norm(omega) <= params.rot_tol

    along = jnp.dot(specific_force, ref.direction) * ref.direction
    horizontal_ok = jnp.linalg.norm(specific_force - along) <= params.horiz_tol

    return norm_ok & rotation_ok & horizontal_ok


def gravity_jacobian(state: InEKFState) -> Array:
    r"""``H = [−R̂ᵀ(e_z)_×  |  0 …]``, shape ``(3, 3N+9)``.

    Rank 2 with its null direction along ``e_z``: ``−R̂ᵀ(e_z)_× e_z = 0``, so the yaw
    column is exactly zero and a gravity update cannot create heading.
    """
    H = jnp.zeros((3, state.dim))
    return H.at[:, 0:3].set(-state.R.T @ skew(UP))


def gravity_measurement_covariance(
    g_hat: Array,
    params: GravityParams,
    pitch_observable: bool | Array = True,
) -> Array:
    r"""Anisotropic ``(3,3)`` ``R``, built on an orthonormal triad about ``g_hat`` in the **body** frame.

    ``g_hat`` must be the **predicted** direction ``R̂ᵀ e_z``, not the
    measured/reference one.  That choice is load-bearing, not cosmetic:
    ``H = −R̂ᵀ(e_z)_×`` satisfies ``(R̂ᵀe_z)ᵀ H = −e_zᵀ(e_z)_× = 0`` exactly, so the
    residual component along ``R̂ᵀe_z`` is *structurally unobservable* — the
    measurement is a unit vector, and its variation is always orthogonal to itself.
    Building ``R``'s triad about the same direction makes ``S`` block diagonal, so that
    unobservable channel decouples and its variance never perturbs the roll/pitch
    correction.  Building it about the measured direction instead leaves the two null
    directions misaligned by exactly the tilt being corrected, and the update then
    fights itself — measured at ~10x slower pitch convergence, enough to fail
    `testGravityUpdateLevelsTiltAndPreservesYaw`.

    The triad::

        p̂ = normalize(e_y × ĝ)     pitch-informing residual direction
        r̂ = normalize(ĝ × p̂)       roll-informing direction
        ĝ                          the gravity-null direction

        R = pitch_var·p̂p̂ᵀ + roll_var·r̂r̂ᵀ + roll_var·ĝĝᵀ

    At ``ĝ = e_z`` this is ``diag(pitch_var, roll_var, roll_var)``.  Because the triad
    is built from the **body** axes, the pitch-distrust axis follows body ``y`` even at
    non-zero yaw (`testPitchDistrustAxisIsBodyYAtNonZeroYaw`).

    ``pitch_observable=False`` swaps in `pitch_disabled_var`, freezing pitch while
    leaving roll untouched.
    """
    g_hat = _normalize(g_hat)

    pitch_dir = jnp.cross(jnp.array([0.0, 1.0, 0.0]), g_hat)
    # Degenerate only if ĝ ∥ e_y; fall back to the x-axis construction.
    fallback = jnp.cross(jnp.array([1.0, 0.0, 0.0]), g_hat)
    pitch_dir = jnp.where(jnp.linalg.norm(pitch_dir) > 1e-8, pitch_dir, fallback)
    pitch_dir = _normalize(pitch_dir)
    roll_dir = _normalize(jnp.cross(g_hat, pitch_dir))

    pitch_var = jnp.where(
        jnp.asarray(pitch_observable, dtype=bool),
        params.pitch_var,
        params.pitch_disabled_var,
    )
    return (pitch_var * jnp.outer(pitch_dir, pitch_dir)
            + params.roll_var * jnp.outer(roll_dir, roll_dir)
            + params.roll_var * jnp.outer(g_hat, g_hat))


def assemble_gravity_leveling(
    ref: GravityRef,
    state: InEKFState,
    specific_force: Array,
    params: GravityParams,
    pitch_observable: bool | Array = True,
) -> GravityMeasurement:
    r"""Assemble ``(residual, H, R)`` plus tilt diagnostics — Java ``assemble``.

    Uses the complementary reference (seeding it from ``specific_force`` on first
    use), **not** the raw specific force: that indirection is what buys the artifact
    rejection of the roll-sway tests while leaving DC tilt authority at full strength.
    """
    ref = seed_reference(ref, specific_force)

    predicted = state.R.T @ UP                     # R̂ᵀ e_z
    residual = ref.direction - predicted

    cos_tilt = jnp.clip(jnp.dot(ref.direction, predicted), -1.0, 1.0)
    return GravityMeasurement(
        residual=residual,
        H=gravity_jacobian(state),
        R=gravity_measurement_covariance(predicted, params, pitch_observable),
        tilt_angle=jnp.arccos(cos_tilt),
        tilt_pitch=residual[0],
        tilt_roll=residual[1],
        ref=ref,
    )


def apply_gravity_leveling(
    state: InEKFState,
    meas: GravityMeasurement,
    params: GravityParams,
    gate: Array | float = 1.0,
) -> tuple[InEKFState, UpdateDiagnostics]:
    r"""Joseph-form gravity update ``X̂⁺ = exp(−(Kν)^∧) X̂`` (I5).

    A thin wrapper over the shared `correct.linear_update`: the gravity update is an
    ordinary linear update with an unusual ``H`` and ``R``, so it must not have its own
    gain/Joseph/gate implementation to drift out of sync.  ``gate`` is where an
    external mask (typically `is_quasi_static`) enters; a gated update leaves
    ``(X̂, P)`` bit-for-bit unchanged.
    """
    return linear_update(
        state, meas.H, meas.residual, meas.R,
        cond_max=params.cond_max, gate=gate,
    )


def tilt_angle(state: InEKFState) -> Array:
    """``acos(clamp((R̂ᵀe_z)·e_z))`` — the test helper's tilt metric."""
    return jnp.arccos(jnp.clip(jnp.dot(state.R.T @ UP, UP), -1.0, 1.0))
