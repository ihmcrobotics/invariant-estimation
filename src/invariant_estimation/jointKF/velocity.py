r"""
jointKF/velocity.py
===================
The **optional** direct joint-velocity measurement channel (Java
`JointLevelKFPreFilter` `getVelocityMeasurementJacobian` /
`getVelocityMeasurementNoise` / `refreshDirectVelocityNoise`, ported test class
`JointLevelKFDirectVelocityMeasurementTest`).

Default **off** (`params.direct_velocity_enabled`, `false` in
`config/filter_cfg.yaml`).  It exists because some drives publish a firmware
velocity estimate that is genuinely better than anything the filter can infer
from the encoder history alone; it is off for sim v1 because that firmware signal
does not exist in MJX, and a channel whose noise model is calibrated against
hardware firmware is worse than no channel at all when fed a perfect derivative.

What is measured is NOT `q_dot`
-------------------------------
The load-bearing subtlety, and the reason this module is more than
``H = [0 | I | 0]``: the drive does not publish `q_dot(t)`.  It publishes the
output of a first-order low-pass with corner `omega_eff`::

    y_dot = omega_eff * (u - y),      u = true q_dot,   y = published value

so the measurement is *lagged*, and the lag error is not noise-like — it is a
deterministic function of how fast the joint is currently accelerating.
Rearranging that one line gives the identity this whole module rests on::

    u - y = y_dot / omega_eff                      (EXACT, not a small-angle
                                                    approximation)

The measurement error is therefore the *published signal's own slope* divided by
the corner frequency.  We do not know `y_dot` exactly, but we can estimate it
from the measurement itself, and then declare an honest variance::

    R_ii(t) = sigma_i^2  +  ( dhat_i / omega_eff,i )^2

The first term is the sensor noise floor; the second is the lag, which is zero at
constant velocity and large during a swing-leg transient.  A *static* `R` has to
be sized for the worst case, which means the channel is uselessly loose exactly
when the robot is standing still — the regime where a velocity measurement is
worth the most.

Why `dhat` is smoothed, and why that is not a free parameter
------------------------------------------------------------
`dhat` is a finite difference of the **noisy** measurement.  Raw, its variance is
``2 sigma^2 / dt^2``: at ``sigma = 1e-2 rad/s`` and ``dt = 1e-3 s`` that is
``2e2 (rad/s^2)^2``, which through ``(dhat/omega_eff)^2`` inflates `R` by roughly
two orders of magnitude *at quiet standing* — the exact opposite of the point.
So the finite difference is low-passed at `params.lag_slew_smoothing_hz` (5 Hz:
above the gait band, so real slew passes; far below the 500 Hz Nyquist, so
differentiation noise is cut by ~(5/500) in amplitude).

Note that the deterministic ramp test (`lagInflationTracksMeasuredSlewExactly`)
**cannot see this**: on a noiseless ramp the raw and smoothed finite differences
both converge to `slope`, and on a constant signal both are exactly zero.  The
smoothing is constrained only by a noisy-input test, which is why the ported
suite here carries one in addition to the Java scenario.

Constant graph (I7)
-------------------
The smoothed slew is a **carry**, not an attribute: `VelocityCarry` is a pytree of
float arrays advanced with `jnp.where`, so it rides in the `lax.scan` state
exactly like the trusted-feet mask.  The "have we seen a sample yet" flag is a
float in the same carry rather than a Python `bool` — the first tick's finite
difference against an unset `z_prev` would otherwise be `z/dt`, a spike of
hundreds of rad/s^2 that inflates `R` for the ~30 ms the smoother needs to
forget it.

Cross-talk (the ported observable of Java's label dispatch)
-----------------------------------------------------------
Java routes diagnostics by an exact-match measurement label — `josephUpdate(H, z,
R, "encoder")` publishes into `jointKF_encNIS_*`, `"encoderVelocity"` into
`jointKF_qdNIS_*` — and the ported test asserts that running the velocity channel
leaves the encoder NIS at `NaN`.  Strings cannot cross a jit boundary (I7), so
the port replaces string dispatch with **separate fields**: `ChannelDiagnostics`
has an encoder half and a velocity half, and `velocity_update` structurally
cannot write the encoder half.  The observable — "the position channel's
consistency statistic never reports a number it did not compute" — is preserved;
the mechanism is a struct field instead of a string compare.
"""
from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..config import section
from .state import JointKFBuild, JointKFParams, JointKFState
from .update import UpdateInfo, joseph_update

__all__ = [
    "VelocityChannel",
    "VelocityCarry",
    "ChannelDiagnostics",
    "velocity_var_for_name",
    "build_velocity_channel",
    "velocity_jacobian",
    "velocity_noise",
    "init_velocity_carry",
    "advance_slew",
    "velocity_update",
    "no_diagnostics",
]


# ---------------------------------------------------------------------------
# Build-time: per-joint noise floor and corner frequency (plain Python, I7)
# ---------------------------------------------------------------------------

def velocity_var_for_name(
    name: str,
    cfg: Mapping | None = None,
    lookup: Mapping[str, float] | None = None,
) -> tuple[float, bool]:
    """Per-joint velocity-measurement VARIANCE, and whether the lookup was wired.

    Mirrors `state.encoder_var_for_name` exactly — including the "return the
    fallback *and say so*" contract, because a joint silently on the fallback is
    the failure this whole per-joint machinery exists to prevent (invariant I9).

    The fallback is `sigma_qd_unfiltered` (0.1 rad/s => 0.01 rad^2/s^2), which is
    Java's `SIGMA_QD_FALLBACK`.  The two are the same constant in Java and are
    kept as one here: both answer "what does an *unmodelled* joint velocity cost
    us", one for a chain joint that is not a filter state and one for a joint
    whose drive noise was never characterised.

    Parameters
    ----------
    name : str
    cfg : mapping, optional
        The `joint_kf` config section; read once by `build_velocity_channel`.
    lookup : mapping, optional
        Overrides `cfg["encoder_vel_std"]` — the test seam standing in for Java's
        `velSigmaFor` function argument.  A non-finite or non-positive entry
        counts as *absent*, which is how the Java test injects an "unmatched"
        joint (it maps that one name to `NaN`).

    Returns
    -------
    (variance, wired) : (float, bool)
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    table = lookup if lookup is not None else cfg.get("encoder_vel_std", {})
    std = table.get(name)
    if std is None or not np.isfinite(std) or std <= 0.0:
        return float(cfg["sigma_qd_unfiltered"]) ** 2, False
    return float(std) ** 2, True


class VelocityChannel(NamedTuple):
    """Static, name-resolved structure of the direct-velocity channel.

    Built once in plain Python (I7: no strings, no lookups inside jit) and closed
    over by the jitted step, exactly like `JointKFBuild`.  It is a separate struct
    rather than extra `JointKFBuild` fields because `JointKFBuild` is frozen and
    because the channel is optional: a build with the channel disabled should not
    carry its arrays at all.

    Attributes
    ----------
    var : (n,) float
        Per-joint sensor-noise variance `sigma_i^2` — the floor `R` decays back
        to when the joint is at constant velocity.
    wired : tuple[bool, ...]
        Per joint: did the name lookup hit?  Reported at build, never read in jit.
    inv_omega : (n,) float
        `1 / (2*pi*f_corner,i)`, the seconds of lag the drive's low-pass adds.
        **Zero** for a joint with no declared corner frequency, which switches the
        lag inflation off for that joint and leaves a static `R` — the port of
        Java's nullable `cornerFn`.  Zero is the right disabled value: an unknown
        corner is not "infinitely laggy", it is "we are not modelling this".
    smoother_alpha : float
        `exp(-2*pi*f_smooth*dt)`, the one-pole coefficient of the slew smoother.
    dt : float
        Kept here so `advance_slew` needs only the channel and the carry.
    """

    var: Array
    wired: tuple[bool, ...]
    inv_omega: Array
    smoother_alpha: float
    dt: float


def build_velocity_channel(
    build: JointKFBuild,
    params: JointKFParams,
    *,
    vel_std: Mapping[str, float] | None = None,
    corner_hz: Mapping[str, float] | float | None = None,
    cfg: Mapping | None = None,
) -> VelocityChannel:
    """Resolve the per-joint velocity noise floor and corner frequency, once.

    Parameters
    ----------
    build, params
        Static structure and scalars.  `params.lag_slew_smoothing_hz` (5 Hz) and
        `params.dt` set the smoother; `params.sigma_qd_unfiltered` is the noise
        fallback.
    vel_std : mapping, optional
        Joint name -> velocity-noise STD [rad/s].  Defaults to the config's
        `encoder_vel_std` sidecar.  Java's `velSigmaFor`.
    corner_hz : mapping or float, optional
        Joint name -> drive low-pass corner [Hz], or one scalar for all joints.
        `None` (the default) disables lag inflation entirely — Java's
        `cornerFn = null`, the configuration every wiring/NIS test uses.
    cfg : mapping, optional
        The `joint_kf` config section; read here so the caller can inject one.

    Returns
    -------
    VelocityChannel
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    names = build.joint_names

    resolved = [velocity_var_for_name(nm, cfg, vel_std) for nm in names]
    var = np.array([v for v, _ in resolved], dtype=float)
    wired = tuple(w for _, w in resolved)

    if corner_hz is None:
        inv_omega = np.zeros(len(names), dtype=float)
    elif isinstance(corner_hz, (int, float)):
        inv_omega = np.full(len(names), 1.0 / (2.0 * np.pi * float(corner_hz)))
    else:
        # A joint absent from the table gets 0.0 => no inflation, same as the
        # global `None` case.  Never a default corner: guessing a corner
        # frequency invents a lag term out of nothing.
        f = np.array([float(corner_hz.get(nm, 0.0)) for nm in names], dtype=float)
        inv_omega = np.where(f > 0.0, 1.0 / (2.0 * np.pi * np.where(f > 0.0, f, 1.0)), 0.0)

    alpha = float(np.exp(-2.0 * np.pi * params.lag_slew_smoothing_hz * params.dt))
    return VelocityChannel(
        var=jnp.asarray(var, dtype=jnp.float64),
        wired=wired,
        inv_omega=jnp.asarray(inv_omega, dtype=jnp.float64),
        smoother_alpha=alpha,
        dt=float(params.dt),
    )


# ---------------------------------------------------------------------------
# The measurement model
# ---------------------------------------------------------------------------

def velocity_jacobian(build: JointKFBuild, params: JointKFParams | None = None) -> Array:
    """`H_qd = [0 | I_n | 0]`, shape `(n, dim)` — Java `getVelocityMeasurementJacobian`.

    Exactly the identity on the velocity block: the drive reports joint velocity
    directly, with no kinematic transformation and no bias term (the drive's own
    offset would be indistinguishable from `q_dot` here and is not modelled — the
    gyro bias state is per-IMU and belongs to a different channel entirely).

    `params` is accepted and unused, matching `measure.encoder_jacobian`'s two
    call styles.
    """
    n = build.n_joints
    return jnp.eye(n, build.dim, k=n, dtype=jnp.float64)


def velocity_noise(channel: VelocityChannel, carry: "VelocityCarry | None" = None) -> Array:
    """`R_qd = diag(sigma_i^2 + (dhat_i * inv_omega_i)^2)` — Java `getVelocityMeasurementNoise`.

    Diagonal: two drives' velocity errors share no mechanism, and the lag term is
    a per-joint function of that joint's own slew.

    `carry=None` gives the static floor `diag(sigma^2)` — the configuration the
    wiring and NIS tests use (Java's `cornerFn = null`).  With `inv_omega = 0` the
    two are identical anyway, so the argument is a convenience, not a second code
    path.
    """
    lag = jnp.zeros_like(channel.var) if carry is None else carry.dhat * channel.inv_omega
    return jnp.diag(channel.var + lag ** 2)


# ---------------------------------------------------------------------------
# The lag-inflation carry
# ---------------------------------------------------------------------------

class VelocityCarry(NamedTuple):
    """Per-tick state of the lag-inflation estimator — a pytree, never an attribute.

    Attributes
    ----------
    dhat : (n,) float
        Smoothed finite difference of the *measured* velocity [rad/s^2].  This is
        an estimate of the published signal's own slope, which by the first-order
        identity `u - y = y_dot / omega_eff` **is** the lag error up to the corner
        frequency.
    z_prev : (n,) float
        Previous tick's measurement, the other half of the finite difference.
    primed : () float
        `0.0` until the first sample has been seen, `1.0` after.  A float in the
        carry rather than a Python flag so the graph is identical on every tick
        (I7).  Without it, tick 0 differences against `z_prev = 0` and produces a
        `z/dt` spike — at 1 kHz, a 0.1 rad/s standing velocity becomes a phantom
        100 rad/s^2 slew, and the 5 Hz smoother then takes ~30 ms to forget it.
    """

    dhat: Array
    z_prev: Array
    primed: Array


def init_velocity_carry(build: JointKFBuild) -> VelocityCarry:
    """Zeroed carry, unprimed: no slew estimate and no history yet."""
    n = build.n_joints
    return VelocityCarry(
        dhat=jnp.zeros(n, dtype=jnp.float64),
        z_prev=jnp.zeros(n, dtype=jnp.float64),
        primed=jnp.zeros((), dtype=jnp.float64),
    )


def advance_slew(channel: VelocityChannel, carry: VelocityCarry, z: Array) -> VelocityCarry:
    r"""One tick of the slew smoother — Java `refreshDirectVelocityNoise`.

    ::

        fd    = (z - z_prev) / dt          (zeroed on the very first sample)
        dhat <- alpha * dhat + (1 - alpha) * fd
        alpha = exp(-2*pi*f_smooth*dt)

    `alpha` is the exact one-pole discretisation (`exp(-dt/tau)`), not the
    `1 - dt/tau` Euler approximation: at 5 Hz and 1 kHz they differ in the fifth
    decimal, but the exact form is unconditionally stable for any `dt` a caller
    might pass, and costs one build-time `exp`.

    Both the smoother and the priming gate are `jnp.where` on float carries, so
    the traced graph is the same on tick 0 and tick 10^6 (I7).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    fd = jnp.where(carry.primed > 0.0, (z - carry.z_prev) / channel.dt, 0.0)
    a = channel.smoother_alpha
    return VelocityCarry(
        dhat=a * carry.dhat + (1.0 - a) * fd,
        z_prev=z,
        primed=jnp.ones_like(carry.primed),
    )


# ---------------------------------------------------------------------------
# Diagnostics — separate fields ARE the port of Java's label dispatch
# ---------------------------------------------------------------------------

class ChannelDiagnostics(NamedTuple):
    """Per-joint consistency statistics, one field per measurement channel.

    Java publishes `jointKF_encNIS_<joint>` / `jointKF_qdNIS_<joint>` and
    dispatches on an exact-match label string.  Here the destination is chosen
    *structurally* — `velocity_update` can only construct the velocity half —
    which is the same guarantee without a string inside jit (I7).

    Every field is `(n,)` per-joint and initialises to `NaN`
    (`no_diagnostics`): a channel that did not run reports "no statistic", never
    a stale or borrowed number.  `NaN` specifically, because it cannot be
    mistaken for "in band" by a downstream consistency check the way `0.0` can.

    Attributes
    ----------
    encoder_nis, encoder_innovation : (n,)
        Written only by the position channel.
    velocity_nis, velocity_innovation : (n,)
        Written only by this module.
    velocity_r_diag : (n,)
        The `R` diagonal actually used this tick, i.e. floor + lag inflation —
        Java's `jointKF_qdR_<joint>`.  Published because the inflation is the
        channel's whole behaviour and is otherwise invisible from outside.
    """

    encoder_nis: Array
    encoder_innovation: Array
    velocity_nis: Array
    velocity_innovation: Array
    velocity_r_diag: Array


def no_diagnostics(build: JointKFBuild) -> ChannelDiagnostics:
    """All-`NaN` diagnostics: nothing has been measured yet on any channel."""
    nan = jnp.full(build.n_joints, jnp.nan, dtype=jnp.float64)
    return ChannelDiagnostics(
        encoder_nis=nan,
        encoder_innovation=nan,
        velocity_nis=nan,
        velocity_innovation=nan,
        velocity_r_diag=nan,
    )


def velocity_update(
    state: JointKFState,
    z: Array,
    build: JointKFBuild,
    params: JointKFParams,
    channel: VelocityChannel,
    carry: VelocityCarry,
    diagnostics: ChannelDiagnostics | None = None,
) -> tuple[JointKFState, VelocityCarry, ChannelDiagnostics, UpdateInfo]:
    """One direct-velocity measurement update — Java `josephUpdate(H, z, R, "encoderVelocity")`.

    Order matters and matches Java: the slew carry is advanced with **this**
    tick's measurement *before* `R` is built, because the lag error being modelled
    is the one present in the sample about to be fused, not the previous one.

    The update itself goes through the shared `joseph_update` so this channel
    inherits the `cond(S)` gate, the finite-mask hardening and the Joseph form
    unchanged — one gating semantics for every channel is the reason that function
    exists.

    Returns
    -------
    state : JointKFState
    carry : VelocityCarry
        Advanced slew state.  Note it is advanced whether or not the update was
        gated out: the drive kept publishing, so the slew estimate must keep
        tracking, or a single gated tick would leave `dhat` stale.
    diagnostics : ChannelDiagnostics
        `diagnostics` with **only** the velocity half replaced.
    info : UpdateInfo
        The shared per-update diagnostics (whole-channel NIS, `S`, gate flag).
    """
    from .diagnostics import per_joint_nis

    z = jnp.asarray(z, dtype=jnp.float64)
    carry = advance_slew(channel, carry, z)

    H = velocity_jacobian(build)
    R = velocity_noise(channel, carry)
    post, info = joseph_update(state, H, z, R, params, label="encoderVelocity")

    # NaN when the update was gated out, matching `UpdateInfo.nis`: a skipped
    # update has no consistency statistic, and NaN cannot be read as "in band".
    nis = jnp.where(info.was_applied > 0.0, per_joint_nis(info.nu, info.S), jnp.nan)

    base = no_diagnostics(build) if diagnostics is None else diagnostics
    diagnostics = base._replace(
        velocity_nis=nis,
        velocity_innovation=info.nu,
        velocity_r_diag=jnp.diag(R),
    )
    return post, carry, diagnostics, info
