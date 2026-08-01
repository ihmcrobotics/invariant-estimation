r"""The **optional** direct joint-velocity measurement channel.

Java `JointLevelKFPreFilter` `getVelocityMeasurementJacobian` /
`getVelocityMeasurementNoise` / `refreshDirectVelocityNoise`, ported test class
`JointLevelKFDirectVelocityMeasurementTest`.

Default **off** (`params.direct_velocity_enabled`).  It exists because some drives
publish a firmware velocity estimate better than anything the filter can infer
from encoder history; it is off for sim v1 because that signal does not exist in
MJX, and a channel whose noise model is calibrated against hardware firmware is
worse than no channel at all when fed a perfect derivative.

**What is measured is NOT `q_dot`** — the reason this module is more than
``H = [0 | I | 0]``.  The drive publishes the output of a first-order low-pass
with corner `omega_eff`::

    y_dot = omega_eff * (u - y),      u = true q_dot,   y = published value

so the measurement is *lagged*, and the lag error is not noise-like but a
deterministic function of the joint's current acceleration.  Rearranging gives
the identity the module rests on::

    u - y = y_dot / omega_eff      (EXACT, not a small-angle approximation)

The error is the *published signal's own slope* over the corner frequency, which
we estimate from the measurement itself and declare as an honest variance::

    R_ii(t) = sigma_i^2  +  ( dhat_i / omega_eff,i )^2

The second term is zero at constant velocity and large during a swing-leg
transient.  A *static* `R` must be sized for the worst case, leaving the channel
uselessly loose exactly when the robot is standing still — the regime where a
velocity measurement is worth the most.

**Why `dhat` is smoothed.**  It is a finite difference of the *noisy*
measurement, so raw its variance is ``2 sigma^2 / dt^2``: at
``sigma = 1e-2 rad/s`` and ``dt = 1e-3 s`` that is ``2e2 (rad/s^2)^2``, which
through ``(dhat/omega_eff)^2`` inflates `R` by roughly two orders of magnitude
*at quiet standing* — the exact opposite of the point.  So it is low-passed at
`params.lag_slew_smoothing_hz` (5 Hz: above the gait band so real slew passes;
far below the 500 Hz Nyquist so differentiation noise is cut by ~(5/500) in
amplitude).  The deterministic ramp test
(`lagInflationTracksMeasuredSlewExactly`) **cannot see this** — on a noiseless
ramp raw and smoothed both converge to `slope`, and on a constant signal both are
exactly zero — so the ported suite carries a noisy-input test in addition to the
Java scenario.

**Constant graph (I7).**  The smoothed slew is a carry, not an attribute:
`VelocityCarry` is a pytree of float arrays advanced with `jnp.where`.  The "have
we seen a sample yet" flag is a float in the same carry rather than a Python
`bool` — the first tick's finite difference against an unset `z_prev` would
otherwise be `z/dt`, a spike of hundreds of rad/s^2 that inflates `R` for the
~30 ms the smoother needs to forget it.

**Cross-talk.**  Java routes diagnostics by an exact-match measurement label
(`"encoder"` -> `jointKF_encNIS_*`, `"encoderVelocity"` -> `jointKF_qdNIS_*`) and
the ported test asserts that running the velocity channel leaves the encoder NIS
at `NaN`.  Strings cannot cross a jit boundary (I7), so the port replaces string
dispatch with **separate fields**: `velocity_update` structurally cannot write
`ChannelDiagnostics`' encoder half.  The observable is preserved; the mechanism
is a struct field instead of a string compare.
"""
from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..config import section
from .state import _ci_get, JointKFBuild, JointKFParams, JointKFState
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


# Build-time: per-joint noise floor and corner frequency (plain Python, I7)

def velocity_var_for_name(
    name: str,
    cfg: Mapping | None = None,
    lookup: Mapping[str, float] | None = None,
) -> tuple[float, bool]:
    """Per-joint velocity-measurement VARIANCE, and whether the lookup was wired.

    Mirrors `state.encoder_var_for_name`, including the "return the fallback *and
    say so*" contract: a joint silently on the fallback is the failure this
    per-joint machinery exists to prevent (invariant I9).

    The fallback is `sigma_qd_unfiltered` (0.1 rad/s => 0.01 rad^2/s^2), Java's
    `SIGMA_QD_FALLBACK` — one constant in Java and kept as one here, since both
    answer "what does an *unmodelled* joint velocity cost us".

    `lookup` overrides `cfg["encoder_vel_std"]` — the test seam standing in for
    Java's `velSigmaFor` function argument.  A non-finite or non-positive entry
    counts as *absent*, which is how the Java test injects an "unmatched" joint.
    """
    cfg = cfg if cfg is not None else section("joint_kf")
    table = lookup if lookup is not None else cfg.get("encoder_vel_std", {})
    std = _ci_get(table, name)
    if std is None or not np.isfinite(std) or std <= 0.0:
        return float(cfg["sigma_qd_unfiltered"]) ** 2, False
    return float(std) ** 2, True


class VelocityChannel(NamedTuple):
    """Static, name-resolved structure of the direct-velocity channel.

    Built once in plain Python (I7) and closed over by the jitted step, like
    `JointKFBuild`.  A separate struct because `JointKFBuild` is frozen and the
    channel is optional: a build with it disabled should not carry its arrays.

    `inv_omega` is **zero** for a joint with no declared corner frequency, which
    switches lag inflation off for that joint and leaves a static `R` — the port
    of Java's nullable `cornerFn`.  Zero is the right disabled value: an unknown
    corner is not "infinitely laggy", it is "we are not modelling this".
    """

    var: Array                  # (n,) sensor-noise variance sigma_i^2, the R floor
    wired: tuple[bool, ...]     # per joint: did the name lookup hit? build-time only
    inv_omega: Array            # (n,) 1/(2*pi*f_corner,i) [s] of drive low-pass lag
    smoother_alpha: float       # exp(-2*pi*f_smooth*dt), the slew smoother's pole
    dt: float                   # here so `advance_slew` needs only channel + carry


def build_velocity_channel(
    build: JointKFBuild,
    params: JointKFParams,
    *,
    vel_std: Mapping[str, float] | None = None,
    corner_hz: Mapping[str, float] | float | None = None,
    cfg: Mapping | None = None,
) -> VelocityChannel:
    """Resolve the per-joint velocity noise floor and corner frequency, once.

    `vel_std` maps joint name -> velocity-noise STD [rad/s] (Java's `velSigmaFor`;
    defaults to the config's `encoder_vel_std` sidecar).  `corner_hz` maps joint
    name -> drive low-pass corner [Hz], or is one scalar for all joints; `None`
    (the default) disables lag inflation entirely — Java's `cornerFn = null`, the
    configuration every wiring/NIS test uses.
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


def velocity_jacobian(build: JointKFBuild, params: JointKFParams | None = None) -> Array:
    """`H_qd = [0 | I_n | 0]`, shape `(n, dim)` — Java `getVelocityMeasurementJacobian`.

    Exactly the identity on the velocity block: no kinematic transformation and no
    bias term — the drive's own offset would be indistinguishable from `q_dot`
    here and is not modelled; the gyro bias state is per-IMU and belongs to a
    different channel.  `params` is accepted and unused, matching
    `measure.encoder_jacobian`'s two call styles.
    """
    n = build.n_joints
    return jnp.eye(n, build.dim, k=n, dtype=jnp.float64)


def velocity_noise(channel: VelocityChannel, carry: "VelocityCarry | None" = None) -> Array:
    """`R_qd = diag(sigma_i^2 + (dhat_i * inv_omega_i)^2)` — Java `getVelocityMeasurementNoise`.

    Diagonal: two drives' velocity errors share no mechanism, and the lag term is
    a per-joint function of that joint's own slew.  `carry=None` gives the static
    floor `diag(sigma^2)` — the configuration the wiring and NIS tests use (Java's
    `cornerFn = null`).  With `inv_omega = 0` the two are identical anyway, so the
    argument is a convenience, not a second code path.
    """
    lag = jnp.zeros_like(channel.var) if carry is None else carry.dhat * channel.inv_omega
    return jnp.diag(channel.var + lag ** 2)


class VelocityCarry(NamedTuple):
    """Per-tick state of the lag-inflation estimator — a pytree, never an attribute.

    `dhat` `(n,)` [rad/s^2] is the smoothed finite difference of the *measured*
    velocity: an estimate of the published signal's own slope, which by
    `u - y = y_dot / omega_eff` **is** the lag error up to the corner frequency.

    `primed` is `0.0` until the first sample has been seen, `1.0` after — a float
    in the carry rather than a Python flag so the graph is identical on every tick
    (I7).  Without it, tick 0 differences against `z_prev = 0` and produces a
    `z/dt` spike: at 1 kHz a 0.1 rad/s standing velocity becomes a phantom
    100 rad/s^2 slew, which the 5 Hz smoother takes ~30 ms to forget.
    """

    dhat: Array
    z_prev: Array    # previous tick's measurement, other half of the difference
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
    might pass, and costs one build-time `exp`.  Both the smoother and the priming
    gate are `jnp.where` on float carries, so the traced graph is the same on tick
    0 and tick 10^6 (I7).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    fd = jnp.where(carry.primed > 0.0, (z - carry.z_prev) / channel.dt, 0.0)
    a = channel.smoother_alpha
    return VelocityCarry(
        dhat=a * carry.dhat + (1.0 - a) * fd,
        z_prev=z,
        primed=jnp.ones_like(carry.primed),
    )


class ChannelDiagnostics(NamedTuple):
    """Per-joint consistency statistics, one field per measurement channel.

    Separate fields ARE the port of Java's label dispatch: it publishes
    `jointKF_encNIS_<joint>` / `jointKF_qdNIS_<joint>` by exact-match label
    string, while here the destination is chosen *structurally* —
    `velocity_update` can only construct the velocity half — which is the same
    guarantee without a string inside jit (I7).

    Every field is `(n,)` per-joint and initialises to `NaN` (`no_diagnostics`):
    a channel that did not run reports "no statistic", never a stale or borrowed
    number.  `NaN` specifically, because it cannot be mistaken for "in band" by a
    downstream consistency check the way `0.0` can.
    """

    encoder_nis: Array          # written only by the position channel
    encoder_innovation: Array
    velocity_nis: Array         # written only by this module
    velocity_innovation: Array
    velocity_r_diag: Array      # R diagonal used this tick (floor + lag
                                # inflation) -- Java `jointKF_qdR_<joint>`.
                                # Published because the inflation is the
                                # channel's whole behaviour and is otherwise
                                # invisible from outside.


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
    is the one present in the sample about to be fused, not the previous one.  The
    update goes through the shared `joseph_update` so this channel inherits the
    `cond(S)` gate, the finite-mask hardening and the Joseph form unchanged.

    The returned carry is advanced whether or not the update was gated out: the
    drive kept publishing, so the slew estimate must keep tracking, or a single
    gated tick would leave `dhat` stale.  Only the velocity half of `diagnostics`
    is replaced.
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
