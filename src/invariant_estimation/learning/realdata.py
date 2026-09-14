r"""Design decisions and pure functions for turning real sensor data into the
`TwoStageInputs` contract (Tier 1, remaining items 2/3).

**Scope boundary**: this module is not a robot-log decoder and does not read
any file format. It supplies the two pieces of that adapter that were
genuinely undesigned before now -- `contact_chol` and `accel_body` -- as pure,
jit/vmap-safe functions with an explicit contract, so they can be written and
tested today against synthetic inputs and wired to a real per-tick reader the
moment a real capture exists.

Contact condition (``contact_chol``)
-------------------------------------
`inEKF/filter.py`'s `InEKFInputs.contact_chol` docstring names ContactNet as
the eventual source, with an explicit fallback: "Default heuristic: a
constant diagonal factor, inflated for swing feet." `contact_chol_heuristic`
implements exactly that fallback, driven by the same per-foot contact
probability `FootSwitchContactProbabilityProvider` already produces on the
Java side (`us.ihmc.stateEstimation.invariantEstimator`) -- p=1 firm stance,
p=0 swing, degrading smoothly in between via the debounced-Schmitt EMA that
provider already applies. This is deliberately NOT a new detector: it consumes
whatever probability the existing production contact logic emits and turns it
into a covariance shape, nothing more.

Accelerometer bias (``accel_body``)
-------------------------------------
`two_stage.py`'s `make_step` docstring is explicit: "Accelerometer bias
correction remains the caller's responsibility; JointKF estimates gyro bias
only" -- there is no accel-bias *state* anywhere in this pipeline (Java or
JAX). The design decision made here is to NOT add one: online accel-bias
estimation would need its own observability argument (accel bias is only
weakly observable from a single IMU without an independent velocity/position
reference) that this project does not need to make, because IMU accelerometer
bias is small and slowly-varying enough to calibrate once per session from a
stationary window at capture start -- standard practice, and exactly what a
mocap-supervised capture protocol can guarantee (the robot starts each session
standing still while mocap locks on). `estimate_static_accel_bias` performs
that one-shot, non-differentiable calibration; `accel_body_from_raw` is the
per-tick correction it feeds, which IS meant to run inside the differentiated
scan.
"""
import jax
import jax.numpy as jnp
from jax import Array


def contact_chol_heuristic(contact_probability: Array, firm_variance: float, swing_variance: float) -> Array:
    r"""Per-contact diagonal Cholesky factor from a scalar trust probability.

    ``variance(p) = swing_variance + p * (firm_variance - swing_variance)``,
    linear in probability so `p=1` (firm) gives exactly `firm_variance` and
    `p=0` (swing/no contact) gives exactly `swing_variance`; `L = sqrt(variance) * I3`.
    Linear (not log-linear) so the map stays smooth and finite-valued at `p=0`
    and `p=1` with no division, keeping BPTT through a live probability signal
    well-defined even though this heuristic itself is not trained.

    Parameters
    ----------
    contact_probability : Array, shape (N,)
        Per-contact trust in [0, 1], e.g. `FootSwitchContactProbabilityProvider`
        `getContactProbability` on the Java side, or its JAX equivalent.
    firm_variance : float
        Contact-position process variance when fully trusted (small).
    swing_variance : float
        Contact-position process variance with no trust (large).

    Returns
    -------
    Array, shape (N, 3, 3)
        Lower-triangular (here: diagonal) Cholesky factors, ready for
        `inEKF.contact.digest`.
    """
    if not (0.0 < firm_variance < swing_variance):
        raise ValueError("firm_variance must be positive and less than swing_variance")
    p = jnp.clip(contact_probability, 0.0, 1.0)
    variance = swing_variance + p * (firm_variance - swing_variance)
    scale = jnp.sqrt(variance)
    return scale[:, None, None] * jnp.eye(3)[None]


def estimate_static_accel_bias(accel_raw_body_samples: Array, gravity_body_at_rest: Array) -> Array:
    r"""One-shot accelerometer bias from a stationary window (not part of the scan).

    Parameters
    ----------
    accel_raw_body_samples : Array, shape (T, 3)
        Raw specific-force samples from a window where the robot is known to
        be stationary (e.g. the first N ticks of a capture, before it starts
        moving) and already rotated into the pelvis/body frame.
    gravity_body_at_rest : Array, shape (3,)
        The specific force a bias-free, stationary IMU should read in the
        body frame (i.e. ``-g`` expressed in body axes at the calibration
        pose) -- NOT the InEKF's internal gravity convention, which is the
        caller's to reconcile.

    Returns
    -------
    Array, shape (3,)
        ``bias = mean(samples) - gravity_body_at_rest``, to subtract from
        every subsequent raw sample via `accel_body_from_raw`.
    """
    if accel_raw_body_samples.ndim != 2 or accel_raw_body_samples.shape[1] != 3:
        raise ValueError("accel_raw_body_samples must have shape (T, 3)")
    if accel_raw_body_samples.shape[0] < 1:
        raise ValueError("at least one stationary sample is required")
    return jnp.mean(accel_raw_body_samples, axis=0) - gravity_body_at_rest


def accel_body_from_raw(accel_raw_body: Array, bias: Array) -> Array:
    r"""Per-tick, differentiable accel-bias correction: ``accel_raw_body - bias``.

    Meant to run inside the `two_stage` scan (`TwoStageInputs.accel_body`),
    unlike `estimate_static_accel_bias`, which runs once per session before
    the scan starts. `bias` is treated as a fixed per-session constant here,
    never re-estimated online.
    """
    return accel_raw_body - bias
