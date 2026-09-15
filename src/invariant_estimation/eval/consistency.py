r"""Filter self-consistency from innovations alone -- the half of the objective that
needs no ground truth.

NIS vs NEES
-----------
``NEES`` compares the filter's error against a *reference* trajectory
(``ξᵀP⁻¹ξ`` with ``ξ`` measured against mocap) and therefore cannot be computed
without one. ``NIS`` compares each measurement's innovation against the
covariance the filter itself predicted for that innovation,
``ν = rᵀS⁻¹r`` -- entirely self-contained. Every quantity it needs is already
computed inside the filter and published: the InEKF's contact and gravity
``UpdateDiagnostics.nis``, and the joint filter's ``encoder_nis`` /
``stacked_nis``.

That makes NIS usable on **any robot log, with or without a mocap session** --
for evaluating a hand-tuned baseline, and (via `nis_consistency_loss`) as a
training objective or regularizer in its own right.

What the number means
---------------------
For a consistent filter each applied update's NIS is distributed ``χ²(d)``,
where ``d`` is that measurement's row count, so its expectation is exactly
``d``. Averaged over ``N`` independent applied updates, ``N·ANIS ~ χ²(N·d)``,
which is what `anis_band` inverts into an acceptance interval. Deviations have
a direction, and the direction is the tuning signal:

* ``ANIS > d`` -- innovations are **larger** than the filter predicted: it is
  overconfident, its Q/R are too small.
* ``ANIS < d`` -- innovations are **smaller** than predicted: it is
  conservative, its Q/R are too large, and it is under-using the measurement.

Both are real failures; only the first is dangerous, but the second is what a
loss minimizing raw error will happily drift into if nothing checks it.

Sampling caveats this module enforces
-------------------------------------
Only **applied** updates carry a meaningful NIS -- a gated update leaves the
state untouched and its diagnostic stale (NaN before the first update ever
runs). Both are filtered out here rather than silently averaged in.

The χ² band assumes independent samples. Consecutive ticks of a 1 kHz estimator
are emphatically not independent, so a band computed over every tick of a long
log is far too tight and will report "inconsistent" for a healthy filter. Pass
``effective_samples`` when the correlation time is known, or read a marginal
band violation as a hint rather than a verdict; the report carries the sample
count so this is visible rather than buried.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammainc


def chi2_cdf(x, dof):
    """``P(X <= x)`` for ``X ~ χ²(dof)``, via the regularized lower incomplete gamma.

    ``chi2.cdf(x; k) == gammainc(k/2, x/2)`` exactly; using JAX's own
    `gammainc` keeps this dependency-free rather than pulling in SciPy for one
    special function.
    """
    return gammainc(dof / 2.0, jnp.asarray(x, dtype=jnp.float64) / 2.0)


def chi2_quantile(probability: float, dof: float) -> float:
    """Inverse χ² CDF by bisection on `chi2_cdf`.

    Bisection rather than a closed-form approximation (Wilson-Hilferty and
    friends) because this runs once per report, never on a hot path, so there is
    no reason to accept an approximation's error -- especially at the small
    ``dof`` where those approximations are worst and where a study with few
    sessions actually lives.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    if not dof > 0:
        raise ValueError(f"dof must be positive, got {dof}")

    low, high = 0.0, max(2.0 * dof, 1.0)
    while float(chi2_cdf(high, dof)) < probability:
        high *= 2.0
        if high > 1e12:
            raise ValueError("chi-squared quantile failed to bracket the target probability")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if float(chi2_cdf(mid, dof)) < probability:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def anis_band(dof: float, samples: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided acceptance interval for the AVERAGE NIS over ``samples`` updates.

    ``N·ANIS ~ χ²(N·d)``, so the band on ANIS is the χ²(N·d) interval divided by
    ``N``. It tightens around ``d`` as ``N`` grows -- which is the whole point,
    and also why an inflated ``N`` from correlated samples makes the test far
    too strict.
    """
    if samples < 1:
        raise ValueError("need at least one applied update to form a band")
    total = dof * samples
    lower = chi2_quantile(alpha / 2.0, total) / samples
    upper = chi2_quantile(1.0 - alpha / 2.0, total) / samples
    return lower, upper


class ChannelConsistency(NamedTuple):
    """One measurement channel's NIS verdict over a trajectory.

    Attributes
    ----------
    channel : str
        Which measurement this is (``"contact"``, ``"encoder"``, ...).
    dof : float
        Measurement row count; the expected NIS under consistency.
    samples : int
        Applied, finite updates that entered the average.
    anis : float
        The average NIS over those samples. NaN when ``samples == 0``.
    lower, upper : float
        The acceptance band for ``anis`` at the requested confidence.
    verdict : str
        ``"consistent"``, ``"overconfident"`` (ANIS above the band -- Q/R too
        small), ``"conservative"`` (below -- Q/R too large), or ``"no-data"``.
    """

    channel: str
    dof: float
    samples: int
    anis: float
    lower: float
    upper: float
    verdict: str

    @property
    def ratio(self) -> float:
        """``ANIS / dof`` -- 1.0 is ideal; 2.0 means innovations are twice the predicted size."""
        return self.anis / self.dof

    def describe(self) -> str:
        if self.verdict == "no-data":
            return f"{self.channel}: no applied updates"
        return (
            f"{self.channel}: ANIS {self.anis:.3f} vs dof {self.dof:g} "
            f"(ratio {self.ratio:.2f}, band [{self.lower:.3f}, {self.upper:.3f}], "
            f"n={self.samples}) -> {self.verdict}"
        )


def _usable(nis, applied):
    """Applied AND finite samples, as a flat numpy array of NIS values."""
    nis = np.asarray(nis, dtype=float).reshape(-1)
    applied = np.asarray(applied, dtype=float).reshape(-1)
    if nis.shape != applied.shape:
        raise ValueError(f"nis and applied must have matching shapes, got {nis.shape} and {applied.shape}")
    # `applied` is a float mask (never a Python branch, so it survives jit); treat
    # anything nonzero as applied rather than testing == 1.0.
    return nis[(applied != 0.0) & np.isfinite(nis)]


def channel_consistency(channel: str, nis, applied, dof: float, *, alpha: float = 0.05,
                        effective_samples: int | None = None) -> ChannelConsistency:
    """Score one channel's stacked per-tick NIS against its χ² band.

    Parameters
    ----------
    nis, applied : array-like
        Per-tick NIS and its applied mask, as stacked by a `lax.scan` over the
        trajectory. Gated and pre-first-update (NaN) entries are dropped.
    dof : float
        Measurement row count for this channel.
    effective_samples : int, optional
        Independent-sample count to use for the band when consecutive ticks are
        correlated. Defaults to the raw applied count, which is optimistic for a
        high-rate log -- see the module docstring.
    """
    if not dof > 0:
        raise ValueError(f"dof must be positive, got {dof}")
    values = _usable(nis, applied)
    samples = int(values.size)
    if samples == 0:
        return ChannelConsistency(channel, dof, 0, float("nan"), float("nan"), float("nan"), "no-data")

    band_samples = samples if effective_samples is None else int(effective_samples)
    if band_samples < 1:
        raise ValueError("effective_samples must be at least 1")
    lower, upper = anis_band(dof, band_samples, alpha)

    anis = float(values.mean())
    if anis > upper:
        verdict = "overconfident"
    elif anis < lower:
        verdict = "conservative"
    else:
        verdict = "consistent"
    return ChannelConsistency(channel, dof, samples, anis, lower, upper, verdict)


def two_stage_consistency(joint_diagnostics, inekf_outputs, *, n_contacts: int, n_joints: int,
                          gravity_dof: float = 3.0, alpha: float = 0.05,
                          effective_samples: int | None = None) -> list[ChannelConsistency]:
    """Score every NIS channel a two-stage rollout publishes.

    Consumes the diagnostics the filters already emit -- `TickDiagnostics` from
    the joint filter and `InEKFOutputs` from the base filter -- rather than
    recomputing any innovation, so this can never disagree with what the filter
    actually did.

    Parameters
    ----------
    n_contacts : int
        Contact count ``N``; the contact update stacks 3 rows per contact.
    n_joints : int
        Joint count ``n``; the encoder channel observes one row per joint.
    gravity_dof : float
        Row count of the gravity-leveling measurement.
    """
    reports = [
        channel_consistency("encoder", joint_diagnostics.encoder_nis, joint_diagnostics.encoder_applied,
                            n_joints, alpha=alpha, effective_samples=effective_samples),
        channel_consistency("contact", inekf_outputs.contact_diagnostics.nis,
                            inekf_outputs.contact_diagnostics.applied,
                            3.0 * n_contacts, alpha=alpha, effective_samples=effective_samples),
        channel_consistency("gravity", inekf_outputs.gravity_diagnostics.nis,
                            inekf_outputs.gravity_diagnostics.applied,
                            gravity_dof, alpha=alpha, effective_samples=effective_samples),
    ]
    # The stacked gyro/anchor channel's row count varies per tick with the active
    # anchor count, so its dof is read from the per-row diagnostic rather than
    # assumed. Rows are NaN-padded to capacity; the finite ones are the live rows.
    per_row = np.asarray(joint_diagnostics.stacked_nis_per_row, dtype=float)
    if per_row.size:
        live_rows = np.isfinite(per_row).sum(axis=-1)
        applied = np.asarray(joint_diagnostics.stacked_applied, dtype=float).reshape(-1)
        active = live_rows.reshape(-1)[applied != 0.0]
        if active.size and active.min() == active.max():
            reports.append(channel_consistency("stacked", joint_diagnostics.stacked_nis,
                                               joint_diagnostics.stacked_applied, float(active[0]),
                                               alpha=alpha, effective_samples=effective_samples))
    return reports


def format_report(reports) -> str:
    """One line per channel, plus the reading key. For a console or a log header."""
    lines = [r.describe() for r in reports]
    lines.append("ratio > 1 => filter is overconfident (Q/R too small); < 1 => conservative (too large)")
    return "\n".join(lines)


def nis_consistency_loss(nis, applied, dof: float):
    """Differentiable, ground-truth-free consistency objective: ``(mean(NIS)/dof - 1)²``.

    Zero exactly when the average NIS equals its expectation. Normalizing by
    ``dof`` before squaring makes channels of different measurement dimension
    comparable, so several can be summed without the widest one dominating.

    Unlike `channel_consistency` this keeps everything in JAX and applies the
    ``applied`` mask by weighting rather than boolean indexing, so it stays
    jit/BPTT-safe: a gated update contributes zero to both the numerator and the
    sample count instead of changing the array's shape.

    Intended as a **regularizer alongside** a state-error loss, not a
    replacement for one. NIS is blind to a whole class of failure: a filter can
    be perfectly self-consistent about an estimate that is steadily wrong, since
    consistency only asks whether the errors match the covariance that was
    advertised -- not whether they are small.
    """
    if not dof > 0:
        raise ValueError(f"dof must be positive, got {dof}")
    nis = jnp.asarray(nis, dtype=jnp.float64).reshape(-1)
    weight = jnp.asarray(applied, dtype=jnp.float64).reshape(-1)
    if nis.shape != weight.shape:
        raise ValueError("nis and applied must have matching shapes")
    # A gated tick's NIS may be NaN; it must be zeroed BEFORE the multiply, since
    # 0.0 * NaN is NaN and would poison the sum and every gradient through it.
    weight = jnp.where(jnp.isfinite(nis), weight, 0.0)
    safe = jnp.where(jnp.isfinite(nis), nis, 0.0)
    count = jnp.sum(weight)
    mean = jnp.sum(safe * weight) / jnp.maximum(count, 1.0)
    return jnp.where(count > 0, (mean / dof - 1.0) ** 2, 0.0)
