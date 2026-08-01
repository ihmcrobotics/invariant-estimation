r"""Two observables the filter publishes but does not act on: the **per-joint NIS**
of a diagonal channel, and the **attribution of a near-singular innovation
covariance** to the physical measurement that caused it.

Java `describeSingularInnovation`, ported class
`JointLevelKFSingularInnovationDiagnosticTest`.

`update.py` already *handles* a singular `S` — the `cond(S)` gate drops the update
and leaves `(x, P)` bit-identical — and is silent by design, which is the problem.
On hardware the failure looked like "the estimator stopped updating", with 40+
measurement rows and no indication of which had gone degenerate.  The gate
protects the filter; this module lets a human find the sensor.

**Port the observable, not the message (CLAUDE.md §5).**  Java returns a
human-readable string and asserts on substrings.  A string is not a portable
contract, so this returns **structured attribution** — row index -> channel,
ordinal, name, dominant state column — with `.summary()` for the log line.  The
ported tests assert on the structure; the text is free to change.

**How it works.**  `S = H P H^T + R` is symmetric PSD, so near-singularity means
some direction `v` in *measurement* space has `v^T S v ~ 0`.  Take `v` = the
eigenvector of the smallest eigenvalue; rows with large `|v_i|` participate in the
degenerate combination.  Two rows measuring the same physical quantity with tight
`R` give `v ~ (1, -1)/sqrt(2)` — the "duplicate measurement" pathology.

Each implicated row is described twice, because neither alone is enough: by **row
block** (channel and ordinal — the stacked layout owns this and `H` cannot tell
you it, since a gyro row's columns identify joints, not the pair), and by
**dominant state column** of that row of `H`.  The second is what names the
*joint* in the encoder case, and it differs from the row ordinal: the Java encoder
test builds two rows both weighted on joint 0's column, so row 1 must be reported
as observing joint 0.

Host-side deliberately: this eigendecomposes, allocates, and returns Python
strings, and is called only after a gate has fired (`UpdateInfo.was_applied ==
0`), never in the hot loop, so I7 does not apply and NumPy is the right tool.
`per_joint_nis` is the exception — pure array arithmetic, called every tick from
inside the traced step.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .state import JointKFBuild

__all__ = [
    "per_joint_nis",
    "RowAttribution",
    "SingularInnovationReport",
    "describe_singular_innovation",
]


def per_joint_nis(nu: Array, S: Array) -> Array:
    r"""Per-row normalised innovation squared `nu_i^2 / S_ii`, from the PRIOR `(nu, S)`.

    The **marginal** consistency statistic, not a slice of the joint one: each
    `nu_i` is marginally `N(0, S_ii)`, so `nu_i^2 / S_ii ~ chi^2_1` regardless of
    `S`'s off-diagonal correlations.  `UpdateInfo.nis` reports the whole-channel
    `nu^T S^-1 nu ~ chi^2_k`; this form is what localises a bad encoder to a
    joint, which the aggregate cannot do.

    Both must be computed on the **prior** `S` and the **prior** residual
    (CLAUDE.md §6) — `UpdateInfo` supplies exactly those, which is why this takes
    `(nu, S)` rather than a state.
    """
    return jnp.asarray(nu) ** 2 / jnp.diag(jnp.asarray(S))


class RowAttribution(NamedTuple):
    """One measurement row's share of a degenerate direction."""

    row: int            # index into the channel's stacked measurement
    channel: str        # "gyro_pair", "anchor", "encoder" or "velocity"
    ordinal: int        # index within the channel: pair, anchor slot, or joint
    name: str           # the physical measurement, e.g. "gyro pair 0 (imu0 -> imu1)"
    state_index: int    # state column this row loads most; -1 for an all-zero row
    state_name: str     # that column named, e.g. "b_omega[1] of imu0"
    weight: float       # v_i^2 for the near-null eigenvector v; rows sum to 1


class SingularInnovationReport(NamedTuple):
    """Structured answer to "which sensor made `S` singular?".

    `condition_number` is the honest eigenvalue `max_eig / min_eig` (`inf` if
    `min_eig <= 0`), not `update.py`'s Cholesky-diagonal proxy: the proxy is what
    the hot path can afford, this is what the diagnostic should report.
    `null_vector` is sign-normalised so its largest-magnitude entry is positive —
    an eigenvector's sign is arbitrary, and pinning it makes the report
    reproducible.  `rows` is heaviest first.
    """

    label: str          # the channel label the caller passed (Java's `label`)
    reason: str         # free text from the caller (Java's `reason`)
    condition_number: float
    min_eigenvalue: float
    null_vector: np.ndarray
    rows: tuple[RowAttribution, ...]

    def summary(self) -> str:
        """The log line — Java's message, rebuilt from the structure.

        Derived from `rows` rather than being the primary product: the tests
        assert on `rows`, so the text can be reworded without breaking anything.
        """
        who = "; ".join(f"{r.name} [{r.state_name}, weight {r.weight:.2f}]" for r in self.rows)
        return (f"near-singular innovation covariance in '{self.label}' "
                f"({self.reason}): cond(S) = {self.condition_number:.3e}, "
                f"min eig = {self.min_eigenvalue:.3e}; degenerate direction carried by {who}")


def _state_name(build: JointKFBuild, index: int) -> str:
    """Name a state column: `q`, `q_dot`, or an axis of some IMU's bias."""
    n = build.n_joints
    if index < 0:
        return "none"
    if index < n:
        return f"q of {build.joint_names[index]}"
    if index < 2 * n:
        return f"q_dot of {build.joint_names[index - n]}"
    k, axis = divmod(index - 2 * n, 3)
    return f"b_omega[{axis}] of {build.imu_names[k]}"


def _row_identity(build: JointKFBuild, row: int, channel: str, state_index: int) -> tuple[str, int, str]:
    """`(channel, ordinal, name)` for one measurement row.

    Stacked rows are grouped in threes, `3e..3e+2` for pair `e` then
    `anchor_row0 + 3k..` for anchor `k`.  A gyro row is named by its pair *and*
    both IMUs, because "pair 0" alone does not tell an operator which box to look
    at — Java's message includes the sensor name for that reason.

    For the diagonal channels (encoder, velocity) the ordinal comes from the
    **dominant state column**, not the row index: a duplicated row observes a
    joint other than its own, and reporting the row's own ordinal would name the
    wrong joint (precisely the Java encoder scenario).
    """
    n = build.n_joints
    if channel == "stacked":
        if row < build.anchor_row0:
            e = row // 3
            parent = build.imu_names[int(build.pair_parent[e])]
            child = build.imu_names[int(build.pair_child[e])]
            return "gyro_pair", e, f"gyro pair {e} ({parent} -> {child})"
        k = (row - build.anchor_row0) // 3
        imu = build.imu_names[int(build.anchor_imu[k])] if len(build.anchor_imu) else "?"
        return "anchor", k, f"stance anchor {k} (pins {imu})"
    if channel == "encoder":
        j = state_index if 0 <= state_index < n else row
        return "encoder", j, f"encoder q of {build.joint_names[j]}"
    if channel == "velocity":
        j = state_index - n if n <= state_index < 2 * n else row
        return "velocity", j, f"direct velocity of {build.joint_names[j]}"
    raise ValueError(f"unknown channel {channel!r} (expected stacked/encoder/velocity)")


def describe_singular_innovation(
    build: JointKFBuild,
    H,
    R,
    P,
    *,
    channel: str,
    label: str = "",
    reason: str = "",
    weight_floor: float = 0.05,
) -> SingularInnovationReport:
    """Attribute a near-singular `S = H P H^T + R` to measurement rows.

    `H` `(k, dim)`, `R` `(k, k)`, `P` `(dim, dim)` are the measurement the gate
    rejected and the prior covariance it was formed against.  `P` is an argument
    rather than a state so the caller passes the *prior* explicitly — attributing
    against a posterior would describe a matrix that was never inverted.

    `channel` is `"stacked"`, `"encoder"` or `"velocity"`: which row layout `H`
    follows, the port of Java's exact-match label dispatch.  Host-side Python
    string, so I7 does not apply.  `label`/`reason` are carried into the report
    for the log line only.

    `weight_floor` omits rows carrying less than `v_i^2` of the degenerate
    direction.  0.05 keeps a 5% participant and drops numerical dust; with `k`
    rows the uniform share is `1/k`, so it never hides a genuine participant at
    the row counts this filter uses.
    """
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)

    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)
    eig, vec = np.linalg.eigh(S)
    v = vec[:, 0]
    # Pin the sign: an eigenvector's sign is arbitrary and would otherwise make
    # the reported null direction flip between runs on the same input.
    if v[np.argmax(np.abs(v))] < 0.0:
        v = -v

    lo, hi = float(eig[0]), float(eig[-1])
    cond = float(hi / lo) if lo > 0.0 else float("inf")

    weights = v ** 2
    order = np.argsort(-weights)
    rows: list[RowAttribution] = []
    for row in order:
        w = float(weights[row])
        if w < weight_floor:
            break
        h = H[row]
        state_index = int(np.argmax(np.abs(h))) if np.any(h != 0.0) else -1
        ch, ordinal, name = _row_identity(build, int(row), channel, state_index)
        rows.append(RowAttribution(
            row=int(row),
            channel=ch,
            ordinal=ordinal,
            name=name,
            state_index=state_index,
            state_name=_state_name(build, state_index),
            weight=w,
        ))

    return SingularInnovationReport(
        label=label,
        reason=reason,
        condition_number=cond,
        min_eigenvalue=lo,
        null_vector=v,
        rows=tuple(rows),
    )
