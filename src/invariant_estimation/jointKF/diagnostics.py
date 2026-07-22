r"""
jointKF/diagnostics.py
======================
Two observables the filter publishes but does not act on: the **per-joint NIS**
of a diagonal channel, and the **attribution of a near-singular innovation
covariance** to the physical measurement that caused it (Java
`describeSingularInnovation`, ported class
`JointLevelKFSingularInnovationDiagnosticTest`).

Why attribution is worth code
-----------------------------
`update.py` already *handles* a singular `S`: the `cond(S)` gate drops the update
and leaves `(x, P)` bit-identical.  That is the correct runtime behaviour and it
is silent by design — which is the problem.  On hardware the failure looked like
"the estimator stopped updating", with 40+ measurement rows and no indication of
which one had gone degenerate.  The gate protects the filter; this module is what
lets a human find the sensor.

Port the observable, not the message (CLAUDE.md §5)
---------------------------------------------------
Java returns a human-readable string and its test asserts on substrings
(`"gyro pair 0"`, the base IMU's sensor name, `"encoder q of joint <name>"`).  A
string is not a portable contract, so this module returns **structured
attribution** — row index -> channel, ordinal, name, dominant state column — and
offers `.summary()` for the log line.  The ported tests assert on the structure;
the text is free to change.

How the attribution works
-------------------------
`S = H P H^T + R` is symmetric PSD.  Near-singularity means some direction `v` in
*measurement* space has almost no innovation variance: `v^T S v ~ 0`.  Take `v` =
the eigenvector of the smallest eigenvalue; the rows with large `|v_i|` are the
rows that participate in the degenerate combination.  Two rows measuring the same
physical quantity with tight `R` give `v ~ (1, -1)/sqrt(2)` — their *difference*
is unobservably small, which is exactly the "duplicate measurement" pathology.

Each implicated row is then described twice over, because neither description
alone is enough:

* by **row block** — which channel and which ordinal within it (gyro pair 3,
  anchor 1, encoder row 5).  The stacked layout owns this and `H` cannot tell you
  it: a gyro row's columns identify joints, not the pair.
* by **dominant state column** of that row of `H` — which state the row actually
  loads.  This is what names the *joint* in the encoder case, and it is not the
  same as the row ordinal: the Java encoder test builds two rows that both put
  their weight on joint 0's column, so row 1 must be reported as observing joint
  0, not joint 1.

Host-side, deliberately
-----------------------
This runs off the jit path: it eigendecomposes, allocates, and returns Python
strings.  It is called when a gate has already fired (`UpdateInfo.was_applied ==
0`), i.e. never in the hot loop, so I7 does not apply and NumPy is the right
tool.  `per_joint_nis` is the exception — it is pure array arithmetic and is
called every tick from inside the traced step.
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


# ---------------------------------------------------------------------------
# Per-joint NIS
# ---------------------------------------------------------------------------

def per_joint_nis(nu: Array, S: Array) -> Array:
    r"""Per-row normalised innovation squared `nu_i^2 / S_ii`.

    This is the **marginal** consistency statistic, not a slice of the joint one:
    each `nu_i` is marginally `N(0, S_ii)`, so `nu_i^2 / S_ii ~ chi^2_1` (mean 1,
    variance 2) regardless of the correlations `S` carries off the diagonal.  The
    whole-channel statistic `nu^T S^-1 nu ~ chi^2_k` is the one `UpdateInfo.nis`
    reports; the per-joint form is what localises a bad encoder to a joint, which
    the aggregate cannot do.

    Both must be computed on the **prior** `S` and the **prior** residual
    (CLAUDE.md §6) — `UpdateInfo` supplies exactly those, which is why this takes
    `(nu, S)` rather than a state.

    Parameters
    ----------
    nu : (k,)   prior innovation
    S : (k, k)  prior innovation covariance

    Returns
    -------
    (k,) array
    """
    return jnp.asarray(nu) ** 2 / jnp.diag(jnp.asarray(S))


# ---------------------------------------------------------------------------
# Near-singular innovation attribution
# ---------------------------------------------------------------------------

class RowAttribution(NamedTuple):
    """One measurement row's share of a degenerate direction.

    Attributes
    ----------
    row : int
        Index into the channel's stacked measurement.
    channel : str
        `"gyro_pair"`, `"anchor"`, `"encoder"` or `"velocity"`.
    ordinal : int
        Index within the channel: pair number, anchor slot, or joint number.
    name : str
        Human-readable identification of the *physical* measurement, e.g.
        `"gyro pair 0 (imu0 -> imu1)"`.
    state_index : int
        The state column this row loads most heavily — `-1` for an all-zero row.
    state_name : str
        That column, named: `"q of joint3"`, `"q_dot of joint3"`,
        `"b_omega[1] of imu0"`.
    weight : float
        `v_i^2` for the near-null eigenvector `v`; the rows sum to 1.
    """

    row: int
    channel: str
    ordinal: int
    name: str
    state_index: int
    state_name: str
    weight: float


class SingularInnovationReport(NamedTuple):
    """Structured answer to "which sensor made `S` singular?".

    Attributes
    ----------
    label : str
        The channel label the caller passed (Java's `label` argument).
    reason : str
        Free text from the caller (Java's `reason`).
    condition_number : float
        `max_eig / min_eig` of `S`, `inf` if `min_eig <= 0`.  This is the honest
        eigenvalue condition number, not `update.py`'s Cholesky-diagonal proxy:
        the proxy is what the hot path can afford, this is what the diagnostic
        should report.
    min_eigenvalue : float
    null_vector : np.ndarray, shape (k,)
        Eigenvector of the smallest eigenvalue, sign-normalised so its
        largest-magnitude entry is positive (an eigenvector's sign is arbitrary;
        pinning it makes the report reproducible).
    rows : tuple[RowAttribution, ...]
        The implicated rows, heaviest first.
    """

    label: str
    reason: str
    condition_number: float
    min_eigenvalue: float
    null_vector: np.ndarray
    rows: tuple[RowAttribution, ...]

    def summary(self) -> str:
        """The log line — Java's message, rebuilt from the structure.

        Deliberately derived from `rows` rather than being the primary product:
        the tests assert on `rows`, so the text can be reworded without breaking
        anything, which is the point of porting the observable instead of the
        message.
        """
        who = "; ".join(f"{r.name} [{r.state_name}, weight {r.weight:.2f}]" for r in self.rows)
        return (f"near-singular innovation covariance in '{self.label}' "
                f"({self.reason}): cond(S) = {self.condition_number:.3e}, "
                f"min eig = {self.min_eigenvalue:.3e}; degenerate direction carried by {who}")


def _state_name(build: JointKFBuild, index: int) -> str:
    """Name a state column: `q`, `q_dot`, or an axis of some IMU's bias."""
    n, m = build.n_joints, build.n_imus
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

    The stacked channel is the interesting one: rows are grouped in threes,
    `3e..3e+2` for pair `e` and then `anchor_row0 + 3k..` for anchor `k`
    (`JointKFBuild.stacked_row_for_pair` / `anchor_row0`).  A gyro row is named by
    its pair *and* both IMUs, because "pair 0" alone does not tell an operator
    which box to go and look at — Java's message includes the sensor name for
    exactly that reason.

    For the diagonal channels (encoder, velocity) the ordinal comes from the
    **dominant state column**, not the row index: a duplicated row observes a
    joint other than its own, and reporting the row's own ordinal would name the
    wrong joint (the Java encoder scenario is precisely this).
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

    Parameters
    ----------
    build : JointKFBuild
        Supplies the row layout and the names.
    H : (k, dim), R : (k, k), P : (dim, dim)
        The measurement the gate rejected, and the prior covariance it was formed
        against.  `P` is taken as an argument rather than a state so the caller
        can pass the *prior* explicitly — attributing against a posterior would
        describe a matrix that was never inverted.
    channel : {"stacked", "encoder", "velocity"}
        Which row layout `H` follows.  This is the port of Java's exact-match
        label dispatch; it is a host-side Python string, so I7 does not apply.
    label, reason : str
        Carried into the report for the log line only.
    weight_floor : float
        Rows with `v_i^2` below this are omitted as noise.  0.05 keeps a row that
        carries 5% of the degenerate direction and drops numerical dust; with `k`
        rows the uniform share is `1/k`, so this floor never hides a genuine
        participant for the row counts this filter uses.

    Returns
    -------
    SingularInnovationReport
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
