r"""State and parameter types for the InEKF on ``SE_{N+2}(3)``, plus the two
state-independent constants the whole filter is built around: the propagation
transition ``Φ`` (§3.2) and the FK observation matrix ``H`` (§4.1).

The estimated group element is the ``(N+5)x(N+5)`` matrix

        ┌ R   v   p   d_1  …  d_N ┐
        │ 0   1   0   0    …  0   │
  X  =  │ 0   0   1   0    …  0   │
        │ 0   0   0   1    …  0   │
        │ ⋮               ⋱      │
        └ 0   0   0   0    …  1   ┘

  R   = {}^{W}R_{B}      base orientation (world ← body)
  v   = {}^{W}v_{B}      base linear velocity in world
  p   = {}^{W}p_{WB}     base position in world
  d_i = {}^{W}p_{WC_i}   world position of contact candidate i

The right-invariant error ``η^r = X̄ X⁻¹ = exp(ξ^)`` lives in ``ξ ∈ R^{3N+9}`` with
the fixed ordering (never permuted — ``P``, ``Φ`` and ``H`` all assume it):

  ξ = [ ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; … ; ξ_{d_N} ]

**``N`` is not stored** — it is implicit in ``d.shape[0]``.  Storing it would force
static-int handling and provoke recompiles.  All ``N`` contact candidates stay in
the state permanently (I2): a candidate not in contact is expressed through a large
contact covariance, never a shape change.
"""
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp

from ..config import section
from .group import skew


# Tangent-space layout (I4 — rotation-first, never permuted), locked by the ported
# `InvariantStateTest.testTangentIndices`.  Plain Python ints (static).

ROTATION_TANGENT_INDEX = 0
BASE_VELOCITY_TANGENT_INDEX = 3
BASE_POSITION_TANGENT_INDEX = 6
CONTACT_TANGENT_OFFSET = 9


def contact_tangent_index(i: int) -> int:
    """Tangent index of contact ``i``: ``9 + 3i`` (no bounds check — see method)."""
    return CONTACT_TANGENT_OFFSET + 3 * i


def _check_contact_index(i: int, N: int) -> None:
    """Reject out-of-range contact indices, Java-style (no negative wraparound)."""
    if not 0 <= i < N:
        raise IndexError(f"contact index {i} out of range for N = {N}")


class InEKFState(NamedTuple):
    """Sufficient statistic for the contact-aided InEKF: the group element's blocks
    ``(R, v, p, d)`` as named in the module docstring, plus the right-invariant error
    covariance ``P`` over ``ξ``.  A `NamedTuple`, so a valid pytree and a clean
    `lax.scan` carry; ``d``'s vmap axis is axis 0.
    """
    R: Array      # (3, 3)
    v: Array      # (3,)
    p: Array      # (3,)
    d: Array      # (N, 3)
    P: Array      # (3N+9, 3N+9)

    @property
    def N(self) -> int:
        """Number of contact candidates, inferred from `d.shape[0]`."""
        return self.d.shape[0]

    @property
    def dim(self) -> int:
        """Tangent / covariance dimension ``3N + 9``."""
        return 3 * self.N + 9

    # Java-parity aliases (`InvariantState`, ported suite): same numbers as
    # `N` / `dim`, named so the ported tests read 1:1 against `InvariantStateTest`.

    @property
    def group_size(self) -> int:
        """Side length ``N + 5`` of the dense group element ``X``."""
        return self.N + 5

    @property
    def tangent_size(self) -> int:
        """Tangent dimension ``3N + 9`` (alias of `dim`)."""
        return self.dim

    @classmethod
    def identity(cls, N: int) -> "InEKFState":
        """Fresh state at the group identity with **zero** covariance.

        Java `InvariantState(int numberOfContacts)`: ``P`` is zeros, *not*
        identity — a prior is applied separately by `init_state`.
        """
        return cls(
            R=jnp.eye(3),
            v=jnp.zeros(3),
            p=jnp.zeros(3),
            d=jnp.zeros((N, 3)),
            P=jnp.zeros((3 * N + 9, 3 * N + 9)),
        )

    def set_to_identity(self) -> "InEKFState":
        """Reset ``X`` to the group identity, leaving ``P`` untouched (Java `setToIdentity()`)."""
        return self._replace(
            R=jnp.eye(3),
            v=jnp.zeros(3),
            p=jnp.zeros(3),
            d=jnp.zeros_like(self.d),
        )

    def get_contact_position(self, i: int) -> Array:
        """World position ``d_i`` of contact ``i``; `IndexError` if out of range."""
        _check_contact_index(i, self.N)
        return self.d[i]

    def set_contact_position(self, i: int, d_i: Array) -> "InEKFState":
        """Return a copy with contact ``i`` set to ``d_i``; `IndexError` if out of range."""
        _check_contact_index(i, self.N)
        return self._replace(d=self.d.at[i].set(d_i))

    def contact_tangent_index(self, i: int) -> int:
        """Tangent index ``9 + 3i`` of contact ``i``; `IndexError` if out of range."""
        _check_contact_index(i, self.N)
        return contact_tangent_index(i)

    @property
    def as_matrix(self) -> Array:
        """Dense ``(N+5, N+5)`` group element ``X``, built on demand from `(R, v, p, d)`."""
        N = self.N
        X = jnp.eye(N + 5)
        X = X.at[0:3, 0:3].set(self.R)
        X = X.at[0:3, 3].set(self.v)
        X = X.at[0:3, 4].set(self.p)
        X = X.at[0:3, 5:].set(self.d.T)   # d_i as columns 5 … N+4
        return X


def build_Phi(g: Array, dt: float, N: int) -> Array:
    r"""Constant right-invariant transition ``Φ = expm(A^r dt)``, shape ``(3N+9, 3N+9)`` (§3.2).

    With no bias in the state (I1) the error dynamics matrix ``A^r`` is constant and
    nilpotent (``(A^r)³ = 0``), so ``Φ`` is the exact closed form

        ┌ I            0      0   0 ┐   (R)
        │ (g)_× dt     I      0   0 │   (v)
        │ ½(g)_× dt²   I dt   I   0 │   (p)
        └ 0            0      0   I ┘   (d, all identity — contacts uncoupled)

    independent of ``R̄, v̄, p̄`` and of the IMU input.  Built directly from the
    closed form; `expm` is never called in the scan body.
    """
    G = skew(g)
    Phi = jnp.eye(3 * N + 9)
    Phi = Phi.at[3:6, 0:3].set(G * dt)               # [v, R]
    Phi = Phi.at[6:9, 3:6].set(jnp.eye(3) * dt)      # [p, v]
    Phi = Phi.at[6:9, 0:3].set(0.5 * G * dt * dt)    # [p, R]
    return Phi


def build_H(N: int) -> Array:
    r"""Constant FK observation matrix ``H``, shape ``(3N, 3N+9)`` (§4.1).

    Each contact's right-invariant FK observation has Jacobian
    ``H_i = [ 0  0  +I  …  −I(col d_i)  … ]``.  Stacked over contacts::

        H = [ 0_{3N×3} | 0_{3N×3} | (+I_3 ×N) | −I_{3N} ]

    State-independent by construction (world-centric + right-invariant).

    **Sign convention** — this is the Java `ContactUpdater.computeJacobian` layout,
    locked element-wise (tol 0.0) by the ported
    `ContactUpdaterTest.testJacobianStructureAndStateIndependence`, and it is what
    makes I5 read literally: the residual linearises as ``ν ≈ +H ξ``, so the
    correction ``ξ⁺ = Kν`` *estimates* the error and is removed by
    ``X̂⁺ = exp(−(Kν)^∧) X̂``.
    """
    H = jnp.zeros((3 * N, 3 * N + 9))
    H = H.at[:, 6:9].set(jnp.tile(jnp.eye(3), (N, 1)))    # p block: +I per contact
    H = H.at[:, 9:9 + 3 * N].set(-jnp.eye(3 * N))         # d block: −I
    return H


class InEKFParams(NamedTuple):
    """Run-fixed configuration plus the precomputed constants ``Φ`` and ``H``.

    Passed into the filter step rather than stored in the mutable state, and
    precomputed here so the scan body never rebuilds them — in particular `expm` is
    never called in the loop.  The IMU noises are continuous **variance** densities.
    """
    g: Array            # (3,) gravity accel [m/s²], world
    dt: float           # [s]
    gyro_var: float     # [(rad/s)²/Hz] → Q_g = gyro_var · I₃
    accel_var: float    # [(m/s²)²/Hz]  → Q_a = accel_var · I₃
    contact_floor: float  # [m²] variance floor on contact covariances
    Phi: Array          # (3N+9, 3N+9) precomputed transition
    H: Array            # (3N, 3N+9) precomputed FK observation


def init_state(
    N: int,
    R0: Array | None = None,
    v0: Array | None = None,
    p0: Array | None = None,
    d0: Array | None = None,
    p_R: float | None = None,
    p_v: float | None = None,
    p_p: float | None = None,
    p_d: float | None = None,
) -> InEKFState:
    """Initial `InEKFState` with a diagonal prior covariance.

    ``R0`` ``(3,3)`` / ``v0`` ``(3,)`` / ``p0`` ``(3,)`` / ``d0`` ``(N,3)`` default
    to identity / zeros.  The prior variances ``p_R, p_v, p_p, p_d`` default to
    ``inekf.init`` in ``config/filter_cfg.yaml``; contacts take a diffuse ``p_d``
    because a candidate not yet in firm contact is "off" via a large covariance.
    """
    prior = section("inekf")["init"]
    p_R = prior["rotation_var"] if p_R is None else p_R
    p_v = prior["velocity_var"] if p_v is None else p_v
    p_p = prior["position_var"] if p_p is None else p_p
    p_d = prior["contact_var"] if p_d is None else p_d

    R0 = jnp.eye(3) if R0 is None else R0
    v0 = jnp.zeros(3) if v0 is None else v0
    p0 = jnp.zeros(3) if p0 is None else p0
    d0 = jnp.zeros((N, 3)) if d0 is None else d0

    P = jnp.diag(
        jnp.concatenate([
            jnp.full(3, p_R),
            jnp.full(3, p_v),
            jnp.full(3, p_p),
            jnp.full(3 * N, p_d),
        ])
    )
    return InEKFState(R=R0, v=v0, p=p0, d=d0, P=P)


def default_params(
    N: int,
    dt: float | None = None,
    g: Array | None = None,
    gyro_var: float | None = None,
    accel_var: float | None = None,
    contact_floor: float | None = None,
) -> InEKFParams:
    """Default `InEKFParams` for an ``N``-contact filter at 1 kHz, with ``Φ`` and ``H`` precomputed.

    Every argument defaults to the ``inekf`` section of ``config/filter_cfg.yaml``;
    pass a value to override it.  The noise densities there are starting-point
    values (untuned) — match them to the IMU spec and the ContactNet covariance
    scale once the baseline is consistent.
    """
    cfg = section("inekf")
    dt = cfg["dt"] if dt is None else dt
    gyro_var = cfg["gyro_var"] if gyro_var is None else gyro_var
    accel_var = cfg["accel_var"] if accel_var is None else accel_var
    contact_floor = cfg["contact_floor"] if contact_floor is None else contact_floor
    g = jnp.asarray(cfg["gravity"], dtype=float) if g is None else g

    return InEKFParams(
        g=g,
        dt=dt,
        gyro_var=gyro_var,
        accel_var=accel_var,
        contact_floor=contact_floor,
        Phi=build_Phi(g, dt, N),
        H=build_H(N),
    )
