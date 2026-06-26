"""
inEKF/state.py
==============
State and parameter types for the world-centric, right-invariant contact-aided
InEKF on ``SE_{N+2}(3)`` (CLAUDE.md §1), plus the two state-independent constants
the whole filter is built around: the propagation transition ``Φ`` (§3.2) and the
FK observation matrix ``H`` (§4.1).

The estimated group element is the ``(N+5)x(N+5)`` matrix (left-superscript /
Traversaro notation)

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

The right-invariant error ``η^r = X̄ X⁻¹ = exp(ξ^)`` lives in ``ξ ∈ R^{3N+9}``
with the fixed ordering (never permuted — ``P``, ``Φ`` and ``H`` all assume it):

  ξ = [ ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; … ; ξ_{d_N} ]

Design notes
------------
* `InEKFState` is a `NamedTuple` → valid JAX pytree, clean `lax.scan` carry.
* **`N` is NOT stored** — it is implicit in `d.shape[0]`.  Storing it would force
  static-int handling and provoke recompiles (CLAUDE.md §1.3).
* All `N` contact candidates are kept in the state *permanently* (CoCo): a
  candidate not in contact is expressed through a large contact covariance, never
  a shape change.  This keeps the computation graph constant — required for
  `jit` + `scan` + BPTT (§1.1, invariant 5).
* `d` is one `(N, 3)` array, not a Python list, so every per-contact op is a
  single `vmap` / broadcast.
* The dense matrix is built on demand (`InEKFState.as_matrix`); only
  `(R, v, p, d)` are stored (§1.3).
* `InEKFParams` carries the run-fixed config **and the precomputed constants
  ``Φ`` and ``H``** (invariant 6): they are never rebuilt inside the scan body —
  in particular `expm` is never called in the loop (§3.2).
"""
from typing import NamedTuple

from jax import Array
import jax.numpy as jnp

from .group import skew


class InEKFState(NamedTuple):
    """Sufficient statistic for the contact-aided InEKF.

    Attributes
    ----------
    R : Array, shape (3, 3)
        Base orientation ``{}^{W}R_{B}`` (world ← body).
    v : Array, shape (3,)
        Base linear velocity ``{}^{W}v_{B}`` in world.
    p : Array, shape (3,)
        Base position ``{}^{W}p_{WB}`` in world.
    d : Array, shape (N, 3)
        Stacked contact-candidate world positions ``{}^{W}p_{WC_i}`` — the
        ``vmap`` axis is axis 0.
    P : Array, shape (3N+9, 3N+9)
        Right-invariant error covariance over
        ``ξ = [ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; … ; ξ_{d_N}]``.
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

    @property
    def as_matrix(self) -> Array:
        """Dense ``(N+5, N+5)`` group element ``X`` built from `(R, v, p, d)`.

        Built on demand for the Lie ops that need the dense form (Adjoint,
        innovation mapping); the state itself stores only the compact parts
        (§1.3).
        """
        N = self.N
        X = jnp.eye(N + 5)
        X = X.at[0:3, 0:3].set(self.R)
        X = X.at[0:3, 3].set(self.v)
        X = X.at[0:3, 4].set(self.p)
        X = X.at[0:3, 5:].set(self.d.T)   # d_i as columns 5 … N+4
        return X


# ---------------------------------------------------------------------------
# Precomputed constants:  Φ (transition)  and  H (FK observation)
# ---------------------------------------------------------------------------

def build_Phi(g: Array, dt: float, N: int) -> Array:
    r"""Constant right-invariant transition ``Φ = expm(A^r dt)`` (§3.2).

    With no bias in the state the error dynamics matrix ``A^r`` is constant and
    nilpotent (``(A^r)³ = 0``), so ``Φ`` is the exact closed form

        ┌ I            0      0   0 ┐   (R)
        │ (g)_× dt     I      0   0 │   (v)
        │ ½(g)_× dt²   I dt   I   0 │   (p)
        └ 0            0      0   I ┘   (d, all identity — contacts uncoupled)

    independent of ``R̄, v̄, p̄`` and of the IMU input.  Built directly from the
    closed form (cheaper and exact); `expm` is never used in the scan body.

    Parameters
    ----------
    g : Array, shape (3,)
        Gravity acceleration vector in world.
    dt : float
        Filter timestep.
    N : int
        Number of contact candidates (static).

    Returns
    -------
    Array, shape (3N+9, 3N+9)
    """
    G = skew(g)
    Phi = jnp.eye(3 * N + 9)
    Phi = Phi.at[3:6, 0:3].set(G * dt)               # [v, R]
    Phi = Phi.at[6:9, 3:6].set(jnp.eye(3) * dt)      # [p, v]
    Phi = Phi.at[6:9, 0:3].set(0.5 * G * dt * dt)    # [p, R]
    return Phi


def build_H(N: int) -> Array:
    r"""Constant FK observation matrix ``H`` (§4.1), shape ``(3N, 3N+9)``.

    Each contact's right-invariant FK observation has Jacobian
    ``H_i = [ 0  0  −I  …  +I(col d_i)  … ]`` — ``−I`` in the ``p`` block and
    ``+I`` in its own ``d_i`` block.  Stacked over contacts this is

        H = [ 0_{3N×3} | 0_{3N×3} | (−I_3 ×N) | I_{3N} ]

    i.e. the ``d`` columns form a plain identity (contact ``i`` selects ``d_i``).
    State-independent by construction (world-centric + right-invariant, §0).

    Parameters
    ----------
    N : int
        Number of contact candidates (static).

    Returns
    -------
    Array, shape (3N, 3N+9)
    """
    H = jnp.zeros((3 * N, 3 * N + 9))
    H = H.at[:, 6:9].set(jnp.tile(-jnp.eye(3), (N, 1)))   # p block: −I per contact
    H = H.at[:, 9:9 + 3 * N].set(jnp.eye(3 * N))          # d block: identity
    return H


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class InEKFParams(NamedTuple):
    """Run-fixed configuration plus the precomputed constants ``Φ`` and ``H``.

    Passed into the filter step rather than stored in the mutable state.  ``Φ``
    and ``H`` are precomputed here (invariant 6) so the scan body never rebuilds
    them — in particular `expm` is never called in the loop.

    Attributes
    ----------
    g : Array, shape (3,)
        Gravity acceleration vector in world [m/s²], e.g. ``[0, 0, -9.81]``.
    dt : float
        Filter timestep [s].
    sigma_gyro : float
        Gyro continuous noise density [rad/s/√Hz].  Builds the isotropic gyro
        block ``Q_g = sigma_gyro² I₃`` of the continuous error density (§3.3).
    sigma_accel : float
        Accelerometer continuous noise density [m/s²/√Hz].  Builds
        ``Q_a = sigma_accel² I₃``.
    contact_floor : float
        Variance floor [m²] applied to the per-contact covariances before they
        enter the ``Q̄_d`` contact block (the digest/clamp of §5).
    Phi : Array, shape (3N+9, 3N+9)
        Precomputed constant transition (`build_Phi`).
    H : Array, shape (3N, 3N+9)
        Precomputed constant FK observation (`build_H`).
    """
    g: Array            # (3,) gravity accel [m/s²], world
    dt: float           # [s]
    sigma_gyro: float   # [rad/s/√Hz]  → Q_g = σ_g² I₃
    sigma_accel: float  # [m/s²/√Hz]   → Q_a = σ_a² I₃
    contact_floor: float  # [m²] variance floor on contact covariances
    Phi: Array          # (3N+9, 3N+9) precomputed transition
    H: Array            # (3N, 3N+9) precomputed FK observation


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_state(
    N: int,
    R0: Array | None = None,
    v0: Array | None = None,
    p0: Array | None = None,
    d0: Array | None = None,
    p_R: float = 1e-2,
    p_v: float = 1e-1,
    p_p: float = 1e-2,
    p_d: float = 1.0,
) -> InEKFState:
    """Construct an initial `InEKFState` with a diagonal prior covariance.

    Parameters
    ----------
    N : int
        Number of contact candidates (static).
    R0, v0, p0 : Array, optional
        Initial base orientation (3, 3), velocity (3,), position (3,).  Default
        to identity / zeros.
    d0 : Array, shape (N, 3), optional
        Initial contact positions.  Defaults to zeros.
    p_R, p_v, p_p, p_d : float
        Diagonal prior variances on the orientation / velocity / position /
        per-contact error blocks.  Contacts default to a diffuse ``p_d`` because
        a candidate not yet in firm contact is "off" via a large covariance
        (CoCo, §1.1).  These are untuned starting points, not tuned constants.

    Returns
    -------
    InEKFState
    """
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
    dt: float = 1e-3,
    g: Array | None = None,
    sigma_gyro: float = 1e-3,
    sigma_accel: float = 1e-2,
    contact_floor: float = 1e-4,
) -> InEKFParams:
    """Sensible default `InEKFParams` for an ``N``-contact filter at 1 kHz.

    Precomputes ``Φ`` and ``H`` from the closed forms.  The noise densities are
    starting-point values (untuned); match them to the IMU spec and the
    ContactNet covariance scale once the baseline is consistent.

    Parameters
    ----------
    N : int
        Number of contact candidates (static — fixes the constant graph).
    dt : float
        Filter timestep [s].  Default matches the IHMC 1 kHz control loop.
    g : Array, shape (3,), optional
        Gravity acceleration vector.  Defaults to ``[0, 0, -9.81]``.
    sigma_gyro, sigma_accel : float
        Isotropic IMU noise densities (see `InEKFParams`).
    contact_floor : float
        Variance floor on the contact covariances [m²].
    """
    g = jnp.array([0.0, 0.0, -9.81]) if g is None else g
    return InEKFParams(
        g=g,
        dt=dt,
        sigma_gyro=sigma_gyro,
        sigma_accel=sigma_accel,
        contact_floor=contact_floor,
        Phi=build_Phi(g, dt, N),
        H=build_H(N),
    )
