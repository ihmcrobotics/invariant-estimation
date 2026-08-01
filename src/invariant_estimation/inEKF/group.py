r"""Matrix Lie-group operations on ``SE_{N+2}(3)``: exp / log / Adjoint and the Γ closed forms.

Tangent vectors are **rotation-first**,
``ξ = [ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; … ; ξ_{d_N}]`` (I4) — never permute; ``P``, ``Φ``
and ``H`` are all laid out in this order.  The group element itself is documented
in `state.py`.

The ``θ → 0`` singularities of the SO(3) closed forms use the **double-where**
trick so ``jax.grad`` stays finite at ``θ = 0``.  ``jaxlie.SO3`` supplies only the
rotation logarithm in `log_SEn3`; the ``Γ_0`` here is cross-checked against it in
the tests.

With ``θ = ‖φ‖``::

  Γ_0 = I + (sinθ/θ)(φ)_× + ((1−cosθ)/θ²)(φ)_×²        (SO(3) exp, Rodrigues)
  Γ_1 = I + ((1−cosθ)/θ²)(φ)_× + ((θ−sinθ)/θ³)(φ)_×²   (left Jacobian, the V matrix)
  Γ_2 = ½I + ((θ−sinθ)/θ³)(φ)_× + ((θ²+2cosθ−2)/2θ⁴)(φ)_×²

``Γ_1`` is applied to every translational tangent component; ``Γ_2`` is the second
integral term used by the exact mean propagation (§3.1).
"""
from jax import Array, vmap
import jax.numpy as jnp
from jaxlie import SO3

from ..config import section

# Below this value of θ² = ‖φ‖² the SO(3) coefficient closed forms suffer
# catastrophic cancellation, so we switch to their truncated Taylor series.
# θ ≈ 1e-4; the series (kept to θ⁴) is then accurate to ~1e-24, far past float64.
# Tunable: `numerics.small_angle_eps` in config/filter_cfg.yaml.
_EPS = section("numerics")["small_angle_eps"]


def skew(phi: Array) -> Array:
    r"""``(3,)`` → ``(3,3)`` skew-symmetric matrix ``(φ)_×`` with ``(φ)_× a = φ × a``."""
    x, y, z = phi[0], phi[1], phi[2]
    zero = jnp.zeros_like(x)
    return jnp.array([
        [zero, -z, y],
        [z, zero, -x],
        [-y, x, zero],
    ])


def _theta_safe(phi: Array) -> tuple[Array, Array, Array, Array]:
    """Return ``(θ, θ²_safe, θ²_raw, is_small)`` for the double-where trick.

    Two squared angles because the two branches need different ones: ``θ²_safe``
    (clamped to 1.0 when small) feeds the **analytic** coefficients — do not feed
    them the raw ``φ·φ``, because dividing by it back-propagates ``0 * NaN = NaN``
    through the `jnp.where` even though the value is discarded.  ``θ²_raw`` feeds
    the **truncated series**, so the small-angle value and its gradient are correct
    at ``θ = 0``.
    """
    theta2_raw = phi @ phi
    is_small = theta2_raw < _EPS
    theta2_safe = jnp.where(is_small, 1.0, theta2_raw)
    theta = jnp.sqrt(theta2_safe)
    return theta, theta2_safe, theta2_raw, is_small


def Gamma0(phi: Array) -> Array:
    r"""``SO(3)`` exponential ``Γ_0(φ) = exp((φ)_×)`` (Rodrigues), ``(3,)`` → ``(3,3)``."""
    theta, theta2, theta2_raw, is_small = _theta_safe(phi)
    K = skew(phi)
    KK = K @ K

    # a1 = sinθ/θ ,  a2 = (1−cosθ)/θ²
    a1 = jnp.where(is_small, 1.0 - theta2_raw / 6.0 + theta2_raw**2 / 120.0,
                   jnp.sin(theta) / theta)
    a2 = jnp.where(is_small, 0.5 - theta2_raw / 24.0 + theta2_raw**2 / 720.0,
                   (1.0 - jnp.cos(theta)) / theta2)
    return jnp.eye(3) + a1 * K + a2 * KK


def Gamma1(phi: Array) -> Array:
    r"""Left Jacobian of ``SO(3)``, ``Γ_1(φ)`` (the ``V`` matrix); ``→ I`` as ``θ → 0``."""
    theta, theta2, theta2_raw, is_small = _theta_safe(phi)
    K = skew(phi)
    KK = K @ K

    # b1 = (1−cosθ)/θ² ,  b2 = (θ−sinθ)/θ³
    b1 = jnp.where(is_small, 0.5 - theta2_raw / 24.0 + theta2_raw**2 / 720.0,
                   (1.0 - jnp.cos(theta)) / theta2)
    b2 = jnp.where(is_small, 1.0 / 6.0 - theta2_raw / 120.0 + theta2_raw**2 / 5040.0,
                   (theta - jnp.sin(theta)) / theta**3)
    return jnp.eye(3) + b1 * K + b2 * KK


def Gamma2(phi: Array) -> Array:
    r"""Second integral term ``Γ_2(φ)`` (exact-mean position integration, §3.1); ``→ ½I`` as ``θ → 0``."""
    theta, theta2, theta2_raw, is_small = _theta_safe(phi)
    K = skew(phi)
    KK = K @ K

    # c1 = (θ−sinθ)/θ³ ,  c2 = (θ²+2cosθ−2)/(2θ⁴)
    c1 = jnp.where(is_small, 1.0 / 6.0 - theta2_raw / 120.0 + theta2_raw**2 / 5040.0,
                   (theta - jnp.sin(theta)) / theta**3)
    c2 = jnp.where(is_small, 1.0 / 24.0 - theta2_raw / 720.0 + theta2_raw**2 / 40320.0,
                   (theta2 + 2.0 * jnp.cos(theta) - 2.0) / (2.0 * theta2**2))
    return 0.5 * jnp.eye(3) + c1 * K + c2 * KK


def exp_SEk3(xi: Array) -> Array:
    r"""``(3+3k,)`` rotation-first tangent → ``(3+k, 3+k)`` group element (Java ``SEK3_Utils.exp``).

    ``Γ_0(φ)`` in the rotation block, ``Γ_1(φ)`` applied to each of the ``k``
    translational components.  `exp_SEn3` is the filter-facing alias at ``k = N+2``.
    ``ValueError`` if ``len(ξ)`` is not ``3 + 3k`` for an integer ``k ≥ 1``.
    """
    m = xi.shape[0]
    if m < 6 or (m - 3) % 3 != 0:
        raise ValueError(
            f"tangent length {m} is not 3 + 3k for an integer k >= 1"
        )
    k = (m - 3) // 3

    phi = xi[:3]
    R = Gamma0(phi)
    J = Gamma1(phi)

    rest = xi[3:].reshape(k, 3)             # (k, 3): the translational components
    cols = J @ rest.T                        # (3, k): Γ_1 applied to each

    X = jnp.eye(k + 3)
    X = X.at[0:3, 0:3].set(R)
    X = X.at[0:3, 3:].set(cols)
    return X


def exp_SEn3(xi: Array, N: int) -> Array:
    r"""``(3N+9,)`` tangent → ``(N+5, N+5)`` element of ``SE_{N+2}(3)``; ``ValueError`` on a length mismatch."""
    if xi.shape[0] != 3 * N + 9:
        raise ValueError(
            f"tangent length {xi.shape[0]} does not match 3N+9 = {3 * N + 9} "
            f"for N = {N} contacts"
        )
    return exp_SEk3(xi)


def log_SEn3(X: Array) -> Array:
    r"""``(N+5, N+5)`` element → ``(3N+9,)`` tangent; inverse of `exp_SEn3`, ``N`` static from the shape.

    Rotation logarithm via ``jaxlie.SO3``; the translational components come from
    ``Γ_1(φ)^{-1}`` applied to the columns by a linear solve (no explicit inverse).

    ``ValueError`` if ``X`` is not square or is smaller than ``4x4`` (``k < 1``).
    This is the port's form of the Java size-consistency guard between ``n = 3+k``
    and the tangent length ``3+3k``: Java packs into a caller-supplied output array
    and rejects a mismatched one, whereas this function *returns* the tangent, so
    only the input can be inconsistent.
    """
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"group element must be a square matrix, got {X.shape}")
    if X.shape[0] < 4:
        raise ValueError(
            f"group element of size {X.shape[0]} is smaller than the minimum "
            f"4x4 (k = 1) SE_k(3) element"
        )

    R = X[0:3, 0:3]
    phi = SO3.from_matrix(R).log()          # (3,)
    J = Gamma1(phi)

    cols = X[0:3, 3:]                        # (3, N+2): translational columns
    rest = jnp.linalg.solve(J, cols)        # (3, N+2): Γ_1^{-1} @ each column
    return jnp.concatenate([phi, rest.T.reshape(-1)])


def Adjoint(X: Array) -> Array:
    r"""``(N+5, N+5)`` element → its adjoint ``Ad_X``, shape ``(3N+9, 3N+9)``.

    Block order ``[R, v, p, d_1, …, d_N]``, each block ``3x3``: ``R`` on every
    diagonal block and ``(t_k)_× R`` in the first block-column for each
    translational component ``t_k ∈ {v, p, d_i}``::

            ┌ R         0   0   …  0 ┐
            │ (v)_× R   R   0   …  0 │
      Ad_X =│ (p)_× R   0   R   …  0 │
            │ (d_1)_×R  0   0   …  0 │
            │ ⋮                ⋱    │
            └ (d_N)_×R  0   0   …  R ┘
    """
    N = X.shape[0] - 5
    nblocks = N + 3                          # R, v, p, d_1 … d_N
    R = X[0:3, 0:3]

    # R on every diagonal block.
    Ad = jnp.kron(jnp.eye(nblocks), R)

    # First block-column coupling: (t_k)_× R for each translational vector.
    translations = X[0:3, 3:].T             # (N+2, 3): v, p, d_1, …, d_N
    coupling = vmap(lambda t: skew(t) @ R)(translations)   # (N+2, 3, 3)
    Ad = Ad.at[3:, 0:3].set(coupling.reshape(-1, 3))
    return Ad
