"""1:1 port of ``InvariantStateTest.java`` (TEST_SUITE_MAP.md §invariant_estimator
core tests) onto `inEKF.state.InEKFState`.

Java constants preserved verbatim: ``EPSILON = 1.0e-12``, ``ITERATIONS = 500``,
per-test seeds 1234 / 2345 / 3456 / 4567.  Java's mutating setters become
functional `_replace`-style returns (the port is a JAX pytree, CLAUDE.md I10) —
the round-trip semantics under test are unchanged.

Coverage (7 tests, one per Java ``@Test``):
  testConstructorIdentityAndSizes, testRotationRoundTrip,
  testBaseVelocityAndPositionRoundTrip, testNamedComponentsAreIndependent,
  testContactIndexOutOfBounds, testTangentIndices, testSetToIdentity.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import state as s

from ._oracles import next_rotation_matrix, next_vector3d

EPSILON = 1.0e-12
ITERATIONS = 500


# ---------------------------------------------------------------------------
# testConstructorIdentityAndSizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [0, 1, 2, 4])
def test_constructor_identity_and_sizes(N):
    st = s.InEKFState.identity(N)

    assert st.N == N
    assert st.group_size == 5 + N
    assert st.tangent_size == 9 + 3 * N

    # X = I_{(5+N)}, P = 0 — covariance starts at zeros, not identity.
    assert jnp.allclose(st.as_matrix, jnp.eye(5 + N), atol=EPSILON)
    assert jnp.allclose(st.P, jnp.zeros((9 + 3 * N, 9 + 3 * N)), atol=EPSILON)

    # I8: float64 at the filter boundary.
    assert st.as_matrix.dtype == jnp.float64
    assert st.P.dtype == jnp.float64


# ---------------------------------------------------------------------------
# testRotationRoundTrip
# ---------------------------------------------------------------------------

def test_rotation_round_trip():
    rng = np.random.default_rng(1234)
    st = s.InEKFState.identity(2)
    for _ in range(ITERATIONS):
        expected = jnp.asarray(next_rotation_matrix(rng))
        st = st._replace(R=expected)
        assert jnp.allclose(st.R, expected, atol=EPSILON)
        # …and it survives the trip through the dense group element.
        assert jnp.allclose(st.as_matrix[0:3, 0:3], expected, atol=EPSILON)


# ---------------------------------------------------------------------------
# testBaseVelocityAndPositionRoundTrip
# ---------------------------------------------------------------------------

def test_base_velocity_and_position_round_trip():
    rng = np.random.default_rng(2345)
    st = s.InEKFState.identity(2)
    for _ in range(ITERATIONS):
        v = jnp.asarray(next_vector3d(rng)[0])
        p = jnp.asarray(next_vector3d(rng)[0])
        st = st._replace(v=v, p=p)
        assert jnp.allclose(st.v, v, atol=EPSILON)
        assert jnp.allclose(st.p, p, atol=EPSILON)
        # Column 3 is velocity, column 4 is position (map: "confirm which is which").
        X = st.as_matrix
        assert jnp.allclose(X[0:3, 3], v, atol=EPSILON)
        assert jnp.allclose(X[0:3, 4], p, atol=EPSILON)


# ---------------------------------------------------------------------------
# testNamedComponentsAreIndependent
# ---------------------------------------------------------------------------

def test_named_components_are_independent():
    rng = np.random.default_rng(3456)
    N = 3
    st = s.InEKFState.identity(N)

    R = jnp.asarray(next_rotation_matrix(rng))
    v = jnp.asarray(next_vector3d(rng)[0])
    p = jnp.asarray(next_vector3d(rng)[0])
    contacts = [jnp.asarray(next_vector3d(rng)[0]) for _ in range(N)]

    st = st._replace(R=R, v=v, p=p)
    for i, d_i in enumerate(contacts):
        st = st.set_contact_position(i, d_i)

    # No aliasing: every named slot reads back what was written into it.
    assert jnp.allclose(st.R, R, atol=EPSILON)
    assert jnp.allclose(st.v, v, atol=EPSILON)
    assert jnp.allclose(st.p, p, atol=EPSILON)
    for i, d_i in enumerate(contacts):
        assert jnp.allclose(st.get_contact_position(i), d_i, atol=EPSILON)

    # Contact i occupies column 5+i of X.
    X = st.as_matrix
    for i, d_i in enumerate(contacts):
        assert jnp.allclose(X[0:3, 5 + i], d_i, atol=EPSILON)


# ---------------------------------------------------------------------------
# testContactIndexOutOfBounds
# ---------------------------------------------------------------------------

def test_contact_index_out_of_bounds():
    st = s.InEKFState.identity(2)          # valid indices: 0, 1
    d = jnp.zeros(3)

    with pytest.raises(IndexError):
        st.get_contact_position(-1)
    with pytest.raises(IndexError):
        st.get_contact_position(2)
    with pytest.raises(IndexError):
        st.set_contact_position(2, d)
    with pytest.raises(IndexError):
        st.contact_tangent_index(2)


# ---------------------------------------------------------------------------
# testTangentIndices  (I4 — load-bearing for every Jacobian/covariance index)
# ---------------------------------------------------------------------------

def test_tangent_indices():
    st = s.InEKFState.identity(3)

    assert s.ROTATION_TANGENT_INDEX == 0
    assert s.BASE_VELOCITY_TANGENT_INDEX == 3
    assert s.BASE_POSITION_TANGENT_INDEX == 6
    assert st.contact_tangent_index(0) == 9
    assert st.contact_tangent_index(1) == 12
    assert st.contact_tangent_index(2) == 15


# ---------------------------------------------------------------------------
# testSetToIdentity
# ---------------------------------------------------------------------------

def test_set_to_identity():
    rng = np.random.default_rng(4567)
    st = s.InEKFState.identity(2)
    st = st._replace(
        R=jnp.asarray(next_rotation_matrix(rng)),
        v=jnp.asarray(next_vector3d(rng)[0]),
        p=jnp.asarray(next_vector3d(rng)[0]),
        P=jnp.eye(15),                      # must survive untouched
    )
    st = st.set_contact_position(0, jnp.asarray(next_vector3d(rng)[0]))
    st = st.set_contact_position(1, jnp.asarray(next_vector3d(rng)[0]))

    st = st.set_to_identity()

    assert jnp.allclose(st.as_matrix, jnp.eye(7), atol=EPSILON)
    # setToIdentity touches X only — P is left alone.
    assert jnp.allclose(st.P, jnp.eye(15), atol=EPSILON)
