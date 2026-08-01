r"""The `InvariantEKF` orchestrator — **pure wiring**, no math of its own.

Every numerical operation delegates: `predict` to `propagate.propagate`, `update` to
`correct.contact_update`, `gravity_update` to `gravity_update.apply_gravity_leveling`.
That is not incidental tidiness — the ported `InvariantEKFTest` asserts the
orchestrator reproduces the standalone propagator and updater **bit-for-bit**
(tol 1e-12), which is only guaranteed if there is exactly one implementation of each
step.

Java's `InvariantEKF` is a mutable object holding `(X, P)` and a bag of collaborators.
The port splits that in two: `InvariantEKF` is the *immutable wiring* (contact count,
filter params, contact noise), built once by `create` and closed over by the jitted
step; `InEKFState` is the *carry* `(R, v, p, d, P)`, threaded explicitly through every
call (I10).  So `ekf.predict(av, la, dt)` becomes `predict(ekf, state, av, la)`
returning a new state, and Java's introspection getters become the returned
`UpdateDiagnostics` pytree rather than fields mutated on the side.
"""
from typing import NamedTuple, Sequence

from jax import Array
import jax.numpy as jnp

from ..config import section
from .correct import UpdateDiagnostics, contact_update, no_update_diagnostics
from .gravity_update import (
    GravityParams,
    GravityRef,
    apply_gravity_leveling,
    assemble_gravity_leveling,
    default_gravity_params,
)
from .propagate import propagate
from .state import InEKFParams, InEKFState, default_params


class InvariantEKF(NamedTuple):
    """Immutable filter wiring — Java `InvariantEKF.create(...)`.

    ``N`` is the contact-candidate count, static for the filter's lifetime (I2);
    ``params`` the propagation/correction constants including the precomputed ``Φ``
    and ``H``; ``sigma_c`` ``(N, 3, 3)`` the per-contact **body-frame** process
    covariances used by the propagation; ``gravity_params`` the gravity-leveling
    configuration.

    ``reseed`` is the touchdown re-seed configuration, or ``None`` (the default and
    every shipped path) to leave the machinery out of the traced graph entirely.  See
    `inEKF/reseed.py`; enabling it also requires driving `InEKFInputs.contact_prob`.
    """
    N: int
    params: InEKFParams
    sigma_c: Array
    gravity_params: GravityParams
    reseed: "ReseedParams | None" = None

    @property
    def number_of_contacts(self) -> int:
        """Java `getNumberOfContacts()`."""
        return self.N

    @property
    def group_size(self) -> int:
        """Java `getState().getGroupSize()` — ``5 + N``."""
        return self.N + 5

    @property
    def tangent_size(self) -> int:
        """Java `getTangentSize()` — ``9 + 3N``."""
        return 3 * self.N + 9


def create(
    number_of_contacts: int,
    gyro_var: float | None = None,
    accel_var: float | None = None,
    contact_var: float | None = None,
    dt: float | None = None,
    gravity_params: GravityParams | None = None,
    reseed: "ReseedParams | None" = None,
) -> InvariantEKF:
    """Java `InvariantEKF.create(numberOfContacts, gyroVar, accelVar, contactVar)`.

    Every noise argument is a **variance** and defaults to the ``inekf`` section
    of ``config/filter_cfg.yaml``.  The contact updater is wired by construction
    — there is no "forgot to install the ContactUpdater" state to fall into,
    which is why the Java `IllegalStateException` has no port analogue.
    """
    if number_of_contacts < 0:
        raise ValueError(f"number of contacts must be >= 0, got {number_of_contacts}")

    cfg = section("inekf")
    contact_var = cfg["contact_var"] if contact_var is None else contact_var
    params = default_params(
        number_of_contacts, dt=dt, gyro_var=gyro_var, accel_var=accel_var,
    )
    sigma_c = jnp.tile(contact_var * jnp.eye(3), (number_of_contacts, 1, 1))
    return InvariantEKF(
        N=number_of_contacts,
        params=params,
        sigma_c=sigma_c,
        gravity_params=default_gravity_params() if gravity_params is None
        else gravity_params,
        reseed=reseed,
    )


def initialize(
    ekf: InvariantEKF,
    rotation: Array,
    velocity: Array,
    position: Array,
    contacts: Sequence[Array] | Array,
    covariance: Array | None = None,
) -> InEKFState:
    """Java `initialize(rotation, velocity, position, contacts[], covariance)`.

    Validates the two shape contracts the Java version throws on: the contact array
    must have exactly ``N`` entries, and the covariance must be ``m x m`` with
    ``m = 9 + 3N``.  Java raises `IllegalArgumentException`; the port raises
    `ValueError`.
    """
    contacts = jnp.asarray(contacts, dtype=float).reshape(-1, 3) if len(contacts) \
        else jnp.zeros((0, 3))
    if contacts.shape[0] != ekf.N:
        raise ValueError(
            f"expected {ekf.N} contact positions, got {contacts.shape[0]}"
        )

    m = ekf.tangent_size
    if covariance is None:
        covariance = section("inekf")["initial_covariance"] * jnp.eye(m)
    covariance = jnp.asarray(covariance, dtype=float)
    if covariance.shape != (m, m):
        raise ValueError(
            f"covariance must be {m}x{m} for {ekf.N} contacts, got {covariance.shape}"
        )

    return InEKFState(
        R=jnp.asarray(rotation, dtype=float),
        v=jnp.asarray(velocity, dtype=float),
        p=jnp.asarray(position, dtype=float),
        d=contacts,
        P=covariance,
    )


def initialize_from_state(
    ekf: InvariantEKF, source: InEKFState, covariance: Array | None = None
) -> InEKFState:
    """Java `initializeFromState` — re-seed from an existing state's components."""
    return initialize(ekf, source.R, source.v, source.p, source.d, covariance)


def predict(
    ekf: InvariantEKF,
    state: InEKFState,
    angular_velocity: Array,
    linear_acceleration: Array,
) -> InEKFState:
    """Java `predict(angularVelocity, linearAcceleration, dt)` — pure delegation.

    ``dt`` lives in ``ekf.params`` (it is baked into the precomputed ``Φ``, so it
    cannot be a per-call argument without rebuilding the constant).
    ``linear_acceleration`` is the IMU **specific force**; gravity is added
    inside the propagation.
    """
    return propagate(
        state, angular_velocity, linear_acceleration, ekf.sigma_c, ekf.params
    )


def update(
    ekf: InvariantEKF,
    state: InEKFState,
    contact_index: int,
    measurement: Array,
    body_covariance: Array,
) -> tuple[InEKFState, UpdateDiagnostics]:
    """Java `update(contactIndex, measurement, bodyCovariance)` — the high-level 3-arg
    contact update, delegating to `correct.contact_update` with ``learned=False``.
    """
    updated, _, diagnostics = contact_update(
        state, contact_index, measurement, body_covariance, learned=False
    )
    return updated, diagnostics


def gravity_leveling_update(
    ekf: InvariantEKF,
    state: InEKFState,
    ref: GravityRef,
    specific_force: Array,
    pitch_observable: bool | Array = True,
    gate: Array | float = 1.0,
) -> tuple[InEKFState, GravityRef, UpdateDiagnostics]:
    """Java `assembleGravityLeveling(...)` + `applyGravityLeveling()`, fused — returns
    the corrected state, the advanced gravity reference, and the update diagnostics.
    """
    meas = assemble_gravity_leveling(
        ref, state, specific_force, ekf.gravity_params, pitch_observable
    )
    updated, diagnostics = apply_gravity_leveling(
        state, meas, ekf.gravity_params, gate=gate
    )
    return updated, meas.ref, diagnostics


def initial_diagnostics() -> UpdateDiagnostics:
    """Diagnostics before any update — NIS is NaN (Java parity)."""
    return no_update_diagnostics()
