"""The contact measurement-noise floor must actually reach the replayed filter.

`contact_meas_var` (main_estimator "landmine #2") is applied at the joint-KF -> InEKF
boundary: `_boundary` sets ``sigma_q_eff = sigma_q + contact_meas_var * I`` and writes
it into `InEKFInputs.joint.sigma_q` (main_estimator.py:697,713). That boundary runs
inside `make_fused_step`, i.e. during COLLECTION.

ContactNet trains and validates by replaying recorded `InEKFInputs` through
`inEKF.filter.make_step`, which consumes `inputs.joint.sigma_q` directly and never
calls `_boundary`. So handing `contact_meas_var` to `build_collector` at training time
sets `fused.contact_meas_var` and changes nothing whatsoever -- the value that matters
was frozen into the recorded `sigma_q` when the pool was collected.

That defect shipped and produced a held-out analytic baseline BIT-IDENTICAL across a
0.0 -> 1e-4 change (vel_rmse 0.0541, NEES 4.65, NIS/dof 0.041 both times), which is the
signature to remember: a filter parameter that moves nothing is not conservative, it is
disconnected. `dataset.apply_contact_meas_floor` closes it on the replay path; these
tests pin both that it is arithmetically the right transformation and -- the part a
value test alone would miss -- that it is LIVE.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.contactnet import dataset
from invariant_estimation.inEKF.filter import contact_position_noise

N_JOINTS = 12
N_CONTACTS = 8


def _sigma_q(seed=0):
    """A plausible coupled Sigma_q: SPD, off-diagonals that matter to J Sigma_q J^T."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((N_JOINTS, N_JOINTS))
    return m @ m.T + N_JOINTS * np.eye(N_JOINTS)


class _FakeJoint:
    """Minimal stand-in exposing the one field the transform touches."""

    def __init__(self, sigma_q):
        self.sigma_q = sigma_q

    def _replace(self, **kw):
        return _FakeJoint(kw.get("sigma_q", self.sigma_q))


class _FakeInputs:
    def __init__(self, joint):
        self.joint = joint

    def _replace(self, **kw):
        return _FakeInputs(kw.get("joint", self.joint))


def _prep(sigma_q):
    import dataclasses
    f = {fld.name: None for fld in dataclasses.fields(dataset.PreparedRollout)}
    f["inputs"] = _FakeInputs(_FakeJoint(sigma_q))
    return dataset.PreparedRollout(**f)


def test_floor_is_the_boundary_transformation():
    """Must reproduce what collecting at the target floor would have produced.

    Collecting at `recorded` gives `raw + recorded*I`; we want `raw + target*I`; the
    exact difference is `(target - recorded)*I`. Composition, not approximation --
    which is what makes re-flooring recorded data equivalent to re-collecting, given
    that the collection policy reads ground truth so the trajectory does not depend
    on the filter configuration at all.
    """
    raw = _sigma_q()
    recorded, target = 0.0, 1.0e-4
    as_collected = raw + recorded * np.eye(N_JOINTS)
    want = raw + target * np.eye(N_JOINTS)

    got = dataset.apply_contact_meas_floor(
        [_prep(as_collected)], target, recorded)[0].inputs.joint.sigma_q

    assert np.allclose(got, want, rtol=0, atol=1e-18)
    # off-diagonals are load-bearing (J Sigma_q J^T needs the coupling) and a floor
    # must not touch them
    off = ~np.eye(N_JOINTS, dtype=bool)
    assert np.array_equal(got[off], as_collected[off])


def test_delta_form_is_not_a_double_add():
    """Re-flooring an already-floored pool must be a no-op, not another +cmv*I."""
    raw = _sigma_q(1)
    once = dataset.apply_contact_meas_floor([_prep(raw)], 1e-4, 0.0)
    twice = dataset.apply_contact_meas_floor(once, 1e-4, 1e-4)
    assert np.array_equal(twice[0].inputs.joint.sigma_q,
                          once[0].inputs.joint.sigma_q)


def test_matching_floor_leaves_inputs_untouched():
    """pool floor == training floor must return the inputs unchanged (identity)."""
    raw = _sigma_q(2)
    p = _prep(raw)
    out = dataset.apply_contact_meas_floor([p], 1e-4, 1e-4)[0]
    assert out.inputs.joint.sigma_q is raw


def test_the_floor_is_live_in_the_contact_noise():
    """The floor must CHANGE the noise the contact update sees.

    This is the test the original defect needed. `contact_position_noise` is the sole
    consumer of `inputs.joint.sigma_q` on the replay path (`filter.py:264`), so if the
    floor does not move `N^p` it cannot move anything downstream -- which is exactly
    what "the analytic baseline was bit-identical" meant.

    Asserted as a RELATIVE change so it cannot pass on numerical dust.
    """
    rng = np.random.default_rng(3)
    J = jnp.asarray(rng.standard_normal((N_CONTACTS, 3, N_JOINTS)))
    raw = _sigma_q(3)

    base = contact_position_noise(J, jnp.asarray(raw))
    floored = contact_position_noise(
        J, jnp.asarray(dataset.apply_contact_meas_floor(
            [_prep(raw)], 1.0e-4, 0.0)[0].inputs.joint.sigma_q))

    rel = float(jnp.max(jnp.abs(floored - base)) / jnp.max(jnp.abs(base)))
    assert rel > 1e-9, (
        f"contact_meas_var=1e-4 moved the contact noise by only {rel:.2e} relative -- "
        f"the floor is not reaching the filter, which is the landmine-#2 defect")
    # and it must INFLATE: J (Sigma + cI) J^T - J Sigma J^T = c J J^T, PSD.
    d = np.asarray(floored - base)
    for i in range(N_CONTACTS):
        w = np.linalg.eigvalsh(0.5 * (d[i] + d[i].T))
        assert w.min() > -1e-12, (
            f"contact {i}: the floor made the contact noise less positive definite; "
            f"it must add c*J J^T, which is PSD")
