r"""
End-to-end wiring test: CaptureManifest (item 3) -> noise/two_stage/optimize (item 2).

What this test is, and is not
------------------------------
Before this file, `captures.py` and `noise.py`/`two_stage.py`/`optimize.py` were two islands: no
code anywhere constructed a `CaptureManifest`, split it, and then actually fed the resulting
train/test session partition into `fit_scalars`. Each half had its own thorough unit tests, but
nothing proved the SEAM between them holds -- that a held-out split survives from the manifest all
the way to "loss computed only on sessions the optimizer never saw."

This is an INTEGRATION test, not an accuracy or real-data claim:
* Sessions are synthetic (`scene()`-style, matching `tests/learning/test_noise.py`'s fixtures), not
  real robot/mocap logs -- no real capture exists yet (robot time starts 2026-09-15), and the
  `ihmclog` decoder `tests/replay/`'s own tests depend on is not installed on this machine.
* `CaptureSession.robot_log`/`mocap_log` Artifacts here point at small placeholder files written to
  a temp directory purely so `Artifact.verify()`'s hash check has something real to check -- their
  BYTES are not what drives the synthetic filter inputs (a per-session integer seed does, read from
  those same files' contents, so the manifest is not just decorative: changing what a session's
  artifact contains changes what trajectory that session trains/evaluates on).
* There is no held-out ACCURACY assertion (this test does not claim the learned scales generalize)
  -- only that the wiring is correct: (a) the optimizer's gradient never sees test-session data, and
  (b) the held-out loss is computed from the SAME manifest-selected partition, not from whatever the
  caller happened to pass in by hand.

The actual real-data loader (reading a real `robot_log`/`mocap_log` Artifact's bytes into
`TwoStageInputs`) is exactly what's still missing before this pipeline can run on a real capture --
see `LEARNED_MATRIX_TIER1.md`'s "item 4 (Java loading), real-data training... remain pending".
"""
from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from invariant_estimation.inEKF import ekf as base_ekf
from invariant_estimation.inEKF import filter as base_filter
from invariant_estimation.inEKF.group import Gamma0
from invariant_estimation.jointKF import filter as joint_filter
from invariant_estimation.jointKF.anchors import AnchorJacobians
from invariant_estimation.jointKF.build import KinematicTree, build_joint_kf
from invariant_estimation.jointKF.state import default_params
from invariant_estimation.learning.captures import (
    Artifact, CaptureManifest, CaptureSession, deterministic_split,
)
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.learning.optimize import body_velocity_l2, fit_scalars
from invariant_estimation.learning.two_stage import TwoStageCarry, TwoStageInputs, run


def _synthetic_session_data(seed: int, ticks: int = 8):
    """One `scene()`-style synthetic (build, params, ekf, kin, carry, inputs) trajectory, varied by
    `seed` -- stands in for "one capture session's worth of sensor data" until a real loader exists.
    """
    tree = KinematicTree(
        joint_names=("j0", "j1", "j2"), joint_body=np.array([1, 2, 3]),
        body_parent=np.array([-1, 0, 1, 2]), joint_dof=np.array([6, 7, 8]),
        base_dofs=np.arange(6), site_body={"base": 0, "mid": 2, "tip": 3, "foot": 3},
        tau_max=np.ones(3) * 10,
    )
    build = build_joint_kf(tree, imu_sites=["base", "mid", "tip"],
                            pairs=[(0, 1), (0, 2)], foot_sites=["foot"])
    build = build._replace(use_mass_matrix=False, gyro_sigma=jnp.stack(
        [jnp.diag(jnp.array([0.003, 0.004, 0.005]) * (i + 1)) for i in range(3)]))
    params = default_params(dt=0.02, sigma_accel=0.3, cond_s_max=1e14)
    ekf = base_ekf.create(1, dt=0.02, gyro_var=0.01, accel_var=0.1)
    rng = np.random.default_rng(seed)
    q = jnp.array(rng.normal(0, 0.1, 3))

    def kin(q, qd):
        return base_filter.ContactFrames(
            jnp.array([[0.0, 0.0, -0.9]]) + 0.1 * q[None],
            0.1 * jnp.eye(3)[None], jnp.zeros((1, 3, 3)))

    state = base_ekf.initialize(
        ekf, Gamma0(jnp.array(rng.normal(0, 0.03, 3))), jnp.array(rng.normal(0, 0.03, 3)),
        jnp.zeros(3), kin(q, q).y, jnp.eye(12) * 0.01)
    joint = joint_filter.init_carry(build, params, q)._replace(trusted_feet=jnp.ones(1))
    carry = TwoStageCarry(joint, base_filter.init_carry(state))

    def tick():
        return TwoStageInputs(
            joint_filter.SensorInputs(
                q + jnp.array(rng.normal(0, 0.01, 3)),
                jnp.array(rng.normal(0, 0.02, (3, 3))), jnp.zeros(0), jnp.ones(1)),
            joint_filter.ModelInputs(
                jnp.stack([jnp.diag(jnp.array([1.0, 1.0, 0.0])), jnp.eye(3)]),
                jnp.tile(jnp.eye(3), (2, 1, 1)),
                AnchorJacobians(jnp.eye(3)[None], jnp.zeros((1, 3, 0)))),
            jnp.array([0.0, 0.0, 9.81]) + jnp.array(rng.normal(0, 0.02, 3)),
            jnp.eye(3)[None] * 0.03,
        )

    inputs = jax.tree.map(lambda *leaves: jnp.stack(leaves), *(tick() for _ in range(ticks)))
    return build, params, ekf, kin, carry, inputs


def _artifact(directory, name: str, content: bytes) -> Artifact:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return Artifact(name, hashlib.sha256(content).hexdigest())


def _write_manifest(tmp_path, seeds: dict[str, int]):
    """Builds a real, hash-verified `CaptureManifest` on disk whose sessions' artifact CONTENT is
    the seed that drives that session's synthetic trajectory -- so this is a genuine manifest->data
    dependency, not a manifest sitting next to unrelated hard-coded fixtures."""
    sessions = []
    for name, seed in seeds.items():
        sessions.append(CaptureSession(
            name, name, "synthetic-walking", "2026-09-15T10:00:00-05:00",
            _artifact(tmp_path, f"{name}/robot.bin", str(seed).encode()),
            _artifact(tmp_path, f"{name}/mocap.bin", str(seed).encode()),
            _artifact(tmp_path, "shared/calibration.json", b"shared-calibration"),
            _artifact(tmp_path, "shared/model.urdf", b"shared-urdf"),
            _artifact(tmp_path, f"{name}/sync.json", b"sync"),
            "aligned_robot_monotonic_ns", "registered_zup", "pelvis",
            0, 1_000_000_000, ("j0", "j1", "j2"), ("base", "mid", "tip"),
        ))
    split = deterministic_split(sessions, n_test=1, seed=7)
    manifest = CaptureManifest(tuple(sessions), split, "2026-09-14T22:00:00Z", "git:synthetic-fixture")
    manifest.save(tmp_path / "manifest.json")
    return CaptureManifest.load(tmp_path / "manifest.json")  # round-trips + verifies hashes for real


def _seed_for_session(manifest, session_id: int | str, tmp_path) -> int:
    """Stand-in for the not-yet-built real loader: reads the session's own artifact bytes back off
    disk (proving the manifest's paths are load-bearing) rather than trusting an in-memory seed map.
    """
    session = next(s for s in manifest.sessions if s.session_id == str(session_id))
    return int((tmp_path / session.robot_log.path).read_bytes().decode())


def test_manifest_driven_split_reaches_the_optimizer_and_holdout_is_never_trained_on(tmp_path):
    seeds = {"0": 10, "1": 11, "2": 12, "3": 13}
    manifest = _write_manifest(tmp_path, seeds)
    train_ids = [s.session_id for s in manifest.partition("train")]
    test_ids = [s.session_id for s in manifest.partition("test")]
    assert set(train_ids) | set(test_ids) == set(seeds)
    assert not (set(train_ids) & set(test_ids))

    train_sessions = [_synthetic_session_data(_seed_for_session(manifest, sid, tmp_path)) for sid in train_ids]
    test_sessions = [_synthetic_session_data(_seed_for_session(manifest, sid, tmp_path)) for sid in test_ids]

    spec = NoiseSpec(train_sessions[0][0].imu_names, arm=7)

    def combined_training_loss(theta):
        """Mean loss over ONLY the training partition -- the optimizer's gradient must never see a
        test-session sample, which is what makes the assertion below meaningful rather than trivial."""
        losses = []
        for build, params, ekf, kin, carry, inputs in train_sessions:
            _, out = run(theta, spec, build, params, ekf, kin, jnp.eye(3), carry, inputs)
            losses.append(body_velocity_l2(
                out.base.state.R, out.base.state.v,
                jnp.tile(jnp.eye(3), (inputs.accel_body.shape[0], 1, 1)),
                jnp.zeros((inputs.accel_body.shape[0], 3)),
            ))
        return jnp.mean(jnp.stack(losses))

    fit = fit_scalars(combined_training_loss, spec.initial_theta(), steps=10, learning_rate=0.05)
    assert np.isfinite(fit.losses).all()

    def holdout_loss(theta):
        build, params, ekf, kin, carry, inputs = test_sessions[0]
        _, out = run(theta, spec, build, params, ekf, kin, jnp.eye(3), carry, inputs)
        return body_velocity_l2(
            out.base.state.R, out.base.state.v,
            jnp.tile(jnp.eye(3), (inputs.accel_body.shape[0], 1, 1)),
            jnp.zeros((inputs.accel_body.shape[0], 3)),
        )

    held_out = float(holdout_loss(fit.theta))
    assert np.isfinite(held_out), "held-out loss must be a real number, not silently NaN/inf"

    # The seam this test exists to prove: swapping WHICH session the manifest calls "test" changes
    # the held-out number. If it didn't, the partition wouldn't actually be reaching the evaluation
    # (e.g. a bug that always evaluated session "0" regardless of what the manifest's split said).
    other_split_manifest = CaptureManifest(
        manifest.sessions,
        deterministic_split(manifest.sessions, n_test=1, seed=1),
        manifest.created_at, manifest.source_revision,
    )
    other_test_ids = [s.session_id for s in other_split_manifest.partition("test")]
    # Asserted, not branched around: seed=1 vs seed=7 on these 4 fixed session IDs is verified (once,
    # offline) to disagree. If a future change to deterministic_split's hashing ever made every seed
    # agree, `if other_test_ids != test_ids:` would silently skip the one assertion this test exists
    # for -- so a mismatch here must fail loudly instead of quietly no-op'ing the real check below.
    assert other_test_ids != test_ids, "fixture seeds must select different held-out sessions"

    other_test = _synthetic_session_data(_seed_for_session(manifest, other_test_ids[0], tmp_path))

    def other_holdout_loss(theta):
        build, params, ekf, kin, carry, inputs = other_test
        _, out = run(theta, spec, build, params, ekf, kin, jnp.eye(3), carry, inputs)
        return body_velocity_l2(
            out.base.state.R, out.base.state.v,
            jnp.tile(jnp.eye(3), (inputs.accel_body.shape[0], 1, 1)),
            jnp.zeros((inputs.accel_body.shape[0], 3)),
        )

    assert float(other_holdout_loss(fit.theta)) != pytest.approx(held_out, abs=1e-12)


def test_manifest_round_trip_survives_save_and_reload_with_hash_verification(tmp_path):
    """`CaptureManifest.load` must actually verify content, not just parse JSON -- corrupt one
    session's artifact after the manifest referencing it is saved and loading must fail loudly."""
    manifest = _write_manifest(tmp_path, {"a": 1, "b": 2, "c": 3})
    (tmp_path / manifest.sessions[0].robot_log.path).write_bytes(b"corrupted-after-the-fact")
    with pytest.raises(ValueError, match="hash mismatch"):
        CaptureManifest.load(tmp_path / "manifest.json")
