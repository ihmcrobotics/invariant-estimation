r"""End-to-end rehearsal of the training pipeline on generated data.

Runs the whole chain the real study will run -- manifest, split, training on the
train partition only, evaluation on held-out sessions, artifact export -- against
`synthetic.generate_session` instead of a robot log. Every stage is the real
one; only the data source is substituted.

What a passing rehearsal establishes
------------------------------------
That the pieces compose: gradients reach every channel through both filter
stages, the split is honoured end to end, training reduces error on sessions the
optimizer never saw, and the result serializes into an artifact the Java reader
accepts. Those are the failures that would otherwise be discovered during
scarce robot time.

What it does not establish
--------------------------
Anything about real robot data. The generated sessions have exactly the i.i.d.
Gaussian noise the filters assume, perfect contact, and no model error -- see
`synthetic`'s own docstring. Held-out improvement here is evidence the machinery
works, not evidence about how much a real capture will gain.

Run it directly::

    python -m invariant_estimation.learning.dryrun
"""
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .artifact import from_fit, save_artifact
from .captures import Artifact, CaptureManifest, CaptureSession, deterministic_split
from .noise import NoiseSpec
from .optimize import body_velocity_l2, fit_scalars
from .synthetic import baseline_ekf_variances, generate_session
from .two_stage import run


class SessionScore(NamedTuple):
    session_id: str
    partition: str
    baseline: float
    fitted: float

    @property
    def improvement(self) -> float:
        """Fractional reduction in loss; positive is better."""
        return (self.baseline - self.fitted) / self.baseline


class DryRunResult(NamedTuple):
    theta: object
    scales: dict
    scores: tuple
    artifact_path: object

    def held_out(self):
        return tuple(s for s in self.scores if s.partition == "test")

    def report(self) -> str:
        lines = [f"{'session':>10} {'split':>6} {'baseline':>12} {'fitted':>12} {'change':>9}"]
        for score in self.scores:
            lines.append(f"{score.session_id:>10} {score.partition:>6} {score.baseline:>12.6f} "
                         f"{score.fitted:>12.6f} {-100 * score.improvement:>8.1f}%")
        improved = sum(s.improvement > 0 for s in self.held_out())
        lines.append(f"held-out sessions improved: {improved}/{len(self.held_out())}")
        if self.artifact_path is not None:
            lines.append(f"artifact: {self.artifact_path}")
        lines.append("generated data, not a robot -- see synthetic.py for what this cannot tell you")
        return "\n".join(lines)


def _manifest(directory: Path, session_ids, n_test: int, seed: int) -> CaptureManifest:
    """A real `CaptureManifest` over placeholder files, exercised the way a real one would be.

    The files are stubs, but the manifest is not: it is hash-verified, split by the real
    `deterministic_split`, and round-tripped through disk, so a regression in any of that
    surfaces here rather than on capture day.
    """
    sessions = []
    for session_id in session_ids:
        artifacts = {}
        for name in ("robot_log", "mocap_log", "calibration", "model", "synchronization"):
            relative = f"{session_id}/{name}.bin"
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"{session_id}:{name}".encode()
            path.write_bytes(content)
            artifacts[name] = Artifact(relative, hashlib.sha256(content).hexdigest())
        sessions.append(CaptureSession(
            session_id, session_id, "synthetic-rehearsal", "2026-09-15T09:00:00-05:00",
            artifacts["robot_log"], artifacts["mocap_log"], artifacts["calibration"],
            artifacts["model"], artifacts["synchronization"],
            "aligned_robot_monotonic_ns", "registered_zup", "pelvis",
            0, 1_000_000_000, ("c0", "c1", "c2"), ("base", "mid", "tip"),
        ))
    manifest = CaptureManifest(tuple(sessions), deterministic_split(sessions, n_test=n_test, seed=seed),
                               "2026-09-15T09:00:00Z", "git:dryrun")
    manifest.save(directory / "manifest.json")
    return CaptureManifest.load(directory / "manifest.json")


def dry_run(directory, *, n_sessions: int = 5, n_test: int = 2, ticks: int = 120,
            steps: int = 60, learning_rate: float = 0.08, arm: int = 7,
            split_seed: int = 7) -> DryRunResult:
    """Run the full rehearsal and write an artifact into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    session_ids = [str(i) for i in range(n_sessions)]
    manifest = _manifest(directory, session_ids, n_test, split_seed)
    # The session's own id seeds its trajectory, so the manifest genuinely selects the data --
    # a split change moves which trajectories the optimizer sees, not just a label.
    data = {sid: generate_session(int(sid) + 1, ticks=ticks) for sid in session_ids}

    train_ids = [s.session_id for s in manifest.partition("train")]
    test_ids = [s.session_id for s in manifest.partition("test")]
    spec = NoiseSpec(tuple(data[train_ids[0]].build.imu_names), arm=arm)

    def session_loss(theta, session):
        _, out = run(theta, spec, session.build, session.joint_params, session.ekf,
                     session.kinematics, jnp.eye(3), session.carry, session.inputs)
        return body_velocity_l2(out.base.state.R, out.base.state.v,
                                session.truth_rotation, session.truth_velocity)

    def training_loss(theta):
        return jnp.mean(jnp.stack([session_loss(theta, data[sid]) for sid in train_ids]))

    initial = spec.initial_theta()
    fit = fit_scalars(training_loss, initial, steps=steps, learning_rate=learning_rate)

    scores = []
    for partition, ids in (("train", train_ids), ("test", test_ids)):
        for sid in ids:
            scores.append(SessionScore(sid, partition,
                                       float(session_loss(initial, data[sid])),
                                       float(session_loss(fit.theta, data[sid]))))

    artifact_path = directory / "noise_artifact.json"
    save_artifact(artifact_path, _artifact(fit.theta, spec, data[train_ids[0]], manifest,
                                           train_ids, test_ids, directory))

    values = spec.scales(fit.theta)
    flat = np.concatenate((np.asarray(values.imu_gyro), np.array(values[1:])))
    return DryRunResult(fit.theta, dict(zip(spec.names, (float(v) for v in flat))),
                        tuple(scores), artifact_path)


def _artifact(theta, spec, session, manifest, train_ids, test_ids, directory):
    gyro_var, accel_var, contact_var = baseline_ekf_variances()
    manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    context = {
        "robot_id": "synthetic-rehearsal", "model_sha256": "0" * 64, "dt": float(session.joint_params.dt),
        "imu_names": list(spec.imu_names), "joint_names": list(session.build.joint_names),
        "contact_names": ["foot"], "base_frame": "pelvis",
        "contact_process_model": "constant_body_isotropic", "fk_noise_model": "joint_covariance",
    }
    provenance = {
        "created_at": manifest.created_at.replace("Z", "+00:00"),
        "python_revision": "git:dryrun", "java_revision": "git:dryrun",
        "capture_manifest_sha256": manifest_hash, "baseline_config_sha256": "1" * 64,
        "train_sessions": list(train_ids), "validation_sessions": [],
        "test_sessions": list(test_ids), "objective": "body_velocity_l2",
    }
    baseline = {
        "base_gyro_q": gyro_var, "base_accel_q": accel_var, "contact_q": contact_var,
        "gravity_roll_r": 2.5e-3, "gravity_pitch_r": 0.19, "encoder_fallback_var": 5.0e-5,
        "imu_gyro_covariances": {name: np.asarray(session.build.gyro_sigma[i])
                                 for i, name in enumerate(spec.imu_names)},
    }
    return from_fit(theta, spec, context=context, provenance=provenance, baseline=baseline)


def main():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        result = dry_run(directory)
        print(result.report())
        print()
        print(json.dumps(result.scales, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
