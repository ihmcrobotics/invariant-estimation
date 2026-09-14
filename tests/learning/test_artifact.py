import copy

import numpy as np
import pytest

from invariant_estimation.learning.artifact import from_fit, validate_artifact, save_artifact, load_artifact
from invariant_estimation.learning.noise import NoiseSpec

def payload(arm=7):
    spec = NoiseSpec(("base", "shin"), arm=arm)
    context = {"robot_id":"alex001", "model_sha256":"a"*64, "dt":.001,
               "imu_names":["base","shin"], "joint_names":["knee"],
               "contact_names":["left","right"], "base_frame":"pelvis",
               "contact_process_model":"constant_body_isotropic", "fk_noise_model":"joint_covariance"}
    provenance = {"created_at":"2026-09-14T12:00:00-05:00", "python_revision":"git:py",
                  "java_revision":"git:java", "capture_manifest_sha256":"b"*64,
                  "baseline_config_sha256":"c"*64, "train_sessions":["s1"],
                  "validation_sessions":[], "test_sessions":["s2"], "objective":"l2+nees"}
    baseline = {"base_gyro_q":1e-4, "base_accel_q":1e-3, "contact_q":1e-6,
                "gravity_roll_r":2.5e-3, "gravity_pitch_r":.19,
                "encoder_fallback_var":5e-5,
                "imu_gyro_covariances":{"base":[[1e-4,0,0],[0,1e-4,0],[0,0,1e-4]],
                                         "shin":[[2e-4,0,0],[0,2e-4,0],[0,0,2e-4]]}}
    return from_fit(spec.initial_theta(), spec, context=context, provenance=provenance, baseline=baseline)

def test_roundtrip_and_explicit_create(tmp_path):
    data = payload(); validate_artifact(data)
    path = tmp_path / "noise.json"; save_artifact(path, data)
    assert load_artifact(path) == data
    with pytest.raises(FileExistsError): save_artifact(path, data)

def test_rejects_frozen_channel_and_bad_fk_contract():
    data = payload(arm=5); data["scales"]["base_gyro_q"] = 2.0
    with pytest.raises(ValueError, match="frozen"): validate_artifact(data)
    data = payload(); data["context"]["fk_noise_model"] = "constant"
    with pytest.raises(ValueError, match="J Sigma"): validate_artifact(data)

def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"): load_artifact(path)


def test_accepts_numpy_scalars_and_arrays_from_the_real_training_pipeline():
    """from_fit's own producer (noise.py/optimize.py) hands back numpy.float64 scalars and
    jax/numpy ndarrays, not hand-typed Python literals -- this is the exact shape a real caller
    passes, unlike `payload()`'s literals above. Regression pin for a bug where the exact-type
    check `type(value) not in (float, int)` rejected numpy.float64 (a real float subclass that
    json.dumps already serializes fine) and json.dumps itself choked on a raw ndarray covariance."""
    spec = NoiseSpec(("base", "shin"), arm=7)
    context = {"robot_id": "alex001", "model_sha256": "a" * 64, "dt": np.float64(0.001),
               "imu_names": ["base", "shin"], "joint_names": ["knee"],
               "contact_names": ["left", "right"], "base_frame": "pelvis",
               "contact_process_model": "constant_body_isotropic", "fk_noise_model": "joint_covariance"}
    provenance = {"created_at": "2026-09-14T12:00:00-05:00", "python_revision": "git:py",
                  "java_revision": "git:java", "capture_manifest_sha256": "b" * 64,
                  "baseline_config_sha256": "c" * 64, "train_sessions": ["s1"],
                  "validation_sessions": [], "test_sessions": ["s2"], "objective": "l2+nees"}
    baseline = {"base_gyro_q": np.float64(1e-4), "base_accel_q": 1e-3, "contact_q": 1e-6,
                "gravity_roll_r": 2.5e-3, "gravity_pitch_r": .19, "encoder_fallback_var": 5e-5,
                "imu_gyro_covariances": {"base": np.eye(3) * 1e-4, "shin": np.eye(3) * 2e-4}}
    data = from_fit(spec.initial_theta(), spec, context=context, provenance=provenance, baseline=baseline)
    validate_artifact(data)  # from_fit already validates; re-checking pins the contract independently
    assert isinstance(data["context"]["dt"], float)
    assert isinstance(data["baseline"]["imu_gyro_covariances"]["base"], list)


def test_bool_is_never_accepted_as_a_variance_even_though_it_is_an_int_subclass():
    """isinstance(True, int) is True in Python -- the fix for the numpy.float64 rejection above
    must not accidentally start accepting True/False as if they were 1/0."""
    data = payload()
    data["context"]["dt"] = True
    with pytest.raises(ValueError, match="positive finite"):
        validate_artifact(data)


def test_from_fit_rejects_a_malformed_baseline_with_validate_artifacts_error_not_a_keyerror():
    """from_fit's covariance-normalization step (added for the ndarray fix above) runs BEFORE
    validate_artifact and must not itself crash with a bare KeyError on a baseline that is
    missing imu_gyro_covariances -- the schema error belongs to validate_artifact, not to a
    normalization helper reaching for a key that was never guaranteed to exist."""
    spec = NoiseSpec(("base", "shin"), arm=7)
    context = {"robot_id": "alex001", "model_sha256": "a" * 64, "dt": 0.001,
               "imu_names": ["base", "shin"], "joint_names": ["knee"],
               "contact_names": ["left", "right"], "base_frame": "pelvis",
               "contact_process_model": "constant_body_isotropic", "fk_noise_model": "joint_covariance"}
    provenance = {"created_at": "2026-09-14T12:00:00-05:00", "python_revision": "git:py",
                  "java_revision": "git:java", "capture_manifest_sha256": "b" * 64,
                  "baseline_config_sha256": "c" * 64, "train_sessions": ["s1"],
                  "validation_sessions": [], "test_sessions": ["s2"], "objective": "l2+nees"}
    baseline_without_covariances = {"base_gyro_q": 1e-4, "base_accel_q": 1e-3, "contact_q": 1e-6,
                                     "gravity_roll_r": 2.5e-3, "gravity_pitch_r": .19,
                                     "encoder_fallback_var": 5e-5}
    with pytest.raises(ValueError, match="baseline"):
        from_fit(spec.initial_theta(), spec, context=context, provenance=provenance,
                  baseline=baseline_without_covariances)
