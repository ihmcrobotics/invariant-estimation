"""Versioned scalar-noise export for the Java loader (Tier 1 item 4).

Carries dimensional baselines as well as multipliers: a number without its
baseline, units, model, and training provenance is not deployable. This schema
does not serialize ContactNet weights or assert Java/JAX policy parity.
"""
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np

from .noise import INEKF_CHANNELS


BASELINE_SCALARS = (
    "base_gyro_q", "base_accel_q", "contact_q", "gravity_roll_r",
    "gravity_pitch_r", "encoder_fallback_var",
)
CONTEXT_FIELDS = {
    "robot_id", "model_sha256", "dt", "imu_names", "joint_names",
    "contact_names", "base_frame", "contact_process_model", "fk_noise_model",
}
PROVENANCE_FIELDS = {
    "created_at", "python_revision", "java_revision", "capture_manifest_sha256",
    "baseline_config_sha256", "train_sessions", "validation_sessions", "test_sessions", "objective",
}


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: nonempty string required")


def _names(value, label, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{label}: list required")
    for name in value:
        _text(name, label)
    if len(value) != len(set(value)):
        raise ValueError(f"{label}: duplicate name")


def _positive(value, label):
    # isinstance, not the exact-type check this replaced: dt/variances arriving from the JAX
    # training pipeline are routinely numpy.float64 (a real float subclass -- json.dumps already
    # serializes it as a plain float), and rejecting them made this schema unusable from its own
    # producer. bool is still excluded even though it IS an int subclass, matching the exact-type
    # check's original (accidental but correct) behavior: True/False must never pass as a variance.
    # numpy integer types stay rejected too -- they are not int subclasses in numpy, and json.dumps
    # cannot serialize them regardless of what this check allows.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: positive finite number required")


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"{label}: lowercase SHA-256 required")


def validate_artifact(data):
    _keys(data, {"schema_version", "kind", "parameterization", "arm", "max_scale",
                 "context", "provenance", "baseline", "scales"}, "artifact")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if data["kind"] != "alex_scalar_noise" or data["parameterization"] != "variance_scale_v1":
        raise ValueError("unsupported kind/parameterization")
    if type(data["arm"]) is not int or data["arm"] not in (4, 5, 6, 7):
        raise ValueError("unsupported arm")
    _positive(data["max_scale"], "max_scale")
    if not 1 < data["max_scale"] <= 100:
        raise ValueError("deployment max_scale must be in (1,100]")
    context, provenance, baseline, scales = (data[k] for k in ("context", "provenance", "baseline", "scales"))
    _keys(context, CONTEXT_FIELDS, "context")
    for name in ("robot_id", "base_frame"):
        _text(context[name], name)
    _sha(context["model_sha256"], "model_sha256")
    _positive(context["dt"], "dt")
    for name in ("imu_names", "joint_names", "contact_names"):
        _names(context[name], name)
    if context["contact_process_model"] != "constant_body_isotropic":
        raise ValueError("v1 cannot deploy a dynamic ContactNet/contact schedule")
    if context["fk_noise_model"] != "joint_covariance":
        raise ValueError("v1 requires the J Sigma_q J.T FK noise path")
    _keys(provenance, PROVENANCE_FIELDS, "provenance")
    for name in ("created_at", "python_revision", "java_revision", "objective"):
        _text(provenance[name], name)
    if datetime.fromisoformat(provenance["created_at"]).utcoffset() is None:
        raise ValueError("created_at must include timezone")
    for name in ("capture_manifest_sha256", "baseline_config_sha256"):
        _sha(provenance[name], name)
    seen = set()
    for name in ("train_sessions", "validation_sessions", "test_sessions"):
        values = provenance[name]
        _names(values, name, empty=name == "validation_sessions")
        if seen.intersection(values):
            raise ValueError("session overlap in artifact provenance")
        seen.update(values)
    _keys(baseline, set(BASELINE_SCALARS) | {"imu_gyro_covariances"}, "baseline")
    for name in BASELINE_SCALARS:
        _positive(baseline[name], name)
    covariances = baseline["imu_gyro_covariances"]
    _keys(covariances, context["imu_names"], "imu_gyro_covariances")
    for name, matrix in covariances.items():
        cov = np.asarray(matrix)
        if cov.shape != (3, 3) or cov.dtype.kind not in "fi" or not np.isfinite(cov).all():
            raise ValueError(f"invalid 3x3 covariance: {name}")
        if not np.allclose(cov, cov.T, rtol=0, atol=1e-14) or np.linalg.eigvalsh(cov).min() <= 0:
            raise ValueError(f"covariance must be symmetric positive definite: {name}")
    _keys(scales, {f"imu_gyro:{n}" for n in context["imu_names"]} | set(INEKF_CHANNELS), "scales")
    for name, value in scales.items():
        _positive(value, name)
        if not 1/data["max_scale"]*(1-1e-12) <= value <= data["max_scale"]*(1+1e-12):
            raise ValueError(f"scale outside declared bounds: {name}")
        active = data["arm"] in ((5, 7) if name.startswith("imu_gyro:") else (6, 7))
        if not active and value != 1:
            raise ValueError(f"frozen arm channel must equal 1: {name}")


def from_fit(theta, spec, *, context, provenance, baseline):
    """Materialize a fit, requiring explicit captured baseline and provenance.

    Baseline IMU covariances must be the same post-floor build.gyro_sigma used
    during training. JSON roundtrip detaches caller-owned mutable dictionaries.
    """
    if tuple(context["imu_names"]) != spec.imu_names:
        raise ValueError("context IMU order differs from training spec")
    values = spec.scales(theta)
    vector = np.concatenate((np.asarray(values.imu_gyro), np.array(values[1:])))
    # build.gyro_sigma (and anything else a real caller pulls a covariance from) is a JAX/numpy
    # array, not a nested Python list, and json.dumps cannot serialize an ndarray -- this is the
    # producer's own output type, so rejecting it here would make the schema unusable from the
    # pipeline that is supposed to fill it. validate_artifact already accepts either shape (it
    # goes through np.asarray itself), so normalizing before validation costs nothing; malformed
    # baseline dicts (wrong keys, missing field) still fall through unnormalized and get
    # validate_artifact's proper error instead of a bare KeyError here.
    covariances = baseline.get("imu_gyro_covariances") if isinstance(baseline, dict) else None
    if isinstance(covariances, dict):
        baseline = dict(baseline)
        baseline["imu_gyro_covariances"] = {
            name: np.asarray(matrix).tolist() for name, matrix in covariances.items()
        }
    data = dict(schema_version=1, kind="alex_scalar_noise", parameterization="variance_scale_v1",
                arm=spec.arm, max_scale=spec.max_scale, context=context, provenance=provenance,
                baseline=baseline, scales=dict(zip(spec.names, map(float, vector))))
    validate_artifact(data)
    return json.loads(json.dumps(data, allow_nan=False))


def save_artifact(path, data):
    validate_artifact(data)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_artifact(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    validate_artifact(data)
    return data
