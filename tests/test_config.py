"""Tests for `invariant_estimation.config` — the single tuning surface.

These guard the config *mechanism*, not the values: that every section the
modules read exists, that the factories actually take their defaults from the
file, and that the YAML 1.1 unsigned-exponent trap cannot reappear silently.
"""
import jax.numpy as jnp
import pytest
import yaml

from invariant_estimation import config
from invariant_estimation.inEKF import gravity_update as gu
from invariant_estimation.inEKF import state as s
from invariant_estimation.jointKF import state as jks

# Every section some module reads. A missing one must fail loudly.
REQUIRED_SECTIONS = ["inekf", "gravity_leveling", "joint_kf", "numerics"]


def test_default_config_file_exists_and_parses():
    cfg = config.load_config()
    assert isinstance(cfg, dict)
    for name in REQUIRED_SECTIONS:
        assert name in cfg, f"missing section '{name}'"


@pytest.mark.parametrize("name", REQUIRED_SECTIONS)
def test_section_accessor(name):
    assert isinstance(config.section(name), dict)


def test_missing_section_raises():
    """A typo'd section must fail loudly, never fall back to a hidden default."""
    with pytest.raises(KeyError):
        config.section("no_such_section")


def test_every_configured_scalar_is_numeric():
    """The YAML 1.1 trap: `1.0e9` parses as a *string*, `1.0e+9` as a float.

    `load_config` rejects it at load time; without that guard the failure
    surfaces thousands of lines away as a dtype error inside a jitted function.
    """
    def walk(node, trail=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{trail}.{k}" if trail else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{trail}[{i}]")
        else:
            assert not isinstance(node, str), f"{trail} = {node!r} is a string"

    walk(config.load_config())


def test_unsigned_exponent_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("inekf:\n  cond_max: 1.0e9\n")
    with pytest.raises(ValueError, match="signed exponent"):
        config.load_config(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# The factories actually read the file
# ---------------------------------------------------------------------------

def test_inekf_params_come_from_config():
    cfg = config.section("inekf")
    params = s.default_params(2)
    assert params.dt == cfg["dt"]
    assert params.gyro_var == cfg["gyro_var"]
    assert params.accel_var == cfg["accel_var"]
    assert params.contact_floor == cfg["contact_floor"]
    assert jnp.allclose(params.g, jnp.asarray(cfg["gravity"]))


def test_gravity_params_come_from_config():
    cfg = config.section("gravity_leveling")
    params = gu.default_gravity_params()
    assert params.roll_var == cfg["roll_var"]
    assert params.pitch_var == cfg["pitch_var"]
    assert params.tau == cfg["reference_tau"]
    assert params.norm_tol == cfg["gates"]["norm_tol"]
    assert params.rot_tol == cfg["gates"]["rot_tol"]
    assert params.horiz_tol == cfg["gates"]["horiz_tol"]


def test_joint_kf_params_come_from_config():
    cfg = config.section("joint_kf")
    params = jks.default_params()
    assert params.dt == cfg["dt"]
    assert params.sigma_tau == cfg["sigma_tau"]
    assert params.encoder_var == cfg["encoder_var"]
    assert params.qa_max == cfg["qa_max"]
    assert params.anchor_var == cfg["anchor_var"]
    assert params.init_pos_var == cfg["init"]["pos_var"]


def test_joint_kf_unknown_override_is_rejected():
    """A typo'd override must fail loudly rather than be silently dropped."""
    with pytest.raises(TypeError, match="unknown JointKFParams field"):
        jks.default_params(sigma_omega=1.0)


def test_explicit_arguments_override_config():
    """Every factory argument is an override — a sweep needn't touch the file."""
    assert s.default_params(1, gyro_var=0.5).gyro_var == 0.5
    assert gu.default_gravity_params(pitch_var=0.25).pitch_var == 0.25
    assert jks.default_params(dt=0.002).dt == 0.002
    assert jks.default_params(qa_max=1.0e9).qa_max == 1.0e9


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------

def test_set_config_swaps_the_active_tree(tmp_path):
    original = config.section("gravity_leveling")["roll_var"]
    custom = yaml.safe_load(yaml.safe_dump(config.load_config()))
    custom["gravity_leveling"]["roll_var"] = 0.123
    try:
        config.set_config(custom)
        assert gu.default_gravity_params().roll_var == 0.123
    finally:
        config.set_config(None)
    assert gu.default_gravity_params().roll_var == original


def test_test_locked_values_match_the_java_suite():
    """The §2b constants table is the acceptance checksum for the config file.

    A mismatch fails the port, not the test (CLAUDE.md §5).
    """
    inekf = config.section("inekf")
    gravity = config.section("gravity_leveling")

    assert inekf["gyro_var"] == 1.0e-4
    assert inekf["accel_var"] == 1.0e-3
    assert inekf["contact_var"] == 1.0e-6
    assert inekf["gravity"] == [0.0, 0.0, -9.81]
    assert inekf["initial_covariance"] == 1.0

    assert gravity["gravity"] == 9.81
    assert gravity["roll_var"] == 2.5e-3
    assert gravity["pitch_var"] == 1.9e-1
    assert gravity["reference_tau"] == 5.0
    assert gravity["gates"]["norm_tol"] == 0.05
    assert gravity["gates"]["rot_tol"] == 0.15
    assert gravity["gates"]["horiz_tol"] == 0.5

    # jointLevel — TEST_SUITE_MAP.md §"Filter constants the tests mirror".
    jkf = config.section("joint_kf")
    assert jkf["dt"] == 1.0e-3
    assert jkf["sigma_accel"] == 50.0
    assert jkf["sigma_tau"] == 5.0
    assert jkf["qa_max"] == 900.0
    assert jkf["encoder_var"] == 5.0e-5
    assert jkf["sigma_qd_unfiltered"] == 0.1
    assert jkf["anchor_var"] == 4.0e-4
    assert jkf["rotor_inertia_default"] == 0.005
    assert jkf["target_qdd_std"] == 20.0
    assert jkf["alpha_default"] == 0.15
    assert jkf["sigma_gyro_floor"] == 1.0e-6
    assert jkf["sigma_gyro_floor_trace"] == 3.0e-6
    assert jkf["cond_s_max"] == 1.0e9
    assert jkf["imu_bias_process_var"] == 1.0e-4
    assert jkf["lag_slew_smoothing_hz"] == 5.0
    assert jkf["init"] == {"pos_var": 1.0e-6, "vel_var": 1.0, "bias_var": 2.5e-3}

    # Rotor-inertia table — locked by testRotorInertiaTableLookup (tol 0.0).
    rotor = jkf["rotor_inertia"]
    assert rotor["HIP_X"] == 0.062
    assert rotor["HIP_Y"] == 0.167
    assert rotor["KNEE"] == 0.167
    assert rotor["ANKLE_Y"] == 0.07
    assert rotor["ANKLE_X"] == 0.05
    assert rotor["SPINE"] == 0.062


def test_rotor_inertia_lookup_matches_java_table():
    """`JointLevelKFRotorAndGramTest.testRotorInertiaTableLookup`, tol 0.0.

    Substring match, case-insensitive, 0.005 default for an unmatched joint.
    """
    for name, expected in [
        ("LEFT_HIP_X", 0.062),
        ("RIGHT_HIP_Y", 0.167),
        ("left_knee_y", 0.167),          # case-insensitive
        ("LEFT_ANKLE_Y", 0.070),
        ("LEFT_ANKLE_X", 0.050),
        ("SPINE_Z", 0.062),
        ("SOME_UNKNOWN_JOINT", 0.005),   # default floor
    ]:
        assert jks.rotor_inertia_for_name(name) == expected, name


def test_alpha_lookup_falls_back_to_default():
    """Calibrated per-joint alphas by substring; unlisted joints hit the default.

    The fallback is deliberate: an unlisted filtered joint must surface via the
    QA_MAX tripwire, not silently inherit a calibrated neighbour's value.
    """
    assert jks.alpha_for_name("LEFT_HIP_X") == 6.01544e-2
    assert jks.alpha_for_name("RIGHT_KNEE_Y") == 2.49771e-2
    assert jks.alpha_for_name("SPINE_Z") == 5.61133e-2
    # LEFT/RIGHT pairs agree to ~0.2% — the legs are physically identical, so
    # that symmetry is what validates the 2026-07-10 calibration measurement.
    for left, right in [("LEFT_HIP_X", "RIGHT_HIP_X"), ("LEFT_KNEE_Y", "RIGHT_KNEE_Y")]:
        a, b = jks.alpha_for_name(left), jks.alpha_for_name(right)
        assert abs(a - b) / a < 0.12, f"{left} vs {right}: {a} vs {b}"
    # Unlisted joint -> the loud default, never a calibrated neighbour's value.
    assert jks.alpha_for_name("LEFT_WRIST_Z") == 0.15
    assert jks.alpha_for_name("SOME_UNKNOWN_JOINT") == 0.15


def test_encoder_var_reports_unwired_joints():
    """An unwired joint takes the 5e-5 fallback and says so.

    That fallback is 2-4 orders ABOVE the hardware-measured per-joint variances,
    so a joint silently on it badly under-trusts its encoder — `build.py` must be
    able to name every such joint at boot (Java parity).
    """
    var, wired = jks.encoder_var_for_name("ANY_UNWIRED_JOINT")
    assert var == 5.0e-5
    assert wired is False
