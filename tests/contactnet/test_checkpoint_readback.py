"""Invariant N5, as a test rather than a paragraph: the diag(L) parameterisation must
be read back from `summary.json` when a checkpoint is loaded.

The failure this prevents is silent by construction. `train.save_params` writes bare
pytree leaves, so a checkpoint carries no metadata; the same weights are a valid head
under every parameterisation, so nothing about them reveals a mismatch; and the
resulting Sigma_C is wrong by orders of magnitude with no shape error, no NaN and no
diagnostic. Every deployment path constructed a default `ContactNetConfig()` until
2026-08-10, which means the `exp` checkpoint could only ever have been evaluated as
softplus.

`test_diag_param.test_mismatched_parameterisation_changes_sigma_c` documents the size
of the error; this file asserts the mechanism that avoids it.
"""
import json

import pytest

import invariant_estimation  # noqa: F401  (enables x64)
from invariant_estimation.contactnet import network
from invariant_estimation.contactnet.checkpoint import (
    RESTORED_FIELDS, config_for_checkpoint, restored_fields)
from invariant_estimation.contactnet.config import ContactNetConfig


def _run_dir(tmp_path, cfg_block, name="run"):
    """A checkpoint directory carrying only the summary the loader reads."""
    d = tmp_path / name
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"cfg": cfg_block}))
    (d / "params.npz").write_bytes(b"")        # never opened by the resolver
    return d


def test_restores_the_recorded_parameterisation(tmp_path):
    d = _run_dir(tmp_path, {"diag_param": "bounded_exp",
                            "diag_lo": 1.0e-6, "diag_hi": 3.0e1})
    cfg = config_for_checkpoint(str(d))
    assert cfg.diag_param == "bounded_exp"
    assert (cfg.diag_lo, cfg.diag_hi) == (1.0e-6, 3.0e1)
    # and it arrives at the network as one object, bounds included
    assert cfg.diag_spec == network.DiagSpec("bounded_exp", 1.0e-6, 3.0e1)


def test_accepts_the_params_file_as_well_as_the_directory(tmp_path):
    """Callers hold either: `run_estimator` takes `--contactnet PARAMS.npz`,
    `evaluate_run` takes a run directory."""
    d = _run_dir(tmp_path, {"diag_param": "exp"})
    assert config_for_checkpoint(str(d)).diag_param == "exp"
    assert config_for_checkpoint(str(d / "params.npz")).diag_param == "exp"


@pytest.mark.parametrize("cfg_block", [
    {},                                   # summary from before the option existed
    {"L": 256, "objective": "l2_velocity"},
])
def test_unrecorded_fields_fall_back_to_the_pre_option_default(tmp_path, cfg_block):
    """Legacy checkpoints -- including the best one to date, whose summary.json has no
    `diag_param` key -- must keep loading exactly as they always did, which means
    softplus."""
    d = _run_dir(tmp_path, cfg_block)
    cfg = config_for_checkpoint(str(d))
    assert cfg.diag_param == "softplus"
    assert (cfg.diag_lo, cfg.diag_hi) == (network.DIAG_LO_DEFAULT,
                                          network.DIAG_HI_DEFAULT)


def test_missing_or_unreadable_summary_does_not_raise(tmp_path):
    """A loader that dies on a missing summary would block every pre-2026-08 run; the
    geometry check next to it takes the same view."""
    assert restored_fields(str(tmp_path / "nonexistent")) == {}
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "summary.json").write_text("{not json")
    assert restored_fields(str(bad)) == {}
    assert config_for_checkpoint(str(bad)).diag_param == "softplus"


def test_base_supplies_the_fields_that_are_not_restored(tmp_path):
    """Callers with their own config (drift_backfill's `L=EVAL_L`, evaluate_run's
    objective) must keep it; only the recorded fields are overwritten."""
    d = _run_dir(tmp_path, {"diag_param": "exp"})
    base = ContactNetConfig(L=64, objective="beta_nll")
    cfg = config_for_checkpoint(str(d), base=base)
    assert (cfg.L, cfg.objective) == (64, "beta_nll")
    assert cfg.diag_param == "exp"


def test_explicit_overrides_beat_the_recording(tmp_path):
    """Deliberately reading a checkpoint under the wrong parameterisation is a real
    diagnostic (`plot_contact_phase --diag-param`); it must stay possible, and it must
    take an explicit argument to get."""
    d = _run_dir(tmp_path, {"diag_param": "exp"})
    assert config_for_checkpoint(str(d), diag_param="softplus").diag_param == "softplus"


def test_every_restored_field_is_a_real_config_field():
    """A typo in `RESTORED_FIELDS` would silently restore nothing, which is precisely
    the defect this module exists to remove."""
    for f in RESTORED_FIELDS:
        assert hasattr(ContactNetConfig(), f), f
