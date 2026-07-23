r"""
G9 on the REAL Alex model — assembly + the R_mount frame parity check.

The synthetic G9 gate (`tests/pipeline/test_main_estimator.py`) proves the fusion
is correct and constant-graph on a hand-built biped. This file proves it also
*assembles and runs on real Alex* (the log's own `model.sdf`), and — the part
that cannot be done synthetically — that the port's `R_mount` is the SAME frame
the Java InEKF used, checked against the logged
`invariantAppliedGyroBiasInPelvisFrame`.

Like the rest of `tests/replay/`, this skips without the hardware log + the
`ihmc-log` decoder (see `conftest.py`): a missing log is a missing fixture.

What "the frame is right" rests on
----------------------------------
`R_mount = ᴮR_S` maps the pelvis-IMU measurement frame `S` into the InEKF body
frame `B` (the pelvis root body). Java publishes the SAME bias in both frames
(`jointKF_gyroBias_pelvis_imu_*` in `S`, `invariantAppliedGyroBiasInPelvisFrame`
in `B`), so `R_mount @ bias_S` must equal `bias_B`. That is a pure rotation check
with no signal processing in the way — if it holds to numerical precision, the
mount rotation is exactly Java's. (The *raw angular velocity* does NOT match this
way: the real InEKF consumes a Mahony-prefiltered pelvis gyro, not the raw
`gyroscope_pelvis_imu` — a Tier-2 input-plumbing concern, not a frame one. See
PORT_NOTES "G9 — real model".)
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from invariant_estimation.config import load_config
from invariant_estimation.model.urdf2mjcf import convert_log_model
from invariant_estimation.pipeline import main_estimator as me
from invariant_estimation.replay.logsource import read_window

from .conftest import FILTERED_JOINTS, WINDOW


@pytest.fixture(scope="module")
def alex_fused(log_dir):
    """The fused estimator built on the real Alex model (with soles + body site)."""
    jk = load_config()["joint_kf"]
    spec = convert_log_model(
        log_dir,
        rotor_inertia=jk["rotor_inertia"],
        rotor_inertia_default=jk["rotor_inertia_default"],
        extra_sites=me.ALEX_EXTRA_SITES,
    )
    return me.build_alex_fused_estimator(spec)


def test_assembles_on_the_real_model(alex_fused):
    """The derived Alex topology reproduces the logged filter dimensions."""
    assert alex_fused.n_joints == len(FILTERED_JOINTS) == 9
    assert alex_fused.build.n_imus == 8, "jointKFNumberOfIMUs was 8 on this run"
    assert alex_fused.n_contacts == 2
    # The four ankle joints are the unfiltered anchor-chain split (Alex has no
    # foot IMUs, so the ankles are not filter states).
    assert alex_fused.build.anchor_unfiltered_mask.shape[1] == 4
    # No gap joints on Alex: the mass-matrix nuisance set is the base 6 DoF only.
    assert list(alex_fused.model.dof_nuisance) == list(range(6))


def test_runs_finite_and_constant_graph_on_the_real_model(alex_fused):
    """A short level-rest run: finite, PSD, and a single compiled graph (I7)."""
    f = alex_fused
    n_u = f.build.anchor_unfiltered_mask.shape[1]

    def sensors():
        return me.FusedSensors(
            encoders=jnp.zeros(f.n_joints),
            gyros=jnp.zeros((f.build.n_imus, 3)),
            accel_base=jnp.array([0.0, 0.0, 9.81]),
            qd_unfiltered=jnp.zeros(n_u),
            contact=jnp.ones(f.n_contacts),
            contact_chol=jnp.tile(jnp.eye(3) * 1e-4, (f.n_contacts, 1, 1)),
        )

    step = jax.jit(me.make_fused_step(f))
    carry = me.init_fused_carry(f, q0=jnp.zeros(f.n_joints))
    for _ in range(5):
        carry, out = step(carry, sensors())
        assert np.all(np.isfinite(np.asarray(out.p)))
        P = np.asarray(carry[1].state.P)
        assert np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -1e-9
    assert step._cache_size() == 1


def test_R_mount_is_a_ninety_degree_yaw(alex_fused):
    """The pelvis IMU is yawed +90° from the body (CLAUDE.md); R_mount must show it."""
    Rm = np.asarray(alex_fused.R_mount)
    assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-9)          # a real rotation
    yaw = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
    assert abs(yaw - 90.0) < 2.0, f"expected ~+90° pelvis-IMU yaw, got {yaw:.1f}°"


def test_R_mount_matches_the_java_inekf_bias_frame(alex_fused, log_dir, ihmclog_module):
    """R_mount @ jointKF_bias(S) == invariantAppliedGyroBiasInPelvisFrame(B), exactly.

    The decisive frame check: same physical bias, two frames Java published, one
    rotation between them. Agreement to sensor precision means the port's mount is
    Java's mount — the one place a G9 frame bug could hide.
    """
    Rm = np.asarray(alex_fused.R_mount)
    bias_s = [f"jointKF_gyroBias_pelvis_imu_{a}" for a in "XYZ"]
    bias_b = [f"invariantAppliedGyroBiasInPelvisFrame{a}" for a in "XYZ"]
    w = read_window(log_dir, bias_s + bias_b, start=WINDOW[0], end=WINDOW[1], stride=20)

    pred_b = w.stack(bias_s) @ Rm.T                              # R_mount @ bias_S
    ref_b = w.stack(bias_b)
    rms = float(np.sqrt(np.mean(np.sum((pred_b - ref_b) ** 2, axis=1))))
    assert rms < 1e-9, f"R_mount disagrees with Java's IMU->pelvis frame: RMS {rms:.2e}"
