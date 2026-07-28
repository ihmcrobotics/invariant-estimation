r"""`contactnet/dataset.py` — the segment loader.

Every test here runs on a **fabricated** rollout written to a tmp dir in exactly
the format `sim.collect.save_rollout` produces, so nothing in this file needs
MJX, a policy, or a 130 MB `.npz`.  The fabrication is deliberately *not* random
noise everywhere: `contact_chol` really switches stance↔swing, the channels
carry per-channel offsets, and the FK vectors are a smooth function of tick, so
a test that asserts "the loader kept the structure" has structure to lose.

The properties with teeth, and the mutants each one kills, are noted per test.
Several tests assert their own non-vacuity in-line (a "the unmutated form would
also pass this" check), because a loader test that compares an array against
itself is the failure mode this suite is most exposed to.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation  # noqa: F401  (x64 side effect, must precede arrays)
from invariant_estimation.contactnet import dataset, features, normalize
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.inEKF import ekf as ekf_mod
from invariant_estimation.inEKF.correct import innovation
from invariant_estimation.inEKF.filter import ContactFrames, InEKFInputs, JointFilterOutput
from invariant_estimation.sim.collect import Rollout, save_rollout
from invariant_estimation.pipeline.main_estimator import FusedSensors

T_FAKE = 1_200
N_C = 2
N_J = 6                 # matches the inEKF fixture kinematics, so `measure_p0` is testable
WARMUP = 100
F = len(features.channel_names())


# ---------------------------------------------------------------------------
# Fabrication
# ---------------------------------------------------------------------------

def _cfg(**kw) -> ContactNetConfig:
    """Small config: H=5, stride=8 (lead-in 32) so a 1200-tick rollout has room."""
    base = dict(F=F, sigma_0=1.0e-4, H=5, window_span_s=0.032, dt=1.0e-3, L=16, B=4)
    base.update(kw)
    return ContactNetConfig(**base)


def _fake_channels(seed: int) -> np.ndarray:
    """(T, N_c, F) with a per-channel offset and scale, so normalization is visible."""
    r = np.random.default_rng(seed)
    t = np.arange(T_FAKE)[:, None, None]
    offs = np.arange(F)[None, None, :] * 0.5
    return (offs + np.sin(0.01 * t + np.arange(N_C)[None, :, None])
            + 0.01 * r.standard_normal((T_FAKE, N_C, F)))


def _fake_rollout(seed: int, terrain: str) -> Rollout:
    r = np.random.default_rng(1000 + seed)
    T = T_FAKE

    def rn(*shape):
        return r.standard_normal(shape)

    # contact_chol carries the sim's stance/swing GROUND TRUTH, exactly as
    # `SimSensorReader` writes it: a scalar times I3, switching 1e-4 <-> 1e1.
    stance = (np.sin(np.arange(T)[:, None] * 0.05 + np.arange(N_C)[None, :]) > 0.0)
    scal = np.where(stance, 1.0e-4, 1.0e1)
    chol = scal[:, :, None, None] * np.eye(3)[None, None]

    sensors = FusedSensors(
        encoders=rn(T, N_J), gyros=rn(T, 3, 3), accel_base=rn(T, 3),
        qd_unfiltered=rn(T, 0), contact=r.random((T, N_C)), contact_chol=chol,
        q_unfiltered=rn(T, 0), torques=rn(T, N_J),
    )
    inputs = InEKFInputs(
        omega=rn(T, 3), accel=rn(T, 3), raw_omega=rn(T, 3),
        joint=JointFilterOutput(
            q=0.3 * np.sin(np.arange(T)[:, None] * 0.01 + np.arange(N_J)[None, :]),
            q_dot=rn(T, N_J),
            sigma_q=np.tile(np.eye(N_J) * 5.0e-5, (T, 1, 1)),
            sigma_q_dot=np.tile(np.eye(N_J) * 1.0e-2, (T, 1, 1)),
        ),
        contact_chol=chol,
        contact_meas_chol=np.zeros((T, N_C, 3, 3)),
    )
    ang = np.arange(T) * 1.0e-4
    R = np.stack([np.array([[np.cos(a), -np.sin(a), 0.0],
                            [np.sin(a), np.cos(a), 0.0],
                            [0.0, 0.0, 1.0]]) for a in ang])
    truth = {"R": R, "omega": rn(T, 3), "v": rn(T, 3) * 0.1,
             "p": np.cumsum(rn(T, 3) * 1e-3, axis=0), "q": rn(T, N_J), "q_dot": rn(T, N_J)}
    aux = {"bias": rn(T, 9), "nis": r.random(T), "est_R": R, "est_v": rn(T, 3),
           "est_p": rn(T, 3)}
    meta = {"terrain": terrain, "seed": seed, "warmup_ticks": WARMUP, "T": T,
            "dt": 1.0e-3, "seconds": 1.2, "settle_s": 0.0}
    return Rollout(sensors=sensors, inputs=inputs, truth=truth, aux=aux, meta=meta)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """Four fabricated rollouts + their channel caches, in `collect`'s own format."""
    d = tmp_path_factory.mktemp("data")
    cache = d / "cache"
    cache.mkdir()
    for i, terr in enumerate(["flat", "waves", "stones", "hard"]):
        roll = _fake_rollout(i, terr)
        p = d / f"{terr}_seed{i:03d}.npz"
        save_rollout(roll, p)
        y = 0.1 * np.sin(np.arange(T_FAKE)[:, None, None] * 0.01
                         + np.arange(N_C)[None, :, None] + np.arange(3)[None, None, :])
        y[:, :, 2] -= 0.9
        np.savez_compressed(cache / f"{p.stem}_feat.npz", channels=_fake_channels(i),
                            y_fk=y, names=np.asarray(features.channel_names()),
                            meta=np.array(json.dumps(roll.meta)))
    return d


@pytest.fixture(scope="module")
def norm(dataset_dir):
    return dataset.fit_normalization(dataset.rollout_paths(dataset_dir),
                                     cache_dir=dataset_dir / "cache", source="test")


@pytest.fixture(scope="module")
def preps(dataset_dir, norm):
    return dataset.prepare(dataset.rollout_paths(dataset_dir), norm, _cfg(),
                           cache_dir=dataset_dir / "cache")


P0 = np.eye(9 + 3 * N_C) * 1.0e-4


# ---------------------------------------------------------------------------
# Windowing — the loader's one piece of index arithmetic
# ---------------------------------------------------------------------------

def test_segment_windows_are_bit_identical_to_features_window(dataset_dir, norm, preps):
    r"""A segment's windows == `features.window` over the whole rollout, sliced.

    The strongest available oracle: `features.window` is the tested reference and
    the loader reimplements only its *gather*, on a global boxcar, precisely so
    the two can be compared bit-for-bit rather than approximately.

    Kills: a missing `swapaxes` (contacts interleaved into history), an
    off-by-one in the stride, a segment offset by `t0`, and boxcar-on-the-slice
    (which would agree only to ~1e-16 and fail `array_equal`).
    """
    cfg = _cfg()
    p = preps[0]
    c = dataset.load_channel_cache(dataset.cache_path(
        dataset.rollout_paths(dataset_dir)[0], dataset_dir / "cache"))
    ref = np.asarray(features.window(
        normalize.apply(jnp.asarray(c["channels"]), norm), cfg.H, cfg.stride))

    for t0 in (p.t_lo, p.t_lo + 37, (p.t_lo + p.t_hi) // 2, p.t_hi):
        seg = dataset.make_segment(p, t0, cfg, P0)
        assert seg.windows.shape == (cfg.L, N_C, cfg.H, F)
        assert np.array_equal(seg.windows, ref[t0:t0 + cfg.L]), f"t0={t0}"

    # Non-vacuity: the reference is not constant along any of the axes being
    # checked, so an axis mix-up really does change the answer.
    w = ref[p.t_lo:p.t_lo + cfg.L]
    assert not np.array_equal(w, np.swapaxes(w, 1, 2).reshape(w.shape))
    assert not np.array_equal(w, ref[p.t_lo + 1:p.t_lo + 1 + cfg.L])


def test_rollout_paths_ignores_non_rollout_npz(dataset_dir, norm, tmp_path):
    r"""``data/`` also holds the constants, ``P0`` and checkpoints — none are rollouts.

    Structural, not name-based: this fires on *any* future artifact dropped in
    the data directory.  It is a regression test — the first training run wrote
    its checkpoint to ``data/`` and the loader tried to read it as a rollout.
    """
    d = tmp_path / "data"
    d.mkdir()
    for p in dataset.rollout_paths(dataset_dir)[:2]:
        (d / p.name).symlink_to(p)
    normalize.save(str(d / "norm_constants.npz"), norm)
    np.savez(d / "p0.npz", P0=np.eye(15))
    np.savez(d / "contactnet_smoke.npz", w=np.zeros((3, 3)))
    assert [p.name for p in dataset.rollout_paths(d)] == \
        [p.name for p in dataset.rollout_paths(dataset_dir)[:2]]
    assert not dataset.is_rollout(d / "p0.npz")
    assert dataset.is_rollout(dataset.rollout_paths(dataset_dir)[0])


def test_window_indices_are_causal_and_never_clamp(preps):
    """Every history index lies in ``[0, k]`` and no legal start clamps.

    Clamping is the trap the lead-in bound exists for: `features.window_indices`
    lower-clamps at 0, which fabricates a window out of repeated samples. The
    loader refuses instead.
    """
    cfg = _cfg()
    p = preps[0]
    for t0 in (p.t_lo, p.t_hi):
        idx = dataset._segment_window_indices(t0, cfg)
        ticks = t0 + np.arange(cfg.L)
        assert idx.max(axis=1).tolist() == ticks.tolist()      # every row ENDS at k
        assert (idx <= ticks[:, None]).all()                   # causal
        assert idx.min() >= 0
    # One tick earlier than the bound would clamp, and the loader says so.
    with pytest.raises(ValueError, match="clamp"):
        dataset._segment_window_indices((cfg.H - 1) * cfg.stride - 1, cfg)


def test_valid_start_range_excludes_warmup_and_lead_in(preps):
    cfg = _cfg()
    lo, hi = dataset.valid_start_range(T_FAKE, WARMUP, cfg)
    assert lo == WARMUP + (cfg.H - 1) * cfg.stride
    assert hi == T_FAKE - cfg.L
    assert preps[0].t_lo == lo and preps[0].t_hi == hi
    with pytest.raises(ValueError, match="no legal segment start"):
        dataset.valid_start_range(T=200, warmup=190, cfg=cfg)


# ---------------------------------------------------------------------------
# The process socket (`contact_chol`)
# ---------------------------------------------------------------------------

def test_segment_contact_chol_passes_the_stance_swing_switch_through(preps):
    r"""By default the sim's stance/swing switch must reach the filter INTACT.

    Freezing it at the stance value pins swing feet as world-static and was
    measured at 10.2× worse body-frame velocity error than not using contacts at
    all (`experiments/measure_tstar.py`); it is what made run 1 drive
    ``Σ_C → ∞``.  Two halves, and the second is what makes the first mean
    anything: the source really does switch 1e-4 ↔ 1e1 inside the segment's tick
    range, and the segment reproduces it elementwise.
    """
    cfg = _cfg()
    assert not cfg.freeze_contact_chol, "pass-through must be the default"
    p = preps[0]
    t0 = (p.t_lo + p.t_hi) // 2
    src = p.inputs.contact_chol[t0:t0 + cfg.L]
    assert np.unique(np.round(src, 12)).size > 2, "fixture lost the stance/swing switch"

    seg = dataset.make_segment(p, t0, cfg, P0)
    assert seg.inputs.contact_chol.shape == (cfg.L, N_C, 3, 3)
    assert np.array_equal(seg.inputs.contact_chol, src)
    # Non-vacuity: a swing tick really is inside this segment and really is
    # swing-valued afterwards.  Without this, a loader that froze everything at
    # 1e1 would pass the equality above on an all-swing slice.
    swing = np.abs(src[:, :, 0, 0] - 1.0e1) < 1e-9
    assert swing.any(), "no swing tick inside the segment — the check is vacuous"
    i, j = np.argwhere(swing)[0]
    assert seg.inputs.contact_chol[i, j, 0, 0] == 1.0e1


def test_freeze_contact_chol_restores_run1_behaviour(preps):
    """The opt-in flag still destroys the switch, for the ablation."""
    cfg = _cfg(freeze_contact_chol=True)
    p = preps[0]
    t0 = (p.t_lo + p.t_hi) // 2
    src = p.inputs.contact_chol[t0:t0 + cfg.L]
    swing = np.abs(src[:, :, 0, 0] - 1.0e1) < 1e-9
    assert swing.any(), "no swing tick inside the segment — the check is vacuous"

    seg = dataset.make_segment(p, t0, cfg, P0)
    want = cfg.contact_chol_const * np.eye(3)
    assert np.array_equal(seg.inputs.contact_chol,
                          np.broadcast_to(want, (cfg.L, N_C, 3, 3)))
    i, j = np.argwhere(swing)[0]
    assert seg.inputs.contact_chol[i, j, 0, 0] == cfg.contact_chol_const


def test_every_other_input_field_is_the_untouched_contiguous_slice(preps):
    """No shuffling within a segment; by default NOTHING is rewritten."""
    cfg = _cfg()
    p = preps[0]
    t0 = p.t_lo + 11
    seg = dataset.make_segment(p, t0, cfg, P0)
    for name in InEKFInputs._fields:
        got, want = getattr(seg.inputs, name), getattr(p.inputs, name)
        for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            assert np.array_equal(np.asarray(g), np.asarray(w)[t0:t0 + cfg.L]), name
    # Non-vacuity: a rolled slice would NOT match, so "contiguous, in order" is
    # a real constraint on `omega` here rather than a property of a constant array.
    assert not np.array_equal(np.asarray(seg.inputs.omega),
                              np.roll(np.asarray(p.inputs.omega)[t0:t0 + cfg.L], 1, axis=0))


# ---------------------------------------------------------------------------
# state0
# ---------------------------------------------------------------------------

def test_state0_is_ground_truth_and_zeroes_the_first_contact_residual(preps):
    r"""``R, v, p`` are truth; ``d`` makes ``ν = R̄y − (d̄ − p̄)`` exactly zero.

    The residual identity is the point: it is what "contacts from the FK" has to
    mean for the seed to be consistent with the measurement the filter takes on
    the segment's first tick.  Checked against `inEKF.correct.innovation` — the
    filter's own function, not a re-derivation here.
    """
    cfg = _cfg()
    p = preps[0]
    t0 = p.t_lo + 5
    seg = dataset.make_segment(p, t0, cfg, P0)
    assert np.array_equal(seg.state0.R, p.R_true[t0])
    assert np.array_equal(seg.state0.v, p.v_true[t0])
    assert np.array_equal(seg.state0.p, p.p_true[t0])
    assert np.array_equal(seg.state0.P, P0)

    nu = np.asarray(innovation(jax.tree.map(jnp.asarray, seg.state0),
                               jnp.asarray(p.y_fk[t0])))
    assert np.abs(nu).max() < 1e-14
    # Non-vacuity: the same seed against a DIFFERENT tick's FK is not zero, so
    # the identity is about `t0` and not about y_fk being small.
    nu_other = np.asarray(innovation(jax.tree.map(jnp.asarray, seg.state0),
                                     jnp.asarray(p.y_fk[t0 + 300])))
    assert np.abs(nu_other).max() > 1e-6


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_batches_are_spread_across_rollouts(preps):
    r"""A batch touches ``min(B, n_rollouts)`` distinct rollouts, never one.

    Segments from one rollout share terrain, seed and one filter trajectory, so
    ``B`` of them is well under ``B`` independent samples.  Kills the obvious
    simplification (`rng.integers(n, size=B)`), which at n=4, B=4 draws all four
    from one rollout about 0.4% of the time and duplicates constantly.
    """
    rng = np.random.default_rng(0)
    n = len(preps)
    for B in (2, n, 2 * n, 2 * n + 1):
        picks = dataset.sample_starts(rng, preps, B)
        assert len(picks) == B
        counts = np.bincount([i for i, _ in picks], minlength=n)
        assert len(set(i for i, _ in picks)) == min(B, n)
        assert counts.max() - counts.min() <= 1        # no rollout over-represented
        for i, t0 in picks:
            assert preps[i].t_lo <= t0 <= preps[i].t_hi


def test_starts_are_random_not_tiled(preps):
    """Starts must not walk the trajectory in order or repeat between batches."""
    rng = np.random.default_rng(3)
    starts = [t for _ in range(40) for _, t in dataset.sample_starts(rng, preps, 4)]
    span = preps[0].t_hi - preps[0].t_lo
    assert len(set(starts)) > 0.9 * len(starts)        # essentially no repeats
    assert np.std(starts) > 0.2 * span                 # spread over the range
    diffs = np.diff(starts)
    assert not np.all(diffs == diffs[0])               # not a tiling


def test_batch_has_the_shapes_and_dtypes_the_loss_expects(preps):
    cfg = _cfg()
    batch = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=0)))
    assert batch.windows.shape == (cfg.B, cfg.L, N_C, cfg.H, F)
    assert batch.v_true.shape == (cfg.B, cfg.L, 3)
    assert batch.R_true.shape == (cfg.B, cfg.L, 3, 3)
    assert batch.state0.P.shape == (cfg.B, 9 + 3 * N_C)[:1] + (9 + 3 * N_C, 9 + 3 * N_C)
    assert batch.inputs.joint.sigma_q.shape == (cfg.B, cfg.L, N_J, N_J)
    bad = [x.dtype for x in jax.tree.leaves(batch) if x.dtype != jnp.float64]
    assert not bad, bad
    assert bool(jnp.all(jnp.isfinite(batch.windows)))


def test_batch_stream_is_reproducible_and_seed_dependent(preps):
    cfg = _cfg()
    a = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=7)))
    b = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=7)))
    c = next(iter(dataset.batch_stream(preps, cfg, P0, steps=1, seed=8)))
    assert np.array_equal(a.windows, b.windows)
    assert not np.array_equal(a.windows, c.windows)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_prepare_applies_the_frozen_constants(dataset_dir, norm):
    """At stride 1 the boxcar is the identity, so `smoothed` is exactly ``(x-m)/s``."""
    cfg = _cfg(window_span_s=0.004)          # -> stride 1
    assert cfg.stride == 1
    paths = dataset.rollout_paths(dataset_dir)
    p = dataset.prepare(paths[:1], norm, cfg, cache_dir=dataset_dir / "cache")[0]
    raw = dataset.load_channel_cache(dataset.cache_path(paths[0], dataset_dir / "cache"))
    want = (raw["channels"] - np.asarray(norm.mean)) / np.asarray(norm.std)
    assert np.allclose(p.smoothed, want, rtol=0, atol=1e-15)
    # Non-vacuity: raw != normalized, so this is not "the loader returned its input".
    assert not np.allclose(p.smoothed, raw["channels"])


def test_prepare_refuses_stale_constants(dataset_dir, norm):
    """A constants artifact whose channel set no longer matches must raise."""
    stale = normalize.NormConstants(
        mean=jnp.zeros(F - 1), std=jnp.ones(F - 1), names=norm.names[:-1],
        n_ticks=1, source="stale", floored=())
    with pytest.raises(ValueError):
        dataset.prepare(dataset.rollout_paths(dataset_dir)[:1], stale, _cfg(),
                        cache_dir=dataset_dir / "cache")


def test_fit_normalization_pools_only_the_usable_region(dataset_dir):
    r"""The calibration set is the post-warm-up union, and the skip is observable."""
    paths = dataset.rollout_paths(dataset_dir)
    cache = dataset_dir / "cache"
    got = dataset.fit_normalization(paths, cache_dir=cache, source="t")
    pooled = np.concatenate(
        [dataset.load_channel_cache(dataset.cache_path(p, cache))["channels"][WARMUP:]
         for p in paths], axis=0)
    assert np.allclose(np.asarray(got.mean), pooled.mean(axis=(0, 1)), atol=1e-12)
    assert got.n_ticks == pooled.shape[0] * N_C
    # The skip really moves the answer, so `usable_only` is not decorative.
    whole = dataset.fit_normalization(paths, cache_dir=cache, usable_only=False)
    assert not np.allclose(np.asarray(whole.mean), np.asarray(got.mean), atol=1e-9)


def test_fit_normalization_floors_a_channel_the_set_never_exercised(dataset_dir, tmp_path):
    r"""A dead channel must land in `floored` with its std at the noise floor.

    This is the gate the `floored` field exists for — the measured example is a
    standing-only calibration set flooring ``base_gyro_y/z``.  Reproduced here by
    freezing that channel and checking both that it is reported and that its std
    is the floor rather than its own ~0 variance.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    paths = dataset.rollout_paths(dataset_dir)[:1]
    c = dataset.load_channel_cache(dataset.cache_path(paths[0], dataset_dir / "cache"))
    x = c["channels"].copy()
    j = features.channel_names().index("base_gyro_y")
    x[:, :, j] = 1.234                      # frozen: the set never exercised it
    np.savez_compressed(cache / dataset.cache_path(paths[0]).name, channels=x,
                        y_fk=c["y_fk"], names=np.asarray(c["names"]),
                        meta=np.array(json.dumps(c["meta"])))
    got = dataset.fit_normalization(paths, cache_dir=cache, source="frozen-gyro-y")
    assert "base_gyro_y" in got.floored
    assert float(got.std[j]) == pytest.approx(normalize.NOISE_FLOOR["base_gyro"])
    assert float(got.mean[j]) == pytest.approx(1.234)
    assert "base_gyro_x" not in got.floored     # the live channels are untouched


# ---------------------------------------------------------------------------
# P0
# ---------------------------------------------------------------------------

def _fixture_kinematics(n_contacts=N_C, n_joints=N_J):
    """The `tests/inEKF/test_filter.py` analytic seam — smooth, jit-able, consistent."""
    offsets = jnp.asarray([[0.0, 0.1 * (-1) ** i, -0.9] for i in range(n_contacts)])
    weights = jnp.asarray([[0.1 * ((i + j) % 3 + 1) for j in range(n_joints)]
                           for i in range(n_contacts)])

    def kinematics(q, q_dot):
        y = offsets + (weights * jnp.sin(q)[None, :]) @ jnp.ones((n_joints, 3))
        J = jnp.einsum("ij,j->ij", weights, jnp.cos(q))[:, None, :] * jnp.ones((1, 3, 1))
        J_dot = -jnp.einsum("ij,j->ij", weights, jnp.sin(q) * q_dot)[:, None, :] \
            * jnp.ones((1, 3, 1))
        return ContactFrames(y=y, J=J, J_dot=J_dot)

    return kinematics


def test_measure_p0_converges_below_the_diffuse_prior(preps):
    r"""The burn-in must actually *converge* something, not hand back the prior.

    `ekf_mod.initialize`'s default prior is ``1.0·I``.  A burn-in that returned
    it unchanged (an unrun scan, a dropped carry) would be indistinguishable from
    a correct one by shape or PSD alone, so the assertion is that the observed
    blocks shrank by orders of magnitude while the filter stayed PSD.
    """
    fused = SimpleNamespace(ekf=ekf_mod.create(N_C, dt=1.0e-3),
                            kinematics=_fixture_kinematics())
    P = dataset.measure_p0(fused, preps[0], _cfg(), ticks=400, verbose=False)
    m = 9 + 3 * N_C
    assert P.shape == (m, m)
    assert np.allclose(P, P.T, atol=0)
    assert np.linalg.eigvalsh(P).min() > 0
    # position and the contact anchors are the directly-observed blocks
    assert np.diag(P)[6:9].max() < 1.0
    assert np.diag(P)[9:].max() < 1.0


def test_measure_p0_uses_the_constant_contact_chol(preps, monkeypatch):
    """The burn-in must run under training conventions, not the sim's ground truth.

    Kills a burn-in that forgot the `contact_chol` overwrite: the sim's swing
    value (1e1 ⇒ Σ = 100·I) inflates the contact process noise by six orders of
    magnitude against the frozen stance value, which the converged ``P`` sees.
    """
    fused = SimpleNamespace(ekf=ekf_mod.create(N_C, dt=1.0e-3),
                            kinematics=_fixture_kinematics())
    cfg = _cfg()
    P_const = dataset.measure_p0(fused, preps[0], cfg, ticks=400, verbose=False)

    seen = {}
    real = dataset._constant_contact_chol

    def spy(cfg_, L, N):
        seen["called"] = True
        return real(cfg_, L, N)

    monkeypatch.setattr(dataset, "_constant_contact_chol", spy)
    P_spy = dataset.measure_p0(fused, preps[0], cfg, ticks=400, verbose=False)
    assert seen.get("called"), "measure_p0 did not freeze contact_chol"
    assert np.array_equal(P_spy, P_const)          # the spy is a pass-through

    # And the frozen value genuinely changes the answer: re-run with the sim's
    # raw (switching) contact_chol.  The converged P is dominated by the
    # measurement, so the shift is small (~1e-4 relative here) — but it is eight
    # orders above float64 noise, which is what makes this a discriminator and
    # not a coincidence.
    monkeypatch.setattr(dataset, "_constant_contact_chol",
                        lambda cfg_, L, N: np.asarray(
                            preps[0].inputs.contact_chol[preps[0].t_lo:preps[0].t_lo + L]))
    P_sim = dataset.measure_p0(fused, preps[0], cfg, ticks=400, verbose=False)
    rel = np.abs(P_sim - P_const).max() / np.abs(P_const).max()
    assert rel > 1e-8, f"freezing contact_chol changed nothing (rel {rel:.2e})"
