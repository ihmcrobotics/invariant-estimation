"""The ContactNet data collector (`invariant_estimation.sim.collect`).

The failure modes this module has are all *silent*: a stream sampled at 50 Hz instead of 1 kHz,
two terrains that are secretly the same terrain, a `.npz` that loses the last bit of a float, an
`inekf_inputs` that got rebuilt from the wrong pieces. Every one of those produces a dataset with
the right shapes that trains a worse network with nothing raising. So the tests here are written
against that: each one is paired with a MUTANT (recorded in its docstring) that it was checked to
catch, per the discipline in `estimator-port-testing-discipline`.

Building the Alex estimator traces MJX over 49 links and compiling the scan costs ~20 s, so the
collector fixture is module-scoped and this module is marked `slow`. Rollouts here are 1-2 s.
"""

import json

import numpy as np
import pytest

from invariant_estimation.sim import collect as co
from invariant_estimation.sim import terrain as tr

pytestmark = pytest.mark.slow

SECONDS = 1.0
SETTLE = 0.5
TICKS = int(round((SECONDS + SETTLE) / co.CONTROL_DT)) * 20      # DECIMATION = 20


@pytest.fixture(scope="module")
def collector():
    return co.build_collector(chunk_ticks=1000, verbose=False)


@pytest.fixture(scope="module")
def roll(collector, tmp_path_factory):
    return co.collect_rollout("waves", seed=0, seconds=SECONDS, settle_s=SETTLE,
                              collector=collector, out_dir=tmp_path_factory.mktemp("data"),
                              verbose=False)


# ---------------------------------------------------------------------------
# Shape, rate, dtype
# ---------------------------------------------------------------------------

def test_every_leaf_is_float64_with_the_full_time_axis(roll):
    """I8 at the boundary, and a time axis at the PHYSICS rate on every leaf."""
    leaves = co._leaves(roll)
    assert leaves, "no arrays collected"
    for k, v in leaves.items():
        assert v.shape[0] == TICKS, f"{k}: T={v.shape[0]}, expected {TICKS}"
        assert v.dtype == np.float64, f"{k}: {v.dtype}"
        assert np.all(np.isfinite(v)), f"{k}: non-finite"


def test_sampling_is_at_the_physics_rate(roll):
    """The trap this module exists to avoid: sampling once per `control_tick`, not per `mj_step`.

    A control-rate stream up-sampled to `T` would come in runs of `DECIMATION` identical samples.
    Assert on the ADJACENT-SAMPLE differences rather than on `T` alone, which any duplication
    scheme satisfies.

    MUTANT: move `reader.read(d)` out of `_RecordingLoop`'s decimation loop and repeat the sample
    20x -> `dup` becomes 0.95 and this fails. (`T` alone still passes, which is the point.)
    """
    enc = roll.sensors.encoders
    dup = np.mean(np.all(np.diff(enc, axis=0) == 0.0, axis=1))
    assert dup < 0.01, f"{dup:.1%} of consecutive encoder samples are identical"
    # And the stream must actually resolve intra-tick motion: within one control period the joints
    # move by much more than the encoder noise floor (2e-4 rad).
    within = np.abs(np.diff(enc[TICKS // 2: TICKS // 2 + 20], axis=0)).max()
    assert within > 1e-3, f"nothing moves inside a control period (max |dq| = {within:.2e})"


def test_the_recorded_inekf_inputs_are_what_the_filter_consumed(roll, collector):
    """`inekf_inputs.joint.sigma_q` is the one field that cannot be rebuilt downstream.

    Check it is a real, evolving, symmetric PSD covariance of the widened joint vector (9 filtered
    + 4 off-path ankles) -- not zeros, not constant, not the 9x9 un-widened block.

    NOT mutation-checked: the mutants for this one (zeroing `sigma_q`, dropping the
    `q_unfiltered` widening) live in `pipeline/main_estimator._boundary`, which this task was
    not permitted to edit. The assertions are written to fail under both, but that is an argument,
    not a measurement -- treat this test as weaker than the others in this file.
    """
    S = roll.inputs.joint.sigma_q
    n_wide = collector.fused.n_joints + collector.fused.n_aux
    assert S.shape[1:] == (n_wide, n_wide), S.shape
    assert np.allclose(S, np.swapaxes(S, 1, 2), atol=0), "sigma_q is not symmetric"
    w = np.linalg.eigvalsh(S)
    assert w.min() > -1e-15, f"sigma_q is not PSD (min eig {w.min():.2e})"
    assert w.max() > 0.0
    # It is a live covariance, not a constant seeded matrix.
    assert np.abs(np.diff(np.diagonal(S, axis1=1, axis2=2), axis=0)).max() > 0.0
    # The off-path block is the ANKLES' encoder variance, so it is diagonal and constant while
    # the filtered block is coupled and moving -- a permuted or truncated widening breaks this.
    off = S[:, collector.fused.n_joints:, collector.fused.n_joints:]
    assert np.allclose(off, off[0]), "the off-path block should be a constant encoder variance"
    assert np.count_nonzero(off[0] - np.diag(np.diagonal(off[0]))) == 0


def test_the_bias_and_the_estimate_are_alive(roll):
    """Aux diagnostics: the filter ran, converged toward the truth, and published.

    The attitude comparison is against `truth["R"]`, which comes from `MjData` down an entirely
    separate path from the estimate -- so this cannot pass by comparing the pipeline with itself.
    (Covered transitively by the M5 sampling mutant, which breaks the sensor stream and takes
    this test with it; no dedicated mutant was run for it.)
    """
    assert roll.aux["bias"].shape[1] == 24                      # 8 IMUs x 3
    assert np.abs(np.diff(roll.aux["bias"], axis=0)).max() > 0.0, "the bias state never moved"
    # Attitude tracks truth: the estimator is genuinely running on these sensors.
    c = np.einsum("tij,tij->t", roll.aux["est_R"][-100:], roll.truth["R"][-100:])
    ang = np.degrees(np.arccos(np.clip((c - 1.0) / 2.0, -1.0, 1.0)))
    assert ang.mean() < 5.0, f"attitude error {ang.mean():.1f}deg -- the estimator is not tracking"


def test_the_recording_loop_is_the_same_sim_as_run_policys(collector):
    """`_RecordingLoop.control_tick` is a COPY of `rp.Loop.control_tick` (no hook exists inside
    its decimation loop). This is the test that notices when the copy drifts from the original.

    Bit-identical, not close: both are the same deterministic ONNX policy on the same MjData, and
    recording sensors cannot change the dynamics. Anything less than exact means the copy diverged.

    MUTANT: drop the `self.d.ctrl[ALL_AID] = ALL_HOME` line (the undriven joints hold home) from
    the copy -> the trajectories separate within a few ticks and this fails.
    """
    import run_policy as rp

    from invariant_estimation.sim.sensors import SimSensorReader

    field = co.terrain_field("flat", 0)
    m = rp.build_sim_model(collector.policy, with_visuals=False, with_imu_sensors=True,
                           floor=tr.HeightfieldFloor(field))
    maps = rp.make_maps(m, collector.policy)
    reader = SimSensorReader(m, collector.fused, foot_geoms=rp.FOOT_GEOMS, dt=collector.dt)
    ref = rp.Loop(m, collector.policy, maps)
    rec = co._RecordingLoop(m, collector.policy, maps, reader)
    for _ in range(20):
        ref.cmd[0:3] = rec.cmd[0:3] = (0.4, 0.0, 0.0)
        ref.cmd[3] = rec.cmd[3] = 0.0
        ref.control_tick()
        rec.control_tick()
    assert np.array_equal(ref.d.qpos, rec.d.qpos), "the recording loop is a different sim"
    assert np.array_equal(ref.d.qvel, rec.d.qvel)


# ---------------------------------------------------------------------------
# The two checks that compare against something INDEPENDENT
# ---------------------------------------------------------------------------

def test_different_terrains_produce_different_data(collector, tmp_path):
    """Two terrains at the SAME seed must differ -- everywhere, not just in a header field.

    The cheap version of this test (compare `meta["terrain"]`) passes even if the floor argument
    is dropped entirely and every rollout runs on a plane.

    MUTANT: `field = tr.flat()` for every terrain -- the hfield-vs-field consistency check still
    passes (it compares the model against the same flat field), the labels are still right, and
    the encoder streams become bit-identical: caught, `max|dq| = 0.0`.
    """
    kw = dict(seconds=SECONDS, settle_s=SETTLE, collector=collector, out_dir=None, verbose=False)
    a = co.collect_rollout("flat", seed=0, **kw)
    b = co.collect_rollout("hard_stepping", seed=0, **kw)

    # Same spawn, same policy, same noise seed: the ONLY difference is the ground.
    assert a.meta["spawn_xy_yaw"] == b.meta["spawn_xy_yaw"]
    assert np.abs(a.sensors.encoders - b.sensors.encoders).max() > 1e-3
    assert np.abs(a.truth["p"][:, 2] - b.truth["p"][:, 2]).max() > 5e-3   # a 7 cm relief field
    assert np.abs(a.inputs.joint.sigma_q - b.inputs.joint.sigma_q).max() > 0.0
    # And the terrain each one actually stood on is different where the robot walked.
    fa, fb = co.terrain_field("flat", 0), co.terrain_field("hard_stepping", 0)
    x, y = a.truth["p"][:, 0], a.truth["p"][:, 1]
    assert np.abs(tr.sample(fa, x, y) - tr.sample(fb, x, y)).max() > 1e-3


def test_seeds_change_the_data_and_the_stone_pattern():
    """Rollout `seed` must reach BOTH the terrain rasteriser and the spawn/noise.

    MUTANT: drop the `1000 * seed` offset in `terrain_field` -> caught (identical stone fields).
    MUTANT: make `spawn_pose` seed-independent -> caught.
    """
    f0, f1 = co.terrain_field("stepping_stones", 0), co.terrain_field("stepping_stones", 1)
    assert np.abs(f0 - f1).max() > 1e-3, "the stone pattern did not change with the seed"
    assert co.spawn_pose(0) != co.spawn_pose(1)
    # seed 0 must still reproduce the registry field exactly (the Stage-1 comparability claim).
    assert np.array_equal(co.terrain_field("stepping_stones", 0), tr.TERRAINS["stepping_stones"]())
    # flat has nothing to randomise -- the docstring says so; assert it rather than implying it.
    assert np.array_equal(co.terrain_field("flat", 0), co.terrain_field("flat", 7))


def test_the_npz_round_trips_bit_identically(roll, tmp_path):
    """Save -> load -> compare every leaf with `array_equal`, not `allclose`.

    MUTANT: `arrays[k].astype(np.float32)` in `save_rollout` -> exact equality fails while
    `allclose(atol=1e-6)` would still pass, which is exactly why this is exact.
    """
    p = co.save_rollout(roll, tmp_path / "r.npz")
    back = co.load_rollout(p)
    a, b = co._leaves(roll), co._leaves(back)
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} changed across the round trip"
    assert back.meta == roll.meta
    assert json.dumps(back.meta)          # metadata survives as JSON, not as a pickle
    # The typed structure survives, not just the numbers: the trainer takes an `InEKFInputs`.
    assert type(back.inputs) is type(roll.inputs)
    assert type(back.inputs.joint) is type(roll.inputs.joint)
    assert type(back.sensors) is type(roll.sensors)


def test_chunking_is_exact(collector, roll):
    """Chunked `run_fused` == one long scan, because the filter carry crosses the boundary.

    MUTANT: re-seed the carry at each chunk (`init_fused_carry` inside the loop) -> caught.
    """
    T = roll.sensors.encoders.shape[0]
    assert T % collector.chunk_ticks != 0 or T // collector.chunk_ticks > 1, \
        "fixture must exercise more than one chunk"
    carry = _seed_like(collector, roll)
    one, _, _ = co._run_fused_chunked(_with_chunk(collector, T), carry, roll.sensors, T)
    many, _, _ = co._run_fused_chunked(_with_chunk(collector, 137), carry, roll.sensors, T)
    assert np.array_equal(one.joint.sigma_q, many.joint.sigma_q)
    assert np.array_equal(one.omega, many.omega)


def _with_chunk(c, n):
    return co.Collector(policy=c.policy, fused=c.fused, dt=c.dt, chunk_ticks=n,
                        policy_name=c.policy_name, _scan=c._scan)


def _seed_like(collector, roll):
    """The same carry `collect_rollout` seeded, rebuilt from the recorded first tick."""
    import jax.numpy as jnp

    from invariant_estimation.pipeline import main_estimator as me
    return me.init_fused_carry(
        collector.fused,
        q0=jnp.asarray(roll.sensors.encoders[0]),
        rotation=jnp.asarray(roll.truth["R"][0]),
        position=jnp.asarray(roll.truth["p"][0]),
        q0_unfiltered=jnp.asarray(roll.sensors.q_unfiltered[0]))


# ---------------------------------------------------------------------------
# Guards: they must actually fire
# ---------------------------------------------------------------------------

def test_the_preflight_refuses_a_rollout_that_cannot_fit():
    """A 100 s rollout at 0.4 m/s does not fit on a 64 m field, and must be refused BEFORE the
    compute is spent -- not discovered by a robot standing on the clamped edge extrusion.

    No collector needed: this must raise before anything is built.
    """
    with pytest.raises(ValueError, match="past the"):
        co.collect_rollout("flat", seed=0, seconds=100.0, out_dir=None, verbose=False)


def test_the_upright_and_on_field_guards_fire():
    """The guards themselves, against fabricated trajectories with a KNOWN answer.

    The pre-flight check makes the off-field branch unreachable from a real short rollout, so it
    is exercised here directly rather than left untested. A clean trajectory must pass -- a guard
    that raises on everything is not a guard.

    MUTANT: the tilt `raise` becomes a `print` -> caught.
    MUTANT: `tilt.max()` -> `tilt.mean()` -> caught; the 1-tick stumble below is exactly the case
    a mean-based guard sleeps through, and a stumble is what ruins a rollout.
    """
    T = 100
    R = np.tile(np.eye(3), (T, 1, 1))
    p = np.zeros((T, 3))
    good = dict(R=R, p=p)
    assert co._check_rollout(good, max_tilt_deg=15.0, limit=30.0, label="ok").max() == 0.0

    stumble = {"R": R.copy(), "p": p}
    a = np.radians(20.0)
    stumble["R"][57] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    with pytest.raises(RuntimeError, match="tilt reached"):
        co._check_rollout(stumble, max_tilt_deg=15.0, limit=30.0, label="x")

    off = {"R": R, "p": p.copy()}
    off["p"][:, 1] = np.linspace(0.0, 31.0, T)
    with pytest.raises(RuntimeError, match="left the heightfield"):
        co._check_rollout(off, max_tilt_deg=15.0, limit=30.0, label="x")


def test_the_upright_guard_fires_end_to_end(collector):
    """And the guard is actually WIRED into `collect_rollout`, not just defined next to it."""
    with pytest.raises(RuntimeError, match="tilt reached"):
        co.collect_rollout("flat", seed=0, seconds=SECONDS, settle_s=SETTLE, collector=collector,
                           max_tilt_deg=0.05, out_dir=None, verbose=False)


def test_nothing_is_saved_when_a_guard_fires(collector, tmp_path):
    """A failed rollout must leave NO file -- a half-written dataset is worse than none."""
    with pytest.raises(RuntimeError):
        co.collect_rollout("flat", seed=0, seconds=SECONDS, settle_s=SETTLE, collector=collector,
                           max_tilt_deg=0.05, out_dir=tmp_path, verbose=False)
    assert list(tmp_path.glob("*.npz")) == []


# ---------------------------------------------------------------------------
# End to end: the consumer
# ---------------------------------------------------------------------------

def test_the_collected_stream_feeds_contactnets_channels(collector, roll):
    """`make_contact_channels` over collected data: right shape, finite, physically right.

    The physical numbers are the teeth. `p_bc_z ~ -0.9 m` (the sole is below the pelvis) and
    `base_accel_z ~ +9.8 m/s^2` at rest (specific force, not acceleration) both fail loudly under
    a permuted subchain gather or a sign/frame error, which a shape assertion cannot see.

    MUTANT: build the collector with `contact_fk_unfiltered=False` -> `q_unfiltered` is empty,
    `make_contact_channels` raises, and the 4 missing ankle channels are caught rather than
    silently absent.
    """
    settle = roll.meta["settle_ticks"]
    rep = co.channel_report(collector, roll.sensors, rest=slice(settle - 200, settle))
    assert rep["shape"] == (TICKS, collector.fused.n_contacts, 24)
    assert rep["finite"]
    assert -1.05 < rep["p_bc_z_mean"] < -0.75, rep["p_bc_z_mean"]
    assert 9.0 < rep["base_accel_z_rest"] < 10.6, rep["base_accel_z_rest"]
    assert rep["base_gyro_norm_rest"] < 0.15, rep["base_gyro_norm_rest"]
    assert rep["v_bc_absmax"] < 20.0            # m/s of foot-relative-to-base speed
    assert 1.0 < rep["tau_knee_absmax"] < 400.0  # N.m; Alex's standing knee torque is ~51


def test_chunked_channels_match_one_pass(collector, roll):
    """Chunking the feature path must change nothing that matters.

    `make_contact_channels` vmaps the MJX FK over the whole time axis and does not survive a 60 s
    rollout (~38 GB), so `channel_report` splits it. The only cross-tick term is the causal
    difference `v`, handled with a one-tick lead-in.

    Bit-identity is asserted where it is achievable and is NOT achievable on the FK channels:
    measured, XLA's batched FK differs by up to 1 ulp of a ~0.9 m position depending on the vmap
    width, and `v = dp/dt` amplifies that by 1/dt = 1000 -> 2.2e-13. So: exact on the 18
    non-FK channels, 1e-11 on the 6 FK ones. That is still 11 orders of margin on the mutant.

    MUTANT: drop the lead-in (`start = lo`) -> `v` is 0 at every chunk boundary, an O(0.1-2 m/s)
    error on 1/257 of the rows -- caught, and it is the artifact that would otherwise ride
    silently into training.
    """
    from invariant_estimation.contactnet.features import (build_subchain_indices,
                                                          make_contact_channels)

    sub = build_subchain_indices(collector.fused.build.joint_names,
                                 co._unfiltered_names(collector))
    chan = make_contact_channels(sub, collector.fused.base_imu, collector.fused.kinematics,
                                 collector.dt)
    whole = co.contact_channels_chunked(chan, roll.sensors, chunk=10 ** 9)
    split = co.contact_channels_chunked(chan, roll.sensors, chunk=257)
    assert whole.shape == split.shape == (TICKS, collector.fused.n_contacts, 24)
    fk = slice(18, 24)                      # p_bc_{x,y,z}, v_bc_{x,y,z}
    assert np.array_equal(np.delete(whole, np.r_[fk], axis=2),
                          np.delete(split, np.r_[fk], axis=2))
    assert np.abs(whole[..., fk] - split[..., fk]).max() < 1e-11
    # The boundary rows are where a missing lead-in would show, so pin them on their own.
    b = np.arange(257, TICKS, 257)
    assert np.abs(whole[b] - split[b]).max() < 1e-11
    assert np.abs(split[b, :, 21:24]).max() > 1e-3, "v is zero at the boundaries -- no lead-in"


def test_contact_chol_carries_the_sims_contact_truth(roll):
    """A documented hazard, asserted so it cannot silently stop being true.

    The saved `inputs.contact_chol` is stance/swing-switched from the SIM's contact state. A
    training segment that feeds it to the filter hands the network a ground-truth contact flag,
    which is why `contactnet.rollout.Segment` says to constant it. Assert the switching is really
    there, so the hazard note in the module docstring stays honest.
    """
    d = np.diagonal(roll.inputs.contact_chol, axis1=-2, axis2=-1)
    assert d.max() > 1.0 and d.min() < 1e-3, "contact_chol never switches -- check the reader"
    assert np.array_equal(roll.inputs.contact_chol, roll.sensors.contact_chol)


def test_collect_all_writes_one_file_per_rollout_and_survives_a_failure(collector, tmp_path):
    """The driver: files land, names are unique, and one bad rollout does not kill the run.

    `max_tilt_deg=0.05` makes EVERY rollout fail, which is the case that matters -- a driver that
    propagates the first `RuntimeError` loses a whole overnight collection to one stumble.
    """
    kw = dict(seconds=0.5, settle_s=0.5, collector=collector, verbose=False)
    metas = co.collect_all(["flat", "waves"], seeds=(0,), seconds=0.5, out_dir=tmp_path,
                           settle_s=0.5, collector=collector, verbose=False)
    assert len(metas) == 2
    assert {p.name for p in tmp_path.glob("*.npz")} == {"flat_seed000.npz", "waves_seed000.npz"}
    assert co.load_rollout(tmp_path / "flat_seed000.npz").meta["terrain"] == "flat"

    bad = tmp_path / "bad"
    assert co.collect_all(["flat"], seeds=(0, 1), out_dir=bad, max_tilt_deg=0.05, **kw) == []
    assert not bad.exists() or list(bad.glob("*.npz")) == []


# ---------------------------------------------------------------------------
# (A) the warm-up criterion
# ---------------------------------------------------------------------------

def test_bias_plateau_finds_a_known_transient():
    """The criterion, against a synthetic bias with a KNOWN settling time -- not against itself.

    An exponential that reaches its final value at 3 s must plateau within a window of 3 s, and a
    signal that never settles must not report that it did.

    MUTANT: normalise by `||b_inf||` instead of the excursion. Caught only by the OFFSET case
    below -- a transient that starts at zero has `||b_inf|| == ||b_inf - b_0||`, so the obvious
    test cannot tell the two criteria apart, and the first version of this test could not.
    """
    dt, T = 1.0e-3, 20_000
    t = np.arange(T) * dt
    b = np.tile((1.0 - np.exp(-t / 0.6))[:, None], (1, 3)) * 0.01     # settles by ~3 s
    p = co.bias_plateau(b, dt)
    assert 1.5 < p["drift"] * dt < 4.0, p
    assert 1.5 < p["settle"] * dt < 4.0, p

    # Same transient riding on a large constant offset: the excursion is 1% of the final value,
    # so a criterion scaled by ||b_inf|| calls it settled at tick 0 while it is still moving.
    off = b + 1.0
    q = co.bias_plateau(off, dt)
    assert 1.5 < q["drift"] * dt < 4.0, q
    assert abs(q["scale"] - p["scale"]) < 1e-9, "the criterion must not see the constant offset"

    # A ramp that never stops moving: the drift criterion must run to the end of the record.
    ramp = np.tile((0.01 * t / t[-1])[:, None], (1, 3))
    assert co.bias_plateau(ramp, dt)["drift"] > int(0.9 * T)

    # Degenerate: a bias that starts at its final value (no noise -> nothing to estimate).
    z = co.bias_plateau(np.zeros((T, 3)), dt)
    assert z["degenerate"] and z["drift"] == 0


def test_the_measured_warmup_is_recorded_but_nothing_is_dropped(roll):
    """The discard is METADATA. The saved stream starts at t=0, including the settle."""
    assert roll.meta["warmup_ticks"] == co.WARMUP_TICKS
    assert roll.meta["settle_ticks"] == int(round(SETTLE / co.CONTROL_DT)) * 20
    assert roll.sensors.encoders.shape[0] == TICKS       # nothing sliced off on the way out
