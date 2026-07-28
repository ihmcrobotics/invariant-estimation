"""`sim/collect.py` — ContactNet training data from terrain-randomised MuJoCo rollouts.

One rollout is: walk the baseline policy across one of `terrain.TERRAINS` for `seconds`,
recording **every physics tick** (1 kHz), then run the fused estimator once over the whole
recorded sensor stream and save

    FusedSensors      what the plant produced          -> ContactNet's feature channels
    InEKFInputs       what the InEKF actually consumed -> `contactnet.rollout.Segment.inputs`
    ground truth      v (world), R, p, omega, q, q̇    -> the L2 objective and the segment reseed

to one `.npz` under `data/` (gitignored).

Three things here are load-bearing and each has a test in `tests/sim/test_collect.py`:

1. **Sampling is at the PHYSICS rate, not the control rate.** `run_policy.Loop.control_tick`
   runs `DECIMATION = 20` `mj_step`s per policy query; a naive `for tick: read(d)` around it
   samples at 50 Hz and then *looks* right — the arrays have a time axis, the filter runs, the
   loss goes down, and every window the network ever sees is 20x too coarse. So `_RecordingLoop`
   samples inside the decimation loop, and `test_sampling_is_at_the_physics_rate` fails on any
   stream whose samples come in runs of 20 identical values.
2. **`inekf_inputs` is recorded, never rebuilt.** `FusedOutputs.inekf_inputs` carries
   `joint.sigma_q`, which comes off `jkf_carry.state.P` — `run_fused` keeps only the FINAL
   carry, so a consumer cannot reconstruct it without duplicating `_boundary`. Recording it is
   the whole reason the field exists.
3. **The estimator runs OPEN LOOP on a ground-truth-driven sim.** The policy reads `MjData`
   (plain `run_policy.Loop`); the filter never touches the gait. That is deliberate: the
   training data must not depend on the filter that ContactNet is about to change, or every
   retrain shifts its own dataset. `run_estimator.py` is the closed-loop counterpart.

Three facts that bite, all measured (see `bias_plateau`, `WARMUP_TICKS` and `__main__ --measure`):

* The gyro-bias state needs a **warm-up**, and until it settles the joint KF's `Σ_q` — hence the
  InEKF's contact `N = J Σ_q Jᵀ` — is a transient that never occurs on hardware, where the filter
  has been running for minutes. Every rollout records `meta["warmup_ticks"]`; slice it off.
  Nothing here silently discards data: the full stream is saved and the discard is metadata.
* **Two of the eight bias states do not converge to the truth, and that is not this module's
  bug — but it is what the warm-up number is measuring, so it is recorded here.** On a 62 s flat
  rollout with a 0.042 rad/s injected bias, six IMUs land within 0.007–0.025 rad/s of their true
  bias while `left_shin_imu` and `right_shin_imu` settle 0.21 rad/s off, which is 97% of the
  whole-vector error and all of the slow tail after ~16 s. The BASE IMU — the only bias the InEKF
  consumes (I1) — converges to 0.007 rad/s (0.4 deg/s), and `Σ_q` itself plateaus in 3.2 s, so
  the collected data is usable; but a shin-bias state absorbing 5x the injected bias is a joint-KF
  observability question worth someone's attention, not a settled one.
* `contact_chol` in the saved inputs is the **sim's contact truth** (stance ⇒ small, swing ⇒
  large, `SimSensorReader.stance_chol/swing_chol`). It is passed through the fused step untouched
  and does not influence anything else that is recorded, but a training segment must overwrite it
  with a constant or the network gets a free ground-truth contact flag
  (`contactnet.rollout.Segment.inputs`, which says exactly this).

    uv run python -m invariant_estimation.sim.collect --seconds 60 --seeds 0 1 2
    uv run python -m invariant_estimation.sim.collect --measure     # (A) warm-up, (B) throughput
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

# `jax_enable_x64` is flipped at package import and MUST precede any array construction (I8).
import invariant_estimation  # noqa: F401  (import for the side effect, not the name)
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from ..pipeline import main_estimator as me
from ..pipeline.main_estimator import FusedSensors
from ..inEKF.filter import InEKFInputs, JointFilterOutput
from . import terrain as tr
from .sensors import IMUNoise, SimSensorReader

__all__ = [
    "DATA_DIR", "Rollout", "Collector", "build_collector", "terrain_field", "spawn_pose",
    "collect_rollout", "collect_all", "save_rollout", "load_rollout",
    "bias_plateau", "channel_report",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

# `run_policy` is a root-level script, not a package module (same accommodation `terrain.py` and
# pytest's `pythonpath = ["."]` make).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import run_policy as rp  # noqa: E402

CONTROL_DT = rp.DT * rp.DECIMATION      # 0.02 s

# Defaults for a collection run. `SETTLE_S` is the stand-still before the walk command; it is
# RECORDED (see the module docstring — nothing is silently dropped) and counted into the warm-up.
SETTLE_S = 2.0
SPAWN_RADIUS = 3.0        # m, half-width of the spawn box about the field centre
FIELD_MARGIN = 2.0        # m of hfield that must remain unused at the end of a rollout
MAX_TILT_DEG = 15.0


# ---------------------------------------------------------------------------
# Terrain / spawn randomisation
# ---------------------------------------------------------------------------

def terrain_field(name: str, seed: int = 0) -> np.ndarray:
    """The `(N, N)` elevation field for `name`, re-seeded for rollout `seed`.

    `seed = 0` reproduces the `TERRAINS` registry entry exactly (so a collection run is
    comparable with the Stage-1 walk that recorded the terrain-is-real numbers).

    Only the stone terrains carry an RNG. `flat` has nothing to randomise and `waves` is a fixed
    sinusoid, so for those two the per-rollout variation comes entirely from the **spawn pose**
    (a different phase of the wave, a different heading) and the IMU-noise seed — which is worth
    stating plainly rather than implying four independently-randomised terrain families.
    """
    if name not in tr.TERRAINS:
        raise KeyError(f"unknown terrain {name!r}; have {list(tr.TERRAINS)}")
    fn = tr.TERRAINS[name]
    kw = dict(fn.keywords)
    if "seed" in inspect.signature(fn.func).parameters:
        # Offset by the registry's own seed so `seed=0` is the registry field and so two stone
        # terrains at the same rollout seed do not share a stone pattern.
        kw["seed"] = int(kw.get("seed", 0)) + 1000 * int(seed)
    return fn.func(**kw)


def spawn_pose(seed: int, *, radius: float = SPAWN_RADIUS) -> tuple[float, float, float]:
    """`(x, y, yaw)` for rollout `seed`: a box about the field centre and a free heading.

    The heading matters more than the position: the policy walks +x in its OWN frame, so a random
    yaw is what makes two rollouts on the same (deterministic) `waves` field traverse different
    ground rather than the same 24 m twice.
    """
    r = np.random.default_rng(0xC0FFEE + int(seed))
    x, y = r.uniform(-radius, radius, 2)
    return float(x), float(y), float(r.uniform(-np.pi, np.pi))


# ---------------------------------------------------------------------------
# The collector (heavy, built once, reused across every rollout)
# ---------------------------------------------------------------------------

@dataclass
class Collector:
    """Policy + fused estimator + compiled scan. Independent of the floor, so it is built once.

    The estimator is assembled from the URDF and knows nothing about the terrain — only the sim
    `MjModel` changes between rollouts — so the MJX trace and XLA compile of the scanned step
    (measured: ~30 s, and it lands on the FIRST chunk of the first rollout, not in
    `build_collector`, which returns in under a second) are paid once for a whole collection run
    rather than once per rollout. Reuse one `Collector` for everything.
    """

    policy: dict
    fused: me.FusedEstimator
    dt: float
    chunk_ticks: int
    policy_name: str
    build_s: float = 0.0
    _scan: object = None

    def scan(self):
        """`(carry, sensors) -> (carry, FusedOutputs)`, jitted, one compiled shape per chunk."""
        if self._scan is None:
            step = me.make_fused_step(self.fused)
            self._scan = jax.jit(lambda c, xs: jax.lax.scan(step, c, xs))
        return self._scan


def build_collector(policy_name: str = "baseline", *, dt: float = rp.DT,
                    chunk_ticks: int = 10_000, contact_meas_var: float = 0.0,
                    verbose: bool = True) -> Collector:
    """Load the policy and build the fused estimator.

    `contact_fk_unfiltered=True` is NOT optional here: without it `FusedSensors.q_unfiltered` is
    empty, the contact FK stands on `qpos0` ankles, and ContactNet's `q_ankle_*`/`tau_ankle_*`
    channels (4 of its 24) do not exist. `contactnet.features.make_contact_channels` raises rather
    than silently producing a narrower feature vector, which is the behaviour that catches it.

    `contact_meas_var = 0.0` is deliberate and is the thing ContactNet replaces: it is the
    isotropic floor `_boundary` folds into `Σ_q`, i.e. a hand-tuned stand-in for exactly the
    quantity the network is being trained to predict. Collecting with it nonzero would train the
    network against a target that already contains a constant version of itself.
    """
    t0 = time.time()
    policy = rp.load_policy(policy_name)
    fused = me.build_alex_fused_estimator_from_urdf(
        rp.cycloid_forearm_urdf(rp.URDF), dt=dt,
        contact_meas_var=contact_meas_var, contact_fk_unfiltered=True)
    c = Collector(policy=policy, fused=fused, dt=dt, chunk_ticks=int(chunk_ticks),
                  policy_name=policy_name, build_s=time.time() - t0)
    if verbose:
        print(f"collector: {fused.n_joints} filtered joints, {fused.build.n_imus} IMUs, "
              f"{fused.n_contacts} contacts, {fused.n_aux} off-path joints, dt={dt} "
              f"({1 / dt:.0f} Hz)  [built in {c.build_s:.1f}s]")
    return c


# ---------------------------------------------------------------------------
# The recording loop
# ---------------------------------------------------------------------------

class _RecordingLoop(rp.Loop):
    """`run_policy.Loop` that samples the sensors after EVERY `mj_step`.

    `control_tick` is reimplemented rather than wrapped because the base class has no hook inside
    its decimation loop — the same accommodation `run_estimator.EstimatedLoop` makes, and the
    policy/actuator lines are a verbatim copy of `rp.Loop.control_tick`. The observation is built
    from `MjData`, i.e. ground truth: this loop does not close the estimator into the gait.
    """

    def __init__(self, m, policy, maps, reader: SimSensorReader):
        super().__init__(m, policy, maps)
        self.reader = reader
        self.sensors: list = []
        self.truth: list = []
        self.sim_s = 0.0        # wall time in mj_step + policy
        self.read_s = 0.0       # wall time in sensor extraction (collector overhead)

    def control_tick(self):
        t0 = time.perf_counter()
        self.cmd[4] = self._height()
        obs = rp.build_obs(self.m, self.d, self.policy, self.maps, self.cmd, self.last_action)
        self.last_action = self.sess.run(
            None, {self.sess.get_inputs()[0].name: obs[None]})[0][0]
        self.d.ctrl[self.maps["ALL_AID"]] = self.maps["ALL_HOME"]
        self.d.ctrl[self.maps["AID"]] = self.maps["HOME"] + self.scale * self.last_action
        for _ in range(rp.DECIMATION):
            mujoco.mj_step(self.m, self.d)
            t1 = time.perf_counter()
            self.sim_s += t1 - t0
            # One read per physics tick, in order: `SimSensorReader` owns the contact-trust state
            # machine, so calling it twice on one tick (or skipping one) silently changes the
            # Schmitt/dwell trajectory that the joint KF's stance anchors ride on.
            self.sensors.append(self.reader.read(self.d))
            self.truth.append(self.reader.truth(self.d))
            t0 = time.perf_counter()
            self.read_s += t0 - t1
        self._ramp_t += rp.DECIMATION * rp.DT


# ---------------------------------------------------------------------------
# One rollout
# ---------------------------------------------------------------------------

class Rollout(NamedTuple):
    """One collected rollout, in NumPy, every leaf with a leading time axis of length `T`."""

    sensors: FusedSensors     # the plant boundary, as ContactNet's features consume it
    inputs: InEKFInputs       # what the InEKF actually consumed (incl. joint.sigma_q)
    truth: dict               # R (T,3,3), v/p/omega (T,3), q/q_dot (T,n)
    aux: dict                 # bias (T,3m), nis (T,), est_R/est_v/est_p — filter health
    meta: dict


def collect_rollout(
    terrain_name: str,
    seed: int = 0,
    seconds: float = 60.0,
    *,
    collector: Collector | None = None,
    vx: float = 0.4,
    settle_s: float = SETTLE_S,
    imu_noise: bool = True,
    stance_chol: float = 1.0e-4,
    swing_chol: float = 1.0e1,
    spawn_radius: float = SPAWN_RADIUS,
    field_margin: float = FIELD_MARGIN,
    max_tilt_deg: float = MAX_TILT_DEG,
    warmup_ticks: int | None = None,
    out_dir: Path | str | None = DATA_DIR,
    verbose: bool = True,
) -> Rollout:
    """Walk `terrain_name` for `seconds`, record at 1 kHz, run the estimator once, save.

    Raises `RuntimeError` — never returns partial data — if the robot falls, leaves the
    heightfield, or goes non-finite. A rollout that walked off the 64 m field is not "slightly
    worse" data: past the edge MuJoCo clamps the hfield, so the robot walks onto an infinite
    extrusion of the boundary row and everything downstream is still perfectly finite.

    `warmup_ticks` (metadata only, nothing is dropped) defaults to the measured joint-KF bias
    plateau; pass an explicit value to override.
    """
    c = collector or build_collector(verbose=verbose)
    n_ticks = int(round(seconds / CONTROL_DT))
    settle_ticks = int(round(settle_s / CONTROL_DT))
    total_ticks = settle_ticks + n_ticks
    T = total_ticks * rp.DECIMATION

    # -- pre-flight: can this rollout even fit on the field? ------------------
    reach = spawn_radius + abs(vx) * seconds + field_margin
    if reach > tr.EXTENT / 2:
        raise ValueError(
            f"a {seconds:.0f}s rollout at vx={vx} from a +-{spawn_radius}m spawn reaches "
            f"{reach:.1f}m, past the {tr.EXTENT / 2:.0f}m half-extent of the heightfield; "
            "shorten the rollout or shrink the spawn box")

    # -- model ---------------------------------------------------------------
    field = terrain_field(terrain_name, seed)
    floor = tr.HeightfieldFloor(field)
    m = rp.build_sim_model(c.policy, with_visuals=False, with_imu_sensors=True, floor=floor)
    # The compiled model must actually carry this terrain (TERRAIN.md §7): a flat field behind a
    # terrain label passes every other check in this function.
    got = m.hfield_data.reshape(field.shape) * tr.EZ
    if not np.allclose(got, field, atol=1e-6):
        raise RuntimeError("hfield_data does not match the rasterised field")

    reader = SimSensorReader(m, c.fused, foot_geoms=rp.FOOT_GEOMS, dt=c.dt,
                             noise=IMUNoise(seed=int(seed)) if imu_noise else None,
                             stance_chol=stance_chol, swing_chol=swing_chol)
    loop = _RecordingLoop(m, c.policy, rp.make_maps(m, c.policy), reader)

    # -- spawn ---------------------------------------------------------------
    x0, y0, yaw = spawn_pose(seed, radius=spawn_radius)
    loop.d.qpos[0:2] = (x0, y0)
    loop.d.qpos[2] += tr.spawn_lift(field)
    loop.d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)      # re-seed the height ramp from the new pose

    # The estimator is seeded from the sim's own state, read straight off `MjData` so that the
    # contact-trust machine is not advanced a tick before the recording starts.
    carry = me.init_fused_carry(
        c.fused,
        q0=jnp.asarray(loop.d.qpos[reader.enc_qadr], dtype=jnp.float64),
        rotation=jnp.asarray(loop.d.xmat[reader.base_bid].reshape(3, 3), dtype=jnp.float64),
        position=jnp.asarray(loop.d.xpos[reader.base_bid], dtype=jnp.float64),
        q0_unfiltered=jnp.asarray(loop.d.qpos[reader.unf_qadr], dtype=jnp.float64),
    )

    # -- run -----------------------------------------------------------------
    if verbose:
        print(f"  {terrain_name}/seed{seed}: relief={floor.relief * 100:.1f}cm  "
              f"spawn=({x0:+.1f},{y0:+.1f})m yaw={np.degrees(yaw):+.0f}deg  "
              f"{settle_s:.0f}s settle + {seconds:.0f}s walk -> T={T} ticks")
    for k in range(total_ticks):
        if k >= settle_ticks:
            loop.cmd[0:3] = (vx, 0.0, 0.0)
            loop.cmd[3] = 0.0
        loop.control_tick()
        if not np.all(np.isfinite(loop.d.qpos)):
            raise RuntimeError(f"{terrain_name}/seed{seed}: non-finite qpos at control tick {k}")

    sensors = _stack(loop.sensors)
    truth = _stack(loop.truth)
    assert len(loop.sensors) == T, f"recorded {len(loop.sensors)} ticks, expected {T}"

    # -- did this rollout produce data at all? --------------------------------
    tilt = _check_rollout(truth, max_tilt_deg=max_tilt_deg,
                          limit=tr.EXTENT / 2 - field_margin,
                          label=f"{terrain_name}/seed{seed}")

    # -- the estimator pass ---------------------------------------------------
    t0 = time.perf_counter()
    inputs, aux, chunk_wall = _run_fused_chunked(c, carry, sensors, T)
    fused_s = time.perf_counter() - t0

    walk0 = settle_ticks * rp.DECIMATION
    travelled = float(np.linalg.norm(truth["p"][-1, :2] - truth["p"][walk0, :2]))
    meta = {
        "terrain": terrain_name,
        "seed": int(seed),
        "policy": c.policy_name,
        "dt": float(c.dt),
        "seconds": float(seconds),
        "settle_s": float(settle_s),
        "vx": float(vx),
        "decimation": int(rp.DECIMATION),
        "T": int(T),
        "settle_ticks": int(settle_ticks * rp.DECIMATION),
        "warmup_ticks": int(WARMUP_TICKS if warmup_ticks is None else warmup_ticks),
        "spawn_xy_yaw": [x0, y0, yaw],
        "relief_m": floor.relief,
        "imu_noise": bool(imu_noise),
        # The gyro bias that was actually INJECTED, (m, 3) in each IMU's own frame. Without it the
        # warm-up measurement can only say the bias estimate stopped moving, not whether it
        # stopped at the right place — and "converged to the wrong constant" looks identical.
        "true_gyro_bias": (reader.noise.bias(reader.n_imus).tolist()
                           if reader.noise is not None else None),
        "stance_chol": float(stance_chol),
        "swing_chol": float(swing_chol),
        "contact_meas_var": float(c.fused.contact_meas_var),
        "chunk_ticks": int(c.chunk_ticks),
        "travelled_m": travelled,
        "tilt_max_deg": float(tilt.max()),
        "git_commit": _git_commit(),
        "wall_sim_s": loop.sim_s,
        "wall_read_s": loop.read_s,
        "wall_fused_s": fused_s,
        "wall_fused_chunks_s": chunk_wall,
    }
    if verbose:
        sim_s = seconds + settle_s
        print(f"    travelled={travelled:.1f}m  tilt_max={tilt.max():.1f}deg  "
              f"wall: sim={loop.sim_s:.1f}s read={loop.read_s:.1f}s fused={fused_s:.1f}s "
              f"({(loop.sim_s + loop.read_s + fused_s) / sim_s:.2f} s/sim-s)")

    roll = Rollout(sensors=sensors, inputs=inputs, truth=truth, aux=aux, meta=meta)
    _assert_float64(roll)
    if out_dir is not None:
        path = Path(out_dir) / f"{terrain_name}_seed{seed:03d}.npz"
        save_rollout(roll, path)
        if verbose:
            print(f"    -> {path}  ({path.stat().st_size / 1e6:.0f} MB)")
    return roll


def _check_rollout(truth: dict, *, max_tilt_deg: float, limit: float, label: str) -> np.ndarray:
    """Raise unless the robot stayed upright and on the field. Returns the tilt trace [deg].

    Split out of `collect_rollout` because it is the part with teeth and the part that must be
    testable against a FABRICATED trajectory: the pre-flight feasibility check upstream makes it
    impossible to provoke the off-field branch from a real 1 s rollout, so without this seam that
    branch would only ever be tested by reading it.
    """
    tilt = np.degrees(np.arccos(np.clip(np.asarray(truth["R"])[:, 2, 2], -1.0, 1.0)))
    if not np.all(np.isfinite(tilt)):
        raise RuntimeError(f"{label}: non-finite attitude")
    if tilt.max() > max_tilt_deg:
        k = int(tilt.argmax())
        raise RuntimeError(
            f"{label}: tilt reached {tilt.max():.1f}deg at t={k * rp.DT:.2f}s "
            f"(bound {max_tilt_deg}deg) — the robot fell or stumbled; the rollout is not data")
    xy = np.abs(np.asarray(truth["p"])[:, :2]).max(axis=0)
    if xy.max() > limit:
        raise RuntimeError(
            f"{label}: left the heightfield — max |x|,|y| = ({xy[0]:.1f},{xy[1]:.1f})m "
            f"against the {limit:.1f}m usable half-extent. Past the edge MuJoCo CLAMPS the "
            "hfield, so the robot keeps walking on an extrusion of the boundary row and every "
            "downstream array stays perfectly finite.")
    return tilt


def _run_fused_chunked(c: Collector, carry, sensors: FusedSensors, T: int):
    """`run_fused` over the whole stream in fixed-length chunks, CARRYING the filter state.

    The carry is the entire point: chunk `k+1` starts from chunk `k`'s final `(jkf_carry,
    inekf_carry)`, so the result is bit-identical to one long scan (asserted by
    `test_chunking_is_exact`).

    **Measured, it was not needed at 62 s** — one unchunked 62 000-tick scan peaked at 4.4 GB
    (~18 kB/tick of `FusedOutputs`, of which ~2.7 kB is the `inekf_inputs` that survive) and took
    186 s against the chunked 175 s, so the honest statement is that this bounds a cost that is
    already affordable. It is kept because `lax.scan` materialises every output of the whole scan
    at once, so the requirement grows linearly and unboundedly with rollout length, and because
    per-chunk wall times are what separate the one-off compile from the steady-state rate in (B).

    The stream is padded to a multiple of `chunk_ticks` with a repeat of the last sample and the
    padding trimmed off afterwards, so exactly ONE scan length is ever compiled, whatever `T` is.
    The padded ticks advance a carry that is then thrown away.
    """
    chunk = min(int(c.chunk_ticks), T)
    pad = (-T) % chunk
    if pad:
        sensors = jax.tree.map(
            lambda a: np.concatenate([a, np.repeat(a[-1:], pad, axis=0)], axis=0), sensors)
    scan = c.scan()
    inputs_chunks, aux_chunks, wall = [], [], []
    for lo in range(0, T + pad, chunk):
        t0 = time.perf_counter()
        xs = jax.tree.map(lambda a: jnp.asarray(a[lo:lo + chunk], dtype=jnp.float64), sensors)
        carry, out = scan(carry, xs)
        inputs = jax.tree.map(np.asarray, out.inekf_inputs)     # blocks until the scan is done
        inputs_chunks.append(inputs)
        aux_chunks.append({
            "bias": np.asarray(out.bias),
            "nis": np.asarray(out.inekf.contact_diagnostics.nis),
            "est_R": np.asarray(out.R),
            "est_v": np.asarray(out.v),
            "est_p": np.asarray(out.p),
        })
        # Per chunk, so the FIRST one (which pays the MJX trace + XLA compile, tens of seconds
        # and once per process) can be separated from the steady-state rate that prices a
        # collection run. Reporting the mean over all chunks understates throughput by ~2x on a
        # 60 s rollout and by 10x on a short one.
        wall.append(time.perf_counter() - t0)
    cat = lambda parts: jax.tree.map(lambda *a: np.concatenate(a, axis=0)[:T], *parts)  # noqa: E731
    return cat(inputs_chunks), cat(aux_chunks), wall


def _stack(records: Sequence):
    """List of pytrees -> one pytree with a leading time axis (float64 NumPy leaves)."""
    return jax.tree.map(lambda *xs: np.asarray(np.stack(xs), dtype=np.float64), *records)


def _assert_float64(roll: Rollout):
    bad = [f"{k}: {v.dtype}" for k, v in _leaves(roll).items()
           if v.dtype != np.float64 and v.dtype.kind == "f"]
    if bad:
        raise TypeError(f"float32 leak at the filter boundary (I8): {bad}")


def _leaves(roll: Rollout) -> dict:
    out = {}
    for name, tree in (("sensors", roll.sensors), ("inputs", roll.inputs),
                       ("truth", roll.truth), ("aux", roll.aux)):
        out.update(_flat(tree, name + "."))
    return out


def _flat(tree, prefix: str) -> dict:
    if isinstance(tree, dict):
        items = tree.items()
    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        items = zip(tree._fields, tree)
    else:
        return {prefix[:-1]: np.asarray(tree)}
    out = {}
    for k, v in items:
        out.update(_flat(v, f"{prefix}{k}."))
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                       # a tarball checkout is not a reason to lose a rollout
        return "unknown"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_rollout(roll: Rollout, path: Path | str, *, compress: bool = True) -> Path:
    """Write one `.npz`. Keys are dotted paths (`inputs.joint.sigma_q`) plus `meta` (JSON).

    Compressed by default, which was NOT the expected answer: the bulk is `Σ_q`/`Σ_q̇`, dense
    float64, and the prior was that zlib would buy little for real time. Measured, it halves the
    file (4.5 -> 2.1 MB per 1000 ticks, i.e. 280 -> 130 MB for a 62 s rollout) for ~4 s of CPU
    against the ~180 s the rollout itself costs. Lossless, so `load_rollout` still round-trips
    bit-identically either way.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _leaves(roll)
    arrays["meta"] = np.array(json.dumps(roll.meta))
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    return path


def load_rollout(path: Path | str) -> Rollout:
    """Inverse of `save_rollout`. Reassembles the typed pytrees, not a bag of arrays.

    Explicit reconstruction, not a generic unflatten: if the `InEKFInputs` structure ever changes,
    this raises instead of quietly handing the trainer a `dict`.
    """
    z = np.load(Path(path), allow_pickle=False)
    g = lambda k: np.asarray(z[k])                                          # noqa: E731
    sensors = FusedSensors(**{f: g(f"sensors.{f}") for f in FusedSensors._fields})
    inputs = InEKFInputs(
        omega=g("inputs.omega"), accel=g("inputs.accel"), raw_omega=g("inputs.raw_omega"),
        joint=JointFilterOutput(
            q=g("inputs.joint.q"), q_dot=g("inputs.joint.q_dot"),
            sigma_q=g("inputs.joint.sigma_q"), sigma_q_dot=g("inputs.joint.sigma_q_dot")),
        contact_chol=g("inputs.contact_chol"),
        contact_meas_chol=g("inputs.contact_meas_chol"),
    )
    pre = lambda p: {k[len(p):]: g(k) for k in z.files if k.startswith(p)}   # noqa: E731
    return Rollout(sensors=sensors, inputs=inputs, truth=pre("truth."), aux=pre("aux."),
                   meta=json.loads(str(z["meta"])))


# ---------------------------------------------------------------------------
# (A) Joint-KF warm-up
# ---------------------------------------------------------------------------

# MEASURED, 2026-07-28, on 62 s walking rollouts over all four terrains (`__main__ --measure`),
# 20-core CPU jaxlib, `imu_noise=True`:
#
#   terrain           bias settle[s]   bias drift-plateau[s]
#   flat                    29.9              9.3
#   waves                   26.0             11.0
#   stepping_stones         28.2             12.6
#   hard_stepping           31.9             15.7
#
# and, on the same flat rollout, the plateau of the quantity that actually reaches the InEKF
# (`diag(Σ_q)`): 3.2 s by the same drift criterion, and immediate by an envelope criterion.
#
# 16 000 ticks = 16 s covers the worst measured BIAS drift-plateau (15.7 s) with a little margin,
# and is 5x the Σ_q plateau. The stricter 5%-of-excursion SETTLING criterion would want 32 s,
# i.e. half of a 62 s rollout; that is not worth paying, because the residual motion after 16 s
# lives almost entirely in the two SHIN IMUs' bias states (see the module docstring), which enter
# neither the InEKF (I1: it consumes the BASE IMU bias, converged to 0.007 rad/s here) nor
# ContactNet's channels (features.py takes the RAW gyro). Collect >= 40 s per rollout so the
# discard is a fraction of it, not most of it.
WARMUP_TICKS = 16_000


def bias_plateau(bias: np.ndarray, dt: float, *, window_s: float = 1.0,
                 tol_frac: float = 0.05, tail_s: float = 5.0) -> dict:
    r"""When does the gyro-bias state stop moving? `(T, 3m) -> dict` of tick counts.

    Two criteria, both scaled by the **excursion** the filter actually had to make,
    :math:`s = \lVert b_\infty - b_0 \rVert` with :math:`b_\infty` the mean over the last
    `tail_s` — never by :math:`\lVert b_\infty \rVert` alone, because with `imu_noise=False` the
    true bias is zero and any criterion normalised by the final value degenerates to 0/0.

    * ``settle``: the last tick with :math:`\lVert b(k) - b_\infty \rVert > \text{tol} \cdot s`,
      i.e. the classical settling time. Sensitive to a late excursion.
    * ``drift``: the first tick after which the 1 s drift
      :math:`\lVert b(k) - b(k - W) \rVert` stays below :math:`\text{tol} \cdot s` forever.
      This is the one that matters for ContactNet: a segment is usable when the bias is no longer
      *moving* across it, which is what makes `Σ_q` stationary.

    Returned ticks are indices into the recorded stream, so they already include the settle.
    """
    b = np.asarray(bias, dtype=float)
    if b.ndim != 2:
        raise ValueError(f"expected (T, 3m), got {b.shape}")
    T = b.shape[0]
    tail = max(1, int(round(tail_s / dt)))
    w = max(1, int(round(window_s / dt)))
    b_inf = b[-tail:].mean(axis=0)
    scale = float(np.linalg.norm(b_inf - b[0]))
    if scale <= 0.0:
        return dict(settle=0, drift=0, scale=scale, b_inf_norm=float(np.linalg.norm(b_inf)),
                    degenerate=True)

    err = np.linalg.norm(b - b_inf, axis=1)
    over = np.flatnonzero(err > tol_frac * scale)
    settle = int(over[-1] + 1) if over.size else 0

    drift = np.full(T, np.inf)
    drift[w:] = np.linalg.norm(b[w:] - b[:-w], axis=1)
    over_d = np.flatnonzero(drift > tol_frac * scale)
    plateau = int(over_d[-1] + 1) if over_d.size else 0

    return dict(settle=settle, drift=plateau, scale=scale,
                b_inf_norm=float(np.linalg.norm(b_inf)),
                err_at_plateau=float(err[min(plateau, T - 1)]), degenerate=False)


# ---------------------------------------------------------------------------
# (B) Throughput
# ---------------------------------------------------------------------------

def fused_rate(meta: dict) -> tuple[float, float]:
    """`(steady_state, including_compile)` wall-seconds of `run_fused` per simulated second.

    The first chunk carries the one-per-process MJX trace + XLA compile; quoting the mean over
    all chunks prices a collection run at roughly twice its real cost.
    """
    sim_s = meta["seconds"] + meta["settle_s"]
    w = list(meta["wall_fused_chunks_s"])
    per_chunk_sim = meta["chunk_ticks"] * meta["dt"]
    steady = (sum(w[1:]) / (len(w) - 1) / per_chunk_sim) if len(w) > 1 else float("nan")
    return steady, sum(w) / sim_s


# ---------------------------------------------------------------------------
# End-to-end channel sanity (the consumer, run over collected data)
# ---------------------------------------------------------------------------

def channel_report(c: Collector, sensors: FusedSensors, *, rest: slice | None = None) -> dict:
    """Run `contactnet.features.make_contact_channels` over a collected stream and score it.

    The point is end-to-end: a collector that produces correctly-shaped garbage passes every test
    inside this module. `p_bc_z` (foot below the pelvis, ~-0.9 m) and `base_accel_z` (+9.81 at
    rest, specific force) are physical numbers with known values, so a permuted gather or a frame
    error shows up as a number that is simply wrong.

    `rest` is the window the "at rest" numbers are averaged over and defaults to the LAST second
    of the standing settle. Not the first: the robot is spawned lifted clear of the terrain
    (`terrain.spawn_lift`) and spends the first few hundred ms falling onto it, which is why the
    naive `slice(0, 500)` reports a 0.2 rad/s "resting" gyro.
    """
    from ..contactnet.features import (build_subchain_indices, channel_names,
                                       make_contact_channels)

    reader_unfiltered = _unfiltered_names(c)
    sub = build_subchain_indices(c.fused.build.joint_names, reader_unfiltered)
    chan = make_contact_channels(sub, c.fused.base_imu, c.fused.kinematics, c.dt)
    x = contact_channels_chunked(chan, sensors)
    names = channel_names()
    idx = {n: i for i, n in enumerate(names)}
    if rest is None:
        settle = int(round(SETTLE_S / c.dt))
        end = min(settle, x.shape[0])
        rest = slice(max(0, end - 1000), end)
    return {
        "shape": x.shape,
        "n_channels": len(names),
        "finite": bool(np.all(np.isfinite(x))),
        "p_bc_z_mean": float(x[:, :, idx["p_bc_z"]].mean()),
        "p_bc_z_range": [float(x[:, :, idx["p_bc_z"]].min()), float(x[:, :, idx["p_bc_z"]].max())],
        "base_accel_z_rest": float(x[rest, :, idx["base_accel_z"]].mean()),
        "base_gyro_norm_rest": float(np.linalg.norm(
            x[rest, 0, idx["base_gyro_x"]:idx["base_gyro_z"] + 1], axis=-1).mean()),
        "v_bc_absmax": float(np.abs(x[:, :, idx["v_bc_x"]:idx["v_bc_z"] + 1]).max()),
        "tau_knee_absmax": float(np.abs(x[:, :, idx["tau_knee_y"]]).max()),
    }


def contact_channels_chunked(channels, sensors: FusedSensors, chunk: int = 2_000) -> np.ndarray:
    """`contactnet.features.make_contact_channels`, evaluated a chunk of ticks at a time.

    **This is a workaround for a real scaling limit in the feature path, not a convenience.**
    `make_contact_channels` does `jax.vmap(kinematics)(q_all)` over the whole leading axis, i.e.
    `T` simultaneous full-body MJX FK evaluations. That is fine for a training window and is not
    fine for a rollout: at T = 62 000 it grew to ~38 GB of RSS on this machine before it was
    killed. Anything that runs the feature path over a whole collected rollout — a normalization
    statistics pass, an export check, this report — needs the same treatment.

    The split is exact in structure, not merely close. The only cross-tick term is
    ``v[k] = (p[k] - p[k-1]) / dt``, so each chunk is evaluated with ONE tick of lead-in and its
    first row discarded. Without it every chunk boundary would silently carry ``v = 0`` — the kind
    of artifact that trains fine and deploys wrong.

    What is NOT reproduced is the last bit: XLA's batched FK depends on the vmap width, so ``p``
    moves by up to 1 ulp of ~0.9 m and ``v = dp/dt`` amplifies that by 1/dt to ~2e-13. The 18
    non-FK channels ARE bit-identical. `test_chunked_channels_match_one_pass` pins both halves of
    that statement rather than papering over it with one loose tolerance.
    """
    T = sensors.encoders.shape[0]
    out = []
    for lo in range(0, T, chunk):
        start = max(0, lo - 1)      # lead-in, so this chunk's first kept row has a real v
        xs = jax.tree.map(lambda a: jnp.asarray(a[start:lo + chunk]), sensors)
        y = np.asarray(channels(xs))
        out.append(y[1:] if start < lo else y)
    return np.concatenate(out, axis=0)


def _unfiltered_names(c: Collector) -> tuple[str, ...]:
    """The off-path anchor joints, by name — the same resolution `SimSensorReader` does."""
    from .sensors import _dof_joint_names
    return _dof_joint_names(c.fused.model.mj_model,
                            np.asarray(c.fused.build.dof_anchor_unfiltered, dtype=int))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def collect_all(terrains: Sequence[str] | None = None, seeds: Sequence[int] = (0, 1, 2),
                seconds: float = 60.0, *, out_dir: Path | str = DATA_DIR,
                collector: Collector | None = None, keep: bool = False, **kw) -> list[dict]:
    """Every terrain x every seed. Returns one metadata dict per rollout.

    A rollout that falls or leaves the field is REPORTED and skipped, not retried and not silently
    replaced — a terrain whose seeds keep falling is a finding about the policy, and hiding it
    behind a resample is how a dataset ends up quietly biased toward the easy seeds.

    `keep=False` drops each `Rollout` after saving; a full collection run does not fit in RAM.
    """
    c = collector or build_collector()
    names = list(terrains) if terrains else list(tr.TERRAINS)
    metas, kept, failures = [], [], []
    for name in names:
        for s in seeds:
            try:
                r = collect_rollout(name, seed=s, seconds=seconds, collector=c,
                                    out_dir=out_dir, **kw)
            except RuntimeError as e:
                print(f"  !! SKIPPED {name}/seed{s}: {e}")
                failures.append((name, s, str(e)))
                continue
            metas.append(r.meta)
            if keep:
                kept.append(r)
    print(f"\ncollected {len(metas)}/{len(names) * len(seeds)} rollouts "
          f"-> {Path(out_dir)}" + (f"; {len(failures)} failed" if failures else ""))
    return (metas, kept) if keep else metas


def _measure(args):
    """(A) warm-up + (B) throughput, on a long rollout per terrain. The two deliverables."""
    c = build_collector(chunk_ticks=args.chunk)
    rows = []
    for name in (args.terrain or list(tr.TERRAINS)):
        r = collect_rollout(name, seed=args.seed, seconds=args.seconds, collector=c,
                            out_dir=args.out if args.save else None)
        pl = bias_plateau(r.aux["bias"], c.dt)
        if r.meta["true_gyro_bias"] is not None:
            b_true = np.asarray(r.meta["true_gyro_bias"]).ravel()
            b_end = r.aux["bias"][-5000:].mean(axis=0)
            print(f"  (A) {name}: |b_true|={np.linalg.norm(b_true):.4f}  "
                  f"|b_hat_end|={np.linalg.norm(b_end):.4f}  "
                  f"|b_hat-b_true|={np.linalg.norm(b_end - b_true):.4f} rad/s")
        sim_s = r.meta["seconds"] + r.meta["settle_s"]
        settle = r.meta["settle_ticks"]
        rows.append((name, pl, r.meta,
                     channel_report(c, r.sensors, rest=slice(settle - 1000, settle))))
        steady, with_compile = fused_rate(r.meta)
        print(f"\n  (A) {name}: bias excursion {pl['scale']:.4f} rad/s, "
              f"|b_inf|={pl['b_inf_norm']:.4f}; settle={pl['settle']} ticks "
              f"({pl['settle'] * c.dt:.1f}s), drift-plateau={pl['drift']} ticks "
              f"({pl['drift'] * c.dt:.1f}s)")
        print(f"  (B) {name}: sim+policy {r.meta['wall_sim_s'] / sim_s:.3f} s/sim-s, "
              f"run_fused {steady:.3f} s/sim-s steady ({with_compile:.3f} incl. compile), "
              f"sensor read {r.meta['wall_read_s'] / sim_s:.3f} s/sim-s")
        print(f"      channels: {rows[-1][3]}")
    print("\n" + "=" * 88)
    print(f"{'terrain':16s} {'settle[s]':>10s} {'plateau[s]':>11s} {'sim[s/s]':>10s} "
          f"{'fused[s/s]':>11s} {'+compile':>10s} {'read[s/s]':>10s}")
    for name, pl, meta, _ in rows:
        sim_s = meta["seconds"] + meta["settle_s"]
        steady, with_compile = fused_rate(meta)
        print(f"{name:16s} {pl['settle'] * c.dt:10.2f} {pl['drift'] * c.dt:11.2f} "
              f"{meta['wall_sim_s'] / sim_s:10.3f} {steady:11.3f} {with_compile:10.3f} "
              f"{meta['wall_read_s'] / sim_s:10.3f}")
    worst = max(pl["drift"] for _, pl, _, _ in rows)
    print(f"\nWARMUP_TICKS should be >= {worst} ({worst * c.dt:.1f}s); module has {WARMUP_TICKS}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--terrain", action="append", choices=list(tr.TERRAINS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--no-noise", action="store_true", help="clean sensors (no bias to converge)")
    ap.add_argument("--measure", action="store_true",
                    help="(A) bias warm-up + (B) throughput, one long rollout per terrain")
    ap.add_argument("--seed", type=int, default=0, help="--measure only")
    ap.add_argument("--save", action="store_true", help="--measure only: also write the .npz")
    args = ap.parse_args()

    if args.measure:
        _measure(args)
    else:
        collect_all(args.terrain, args.seeds, args.seconds, out_dir=args.out,
                    collector=build_collector(chunk_ticks=args.chunk),
                    vx=args.vx, imu_noise=not args.no_noise)
