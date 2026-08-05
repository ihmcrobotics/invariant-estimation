r"""Flat-ground rollouts recorded as ContactNet training data.

A MINIMAL port of the reference sim/collect.py (contact-net-integration), flat
terrain only: it drops terrain heightfields, domain randomisation, slip
instrumentation and ramp friction (all out of scope for the overnight training
run). Diversity comes from spawn yaw + IMU-noise seed + forward command.

Runs the policy in MuJoCo, records per-tick FusedSensors + ground truth, runs
the fused estimator ONCE over the recorded stream to freeze the InEKFInputs the
network trains against, and saves one .npz per rollout. Save/load are generic
over FusedSensors._fields, so encoders_vel/torques serialize automatically.

Coherent 1 kHz regime: the whole ContactNet stack (config dt=1e-3, warmup_ticks
=16000, window bandwidths, InEKF/jointKF config dt) is designed at 1 kHz. Set
rp.DT=0.001, rp.DECIMATION=20 before collecting (CONTROL_DT=0.02 unchanged, so
policy behaviour is identical). take-two's default rp.DT=0.005 (200 Hz) would
mismatch every one of those constants.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

# `jax_enable_x64` is flipped at package import and MUST precede any array construction (I8).
import invariant_estimation  # noqa: F401
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from invariant_estimation.contactnet.config import ContactNetConfig

from ..pipeline import main_estimator as me
from ..pipeline.main_estimator import FusedSensors
from ..inEKF.filter import InEKFInputs, JointFilterOutput
from .sensors import IMUNoise, SimSensorReader

__all__ = [
    "DATA_DIR", "Rollout", "Collector", "build_collector", "spawn_pose",
    "collect_rollout", "collect_all", "save_rollout", "load_rollout",
    "contact_channels_chunked", "WARMUP_TICKS",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import run_policy as rp  # noqa: E402

SETTLE_S = 2.0
SPAWN_RADIUS = 3.0
MAX_TILT_DEG = 15.0
WARMUP_TICKS = 16_000
"""Joint-KF bias/Sigma_q warm-up (measured, 1 kHz): 16 s covers the worst bias
drift-plateau and is 5x the Sigma_q plateau (reference collect.py note (A))."""


def spawn_pose(seed: int, *, radius: float = SPAWN_RADIUS) -> tuple[float, float, float]:
    """(x, y, yaw) for rollout `seed`. The heading matters more than the position:
    the policy walks +x in its OWN frame, so random yaw is what makes two rollouts
    traverse different ground."""
    r = np.random.default_rng(0xC0FFEE + int(seed))
    x, y = r.uniform(-radius, radius, 2)
    return float(x), float(y), float(r.uniform(-np.pi, np.pi))


@dataclass
class Collector:
    """Policy + fused estimator + compiled scan. Independent of the floor, built
    once: the MJX trace + XLA compile (~30 s, on the first chunk of the first
    rollout) is paid once for a whole collection run. Reuse one Collector."""

    policy: dict
    fused: me.FusedEstimator
    dt: float
    chunk_ticks: int
    policy_name: str
    build_s: float = 0.0
    _scan: object = None

    def scan(self):
        """(carry, sensors) -> (carry, FusedOutputs), jitted, one compiled shape per chunk."""
        if self._scan is None:
            step = me.make_fused_step(self.fused)
            self._scan = jax.jit(lambda c, xs: jax.lax.scan(step, c, xs))
        return self._scan


def build_collector(policy_name: str = "baseline", *, dt: float = None,
                    chunk_ticks: int = 10_000, contact_meas_var: float = 0.0,
                    contacts_per_foot: int = 1, verbose: bool = True) -> Collector:
    """Load the policy and build the fused estimator.

    contact_fk_unfiltered=True is NOT optional: without it FusedSensors.q_unfiltered
    is empty, the contact FK stands on qpos0 ankles, and ContactNet's ankle q/tau
    channels do not exist (features raises rather than narrowing silently).
    contact_meas_var=0.0 is the isotropic floor ContactNet replaces (process socket).
    """
    dt = float(rp.DT) if dt is None else float(dt)
    t0 = time.time()
    policy = rp.load_policy(policy_name)
    fused = me.build_alex_fused_estimator_from_urdf(
        rp.cycloid_forearm_urdf(rp.URDF), contacts_per_foot=contacts_per_foot, dt=dt,
        contact_meas_var=contact_meas_var, contact_fk_unfiltered=True)
    c = Collector(policy=policy, fused=fused, dt=dt, chunk_ticks=int(chunk_ticks),
                  policy_name=policy_name, build_s=time.time() - t0)
    if verbose:
        print(f"collector: {fused.n_joints} filtered joints, {fused.build.n_imus} IMUs, "
              f"{fused.n_contacts} contacts, {fused.n_aux} off-path joints, dt={dt} "
              f"({1 / dt:.0f} Hz)  [built in {c.build_s:.1f}s]")
    return c

@dataclass
class Disturb:
    rate_hz: float = 0.0
    mag_N: tuple = (30.0, 120.0)
    dur_s: float = 0.1

class _RecordingLoop(rp.Loop):
    """run_policy.Loop that samples the sensors after EVERY mj_step. control_tick
    is reimplemented (the base has no hook inside its decimation loop); the
    policy/actuator lines are a verbatim copy of rp.Loop.control_tick."""

    def __init__(self, m, policy, maps, reader: SimSensorReader, disturb = None, dr_seed=0):
        super().__init__(m, policy, maps)
        self.reader = reader
        self.sensors: list = []
        self.truth: list = []
        self.sim_s = 0.0
        self.read_s = 0.0
        self.tick = 0

        self.disturb = disturb
        self._drng = np.random.default_rng((int(dr_seed) << 20) ^ 0xF00D)
        self._push_left = 0
        self._push_vec = np.zeros(3)
        self._push_bid = reader.base_bid # pelvis

    def control_tick(self):
        if self.disturb is not None and self.disturb.rate_hz > 0.0:
            dt_c = rp.DT * rp.DECIMATION
            if self._push_left <= 0 and self._drng.random() < self.disturb.rate_hz * dt_c:
                mag = self._drng.uniform(*self.disturb.mag_N)
                ang = self._drng.uniform(0.0, 2 * np.pi)
                self._push_vec = mag * np.array([np.cos(ang), np.sin(ang), 0.0])
                self._push_left = max(1, int(round(self.disturb.dur_s / dt_c)))
            # xrfc applied is not auto cleared, we need to set it every tick, and zero when its not doing anything.
            self.d.xfrc_applied[self._push_bid, :3] = self._push_vec if self._push_left > 0 else 0.0
            self._push_left -= 1

        # Existing code before change to disturbances
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
            # One read per physics tick: SimSensorReader owns the contact-trust
            # state machine, so a double/skip read silently changes the Schmitt/dwell
            # trajectory the joint-KF stance anchors ride on.
            self.sensors.append(self.reader.read(self.d))
            self.truth.append(self.reader.truth(self.d))
            self.tick += 1
            t0 = time.perf_counter()
            self.read_s += t0 - t1
        self._ramp_t += rp.DECIMATION * rp.DT


class Rollout(NamedTuple):
    """One collected rollout, NumPy, every leaf with a leading time axis of length T."""
    sensors: FusedSensors     # the plant boundary, as ContactNet's features consume it
    inputs: InEKFInputs       # what the InEKF actually consumed (incl. joint.sigma_q)
    truth: dict               # R (T,3,3), v/p/omega (T,3), q/q_dot (T,n)
    aux: dict                 # bias, nis, est_R/est_v/est_p -- filter health
    meta: dict

def _command_schedule(seed, n_control_ticks, control_dt, cfg):
    rng = np.random.default_rng((int(seed) << 8) ^ 0xBADA55) # hell yeah brother lol
    per = max(1, round(cfg.cmd_resample_s / control_dt))
    def draw(lo, hi): return rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi)
    n = -(-n_control_ticks // per)
    sched = np.array(
        [[draw(*cfg.cmd_vx_range), draw(*cfg.cmd_vy_range), draw(*cfg.cmd_yaw_range)] for _ in range(n)]
    )
    return sched, per


def collect_rollout(seed: int = 0, seconds: float = 60.0, *,
                    terrain="flat",
                    collector: Collector | None = None, vx: float = 0.4,
                    settle_s: float = SETTLE_S, imu_noise: bool = True,
                    stance_chol: float = 1.0e-4, swing_chol: float = 1.0e1,
                    spawn_radius: float = SPAWN_RADIUS, max_tilt_deg: float = MAX_TILT_DEG,
                    warmup_ticks: int | None = None,
                    out_dir: Path | str | None = DATA_DIR, cfg: ContactNetConfig = ContactNetConfig(),
                    cmd_override = None, verbose: bool = True) -> Rollout:
    """Walk for `seconds`, record at 1/rp.DT Hz, run the estimator once, save.

    Raises RuntimeError -- never returns partial data -- if the robot falls or goes
    non-finite. terrain_name is fixed to "flat" (the built-in plane floor)."""
    from invariant_estimation.sim import terrain as terr
    c = collector or build_collector(verbose=verbose)
    dr_rng = np.random.default_rng((int(seed) << 24) ^ 0xDEADBEEF)

    use_terrain = terrain != "flat"
    field = terr.sample_field(terrain, seed) if use_terrain else None
    m = rp.build_sim_model(c.policy, with_visuals=False, with_imu_sensors=True, terrain=field)

    if cfg.env_dr:
        # MuJoCo mixes pairwise friction by ELEMENT-WISE MAX -> a low mu on the foot
        # alone is clipped back up by the mu = 1 floor, so we set both, so the effective mu is the one that we want.
        if dr_rng.random() < cfg.friction_low_tail_prob:
            lo, hi = cfg.friction_low_tail
        else:
            lo, hi = cfg.friction_range
        mu = float(dr_rng.uniform(lo, hi))
        gids = [m.geom(n).id for n in rp.FOOT_GEOMS] + [m.geom("floor").id]
        m.geom_friction[gids, 0] = mu
    else:
        mu = float(rp.CONTACT["friction"].split()[0])

    control_dt = rp.DT * rp.DECIMATION
    n_ticks = int(round(seconds / control_dt))
    settle_ticks = int(round(settle_s / control_dt))
    total_ticks = settle_ticks + n_ticks
    T = total_ticks * rp.DECIMATION

    reader = SimSensorReader(m, c.fused, foot_geoms=rp.FOOT_GEOMS, dt=c.dt,
                             noise=IMUNoise(seed=int(seed)) if imu_noise else None,
                             stance_chol=stance_chol, swing_chol=swing_chol)
    disturb = (Disturb(cfg.disturb_rate_hz, cfg.disturb_mag_N, cfg.disturb_dur_s) if cfg.env_dr else None)
    loop = _RecordingLoop(m, c.policy, rp.make_maps(m, c.policy), reader, disturb=disturb, dr_seed=seed)

    x0, y0, yaw = spawn_pose(seed, radius=spawn_radius)
    loop.d.qpos[0:2] = (x0, y0)
    loop.d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    if use_terrain:
        loop.d.qpos[2] = float(field.max()) + 0.02
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)

    # Seed the estimator from the sim's own state, read straight off MjData so the
    # contact-trust machine is not advanced a tick before recording starts.
    carry = me.init_fused_carry(
        c.fused,
        q0=jnp.asarray(loop.d.qpos[reader.enc_qadr], dtype=jnp.float64),
        rotation=jnp.asarray(loop.d.xmat[reader.base_bid].reshape(3, 3), dtype=jnp.float64),
        position=jnp.asarray(loop.d.xpos[reader.base_bid], dtype=jnp.float64),
        q0_unfiltered=jnp.asarray(loop.d.qpos[reader.unf_qadr], dtype=jnp.float64),
    )

    if verbose:
        print(f"  {terrain}/seed{seed}: spawn=({x0:+.1f},{y0:+.1f})m yaw={np.degrees(yaw):+.0f}deg  "
              f"{settle_s:.0f}s settle + {seconds:.0f}s walk -> T={T} ticks")
    if cmd_override is not None:
        sched, per = np.asarray([cmd_override], dtype=float), max(1, total_ticks)  # override the schedule with a single command
    else:
        sched, per = _command_schedule(seed, total_ticks, control_dt, cfg)
    for k in range(total_ticks):
        if k >= settle_ticks:
            # loop.cmd[0:3] = (vx, 0.0, 0.0) #WARNING: this is only forward command, and this isn't randomized - explain why we aren't robust to a wide range of motions.
            idx = min((k - settle_ticks) // per, len(sched) - 1)
            loop.cmd[0:3] = sched[idx]
            loop.cmd[3] = 0.0
        loop.control_tick()
        if not np.all(np.isfinite(loop.d.qpos)):
            raise RuntimeError(f"{terrain}/seed{seed}: non-finite qpos at control tick {k}")

    sensors = _stack(loop.sensors)
    truth = _stack(loop.truth)
    assert len(loop.sensors) == T, f"recorded {len(loop.sensors)} ticks, expected {T}"

    tilt = _check_rollout(truth, max_tilt_deg=max_tilt_deg, label=f"{terrain}/seed{seed}")

    t0 = time.perf_counter()
    inputs, aux, chunk_wall = _run_fused_chunked(c, carry, sensors, T)
    fused_s = time.perf_counter() - t0

    walk0 = settle_ticks * rp.DECIMATION
    travelled = float(np.linalg.norm(truth["p"][-1, :2] - truth["p"][walk0, :2]))
    meta = {
        "terrain": "flat",
        "seed": int(seed),
        "policy": c.policy_name,
        "dt": float(c.dt),
        "seconds": float(seconds),
        "settle_s": float(settle_s),
        # "vx": float(vx),
        "cmd_seed": int(seed),
        "cmd_override": None if cmd_override is None else [float(x) for x in cmd_override],
        "cmd_ranges": {
            "vx": cfg.cmd_vx_range,
            "vy": cfg.cmd_vy_range,
            "yaw": cfg.cmd_yaw_range
        },
        "decimation": int(rp.DECIMATION),
        "T": int(T),
        "settle_ticks": int(settle_ticks * rp.DECIMATION),
        "warmup_ticks": int(WARMUP_TICKS if warmup_ticks is None else warmup_ticks),
        "spawn_xy_yaw": [x0, y0, yaw],
        "imu_noise": bool(imu_noise),
        "true_gyro_bias": (reader.noise.bias(reader.n_imus).tolist()
                           if reader.noise is not None else None),
        "stance_chol": float(stance_chol),
        "swing_chol": float(swing_chol),
        "contact_meas_var": float(c.fused.contact_meas_var),
        "chunk_ticks": int(c.chunk_ticks),
        "travelled_m": travelled,
        "tilt_max_deg": float(tilt.max()),
        "wall_sim_s": loop.sim_s,
        "wall_read_s": loop.read_s,
        "wall_fused_s": fused_s,
        "wall_fused_chunks_s": chunk_wall,
        "terrain": terrain,
        "friction_mu": mu,
        "env_dr": bool(cfg.env_dr),
        "disturb": (None if disturb is None else {
            "rate_hz": float(cfg.disturb_rate_hz),
            "mag_N": list(cfg.disturb_mag_N),
            "dur_s": float(cfg.disturb_dur_s)
        })
    }
    if verbose:
        sim_s = seconds + settle_s
        print(f"    travelled={travelled:.1f}m  tilt_max={tilt.max():.1f}deg  "
              f"wall: sim={loop.sim_s:.1f}s read={loop.read_s:.1f}s fused={fused_s:.1f}s "
              f"({(loop.sim_s + loop.read_s + fused_s) / sim_s:.2f} s/sim-s)")

    roll = Rollout(sensors=sensors, inputs=inputs, truth=truth, aux=aux, meta=meta)
    _assert_float64(roll)
    if out_dir is not None:
        # N is in the filename: raw sensors are N-agnostic but `InEKFInputs` and
        # `contact_chol` are not, so an N=2 pool must never be picked up by an N=8
        # run. The N=2 name is left bare so the existing flat pool stays valid.
        tag = "" if c.fused.n_contacts == 2 else f"_n{c.fused.n_contacts}"
        path = Path(out_dir) / f"{terrain}{tag}_seed{seed:03d}.npz"
        save_rollout(roll, path)
        if verbose:
            print(f"    -> {path}  ({path.stat().st_size / 1e6:.0f} MB)")
    return roll


def _check_rollout(truth: dict, *, max_tilt_deg: float, label: str) -> np.ndarray:
    """Raise unless the robot stayed upright. Returns the tilt trace [deg]."""
    tilt = np.degrees(np.arccos(np.clip(np.asarray(truth["R"])[:, 2, 2], -1.0, 1.0)))
    if not np.all(np.isfinite(tilt)):
        raise RuntimeError(f"{label}: non-finite attitude")
    tail = tilt[-int(0.5 / rp.DT):]
    # if tilt.max() > max_tilt_deg:
    #     k = int(tilt.argmax())
    #     raise RuntimeError(
    #         f"{label}: tilt reached {tilt.max():.1f}deg at t={k * rp.DT:.2f}s "
    #         f"(bound {max_tilt_deg}deg) -- the robot fell; the rollout is not data")
    if tail.mean() > max_tilt_deg or tilt.max() > 75.0:
        raise RuntimeError(
            f"{label}: end-tilt {tail.mean():.1f}deg / peak {tilt.max():.1f} deg"
            f"(bound {max_tilt_deg}deg) -- fell and did not recover, the rollout is not data"
        )
    return tilt


def _run_fused_chunked(c: Collector, carry, sensors: FusedSensors, T: int):
    """run_fused over the whole stream in fixed-length chunks, CARRYING filter state.
    Chunk k+1 starts from chunk k's final carry, so the result is bit-identical to
    one long scan. Padded to a multiple of chunk_ticks so ONE scan length compiles."""
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


def save_rollout(roll: Rollout, path: Path | str, *, compress: bool = True) -> Path:
    """Write one .npz. Keys are dotted paths (inputs.joint.sigma_q) plus meta (JSON).
    Generic over _fields, so encoders_vel/torques serialize automatically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _leaves(roll)
    arrays["meta"] = np.array(json.dumps(roll.meta))
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    return path


def load_rollout(path: Path | str) -> Rollout:
    """Inverse of save_rollout. Reassembles the typed pytrees, not a bag of arrays."""
    z = np.load(Path(path), allow_pickle=False)
    g = lambda k: jnp.asarray(z[k])                                          # noqa: E731
    sensors = FusedSensors(**{f: g(f"sensors.{f}") for f in FusedSensors._fields})
    # NOTE: take-two's InEKFInputs is PROCESS-SOCKET-ONLY -- no contact_meas_chol
    # field (the hard invariant, enforced structurally). ContactNet drives contact_chol.
    inputs = InEKFInputs(
        omega=g("inputs.omega"), accel=g("inputs.accel"), raw_omega=g("inputs.raw_omega"),
        joint=JointFilterOutput(
            q=g("inputs.joint.q"), q_dot=g("inputs.joint.q_dot"),
            sigma_q=g("inputs.joint.sigma_q"), sigma_q_dot=g("inputs.joint.sigma_q_dot")),
        contact_chol=g("inputs.contact_chol"),
    )
    pre = lambda p: {k[len(p):]: g(k) for k in z.files if k.startswith(p)}   # noqa: E731
    return Rollout(sensors=sensors, inputs=inputs, truth=pre("truth."), aux=pre("aux."),
                   meta=json.loads(str(z["meta"])))


def contact_channels_chunked(channels, sensors: FusedSensors, chunk: int = 2_000) -> np.ndarray:
    """features.make_contact_channels a chunk of ticks at a time (the full-rollout
    vmapped FK OOMs). The only cross-tick term is v[k]=(p[k]-p[k-1])/dt, so each
    chunk is evaluated with ONE tick of lead-in and its first row discarded."""
    T = sensors.encoders.shape[0]
    out = []
    for lo in range(0, T, chunk):
        start = max(0, lo - 1)
        xs = jax.tree.map(lambda a: jnp.asarray(a[start:lo + chunk]), sensors)
        y = np.asarray(channels(xs))
        out.append(y[1:] if start < lo else y)
    return np.concatenate(out, axis=0)


def _unfiltered_names(c: Collector) -> tuple[str, ...]:
    """The off-path anchor joints, by name -- the same resolution SimSensorReader does."""
    from .sensors import _dof_joint_names
    return _dof_joint_names(c.fused.model.mj_model,
                            np.asarray(c.fused.build.dof_anchor_unfiltered, dtype=int))


def collect_all(seeds: Sequence[int] = (0, 1, 2), seconds: float = 60.0, *,
                out_dir: Path | str = DATA_DIR, collector: Collector | None = None,
                **kw) -> list[dict]:
    """Collect one flat rollout per seed, skipping falls. Returns the metas."""
    c = collector or build_collector()
    metas = []
    for s in seeds:
        try:
            roll = collect_rollout(int(s), seconds, collector=c, out_dir=out_dir, **kw)
            metas.append(roll.meta)
        except RuntimeError as e:
            print(f"  skipped {kw.get('terrain', 'flat')}/seed{s}: {e}")
    return metas
