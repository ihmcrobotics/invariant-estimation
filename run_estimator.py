"""run_estimator.py — the RL policy driven by the ESTIMATOR, in the MuJoCo sim.

`run_policy.py` runs the policy on ground truth. This runs the same sim with the fused
estimator (joint KF -> InEKF) in the loop: simulated IMUs and encoders go in, and the policy's
`base_ang_vel` / `projected_gravity` come out of the filter instead of out of `MjData` — which
is the arrangement on the real robot.

    uv run python run_estimator.py --policy baseline --headless --ticks 500
    uv run python run_estimator.py --policy baseline                     # viewer
    uv run python run_estimator.py --policy baseline --headless --vx 0.6 --imu-noise
    uv run python run_estimator.py --policy baseline --headless --source truth   # A/B: estimator
                                                                                 # runs but does
                                                                                 # not drive
    uv run python run_estimator.py --policy baseline --headless --ticks 1500 --vx 0.6 \
        --contactnet artifacts/contactnet_run4.npz \
        --contactnet-norm data/dr/norm_constants.npz     # learned contact noise in the loop
Every run scores the estimate against the sim's own state (attitude, tilt-as-seen-by-the-policy,
gyro, velocity, position drift, joint state) and can dump the full history to `.npz`.

Speed (CPU-only jaxlib, measured): a control tick costs ~35 ms against its 20 ms real-time
budget while walking and ~21 ms standing, so the viewer runs at roughly 0.6x speed. Building the
estimator and compiling the step costs ~55 s up front — `make_estimated_loop` compiles eagerly
(`EstimatorRuntime.warmup`) so that cost lands there and not as an 11 s freeze on the first tick.

Read `run_policy.py` first — the sim, the policy contract and every magic number live there.
"""
import argparse
import os
import sys
import time

# `MUJOCO_GL` is read when `mujoco` is imported, so the offscreen backend has to be chosen before
# the import below -- hence the argv peek. EGL renders without a window, which is what `--video`
# wants; the interactive viewer is left on the platform default.
if any(a.startswith("--video") for a in sys.argv):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

import run_policy as rp
from invariant_estimation.pipeline import main_estimator as me
from invariant_estimation.sim.estimator_loop import (
    EstimatorRuntime,
    attitude_error_deg,
    tilt_error_deg,
)
from invariant_estimation.sim.sensors import IMUNoise, SimSensorReader

# Terms the policy may take from the estimate. `base_ang_vel` and `projected_gravity` are the
# only two the estimator feeds on hardware; `joints` additionally routes the 9 FILTERED joints'
# position/velocity through the joint KF (the arms always stay on raw encoders — the KF does not
# filter them).
SOURCES = ("base_ang_vel", "projected_gravity", "joints")
DEFAULT_SOURCES = ("base_ang_vel", "projected_gravity")


class EstimatedLoop(rp.Loop):
    """`run_policy.Loop` with the estimator in the observation path.

    One control tick, in order:

        advance the estimator over the DECIMATION physics steps that just happened
        -> obs (with the estimated terms substituted) -> policy -> DECIMATION * mj_step

    so the estimate the policy reads is current. The sensor batch handed to the estimator is
    always exactly DECIMATION samples long, which keeps the scanned graph constant (I7).
    """

    def __init__(self, m, policy, maps, *, fused, reader, sources=DEFAULT_SOURCES, est_every=1):
        super().__init__(m, policy, maps)
        self.reader = reader
        self.sources = tuple(sources)
        # `est_every` physics steps per estimator step. 1 = the physics rate (200 Hz), the
        # faithful setting. 4 = one estimator step per control tick (50 Hz), which is what makes
        # the viewer run at roughly real time; the filter is rate-parameterised so this is a
        # legitimate configuration, just a coarser one.
        if rp.DECIMATION % est_every:
            raise ValueError(f"est_every must divide DECIMATION={rp.DECIMATION}")
        self.est_every = int(est_every)
        substeps = rp.DECIMATION // self.est_every
        self.rt = EstimatorRuntime(fused, reader, substeps=substeps)
        self.rt.seed(self.d)
        # t=0 the robot is at rest and the sensors are already meaningful, so prime the batch
        # rather than special-casing the first tick.
        self.batch = [reader.read(self.d) for _ in range(substeps)]
        # Filtered joints -> their slots in the policy's joint ordering.
        order = list(policy["order"])
        self.filtered_slots = np.array([order.index(n) for n in fused.build.joint_names])
        self.history = []

    # -- one control tick ---------------------------------------------------

    def control_tick(self):
        est = self.rt.advance(self.batch)
        self.batch = []
        self.cmd[4] = self._height()

        overrides = {}
        if "base_ang_vel" in self.sources:
            overrides["base_ang_vel"] = est.omega_body
        if "projected_gravity" in self.sources:
            overrides["projected_gravity"] = est.projected_gravity
        if "joints" in self.sources:
            qpos = self.d.qpos[self.maps["QADR"]] - self.maps["HOME"]
            qvel = self.d.qvel[self.maps["DOFADR"]].copy()
            qpos[self.filtered_slots] = est.q - self.maps["HOME"][self.filtered_slots]
            qvel[self.filtered_slots] = est.q_dot
            overrides["joint_pos_rel"] = qpos
            overrides["joint_vel_rel"] = qvel

        obs = rp.build_obs(self.m, self.d, self.policy, self.maps, self.cmd,
                           self.last_action, est=overrides)
        self.last_action = self.sess.run(
            None, {self.sess.get_inputs()[0].name: obs[None]})[0][0]
        self.d.ctrl[self.maps["ALL_AID"]] = self.maps["ALL_HOME"]
        self.d.ctrl[self.maps["AID"]] = self.maps["HOME"] + self.scale * self.last_action

        self._record(est)
        for k in range(rp.DECIMATION):
            mujoco.mj_step(self.m, self.d)
            if (k + 1) % self.est_every == 0:
                self.batch.append(self.reader.read(self.d))
        self._ramp_t += rp.DECIMATION * rp.DT

    # -- scoring ------------------------------------------------------------

    def _record(self, est):
        t = self.reader.truth(self.d)
        self.history.append({
            "t": self.d.time,
            "att_deg": attitude_error_deg(est.R, t["R"]),
            "tilt_deg": tilt_error_deg(est.R, t["R"]),
            "omega_err": np.linalg.norm(est.omega_body - t["omega"]),
            "omega_norm": np.linalg.norm(t["omega"]),
            "v_err": np.linalg.norm(est.v - t["v"]),
            "p_err": np.linalg.norm(est.p - t["p"]),
            # Signed VERTICAL error, kept separately from the 3D norm. The failure this run
            # exists to detect -- the estimate sinking into the ground -- is a one-sided z
            # error, and `p_err` (a norm, dominated by the forward-travel scale error) cannot
            # show it. `dz_abs` is what the RMS/max columns score.
            "dz": float(est.p[2] - t["p"][2]),
            "dz_abs": float(abs(est.p[2] - t["p"][2])),
            "q_err": float(np.abs(est.q - t["q"]).max()),
            "qd_err": float(np.abs(est.q_dot - t["q_dot"]).max()),
            "bias_norm": float(np.linalg.norm(est.bias)),
            "nis": est.nis,
            "trusted": est.trusted.sum(),
            "est_rpy": est.rpy,
            # Vectors, not just norms: a drift that is 11% of forward travel is a stride-length
            # scale error, while one that wanders in yaw is slip. The norm cannot tell them apart.
            "est_p": est.p.copy(),
            "true_p": t["p"].copy(),
            "est_v": est.v.copy(),
            "true_v": t["v"].copy(),
        })

    def est_status(self):
        h = self.history[-1]
        return (f"est: tilt_err={h['tilt_deg']:5.2f}deg att={h['att_deg']:5.2f}deg "
                f"|dw|={h['omega_err']:.3f} (|w|={h['omega_norm']:.2f}) "
                f"|dv|={h['v_err']:.3f} |dp|={h['p_err']:.3f} dz={h['dz']:+.3f} "
                f"feet={h['trusted']:.0f} NIS={h['nis']:.1f}")


def summarise(history, tail_frac=0.5):
    """Error summary over the whole run and over its last `tail_frac` (post-transient)."""
    if not history:
        return {}
    keys = ("att_deg", "tilt_deg", "omega_err", "v_err", "p_err", "dz_abs", "q_err", "qd_err")
    a = {k: np.array([h[k] for h in history]) for k in keys}
    n0 = int(len(history) * (1.0 - tail_frac))
    out = {}
    for k in keys:
        out[f"{k}_rms"] = float(np.sqrt((a[k] ** 2).mean()))
        out[f"{k}_max"] = float(a[k].max())
        out[f"{k}_tail_rms"] = float(np.sqrt((a[k][n0:] ** 2).mean()))
    out["final_p_err"] = float(history[-1]["p_err"])
    # Signed, so a report cannot hide which way it went: negative = the estimate believes it
    # is BELOW where it really is, i.e. sinking into the ground.
    dz = np.array([h["dz"] for h in history])
    out["final_dz"] = float(dz[-1])
    out["mean_dz"] = float(dz.mean())
    out["tail_mean_dz"] = float(dz[n0:].mean())
    out["min_dz"] = float(dz.min())
    out["max_dz"] = float(dz.max())
    out["duration_s"] = float(history[-1]["t"])
    return out


def print_summary(history):
    s = summarise(history)
    if not s:
        print("  (no ticks recorded)")
        return s
    print(f"\n  estimator vs sim truth over {s['duration_s']:.1f} s "
          f"({len(history)} control ticks)")
    print(f"    {'quantity':22s} {'rms':>10s} {'max':>10s} {'rms(last half)':>16s}")
    for k, label, unit in (("tilt_deg", "tilt error (policy)", "deg"),
                           ("att_deg", "attitude error", "deg"),
                           ("omega_err", "base gyro error", "rad/s"),
                           ("v_err", "base velocity error", "m/s"),
                           ("p_err", "base position error", "m"),
                           ("dz_abs", "VERTICAL |dz| error", "m"),
                           ("q_err", "joint pos error", "rad"),
                           ("qd_err", "joint vel error", "rad/s")):
        print(f"    {label + ' [' + unit + ']':22s} {s[k + '_rms']:10.4f} {s[k + '_max']:10.4f} "
              f"{s[k + '_tail_rms']:16.4f}")
    print(f"    signed dz (est_z - true_z) [m]: final={s['final_dz']:+.4f} "
          f"mean={s['mean_dz']:+.4f} tail_mean={s['tail_mean_dz']:+.4f} "
          f"range=[{s['min_dz']:+.4f}, {s['max_dz']:+.4f}]   "
          f"(negative = estimate sinking into the ground)")
    return s


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def attach_contactnet(fused, reader, ckpt, norm_path, *, verbose=True):
    """Attach a trained ContactNet to `fused`, returning the new estimator.

    The two paths are printed together on purpose: a checkpoint is only meaningful next to the
    normalization constants it was TRAINED under. Load a run-4 checkpoint against
    `data/norm_constants.npz` (the pre-DR constants) and the network sees a shifted input
    distribution -- nothing raises, and every number the run produces is worthless. The log line
    is the record of which pair was actually used.
    """
    import jax

    from invariant_estimation.contactnet import network, normalize, train
    from invariant_estimation.contactnet.config import ContactNetConfig
    from invariant_estimation.contactnet.features import build_subchain_indices

    # The trained config (artifacts/contactnet_run{2,4}.history.json): F=24, sigma_0=1e-4, and
    # the ContactNetConfig defaults for everything else (H=50, window_span_s=0.392, dt=1e-3,
    # widths=(256,256)). dt matches `run_policy.DT`, which is what turns the window SPAN into
    # ticks -- a mismatch there is a silently different network input.
    #
    # THE SOCKET MOVED (2026-07-29). `with_contactnet` now writes the network's
    # output into `contact_chol`, the stance-anchor PROCESS noise, not into
    # `contact_meas_chol`. Runs 1-4 were trained against the measurement socket,
    # where ~1e-4 is a sensible FK measurement std; the same number on the process
    # socket is the stance value, i.e. every anchor -- swing feet included --
    # asserted world-static. That is `freeze_contact_chol`, measured 10.2x worse
    # than not using contacts at all, and in the closed loop it is a fall.
    #
    # So this warns rather than silently running the demo into the ground. It does
    # not refuse: replaying an old checkpoint on the new socket deliberately is a
    # legitimate experiment, and `experiments/replay_eval.py --socket meas` is the
    # way to score one on the socket it was trained for.
    # `sigma_0` is STRUCTURAL ONLY here: `load_params` takes the tree structure
    # from `like` and overwrites every leaf, and nothing on the inference path
    # reads it (the warm-up fallback is `sensors.contact_chol`, not `sigma_0 * I`).
    # So it does not need to match the value the checkpoint was trained at --
    # run 5 used 1e-1 -- and the log line below reports the checkpoint, not this.
    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4)
    if "run1" in ckpt or "run2" in ckpt or "run3" in ckpt or "run4" in ckpt:
        print(f"WARNING: {ckpt} looks like a run-1..4 checkpoint, which was trained "
              f"on the MEASUREMENT socket. It is being attached to the PROCESS "
              f"socket, where its output range means something different -- expect "
              f"pinned anchors and a fall. See PORT_NOTES.md, 'ContactNet moves to "
              f"the process socket'.")
    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    params = train.load_params(ckpt, like)
    consts = normalize.load(norm_path)
    # `reader.unfiltered_names` is the same resolution `sim.collect._unfiltered_names` does
    # (`_dof_joint_names(mj_model, build.dof_anchor_unfiltered)`) -- one source of truth, never
    # a hand-written ankle list.
    sub = build_subchain_indices(fused.build.joint_names, reader.unfiltered_names)
    fused = me.with_contactnet(fused, params, cfg, consts, sub)
    if verbose:
        print(f"ContactNet: ATTACHED  ckpt={ckpt}  norm={norm_path} "
              f"({consts.n_ticks} ticks, source={consts.source!r})")
        print(f"            cfg F={cfg.F} H={cfg.H} span={cfg.window_span_s}s "
              f"stride={cfg.stride} dt={cfg.dt} widths={cfg.widths}; "
              f"warm-up {online_span(cfg)} ticks of the analytic heuristic before it acts")
    return fused


def online_span(cfg):
    from invariant_estimation.contactnet import online as cn_online
    return cn_online.span_ticks(cfg)


def make_estimated_loop(policy_name, *, with_visuals, sources=DEFAULT_SOURCES,
                        noise=None, est_dt=None, contact_meas_var=0.0,
                        stance_chol=1.0e-4, swing_chol=1.0e1,
                        contact_fk_unfiltered=True, est_every=1, verbose=True,
                        contactnet=None, contactnet_norm=None):
    t0 = time.time()
    policy = rp.load_policy(policy_name)
    m = rp.build_sim_model(policy, with_visuals=with_visuals, with_imu_sensors=True)
    maps = rp.make_maps(m, policy)

    urdf = rp.cycloid_forearm_urdf(rp.URDF)
    dt = est_dt or rp.DT * est_every
    fused = me.build_alex_fused_estimator_from_urdf(
        urdf, dt=dt, contact_meas_var=contact_meas_var,
        contact_fk_unfiltered=contact_fk_unfiltered)
    reader = SimSensorReader(m, fused, foot_geoms=rp.FOOT_GEOMS, dt=dt, noise=noise,
                             stance_chol=stance_chol, swing_chol=swing_chol)
    if verbose:
        print(f"estimator: {fused.n_joints} filtered joints {list(fused.build.joint_names)}")
        print(f"           {fused.build.n_imus} IMUs {list(fused.build.imu_names)}, "
              f"{fused.n_contacts} contacts, dt={est_dt or rp.DT}s "
              f"({1 / dt:.0f} Hz), unfiltered anchor joints "
              f"{list(reader.unfiltered_names)}")
        print(f"           policy reads {list(sources)} from the estimate; "
              f"noise={'on' if noise else 'off'}; contact FK uses "
              f"{'MEASURED' if fused.n_aux else 'qpos0-pinned'} off-path joints")
    # After the estimator is built (the seam takes its kinematics and base IMU from `fused`) and
    # before the loop -- `EstimatedLoop` builds the runtime, which closes over the fused step.
    if contactnet:
        fused = attach_contactnet(fused, reader, contactnet, contactnet_norm, verbose=verbose)
    elif verbose:
        print("ContactNet: not attached (analytic contact noise)")
    loop = EstimatedLoop(m, policy, maps, fused=fused, reader=reader, sources=sources,
                         est_every=est_every)
    loop.rt.warmup(loop.batch)      # pay the ~11 s XLA compile here, not on the first tick
    if verbose:
        print(f"           built + compiled in {time.time() - t0:.1f}s")
    return loop


def run_headless(loop, ticks, cmd=None, out=None, every=25, video=None, video_fps=50,
                 video_size=(1280, 720)):
    """Run `ticks` control ticks with no window, scoring the estimate; optionally record `video`.

    Recording renders the same sim the estimator is driving -- the robot on screen is being walked
    by the filter, not by ground truth (unless `--source truth`).
    """
    if cmd is not None:
        loop.cmd[0:3] = cmd
        loop.cmd[3] = 0.0 if np.any(np.abs(np.asarray(cmd)) > 1e-9) else 1.0
    rec = None
    if video:
        # The control loop runs at 50 Hz, so that is the ceiling on frame rate; a lower --video-fps
        # renders every `stride`-th tick and the result is still real time.
        control_hz = 1.0 / (rp.DECIMATION * rp.DT)
        stride = max(1, int(round(control_hz / video_fps)))
        w, h = video_size
        rec = rp.VideoRecorder(loop.m, video, body=loop.maps["BASE_BID"], width=w, height=h,
                               fps=control_hz / stride)
        print(f"  recording {w}x{h} @ {control_hz / stride:.0f} fps -> {video}")
    x0, y0 = loop.d.qpos[0], loop.d.qpos[1]
    t0 = time.time()
    for k in range(ticks):
        loop.control_tick()
        if rec is not None and k % stride == 0:
            rec.capture(loop.d)
        if k % every == 0:
            print(f"  t={k * rp.DECIMATION * rp.DT:5.2f}s  {loop.status()}\n"
                  f"            {loop.est_status()}")
        if not np.all(np.isfinite(loop.d.qpos)):
            print(f"  !! non-finite state at tick {k}")
            break
    print(f"\nfinal {loop.status()}"
          f"  travelled=({loop.d.qpos[0] - x0:+.2f},{loop.d.qpos[1] - y0:+.2f})m"
          f"  wall={time.time() - t0:.1f}s for {ticks * rp.DECIMATION * rp.DT:.1f}s of sim")
    if rec is not None:
        rec.close()
    s = print_summary(loop.history)
    if out:
        np.savez(out, **{k: np.array([h[k] for h in loop.history]) for k in loop.history[0]})
        print(f"  history -> {out}")
    return s


def run_viewer(loop):
    import mujoco.viewer
    pad = rp.Gamepad()
    print(f"  gamepad: {pad.name}\n{rp.GAMEPAD_HELP}" if pad.present
          else f"  no gamepad at {rp.JS_DEVICE}; use the keypad or the terminal\n{rp.KEYMAP_HELP}")
    print("  NOTE: a control tick costs ~35 ms against its 20 ms budget on CPU, so the viewer "
          "runs at roughly 0.6x speed.")
    rp.stdin_commands(loop)
    with mujoco.viewer.launch_passive(loop.m, loop.d, key_callback=loop.key) as v:
        last_print = 0.0
        while v.is_running():
            t0 = time.time()
            pad.apply(loop, rp.DECIMATION * rp.DT)
            loop.control_tick()
            v.sync()
            if t0 - last_print > 0.5:
                print(f"  {loop.status()} | {loop.est_status()}   ", end="\r", flush=True)
                last_print = t0
            sleep = rp.DECIMATION * rp.DT - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    print_summary(loop.history)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="baseline", choices=list(rp.POLICIES))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=500, help="control ticks (50 Hz)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--source", nargs="*", default=list(DEFAULT_SOURCES),
                    help=f"obs terms taken from the estimate; one of {SOURCES}, or 'truth' for "
                         "none (the estimator still runs and is scored)")
    ap.add_argument("--imu-noise", action="store_true",
                    help="corrupt the simulated IMUs/encoders (constant gyro bias + white noise)")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--stance-chol", type=float, default=1.0e-4)
    ap.add_argument("--swing-chol", type=float, default=1.0e1)
    ap.add_argument("--contact-meas-var", type=float, default=0.0,
                    help="isotropic floor on the InEKF contact measurement noise "
                         "(flight uses 1e-4; the port default is 0)")
    ap.add_argument("--contact-fk", choices=("measured", "pinned"), default="measured",
                    help="whether the InEKF's contact FK uses the MEASURED off-path joints "
                         "(the ankles) or pins them at qpos0 as the library default does")
    ap.add_argument("--est-every", type=int, default=1,
                    help="physics steps per estimator step: 1 = the 200 Hz physics rate "
                         "(faithful), 4 = one step per control tick (50 Hz), which makes the "
                         "viewer run at roughly real time")
    ap.add_argument("--contactnet", default=None, metavar="CKPT.npz",
                    help="attach a trained ContactNet (e.g. artifacts/contactnet_run4.npz); "
                         "off by default, which is the analytic contact noise")
    ap.add_argument("--contactnet-norm", default="data/dr/norm_constants.npz",
                    metavar="NORM.npz",
                    help="normalization constants -- MUST be the ones the checkpoint was "
                         "trained under (run 4 -> data/dr/norm_constants.npz; "
                         "run 2 -> data/norm_constants.npz). A mismatch shifts the network's "
                         "input distribution and nothing raises")
    ap.add_argument("--out", default=None, help="write the per-tick history to this .npz")
    ap.add_argument("--video", default=None, metavar="PATH.mp4",
                    help="record the run offscreen to an H.264 file (implies --headless; needs "
                         "ffmpeg on PATH)")
    ap.add_argument("--video-fps", type=float, default=50.0,
                    help="frame rate, capped by the 50 Hz control loop (default 50 = real time)")
    ap.add_argument("--video-size", default="1280x720", metavar="WxH")
    args = ap.parse_args()

    video_size = tuple(int(v) for v in args.video_size.lower().split("x"))

    sources = () if args.source == ["truth"] else tuple(args.source)
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise SystemExit(f"unknown --source {bad}; choose from {SOURCES} or 'truth'")

    # Recording is a headless run that still needs the visual meshes -- without them there is
    # nothing in the scene but the hidden collision boxes.
    headless = args.headless or bool(args.video)
    loop = make_estimated_loop(
        args.policy, with_visuals=not headless or bool(args.video), sources=sources,
        noise=IMUNoise(seed=args.noise_seed) if args.imu_noise else None,
        contact_meas_var=args.contact_meas_var,
        stance_chol=args.stance_chol, swing_chol=args.swing_chol,
        contact_fk_unfiltered=(args.contact_fk == "measured"), est_every=args.est_every,
        contactnet=args.contactnet, contactnet_norm=args.contactnet_norm)
    if headless:
        run_headless(loop, args.ticks, cmd=(args.vx, args.vy, args.yaw), out=args.out,
                     video=args.video, video_fps=args.video_fps, video_size=video_size)
    else:
        run_viewer(loop)
