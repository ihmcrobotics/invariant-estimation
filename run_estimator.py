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
Every run scores the estimate against the sim's own state (attitude, tilt-as-seen-by-the-policy,
gyro, velocity, position drift, joint state) and can dump the full history to `.npz`.

Speed (measured, 2026-07-27): a control tick costs ~17 ms against its 20 ms real-time budget while
walking, so the loop KEEPS REAL TIME on CPU (~1.2x) and ~2.2x on GPU. It used to cost ~35 ms and
run at 0.6x; pinning the ONNX session to one non-spinning thread (`run_policy._ort_session`) was
the fix — ORT's default pool was fighting XLA's. Building the estimator and compiling the step
costs ~55 s up front — `make_estimated_loop` compiles eagerly (`EstimatorRuntime.warmup`) so that
cost lands there and not as an 11 s freeze on the first tick. See RUNNING.md for the A/B tables,
`--ghost` for seeing the estimate, and `--realtime` for the (now largely unnecessary) threaded mode.

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
from invariant_estimation.sim.estimator_thread import ThreadedEstimator
from invariant_estimation.sim.ghost import Ghost
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

    def __init__(self, m, policy, maps, *, fused, reader, sources=DEFAULT_SOURCES, est_every=1,
                 threaded=False, max_backlog_ticks=2):
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
        # Threading is OPT-IN and viewer-only. `run_headless` and every test keep the synchronous
        # path, which is the one `test_sources_truth_bypasses_the_estimate` pins at atol=0.
        self.threaded = bool(threaded)
        self.te = None
        self._max_backlog_ticks = int(max_backlog_ticks)
        self._submitted = 0

    # -- one control tick ---------------------------------------------------

    def start_thread(self):
        """Hand the estimator to its own thread. Viewer entry points only; idempotent."""
        if not self.threaded or self.te is not None:
            return
        self.te = ThreadedEstimator(self.rt, max_backlog_ticks=self._max_backlog_ticks)
        # The priming batch is already exactly `substeps` long, so the first estimate is available
        # synchronously and the policy never has to read a null one.
        self.te.prime(self.batch, self.reader.truth(self.d))
        self._submitted = len(self.batch)
        self.batch = []

    def stop_thread(self):
        if self.te is not None:
            self.te.stop()
            self.te = None

    def current_estimate(self):
        """The estimate a viewer should draw: the published one when threaded, else the last."""
        if self.te is not None:
            pub = self.te.latest()
            return None if pub is None else pub.est
        return self.rt.last

    def control_tick(self):
        if self.threaded and self.te is not None:
            return self._tick_threaded()
        return self._tick_sync()

    def _tick_sync(self):
        est = self.rt.advance(self.batch)
        self.batch = []
        self._act_on(est)
        self._record(est)
        for k in range(rp.DECIMATION):
            mujoco.mj_step(self.m, self.d)
            if (k + 1) % self.est_every == 0:
                self.batch.append(self.reader.read(self.d))
        self._ramp_t += rp.DECIMATION * rp.DT

    def _tick_threaded(self):
        """Same tick, except the estimate is whatever the worker has published most recently.

        The estimate is therefore a few milliseconds STALE, which is the whole point: the sim
        thread never waits on the filter. Sensor reading stays here -- `reader.read` advances the
        contact-trust state machine and reads `MjData`, so it cannot move to the worker.
        """
        pub = self.te.latest()
        self._act_on(pub.est)
        # Scored against the truth PAIRED with that estimate, never the current MjData: the error
        # of a 3-tick-old estimate against the present state is not the estimator's error.
        self._record(pub.est, truth=pub.truth,
                     age_ticks=(self._submitted - pub.seq) / self.rt.substeps)
        for k in range(rp.DECIMATION):
            mujoco.mj_step(self.m, self.d)
            if (k + 1) % self.est_every == 0:
                # Truth is snapshotted HERE, alongside the sensors it belongs with (~10 us).
                self.te.submit(self.reader.read(self.d), self.reader.truth(self.d))
                self._submitted += 1
        self._ramp_t += rp.DECIMATION * rp.DT

    def _act_on(self, est):
        """Estimate -> observation overrides -> policy -> ctrl. Shared by both tick paths."""
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

    # -- scoring ------------------------------------------------------------

    def _record(self, est, truth=None, age_ticks=0.0):
        # `truth=None` means "score against the present", which is correct only when the estimate
        # was produced from the sensors of this very tick -- i.e. the synchronous path.
        t = self.reader.truth(self.d) if truth is None else truth
        self.history.append({
            "t": self.d.time,
            "age_ticks": float(age_ticks),
            "att_deg": attitude_error_deg(est.R, t["R"]),
            "tilt_deg": tilt_error_deg(est.R, t["R"]),
            "omega_err": np.linalg.norm(est.omega_body - t["omega"]),
            "omega_norm": np.linalg.norm(t["omega"]),
            "v_err": np.linalg.norm(est.v - t["v"]),
            "p_err": np.linalg.norm(est.p - t["p"]),
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
        # In threaded mode the age is the number worth watching: if it climbs, the estimator is
        # not keeping up and back-pressure is about to start slowing the sim.
        age = (f" age={h['age_ticks']:.1f}tk backlog={self.te.backlog}"
               if self.te is not None else "")
        return (f"est: tilt_err={h['tilt_deg']:5.2f}deg att={h['att_deg']:5.2f}deg "
                f"|dw|={h['omega_err']:.3f} (|w|={h['omega_norm']:.2f}) "
                f"|dv|={h['v_err']:.3f} |dp|={h['p_err']:.3f} feet={h['trusted']:.0f} "
                f"NIS={h['nis']:.1f}{age}")


def summarise(history, tail_frac=0.5):
    """Error summary over the whole run and over its last `tail_frac` (post-transient)."""
    if not history:
        return {}
    keys = ("att_deg", "tilt_deg", "omega_err", "v_err", "p_err", "q_err", "qd_err")
    a = {k: np.array([h[k] for h in history]) for k in keys}
    n0 = int(len(history) * (1.0 - tail_frac))
    out = {}
    for k in keys:
        out[f"{k}_rms"] = float(np.sqrt((a[k] ** 2).mean()))
        out[f"{k}_max"] = float(a[k].max())
        out[f"{k}_tail_rms"] = float(np.sqrt((a[k][n0:] ** 2).mean()))
    out["final_p_err"] = float(history[-1]["p_err"])
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
                           ("q_err", "joint pos error", "rad"),
                           ("qd_err", "joint vel error", "rad/s")):
        print(f"    {label + ' [' + unit + ']':22s} {s[k + '_rms']:10.4f} {s[k + '_max']:10.4f} "
              f"{s[k + '_tail_rms']:16.4f}")
    return s


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def make_estimated_loop(policy_name, *, with_visuals, sources=DEFAULT_SOURCES,
                        noise=None, est_dt=None, contact_meas_var=0.0,
                        stance_chol=1.0e-4, swing_chol=1.0e1,
                        contact_fk_unfiltered=True, est_every=1, verbose=True,
                        threaded=False, max_backlog_ticks=2):
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
    loop = EstimatedLoop(m, policy, maps, fused=fused, reader=reader, sources=sources,
                         est_every=est_every, threaded=threaded,
                         max_backlog_ticks=max_backlog_ticks)
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
    if loop.threaded:
        print("  --realtime: the estimator runs on its own thread, so the viewer keeps real time "
              "and the policy reads a slightly stale estimate (age is printed below).")
    else:
        print("  a control tick costs ~17 ms against its 20 ms budget on CPU (~9 ms on GPU), so "
              "this keeps real time. --realtime threads the estimator but buys ~nothing now.")
    if loop.ghost is not None:
        print(f"  ghost: {loop.ghost.mode} (keypad * or terminal 'g' to cycle)")
    loop.start_thread()
    rp.stdin_commands(loop)
    try:
        with mujoco.viewer.launch_passive(loop.m, loop.d, key_callback=loop.key) as v:
            last_print = 0.0
            while v.is_running():
                t0 = time.time()
                pad.apply(loop, rp.DECIMATION * rp.DT)
                loop.control_tick()
                est = loop.current_estimate()
                if loop.ghost is not None and est is not None:
                    # `user_scn` persists across frames and is NOT cleared by `sync()`, so the
                    # ghost would otherwise accumulate a robot every tick until it hit MAX_GEOM.
                    loop.ghost.update(est, loop.d)
                    v.user_scn.ngeom = 0
                    loop.ghost.draw(v.user_scn)
                v.sync()
                if t0 - last_print > 0.5:
                    print(f"  {loop.status()} | {loop.est_status()}   ", end="\r", flush=True)
                    last_print = t0
                sleep = rp.DECIMATION * rp.DT - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        loop.stop_thread()
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
    ap.add_argument("--out", default=None, help="write the per-tick history to this .npz")
    ap.add_argument("--video", default=None, metavar="PATH.mp4",
                    help="record the run offscreen to an H.264 file (implies --headless; needs "
                         "ffmpeg on PATH)")
    ap.add_argument("--video-fps", type=float, default=50.0,
                    help="frame rate, capped by the 50 Hz control loop (default 50 = real time)")
    ap.add_argument("--video-size", default="1280x720", metavar="WxH")
    ap.add_argument("--ghost", nargs="?", const="full", default="off",
                    choices=Ghost.MODES,
                    help="draw a translucent robot at the ESTIMATED state. 'full' is the honest "
                         "view (it sinks as position drifts); 'attitude' pins it at the true "
                         "position to isolate orientation error. Cycle live with keypad * or 'g'")
    ap.add_argument("--ghost-offset", type=float, default=0.0, metavar="METRES",
                    help="displace the ghost sideways for side-by-side viewing instead of overlaid")
    ap.add_argument("--wasd", action="store_true",
                    help="use the standalone WASD window instead of the passive viewer")
    ap.add_argument("--realtime", action="store_true",
                    help="run the estimator on its own thread so the viewer keeps real time. The "
                         "policy then reads a slightly STALE estimate, so a threaded run is not "
                         "bit-reproducible -- viewer only, never for recording numbers")
    ap.add_argument("--max-backlog-ticks", type=int, default=2,
                    help="how far the estimator may fall behind before the sim thread waits for "
                         "it (default 5 = 100 ms). Samples are never dropped, only delayed")
    args = ap.parse_args()

    video_size = tuple(int(v) for v in args.video_size.lower().split("x"))

    sources = () if args.source == ["truth"] else tuple(args.source)
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise SystemExit(f"unknown --source {bad}; choose from {SOURCES} or 'truth'")

    # Recording is a headless run that still needs the visual meshes -- without them there is
    # nothing in the scene but the hidden collision boxes.
    headless = args.headless or bool(args.video)
    # A threaded run is deliberately not reproducible; letting it write an .npz or a video would
    # put an irreproducible number somewhere it looks authoritative.
    if args.realtime and headless:
        raise SystemExit("--realtime is viewer-only: it makes the estimate slightly stale and the "
                         "run irreproducible. Drop --headless/--video, or drop --realtime.")
    loop = make_estimated_loop(
        args.policy, with_visuals=not headless or bool(args.video), sources=sources,
        noise=IMUNoise(seed=args.noise_seed) if args.imu_noise else None,
        contact_meas_var=args.contact_meas_var,
        stance_chol=args.stance_chol, swing_chol=args.swing_chol,
        contact_fk_unfiltered=(args.contact_fk == "measured"), est_every=args.est_every,
        threaded=args.realtime, max_backlog_ticks=args.max_backlog_ticks)
    # The ghost is a viewer feature: it draws, and headless has nothing to draw into.
    if args.ghost != "off" and headless:
        raise SystemExit("--ghost needs a viewer; drop --headless/--video")
    if not headless:
        loop.ghost = Ghost(loop.m, loop.maps, loop.filtered_slots,
                           offset=args.ghost_offset, mode=args.ghost)
    if headless:
        run_headless(loop, args.ticks, cmd=(args.vx, args.vy, args.yaw), out=args.out,
                     video=args.video, video_fps=args.video_fps, video_size=video_size)
    elif args.wasd:
        loop.start_thread()
        try:
            rp.run_free_viewer(args.policy, loop=loop)
        finally:
            loop.stop_thread()
        print_summary(loop.history)
    else:
        run_viewer(loop)
