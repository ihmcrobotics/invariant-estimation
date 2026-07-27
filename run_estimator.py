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

Speed: the estimator costs ~20 ms per physics step on CPU (MJX FK + CRB per tick), i.e. ~4x
slower than real time at 200 Hz. Headless runs are unaffected; the viewer runs in slow motion.

Read `run_policy.py` first — the sim, the policy contract and every magic number live there.
"""
import argparse
import os
import time

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

    def __init__(self, m, policy, maps, *, fused, reader, sources=DEFAULT_SOURCES):
        super().__init__(m, policy, maps)
        self.reader = reader
        self.sources = tuple(sources)
        self.rt = EstimatorRuntime(fused, reader, substeps=rp.DECIMATION)
        self.rt.seed(self.d)
        # t=0 the robot is at rest and the sensors are already meaningful, so prime the batch
        # with DECIMATION copies of the rest reading rather than special-casing the first tick.
        self.batch = [reader.read(self.d) for _ in range(rp.DECIMATION)]
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
        for _ in range(rp.DECIMATION):
            mujoco.mj_step(self.m, self.d)
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
            "q_err": float(np.abs(est.q - t["q"]).max()),
            "qd_err": float(np.abs(est.q_dot - t["q_dot"]).max()),
            "bias_norm": float(np.linalg.norm(est.bias)),
            "nis": est.nis,
            "trusted": est.trusted.sum(),
            "est_rpy": est.rpy,
            "true_z": t["p"][2],
            "est_z": est.p[2],
        })

    def est_status(self):
        h = self.history[-1]
        return (f"est: tilt_err={h['tilt_deg']:5.2f}deg att={h['att_deg']:5.2f}deg "
                f"|dw|={h['omega_err']:.3f} (|w|={h['omega_norm']:.2f}) "
                f"|dv|={h['v_err']:.3f} |dp|={h['p_err']:.3f} feet={h['trusted']:.0f} "
                f"NIS={h['nis']:.1f}")


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
                        contact_fk_unfiltered=True, verbose=True):
    t0 = time.time()
    policy = rp.load_policy(policy_name)
    m = rp.build_sim_model(policy, with_visuals=with_visuals, with_imu_sensors=True)
    maps = rp.make_maps(m, policy)

    urdf = rp.cycloid_forearm_urdf(rp.URDF)
    fused = me.build_alex_fused_estimator_from_urdf(
        urdf, dt=est_dt or rp.DT, contact_meas_var=contact_meas_var,
        contact_fk_unfiltered=contact_fk_unfiltered)
    reader = SimSensorReader(m, fused, foot_geoms=rp.FOOT_GEOMS, dt=rp.DT, noise=noise,
                             stance_chol=stance_chol, swing_chol=swing_chol)
    if verbose:
        print(f"estimator: {fused.n_joints} filtered joints {list(fused.build.joint_names)}")
        print(f"           {fused.build.n_imus} IMUs {list(fused.build.imu_names)}, "
              f"{fused.n_contacts} contacts, dt={est_dt or rp.DT}s "
              f"({1 / (est_dt or rp.DT):.0f} Hz), unfiltered anchor joints "
              f"{list(reader.unfiltered_names)}")
        print(f"           policy reads {list(sources)} from the estimate; "
              f"noise={'on' if noise else 'off'}; contact FK uses "
              f"{'MEASURED' if fused.n_aux else 'qpos0-pinned'} off-path joints")
    loop = EstimatedLoop(m, policy, maps, fused=fused, reader=reader, sources=sources)
    if verbose:
        print(f"           built + compiled in {time.time() - t0:.1f}s")
    return loop


def run_headless(loop, ticks, cmd=None, out=None, every=25):
    if cmd is not None:
        loop.cmd[0:3] = cmd
        loop.cmd[3] = 0.0 if np.any(np.abs(np.asarray(cmd)) > 1e-9) else 1.0
    x0, y0 = loop.d.qpos[0], loop.d.qpos[1]
    t0 = time.time()
    for k in range(ticks):
        loop.control_tick()
        if k % every == 0:
            print(f"  t={k * rp.DECIMATION * rp.DT:5.2f}s  {loop.status()}\n"
                  f"            {loop.est_status()}")
        if not np.all(np.isfinite(loop.d.qpos)):
            print(f"  !! non-finite state at tick {k}")
            break
    print(f"\nfinal {loop.status()}"
          f"  travelled=({loop.d.qpos[0] - x0:+.2f},{loop.d.qpos[1] - y0:+.2f})m"
          f"  wall={time.time() - t0:.1f}s for {ticks * rp.DECIMATION * rp.DT:.1f}s of sim")
    s = print_summary(loop.history)
    if out:
        keys = [k for k in loop.history[0] if k != "est_rpy"]
        np.savez(out, **{k: np.array([h[k] for h in loop.history]) for k in keys},
                 est_rpy=np.array([h["est_rpy"] for h in loop.history]))
        print(f"  history -> {out}")
    return s


def run_viewer(loop):
    import mujoco.viewer
    pad = rp.Gamepad()
    print(f"  gamepad: {pad.name}\n{rp.GAMEPAD_HELP}" if pad.present
          else f"  no gamepad at {rp.JS_DEVICE}; use the keypad or the terminal\n{rp.KEYMAP_HELP}")
    print("  NOTE: the estimator costs ~80 ms per control tick on CPU, so the viewer runs at "
          "roughly 1/4 speed.")
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
    ap.add_argument("--out", default=None, help="write the per-tick history to this .npz")
    args = ap.parse_args()

    sources = () if args.source == ["truth"] else tuple(args.source)
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise SystemExit(f"unknown --source {bad}; choose from {SOURCES} or 'truth'")

    loop = make_estimated_loop(
        args.policy, with_visuals=not args.headless, sources=sources,
        noise=IMUNoise(seed=args.noise_seed) if args.imu_noise else None,
        contact_meas_var=args.contact_meas_var,
        stance_chol=args.stance_chol, swing_chol=args.swing_chol,
        contact_fk_unfiltered=(args.contact_fk == "measured"))
    if args.headless:
        run_headless(loop, args.ticks, cmd=(args.vx, args.vy, args.yaw), out=args.out)
    else:
        run_viewer(loop)
