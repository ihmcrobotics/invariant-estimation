r"""Does the touchdown re-seed remove the vertical drift?  (PORT_NOTES Finding 2)

Replays a **recorded** rollout through the InEKF alone — the same construction
`contactnet.dataset.measure_p0` uses — twice, from a bit-identical seed, with the
touchdown re-seed off and on.  Recorded input rather than a live sim so the two
arms see *exactly* the same sensor stream: the estimate does reach the policy
(`--source`), so a closed-loop A/B would compare two different trajectories and
could not attribute a difference to the filter.

What it reports, and why each number is here
--------------------------------------------
* **drift rate** — a linear fit of the vertical error.  Finding 2 measured
  ~0.09 m/s, *linear*, while horizontal odometry was fine.
* **linear vs √t** — the diagnostic that decides the whole question.  A biased
  innovation being gain-split into the base gives error ∝ t; an honest random
  walk in an unobservable direction gives ∝ √t.  Re-seed can only fix the first.
  Reported as the RMS of each fit's residual: lower is the better description.
* **touchdown concentration** — the share of vertical error accumulated in the
  ±40 ms around a contact rising edge, against that window's share of ticks.
  Finding 2: 63 % of the error in 25 % of the ticks, 5x the background rate.
* **fires** — how many times the latch actually fired.  A zero here means the
  experiment measured nothing and the latch wiring is wrong, not that the
  re-seed does not help.

Usage
-----
    python3 experiments/reseed_drift.py --rollout data/flat_seed005.npz --ticks 20000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from invariant_estimation.inEKF import ekf as inekf_mod
from invariant_estimation.inEKF.filter import (
    InEKFInputs,
    JointFilterOutput,
    init_carry,
    make_step,
)
from invariant_estimation.inEKF.contact import default_rolling_anchor_params
from invariant_estimation.inEKF.reseed import default_reseed_params
from invariant_estimation.pipeline import main_estimator as me


def load_rollout(path: Path, t0: int, ticks: int):
    """Recorded InEKF inputs + ground truth for ``[t0, t0+ticks)``."""
    z = np.load(path, allow_pickle=True)
    sl = slice(t0, t0 + ticks)
    joint = JointFilterOutput(
        q=z["inputs.joint.q"][sl],
        q_dot=z["inputs.joint.q_dot"][sl],
        sigma_q=z["inputs.joint.sigma_q"][sl],
        sigma_q_dot=z["inputs.joint.sigma_q_dot"][sl],
    )
    inputs = InEKFInputs(
        omega=z["inputs.omega"][sl],
        accel=z["inputs.accel"][sl],
        raw_omega=z["inputs.raw_omega"][sl],
        joint=joint,
        contact_chol=z["inputs.contact_chol"][sl],
        contact_prob=z["sensors.contact"][sl],
    )
    truth = dict(R=z["truth.R"][sl], p=z["truth.p"][sl], v=z["truth.v"][sl])
    return inputs, truth


def build(dt, *, rolling_enabled=False, tau=None, sigma_r=None):
    """Build the fused estimator for one arm.

    Rebuilt per arm rather than reused, because `rolling.enabled` is consumed at
    BUILD time by the kinematics seam (it decides whether to stage out the extra
    FK differentiation at all), not just at step time. Reusing one `fused` across
    arms would silently give the omega arm a kinematics with no `omega_rel`.
    """
    import run_policy as rp

    return me.build_alex_fused_estimator_from_urdf(
        rp.cycloid_forearm_urdf(rp.URDF), dt=dt, contact_fk_unfiltered=True,
        rolling=default_rolling_anchor_params(
            enabled=rolling_enabled, tau=tau, sigma_r=sigma_r)
        if rolling_enabled else None,
    )


def run_arm(fused, inputs, truth, *, reseed_enabled=False, **_):
    """One replay.  Returns a dict of NumPy per-tick outputs."""
    ekf = fused.ekf
    if reseed_enabled:
        ekf = ekf._replace(reseed=default_reseed_params(enabled=True))

    # Seed exactly as `dataset.make_segment` does: truth pose, contacts placed by
    # FK so the filter starts self-consistent (zero contact innovation at t0).
    R0, p0, v0 = truth["R"][0], truth["p"][0], truth["v"][0]
    y0 = np.asarray(fused.kinematics(jnp.asarray(inputs.joint.q[0]),
                                     jnp.asarray(inputs.joint.q_dot[0])).y)
    d0 = np.einsum("ij,kj->ki", R0, y0) + p0[None, :]
    state0 = inekf_mod.initialize(
        ekf, rotation=jnp.asarray(R0), velocity=jnp.asarray(v0),
        position=jnp.asarray(p0), contacts=jnp.asarray(d0),
    )

    xs = jax.tree.map(jnp.asarray, inputs)
    _, out = jax.lax.scan(make_step(ekf, fused.kinematics), init_carry(state0, ekf), xs)
    return dict(
        est_p=np.asarray(out.state.p),
        est_v=np.asarray(out.state.v),
        est_d=np.asarray(out.state.d),
        nu=np.asarray(out.contact_innovation),
        nis=np.asarray(out.contact_diagnostics.nis),
        fires=np.asarray(out.reseed_fire).sum(),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def fit_rates(err, dt):
    """(linear slope, linear residual RMS, sqrt residual RMS) for a 1-D error."""
    t = np.arange(err.size) * dt
    slope = np.polyfit(t, err, 1)
    lin_res = err - np.polyval(slope, t)
    # sign-aware sqrt basis: a random walk's |error| grows as sqrt(t)
    s = np.sqrt(t)
    coef = np.polyfit(s, err, 1)
    sqrt_res = err - np.polyval(coef, s)
    return slope[0], float(np.sqrt((lin_res ** 2).mean())), float(np.sqrt((sqrt_res ** 2).mean()))


def touchdown_concentration(err_z, contact, dt, half_window_s=0.04):
    """Share of |Δ vertical error| accumulated near a contact rising edge.

    Edges are detected **per foot** and unioned. An earlier version OR-ed the
    feet *before* differencing, which on an alternating gait is high almost
    always and reported 3 edges in 20 s instead of ~45 — a broken detector that
    made the concentration test agree on a zero.
    """
    up = contact > 0.5                                      # (T, N)
    rising = np.zeros(contact.shape[0], dtype=bool)
    rising[1:] = (up[1:] & ~up[:-1]).any(axis=1)
    h = int(round(half_window_s / dt))
    near = np.zeros_like(rising)
    for k in np.flatnonzero(rising):
        near[max(0, k - h): k + h + 1] = True
    d = np.abs(np.diff(err_z, prepend=err_z[0]))
    share_err = d[near].sum() / max(d.sum(), 1e-12)
    return share_err, near.mean(), int(rising.sum())


def report(name, arm, truth, contact, dt):
    est_p, est_v, est_d = arm["est_p"], arm["est_v"], arm["est_d"]
    err = est_p - truth["p"]
    ez = err[:, 2]
    slope, lin_res, sqrt_res = fit_rates(ez, dt)
    share_err, share_ticks, n_td = touchdown_concentration(ez, contact, dt)
    horiz = np.linalg.norm(err[-1, :2])

    # Common mode: does the base sink WITH its anchors (invisible to H), or
    # AGAINST them (visible, and therefore a gain/noise problem instead)?
    anchor_z = est_d[:, :, 2].mean(axis=1)
    anchor_sink = anchor_z[-1] - anchor_z[0]
    common = anchor_sink / ez[-1] if abs(ez[-1]) > 1e-9 else float("nan")

    vz_err = est_v[:, 2] - truth["v"][:, 2]

    print(f"\n--- {name} ---")
    print(f"  reseed fires             : {arm['fires']:.0f}   ({n_td} touchdown edges)")
    print(f"  final vertical error     : {ez[-1]:+.4f} m")
    print(f"  vertical drift rate      : {slope:+.4f} m/s   (linear fit)")
    print(f"  final horizontal error   : {horiz:.4f} m")
    print(f"  fit residual RMS  linear : {lin_res:.4f} m")
    print(f"                    sqrt(t): {sqrt_res:.4f} m   "
          f"-> {'LINEAR (biased)' if lin_res < sqrt_res else 'SQRT (diffusive)'}")
    print(f"  |Δz| near touchdown      : {share_err:5.1%} of error in "
          f"{share_ticks:5.1%} of ticks  ({share_err / max(share_ticks, 1e-9):.1f}x)")
    print(f"  anchors sank             : {anchor_sink:+.4f} m "
          f"({common:.2f}x the base error -> "
          f"{'COMMON MODE' if 0.8 < common < 1.2 else 'not common mode'})")
    print(f"  mean vertical vel error  : {vz_err.mean():+.5f} m/s "
          f"(x {dt * len(vz_err):.0f}s = {vz_err.mean() * dt * len(vz_err):+.3f} m)")
    print(f"  mean contact NIS         : {np.nanmean(arm['nis']):.3f}  "
          f"| RMS innovation {np.sqrt((arm['nu'] ** 2).mean()) * 1e3:.3f} mm")
    return dict(final_z=ez[-1], rate=slope, horiz=horiz, fires=arm["fires"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollout", default="data/flat_seed005.npz")
    ap.add_argument("--t0", type=int, default=16000, help="skip the joint-KF warm-up")
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--arm", choices=("off", "on", "omega", "both", "all"), default="all",
                    help="run a single arm and cache it to --cache; 'all' scores them")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="rolling-anchor correlation time [s]")
    ap.add_argument("--sigma_r", type=float, default=0.0985,
                    help="rolling-anchor lever-arm prior std [m]")
    ap.add_argument("--cache", default="/tmp/reseed_arms",
                    help="directory holding per-arm .npz results between invocations")
    args = ap.parse_args()

    inputs, truth = load_rollout(Path(args.rollout), args.t0, args.ticks)
    contact = np.asarray(inputs.contact_prob)
    print(f"replaying {args.ticks} ticks ({args.ticks * args.dt:.1f}s) "
          f"of {args.rollout} from t0={args.t0}")

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    arms = {
        "off":    dict(),                                          # shipped filter
        "on":     dict(reseed_enabled=True),                       # touchdown re-seed
        "omega":  dict(rolling_enabled=True, tau=args.tau,         # rolling anchor
                       sigma_r=args.sigma_r),
        "both":   dict(reseed_enabled=True, rolling_enabled=True,
                       tau=args.tau, sigma_r=args.sigma_r),
    }
    todo = ["off", "on", "omega"] if args.arm == "all" else [args.arm]

    # Each arm is run at most once and cached, so a machine that cannot fit both
    # traces in one process (or one wall-clock budget) can still produce the
    # comparison by invoking --arm off and --arm on separately.
    for key in todo:
        f = cache / f"{Path(args.rollout).stem}_{key}_{args.t0}_{args.ticks}.npz"
        if f.exists():
            print(f"[{key}] cached -> {f}")
            continue
        t = time.time()
        fused = build(args.dt, **{k: v for k, v in arms[key].items()
                                  if k != "reseed_enabled"})
        out = run_arm(fused, inputs, truth, **arms[key])
        print(f"[{key}] {time.time() - t:.1f}s")
        np.savez(f, **out)

    if args.arm != "all":
        return

    # Harness validation: the recorded rollout carries the estimate its ORIGINAL
    # full fused run produced (`aux.est_p`). If this replay's OFF arm does not
    # reproduce that drift, the replay is measuring its own artefact and nothing
    # below means anything.
    ref = np.load(Path(args.rollout), allow_pickle=True)["aux.est_p"][
        args.t0: args.t0 + args.ticks]
    ref_rate = fit_rates(ref[:, 2] - truth["p"][:, 2], args.dt)[0]
    print(f"\nrecorded fused-run drift (aux.est_p): {ref_rate:+.4f} m/s"
          f"  <- the replay's OFF arm should match this")

    results = {}
    labels = (("off", "shipped (no reseed, no omega-term)"),
              ("on", "touchdown reseed"),
              ("omega", f"rolling anchor  tau={args.tau} sigma_r={args.sigma_r}"))
    for key, name in labels:
        z = np.load(cache / f"{Path(args.rollout).stem}_{key}_{args.t0}_{args.ticks}.npz")
        results[name] = report(name, z, truth, contact, args.dt)

    # Where the vertical error comes from, in quarters: a velocity bias that is
    # already present in Q1 is not something a touchdown-timed fix can reach.
    off_v = np.load(cache / f"{Path(args.rollout).stem}_off_{args.t0}_{args.ticks}.npz")["est_v"][:, 2]
    q = np.array_split(off_v - truth["v"][:, 2], 4)
    print("\n  vertical velocity error by quarter (reseed OFF): "
          + "  ".join(f"Q{i + 1} {x.mean():+.4f}" for i, x in enumerate(q)) + " m/s")

    base = results[labels[0][1]]
    print("\n=== verdict (vs the shipped filter) ===")
    print(f"  {'arm':38s} {'drift m/s':>10s} {'x':>6s} {'final z':>9s} {'horiz':>8s}")
    for _, name in labels:
        r = results[name]
        print(f"  {name:38s} {r['rate']:+10.4f} "
              f"{abs(r['rate']) / max(abs(base['rate']), 1e-12):6.2f} "
              f"{r['final_z']:+9.4f} {r['horiz']:8.4f}")


if __name__ == "__main__":
    main()
