r"""Does a trained ``Sigma_C`` make the filter better -- and better at *what*?

`check_sigma.py` reports what a checkpoint does to the Kalman gain.  That is a
property of the network, not of the filter, and it cannot distinguish two very
different situations:

* the network switched an axis off because the objective was degenerate
  (run 1, and the reason that gate exists), or
* the network switched an axis off because that axis' contact residual carries
  little velocity information -- a **correct** inference that a gain ratio looks
  identical to.

Only running the filter can tell them apart.  This replays one rollout from a
truth seed under two contact measurement noises, identical in every other
respect, and reports error in the quantity the loss scores (body-frame velocity)
*and* in the quantities it does not (position, height, tilt).

The distinction matters because `losses.l2_velocity` has no position term at all.
A `Sigma_C` that trades vertical accuracy for forward-velocity accuracy is
strictly rewarded by L2 and would show up here as "velocity better, height
worse".  CoCo-InEKF notes the same gap (arXiv 2605.15122 §IV-A: "Adding
additional loss terms based on the position and orientation states is
straightforward, however the InEKF's global drift over the episode must be
accounted for").

Note the rollouts are all in-sample -- 12 were collected and 12 trained on.  This
is a "does it help the filter" comparison, not a generalisation test.

Usage
-----
    uv run python -m experiments.replay_eval artifacts/contactnet_run2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from invariant_estimation.contactnet import (dataset, features, network,
                                             normalize, train)
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.inEKF import ekf as inekf_mod
from invariant_estimation.inEKF.filter import init_carry, make_step
from invariant_estimation.sim import collect

REPO = Path(__file__).resolve().parent.parent


def run_arm(fused, prep, cfg, t0: int, ticks: int, chol, P0) -> dict[str, float]:
    """Filter from a truth seed at `t0` for `ticks`, under contact chol `chol`."""
    xs = jax.tree.map(lambda a: jnp.asarray(a[t0:t0 + ticks]), prep.inputs)
    d0 = jnp.asarray(np.einsum("ij,kj->ki", prep.R_true[t0], prep.y_fk[t0])
                     + prep.p_true[t0][None, :])
    state0 = inekf_mod.initialize(
        fused.ekf, rotation=jnp.asarray(prep.R_true[t0]),
        velocity=jnp.asarray(prep.v_true[t0]),
        position=jnp.asarray(prep.p_true[t0]), contacts=d0)
    state0 = state0._replace(P=jnp.asarray(P0))

    step = make_step(fused.ekf, fused.kinematics)
    _, out = jax.lax.scan(step, init_carry(state0),
                          xs._replace(contact_meas_chol=chol))

    R_t = jnp.asarray(prep.R_true[t0:t0 + ticks])
    v_t = jnp.asarray(prep.v_true[t0:t0 + ticks])
    p_t = jnp.asarray(prep.p_true[t0:t0 + ticks])

    e_v = (jnp.einsum("kji,kj->ki", out.state.R, out.state.v)
           - jnp.einsum("kji,kj->ki", R_t, v_t))
    e_p = out.state.p - p_t
    # Tilt: angle between the estimated and true gravity direction in body frame.
    g_est = out.state.R[:, 2, :]
    g_true = R_t[:, 2, :]
    tilt = jnp.arccos(jnp.clip(jnp.sum(g_est * g_true, axis=-1), -1.0, 1.0))
    return {
        "vel_rms": float(jnp.sqrt(jnp.mean(jnp.sum(e_v ** 2, axis=-1)))),
        "pos_rms": float(jnp.sqrt(jnp.mean(jnp.sum(e_p ** 2, axis=-1)))),
        "height_rms": float(jnp.sqrt(jnp.mean(e_p[:, 2] ** 2))),
        "height_final": float(jnp.abs(e_p[-1, 2])),
        "tilt_deg": float(jnp.mean(tilt)) * 180.0 / np.pi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    # Explicit, because a hardcoded `data/` would silently score a network
    # trained on one dataset against a different one and look perfectly healthy.
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--cache", default=None, help="default <data>/cache")
    ap.add_argument("--norm", default=None, help="default <data>/norm_constants.npz")
    # P0 is MEASURED per dataset. Friction randomisation changes what the filter
    # covariance settles to, so scoring DR data against the original set's P0
    # would seed every arm from the wrong prior.
    ap.add_argument("--p0", default=str(REPO / "artifacts/p0.npz"))
    ap.add_argument("--ticks", type=int, default=20_000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=2)
    args = ap.parse_args()

    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4)
    fused = collect.build_collector(verbose=False).fused
    data = Path(args.data)
    cache = Path(args.cache) if args.cache else data / "cache"
    norm = normalize.load(str(Path(args.norm) if args.norm
                              else data / "norm_constants.npz"))
    preps = dataset.prepare(dataset.rollout_paths(data)[:args.rollouts],
                            norm, cfg, cache_dir=cache, verbose=False)
    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                        cfg.sigma_0, cfg.eps)
    params = train.load_params(args.checkpoint, like)
    fwd = jax.jit(jax.vmap(jax.vmap(
        lambda x: network.forward(params, x, cfg.eps))))

    P0 = np.load(args.p0)["P0"]
    print(f"replay eval: {args.checkpoint}  (P0 from {args.p0})")
    print(f"  data={data}  {len(preps)} rollouts x {args.starts} seeds, "
          f"{args.ticks * cfg.dt:.0f} s each\n")

    rows = {"heuristic": [], "trained": []}
    for prep in preps:
        # Window the whole stream once; the cache makes this a gather.
        w = features.window(jnp.asarray(prep.smoothed), cfg.H, cfg.stride)
        hi = min(prep.t_hi, prep.smoothed.shape[0] - args.ticks - 1)
        if hi <= prep.t_lo:
            continue
        for t0 in np.linspace(prep.t_lo, hi, args.starts).astype(int):
            t0 = int(t0)
            flat = w[t0:t0 + args.ticks].reshape(args.ticks, w.shape[1], -1)
            L_net = fwd(flat)
            L_heur = jnp.broadcast_to(
                cfg.sigma_0 * jnp.eye(3, dtype=jnp.float64), L_net.shape)
            rows["heuristic"].append(run_arm(fused, prep, cfg, t0, args.ticks, L_heur, P0))
            rows["trained"].append(run_arm(fused, prep, cfg, t0, args.ticks, L_net, P0))
            print(f"  {prep.name} t0={t0:6d}  "
                  f"vel {rows['heuristic'][-1]['vel_rms']:.4f} -> "
                  f"{rows['trained'][-1]['vel_rms']:.4f}   "
                  f"height {rows['heuristic'][-1]['height_rms']:.4f} -> "
                  f"{rows['trained'][-1]['height_rms']:.4f}")

    keys = ["vel_rms", "pos_rms", "height_rms", "height_final", "tilt_deg"]
    units = {"vel_rms": "m/s", "pos_rms": "m", "height_rms": "m",
             "height_final": "m", "tilt_deg": "deg"}
    print(f"\n{'metric':>14}  {'heuristic':>12}  {'trained':>12}  {'ratio':>8}")
    verdict = {}
    for k in keys:
        h = float(np.mean([r[k] for r in rows["heuristic"]]))
        t = float(np.mean([r[k] for r in rows["trained"]]))
        verdict[k] = t / h
        mark = "  <- SCORED BY L2" if k == "vel_rms" else ""
        print(f"{k:>14}  {h:12.5f}  {t:12.5f}  {t / h:8.3f}  {units[k]}{mark}")

    print()
    better_v = verdict["vel_rms"] < 0.95
    worse_p = verdict["height_rms"] > 1.05 or verdict["pos_rms"] > 1.05
    if better_v and worse_p:
        print("L2 IS DOING ITS JOB AND ONLY ITS JOB — velocity improved, position/"
              "height regressed. The trade is exactly what an objective with no "
              "position term rewards.")
    elif better_v:
        print("The trained Sigma_C helps on every axis measured. Whatever the "
              "gain table showed, the filter is better.")
    elif verdict["vel_rms"] > 1.05:
        print("The trained Sigma_C is WORSE at the very thing it optimised. "
              "Something upstream of the objective is wrong.")
    else:
        print("No meaningful change in either direction.")


if __name__ == "__main__":
    main()
