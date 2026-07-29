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

**Score a checkpoint on the socket it was TRAINED for.**  Runs 1-4 drove
``contact_meas_chol`` (``--socket meas``); everything from the process-socket move
on 2026-07-29 drives ``contact_chol`` (``--socket process``, the default).  The
baseline arm differs with it -- see the comment in `main` -- because "the shipped
filter's value" is zero on one socket and the Schmitt-switched heuristic on the
other.

Usage
-----
    uv run python -m experiments.replay_eval artifacts/contactnet_run5.npz \
        --data data/dr --p0 artifacts/p0_process_dr.npz
    uv run python -m experiments.replay_eval artifacts/contactnet_run4.npz \
        --data data/dr --p0 artifacts/p0_dr.npz --socket meas    # the old runs
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
from invariant_estimation.inEKF.contact import digest
from invariant_estimation.inEKF.filter import init_carry, make_step
from invariant_estimation.inEKF.propagate import propagate
from invariant_estimation.sim import collect

REPO = Path(__file__).resolve().parent.parent


def run_arm(fused, prep, cfg, t0: int, ticks: int, chol, P0,
            socket: str = "meas", want_traces: bool = False) -> dict[str, float]:
    r"""Filter from a truth seed at `t0` for `ticks`, under contact chol `chol`.

    `socket` selects **which** contact covariance `chol` drives, and it is the
    whole point of `experiments/process_socket_ablation.py`:

    ``"meas"``
        ``contact_meas_chol`` — the measurement noise ``N``, additive on the
        encoder term.  This is the socket ContactNet was trained on through
        run 4, and the arm-D control.
    ``"process"``
        ``contact_chol`` — the stance-anchor process noise that reaches ``Q_d``.
        The recorded heuristic value is **overwritten**, so an arm driving this
        socket must supply a full replacement, not a perturbation.

    Whichever socket is driven, the *other* one keeps its recorded value, so the
    two arms differ in exactly one field.

    `want_traces` adds the ``P_pp``/``P_dd`` vertical variances and the contact
    innovation to the result — `branch_out.md` §6's two instrumentation items,
    both of which the scan already computes and would otherwise throw away.
    Getting them here rather than from `sim/collect.py` means the §6 check costs
    a replay, not a re-collection of the whole dataset.
    """
    if socket not in ("meas", "process"):
        raise ValueError(f"socket must be 'meas' or 'process', got {socket!r}")
    xs = jax.tree.map(lambda a: jnp.asarray(a[t0:t0 + ticks]), prep.inputs)
    d0 = jnp.asarray(np.einsum("ij,kj->ki", prep.R_true[t0], prep.y_fk[t0])
                     + prep.p_true[t0][None, :])
    state0 = inekf_mod.initialize(
        fused.ekf, rotation=jnp.asarray(prep.R_true[t0]),
        velocity=jnp.asarray(prep.v_true[t0]),
        position=jnp.asarray(prep.p_true[t0]), contacts=d0)
    state0 = state0._replace(P=jnp.asarray(P0))

    step = make_step(fused.ekf, fused.kinematics)
    field = "contact_meas_chol" if socket == "meas" else "contact_chol"
    xs = xs._replace(**{field: chol})
    _, out = jax.lax.scan(step, init_carry(state0), xs)

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

    # The sink, in `z_bias_diag.py`'s own terms so the two are comparable: the
    # WORLD-frame vertical velocity error and the slope of the vertical position
    # error.  `e_v` above is body-frame (what the L2 objective scores); this is
    # not, on purpose -- the sink is a world-z quantity.
    e_vz = out.state.v[:, 2] - v_t[:, 2]
    t_s = jnp.arange(ticks, dtype=jnp.float64) * cfg.dt
    slope = jnp.polyfit(t_s, e_p[:, 2], 1)[0]

    res = {
        "vel_rms": float(jnp.sqrt(jnp.mean(jnp.sum(e_v ** 2, axis=-1)))),
        "pos_rms": float(jnp.sqrt(jnp.mean(jnp.sum(e_p ** 2, axis=-1)))),
        "height_rms": float(jnp.sqrt(jnp.mean(e_p[:, 2] ** 2))),
        "height_final": float(jnp.abs(e_p[-1, 2])),
        "tilt_deg": float(jnp.mean(tilt)) * 180.0 / np.pi,
        "e_vz": float(jnp.mean(e_vz)),
        "slope_e_pz": float(slope),
    }
    # Ratio ~= 1 is the signature that the sink IS the integrated velocity bias
    # (z_bias_diag.py diagnostic B).  A departure from 1 is a finding, not noise.
    res["ratio"] = res["slope_e_pz"] / res["e_vz"] if res["e_vz"] != 0.0 else float("nan")
    if want_traces:
        # Tangent layout is rotation-first (I4): p at 6, contact i at 9+3i, so
        # the vertical component of each is that index + 2.
        #
        # All three blocks, not just the diagonals: the anchor is seeded from the
        # base pose, so `P_pd` is nearly as large as either variance and the
        # naive `P_pp/(P_pp+P_dd)` reads exactly 0.5 forever.  The apportionment
        # the mechanism argument is about is the p-row of `P Hᵀ` over `H P Hᵀ`,
        # which for `H = [0 0 I −I]` is `(P_pp−P_pd)/(P_pp−P_pd−P_dp+P_dd)` —
        # the pure-prior share, exact in the `N → 0` limit.
        #
        # The `v` row is here too, and it is the one that matters: base POSITION
        # is unobservable in this filter, so `P_pp`, `P_pd` and `P_dd` are all
        # dominated by the same common global-position mode and agree to four
        # digits.  The sink is an integrated VELOCITY bias, and the velocity
        # correction per unit vertical residual is `(P_vp − P_vd)_z / (H P Hᵀ)_z`
        # — the row `branch_out.md` §0 names.
        #
        # PRIOR, not posterior.  `Σ_C` enters through `Q_d`, so reading the
        # end-of-tick covariance measures the quantity *after* the update has
        # undone most of what the process noise just did — which is exactly the
        # difference under test.  Reconstructed by calling the filter's own
        # `propagate` on each tick's predecessor state: the real function, not a
        # re-derivation of it.
        sigma_c = digest(xs.contact_chol, fused.ekf.params)
        prev = jax.tree.map(lambda a, s: jnp.concatenate([s[None], a[:-1]], axis=0),
                            out.state, state0)
        prior = jax.vmap(propagate, in_axes=(0, 0, 0, 0, None))(
            prev, xs.omega, xs.accel, sigma_c, fused.ekf.params)

        n_c = out.state.d.shape[1]
        z_d = [11 + 3 * i for i in range(n_c)]
        col = lambda r, cs: np.asarray(jnp.stack([prior.P[:, r, c] for c in cs], axis=1))  # noqa: E731
        res["P_pp_z"] = np.asarray(prior.P[:, 8, 8])
        res["P_pd_z"] = col(8, z_d)
        res["P_dd_z"] = np.asarray(jnp.stack([prior.P[:, j, j] for j in z_d], axis=1))
        res["P_vp_z"] = np.asarray(prior.P[:, 5, 8])
        res["P_vd_z"] = col(5, z_d)
        # `contact_innovation` is (3N,) per tick, contact-major: contact i's z
        # residual is 3i+2.  This is the quantity §1's mechanism argument infers
        # the sign of; logging it makes the per-step dose directly observable.
        res["nu_z"] = np.asarray(out.contact_innovation[:, 2::3])
    return res


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
    # Which socket the checkpoint is scored on. `process` is where the deployed
    # seam writes since 2026-07-29; `meas` is what runs 1-4 were trained against,
    # and is the only fair way to score one of those.
    ap.add_argument("--socket", default="process", choices=["process", "meas"])
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
    print(f"replay eval: {args.checkpoint}  (P0 from {args.p0}, "
          f"socket={args.socket})")
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
            # The baseline is "what the shipped filter puts in this socket", and
            # that differs BY SOCKET. On `meas` the shipped value is zero, and
            # `sigma_0 * I` is the network's own initialization -- three orders
            # below `J Sigma_q J^T`, so indistinguishable from zero and a fair
            # stand-in. On `process` the shipped value is the recorded
            # Schmitt-switched heuristic, and `sigma_0 * I` there would be the
            # stance constant, i.e. `freeze_contact_chol` -- a straw man that
            # would flatter any checkpoint.
            if args.socket == "meas":
                L_heur = jnp.broadcast_to(
                    cfg.sigma_0 * jnp.eye(3, dtype=jnp.float64), L_net.shape)
            else:
                L_heur = jnp.asarray(prep.inputs.contact_chol[t0:t0 + args.ticks])
            rows["heuristic"].append(run_arm(fused, prep, cfg, t0, args.ticks, L_heur,
                                             P0, socket=args.socket))
            rows["trained"].append(run_arm(fused, prep, cfg, t0, args.ticks, L_net,
                                           P0, socket=args.socket))
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
