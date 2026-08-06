r"""Is the FK contact point actually world-static during stance?

The InEKF's contact model is ``d_i = const + noise``. If that is false in a
*signed* way, the filter is correct given a wrong model, and no covariance, loss
function or re-seed can repair it — only the model can.

This asks the question with **no filter in the loop**: take ground-truth base
pose and the recorded encoders, put the FK contact point in the world,

    d_i^true(t) = p_true(t) + R_true(t) · h_{p,i}(q(t)),

and watch it over each trusted stance phase. A perfect model gives a flat line.
The measured slope, converted to an equivalent base-velocity bias by the stance
duty cycle, is directly comparable to the -0.033 m/s the filter actually carries
(`experiments/reseed_drift.py`).

Usage
-----
    python3 experiments/anchor_static_check.py --ticks 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from invariant_estimation.pipeline import main_estimator as me


def stance_spans(contact_col, min_ticks=50):
    """[(start, stop)] index spans where one foot is continuously trusted."""
    up = np.asarray(contact_col) > 0.5
    edges = np.diff(up.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_ticks]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollout", default="data/flat_seed005.npz")
    ap.add_argument("--t0", type=int, default=16000)
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    args = ap.parse_args()

    import run_policy as rp

    fused = me.build_alex_fused_estimator_from_urdf(
        rp.cycloid_forearm_urdf(rp.URDF), dt=args.dt, contact_fk_unfiltered=True)

    z = np.load(Path(args.rollout), allow_pickle=True)
    sl = slice(args.t0, args.t0 + args.ticks)
    q = jnp.asarray(z["inputs.joint.q"][sl])
    qd = jnp.asarray(z["inputs.joint.q_dot"][sl])
    R_true = z["truth.R"][sl]
    p_true = z["truth.p"][sl]
    contact = z["sensors.contact"][sl]

    # `lax.map`, not `vmap`: vmapping MJX FK over 20000 ticks materialises the
    # whole batched pipeline at once and needs several GB. This is sequential and
    # allocates one tick at a time.
    y = np.asarray(jax.lax.map(lambda ab: fused.kinematics(*ab).y, (q, qd)))  # (T, N, 3)
    d_true = np.einsum("tij,tkj->tki", R_true, y) + p_true[:, None, :]       # (T, N, 3)

    print(f"{args.rollout}  ticks {args.t0}..{args.t0 + args.ticks} "
          f"({args.ticks * args.dt:.0f}s), {y.shape[1]} contacts\n")

    all_slopes, all_deltas, total_stance = [], [], 0
    for i in range(d_true.shape[1]):
        spans = stance_spans(contact[:, i])
        slopes, deltas, durs = [], [], []
        for a, b in spans:
            zz = d_true[a:b, i, 2]
            t = np.arange(zz.size) * args.dt
            slopes.append(np.polyfit(t, zz, 1)[0])
            deltas.append(zz[-1] - zz[0])
            durs.append((b - a) * args.dt)
        total_stance += sum(durs)
        all_slopes += slopes
        all_deltas += deltas
        print(f"  contact {i}: {len(spans):3d} stance phases, "
              f"mean {np.mean(durs):.3f}s   "
              f"height {d_true[:, i, 2][contact[:, i] > 0.5].mean() * 1e3:+7.1f} mm mean")
        print(f"              anchor z drift per stance : "
              f"{np.mean(deltas) * 1e3:+7.2f} mm   (slope {np.mean(slopes) * 1e3:+7.1f} mm/s)")

    duty = total_stance / (args.ticks * args.dt)
    equiv = np.mean(all_slopes) * duty
    print(f"\n  stance duty cycle          : {duty:.2f}  (sum over feet)")
    print(f"  duty-weighted anchor slope : {equiv:+.4f} m/s")
    print(f"  filter's vertical vel bias : -0.0329 m/s   "
          f"(experiments/reseed_drift.py, reseed OFF)")
    print(f"  ratio                      : {equiv / -0.0329:.2f}")

    # Split the stance into loading (first third) and unloading (last third):
    # a ratchet needs the two halves to be ASYMMETRIC, since a purely elastic
    # penetration that recovers before lift-off injects nothing net.
    load, unload = [], []
    for i in range(d_true.shape[1]):
        for a, b in stance_spans(contact[:, i]):
            zz = d_true[a:b, i, 2]
            k = max(zz.size // 3, 1)
            load.append(zz[k] - zz[0])
            unload.append(zz[-1] - zz[-k - 1])
    print(f"\n  loading   (first 1/3 of stance): {np.mean(load) * 1e3:+7.2f} mm")
    print(f"  unloading (last  1/3 of stance): {np.mean(unload) * 1e3:+7.2f} mm")
    print(f"  net per stance                 : {np.mean(all_deltas) * 1e3:+7.2f} mm")


if __name__ == "__main__":
    main()
