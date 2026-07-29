r"""What is actually in the ContactNet training set — slip, contact events, terrain.

PORT_NOTES has said since 2026-07-28 that the number to reason about is
**independent contact events, not samples**, and that the specific gap is
friction: `Sigma_C` is contact *measurement* uncertainty (sole compliance,
geometry, slip), the dataset varies terrain but not `geom_friction`, and

    "If the sim never slips, ContactNet can only learn `Sigma_C` ~ constant,
     modulated slightly by load and geometry.  That is a real result but a small
     one, and a good L2 number would be exactly what you would expect for the
     wrong reason."

It also said the measurement was ~20 minutes and should happen before collecting
a larger dataset.  This is that measurement.

Everything here is read off the saved rollouts — no sim, no MJX, no estimator.
The signals used are the ones `sim/collect.py` already records:

* ``sensors.contact`` — the `ContactTrust` per-foot trust in [0, 1], which is the
  same signal the filter's process socket switches on.  A **contact event** is a
  rising edge of ``trust > 0.5``.
Slip: **not answerable from the saved rollouts** — measured and reported here
-----------------------------------------------------------------------------
The obvious reconstruction is the world velocity of the FK contact point,
``W v_C = v_B + R(omega x p_bc) + R v_bc``, which is exactly zero for a planted
foot.  It does not come out zero.  In deep mid-stance (trust > 0.95, eroded
60 ms from each edge) it reads **0.29 m/s p50 against a 0.406 m/s base speed**,
and an independent finite difference of the world contact point agrees to three
digits — so this is not an algebra error.

It is also not 0.29 m/s of sliding.  ``p_bc`` is FK to the **sole site**, not to
the contact patch, so a foot rolling heel-to-toe translates that site through the
world without sliding at all; over a ~0.64 s stance a few tens of cm of roll is
0.2-0.3 m/s on its own.  Differentiating a noisy FK at 1 kHz adds more.  Nothing
in the recorded signals separates the two: the rollouts store no contact-patch
position and no contact forces.

So this script reports the quantity and refuses to call it slip.  **The slip
measurement PORT_NOTES asked for has to go into the collector** — per-contact
tangential sole velocity while loaded, computed where MuJoCo still knows the
contact geometry — and cannot be recovered afterwards.

Usage
-----
    uv run python -m experiments.dataset_stats
    uv run python -m experiments.dataset_stats --data data --cache data/cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _events(trust: np.ndarray, hi: float = 0.5) -> np.ndarray:
    """Rising edges of ``trust > hi``, per foot: ``(n_feet,)`` counts."""
    on = trust > hi
    rises = on[1:] & ~on[:-1]
    return rises.sum(axis=0)


def _duty(trust: np.ndarray, hi: float = 0.5) -> np.ndarray:
    return (trust > hi).mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--cache", default=str(REPO / "data/cache"))
    ap.add_argument("--warmup", type=int, default=16_000,
                    help="ticks to drop; the joint-KF bias plateau")
    args = ap.parse_args()

    paths = sorted(Path(args.data).glob("*.npz"))
    paths = [p for p in paths if p.name != "norm_constants.npz"]

    names = None
    tot_events = np.zeros(2)
    tot_ticks = 0
    slip_all, load_all = [], []
    rows = []

    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            trust = np.asarray(z["sensors.contact"])          # (T, n_feet)
        cache = Path(args.cache) / f"{p.stem}_feat.npz"
        if not cache.exists():
            print(f"  {p.name}: no channel cache, skipped")
            continue
        with np.load(cache, allow_pickle=True) as z:
            chan = np.asarray(z["channels"])                  # (T, N_c, F)
            names = names or [str(x) for x in z["names"]]

        w = slice(args.warmup, None)
        trust, chan = trust[w], chan[w]
        iy = names.index("v_bc_y")
        ix, iz = names.index("v_bc_x"), names.index("v_bc_z")
        v_tan = np.linalg.norm(chan[..., [ix, iy]], axis=-1)   # (T, N_c)

        loaded = trust > 0.5
        slip = v_tan[loaded]
        slip_all.append(slip)
        load_all.append(v_tan[~loaded])

        ev = _events(trust)
        tot_events += ev
        tot_ticks += trust.shape[0]
        rows.append((f"{meta['terrain']}/s{meta['seed']}", ev.sum(),
                     _duty(trust).mean(), float(np.median(slip)),
                     float(np.percentile(slip, 99)), meta["relief_m"],
                     meta["travelled_m"], meta["tilt_max_deg"]))

    print(f"{'rollout':>24} {'events':>7} {'duty':>6} {'slip p50':>10} "
          f"{'slip p99':>10} {'relief':>8} {'travel':>7} {'tilt':>6}")
    for r in rows:
        print(f"{r[0]:>24} {r[1]:7d} {r[2]:6.3f} {r[3]:10.4f} {r[4]:10.4f} "
              f"{r[5]:8.3f} {r[6]:7.2f} {r[7]:6.2f}")

    slip = np.concatenate(slip_all)
    swing = np.concatenate(load_all)
    print(f"\ntotals over {len(rows)} rollouts, {tot_ticks} post-warm-up ticks:")
    print(f"  contact events (rising edges of trust>0.5): "
          f"{int(tot_events.sum())}  ({tot_events})")
    print(f"  usable trajectory: {tot_ticks * 1e-3:.0f} s")

    print(f"\ntangential BODY-frame contact speed |v_bc_xy| [m/s] "
          f"-- NOT slip, see the module docstring:")
    for tag, v in (("LOADED (trust>0.5)", slip), ("swing  (trust<=0.5)", swing)):
        q = np.percentile(v, [50, 90, 99, 99.9])
        print(f"  {tag}:  p50 {q[0]:.4f}  p90 {q[1]:.4f}  p99 {q[2]:.4f}  "
              f"p99.9 {q[3]:.4f}  max {v.max():.4f}")
    print("  A planted foot reads ~ the base speed here by construction, because")
    print("  `v_bc` is a BODY-frame derivative. The world-frame reconstruction is")
    print("  in the docstring; it does not isolate slip either.")

    spread = lambda i: (max(r[i] for r in rows) - min(r[i] for r in rows))
    print(f"\nspread across all {len(rows)} rollouts (4 terrains x 3 seeds):")
    print(f"  contact events   {min(r[1] for r in rows)}-{max(r[1] for r in rows)}"
          f"   (spread {spread(1)})")
    print(f"  stance duty      {min(r[2] for r in rows):.3f}-{max(r[2] for r in rows):.3f}")
    print(f"  distance walked  {min(r[6] for r in rows):.2f}-{max(r[6] for r in rows):.2f} m")
    print(f"  max tilt         {min(r[7] for r in rows):.2f}-{max(r[7] for r in rows):.2f} deg"
          f"   <- the only column the terrain moves")


if __name__ == "__main__":
    main()
