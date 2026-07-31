r"""One table from a set of `run_estimator.py --out` histories.

Written because a video without its numbers next to it is not evidence.  Reports the
three quantities a ghost clip is judged on, plus the split that makes the vertical
number honest:

* ``|dz|`` -- the signed vertical error, which is the one-sided failure the ghost shows;
* ``yaw`` -- the yaw component of attitude error, ``sqrt(att^2 - tilt^2)``.  ``tilt`` is
  the gravity-observable part and ``att`` the whole rotation error, so the residual is the
  unobservable heading.  Reported separately for exactly that reason;
* ``horiz`` -- ``|| (p_est - p_true)_xy ||``, and its ratio to distance travelled.

**Global x, y and yaw are unobservable in a proprioceptive InEKF.**  There is no
measurement with a world-frame position or heading row anywhere in this filter, so those
three drift for as long as it runs and no covariance, schedule or network can stop them.
A change can only reduce the RATE.  The table prints drift per metre travelled so that a
longer run cannot look worse than a shorter one for free.

Usage
-----
    uv run python -m experiments.summarise_runs artifacts/video/*.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def row(path: Path, tail_frac: float = 0.5) -> dict:
    with np.load(path) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    n0 = int(len(d["t"]) * (1.0 - tail_frac))
    ep, tp = d["est_p"], d["true_p"]
    horiz = np.linalg.norm(ep[:, :2] - tp[:, :2], axis=1)
    travelled = np.linalg.norm(tp[-1, :2] - tp[0, :2])
    # Yaw is what attitude error has left once the gravity-observable tilt is removed.
    yaw = np.sqrt(np.maximum(d["att_deg"] ** 2 - d["tilt_deg"] ** 2, 0.0))
    return {
        "name": path.stem,
        "s": float(d["t"][-1]),
        "dz_final_cm": 100.0 * float(d["dz"][-1]),
        "dz_tail_cm": 100.0 * float(d["dz"][n0:].mean()),
        "dz_rms_cm": 100.0 * float(np.sqrt((d["dz"] ** 2).mean())),
        "tilt_tail_deg": float(np.sqrt((d["tilt_deg"][n0:] ** 2).mean())),
        "yaw_tail_deg": float(np.sqrt((yaw[n0:] ** 2).mean())),
        "horiz_final_m": float(horiz[-1]),
        "travelled_m": float(travelled),
        "horiz_pct": 100.0 * float(horiz[-1]) / max(travelled, 1e-9),
        "vel_rms": float(np.sqrt((d["v_err"] ** 2).mean())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    rows = [row(Path(p)) for p in args.runs]
    hdr = (f"{'run':40} {'s':>5} {'dz fin':>8} {'dz tail':>8} {'dz rms':>7} "
           f"{'tilt':>6} {'yaw':>6} {'horiz':>7} {'trav':>6} {'h/trav':>7} {'vel':>7}")
    print(hdr)
    print(f"{'':40} {'':>5} {'[cm]':>8} {'[cm]':>8} {'[cm]':>7} "
          f"{'[deg]':>6} {'[deg]':>6} {'[m]':>7} {'[m]':>6} {'[%]':>7} {'[m/s]':>7}")
    for r in rows:
        print(f"{r['name'][:40]:40} {r['s']:5.0f} {r['dz_final_cm']:8.1f} "
              f"{r['dz_tail_cm']:8.1f} {r['dz_rms_cm']:7.1f} {r['tilt_tail_deg']:6.2f} "
              f"{r['yaw_tail_deg']:6.2f} {r['horiz_final_m']:7.2f} {r['travelled_m']:6.1f} "
              f"{r['horiz_pct']:7.1f} {r['vel_rms']:7.4f}")
    print("\n  Global x/y/yaw are UNOBSERVABLE in a proprioceptive InEKF: no H row sees them.")
    print("  Those drifts are reduced by a better schedule, never eliminated.")


if __name__ == "__main__":
    main()
