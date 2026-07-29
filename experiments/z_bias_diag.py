r"""Where does the vertical sink come from? Two read-only diagnostics.

`PORT_NOTES.md` ("Run 4 converged, and the residual z drift is a BIAS") lists
three candidate sources for the constant-rate sink and names one diagnostic to
separate them.  This runs that diagnostic, plus a second one that turned out to
be the decisive one, entirely from recorded rollouts.  **Nothing here runs the
filter** -- every number comes from arrays already in the ``.npz``, so it is
seconds of numpy and needs neither JAX nor MJX.

Diagnostic A -- is the FK contact point sinking?
------------------------------------------------
Reconstruct the world-frame sole position from the *true* base pose and the
cached FK, ``sole_w = p_true + R_true · y_fk``.  A planted foot is world-static,
so during deep stance ``d(sole_w)/dt`` must be zero; any persistent negative
mean is a contact-point/penetration model error and would inject a downward
ramp on every step.

**Read the erosion parameter before trusting a number here.**  Taking a tick-wise
`np.gradient` over a stance mask gives ``-0.0096 m/s`` and that value is an
artifact: at the two ends of a stance block the centred difference straddles the
swing transition, and those few samples carry ~100x the interior magnitude.
`ERODE` trims each block; the interior drift is then computed as a *secant* over
the block rather than a mean of derivatives, which cannot be contaminated at all.
With that, the answer changes sign and becomes statistically zero.  Per-phase
statistics (not per-tick) are also mandatory: 1 kHz samples inside one stance are
almost perfectly correlated, so a per-tick standard error understates by ~25x.

Diagnostic B -- is the sink a position injection or a velocity bias?
--------------------------------------------------------------------
Compare ``slope(est_p_z - p_z)`` against ``mean(est_v_z - v_z)``.  If the base
position is being pushed down by the contact update, these are unrelated.  If
the velocity estimate simply sits low, the position error is its integral and
the two are equal by construction.

The third block then asks whether the propagation could be responsible, by
measuring the world-z specific-force error the filter actually integrates,
``(R_est a)_z - (R_true a)_z``, and comparing it against the second-order tilt
rectification ``-g|delta|^2 / 2`` -- which is a genuine bias mechanism (it is
negative for *any* tilt error direction) and worth ruling in or out explicitly.

Usage
-----
    python -m experiments.z_bias_diag                 # data/dr + data/control
    python -m experiments.z_bias_diag --glob 'data/*.npz'
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

GRAVITY = 9.81
LOAD_FRAC = 0.7
"""Share of body weight on one foot that counts as deep stance."""

ERODE = 50
"""Ticks trimmed from each end of a stance block. See the module docstring --
this is the difference between ``-0.0096`` and ``+0.0007`` m/s."""

MIN_BLOCK = 200
"""Ticks a stance block must retain after erosion to be scored."""


def _blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of True."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[0::2], idx[1::2]))


def _cache_path(path: Path) -> Path:
    return path.parent / "cache" / f"{path.stem}_feat.npz"


def sole_world(z, y_fk: np.ndarray) -> np.ndarray:
    """``(T, N_c, 3)`` world sole position from the TRUE base pose and cached FK.

    Truth on both sides on purpose: this isolates the kinematic model from the
    filter, so a nonzero result is a model statement and not an estimator one.
    """
    return z["truth.p"][:, None, :] + np.einsum(
        "tij,tkj->tki", z["truth.R"], y_fk)


def diagnostic_a(path: Path) -> list[dict]:
    """Per-stance-phase vertical drift of the FK sole point, per foot."""
    z = np.load(path)
    cache = _cache_path(path)
    if not cache.exists():
        return []
    meta = json.loads(str(z["meta"]))
    W, dt = meta["warmup_ticks"], meta["dt"]
    fn = z["truth.contact_fn"]
    sw = sole_world(z, np.load(cache)["y_fk"])
    weight = np.median(fn.sum(1)[W:])

    out = []
    for i in range(fn.shape[1]):
        mask = fn[:, i] / weight > LOAD_FRAC
        mask[:W] = False
        drift, height = [], []
        for a, b in _blocks(mask):
            a, b = a + ERODE, b - ERODE
            if b - a < MIN_BLOCK:
                continue
            # Secant across the block interior: immune to boundary contamination.
            drift.append((sw[b - 1, i, 2] - sw[a, i, 2]) / ((b - 1 - a) * dt))
            height.append(sw[a:b, i, 2].mean())
        if not drift:
            continue
        d = np.asarray(drift)
        out.append(dict(
            foot=i, n=len(d), drift_mean=d.mean(),
            drift_se=d.std(ddof=1) / np.sqrt(len(d)),
            height_mean=float(np.mean(height)),
            min_sole_z=float(sw[W:, i, 2].min()),
            # Sole HEIGHT is only interpretable where the ground is known to sit
            # at z = 0.  On `waves` / `stepping_stones` the heightfield has
            # relief, so the column is a foot-vs-terrain quantity this script
            # cannot resolve without a terrain query.  Drift is unaffected: it is
            # differential within one stance, on one spot of ground.
            flat=meta.get("relief_m", 0.0) == 0.0,
        ))
    return out


def diagnostic_b(path: Path) -> dict:
    """Velocity-bias vs position-injection, plus the propagation check."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    W, dt = meta["warmup_ticks"], meta["dt"]

    e_v = z["aux.est_v"][:, 2] - z["truth.v"][:, 2]
    e_p = z["aux.est_p"][:, 2] - z["truth.p"][:, 2]
    t = np.arange(len(e_p)) * dt
    slope = float(np.polyfit(t[W:], e_p[W:], 1)[0])

    Re, Rt, a = z["aux.est_R"], z["truth.R"], z["inputs.accel"]
    M = np.einsum("tij,tkj->tik", Re, Rt)
    tilt = np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1))[W:]
    # Exactly the quantity `propagate` integrates into v.
    a_err_z = (np.einsum("tij,tj->ti", Re[W:], a[W:])
               - np.einsum("tij,tj->ti", Rt[W:], a[W:]))[:, 2]

    return dict(
        tilt_deg=float(np.degrees(tilt).mean()),
        rectified=float(-0.5 * GRAVITY * np.mean(tilt ** 2)),
        a_err_z=float(a_err_z.mean()),
        e_vz=float(e_v[W:].mean()),
        slope_e_pz=slope,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", action="append", default=None,
                    help="rollout globs; default data/dr/*.npz + data/control/*.npz")
    args = ap.parse_args()
    pats = args.glob or [str(REPO / "data/dr/*.npz"), str(REPO / "data/control/*.npz")]
    paths = [Path(p) for g in pats for p in sorted(globmod.glob(g))
             if "norm_constants" not in p]

    print("=" * 96)
    print("A. Vertical drift of the FK sole point during deep stance "
          f"(load > {LOAD_FRAC:g} BW, erode {ERODE} ticks)")
    print("   A planted foot is world-static, so zero is the null hypothesis.\n")
    print(f"{'rollout':>28} {'foot':>5} {'phases':>7} {'drift [m/s]':>14} "
          f"{'sole z [m]':>11} {'min z [m]':>10}")
    pooled = []
    for p in paths:
        for r in diagnostic_a(p):
            name = f"{p.parent.name}/{p.stem}"
            # Height is only meaningful against a known ground plane -- see `flat`.
            hz = (f"{r['height_mean']:+11.4f} {r['min_sole_z']:+10.4f}"
                  if r["flat"] else f"{'(relief)':>11} {'':>10}")
            print(f"{name:>28} {r['foot']:>5} {r['n']:>7} "
                  f"{r['drift_mean']:+9.4f} +/-{r['drift_se']:.4f} {hz}")
            pooled.append(r)
    if pooled:
        d = np.array([r["drift_mean"] for r in pooled])
        h = np.array([r["height_mean"] for r in pooled if r["flat"]])
        print(f"\n   pooled over {len(d)} foot-rollouts: drift "
              f"{d.mean():+.5f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.5f} m/s")
        if len(h):
            print(f"   flat-terrain sole height above z=0 ({len(h)} foot-rollouts): "
                  f"{h.mean():+.4f} m  [never below {min(r['min_sole_z'] for r in pooled if r['flat']):+.4f}]")

    print("\n" + "=" * 96)
    print("B. Is the sink an integrated velocity bias?\n")
    print(f"{'rollout':>28} {'tilt[deg]':>9} {'-g|d|^2/2':>10} {'a_err_z':>9} "
          f"{'mean e_vz':>10} {'slope e_pz':>11} {'ratio':>7}")
    rows = []
    for p in paths:
        r = diagnostic_b(p)
        rows.append(r)
        print(f"{p.parent.name + '/' + p.stem:>28} {r['tilt_deg']:9.3f} "
              f"{r['rectified']:10.5f} {r['a_err_z']:9.5f} {r['e_vz']:10.5f} "
              f"{r['slope_e_pz']:11.5f} {r['slope_e_pz'] / r['e_vz']:7.3f}")

    ev = np.array([r["e_vz"] for r in rows])
    sl = np.array([r["slope_e_pz"] for r in rows])
    ae = np.array([r["a_err_z"] for r in rows])
    t2 = np.array([-2 * r["rectified"] / GRAVITY for r in rows])
    print(f"\n   mean e_vz vs slope(e_pz):  r = {np.corrcoef(ev, sl)[0, 1]:+.4f}   "
          f"mean ratio {np.mean(sl / ev):.3f}")
    print(f"   |delta|^2 vs a_err_z:      r = {np.corrcoef(t2, ae)[0, 1]:+.3f}")
    print(f"   a_err_z   vs mean e_vz:    r = {np.corrcoef(ae, ev)[0, 1]:+.3f}   "
          f"(sign of a_err_z: {np.sign(ae).astype(int).tolist()})")


if __name__ == "__main__":
    main()
