r"""experiments/b_phase.py — is ContactNet's Sigma_C just a stride clock?

The known failure mode of this dataset: trained on essentially one gait, the
network can learn the *phase* of the stride rather than the *state* of the
contact, and still score well. If `Sigma_C` is (almost) a deterministic function
of gait phase, then it carries no information the analytic heuristic did not
already have, and widening the domain randomisation is attacking the right
problem. If it is not, hypothesis (b) loses its mechanism.

Method: recover gait phase from MuJoCo's own contact set (heel-strike edges of
slot 0 bound each stride), then regress `log trace(Sigma_C)` on a Fourier basis
of phase. `R^2` is the share of the network output explained by phase alone.

    uv run python experiments/b_phase.py results/adetect/contactnet_c1_trust_s0.npz
    uv run python experiments/b_phase.py results/bood/contactnet_*.npz --compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def gait_phase(contact_true: np.ndarray, slot: int = 0) -> np.ndarray:
    """Phase in [0, 1) from one slot's rising edges; NaN outside complete strides."""
    c = contact_true[:, slot] > 0
    edges = np.flatnonzero(np.diff(c.astype(int)) > 0) + 1
    phase = np.full(len(c), np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            phase[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
    return phase


def fourier(phase: np.ndarray, harmonics: int = 6) -> np.ndarray:
    cols = [np.ones_like(phase)]
    for k in range(1, harmonics + 1):
        cols += [np.cos(2 * np.pi * k * phase), np.sin(2 * np.pi * k * phase)]
    return np.stack(cols, axis=1)


def phase_r2(sigma_c: np.ndarray, phase: np.ndarray, harmonics: int = 6):
    """Per-slot R^2 of log trace(Sigma_C) against a Fourier basis of gait phase."""
    ok = np.isfinite(phase)
    X = fourier(phase[ok], harmonics)
    out = []
    for s in range(sigma_c.shape[1]):
        y = np.log(np.trace(sigma_c[ok, s], axis1=1, axis2=2) + 1e-300)
        if np.std(y) < 1e-12:                       # constant output: phase explains nothing
            out.append(0.0)
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out.append(1.0 - float(np.var(resid) / np.var(y)))
    return np.asarray(out)


def describe(path: Path, harmonics: int) -> dict | None:
    z = np.load(path, allow_pickle=False)
    if "contact_true" not in z.files or "sigma_c" not in z.files:
        print(f"  {path.name}: no contact truth / sigma_c recorded — skipped")
        return None
    sc, ct = z["sigma_c"], z["contact_true"]
    if len(sc) != len(ct):
        # Refuse rather than truncate. A record written before the recorder's
        # priming bug was fixed is short by one control tick, and silently
        # aligning the two series from the left shifts the phase by a few percent
        # of a stride — small enough to look plausible and wrong enough to matter.
        print(f"  {path.name}: MISALIGNED (sigma_c {len(sc)} vs contact_true "
              f"{len(ct)}) — skipped; re-record with the fixed z_budget.py")
        return None
    ph = gait_phase(ct)
    r2 = phase_r2(sc, ph, harmonics)
    tr = np.trace(sc, axis1=2, axis2=3)             # (T, N)
    return {
        "name": path.stem,
        "r2_mean": float(r2.mean()),
        "r2": r2,
        "strides": int(np.sum(np.diff(ct[:, 0] > 0) > 0)),
        "log_spread": float(np.std(np.log(tr + 1e-300))),
        "tr_min": float(tr.min()), "tr_max": float(tr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--harmonics", type=int, default=6)
    args = ap.parse_args()

    rows = [d for p in args.paths if (d := describe(Path(p), args.harmonics))]
    if not rows:
        print("nothing to report")
        return 1

    print(f"\n{'run':38s} {'strides':>8s} {'phase R^2':>10s} "
          f"{'log-trace sd':>13s} {'trace min/max':>22s}")
    print("-" * 96)
    for r in rows:
        print(f"{r['name']:38s} {r['strides']:8d} {r['r2_mean']:10.4f} "
              f"{r['log_spread']:13.4f} {r['tr_min']:10.3e}/{r['tr_max']:.3e}")
    print("\n  R^2 -> 1 means Sigma_C is a deterministic function of gait phase, i.e. a")
    print("  stride clock carrying nothing the analytic heuristic did not have.")
    if len(rows) > 1:
        sd = np.std([r["r2_mean"] for r in rows])
        print(f"  spread of phase-R^2 across conditions: {sd:.4f} "
              f"(large = the net does respond to the environment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
