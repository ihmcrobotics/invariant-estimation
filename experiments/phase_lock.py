r"""How much of the contact signal is just a stride-phase clock?

This is the acceptance gate for any change to the training data.

The finding it exists to track
------------------------------
On the original 12-rollout set, the ContactNet covariance learned by runs 2 and 3
is **79% explained by gait phase alone**: regressing ``log10 std_z`` on
time-since-last-touchdown gives ``R^2 = 0.790``, while the `ContactTrust` signal
explains ``R^2 = 0.020``.  With a single gait, "contact quality" and "stride
phase" are the same variable, so a phase clock is the most the network can learn
— and more of that gait teaches it nothing new.

Any intervention meant to fix that (friction randomisation, disturbance forces,
command randomisation, different motions) has to be judged on whether it
*decorrelates contact condition from phase*.  This measures that, two ways:

* **model-side** — ``R^2`` of a trained network's ``log10 std_z`` on phase.
  Needs a checkpoint; directly comparable to the 0.790 baseline.
* **data-side** — ``R^2`` of the recorded friction-cone saturation on phase.
  Needs no training at all, so it can gate a dataset *before* a GPU-hour is
  spent on it.  Only available for rollouts collected with the slip
  instrumentation.

Read the numbers as: **lower is better**.  ``R^2`` near 0.79 means the new data
is still a clock.  ``R^2`` well below it means contact condition now varies for
reasons phase cannot predict, which is the whole point.

Usage
-----
    uv run python -m experiments.phase_lock --data data
    uv run python -m experiments.phase_lock --data data_dr \
        --checkpoint artifacts/contactnet_run4.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASELINE_R2 = 0.721
"""Model-side R^2 of run 2 on the ORIGINAL 12-rollout set, pooled, at ``--stride 20``.

The number any new dataset has to beat, and it must be compared like for like:
the single-rollout fine-stride measurement on `flat_seed000` reads **0.790**, and
quoting that against a pooled run would flatter the new data by ~0.07 for free.
Both say the same thing — the learned covariance is a stride-phase clock.
"""


def gait_phase(trust: np.ndarray, hi: float = 0.5) -> np.ndarray:
    """Ticks since this foot's last touchdown; NaN before the first one.

    Vectorised: a running maximum of the touchdown indices, which is the same
    thing as "index of the most recent rising edge" without a Python loop.
    """
    on = trust > hi
    rise = np.zeros_like(on)
    rise[1:] = on[1:] & ~on[:-1]
    idx = np.arange(trust.shape[0])[:, None]
    last = np.where(rise, idx, -1)
    last = np.maximum.accumulate(last, axis=0)
    ph = (idx - last).astype(float)
    ph[last < 0] = np.nan
    return ph


def r2_on_phase(y: np.ndarray, phase: np.ndarray, *, bin_ticks: int = 10,
                n_bins: int = 300) -> float:
    r"""Fraction of ``y``'s variance explained by binned gait phase alone.

    A binned conditional mean rather than a linear fit: the relationship is
    strongly non-linear (stance and swing are different regimes), and a linear
    ``R^2`` would understate the lock and flatter the dataset.
    """
    y = np.asarray(y).ravel()
    ph = np.asarray(phase).ravel()
    good = np.isfinite(y) & np.isfinite(ph)
    y, ph = y[good], ph[good]
    if y.size < 100:
        return float("nan")
    b = np.clip((ph // bin_ticks).astype(int), 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = b == k
        if sel.sum() > 3:
            means[k] = y[sel].mean()
    pred = means[b]
    m = np.isfinite(pred)
    if m.sum() < 100 or np.var(y[m]) == 0.0:
        return float("nan")
    return float(1.0 - np.var(y[m] - pred[m]) / np.var(y[m]))


SLIP_KEY = "truth.slip_sat"
"""Friction-cone saturation per foot, as `sim/collect.py` actually writes it."""
LOAD_KEY = "truth.contact_fn"
"""Summed normal force per foot. Zero means the sample is not evidence about slip."""


def _slip_key(z) -> str | None:
    return SLIP_KEY if SLIP_KEY in z.files else None


def data_side(path: Path, warmup: int) -> tuple[float, dict] | None:
    """R^2 of recorded friction-cone saturation on phase, if instrumented."""
    with np.load(path, allow_pickle=True) as z:
        key = _slip_key(z)
        if key is None:
            return None
        sat = np.asarray(z[key])[warmup:]
        trust = np.asarray(z["sensors.contact"])[warmup:]
        # Load from the recorded NORMAL FORCE, not from `ContactTrust`. Trust is a
        # hysteretic estimate with a 40 ms dwell; f_n > 0 is the physical fact of
        # whether this foot was bearing load, and an unloaded sample carries no
        # evidence about slip either way.
        fn = (np.asarray(z[LOAD_KEY])[warmup:] if LOAD_KEY in z.files
              else np.where(trust > 0.5, 1.0, 0.0))
        meta = json.loads(str(z["meta"]))
    # Phase still comes from trust: it is the touchdown *event* detector, and it
    # is what the network itself sees.
    ph = gait_phase(trust)
    loaded = fn > 0.0
    return r2_on_phase(np.where(loaded, sat, np.nan),
                       np.where(loaded, ph, np.nan)), meta


def model_side(paths: list[Path], checkpoint: str, cache_dir: Path,
               norm_path: Path, warmup: int, stride: int) -> float:
    """R^2 of a trained network's log10 std_z on phase, pooled over rollouts."""
    import jax
    import jax.numpy as jnp
    from invariant_estimation.contactnet import features, network, normalize, train
    from invariant_estimation.contactnet.config import ContactNetConfig

    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4)
    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                        cfg.sigma_0, cfg.eps)
    params = train.load_params(checkpoint, like)
    nc = normalize.load(str(norm_path))
    fwd = jax.jit(jax.vmap(jax.vmap(
        lambda x: network.forward(params, x, cfg.eps))))

    ys, phs = [], []
    for p in paths:
        cache = cache_dir / f"{p.stem}_feat.npz"
        if not cache.exists():
            continue
        with np.load(cache) as z:
            chan = np.asarray(z["channels"])
        with np.load(p, allow_pickle=True) as z:
            trust = np.asarray(z["sensors.contact"])
        starts = np.arange(warmup, chan.shape[0] - 1, stride)
        w = features.window(normalize.apply(jnp.asarray(chan), nc),
                            cfg.H, cfg.stride)
        flat = w[starts].reshape(len(starts), chan.shape[1], -1)
        L = fwd(flat)
        S = np.asarray(jnp.einsum("...ij,...kj->...ik", L, L))
        sd = np.sqrt(np.diagonal(S, axis1=-2, axis2=-1))
        ys.append(np.log10(sd[..., 2]))
        phs.append(gait_phase(trust)[starts])
    if not ys:
        return float("nan")
    return r2_on_phase(np.concatenate(ys), np.concatenate(phs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--cache", default=None, help="default <data>/cache")
    ap.add_argument("--norm", default=None, help="default <data>/norm_constants.npz")
    ap.add_argument("--checkpoint", default=None,
                    help="a trained ContactNet; enables the model-side R^2")
    ap.add_argument("--warmup", type=int, default=16_000)
    ap.add_argument("--stride", type=int, default=5, help="tick stride for the model-side pass")
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache) if args.cache else data / "cache"
    norm = Path(args.norm) if args.norm else data / "norm_constants.npz"
    paths = [p for p in sorted(data.glob("*.npz")) if p.name != "norm_constants.npz"]
    print(f"phase-lock gate on {data} ({len(paths)} rollouts)")
    print(f"  baseline to beat: model-side R^2 = {BASELINE_R2:.3f} "
          f"(original 12-rollout set, runs 2 and 3)\n")

    rows = []
    for p in paths:
        got = data_side(p, args.warmup)
        if got is None:
            continue
        r2, meta = got
        rows.append((p.stem, r2, meta.get("terrain", "?")))
    if rows:
        print(f"{'rollout':>30} {'terrain':>16} {'data-side R^2':>14}")
        for n, r2, t in rows:
            print(f"{n:>30} {t:>16} {r2:14.3f}")
        vals = np.array([r for _, r, _ in rows])
        print(f"\n  pooled data-side R^2 (cone saturation on phase): "
              f"{np.nanmean(vals):.3f}")
    else:
        print("  no slip instrumentation in these rollouts -- data-side R^2 "
              "unavailable.\n  (Only datasets collected with the friction-cone "
              "recording carry it.)")

    if args.checkpoint:
        r2 = model_side(paths, args.checkpoint, cache, norm, args.warmup, args.stride)
        print(f"\n  model-side R^2 (log10 std_z on phase), {args.checkpoint}: {r2:.3f}")
        if np.isfinite(r2):
            if r2 < 0.5 * BASELINE_R2:
                print("  PASS -- the stride-phase clock is substantially broken.")
            elif r2 < 0.85 * BASELINE_R2:
                print("  PARTIAL -- some decorrelation, but phase still dominates.")
            else:
                print(f"  FAIL -- still a clock ({r2:.3f} vs {BASELINE_R2:.3f}). "
                      f"This intervention did not add contact diversity.")


if __name__ == "__main__":
    main()
