r"""What a trained checkpoint does to the contact update.

The acceptance check for any ContactNet run.  Run 1 passed every *process*
metric — loss down 82x, gradients finite, ``applied_frac`` 1.000 throughout —
and was still degenerate: its ``Sigma_C`` had a median per-axis std of 0.68 m,
four orders above the ``N = J Sigma_q J^T`` term in the same innovation, which
suppressed the velocity Kalman gain **3835x**.  The filter had learned to ignore
its feet.

Loss curves cannot see that.  This can:

1. Evaluate ``Sigma_C`` on real feature windows and report its scale, anisotropy
   and time variation against the ``sigma_0`` the network was initialized at.
2. Plug it into ``S = H P H^T + N + Sigma_C`` on the measured ``P0`` and report
   the Kalman gain per state block, against the same gain at initialization.

The number that matters is the **velocity** row's suppression factor: that is
the block the L2 objective scores, and a factor far above ~1 means training
bought its loss by switching the update off rather than by calibrating it.

Usage
-----
    uv run python -m experiments.check_sigma artifacts/contactnet_run2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from invariant_estimation.contactnet import features, network, normalize, train
from invariant_estimation.contactnet.config import ContactNetConfig

REPO = Path(__file__).resolve().parent.parent

N_ENCODER = 1.26e-5
"""``tr(J Sigma_q J^T)/3`` per axis [m^2], measured on the test fixture.

The other term in the contact innovation.  `Sigma_C` exceeding this by orders of
magnitude is the definition of "the update has been switched off".
"""


def gain_table(P0: np.ndarray, sigma_c: np.ndarray) -> dict[str, float]:
    """``||K||_F`` per state block for one ``Sigma_C``.

    ``H`` differences base position against the contact anchor, so the column
    for a state block is ``P[blk, pos] - P[blk, contact]``.
    """
    pos, c0 = slice(6, 9), slice(9, 12)
    HPH = P0[pos, pos] + P0[c0, c0] - P0[pos, c0] - P0[c0, pos]
    S = HPH + N_ENCODER * np.eye(3) + sigma_c
    blocks = {"rotation": slice(0, 3), "velocity": slice(3, 6),
              "position": pos, "anchor": c0}
    Sinv = np.linalg.inv(S)
    return {name: float(np.linalg.norm((P0[b, pos] - P0[b, c0]) @ Sinv))
            for name, b in blocks.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--cache", default=str(REPO / "data/cache"))
    ap.add_argument("--norm", default=str(REPO / "data/norm_constants.npz"))
    ap.add_argument("--p0", default=str(REPO / "artifacts/p0.npz"))
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=16_000)
    args = ap.parse_args()

    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4)
    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                        cfg.sigma_0, cfg.eps)
    params = train.load_params(args.checkpoint, like)
    print(f"{args.checkpoint}: "
          f"{sum(int(np.size(x)) for x in jax.tree.leaves(params))} params")
    d_head = float(jnp.max(jnp.abs(params.head.W - like.head.W)))
    d_trunk = max(float(jnp.max(jnp.abs(a.W - b.W)))
                  for a, b in zip(params.trunk, like.trunk))
    print(f"  moved off init: head.W {d_head:.3e}  trunk.W {d_trunk:.3e}")

    cache = sorted(Path(args.cache).glob("*.npz"))
    with np.load(cache[0]) as z:
        chan = z["channels"]
    nc = normalize.load(args.norm)
    starts = np.linspace(args.warmup, chan.shape[0] - 1, args.samples).astype(int)
    w = features.window(normalize.apply(jnp.asarray(chan), nc), cfg.H, cfg.stride)
    flat = w[starts].reshape(len(starts), chan.shape[1], -1)

    fwd = jax.jit(jax.vmap(jax.vmap(
        lambda x: network.forward(params, x, cfg.eps))))
    L = fwd(flat)
    S = jnp.einsum("...ij,...kj->...ik", L, L)
    sd = jnp.sqrt(jnp.diagonal(S, axis1=-2, axis2=-1))

    print(f"\nSigma_C over {L.shape[0]} ticks x {L.shape[1]} contacts "
          f"({cache[0].name}):")
    for i, ax in enumerate("xyz"):
        v = sd[..., i]
        print(f"  std_{ax}  min {float(v.min()):.3e}  med {float(jnp.median(v)):.3e}"
              f"  max {float(v.max()):.3e}   (init {cfg.sigma_0:.1e})")
    print(f"  anisotropy max/min std, median "
          f"{float(jnp.median(sd.max(-1) / sd.min(-1))):.3f}")
    per_tick = sd.reshape(sd.shape[0], -1).mean(-1)
    print(f"  temporal CoV {float(per_tick.std() / per_tick.mean()):.1%}")
    ev = jnp.linalg.eigvalsh(S)
    print(f"  SPD {bool(ev.min() > 0)}   finite {bool(jnp.all(jnp.isfinite(S)))}")

    med = np.diag(np.asarray(jnp.median(sd.reshape(-1, 3), axis=0)) ** 2)
    P0 = np.load(args.p0)["P0"]
    k_init = gain_table(P0, cfg.sigma_0 ** 2 * np.eye(3))
    k_now = gain_table(P0, med)
    print(f"\ncontact-update Kalman gain on the measured P0:")
    print(f"  {'block':10s}  {'||K|| init':>12s}  {'||K|| trained':>14s}  "
          f"{'suppression':>12s}")
    for name in k_init:
        f = k_init[name] / k_now[name]
        mark = "  <- scored by L2" if name == "velocity" else ""
        print(f"  {name:10s}  {k_init[name]:12.4e}  {k_now[name]:14.4e}  "
              f"{f:11.1f}x{mark}")

    supp = k_init["velocity"] / k_now["velocity"]
    print()
    if supp > 10.0:
        print(f"DEGENERATE — velocity gain suppressed {supp:.0f}x. The network "
              f"has switched the contact update off, as run 1 did.")
    elif supp < 0.1:
        print(f"velocity gain AMPLIFIED {1/supp:.1f}x — the filter is being "
              f"pushed to trust contacts far more than at init. Check NEES.")
    else:
        print(f"OK — velocity gain within {max(supp, 1/supp):.1f}x of "
              f"initialization. The update is still live.")


if __name__ == "__main__":
    main()
