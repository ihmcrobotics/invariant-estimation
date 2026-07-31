r"""Does the training objective have an interior optimum in ``Sigma_C``?

This is the acceptance gate for the run-2 configuration, and it is the check
that would have caught run 1 before spending a GPU hour on it.

The theory
----------
Theory doc S7.2.1, Claim 1 proves that a quadratic-only objective has **no
interior minimum** over SPD ``S``: along ``S = alpha*Sigma``,

    tr((alpha*Sigma)^-1 Sigma) = tr(I)/alpha = k/alpha  ->  0   as alpha -> inf

so inflating the covariance improves the term forever.  That analysis was
performed on the beta-NLL quadratic term and never on the L2 objective run 1
actually used -- and L2 has the same pathology whenever the contact update
cannot pay for itself over the segment horizon.  Run 1 duly drove ``Sigma_C``
to a median 0.68 m per-axis std, suppressing the velocity Kalman gain 3835x.

So: sweep a global scale ``alpha`` on ``Sigma_C``, evaluate the *actual*
training loss, and ask whether ``argmin_alpha`` is interior.

    interior  ->  "ignore the feet" is not optimal; the objective is trainable.
    at alpha_max ->  degenerate; training will find Sigma_C -> infinity.

This is a property of the *protocol* -- seeding, horizon, process socket -- and
not of the network, so it runs at initialization and needs no training at all.

Usage
-----
    uv run python -m experiments.alpha_sweep                  # run-2 config
    uv run python -m experiments.alpha_sweep --run1           # reproduce run 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from invariant_estimation.contactnet import dataset, network, normalize
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet.rollout import make_batch_loss, make_warm_in
from invariant_estimation.sim import collect

REPO = Path(__file__).resolve().parent.parent


def sweep(batch_loss, params, batch, carry0, alphas) -> np.ndarray:
    r"""Loss at each ``alpha``, with ``Sigma_C -> alpha^2 * Sigma_C``.

    The network emits the Cholesky factor ``L``, and ``Sigma = L L^T``, so
    scaling the *factor* by ``alpha`` scales the covariance by ``alpha^2``.
    Applied by scaling the head bias' diagonal entries in log-space would be
    fiddly and would not be exact; scaling the emitted factor is, so the sweep
    perturbs `params.head.b` -- at initialization the head weight is zero, so
    the output is exactly ``softplus(b_diag) + eps`` and the map is transparent.
    """
    out = []
    for a in alphas:
        # Shift the softplus pre-activation so diag(L) scales by `a`.  At init,
        # diag(L) = sigma_0 exactly, so the target is a*sigma_0.
        scaled = _scale_head(params, float(a))
        loss, _ = batch_loss(scaled, batch, carry0)
        out.append(float(loss))
    return np.asarray(out)


def _scale_head(params, a: float):
    """Head bias whose softplus gives ``a`` times the current diagonal."""
    b = params.head.b
    cur = jax.nn.softplus(b[:3])
    new = jnp.log(jnp.expm1(jnp.clip(a * cur, 1e-300)))   # softplus^-1
    return params._replace(head=params.head._replace(b=jnp.concatenate([new, b[3:]])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--cache", default=str(REPO / "data/cache"))
    ap.add_argument("--norm", default=str(REPO / "data/norm_constants.npz"))
    ap.add_argument("--p0", default=str(REPO / "artifacts/p0.npz"),
                    help="npz to load P0 from; measured and cached here if absent")
    ap.add_argument("--p0-ticks", type=int, default=3_000)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--decades", type=float, default=4.0,
                    help="sweep alpha over 10^[-d, +d]")
    ap.add_argument("--points", type=int, default=17)
    ap.add_argument("--run1", action="store_true",
                    help="reproduce run 1: frozen process socket, truth-seeded segments")
    ap.add_argument("--seed", type=int, default=0)
    # An N=4 dataset stores (T, 4, 3, 3) contact arrays and needs an N=4 estimator: without
    # this the gate builds the N=2 default and dies inside the first propagation with
    # `dot_general ... got (15,) and (21,)` -- i.e. it does not gate, it crashes. Same defect
    # `replay_eval` carried until 2026-07-29; found here on the run-7 dataset.
    ap.add_argument("--toe-heel", dest="toe_heel", action="store_true",
                    help="gate an N=4 (heel+toe per foot) dataset; must match how it was collected")
    args = ap.parse_args()

    n_c = 4 if args.toe_heel else 2
    cfg = ContactNetConfig(
        F=24, sigma_0=1.0e-4, B=args.B, remat=False,
        n_contacts=n_c,
        freeze_contact_chol=args.run1,
    )
    label = "run-1 (frozen chol, truth-seeded)" if args.run1 else "run-2 (chained)"
    print(f"alpha sweep, {label}")
    print(f"  B={cfg.B}  L={cfg.L}  freeze_contact_chol={cfg.freeze_contact_chol}  "
          f"chained={not args.run1}")

    fused = collect.build_collector(verbose=False, toe_heel=args.toe_heel).fused
    if int(fused.n_contacts) != n_c:
        raise SystemExit(f"estimator has {fused.n_contacts} contacts, expected {n_c}")
    norm = normalize.load(args.norm)
    preps = dataset.prepare(dataset.rollout_paths(args.data)[:args.rollouts],
                            norm, cfg, cache_dir=args.cache, verbose=False)
    # P0 is *measured*, not configured. `train` is the only other code path that
    # mints it, so on a fresh clone this gate used to die with FileNotFoundError
    # before it could gate anything -- fall back to measuring it the same way.
    if args.p0 and Path(args.p0).exists():
        P0 = np.load(args.p0)["P0"]
    else:
        P0 = dataset.measure_p0(fused, preps[0], cfg, ticks=args.p0_ticks)
        if args.p0:
            Path(args.p0).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.p0, P0=P0)
    params = network.init(jax.random.PRNGKey(args.seed), cfg.d_in, cfg.widths,
                          cfg.sigma_0, cfg.eps)
    batch_loss = jax.jit(make_batch_loss(fused.ekf, fused.kinematics, cfg.eps,
                                         beta=cfg.beta, objective=cfg.objective,
                                         remat=cfg.remat))

    if args.run1:
        batch = dataset.make_batch(
            preps, dataset.sample_starts(np.random.default_rng(args.seed), preps, cfg.B),
            cfg, P0)
        carry0 = None
    else:
        warm_in = make_warm_in(fused.ekf, fused.kinematics)   # heuristic; see make_warm_in
        batcher = dataset.ChainedBatcher(preps, cfg, P0, warm_in, seed=args.seed)
        batch, carry0 = batcher.batch()

    alphas = np.logspace(-args.decades, args.decades, args.points)
    losses = sweep(batch_loss, params, batch, carry0, alphas)

    print(f"\n{'alpha':>12}  {'Sigma_C std [m]':>16}  {'loss':>14}")
    for a, l in zip(alphas, losses):
        mark = "  <- min" if l == losses.min() else ""
        print(f"{a:12.3e}  {a * cfg.sigma_0:16.3e}  {l:14.6e}{mark}")

    k = int(np.argmin(losses))
    interior = 0 < k < len(alphas) - 1
    print(f"\nargmin at alpha = {alphas[k]:.3e} (index {k} of {len(alphas) - 1})")
    if interior:
        print("PASS — interior optimum. The objective is trainable: 'ignore the "
              "feet' is not the best answer.")
    elif k == len(alphas) - 1:
        print("FAIL — minimum at the largest alpha. The objective is DEGENERATE: "
              "training will drive Sigma_C -> infinity and switch the contact "
              "update off, exactly as run 1 did.")
    else:
        print("FAIL — minimum at the smallest alpha. Sigma_C -> 0; the filter is "
              "being pushed to trust contacts without limit.")
    raise SystemExit(0 if interior else 1)


if __name__ == "__main__":
    main()
