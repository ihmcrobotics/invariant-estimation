r"""How much is the process socket *capable* of? An achievability bound on ``Sigma_C``.

Every ContactNet result so far conflates two questions that need separating before another
training run is worth its GPU hour:

1. **Is the socket capable?** Does *any* ``Sigma_C`` schedule materially reduce the L2 velocity
   loss on this data?
2. **Is the target learnable?** Can a causal function of the feature channels produce it?

This script answers (1), and bounds (2), with **no training and no network**. It optimises
``Sigma_C`` directly — as free variables, with hindsight, against the ground truth — on the same
``B x L`` segments, the same ``P0`` and the same ``l2_velocity`` loss the trainer uses, so every
number here is directly comparable to a reported training loss.

The four arms
-------------

======================  ===============================================  =========================
arm                     ``Sigma_C``                                      what it establishes
======================  ===============================================  =========================
``heuristic``           the recorded Schmitt-switched schedule           the baseline to beat
``network``             a trained checkpoint's output (optional)         where we actually are
``speed``               ``softplus(a + b*s)`` from the TRUE contact      can a simple causal rule,
                        speed ``s``; 6 free parameters                   *given perfect qdot*, win?
``free``                per-tick, per-contact Cholesky, unconstrained    upper bound on the socket
======================  ===============================================  =========================

**The value of the arms is asymmetric, and the conclusions must respect that.**

* ``free`` **fails to beat** ``heuristic`` ⇒ decisive negative. It was handed the best
  ``Sigma_C`` that exists on this data — acausal, per-tick, tuned to this exact noise realisation,
  using ground truth in its objective. If that cannot help, no causal function of any feature set
  can, and the process socket is the wrong lever. Branch closed.
* ``free`` **wins** ⇒ weak on its own. With ~6 free parameters per contact per tick it may simply
  be memorising the noise realisation. It says the socket *can* move the loss; it says nothing
  about learnability.
* ``speed`` **captures most of** ``free``'s **gain** ⇒ the target is a simple function of contact
  speed, which is what CoCo-InEKF's Fig. 3 shows, and ``qdot`` is the missing ingredient — that
  justifies the (expensive) honest re-collection with a measured joint-velocity channel.
* ``speed`` **does not** ⇒ the target needs the anisotropic, directional structure the paper
  describes, and ``qdot`` alone will not deliver it.

``speed`` is a **cheat by construction**: it is driven by contact speed computed from the
*ground-truth* joint state, i.e. the noiseless ``J_C qdot`` our feature path does not have (ours is
a 1 kHz finite difference of FK on noisy encoders, whose noise floor sits at or above the median
loaded contact speed). That is the point — it is an upper bound on what a perfect ``qdot`` channel
could buy, run before anyone pays to collect one.

Reported per arm: the loss, the dynamic range of ``sqrt(tr(Sigma_C))`` (the heuristic spans ~1e5;
run 7's learned output spans 2.4x), and its correlation with true contact speed — the statistic
CoCo's Fig. 3 plots.

    uv run python -m experiments.sigma_bound --data data/dr5 --p0 artifacts/p0_dr5.npz
    uv run python -m experiments.sigma_bound --checkpoint artifacts/contactnet_run7.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from invariant_estimation.contactnet import dataset, network, normalize, train
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet.losses import l2_velocity
from invariant_estimation.contactnet.rollout import contact_factors
from invariant_estimation.inEKF.filter import init_carry, make_step
from invariant_estimation.sim import collect

REPO = Path(__file__).resolve().parent.parent
EPS = 1.0e-6


# ---------------------------------------------------------------------------
# the loss, taken directly on Sigma_C rather than on network parameters
# ---------------------------------------------------------------------------

def make_direct_loss(ekf, kinematics, remat: bool = True):
    """``(L_c, segment) -> (loss, outputs)``, differentiable in ``L_c``.

    A copy of `rollout.make_segment_loss` with `contact_factors` removed: the Cholesky factors
    ARE the free variable here, not the output of a network. Everything downstream — the scan, the
    filter, the objective — is identical, which is what makes the numbers comparable to training.
    """
    step = make_step(ekf, kinematics)
    if remat:
        step = jax.checkpoint(step, prevent_cse=False)

    def loss_fn(L_c, segment):
        inputs = segment.inputs._replace(contact_chol=L_c)
        _, out = jax.lax.scan(step, init_carry(segment.state0), inputs)
        loss = l2_velocity(out.state.v, out.state.R, segment.v_true, segment.R_true)
        return loss, out

    return loss_fn


def batch_mean_loss(loss_fn):
    """Mean over the batch axis — the same reduction `make_batch_loss` uses."""
    def f(L_c, batch):
        losses, _ = jax.vmap(loss_fn)(L_c, batch)
        return jnp.mean(losses)
    return f


# ---------------------------------------------------------------------------
# parameterisations
# ---------------------------------------------------------------------------

def chol_from_raw(raw):
    """``(..., 3, 3)`` unconstrained -> a valid lower-triangular Cholesky factor.

    `softplus` on the diagonal keeps it strictly positive, so ``L Lᵀ`` is SPD for any input and the
    optimiser cannot wander into an indefinite covariance and produce a meaningless bound.
    """
    tril = jnp.tril(raw, -1)
    diag = jax.nn.softplus(jnp.diagonal(raw, axis1=-2, axis2=-1)) + EPS
    return tril + diag[..., None] * jnp.eye(3)


def speed_chol(theta, speed):
    """6 parameters: an isotropic-per-axis affine map from contact speed to a diagonal chol.

    ``diag_k = softplus(a_k + b_k * s)``, i.e. the simplest thing that could reproduce Fig. 3's
    "the standard deviations agree with the instantaneous velocities". Per-axis rather than a
    single scalar so it can at least express *some* anisotropy, but it is still diagonal — it
    cannot express the directional structure the paper mentions, and that limit is the point of
    comparing it against `free`.
    """
    a, b = theta[:3], theta[3:]
    d = jax.nn.softplus(a[None, None, :] + b[None, None, :] * speed[..., None]) + EPS
    return d[..., None] * jnp.eye(3)


# ---------------------------------------------------------------------------
# true contact speed — the "cheat"
# ---------------------------------------------------------------------------

def true_contact_speed(fused, prep, t0: int, L: int) -> np.ndarray:
    """``(L, N_c)`` body-frame contact-point speed from the GROUND-TRUTH joint state.

    Finite difference of ``FK(q_true)``, not of ``FK(q_measured)``. That distinction is the whole
    experiment: differencing the *measured* FK is what the feature path already does, and its noise
    floor sits at or above the median loaded contact speed, so the channel carries ~1 bit
    (swinging or not). Differencing the noiseless FK gives the quantity CoCo's covariance tracks.
    """
    q = np.asarray(prep.meta["_q_true"])[t0 - 1:t0 + L]          # one tick of lead-in for the diff
    zero = jnp.zeros(q.shape[-1])
    p = np.asarray(jax.vmap(lambda qi: fused.kinematics(jnp.asarray(qi), zero).y)(jnp.asarray(q)))
    v = np.diff(p, axis=0) / float(fused.params.dt if hasattr(fused, "params") else 1e-3)
    return np.linalg.norm(v, axis=-1)                             # (L, N_c)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def describe(L_c, speed) -> dict:
    """Dynamic range and the Fig. 3 correlation — the two things that separate the arms."""
    S = np.asarray(jnp.einsum("...ij,...kj->...ik", L_c, L_c))
    tot = np.sqrt(np.trace(S, axis1=-2, axis2=-1)).ravel()
    s = np.asarray(speed).ravel()
    ok = np.isfinite(tot) & np.isfinite(s) & (tot > 0)
    lo, hi = np.percentile(tot[ok], [1, 99])
    return {
        "sqrt_tr_p1": float(lo), "sqrt_tr_p50": float(np.percentile(tot[ok], 50)),
        "sqrt_tr_p99": float(hi), "dynamic_range": float(hi / max(lo, 1e-30)),
        "corr_with_contact_speed": float(np.corrcoef(np.log10(tot[ok] + 1e-30), s[ok])[0, 1]),
    }


def optimise(f, init, batch, steps: int, lr: float, label: str) -> tuple:
    opt = optax.adam(lr)
    state = opt.init(init)
    val_grad = jax.jit(jax.value_and_grad(f))
    x = init
    first = None
    for i in range(steps):
        loss, g = val_grad(x, batch)
        if first is None:
            first = float(loss)
        upd, state = opt.update(g, state)
        x = optax.apply_updates(x, upd)
        if i % max(1, steps // 10) == 0:
            print(f"    {label} step {i:4d}  loss {float(loss):.6e}", flush=True)
    final = float(f(x, batch))
    print(f"    {label} DONE  {first:.6e} -> {final:.6e}", flush=True)
    return x, final


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(REPO / "data/dr5"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--p0", default=str(REPO / "artifacts/p0_dr5.npz"))
    ap.add_argument("--checkpoint", default=None, help="optional trained net, for the reference arm")
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr-free", type=float, default=3e-2)
    ap.add_argument("--lr-speed", type=float, default=3e-2)
    ap.add_argument("--toe-heel", dest="toe_heel", action="store_true", default=True)
    ap.add_argument("--out", default=str(REPO / "artifacts/sigma_bound.json"))
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache) if args.cache else data / "cache"
    norm_path = Path(args.norm) if args.norm else data / "norm_constants.npz"

    n_c = 4 if args.toe_heel else 2
    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4, B=args.B, n_contacts=n_c, remat=True)
    col = collect.build_collector(verbose=False, toe_heel=args.toe_heel)
    fused = col.fused
    norm = normalize.load(str(norm_path))
    preps = dataset.prepare(dataset.rollout_paths(data)[:args.rollouts], norm, cfg,
                            cache_dir=cache, verbose=False)
    P0 = np.load(args.p0)["P0"]

    # One segment per (rollout, start), deterministic starts so the arms share segments exactly.
    segs, speeds = [], []
    per = max(1, args.B // len(preps))
    rollout_q = {}
    for p in preps:
        with np.load(Path(data) / f"{p.meta['terrain']}_seed{p.meta['seed']:03d}.npz",
                     allow_pickle=False) as z:
            q_true = np.asarray(z["truth.q"])
            q_unf = np.asarray(z["sensors.q_unfiltered"])
        rollout_q[p.name] = np.concatenate([q_true, q_unf], axis=-1)
        hi = min(p.t_hi, p.smoothed.shape[0] - cfg.L - 1)
        for t0 in np.linspace(max(p.t_lo, 1), hi, per).astype(int):
            t0 = int(t0)
            segs.append(dataset.make_segment(p, t0, cfg, P0))
            qa = rollout_q[p.name][t0 - 1:t0 + cfg.L]
            zero = jnp.zeros(qa.shape[-1])
            pos = np.asarray(jax.vmap(
                lambda qi: fused.kinematics(jnp.asarray(qi), zero).y)(jnp.asarray(qa)))
            speeds.append(np.linalg.norm(np.diff(pos, axis=0) / cfg.dt, axis=-1))
    batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs), dtype=jnp.float64), *segs)
    speed = jnp.asarray(np.stack(speeds), dtype=jnp.float64)          # (B, L, N_c)
    B, L = speed.shape[0], speed.shape[1]
    print(f"sigma bound: {data.name}  B={B} L={L} N_c={n_c}  "
          f"true contact speed p50={float(jnp.median(speed)):.4f} m/s", flush=True)

    loss_fn = make_direct_loss(fused.ekf, fused.kinematics)
    f = jax.jit(batch_mean_loss(loss_fn))

    out: dict = {"data": str(data), "B": B, "L": L, "n_contacts": n_c, "arms": {}}

    heur = batch.inputs.contact_chol
    out["arms"]["heuristic"] = {"loss": float(f(heur, batch)), **describe(heur, speed)}
    print(f"  heuristic  loss {out['arms']['heuristic']['loss']:.6e}  "
          f"range {out['arms']['heuristic']['dynamic_range']:.3g}", flush=True)

    if args.checkpoint:
        like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
        params = train.load_params(args.checkpoint, like)
        L_net = jax.vmap(lambda w: contact_factors(params, w, cfg.eps))(batch.windows)
        out["arms"]["network"] = {"loss": float(f(L_net, batch)), "checkpoint": args.checkpoint,
                                  **describe(L_net, speed)}
        print(f"  network    loss {out['arms']['network']['loss']:.6e}  "
              f"range {out['arms']['network']['dynamic_range']:.3g}", flush=True)

    # -- the cheat: 6 parameters driven by TRUE contact speed --------------------
    g_speed = lambda th, b: f(speed_chol(th, speed), b)               # noqa: E731
    th0 = jnp.asarray(np.concatenate([np.full(3, -4.0), np.full(3, 1.0)]))
    th, l_speed = optimise(g_speed, th0, batch, args.steps, args.lr_speed, "speed")
    Ls = speed_chol(th, speed)
    out["arms"]["speed"] = {"loss": l_speed, "theta": np.asarray(th).tolist(),
                            **describe(Ls, speed)}

    # -- the bound: free per-tick Cholesky ---------------------------------------
    g_free = lambda raw, b: f(chol_from_raw(raw), b)                  # noqa: E731
    raw0 = jnp.asarray(np.tile(np.diag(np.full(3, -4.0)), (B, L, n_c, 1, 1)))
    raw, l_free = optimise(g_free, raw0, batch, args.steps, args.lr_free, "free")
    out["arms"]["free"] = {"loss": l_free, **describe(chol_from_raw(raw), speed)}

    h = out["arms"]["heuristic"]["loss"]
    out["verdict"] = {
        "free_vs_heuristic": l_free / h,
        "speed_vs_heuristic": l_speed / h,
        "speed_captures_of_free": ((h - l_speed) / (h - l_free)) if h > l_free else None,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdict"], indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
