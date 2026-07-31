r"""CoCo-InEKF Figure 3, reproduced on our nets — and why our `v_bc` channel may not carry it.

Two measurements, both read-only over the collected rollouts.

(1) The Figure-3 analogue
------------------------
CoCo-InEKF §V-A1 plots the total contact-covariance standard deviation
``sqrt(tr(B Sigma_Ci))`` against the contact point's instantaneous VELOCITY MAGNITUDE
over a forward-walking gait, and reports that the two "agree".  That is the single
most directly comparable figure between their system and ours, so this computes the
same two quantities and correlates them.

The velocity is taken from GROUND TRUTH, not from the ``v_bc`` feature: the world
contact point is ``p_C = p_true + R_true @ y_fk`` (both in the rollout, ``y_fk`` in the
feature cache), differentiated with a centred secant over ``vel_smooth`` ticks.  A
tick-wise gradient of a 1 kHz signal is noise, and `PORT_NOTES.md`'s z-bias entry
records that exact artifact flipping a sign.

(2) The noise floor of the `v_bc` channel
-----------------------------------------
`features.make_contact_channels` builds ``v_bc`` as a **causal first difference of
FK(q) at 1 kHz**, on NOISY encoders, deliberately -- "not J q̇, which would
reintroduce the joint-KF coupling this feature set exists to avoid".  Differentiating
white encoder noise at 1 ms amplifies it by ``sqrt(2)/dt = 1414``, so the question is
whether anything survives.

The noise is measured, not assumed.  For a signal that is smooth on a 1 ms scale plus
white noise ``eps``, the second difference ``p[k+1] - 2p[k] + p[k-1]`` has variance
``6 sigma_eps^2``, so ``sigma_eps = std(d2p)/sqrt(6)``.  From that:

* ``sigma_v = sqrt(2) sigma_eps / dt``            -- the noise on the channel as built;
* ``|J|_eff = sigma_eps / sigma_q``               -- the effective FK gain, from the
  known encoder noise (`sim.sensors.IMUNoise.encoder_std`);
* ``sigma_v,analytic = |J|_eff * sigma_qdot``     -- what an analytic ``J_C q̇`` would
  cost instead, from the measured joint-velocity noise.

The ratio of the last two is the case for or against option B (replace the finite
difference with an analytic contact-point velocity).

Usage
-----
    uv run python -m experiments.contact_velocity --data data/dr5 \
        --checkpoint artifacts/contactnet_run7.npz
    uv run python -m experiments.contact_velocity --data data/dr5 --no-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WARMUP = 16_000
DT = 1.0e-3

ENCODER_STD = 2.0e-4
"""`sim.sensors.IMUNoise.encoder_std` [rad] -- the corruption actually injected."""
ENCODER_VEL_STD = 5.0e-3
"""`sim.sensors.IMUNoise.encoder_vel_std` [rad/s] -- what an analytic J q̇ would use."""


def secant(x: np.ndarray, half: int) -> np.ndarray:
    """Centred secant derivative over ``2*half`` ticks; edge-padded, same length.

    A secant rather than `np.gradient`: at 1 kHz the tick-wise centred difference of
    an encoder-derived quantity is dominated by the differentiated noise, which is the
    whole point of measurement (2).
    """
    pad = np.concatenate([np.repeat(x[:1], half, 0), x, np.repeat(x[-1:], half, 0)])
    return (pad[2 * half:] - pad[:-2 * half]) / (2 * half * DT)


def world_contact_velocity(path: Path, cache: Path, smooth: int) -> np.ndarray | None:
    """`(T, N_c)` true world-frame contact-point speed [m/s]."""
    c = cache / f"{path.stem}_feat.npz"
    if not c.exists():
        return None
    with np.load(c) as z:
        y = np.asarray(z["y_fk"])                       # (T, N_c, 3), body frame
    with np.load(path, allow_pickle=True) as z:
        R = np.asarray(z["truth.R"])
        p = np.asarray(z["truth.p"])
    n = min(len(y), len(R))
    pc = p[:n, None, :] + np.einsum("tij,tcj->tci", R[:n], y[:n])
    return np.linalg.norm(secant(pc, smooth), axis=-1)


def vbc_noise(path: Path, cache: Path) -> dict:
    """Noise floor of the `p_bc` / `v_bc` channels, from the second difference."""
    with np.load(cache / f"{path.stem}_feat.npz") as z:
        chan = np.asarray(z["channels"])[WARMUP:]
    p = chan[..., 18:21]                                 # p_bc_{x,y,z}
    v = chan[..., 21:24]                                 # v_bc_{x,y,z}, as the net sees it
    d2 = p[2:] - 2.0 * p[1:-1] + p[:-2]
    sig_eps = float(np.std(d2)) / np.sqrt(6.0)
    sig_v = np.sqrt(2.0) * sig_eps / DT
    j_eff = sig_eps / ENCODER_STD
    return {
        "sigma_p_noise_m": sig_eps,
        "sigma_v_noise_ms": sig_v,
        "J_eff_m_per_rad": j_eff,
        "sigma_v_analytic_ms": j_eff * ENCODER_VEL_STD,
        "ratio": sig_v / max(j_eff * ENCODER_VEL_STD, 1e-30),
        "v_bc_std_ms": float(np.std(v)),
        "v_bc_rms_ms": float(np.sqrt((v ** 2).mean())),
    }


def learned_std(paths, cache, checkpoint, norm_path, stride):
    """`(rollout -> (T', N_c))` total std `sqrt(tr(Sigma))`, on a tick subsample."""
    import jax
    import jax.numpy as jnp
    from invariant_estimation.contactnet import features, network, normalize, train
    from invariant_estimation.contactnet.config import ContactNetConfig

    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4)
    like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths, cfg.sigma_0, cfg.eps)
    params = train.load_params(checkpoint, like)
    nc = normalize.load(str(norm_path))
    fwd = jax.jit(jax.vmap(jax.vmap(lambda x: network.forward(params, x, cfg.eps))))
    out = {}
    for p in paths:
        c = cache / f"{p.stem}_feat.npz"
        if not c.exists():
            continue
        with np.load(c) as z:
            chan = np.asarray(z["channels"])
        w = features.window(normalize.apply(jnp.asarray(chan), nc), cfg.H, cfg.stride)
        starts = np.arange(WARMUP, chan.shape[0], stride)
        tot = []
        for k in range(0, len(starts), 4096):
            s = starts[k:k + 4096]
            L = np.asarray(fwd(w[s].reshape(len(s), chan.shape[1], -1)))
            # tr(L Lᵀ) = sum of every element squared.
            tot.append(np.sqrt((L ** 2).sum(axis=(-2, -1))))
        out[p.stem] = np.concatenate(tot)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(REPO / "data" / "dr5"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--vel-smooth", type=int, default=10, help="secant half-width [ticks]")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache) if args.cache else data / "cache"
    norm = Path(args.norm) if args.norm else data / "norm_constants.npz"
    paths = [p for p in sorted(data.glob("*.npz")) if p.name != "norm_constants.npz"]
    paths = [p for p in paths if (cache / f"{p.stem}_feat.npz").exists()]
    report = {"data": str(data)}

    # -- (2) the noise floor ------------------------------------------------
    rows = [vbc_noise(p, cache) for p in paths]
    agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    report["vbc_noise"] = agg
    print(f"(2) v_bc noise floor, median over {len(rows)} rollouts")
    print(f"    FK position noise    sigma_p       = {agg['sigma_p_noise_m']:.3e} m")
    print(f"    effective FK gain    |J|_eff       = {agg['J_eff_m_per_rad']:.3f} m/rad")
    print(f"    v_bc AS BUILT        sigma_v       = {agg['sigma_v_noise_ms']:.4f} m/s "
          f"(1-tick difference at dt={DT})")
    print(f"    v_bc analytic J qdot sigma_v       = {agg['sigma_v_analytic_ms']:.4f} m/s")
    print(f"    ratio (as built / analytic)        = {agg['ratio']:.1f}x")
    print(f"    channel std / rms                  = {agg['v_bc_std_ms']:.4f} / "
          f"{agg['v_bc_rms_ms']:.4f} m/s")

    # -- (1) the Figure-3 analogue -----------------------------------------
    if args.checkpoint and not args.no_model:
        stds = learned_std(paths, cache, args.checkpoint, norm, args.stride)
        S, V = [], []
        for p in paths:
            if p.stem not in stds:
                continue
            spd = world_contact_velocity(p, cache, args.vel_smooth)
            sd = stds[p.stem]
            starts = np.arange(WARMUP, len(spd), args.stride)[:len(sd)]
            S.append(sd[:len(starts)])
            V.append(spd[starts])
        S, V = np.concatenate(S).ravel(), np.concatenate(V).ravel()
        good = np.isfinite(S) & np.isfinite(V)
        S, V = S[good], V[good]
        from experiments.slip_probe import r2_binned
        r_p = float(np.corrcoef(np.log10(np.maximum(S, 1e-30)), V)[0, 1])
        r2 = r2_binned(np.log10(np.maximum(S, 1e-30)), V, 24)
        report["figure3"] = {"pearson_log10std_vs_speed": r_p, "r2_binned": r2,
                             "n": int(S.size),
                             "std_p10": float(np.percentile(S, 10)),
                             "std_p50": float(np.median(S)),
                             "std_p90": float(np.percentile(S, 90))}
        print(f"\n(1) Figure-3 analogue, {args.checkpoint} ({S.size} samples)")
        print(f"    total std sqrt(tr Sigma): p10/p50/p90 = "
              f"{np.percentile(S, 10):.3f} / {np.median(S):.3f} / "
              f"{np.percentile(S, 90):.3f}")
        print(f"    true contact speed [m/s]: p10/p50/p90 = "
              f"{np.percentile(V, 10):.3f} / {np.median(V):.3f} / "
              f"{np.percentile(V, 90):.3f}")
        print(f"    corr(log10 std, speed) = {r_p:+.3f}")
        print(f"    R^2 of log10 std on binned speed = {r2:.3f}")
        # Split by regime: the claim is about the SHAPE, and a swing/stance
        # step function would produce a high correlation on its own.
        for lo, hi, lab in ((0.0, 0.05, "near-static"), (0.05, 0.3, "creeping"),
                            (0.3, 99.0, "swinging")):
            m = (V >= lo) & (V < hi)
            if m.sum() > 500:
                print(f"      {lab:12s} v in [{lo:.2f},{hi:.2f}): n={m.sum():7d}  "
                      f"median std = {np.median(S[m]):.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
        print(f"  -> {args.json}")


if __name__ == "__main__":
    main()
