r"""Measure ``T*`` — the horizon at which the contact update starts paying off.

Why this number exists
----------------------
`contactnet.dataset.make_segment` seeds every training segment from ground
truth, so a segment begins with **zero** estimation error and runs for
``L*dt = 128 ms``.  The contact update is a bias-variance trade: it removes
accumulated IMU dead-reckoning error and injects the contact measurement's own
error (slip, sole compliance, FK error).  It is worth taking only once the
former exceeds the latter.

``T*`` is the crossover.  If ``T* > L*dt`` the objective
``L_L2(Sigma_C; x0_hat, L)`` has **no interior minimum in Sigma_C** — the best
available answer is "ignore the feet", ``Sigma_C -> infinity`` — which is
exactly what run 1 learned (PORT_NOTES, "Run 1 learned to switch the contact
update OFF": 3835x velocity-gain suppression).

So this script does not measure a property of the network.  It measures a
property of the *protocol*, and it is the number that decides whether the
training objective is degenerate at all.

Method
------
From a truth seed at ``t0``, run the InEKF forward twice over the *same*
recorded inputs, under exactly the training conventions (`measure_p0`'s
docstring lists them), differing only in the contact **measurement** noise:

* **ON**  — ``Sigma_C = sigma_0^2 I``, the network's initialization, which is
  negligible against ``N = J Sigma_q J^T`` and therefore reproduces the
  shipped filter's contact update.
* **OFF** — ``Sigma_C = (1e3 m)^2 I``.  Not a code path change: the same update
  runs with a gain driven to ~0, which is what an infinite ``Sigma_C`` means and
  what run 1 converged to.  Masking the update instead would change the graph.

Both carry ``contact_chol`` frozen at ``contact_chol_const`` — the process
socket is *not* the variable here (theory doc S3.3: in stance it is the floor
alone), and freezing it is what a training segment does.

We record per-tick body-frame velocity error — `losses.l2_velocity`'s exact
integrand, so the number is the training objective and not a proxy — and report
the first horizon at which ON beats OFF and stays beating it.

Usage
-----
    uv run python -m experiments.measure_tstar --ticks 5000 --starts 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from invariant_estimation.contactnet import dataset, normalize
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet.losses import l2_velocity
from invariant_estimation.inEKF import ekf as inekf_mod
from invariant_estimation.inEKF.filter import init_carry, make_step
from invariant_estimation.sim import collect

REPO = Path(__file__).resolve().parent.parent
SIGMA_OFF = 1.0e3
"""Cholesky diagonal for the "contact update off" arm: ``Sigma_C = 1e6 m^2``.

Nine orders above ``N = J Sigma_q J^T = 1.26e-5 m^2``, so the gain is ~0 to
machine precision, without a branch or a mask that would change the graph.
"""


def error_curves(fused, prep: dataset.PreparedRollout, cfg: ContactNetConfig,
                 t0: int, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """``(A, B, C)``, each ``(ticks,)`` body-frame squared velocity error.

    One truth seed, one input stream, two contact measurement noises.  Anything
    that differs between the arms other than ``contact_meas_chol`` would make
    the crossover meaningless, so the seed and the inputs are built once.
    """
    n_c = prep.y_fk.shape[1]
    xs = jax.tree.map(lambda a: jnp.asarray(a[t0:t0 + ticks]), prep.inputs)
    chol_recorded = xs.contact_chol                    # sim stance/swing switching
    chol_frozen = jnp.asarray(
        dataset._constant_contact_chol(cfg, ticks, n_c))   # what training uses

    d0 = jnp.asarray(np.einsum("ij,kj->ki", prep.R_true[t0], prep.y_fk[t0])
                     + prep.p_true[t0][None, :])
    state0 = inekf_mod.initialize(
        fused.ekf,
        rotation=jnp.asarray(prep.R_true[t0]),
        velocity=jnp.asarray(prep.v_true[t0]),
        position=jnp.asarray(prep.p_true[t0]),
        contacts=d0,
    )
    # Seeded from truth => zero error, so P must be the covariance a *running*
    # filter carries, not the diffuse prior.  Same argument as `measure_p0`.
    P0 = dataset.measure_p0(fused, prep, cfg, ticks=3000, verbose=False)
    state0 = state0._replace(P=jnp.asarray(P0))

    step = make_step(fused.ekf, fused.kinematics)
    v_true = jnp.asarray(prep.v_true[t0:t0 + ticks])
    R_true = jnp.asarray(prep.R_true[t0:t0 + ticks])

    def run(chol_diag: float, contact_chol) -> np.ndarray:
        L_c = jnp.broadcast_to(
            chol_diag * jnp.eye(3, dtype=jnp.float64), (ticks, n_c, 3, 3))
        _, out = jax.lax.scan(
            step, init_carry(state0),
            xs._replace(contact_meas_chol=L_c, contact_chol=contact_chol))
        # l2_velocity means over its leading axis; we want it per tick, so the
        # integrand is recomputed here rather than the mean.
        e = (jnp.einsum("kji,kj->ki", out.state.R, out.state.v)
             - jnp.einsum("kji,kj->ki", R_true, v_true))
        return np.asarray(jnp.sum(e * e, axis=-1))

    # Three arms.  A vs B isolates the contact update under the TRAINING config;
    # A vs C isolates the frozen process socket, which is the only thing the
    # training config changes about the filter (`make_segment`).
    return (run(cfg.sigma_0, chol_frozen),      # A: training, contacts on
            run(SIGMA_OFF, chol_frozen),        # B: training, contacts off
            run(cfg.sigma_0, chol_recorded))    # C: deployed, contacts on


def crossover(err_on: np.ndarray, err_off: np.ndarray, dt: float,
              hold: int) -> float | None:
    """First time [s] after which ON beats OFF for at least `hold` consecutive ticks.

    `hold` exists because the two curves cross noisily near the start; a single
    tick where ON happens to be lower is not the crossover.
    """
    better = err_on < err_off
    if not better.any():
        return None
    # Run-length: earliest index starting a `hold`-long all-True run.
    c = np.convolve(better.astype(int), np.ones(hold, dtype=int), mode="valid")
    idx = np.flatnonzero(c == hold)
    return None if idx.size == 0 else float(idx[0] * dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--cache", default=str(REPO / "data/cache"))
    ap.add_argument("--norm", default=str(REPO / "data/norm_constants.npz"))
    ap.add_argument("--ticks", type=int, default=5000, help="horizon per run [ticks]")
    ap.add_argument("--starts", type=int, default=8, help="truth seeds to average over")
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--hold", type=int, default=200,
                    help="consecutive ticks ON must stay ahead to count as crossover")
    ap.add_argument("--out", default=str(REPO / "artifacts/tstar.npz"))
    args = ap.parse_args()

    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4, remat=False)
    fused = collect.build_collector(verbose=False).fused
    norm = normalize.load(args.norm)
    paths = dataset.rollout_paths(args.data)[:args.rollouts]
    preps = dataset.prepare(paths, norm, cfg, cache_dir=args.cache, verbose=False)

    print(f"T* measurement: {len(preps)} rollouts x {args.starts} seeds, "
          f"{args.ticks} ticks ({args.ticks * cfg.dt:.1f} s) each")
    print(f"  Sigma_C  ON = {cfg.sigma_0**2:.2e} m^2   OFF = {SIGMA_OFF**2:.2e} m^2")
    print(f"  segment horizon for comparison: L*dt = {cfg.L * cfg.dt * 1e3:.0f} ms\n")

    A, B, C = [], [], []
    for prep in preps:
        hi = min(prep.t_hi, prep.smoothed.shape[0] - args.ticks - 1)
        if hi <= prep.t_lo:
            print(f"  {prep.name}: too short for {args.ticks} ticks, skipped")
            continue
        for t0 in np.linspace(prep.t_lo, hi, args.starts).astype(int):
            a, b, c = error_curves(fused, prep, cfg, int(t0), args.ticks)
            A.append(a); B.append(b); C.append(c)
            print(f"  {prep.name} t0={t0:6d}  |v|err@128ms  "
                  f"A(train,on) {np.sqrt(a[cfg.L]):.3e}  "
                  f"B(train,off) {np.sqrt(b[cfg.L]):.3e}  "
                  f"C(deploy,on) {np.sqrt(c[cfg.L]):.3e}")

    A, B, C = np.stack(A), np.stack(B), np.stack(C)
    mA, mB, mC = A.mean(0), B.mean(0), C.mean(0)
    t_star = crossover(mA, mB, cfg.dt, args.hold)

    print("\n  A = training config (contact_chol frozen at stance), contact update ON")
    print("  B = training config, contact update OFF  (Sigma_C -> inf)")
    print("  C = deployed config (recorded stance/swing chol), contact update ON")
    print(f"\n{'horizon':>10s}  {'A RMS|v|err':>13s}  {'B':>11s}  {'C':>11s}  "
          f"{'A/B':>7s}  {'C/B':>7s}")
    for t_ms in (10, 50, 128, 250, 500, 1000, 2000, 4000):
        k = int(t_ms / (cfg.dt * 1e3))
        if k >= A.shape[1]:
            continue
        ra, rb, rc = np.sqrt(mA[k]), np.sqrt(mB[k]), np.sqrt(mC[k])
        mark = "  <- L*dt" if t_ms == 128 else ""
        print(f"{t_ms:8d}ms  {ra:13.4e}  {rb:11.4e}  {rc:11.4e}  "
              f"{ra/rb:7.3f}  {rc/rb:7.3f}{mark}")

    print()
    if t_star is None:
        print(f"T* > {args.ticks * cfg.dt:.1f} s — the contact update never overtakes "
              f"dead reckoning within the measured horizon.")
    else:
        print(f"T* = {t_star:.3f} s  ({t_star / (cfg.L * cfg.dt):.1f}x the "
              f"{cfg.L * cfg.dt * 1e3:.0f} ms segment horizon)")
    print("DEGENERATE: the L2 objective has no interior optimum in Sigma_C "
          "on truth-seeded segments."
          if (t_star is None or t_star > cfg.L * cfg.dt) else
          "Segments are long enough for the contact update to pay off.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, err_a=A, err_b=B, err_c=C, dt=cfg.dt, L=cfg.L,
             t_star=np.nan if t_star is None else t_star)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
