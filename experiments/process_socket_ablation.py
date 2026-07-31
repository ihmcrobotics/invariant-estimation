r"""Phase 0 of `branch_out.md`: is the vertical sink sensitive to *process-side*
covariance at all?  No network, no training, recorded data only.

This is the gate that can cancel the whole `contactnet/process-socket` branch, so
it runs before anything is rewired.  The mechanism `branch_out.md` §1 proposes,
in one paragraph -- **refuted by this script's own §6 output; see "What it
measured" below, and read the conclusion before the premise**:

    At liftoff the foot rises, so ``y_z`` grows while ``d̂`` is still pinned where
    stance left it, giving ``ν_z > 0``; ``exp(−Kν)`` then drives the base down.
    The fraction of that residual landing on the base is ``P_pp/(P_pp + P_dd)``,
    and ``P_dd`` is at its **tightest** exactly then, because the foot just spent
    a whole stance being told it was world-static.  Late in swing ``P_dd`` is
    large and the anchor absorbs the descent instead.  One rectified downward
    dose per step per foot.

Every arm replays the *same* recorded rollout from the *same* truth seed and
differs in exactly one field of `InEKFInputs`:

    A   the recorded heuristic ``contact_chol``                    baseline
    B   the same, but loosened ``N`` ticks BEFORE liftoff          is it timing?
    C   the same, with the stance value swept up                   is it tightness?
    D   run 4's network on ``contact_meas_chol``                   control
    F   the heuristic, with ``contact_floor`` swept                 what does the
                                                                   floor buy?
    I   a CONSTANT chol at every phase                             where does a
                                                                   run start?
    N   the same network, on ``contact_chol``                       the head-to-head

Arm B is deliberately **non-causal** — liftoff is known offline, so the swing
value is dilated backwards in time.  It is a mechanism test, not a deployable
filter; nothing here is meant to ship.

Decision rule (`branch_out.md` §1)
----------------------------------
* The sink drops materially in **B** or **C** ⇒ the process socket is a real
  lever, and Phase 1 (the socket move) is worth doing.
* The sink is flat across B and C ⇒ **stop.**  The process socket is not the
  lever and a retrain would be wasted.

What it measured (2026-07-29, `data/dr`, 3 rollouts x 2 seeds x 20 s)
---------------------------------------------------------------------
``slope(e_pz)`` [m/s]: A −0.03750; B at −10/−25/−50/−100 ticks −0.03110 /
−0.02290 / −0.01253 / **−0.00200**; C 1e−4 → 1e−2 flat at −0.0375; D −0.00490.
Monotone in the shift, **19x** at 100 ticks, with ``height_rms`` down 12x — and
no response at all to the stance *value*.  So the sink is controlled by **when**
the anchor is released, not by how tightly it is held.  Gate passed.

Two things the run corrected, both of which are in `PORT_NOTES.md`:

* **Arm D is 7.7x better than baseline too.**  ``N`` cannot change the base/anchor
  *split* — that argument holds — but it scales every correction, and the sink is
  an accumulated dose.  "ContactNet was holding a knob that provably cannot reach
  the sink" is too strong; the process socket is the *more direct* lever, not the
  only one.
* **The §1 mechanism above is wrong.**  The Schmitt trigger releases *at* liftoff,
  so ``P_dd`` is already at its swing value on the first tick of swing and the
  early-vs-late-swing asymmetry it posits does not exist.  The real asymmetry is
  **stance vs swing**, and it is a factor of ~1e6 in the velocity gain (2.673 /s
  in late stance against 2.8e-6 /s in swing).  The dose is injected in the last
  ~100 ms of stance, while the foot is unloading but the trigger still says
  "planted" — which is exactly the interval arm B moves, and exactly why arm C
  does nothing.

Section 6 comes for free
------------------------
`--traces` runs the §6 instrumentation on arm A: the ``P_dd``/``P_pp`` vertical
variances through the gait cycle and the contact innovation ``ν_z``.  Both come
out of the replay, so this costs one scan rather than a re-collection of the
dataset.  Two things it must do to be worth reading, both learned the hard way:

* Report the **prior** covariance, not the posterior.  ``Σ_C`` enters through
  ``Q_d`` and the update immediately shrinks most of it back, so the posterior
  measures the quantity after the effect under test has been undone.
* Report the **velocity** row ``(P_vp − P_vd)/S``, not the position row.  Base
  position is unobservable, so ``P_pp``, ``P_pd`` and ``P_dd`` agree to four
  digits and the naive ``P_pp/(P_pp+P_dd)`` reads exactly 0.5 forever.

Usage
-----
    uv run python -m experiments.process_socket_ablation
    uv run python -m experiments.process_socket_ablation --arms A,B --ticks 30000
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from invariant_estimation.contactnet import (dataset, features, network,
                                             normalize, train)
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.sim import collect

from experiments.replay_eval import run_arm

REPO = Path(__file__).resolve().parent.parent

SWING_SPLIT = 1.0e-2
"""Cholesky value separating the heuristic's stance (1e-4) and swing (1e1)
states.  Geometric-mean-ish and nowhere near either, so the classification is not
sensitive to it -- asserted in `stance_mask`."""


# ---------------------------------------------------------------------------
# Rebuilding `contact_chol` from the recorded heuristic
# ---------------------------------------------------------------------------

def chol_scale(chol: np.ndarray) -> np.ndarray:
    r"""``(T, N_c)`` scalar factor from a ``(T, N_c, 3, 3)`` isotropic chol stack.

    `sim/sensors.py` emits ``s·I₃`` with ``s`` switched by the Schmitt trigger,
    so the whole recorded field is one scalar per foot per tick.  Verified rather
    than assumed: an anisotropic recording would silently lose information here.
    """
    s = chol[..., 0, 0]
    if not np.allclose(chol, s[..., None, None] * np.eye(3), atol=0.0, rtol=1e-12):
        raise ValueError(
            "recorded contact_chol is not isotropic; this harness rebuilds it "
            "from a scalar per foot and would discard the anisotropy")
    return s


def as_chol(s: np.ndarray) -> jnp.ndarray:
    """``(T, N_c)`` scalars → ``(T, N_c, 3, 3)`` isotropic Cholesky factors."""
    return jnp.asarray(s[..., None, None] * np.eye(3, dtype=np.float64))


def stance_mask(s: np.ndarray) -> np.ndarray:
    """``True`` where the heuristic says stance.  Bimodal by construction."""
    lo, hi = s[s < SWING_SPLIT], s[s >= SWING_SPLIT]
    if lo.size and (lo.max() > 1e-3 or (hi.size and hi.min() < 1.0)):
        raise ValueError(f"contact_chol is not bimodal around {SWING_SPLIT}: "
                         f"stance max {lo.max():g}, swing min {hi.min() if hi.size else float('nan'):g}")
    return s < SWING_SPLIT


def loosen_early(s: np.ndarray, n: int) -> np.ndarray:
    r"""Dilate the SWING (large) value ``n`` ticks backwards in time.

    A running maximum over the lookahead window ``[t, t+n]``.  Maximum, not a
    plain shift: shifting the whole mask would also move *touchdown* earlier,
    tightening the anchor before the foot is actually down, which is a second
    change in the opposite direction and would make the arm uninterpretable.
    Dilating only the loose state moves liftoff and leaves touchdown alone.

    The tail is edge-padded, so the last `n` ticks see no lookahead.
    """
    if n == 0:
        return s.copy()
    pad = np.concatenate([s, np.repeat(s[-1:], n, axis=0)], axis=0)
    return np.max(np.stack([pad[k:k + len(s)] for k in range(n + 1)]), axis=0)


def retighten(s: np.ndarray, stance_value: float) -> np.ndarray:
    """Same switching, different stance level (arm C)."""
    return np.where(stance_mask(s), stance_value, s)


IMPACT_BLANK_TICKS = 150
"""Ticks of stance excluded from the peak reference in `causal_early_release`.

**Measured, and the reason the first version of arm E failed.** Touchdown is an impact: on
`data/dr5/flat_seed000` the per-stance peak normal force is **2.0-10.8x the stance median**, and it
occurs 1-16% into the stance. A running peak therefore locks onto the impact spike, `frac * peak`
sits above where the foot spends the rest of its stance, and the latch fires almost immediately --
70-86% of stance released against the ~10-20% intended. That is not early release, it is arm I
(constant loose), and it scored monotonically WORSE than the baseline: slope `e_pz` -0.0149 ->
-0.0239 / -0.0319 / -0.0487 at frac 0.3 / 0.5 / 0.7, with `vel_rms` climbing 0.059 -> 0.35.

150 ticks (150 ms) clears the transient on every episode measured while leaving the bulk of a
~514-tick stance to set the reference.
"""


def causal_early_release(s: np.ndarray, fn: np.ndarray, frac: float,
                         blank: int = IMPACT_BLANK_TICKS) -> np.ndarray:
    r"""Arm E — release the anchor when load falls to ``frac`` of its running stance peak.

    The causal counterpart of `loosen_early`, which is arm B's oracle. Arm B reads liftoff from the
    future; this reads only the past, so it is implementable on hardware (the signal is normal
    force, exactly what `ContactTrust` already consumes).

    **Why a fraction of the peak and not a threshold on the level.** A level threshold fires when
    the load has *already gone*; it can be made lower but never earlier, so it is structurally late
    (`docs/theory/anchor_release_timing.md`). The ratio to the stance peak instead tracks where the
    foot is on its own unloading ramp, which starts well before the load reaches any small absolute
    value. ``frac`` is the one knob: 0 recovers the baseline (release only at zero load), larger
    values release progressively earlier on the ramp.

    Two design choices that mirror arm B's:

    * **Only ever loosens.** The result is ``max`` with the heuristic, so this can move liftoff
      earlier but can never move *touchdown* earlier -- tightening the anchor before the foot is
      down would be a second change in the opposite direction and make the arm uninterpretable.
    * **Latched within a stance.** Ground reaction force is double-humped, so a mid-stance dip
      below ``frac * peak`` would otherwise release and then re-tighten, chattering. Once a foot is
      judged to be on its way out it stays out until the next touchdown -- the same fire-once logic
      `inEKF/reseed.py`'s latch uses, and for the same reason.

    Parameters
    ----------
    s : (T, N_c)
        The heuristic chol scalars, per contact.
    fn : (T, N_c)
        Per-contact normal force (``truth.contact_fn``, expanded foot -> contact).
    frac : float
        Release when ``fn < frac * running_peak`` within the current stance.

    Returns
    -------
    (T, N_c) chol scalars, loose wherever the heuristic OR this predictor says so.
    """
    stance = stance_mask(s)
    swing_val = float(s[~stance].max()) if (~stance).any() else float(s.max())
    released = np.zeros_like(fn, dtype=bool)

    for c in range(fn.shape[1]):
        f = fn[:, c]
        loaded = f > 0.0
        # Contact episodes = maximal runs of `loaded`. Within each, the running peak is a
        # prefix maximum and the latch is a prefix OR -- both vectorised, no per-tick loop.
        edges = np.flatnonzero(np.diff(np.concatenate([[0], loaded.view(np.int8), [0]])))
        for lo, hi in zip(edges[::2], edges[1::2]):
            seg = f[lo:hi]
            idx = np.arange(len(seg))
            # Running peak over the POST-IMPACT load only (see `IMPACT_BLANK_TICKS`). Still
            # causal: the blanking window looks backwards from the current tick, never forwards.
            ref = np.maximum(np.maximum.accumulate(np.where(idx < blank, 0.0, seg)), 1e-9)
            rel = (idx >= blank) & (seg < frac * ref)
            released[lo:hi, c] = np.maximum.accumulate(rel)      # latch: stays released
        released[~loaded, c] = True          # no load at all: unambiguously not planted

    return np.where(released, swing_val, s)


def expand_feet(a: np.ndarray, n_c: int) -> np.ndarray:
    """Per-foot ``(T, K)`` -> per-contact ``(T, N_c)``, foot-major (heel, toe) per foot."""
    K = a.shape[1]
    if n_c == K:
        return a
    if n_c % K:
        raise ValueError(f"cannot map {K} feet onto {n_c} contacts")
    return np.repeat(a, n_c // K, axis=1)


# ---------------------------------------------------------------------------
# Section 6: the P_dd / P_pp asymmetry, and the innovation dose
# ---------------------------------------------------------------------------

def liftoffs(mask: np.ndarray) -> list[int]:
    """Indices where a foot leaves stance (True → False)."""
    return list(np.flatnonzero(mask[:-1] & ~mask[1:]) + 1)


def _s_zz(res: dict, i: int) -> np.ndarray:
    """``(H P Hᵀ)_zz`` for contact ``i`` — the prior residual variance."""
    return res["P_pp_z"] - 2.0 * res["P_pd_z"][:, i] + res["P_dd_z"][:, i]


def base_share(res: dict, i: int) -> np.ndarray:
    r"""``(T,)`` prior share of contact ``i``'s vertical residual landing on the base.

    For one contact ``H = [0 0 I −I]`` over ``(R, v, p, d)``, so the ``p``-row of
    ``P Hᵀ`` is ``P_pp − P_pd`` and ``H P Hᵀ = P_pp − 2P_pd + P_dd``.  The ratio
    is what the gain applies to the base before the measurement noise ``N`` is
    added — and ``N`` sits *inside* the inverted factor, so it scales the
    correction without changing this split.  That is the apportionment argument,
    evaluated rather than asserted.

    In practice this is ~1e-6: base position is unobservable, every block is
    dominated by the same global mode, and the anchor absorbs essentially the
    whole positional residual.  `velocity_gain` is the channel that carries the
    sink.
    """
    return (res["P_pp_z"] - res["P_pd_z"][:, i]) / _s_zz(res, i)


def velocity_gain(res: dict, i: int) -> np.ndarray:
    r"""``(T,)`` vertical velocity correction per unit vertical contact residual.

    The ``v``-row of ``P Hᵀ`` is ``P_vp − P_vd``, so ``(P_vp − P_vd)_z / S_z``
    [1/s] is how much world-z velocity the filter injects per metre of ``ν_z``.
    A rectified per-step dose on **this** row integrates straight into a position
    ramp, which is what the sink measures.
    """
    return (res["P_vp_z"] - res["P_vd_z"][:, i]) / _s_zz(res, i)


PHASES = ("stance", "swing0", "swing1")
"""The three stride windows the trace is reported over: the last `window` ticks
of stance, the first `window` after liftoff, and the last `window` before the
foot lands again."""


def trace_report(res: dict, s_heur: np.ndarray, dt: float, window: int = 50) -> dict:
    r"""Where in the stride does the anchor actually tighten?

    §1 proposes that ``P_dd`` is tightest just after liftoff, "because the foot
    just spent a whole stance being told it was world-static".  Comparing two
    *swing* windows cannot test that, because the Schmitt trigger releases **at**
    liftoff: the process noise is already at its swing value on the first tick of
    swing.  The comparison that carries information is therefore **late stance vs
    early swing** — which is also precisely the interval arm B moves.

    Also reports the per-liftoff **velocity dose** ``−Σ g_v ν_z Δt`` [m/s]: the
    world-z velocity the contact update injects over the first `window` ticks of
    swing.  Applied once per step per foot and never undone, that is an
    integrated position ramp, i.e. the sink.
    """
    nu = res["nu_z"]
    T, n_c = res["P_dd_z"].shape
    gv_at: dict[str, list[float]] = {p: [] for p in PHASES}
    dose: list[float] = []
    blocks: dict[str, list[float]] = {}
    for i in range(n_c):
        f, gv = base_share(res, i), velocity_gain(res, i)
        m = stance_mask(s_heur[:, i])
        for k in liftoffs(m):
            a, b = k, min(k + window, T)
            if b - a < window or k - window < 0 or not m[k - window]:
                continue                             # truncated, or stance too short
            # The last `window` ticks before the foot lands again.
            nxt = next((j for j in range(b, T) if m[j]), None)
            if nxt is None or nxt - window <= b:
                continue
            spans = dict(stance=(k - window, k), swing0=(a, b), swing1=(nxt - window, nxt))
            for tag, (lo, hi) in spans.items():
                gv_at[tag].append(float(np.mean(gv[lo:hi])))
                for key, arr in (("P_pp", res["P_pp_z"]), ("P_pd", res["P_pd_z"][:, i]),
                                 ("P_dd", res["P_dd_z"][:, i]), ("f", f), ("gv", gv),
                                 ("nu", nu[:, i])):
                    blocks.setdefault(f"{key}_{tag}", []).append(float(np.mean(arr[lo:hi])))
            # NEGATIVE is downward: the correction is `exp(−Kν)`, so a positive
            # `ν_z` against a positive `g_v` removes upward velocity.
            dose.append(float(-np.sum(gv[a:b] * nu[a:b, i]) * dt))
    out = dict(n=len(gv_at["swing0"]), nu_z_mean=float(np.mean(nu)),
               dose=float(np.mean(dose)) if dose else float("nan"))
    out.update({f"gv_{p}": float(np.mean(v)) if v else float("nan")
                for p, v in gv_at.items()})
    out.update({k: float(np.mean(v)) for k, v in blocks.items()})
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def with_floor(fused, floor: float):
    r"""``fused`` with `InEKFParams.contact_floor` replaced — arm F's only change.

    The floor is **not** an input, it is filter configuration: `contact.digest`
    applies ``Σ ← Σ + floor·I`` before ``Σ_C`` reaches ``Q_d``.  So unlike every
    other arm here, arm F patches the estimator rather than the per-tick chol.
    `run_arm` rebuilds its own `make_step` from `fused.ekf`, so replacing the
    params is sufficient and nothing stale survives.
    """
    return dataclasses.replace(
        fused, ekf=fused.ekf._replace(
            params=fused.ekf.params._replace(contact_floor=floor)))


def build_arms(spec: str, s_heur: np.ndarray, shifts, stances, floors,
               fn: np.ndarray | None = None, early_fracs=()) -> list[tuple]:
    """``[(label, socket, chol, floor), ...]`` for the requested arm letters.

    ``floor=None`` means "leave the estimator alone"; only arm F sets it.
    """
    want = {c.strip().upper() for c in spec.split(",") if c.strip()}
    arms = []
    if "A" in want:
        arms.append(("A  heuristic", "process", as_chol(s_heur), None))
    if "B" in want:
        for n in shifts:
            arms.append((f"B  loosen -{n:>3d} ticks", "process",
                         as_chol(loosen_early(s_heur, n)), None))
    if "C" in want:
        for v in stances:
            arms.append((f"C  stance {v:.0e}", "process",
                         as_chol(retighten(s_heur, v)), None))
    if "E" in want:
        if fn is None:
            raise SystemExit("arm E needs `truth.contact_fn`; the dataset was collected "
                             "without slip instrumentation (`dr.slip.record`)")
        for v in early_fracs:
            arms.append((f"E  causal frac {v:.2f}", "process",
                         as_chol(causal_early_release(s_heur, fn, v)), None))
    if "I" in want:
        # What the RUN ACTUALLY STARTS FROM. `network.init` zeroes the output
        # head, so iteration 0 emits a constant `sigma_0 * I` for every foot at
        # every gait phase -- stance and swing alike. Arm C only moves the stance
        # value, so it does not answer this; and "the constant is between the two
        # heuristic values" is not an argument, it is a guess.
        for v in stances:
            arms.append((f"I  init const {v:.0e}", "process",
                         as_chol(np.full_like(s_heur, v)), None))
    if "F" in want:
        # `contact_floor` saturates everything below chol ~1e-2: at the
        # heuristic's stance value (chol 1e-4 => Sigma 1e-8) the floor supplies
        # 100% of the digested covariance, so the network's output there has no
        # effect on the filter and therefore no gradient. This arm asks what the
        # floor is actually buying, since it costs the learned Sigma_C the bottom
        # four decades of its range.
        for f in floors:
            arms.append((f"F  floor {f:.0e}", "process", as_chol(s_heur), f))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "data/dr"))
    ap.add_argument("--cache", default=None, help="default <data>/cache")
    ap.add_argument("--norm", default=None, help="default <data>/norm_constants.npz")
    # P0 is measured per dataset (see replay_eval); the DR set has its own.
    ap.add_argument("--p0", default=str(REPO / "artifacts/p0_dr.npz"),
                help="p0_dr is right for arms driven at the OLD conventions; "
                     "use p0_process_dr.npz once the network drives contact_chol")
    ap.add_argument("--checkpoint", default=str(REPO / "artifacts/contactnet_run4.npz"),
                    help="arms D and N")
    # Third tool to need this (after `replay_eval` and `alpha_sweep`): an N=4 dataset stores
    # (T, 4, 3, 3) contact arrays and the N=2 default dies inside the first propagation with
    # `dot_general ... got (15,) and (21,)`.
    ap.add_argument("--toe-heel", dest="toe_heel", action="store_true",
                    help="N=4 (heel+toe per foot); must match how the dataset was collected")
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--shifts", default="0,10,25,50,100")
    ap.add_argument("--stances", default="1e-4,1e-3,1e-2")
    ap.add_argument("--early-fracs", default="0.3,0.5,0.7,0.9",
                    help="arm E: release when load falls below this fraction of its running "
                         "stance peak. Causal; arm B at the same lead is its ceiling.")
    ap.add_argument("--floors", default="1e-4,1e-5,1e-6",
                    help="arm F: InEKFParams.contact_floor sweep")
    ap.add_argument("--ticks", type=int, default=20_000)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--traces", action="store_true", default=True,
                    help="run the §6 P_dd/P_pp + innovation report (default on)")
    ap.add_argument("--no-traces", dest="traces", action="store_false")
    args = ap.parse_args()

    shifts = [int(x) for x in args.shifts.split(",")]
    stances = [float(x) for x in args.stances.split(",")]
    floors = [float(x) for x in args.floors.split(",")]
    early_fracs = [float(x) for x in args.early_fracs.split(",") if x.strip()]
    _want = {c.strip().upper() for c in args.arms.split(",")}
    want_d, want_n = "D" in _want, "N" in _want

    n_c = 4 if args.toe_heel else 2
    cfg = ContactNetConfig(F=24, sigma_0=1.0e-4, n_contacts=n_c)
    fused = collect.build_collector(verbose=False, toe_heel=args.toe_heel).fused
    if int(fused.n_contacts) != n_c:
        raise SystemExit(f"estimator has {fused.n_contacts} contacts, expected {n_c}")
    data = Path(args.data)
    norm = normalize.load(str(Path(args.norm) if args.norm
                              else data / "norm_constants.npz"))
    preps = dataset.prepare(dataset.rollout_paths(data)[:args.rollouts],
                            norm, cfg, cache_dir=Path(args.cache) if args.cache else data / "cache",
                            verbose=False)
    P0 = np.load(args.p0)["P0"]

    # Arm E's input signal. `truth.contact_fn` is the summed per-foot normal force recorded at
    # collection time -- a SENSOR quantity, the same class `ContactTrust` already consumes, not
    # privileged ground truth about liftoff. Absent on datasets collected without slip
    # instrumentation, in which case arm E refuses rather than silently degrading.
    fn_all: dict[str, np.ndarray] = {}
    for p in dataset.rollout_paths(data)[:args.rollouts]:
        with np.load(p, allow_pickle=False) as z:
            if "truth.contact_fn" in z.files:
                meta = __import__("json").loads(str(np.load(p, allow_pickle=False)["meta"]))
                fn_all[f"{meta['terrain']}/seed{meta['seed']}"] = np.asarray(z["truth.contact_fn"])

    fwd = None
    if want_d or want_n:
        like = network.init(jax.random.PRNGKey(0), cfg.d_in, cfg.widths,
                            cfg.sigma_0, cfg.eps)
        params = train.load_params(args.checkpoint, like)
        fwd = jax.jit(jax.vmap(jax.vmap(
            lambda x: network.forward(params, x, cfg.eps))))

    print(f"process-socket ablation: {data}  P0={args.p0}")
    print(f"  {len(preps)} rollouts x {args.starts} seeds x "
          f"{args.ticks * cfg.dt:.0f} s\n")

    rows: dict[str, list[dict]] = {}
    traces: list[dict] = []
    for prep in preps:
        w = features.window(jnp.asarray(prep.smoothed), cfg.H, cfg.stride) if want_d else None
        hi = min(prep.t_hi, prep.smoothed.shape[0] - args.ticks - 1)
        if hi <= prep.t_lo:
            continue
        for t0 in np.linspace(prep.t_lo, hi, args.starts).astype(int):
            t0 = int(t0)
            sl = slice(t0, t0 + args.ticks)
            s_heur = chol_scale(np.asarray(prep.inputs.contact_chol[sl]))
            fn_c = (expand_feet(fn_all[prep.name][sl], s_heur.shape[1])
                    if prep.name in fn_all else None)
            arms = build_arms(args.arms, s_heur, shifts, stances, floors,
                              fn=fn_c, early_fracs=early_fracs)
            if want_d or want_n:
                flat = w[sl].reshape(args.ticks, w.shape[1], -1)
                L_net = fwd(flat)
                # The SAME network output, driven into the two different sockets --
                # which is the only apples-to-apples statement about the move,
                # since every other confound (P0, seeds, rollouts) is held fixed.
                if want_d:
                    arms.append((f"D  net on N (meas)", "meas", L_net, None))
                if want_n:
                    arms.append((f"N  net on Q_d (proc)", "process", L_net, None))

            for label, socket, chol, floor in arms:
                first = label not in rows
                res = run_arm(with_floor(fused, floor) if floor is not None else fused,
                              prep, cfg, t0, args.ticks, chol, P0,
                              socket=socket,
                              want_traces=args.traces and label.startswith("A"))
                rows.setdefault(label, []).append(res)
                if args.traces and label.startswith("A"):
                    traces.append(trace_report(res, s_heur, cfg.dt))
                if first:
                    print(f"  ran {label}")

    if traces:
        print("\n" + "=" * 92)
        print("§6  Vertical covariance through the gait cycle, and the innovation dose.")
        print("    f  = (P_pp-P_pd)/S   base POSITION share of a contact residual  [-]")
        print("    gv = (P_vp-P_vd)/S   base VELOCITY injected per metre of nu_z   [1/s]")
        print("    Covariance is the PRIOR (post-propagate), which is what the gain "
              "is built from.")
        n = sum(t["n"] for t in traces)
        g = lambda k: float(np.mean([t[k] for t in traces]))      # noqa: E731
        print(f"\n{'':>13} {'P_pp':>11} {'P_pd':>11} {'P_dd':>11} {'f':>10} "
              f"{'gv':>10} {'nu_z':>11}")
        for tag, lab in (("stance", "late stance"), ("swing0", "early swing"),
                         ("swing1", "late swing")):
            print(f"{lab:>13} {g('P_pp_' + tag):11.3e} {g('P_pd_' + tag):11.3e} "
                  f"{g('P_dd_' + tag):11.3e} {g('f_' + tag):10.3e} "
                  f"{g('gv_' + tag):10.3e} {g('nu_' + tag):+11.3e}")
        gs, g0 = g("gv_stance"), g("gv_swing0")
        print(f"\n    {n} liftoffs:  gv(late stance)/gv(early swing) = "
              f"{gs / g0 if g0 else float('nan'):.3f}")
        print(f"    velocity dose per liftoff {g('dose'):+.6f} m/s  (- = downward)")
        if not gs > 1.5 * g0:
            print("\n    *** The anchor is no tighter in late stance than in early "
                  "swing on the channel that carries the sink, so §1's asymmetry is "
                  "not the mechanism as stated. Read arm B's response instead. ***")

    print("\n" + "=" * 92)
    print(f"{'arm':>22} {'slope e_pz':>11} {'mean e_vz':>10} {'ratio':>7} "
          f"{'vel_rms':>9} {'height_rms':>11} {'tilt_deg':>9}")
    base = None
    for label, rs in rows.items():
        m = {k: float(np.mean([r[k] for r in rs]))
             for k in ("slope_e_pz", "e_vz", "ratio", "vel_rms", "height_rms", "tilt_deg")}
        if base is None:
            base = m
        print(f"{label:>22} {m['slope_e_pz']:+11.5f} {m['e_vz']:+10.5f} "
              f"{m['ratio']:7.3f} {m['vel_rms']:9.4f} {m['height_rms']:11.4f} "
              f"{m['tilt_deg']:9.3f}")

    print()
    sinks = {k: abs(float(np.mean([r["slope_e_pz"] for r in v]))) for k, v in rows.items()}
    a = next((v for k, v in sinks.items() if k.startswith("A")), None)
    proc = [v for k, v in sinks.items() if k[0] in "BC"]
    if a is None or not proc:
        print("(no process-side comparison arms ran — nothing to decide)")
        return
    best = min(proc)
    if best < 0.8 * a:
        print(f"MECHANISM CONFIRMED — the sink drops from {a:.5f} to {best:.5f} m/s "
              f"({a / best:.1f}x) under a process-side change alone. The process "
              "socket is a real lever on the sink; Phase 1 is worth doing.")
    else:
        print(f"SINK IS FLAT — best process arm {best:.5f} m/s against baseline "
              f"{a:.5f}. The process socket is not the lever, and a retrain on it "
              "would be wasted. Reopen the ranking in Z_BIAS_FACTS.md §5 first.")

    # Arm D is the control, and it is the arm that can falsify the STRONG form of
    # the claim -- "the measurement socket provably cannot reach this". `N` sits
    # inside the inverted factor, so it cannot change the base/anchor SPLIT; but
    # it does scale the correction, and the sink is an accumulated dose, so a
    # trained `N` can still shrink it. Report that rather than assume it away.
    d = next((v for k, v in sinks.items() if k.startswith("D")), None)
    if d is not None:
        print(f"\ncontrol: the trained MEASUREMENT-socket network is {d:.5f} m/s "
              f"({a / d:.1f}x better than baseline).")
        if d < 0.8 * a:
            print("  So the measurement socket is NOT powerless on this metric. It "
                  "cannot change how a residual is apportioned between base and "
                  "anchor -- that part of the argument holds -- but it scales every "
                  "correction, and the sink accumulates corrections. Any claim that "
                  "ContactNet was holding a knob that 'provably cannot reach' the "
                  "sink is too strong; the honest claim is that the process socket "
                  "is the more direct lever.")


if __name__ == "__main__":
    main()
