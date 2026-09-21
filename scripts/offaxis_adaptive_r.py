"""Off-axis-residual-driven adaptive R for the distributed-IMU (`stacked`) channel.

The idea
--------
An IMU pair spanning a joint measures ``z_e = omega_child - {}^{c}R_{p} omega_parent``.
For a rigid link that quantity must lie in the column space of the pair's relative
angular Jacobian ``J_e(q)`` -- a 1-D line for a 1-DoF pair, a 2-D plane for a 2-DoF
pair.  Everything orthogonal to that span carries **no joint information**: it is
link flex, mount shift, vibration and gyro bias.  So the off-axis residual

    r_off_e(t) = (I - P_e(q)) z_e(t),   P_e = projector onto colspace(J_e)

is an online, mocap-free measurement of how badly the rigid-body model is being
violated *right now*, on that pair.  Inflate that pair's R by a function of it.

Slow vs fast (task 1)
---------------------
``r_off`` is not all model violation.  Gyro bias and a constant mount
misalignment both land in it and are *slow*; impact flex and vibration are
*fast*.  Only the fast part should drive R -- a constant misalignment does not
make this tick's on-axis measurement any worse than the last one's, and folding
it in would just add a constant offset to every g.  The split used here is a
**causal trailing moving median** of the off-axis vector, subtracted:

    slow_e(t) = median(r_off_e(t-W+1 .. t))   (per component)
    fast_e(t) = r_off_e(t) - slow_e(t)

Median rather than a mean or a linear high-pass because the fast part is
impulsive (heel strike), and an impulse drags a moving mean -- and therefore the
"slow" estimate -- toward itself, which is exactly the leakage the split exists
to prevent.  Trailing rather than centred so nothing looks into the future.

The inflation
-------------
    g_e(t) = 1 + (|fast_e(t)| / s_e)^2,        s_e = alpha * median_fit(|fast_e|)

which is 1 when the robot is still (fast -> 0), so a standing window is
untouched, and grows quadratically -- matching the fact that a gyro-difference
error enters the innovation linearly and NIS quadratically.  ``alpha`` is a
SINGLE global scalar; the per-pair normalisation ``median_fit(|fast_e|)`` is
seven numbers.  All eight come from the fit window only.

R is inflated additively and isotropically on pair ``e``'s own 3x3 diagonal
block of the stacked ``R``::

    R[3e:3e+3, 3e:3e+3] += (g_e(t) - 1) * sigma0_e^2 * I3

with ``sigma0_e^2 = trace(Sigma_child + {}^{c}R_{p} Sigma_parent {}^{c}R_{p}^T)/3``
-- which is rotation-invariant, hence a static per-pair constant.  Additive keeps
``R`` PSD by construction; scaling a diagonal block of a correlated matrix does
not.

Library change
--------------
One additive, defaulted field: ``jointKF.filter.SensorInputs.pair_r_extra``
(``(n_pairs,)``, per tick), forwarded to ``measure.build_stacked(...,
pair_r_extra=)`` which adds ``diag`` of it into the pair block.  ``None`` (the
default, and what ``prepare_session`` still builds) reproduces the old code path
bit-for-bit.

Usage
-----
    python scripts/offaxis_adaptive_r.py decompose <t0> <t1>
    python scripts/offaxis_adaptive_r.py fit  <fit_t0> <fit_t1>
    python scripts/offaxis_adaptive_r.py eval <fit_t0> <fit_t1> <test_t0> <test_t1>

Windows are in dt-seconds (tick index / 1000), not wall clock.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("ALEX_AB_LOG", "/home/bpark/.claude/jobs/e19f3ddd/tmp/log0717")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

import ab_contact_anchor_offset as ab
from invariant_estimation.eval.consistency import two_stage_consistency, format_report
from invariant_estimation.inEKF import ekf as base_ekf
from invariant_estimation.jointKF.state import default_params
from invariant_estimation.learning.channels import alex_channel_map_from_log
from invariant_estimation.learning.log_adapter import (ClockMapping, InitialState,
                                                       prepare_session, run_session)
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.model.mounts import imu_to_body
from invariant_estimation.replay.logsource import read_window

BIAS = np.load("/home/bpark/.claude/jobs/e19f3ddd/tmp/accel_bias_0717.npy")
MEDIAN_WINDOW = 500          # ticks (~0.43 s at the log's 863 us median step)
RANK_TOL = 1e-8


# ---------------------------------------------------------------------------
# session construction (shares ab_contact_anchor_offset's stack verbatim)
# ---------------------------------------------------------------------------

def make_session(window_s):
    cfg, jk, mj_model, build, session_model = ab.build_stack(None)
    channels = alex_channel_map_from_log(ab.LOG, session_model.joint_names, build.imu_names,
                                         ab.FEET, base_imu="pelvis_imu",
                                         trust_format=ab.TRUST_FORMAT)
    window = read_window(ab.LOG, channels.required_channels(),
                         start=window_s[0], end=window_s[1], stride=1)
    ekf = base_ekf.create(len(ab.FEET), dt=jk["dt"], gyro_var=cfg["inekf"]["gyro_var"],
                          accel_var=cfg["inekf"]["accel_var"],
                          contact_var=cfg["inekf"]["contact_var"])
    joint_params = default_params(dt=jk["dt"], sigma_accel=jk["sigma_accel"],
                                  cond_s_max=jk["cond_s_max"])
    initial = InitialState(rotation=np.eye(3), velocity=np.zeros(3), position=np.zeros(3),
                           covariance=np.eye(ekf.tangent_size) * 1e-3,
                           source="identity prior for an adaptive-R run")
    clock = ClockMapping(source_origin_s=float(window.time[0]), target_origin_ns=0,
                         clock_domain="aligned_robot_monotonic_ns")
    session = prepare_session(window, channels, clock, session_model, build, joint_params,
                              ekf, initial, imu_to_body=imu_to_body(mj_model),
                              world_frame="registered_zup", accel_bias_body=BIAS)
    # `window` is carried so callers can reconstruct the log's tick axis. run_comparison_arms.py
    # needs it to write Java's timestamp convention (tick_index * dt), which neither the session
    # nor LogWindow.time preserves.
    return dict(cfg=cfg, build=build, model=session_model, ekf=ekf,
                joint_params=joint_params, session=session, window=window)


def rollout(ctx, pair_r_extra=None):
    """One two-stage rollout, optionally with a per-tick per-pair R inflation."""
    build, session = ctx["build"], ctx["session"]
    if pair_r_extra is not None:
        import dataclasses
        session = dataclasses.replace(session, sensors=session.sensors._replace(
            pair_r_extra=jnp.asarray(np.asarray(pair_r_extra, float))))
    spec = NoiseSpec(tuple(build.imu_names), arm=7)
    _, out = run_session(spec.initial_theta(), spec, build, ctx["joint_params"],
                         ctx["ekf"], ctx["model"], session)
    return out


# ---------------------------------------------------------------------------
# off-axis residual
# ---------------------------------------------------------------------------

def offaxis(ctx):
    """Per-tick, per-pair (z_e, on-axis part, off-axis part) from the LOG's own q.

    Uses the logged encoder positions rather than the filter's posterior q: this
    is a pre-filter signal by construction, so nothing here can be contaminated
    by the estimate it is about to reweight.
    """
    build, model, session = ctx["build"], ctx["model"], ctx["session"]
    q = np.asarray(session.positions)
    gyros = np.asarray(session.sensors.gyros)
    mask = np.asarray(build.pair_velocity_mask, float)          # (P, n)

    @jax.jit
    def frames(qi):
        mi = model.model_inputs(qi[np.asarray(model.filtered_indices)], qi)
        return mi.J_rel, mi.R_rel

    J_all, R_all = jax.vmap(frames)(jnp.asarray(q))
    J = np.asarray(J_all) * mask[None, :, None, :]              # (T, P, 3, n)
    R_rel = np.asarray(R_all)
    T, P = J.shape[0], J.shape[1]

    parent = np.asarray(build.pair_parent)
    child = np.asarray(build.pair_child)
    z = gyros[:, child] - np.einsum("tepq,teq->tep", R_rel, gyros[:, parent])  # (T,P,3)

    on = np.zeros_like(z)
    rank = np.zeros((T, P), int)
    for e in range(P):
        # colspace(J) via SVD, per tick; the span is what the pair can observe.
        u, s, _ = np.linalg.svd(J[:, e], full_matrices=False)    # u:(T,3,3) s:(T,3)
        keep = (s > RANK_TOL * np.maximum(s[:, :1], 1e-30))
        rank[:, e] = keep.sum(axis=1)
        w = np.einsum("tij,tj->ti", u.transpose(0, 2, 1), z[:, e]) * keep
        on[:, e] = np.einsum("tij,tj->ti", u, w)
    off = z - on
    return z, on, off, rank, R_rel


def trailing_median(x, window):
    """Causal trailing median along axis 0 (ramps in over the first `window` ticks)."""
    T = x.shape[0]
    out = np.empty_like(x)
    flat = x.reshape(T, -1)
    o = out.reshape(T, -1)
    for t in range(T):
        o[t] = np.median(flat[max(0, t - window + 1):t + 1], axis=0)
    return out


def split(off, window=MEDIAN_WINDOW):
    slow = trailing_median(off, window)
    return slow, off - slow


def sigma0(ctx, R_rel):
    """Static per-pair isotropic baseline gyro-difference variance."""
    build = ctx["build"]
    S = np.asarray(build.gyro_sigma)
    parent, child = np.asarray(build.pair_parent), np.asarray(build.pair_child)
    # trace(R Sigma R^T) == trace(Sigma), so this does not depend on the tick.
    return np.array([(np.trace(S[child[e]]) + np.trace(S[parent[e]])) / 3.0
                     for e in range(build.n_pairs)])


def inflation(fast, scale):
    """g_e(t) - 1 = (|fast_e(t)| / s_e)^2, as the additive variance multiplier."""
    mag = np.linalg.norm(fast, axis=-1)                          # (T, P)
    return (mag / scale[None, :]) ** 2


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_decompose(t0, t1):
    ctx = make_session((t0, t1))
    z, on, off, rank, R_rel = offaxis(ctx)
    slow, fast = split(off)
    build = ctx["build"]
    print(f"window ({t0}, {t1})  ticks {int(t0*1000)}..{int(t1*1000)}  T={z.shape[0]}")
    print(f"{'pair':>4} {'chain':38} {'dof':>3} {'|z|':>8} {'|off|':>8} "
          f"{'off/z':>6} {'|slow|':>8} {'|fast|':>8} {'fast%':>6}")
    for e in range(build.n_pairs):
        chain = (f"{build.imu_names[int(build.pair_parent[e])]}->"
                 f"{build.imu_names[int(build.pair_child[e])]}")
        ez = (z[:, e] ** 2).sum()
        eo = (off[:, e] ** 2).sum()
        es = (slow[:, e] ** 2).sum()
        ef = (fast[:, e] ** 2).sum()
        print(f"{e:>4} {chain:38} {int(np.median(rank[:, e])):>3} "
              f"{np.linalg.norm(z[:, e],axis=-1).mean():8.4f} "
              f"{np.linalg.norm(off[:, e],axis=-1).mean():8.4f} "
              f"{eo/ez*100:5.1f}% "
              f"{np.linalg.norm(slow[:, e],axis=-1).mean():8.4f} "
              f"{np.linalg.norm(fast[:, e],axis=-1).mean():8.4f} "
              f"{ef/eo*100:5.1f}%")
    return ctx, off, slow, fast


def fit_scale(ctx, alpha):
    """s_e = alpha * median(|fast_e|) on this window."""
    _, _, off, _, _ = offaxis(ctx)
    _, fast = split(off)
    med = np.median(np.linalg.norm(fast, axis=-1), axis=0)
    return alpha * med, fast


def stacked_anis(out):
    for r in two_stage_consistency(out.joint_diagnostics, out.base, n_contacts=2, n_joints=9):
        if r.channel == "stacked":
            return r.anis
    return float("nan")


def fit_constants(ctx, alpha):
    """The eight fitted numbers, all from THIS window: (s_e, sigma0_e, const_e).

    `const_e` is the matched CONTROL arm: the fit-window time-average of the
    adaptive inflation, frozen. Same per-pair level, no time variation -- so a
    difference between the two arms is attributable to the time variation alone,
    which is the only thing the off-axis hypothesis actually claims.
    """
    _, _, off, _, R_rel = offaxis(ctx)
    _, fast = split(off)
    med = np.median(np.linalg.norm(fast, axis=-1), axis=0)
    s0 = sigma0(ctx, R_rel)
    scale = alpha * med
    extra = inflation(fast, scale) * s0[None, :]
    return scale, s0, extra.mean(axis=0)


def cmd_fit(t0, t1, alphas, mode="adaptive"):
    ctx = make_session((t0, t1))
    _, _, off, _, R_rel = offaxis(ctx)
    _, fast = split(off)
    med = np.median(np.linalg.norm(fast, axis=-1), axis=0)
    s0 = sigma0(ctx, R_rel)
    print("per-pair median |fast| (rad/s):", np.array2string(med, precision=5))
    print("per-pair sigma0^2            :", np.array2string(s0, precision=6))
    print(f"\nFIT window ({t0}, {t1}); mode={mode}; target stacked ANIS = 27")
    T = fast.shape[0]
    results = []
    for alpha in alphas:
        adaptive = inflation(fast, alpha * med) * s0[None, :]
        extra = adaptive if mode == "adaptive" else np.broadcast_to(
            adaptive.mean(axis=0)[None, :], adaptive.shape).copy()
        out = rollout(ctx, extra)
        a = stacked_anis(out)
        results.append((alpha, a))
        print(f"  alpha={alpha:8.4f}  stacked ANIS={a:12.3f}  ratio={a/27:8.3f}")
    return med, s0, results


def nis_shape(out):
    """Distribution of the stacked channel's NIS, not just its mean.

    The control arm (a CONSTANT R inflation) can always be tuned to hit the mean.
    What an adaptive R is supposed to buy is a NIS whose *distribution* matches
    chi2(27): the mean is one moment, and the tail is the one that a constant R
    cannot fix, because a constant R is too loose during quiet swing and still
    too tight at heel strike.
    """
    nis = np.asarray(out.joint_diagnostics.stacked_nis, float)
    ap = np.asarray(out.joint_diagnostics.stacked_applied, float)
    v = nis[(ap != 0.0) & np.isfinite(nis)]
    return dict(mean=v.mean(), median=np.median(v), p90=np.percentile(v, 90),
                p99=np.percentile(v, 99), frac_gt_3dof=float((v > 81).mean()), n=v.size)


def cmd_eval(fit_w, test_w, alpha, beta=1.0):
    """Fit on `fit_w`, report on `test_w`. NOTHING here is tuned on `test_w`."""
    fit_ctx = make_session(fit_w)
    scale, s0, const = fit_constants(fit_ctx, alpha)
    const = const * beta
    print(f"FIT window ({fit_w[0]}, {fit_w[1]}): alpha={alpha}, beta={beta}")
    print(f"  s_e            = {np.array2string(scale, precision=5)}")
    print(f"  const g_e - 1  = {np.array2string(const / s0, precision=1)}")

    ctx = make_session(test_w)
    _, _, off_t, _, _ = offaxis(ctx)
    _, fast_t = split(off_t)
    extra = inflation(fast_t, scale) * s0[None, :]
    gm = extra / s0[None, :]
    print(f"HELD-OUT ({test_w[0]}, {test_w[1]}): g-1 mean={gm.mean():.1f} "
          f"median={np.median(gm):.2f} p99={np.percentile(gm,99):.1f} max={gm.max():.1f}")

    arms = (("BEFORE   baseline frozen R", None),
            ("CONTROL  constant inflated R",
             np.broadcast_to(const[None, :], extra.shape).copy()),
            ("AFTER    adaptive off-axis R", extra))
    for label, e in arms:
        out = rollout(ctx, e)
        v = np.asarray(out.base.state.v)
        print(f"\n--- {label} ---")
        print(format_report(two_stage_consistency(out.joint_diagnostics, out.base,
                                                  n_contacts=2, n_joints=9)))
        s = nis_shape(out)
        print(f"stacked NIS shape: median={s['median']:.2f} p90={s['p90']:.2f} "
              f"p99={s['p99']:.2f} frac(NIS>3*dof)={s['frac_gt_3dof']:.3f}")
        print(f"base |v| mean={np.linalg.norm(v,axis=-1).mean():.4f} m/s  "
              f"max={np.linalg.norm(v,axis=-1).max():.4f}  "
              f"v_z rms={np.sqrt((v[:,2]**2).mean()):.4f}")


def delay_extra(extra, k):
    """`extra` shifted k ticks later: R(t) is driven by the residual at t-k.

    The head holds the first available value rather than zero, so the filter never starts from a
    fabricated "no disturbance" tick -- the same edge policy the contact-timing shift uses, and for
    the same reason.
    """
    if k <= 0:
        return extra
    out = np.empty_like(extra)
    out[k:] = extra[:-k]
    out[:k] = extra[0]
    return out


def causal_extra(extra, window):
    """R(t) from a trailing window of PAST residual only -- the current tick is never read.

    Uses the trailing max rather than the mean: the disturbance is impulsive, and a mean over a
    window long enough to exclude the present tick would smear a heel strike into nothing. The max
    holds the recent worst case, which is the conservative reading and also what an online
    implementation would do.
    """
    T = extra.shape[0]
    out = np.empty_like(extra)
    for t in range(T):
        lo = max(0, t - window)
        hi = max(lo + 1, t)          # strictly excludes t
        out[t] = extra[lo:hi].max(axis=0)
    return out


def cmd_circularity(fit_w, test_w, alpha, beta=1.0):
    """Does the result survive when R can no longer see the innovation it weights?

    The objection this answers: g(t) is built from the current tick's own measurement, so R is
    correlated with the innovation it multiplies and systematically down-weights exactly the ticks
    whose innovations are large. The constant control rules out "any inflation would do"; it does
    not rule out this. If a strictly causal variant -- one that never reads tick t -- keeps the
    cross-regime behaviour, the objection is answered and the method is also online-implementable.
    """
    fit_ctx = make_session(fit_w)
    scale, s0, const = fit_constants(fit_ctx, alpha)
    const = const * beta
    print(f"FIT ({fit_w[0]}, {fit_w[1]}): alpha={alpha} beta={beta}  "
          f"-- every variant below reuses THESE constants, nothing is refitted\n")

    ctx = make_session(test_w)
    _, _, off_t, _, _ = offaxis(ctx)
    _, fast_t = split(off_t)
    extra = inflation(fast_t, scale) * s0[None, :]

    variants = [("baseline frozen R", None),
                ("constant control", np.broadcast_to(const[None, :], extra.shape).copy()),
                ("adaptive (same tick)", extra)]
    for k in (1, 5, 20, 50, 200):
        variants.append((f"adaptive delayed {k} ticks", delay_extra(extra, k)))
    for w in (20, 100):
        variants.append((f"adaptive causal (past {w} only)", causal_extra(extra, w)))

    print(f"{'variant':34s} {'stacked/27':>11s} {'contact/6':>10s} {'|v| mean':>9s} {'std v_y':>8s}")
    for label, e in variants:
        out = rollout(ctx, e)
        reports = {r.channel: (float(r.anis), float(r.dof))
                   for r in two_stage_consistency(out.joint_diagnostics, out.base,
                                                  n_contacts=2, n_joints=9)}
        R = np.asarray(out.base.state.R)
        v = np.asarray(out.base.state.v)
        vb = np.einsum('tji,tj->ti', R, v)
        print(f"{label:34s} {reports['stacked'][0]/27:>11.3f} {reports['contact'][0]/6:>10.3f} "
              f"{np.linalg.norm(v, axis=-1).mean():>9.4f} {vb[:, 1].std():>8.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "decompose":
        cmd_decompose(float(sys.argv[2]), float(sys.argv[3]))
    elif cmd == "fit":
        mode = "constant" if "--constant" in sys.argv else "adaptive"
        alphas = [float(a) for a in sys.argv[4:] if not a.startswith("--")] or [1.0]
        cmd_fit(float(sys.argv[2]), float(sys.argv[3]), alphas, mode)
    elif cmd == "circ":
        cmd_circularity((float(sys.argv[2]), float(sys.argv[3])),
                        (float(sys.argv[4]), float(sys.argv[5])), float(sys.argv[6]),
                        float(sys.argv[7]) if len(sys.argv) > 7 else 1.0)
    elif cmd == "eval":
        cmd_eval((float(sys.argv[2]), float(sys.argv[3])),
                 (float(sys.argv[4]), float(sys.argv[5])), float(sys.argv[6]),
                 float(sys.argv[7]) if len(sys.argv) > 7 else 1.0)
    else:
        raise SystemExit(__doc__)


# ---------------------------------------------------------------------------
# circularity test (task 2)
# ---------------------------------------------------------------------------
