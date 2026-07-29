# TODO — deferred from the process-socket branch

Written 2026-07-29 alongside `contactnet/process-socket`, which landed **only** the
socket move: ContactNet's `Σ_C` now drives `InEKFInputs.contact_chol` (process
noise, `Q_d`) instead of `contact_meas_chol` (measurement noise, `N`).

`branch_out.md` asked for more than that. Everything below was either implemented
and deliberately reverted, or specified and not started. It is here rather than in
the branch because each item is a separate decision, and because two of them rest
on claims the measurements contradicted.

Ordered by what blocks the next retrain.

---

## 1. Initialization on the process socket — BLOCKS THE NEXT RUN

`network.init` zeroes the output head, so **iteration 0 emits a constant `σ₀·I`
for every input at every gait phase.** On the measurement socket that was
harmless: `Σ_C = σ₀²I` = 1e-8 m² sat three orders below `J Σ_q Jᵀ` = 1.26e-5 m².
On the process socket a constant at the stance value **is**
`ContactNetConfig.freeze_contact_chol`, measured (`experiments/measure_tstar.py`)
at **10.2x worse in body-frame velocity than not using contacts at all**, and it
is the configuration run 1 escaped only by driving `Σ_C → ∞`.

Training from today's default `sigma_0 = 1e-4` will reproduce run 1. The
docstring on `ContactNetConfig.sigma_0` says so; nothing enforces it.

Three options, in the order I'd try them:

1. **Supervised warm-start.** Regress the network onto the recorded heuristic
   `inputs.contact_chol` for a few hundred steps, then switch to the real
   objective. The network then starts *at* the shipped filter including its
   swing/stance switching, and BPTT only has to improve on it. Two details that
   matter: the regression must be in **log space on `diag(L)`** (the heuristic
   spans five decades, so an MSE on the factor is ~1e8-weighted toward the swing
   value and would not constrain stance at all), and it must leave the
   off-diagonals free (the heuristic is isotropic and has no anisotropy to
   teach — training them to zero fights the one thing the network exists to
   produce). This was implemented and reverted; it is ~40 lines plus an
   `objective="heuristic"` branch.
2. **Conservative constant**: `sigma_0` at the *swing* end (1e-1 … 1e0). Every
   foot starts loose, the filter degrades to "no contact information" — the
   better end of the `measure_tstar` comparison — and the network must learn to
   tighten. Safe, slow, no code.
3. Constant at the stance value. **This is run 1. Do not.**

### The gradient problem underneath it

`network.forward` emits `softplus(o) + eps`, and `d softplus/dx = σ(x)`, which at
the tight end **equals the emitted value** to first order. So the network receives
almost no gradient exactly where it is predicting tight stance covariances: expect
slow learning in stance, fast in swing. Log `min`/`max` of the predicted `diag(L)`
per step; if the min never moves across a run, that is the cause, and the fix is a
**log-scale output head**, not more steps. (Also reverted; it needs `L_c` in the
loss's aux tuple, which is a 3-line change.)

## 2. `P0` is stale for the new conventions

`dataset.measure_p0` now burns in on the recorded heuristic rather than on the
frozen constant, because that is what a training segment runs under after the
move. `train_contactnet.py p0 --p0 <path>` writes a fresh one;
`artifacts/p0_process_dr.npz` is the one for `data/dr` and differs from
`artifacts/p0_dr.npz` by **1.06% relative**, concentrated in `diag(P)[6:9]`
(0.3363/0.3339/0.3350 → 0.3469/0.3443/0.3456) — the position/anchor block the
process socket drives.

**Do not reuse `p0_dr.npz` for a post-move run.** A stale `P0` seeds every segment
from the wrong prior and moves `train.Metrics.nis_over_dof` for reasons that have
nothing to do with the network.

## 3. `N^v` — the contact zero-velocity block

**Deferred deliberately, and this is the item to read before reviving it.**

`filter.contact_velocity_noise(J, Σ_q̇)` exists with zero call sites, and
`JointFilterOutput.sigma_q_dot` is computed and plumbed to the boundary
unconsumed. `branch_out.md` §3 wanted them wired as a zero-velocity measurement:
a world-static contact has zero world velocity, which would give the filter its
**only direct observation of base velocity** — nothing else in the InEKF has a
`ξ_v` block, which is precisely why a velocity bias can survive indefinitely.

Three reasons it is not in the branch:

**(a) Neither derivation has it.** Lucas's original derivation does not include
`H_v`, and neither does CoCo-InEKF, whose only correction is its Eq. (8) — the FK
position update with `N̄ = R̂(J_C Σ_q J_Cᵀ + Σ_C)R̂ᵀ`. Adding a velocity-level
measurement is a departure from both, not a port of either.

**(b) `H_v` is *not* exactly state-independent.** §3.1 argues that
`H_v = [0 I 0 …]` is exact because "the `ξ_R` coupling cancels identically — the
quantity being observed is zero", and instructs that the block must not ship if a
two-random-state comparison disagrees. It disagrees. Under the port's left
perturbation `X̂ = exp(ξ)X` (I5), with `u = ω̄ × h_i + J_{C_i} q̇̂` and
`r = v̂ + R̂u`:

    r = (v + Ru) + ξ_v + ξ_R × (v + Ru) = r₀ + ξ_v − [r₀]_× ξ_R

so the exact observation matrix is

    H^v = [ −[r₀]_×  |  I  |  0  |  0 ⋯ ]

with `r₀` the residual *at the linearisation point*. Verified by autodiff through
the port's own `exp_SEn3` to machine zero (0.0 elementwise at two unrelated random
states, and `H_v` exact to 4.4e-16 at a state constructed with `v = −Ru`). The
coupling cancels at **truth**, not at an arbitrary estimate — and the residual is
exactly what is nonzero when the update has work to do.

The dropped term is `[r₀]_× ξ_R`, second order in (residual × error), which is the
same order of approximation every EKF linearisation already carries. So shipping
the constant is defensible; it is just not the *exact* `H` that the contact
position block enjoys. Making it exact costs one line (`Hv − skew(r₀)` blockwise,
fixed shape, branch-free, I7-safe). **That is the open decision.**

**(c) The measurement is false during swing, and the fix reintroduces a
contact-condition input on the measurement side.** `filter.py`'s DECISION note
argues against masking because "the encoders still locate the foot relative to the
base perfectly well" — true for *position*, and plainly false for zero-velocity on
a foot in flight. The least-bad answer found was to drive `R_v` from the same
`Σ_C` the process side already carries: `Σ_C` is a position random-walk density
[m²/s], so `Σ_C/Δt` is the velocity covariance the anchor is already permitted,
which is exactly "may this contact be moving?". One learned quantity, two
consumers, no new socket. Measured de-weighting at the heuristic's swing value was
>100x on the fixture.

If it is revived: it **stacks, it does not fold**. `J_C Σ_q̇ J_Cᵀ` must never be
added into `N^p` — they are noises on two different measurements. A separate
sequential `linear_update` is exactly equivalent to stacked rows here (both blocks
are exactly linear in `ξ` with block-diagonal noise), and gives separable
diagnostics plus a build-time off switch. The noise also wants a gyro term
`[h_i]_× (gyro_var/Δt) [h_i]_×ᵀ` from `∂(ω × h)/∂ω = −[h]_×`, which **omits the
gyro bias covariance** — the joint KF owns it (I1) and does not plumb it here.

### And the `Σ_q̇`-into-`N` question specifically

There is no dimensionally consistent way to add `Σ_q̇` to the *position* block's
`N`: `J Σ_q̇ Jᵀ` is (m/s)² against a m² block. The nearest defensible thing is
`J Σ_q̇ Jᵀ · Δt²` — position uncertainty accumulated over one tick of velocity
uncertainty, i.e. encoder/IMU sampling skew — but at `Δt = 1e-3` that is
~1e-8·JJᵀ against the 5e-5·JJᵀ encoder term, so it is invisible in sim and only
meaningful on hardware with real sync jitter. Not implemented; `Σ_q̇`'s only
correct destination is a velocity-level measurement, i.e. item 3 above.

## 4. Guard rails that were built and reverted

Small, independent, each a few lines:

* **`ContactNetConfig.validate` warning** when `sigma_0` is stance-like *and* the
  objective is not the warm-start — i.e. exactly the run-1 configuration. Reverted
  because it fired in 25 existing tests that construct the historical config for
  unrelated reasons; if revived, suppress it in those fixtures rather than
  changing them.
* **`run_estimator.py` warning** when a `run1`..`run4` checkpoint is attached.
  Those were trained against the measurement socket, where ~1e-4 is a sensible FK
  measurement std; the same output in `contact_chol` is the *stance* value, so
  every anchor — swing feet included — is asserted world-static. In the closed
  loop that is a fall, and it is currently silent. Score old checkpoints with
  `experiments/replay_eval.py --socket meas` instead.
* **`aux.contact_innovation` and `aux.P_diag` in `sim/collect.py`.** Both are
  already computed by the filter and thrown away. `contact_innovation` makes the
  per-step vertical dose directly observable instead of inferred; `P_diag` is 15
  floats a tick against the full `P`'s 111 MB per rollout. Note `P_diag` would be
  the **posterior** — the gain is built from the prior, and `Σ_C` enters through
  `Q_d`, so the update shrinks most of it straight back.
  `experiments/replay_eval.run_arm(want_traces=True)` reconstructs the prior by
  replaying `propagate`, which is what the §6 analysis actually used.

---

## Not in scope, recorded so it is not rediscovered

* **The learned moving mean**: `Bw_Ci ~ N(v̂_c, Σ_C)` rather than `N(0, Σ_C)`.
  Both CoCo Eq. (5) and this branch keep the contact noise **zero-mean**, so
  neither can represent a genuinely non-zero-mean contact error. Moving the socket
  converts an *accumulating* drift into a *bounded offset* if the apportionment
  argument holds; it is not a bias correction. The socket move is the
  **prerequisite** for this work, not a substitute for it.
* **`reseedContact` + `TouchdownReseedLatch`.** Absent from the port, and the
  obvious next lever now that the stance→swing transition is implicated (see
  PORT_NOTES, "the real mechanism is late stance"). Kept out of this branch as a
  confound; `tests/inEKF/test_invariant_ekf.py::test_reseed_is_not_implemented`
  stays green. Note the Schmitt trigger is **not** missing —
  `sim/sensors.py::ContactTrust` is a live port of
  `FootSwitchContactProbabilityProvider` and drove every measurement on record.
* **The accelerometer-bias state.** `sim/sensors.py` injects no accel bias and the
  measured specific-force error is *positive* on the control set that sinks worst,
  so it cannot be the sim cause. It remains a real **hardware** exposure — no
  accel-bias state exists anywhere in the port, the joint KF carries `b_ω` only —
  and it needs its own branch and its own argument.
