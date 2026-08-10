# NEXT_STEP — ContactNet / z-drift handoff

Written 2026-08-10 at the end of a long session. Branch `contactnet/z-drift`, clean
tree, nothing running. This is the plan as it stood; read it before starting work.

---

**Don't start with the bounded-`exp` run.** It would be trained at
`contact_meas_var=1e-3`, and that floor was chosen by the sweep we then invalidated —
the replay-derived one. Retraining on it inherits the error. The 2.5 h run would rest
on a number we no longer trust.

**Order suggested:**

1. **Closed-loop floor sweep (~20 min).** Six floors × ~3 min, using the harness that
   already validated itself against the analytic baseline. This re-establishes the
   operating point everything else is measured against. Cheapest item on the list and
   it unblocks the rest.
2. **Then decide amplitude vs timing vs structural**, which is really Lucas's call
   about what he thinks the cause is:
   - *amplitude* → bounded `exp` + lower LR (~2.5 h), closes the 8×-too-tight swing gap
   - *timing* → the ±100 ms smear; a feature/horizon problem, no parameterisation
     fixes it
   - *structural* → the zero-velocity constraint, `contact-zero-velocity.pdf`
3. The phase-residual regression (~10 min, CPU) if you want to know whether there's
   real contact information under the clock before investing in any of the three.

**Reading order:** `CLAUDE.md` §7 (status + the six N-invariants — N1 is the one that
matters most: never evaluate a learned Σ_C by replay), `PORT_NOTES.md` §"ContactNet:
three dead knobs, one invalid metric", `results.md` §8 for outcomes, and the three
PDFs in `~/Documents/filter-debugging/`.

**Two corrections to carry so they don't get re-derived:** `remat` was a regression
from `a492d69`, not an original defect — `take-two` applies it correctly. And softplus
fails at the *top* of its range, not at init, where it is exactly `exp`.

**One thing genuinely unresolved rather than done:** we never established *why* replay
and closed loop disagree for a learned Σ_C. We proved the code path is faithful to
1e-15 and stopped there. That's a real open question, and if it can be answered
cheaply it is worth more than any of the three options above — it would tell us which
of our measurements can be trusted going forward.

---

## Concrete details the above assumes

**The bounded-`exp` change**, when it is run. Not a hard clip (that kills gradients at
the bounds) but a sigmoid in log-space, in `network.py`'s `_diag_fwd`:

```
L_ii = exp( lo + (hi − lo)·sigmoid(r) ),   lo = ln(1e-5), hi = ln(1e2)
```

Spans the physically meaningful range with headroom either side of the analytic
1e-4 → 1e1, keeps near-uniform relative sensitivity through the interior, saturates
smoothly instead of overflowing, and cannot reach the 2.7e5 the unbounded run hit.
Pair it with `peak_lr` ≈ 3e-5: `dlogL/dr` is 1 everywhere under `exp` versus 0.1–0.8
under softplus, so the softplus-tuned 1e-4 is 1.2–10× more aggressive in Σ_C-space.
Add it as a third `diag_param` value; do not repurpose `"exp"`, since a checkpoint
already exists under it.

**Why the closed-loop sweep is cheap.** `scripts/record_contactnet_demo.py` drives all
six validation motions in one 25 s clip and reports per-motion vertical drift
(`--metrics`). ~3 min per configuration, *less* than the ~8 min replay evaluation
it replaces. Use `--contact-meas-var` matching whatever the checkpoint was trained at,
and `--imu-noise` if matching the pool's collection conditions matters.

**Commands.** Everything that touches the GPU goes through `bash scripts/gpu_lock.sh`
— JAX preallocates ~75% of the device, so a second process OOMs *and kills the first*.
CPU-only analyses (`plot_contact_phase.py`, the span diagnostic) take
`JAX_PLATFORMS=cpu` and can run alongside anything.

## State

- Best checkpoint: `results/zdrift/L256_A_l2vel_cmv1e-3` (softplus, L=256, cmv=1e-3).
  **Not deployable** — +2.49 m closed-loop vs the analytic heuristic's −0.117 m.
- `results/zdrift_exp/L256_A_cmv1e-3_exp` — the unbounded `exp` run. Span opened
  (12.8 → 21.9 raw units, swing Σ_C 685× → 8× too tight) but diverged: RMSE 1.374 vs
  0.0607, loss rising, 30 non-finite steps.
- `diag_param` defaults to `softplus` and is recorded in `summary.json`. It **must**
  be read back when loading a checkpoint (N5).
- Evidence JSONs are committed under `results/`; checkpoints and plots are not.
