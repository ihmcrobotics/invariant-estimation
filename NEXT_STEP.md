# NEXT_STEP — ContactNet / z-drift handoff

Written 2026-08-10, second session. Branch `contactnet/z-drift`. **A training run is in
flight** — see "What is running" below before starting anything on the GPU.

---

## What is running

`scripts/bexp_train_and_eval.sh` (launched with `F=1e-3`), logging to
`results/zdrift_bexp/train.log`. It is a chain, deliberately:

1. `run_contactnet.py --diag-param bounded_exp --peak-lr 3e-5 --contact-meas-var 1e-3`,
   6000 steps, ~2.5 h → `results/zdrift_bexp/L256_A_cmv1e-3_bexp`
2. **then** `record_contactnet_demo.py` on the resulting checkpoint →
   `results/zdrift_bexp/closed_loop/cmv_1e-3_bexp.json` (~3 min)
3. **then** the span diagnostic (`plot_contact_phase.py`, CPU)

Step 2 is the verdict, not a follow-up: N1 says a learned Σ_C may only be judged closed
loop. It is chained so it cannot be skipped.

Everything else on the GPU must wait for the `flock` in `scripts/gpu_lock.sh` (N6).

## How to read the result when it lands

    uv run python scripts/cl_floor_summary.py --dir results/zdrift_bexp/closed_loop

The comparison is three-way at the same floor, and the other two arms are already
measured (results.md §9): **analytic −0.117 m**, **softplus +2.488 m**. So:

| bounded_exp total Δe_z | reading |
|---|---|
| beats +2.488 and lands within ~2× of −0.117 | amplitude was the cause; the parameterisation is the fix |
| beats +2.488 but stays ≫ analytic | amplitude is part of it; the residual is timing or structural |
| ≈ +2.488 | amplitude is **not** the cause — the head was not range-limited, it was pointed the wrong way. Kills the hypothesis, which is worth knowing |
| diverges | check `applied` and the non-finite count in `summary.json` first; the bounds held, so it would be the LR or the objective |

Also read the per-motion signs (N3) and the span diagnostic — the mechanism claim is
"the bounds opened the range", and that is measurable independently of the drift number.

## What changed this session

* **`diag_param="bounded_exp"`** — a sigmoid in log space between `diag_lo`/`diag_hi`
  (default 1e-5 → 1e2, config fields, recorded and read back). At the diverged `exp`
  run's p99 raw output (+12.5) it gives 99.99 where `exp` gives 2.7e5. `--peak-lr`,
  `--diag-lo`, `--diag-hi` added to `run_contactnet.py`.
* **N5 finally has code behind it.** `contactnet/checkpoint.py::config_for_checkpoint`
  restores the parameterisation from `summary.json`; every loader used a default
  `ContactNetConfig()` before, so the `exp` checkpoint could only ever have been
  evaluated as softplus. Passing a bare `"bounded_exp"` string now raises.
* **The floor sweep, redone closed-loop** (results.md §9). Twelve runs. The analytic
  response is *monotone* over four decades, so it identifies no operating point; the
  floor buys vertical error with horizontal (~10×) by de-weighting the contact
  measurement. 1e-3 was chosen to match the softplus baseline, not by minimisation.
* **The harness is bit-reproducible and the committed reference used CLEAN sensors** —
  clean reproduces +2.4884 m in all seven motions; `--imu-noise` is a 21% level shift.
  Never mix them inside a comparison.

## If the run fails

The first launch died at step 0: `DiagSpec.log_bounds` used `float(jnp.log(lo))`, which
is fine eagerly and raises `ConcretizationTypeError` under `jit`. Fixed (`math.log`) and
`test_forward_traces_under_jit` now reproduces it. The lesson generalises — **the unit
tests were all eager**, and every real call site is inside `jit`; if you add anything to
the forward path, trace it under `jit` in a test.

## Still open, in the order they are worth doing

1. **Why replay and closed loop disagree for a learned Σ_C.** Never established. The
   code path is faithful to 1e-15 (`online_offline_oracle.py`), so it is a property of
   the metric, not a bug. Answering it would tell us which past measurements can be
   trusted; nothing else on this list has that leverage.
2. **Timing, if `bounded_exp` does not close the gap.** The ±100 ms smear is a
   feature/horizon problem; no parameterisation fixes it.
3. **The structural option:** the contact zero-velocity constraint, which adds rows to
   `H` instead of reweighting existing ones (`contact-zero-velocity.pdf`). 98.6% of the
   sink flows through the contact update's write into `v̂`/`R̂`, and `H` has no velocity
   columns today.
4. The phase-residual regression (~10 min, CPU) — whether there is real contact
   information under the stride clock at all.

**Reading order for a new agent:** `CLAUDE.md` §7 (status + N1–N6), `PORT_NOTES.md`
§"ContactNet: three dead knobs, one invalid metric" and §"bounded `exp`, and the N5
read-back that was never implemented", `results.md` §8–9, and the three PDFs in
`~/Documents/filter-debugging/`.
