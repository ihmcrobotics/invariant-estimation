#!/usr/bin/env bash
# The bounded-`exp` run: train, then IMMEDIATELY evaluate closed-loop, then measure the
# span. Chained in one shell so the eval cannot be forgotten -- invariant N1 says the
# closed-loop number is the only valid verdict on a learned Sigma_C, and it costs ~3 min
# against the run's ~2.5 h.
#
#   F=1e-4 nohup bash scripts/bexp_train_and_eval.sh > results/zdrift_bexp/train.log 2>&1 &
#
# One delta from the softplus baseline beyond the parameterisation and its paired LR:
# deliberately none. `--no-remat`, L=256, l2_velocity, 6000 steps and the n8fix pool are
# all the baseline's, so the comparison attributes to the parameterisation.
set -euo pipefail
cd "$(dirname "$0")/.."

F=${F:-1e-4}                    # contact_meas_var, from the closed-loop floor sweep
LR=${LR:-3e-5}
STEPS=${STEPS:-6000}
OUT=${OUT:-results/zdrift_bexp/L256_A_cmv${F}_bexp}
CL=results/zdrift_bexp/closed_loop
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"

mkdir -p "$CL" /tmp/cl_bexp
echo "== bounded_exp: cmv=$F peak_lr=$LR steps=$STEPS -> $OUT"

$GPU scripts/run_contactnet.py \
    --pool n8fix --contacts-per-foot 4 \
    --objective l2_velocity --L 256 --no-remat \
    --diag-param bounded_exp --peak-lr "$LR" \
    --contact-meas-var "$F" \
    --steps "$STEPS" --warmup-steps 100 --time-budget-s 86400 \
    --out-dir "$OUT"

echo "== training done; closed-loop eval (the acceptance test)"
# Clean sensors and the same floor as the sweep, so this lands on the same curve as the
# analytic and softplus arms already measured at this floor.
$GPU scripts/record_contactnet_demo.py \
    --contacts-per-foot 4 --contact-meas-var "$F" \
    --contactnet "$OUT/params.npz" \
    --out /tmp/cl_bexp/bexp.mp4 --metrics "$CL/cmv_${F}_bexp.json"

echo "== span diagnostic (CPU, so it cannot collide with anything on the GPU)"
JAX_PLATFORMS=cpu uv run python scripts/plot_contact_phase.py \
    --ckpt "$OUT" --out "$OUT/contact_phase.png" || echo "(span diagnostic failed, non-fatal)"

echo "== all done"
uv run python scripts/cl_floor_summary.py --dir "$CL" || true
