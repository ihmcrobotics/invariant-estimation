#!/usr/bin/env bash
# experiments/rfloor_transfer.sh — is the R-floor gain a fix or a cancellation?
#
# Drift is monotone in contact_meas_var and CROSSES ZERO between 1e-1 and 1.0
# (-0.00007 -> +0.00010). So minimising |drift| over R is fitting a cancellation
# between the softened contact pull and the sink -- and a cancellation only transfers
# if it is condition-independent. 1e-2 is the largest value that is still clearly on
# the negative side (-0.00061), i.e. not tuned to the crossing, and it holds at 60 s.
#
# The test: does 1e-2 keep its ~3x advantage over 1e-4 at a different speed and on
# both terrains? If the ratio survives, the lever is real and 1e-2 is shippable. If it
# collapses or inverts, the gain was flat-ground/one-speed cancellation and the honest
# recommendation stays at flight's 1e-4.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/rtrans "$TMPDIR"

for v in 1.0e-4 1.0e-2; do
  for seed in 0 1; do
    out="results/rtrans/vx0.6_rf${v}_s${seed}.npz"
    [[ -f "$out" ]] || uv run python experiments/z_budget.py --ticks 1500 --vx 0.6 \
      --contacts-per-foot 4 --imu-noise --noise-seed "$seed" --contact-source trust \
      --dwell 0.08 --contact-meas-var "$v" --out "$out" >> results/rtrans/rtrans.log 2>&1 \
      || echo "!! vx0.6 rf=$v s=$seed FAILED"
  done
  for terr in hard_stepping waves; do
    for seed in 0 1; do
      out="results/rtrans/${terr}_rf${v}_s${seed}.npz"
      [[ -f "$out" ]] || timeout 1500 uv run python experiments/z_budget.py --ticks 700 \
        --vx 0.4 --contacts-per-foot 4 --imu-noise --noise-seed "$seed" \
        --contact-source trust --dwell 0.08 --contact-meas-var "$v" --terrain "$terr" \
        --terrain-seed "$seed" --out "$out" >> results/rtrans/rtrans.log 2>&1 \
        || echo "!! $terr rf=$v s=$seed FAILED (may have fallen)"
    done
  done
done
echo "### $(date +%H:%M:%S) rfloor transfer sweep done"
