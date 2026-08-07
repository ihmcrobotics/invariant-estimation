#!/usr/bin/env bash
# experiments/rfloor_limit.sh — where does the R floor turn around, and does it hold up?
#
# The shape sweep found drift improving monotonically to 1e-2, the largest value tested
# (-0.00061, 3.1x better than the 1e-4 I was about to recommend). The ledger says the
# contact update is NOT being switched off (vel_CV_CONT_TRANS moves only 8% across three
# decades), so the gain looks real rather than an artefact of ignoring contact. Two checks
# before it becomes a recommendation:
#   1. Push R to 1e-1 and 1.0. At some point the update MUST stop pulling and drift must
#      get worse; if it never does, the 30 s drift metric is not measuring what we think.
#   2. Re-run the best value over 60 s. A softer contact update could look good at 30 s
#      and diverge later -- the failure mode a short window cannot see.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/rlimit "$TMPDIR"

for v in 1.0e-1 1.0e0; do
  for seed in 0 1; do
    out="results/rlimit/analytic_dw0.08_rf${v}_s${seed}.npz"
    [[ -f "$out" ]] && continue
    echo "### $(date +%H:%M:%S) limit rf=$v seed=$seed"
    uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
      --imu-noise --noise-seed "$seed" --contact-source trust --dwell 0.08 \
      --contact-meas-var "$v" --out "$out" >> results/rlimit/rlimit.log 2>&1 \
      || echo "!! rf=$v s=$seed FAILED"
  done
done

# 60 s horizon at the two candidate values plus the shipped one, same seeds.
for v in 0.0 1.0e-4 1.0e-2; do
  for seed in 0 1; do
    out="results/rlimit/long60_rf${v}_s${seed}.npz"
    [[ -f "$out" ]] && continue
    echo "### $(date +%H:%M:%S) long60 rf=$v seed=$seed"
    uv run python experiments/z_budget.py --ticks 3000 --vx 0.4 --contacts-per-foot 4 \
      --imu-noise --noise-seed "$seed" --contact-source trust --dwell 0.08 \
      --contact-meas-var "$v" --out "$out" >> results/rlimit/rlimit.log 2>&1 \
      || echo "!! long60 rf=$v s=$seed FAILED"
  done
done
echo "### $(date +%H:%M:%S) rfloor limit sweep done"
