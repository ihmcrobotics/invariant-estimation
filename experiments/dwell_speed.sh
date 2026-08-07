#!/usr/bin/env bash
# experiments/dwell_speed.sh — is the dwell recommendation gait-period specific?
#
# At 200 Hz with stance ~48 ticks, an 80 ms dwell is ~1/3 of stance. If the win is
# really "wait until the foot has stopped rolling", the optimal dwell should scale
# with the gait period — so a different walking speed is the cleanest check that
# 80 ms is not tuned to vx=0.4 specifically. The turning condition already gave
# partial evidence (3.6x at 80 ms), but it keeps the same forward speed.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/speed "$TMPDIR"

while pgrep -f "dwell_terrain.sh|terrain_arms.sh|dwell_120_breadth.sh|rfloor_confirm.sh" \
      >/dev/null 2>&1; do sleep 30; done

for vx in 0.2 0.6; do
  for dw in 0.04 0.08; do
    for seed in 0 1; do
      out="results/speed/vx${vx}_dw${dw}_s${seed}.npz"
      [[ -f "$out" ]] && continue
      echo "### $(date +%H:%M:%S) vx=$vx dwell=$dw seed=$seed"
      uv run python experiments/z_budget.py --ticks 1500 --vx "$vx" \
        --contacts-per-foot 4 --imu-noise --noise-seed "$seed" --contact-source trust \
        --dwell "$dw" --out "$out" >> results/speed/speed.log 2>&1 \
        || echo "!! vx=$vx dw=$dw s=$seed FAILED"
    done
  done
done
echo "### $(date +%H:%M:%S) speed sweep done"
