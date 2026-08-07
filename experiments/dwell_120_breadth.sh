#!/usr/bin/env bash
# experiments/dwell_120_breadth.sh — give 120 ms the same evidence breadth as 80 ms.
#
# 120 ms is the best dwell measured (-0.00122, 5.8x better than shipped, horizontal
# still better) but so far only on FLAT ground walking STRAIGHT, so the report has
# to hedge and recommend 80 ms. This runs 120 ms through the same checks 80 ms
# already passed — turning, and both terrains — so the recommendation can be made
# on evidence rather than caution.
#
# Terrain runs are 700 ticks: the heightfield is +-8 m and a 30 s walk leaves it.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/dwell results/terrain "$TMPDIR"

while pgrep -f "dwell_terrain.sh|terrain_arms.sh|d_levers.sh" >/dev/null 2>&1; do sleep 30; done

# turning, flat, 30 s — directly comparable to analytic_c2_dw0.08
for seed in 0 1; do
  out="results/dwell/analytic_c2_dw0.12_s${seed}.npz"
  [[ -f "$out" ]] || uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --yaw 0.3 \
    --contacts-per-foot 4 --imu-noise --noise-seed "$seed" --contact-source trust \
    --dwell 0.12 --out "$out" >> results/dwell/dwell120.log 2>&1 \
    || echo "!! c2 dw0.12 s$seed FAILED"
done

# both terrains, 14 s — directly comparable to the terrain dwell sweep
for terr in hard_stepping waves; do
  for seed in 0 1; do
    out="results/terrain/${terr}_dw0.12_s${seed}.npz"
    [[ -f "$out" ]] || timeout 1500 uv run python experiments/z_budget.py --ticks 700 --vx 0.4 \
      --contacts-per-foot 4 --imu-noise --noise-seed "$seed" --contact-source trust \
      --dwell 0.12 --terrain "$terr" --terrain-seed "$seed" --out "$out" \
      >> results/terrain/terrain.log 2>&1 \
      || echo "!! $terr dw0.12 s$seed FAILED (may have fallen)"
  done
done
echo "### $(date +%H:%M:%S) dwell-120 breadth done"
