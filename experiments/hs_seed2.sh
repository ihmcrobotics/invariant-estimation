#!/usr/bin/env bash
# experiments/hs_seed2.sh — third seed on the hard_stepping dwell claim.
#
# The 80-vs-120 ms recommendation now rests on a sign instability: at 120 ms the two
# hard_stepping seeds are +0.00256 and -0.00297, against a tight -0.00159/-0.00152 at
# 80 ms. That is a two-seed claim carrying a shipping recommendation, so it gets a
# third seed. Terrain runs are 700 ticks (the heightfield is +-8 m; see dwell_terrain.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/terrain "$TMPDIR"
for dw in 0.12 0.08; do
  out="results/terrain/hard_stepping_dw${dw}_s2.npz"
  [[ -f "$out" ]] && continue
  echo "### $(date +%H:%M:%S) hard_stepping dwell=$dw seed=2"
  timeout 1500 uv run python experiments/z_budget.py --ticks 700 --vx 0.4 \
    --contacts-per-foot 4 --imu-noise --noise-seed 2 --contact-source trust \
    --dwell "$dw" --terrain hard_stepping --terrain-seed 2 --out "$out" \
    >> results/terrain/terrain.log 2>&1 || echo "!! dw=$dw s=2 FAILED (may have fallen)"
done
echo "### $(date +%H:%M:%S) hard_stepping seed-2 done"
