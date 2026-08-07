#!/usr/bin/env bash
# experiments/dwell_terrain.sh — the decisive robustness test for a dwell change.
#
# Every closed-loop drift number to date — mine and every earlier report's — is
# FLAT GROUND. Where contact is intermittent, a longer entry dwell could starve
# the filter rather than protect it, which would invalidate the recommendation.
# `hard_stepping` was the most reliable terrain during pool collection (7 ok / 0
# failed); `waves` next (6 / 2). A run that falls is reported, not hidden.
# NOTE: the heightfield is EXTENT=16 m centred at the origin, i.e. +-8 m. At
# vx=0.4 a 30 s run travels ~12 m and walks straight off the edge (measured: the
# robot fell, z=-378 m, ncon=0). 700 control ticks = 14 s ~ 5.6 m keeps it on the
# field. Drift is a rate, so the numbers stay comparable to the flat runs; the
# shorter window just makes them noisier.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/terrain "$TMPDIR"

while pgrep -f "dwell_long.sh|dwell_fine.sh" >/dev/null 2>&1; do sleep 30; done

for terr in hard_stepping waves; do
  for dw in 0.005 0.04 0.08; do
    for seed in 0 1; do
      out="results/terrain/${terr}_dw${dw}_s${seed}.npz"
      [[ -f "$out" ]] && continue
      echo "### $(date +%H:%M:%S) $terr dwell=$dw seed=$seed"
      timeout 1500 uv run python experiments/z_budget.py --ticks 700 --vx 0.4 \
        --contacts-per-foot 4 --imu-noise --noise-seed "$seed" --contact-source trust \
        --dwell "$dw" --terrain "$terr" --terrain-seed "$seed" --out "$out" \
        >> results/terrain/terrain.log 2>&1 \
        || echo "!! $terr dw=$dw s=$seed FAILED (may have fallen)"
    done
  done
done
echo "### $(date +%H:%M:%S) terrain dwell sweep done"
