#!/usr/bin/env bash
# experiments/terrain_arms.sh — is the "ContactNet is 2.5x worse" result specific
# to flat ground?
#
# Terrain is IN ContactNet's training distribution (the pool is stratified four
# ways), so if the penalty were a flat-ground artifact this is where it would
# close. Shipped dwell throughout, so this isolates the arm and not the dwell.
# NOTE: the heightfield is EXTENT=16 m centred at the origin, i.e. +-8 m. At
# vx=0.4 a 30 s run travels ~12 m and walks straight off the edge (measured: the
# robot fell, z=-378 m, ncon=0). 700 control ticks = 14 s ~ 5.6 m keeps it on the
# field. Drift is a rate, so the numbers stay comparable to the flat runs; the
# shorter window just makes them noisier.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/terrain "$TMPDIR"
CKPT=results/2026-08-06_06-42-29_D_l2velposori/params.npz
NORM=results/2026-08-06_06-42-29_D_l2velposori/norm_constants.npz

while pgrep -f "dwell_terrain.sh|dwell_long.sh" >/dev/null 2>&1; do sleep 30; done

for terr in hard_stepping waves; do
  for seed in 0 1; do
    out="results/terrain/contactnet_${terr}_dw0.04_s${seed}.npz"
    [[ -f "$out" ]] && continue
    echo "### $(date +%H:%M:%S) contactnet $terr seed=$seed"
    timeout 1500 uv run python experiments/z_budget.py --ticks 700 --vx 0.4 \
      --contacts-per-foot 4 --imu-noise --noise-seed "$seed" --contact-source trust \
      --terrain "$terr" --terrain-seed "$seed" --contactnet "$CKPT" \
      --contactnet-norm "$NORM" --out "$out" \
      >> results/terrain/terrain.log 2>&1 \
      || echo "!! contactnet $terr s=$seed FAILED (may have fallen)"
  done
done
echo "### $(date +%H:%M:%S) terrain arms done"
