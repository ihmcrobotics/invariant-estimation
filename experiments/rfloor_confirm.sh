#!/usr/bin/env bash
# experiments/rfloor_confirm.sh — confirm the contact_meas_var finding across seeds.
#
# `contact_meas_var` is the port's own "landmine #2" (main_estimator.py:405): an
# isotropic floor folded into Sigma_q before the contact update's R, described as
# the stand-in for flight's ConstantContactMeasurementNoiseProvider, and defaulted
# to 0.0 == "current port behaviour". With the default, R = J Sigma_q J^T is
# ~1e-8..1e-6 against H P H^T ~1e-4, so the contact update is nearly a hard
# constraint.
#
# First measurement (n=1): setting it to 1e-4 took ContactNet from -0.01728 to
# -0.01209 (30% better) with horizontal slightly better too. That is the largest
# Sigma_C-adjacent lever found, and it matches what flight already does — so it
# needs seeds before it goes in a recommendation.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/rfloor "$TMPDIR"
CKPT=results/2026-08-06_06-42-29_D_l2velposori/params.npz
NORM=results/2026-08-06_06-42-29_D_l2velposori/norm_constants.npz

while pgrep -f "dwell_terrain.sh|terrain_arms.sh|dwell_120_breadth.sh" >/dev/null 2>&1; do
  sleep 30
done

for seed in 0 1 2; do
  for v in 0.0 1.0e-4; do
    for arm in analytic contactnet; do
      out="results/rfloor/${arm}_rf${v}_s${seed}.npz"
      [[ -f "$out" ]] && continue
      extra=""
      [[ "$arm" == "contactnet" ]] && extra="--contactnet $CKPT --contactnet-norm $NORM"
      echo "### $(date +%H:%M:%S) $arm contact_meas_var=$v seed=$seed"
      # shellcheck disable=SC2086
      uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
        --imu-noise --noise-seed "$seed" --contact-meas-var "$v" --out "$out" $extra \
        >> results/rfloor/rfloor.log 2>&1 || echo "!! $arm v=$v s=$seed FAILED"
    done
  done
done
echo "### $(date +%H:%M:%S) rfloor confirmation done"
