#!/usr/bin/env bash
# experiments/cn_full_config.sh — ContactNet WITH both config fixes applied.
#
# The proposal: keep ContactNet but give it the two config wins (dwell 80 ms, R floor
# 1e-4) so the network "has less to do". Measured separately, ContactNet is immune to
# the dwell (-0.01755 vs -0.01746, 0.9%) because it OVERWRITES contact_chol and never
# sees the trust decision -- but it does respond to the R floor (+31.1%). This runs the
# combination end to end so the answer is measured rather than inferred from the parts.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/cnfull "$TMPDIR"
CKPT=results/2026-08-06_06-42-29_D_l2velposori/params.npz
NORM=results/2026-08-06_06-42-29_D_l2velposori/norm_constants.npz
for seed in 0 1 2; do
  out="results/cnfull/contactnet_dw0.08_rf1e-4_s${seed}.npz"
  [[ -f "$out" ]] && continue
  echo "### $(date +%H:%M:%S) contactnet + both config fixes, seed=$seed"
  uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
    --imu-noise --noise-seed "$seed" --contact-source trust --dwell 0.08 \
    --contact-meas-var 1.0e-4 --contactnet "$CKPT" --contactnet-norm "$NORM" \
    --out "$out" >> results/cnfull/cnfull.log 2>&1 || echo "!! seed=$seed FAILED"
done
echo "### $(date +%H:%M:%S) done"
