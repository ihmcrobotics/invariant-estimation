#!/usr/bin/env bash
# Retrain at contact_meas_var=1e-3 -- the ONLY floor whose drift sign agrees across
# all four held-out terrains. Every floor that looks better on drift is a
# cancellation: 0, 3e-5, 1e-4 and 3e-4 all have terrains on both sides of zero.
#
# Note what we are retraining at: 1e-3 is simultaneously the WORST floor on
# accumulated error (-0.133/-0.165) and the worst on consistency (NIS/dof
# 0.022/0.014) in evaluation. That is not a mistake -- it is the only honest
# operating point, and the eval numbers are train/test mismatched because these nets
# were TRAINED at 1e-4. Whether a net trained at 1e-3 recovers drift while keeping
# the sign honest is precisely the open question.
#
# Scoring runs after EACH arm, so a late second retrain never leaves an unscored
# checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/zdrift; LOG=$OUT/zdrift.log
log(){ echo "[$(date +%F' '%H:%M:%S)] [stage4] $*" | tee -a "$LOG"; }
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"
BEST=1e-3; STEPS=6000

declare -A OBJ=( [L256_A_l2vel]=l2_velocity [L256_C_l2velori]=l2_vel_ori )
for cell in L256_A_l2vel L256_C_l2velori; do
  dst="$OUT/${cell}_cmv${BEST}"
  if [ ! -f "${dst}/summary.json" ]; then
    log "retrain ${cell} (${OBJ[$cell]}) at L=256, contact_meas_var=${BEST}"
    $GPU scripts/run_contactnet.py --pool n8fix --contacts-per-foot 4 \
       --objective "${OBJ[$cell]}" --L 256 --no-remat --contact-meas-var "${BEST}" \
       --steps ${STEPS} --warmup-steps 100 --time-budget-s 86400 \
       --out-dir "${dst}" 2>&1 | tee -a "$LOG"
    ran=$(uv run python -c "
import json
try:  print(json.load(open('${dst}/summary.json'))['steps_run'])
except Exception: print(-1)" 2>/dev/null | tail -1)
    [ "$ran" = "${STEPS}" ] && log "retrain ${cell}: complete" || { log "!! ${cell} FAILED (${ran})"; continue; }
  else
    log "retrain ${cell}: present, skipping"
  fi
  log "scoring everything trained so far at contact_meas_var=${BEST}"
  $GPU scripts/drift_backfill.py --root "$OUT" --pool n8fix \
     --contact-meas-var "${BEST}" --out "$OUT/drift_retrained.json" 2>&1 | tee -a "$LOG"
done
log "=== stage 4 complete ==="
