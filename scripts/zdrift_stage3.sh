#!/usr/bin/env bash
# Verify the OUTER floors per-terrain, then retrain at one that is genuinely
# sign-consistent -- or report that none is.
#
# Both inner candidates (3e-5, 1e-4) came back MIXED across the four held-out
# terrains, so the drift zero-crossing is not a narrow feature: it spans the whole
# region where drift is small. The outer floors sit further from the crossing and may
# be consistently signed (all sinking) -- honest but larger. If even they are mixed,
# then contact_meas_var never yields sign-consistent drift on this pool, and it is
# acting as a BIAS trim rather than a noise parameter. That is a structural finding,
# not a tuning one: a correctly specified measurement covariance cannot move the
# estimate's mean at all, yet this one drives drift monotonically through zero.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/zdrift
LOG=$OUT/zdrift.log
log(){ echo "[$(date +%F' '%H:%M:%S)] [stage3] $*" | tee -a "$LOG"; }
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"
CELLS="L256_A_l2vel L256_C_l2velori"
STEPS=6000

for F in 3e-4 1e-3; do
  log "per-terrain verification at contact_meas_var=${F}"
  $GPU scripts/drift_backfill.py --root results/l_ablation --pool n8fix \
     --only ${CELLS} --contact-meas-var "${F}" \
     --out "$OUT/sweep_cmv_${F}.json" 2>&1 | tee -a "$LOG"
done

log "--- full table ---"
uv run python scripts/zdrift_summary.py 2>&1 | tee -a "$LOG"
BEST=$(uv run python scripts/zdrift_summary.py --best 2>/dev/null | tail -1)

if [ -z "$BEST" ]; then
  log "!! NO floor gives sign-consistent drift across the four held-out terrains."
  log "   contact_meas_var is trimming a BIAS, not setting a noise level. A correctly"
  log "   specified measurement covariance cannot move the estimate's MEAN, yet this"
  log "   one drives drift monotonically through zero -- the signature of it"
  log "   compensating for a missing observation rather than describing a noise."
  log "   See ~/Documents/filter-debugging/contact-zero-velocity.pdf: H has no"
  log "   velocity columns, so Sigma_C can only scale a correction whose DIRECTION is"
  log "   fixed. Not retraining -- there is no defensible floor to retrain at, and a"
  log "   6 h run at an arbitrary one would produce a number with no meaning."
  exit 0
fi

log "selected contact_meas_var=${BEST} (verified sign-consistent on all four terrains)"
declare -A OBJ=( [L256_A_l2vel]=l2_velocity [L256_C_l2velori]=l2_vel_ori )
for cell in ${CELLS}; do
  dst="$OUT/${cell}_cmv${BEST}"
  [ -f "${dst}/summary.json" ] && { log "retrain ${cell}: present, skipping"; continue; }
  log "retrain ${cell} (${OBJ[$cell]}) at L=256, contact_meas_var=${BEST}"
  $GPU scripts/run_contactnet.py --pool n8fix --contacts-per-foot 4 \
     --objective "${OBJ[$cell]}" --L 256 --no-remat --contact-meas-var "${BEST}" \
     --steps ${STEPS} --warmup-steps 100 --time-budget-s 86400 \
     --out-dir "${dst}" 2>&1 | tee -a "$LOG"
  ran=$(uv run python -c "
import json
try:  print(json.load(open('${dst}/summary.json'))['steps_run'])
except Exception: print(-1)" 2>/dev/null | tail -1)
  [ "$ran" = "${STEPS}" ] && log "retrain ${cell}: complete" || log "!! retrain ${cell} FAILED (${ran})"
done

log "final scoring at contact_meas_var=${BEST}"
$GPU scripts/drift_backfill.py --root "$OUT" --pool n8fix \
   --contact-meas-var "${BEST}" --out "$OUT/drift_retrained.json" 2>&1 | tee -a "$LOG"
log "=== stage 3 complete ==="
