#!/usr/bin/env bash
# The z-drift pivot, overnight 2026-08-09 -> 10. One hypothesis, one cheap test, one
# confirmation. Everything serialized through scripts/gpu_lock.sh.
#
# TARGET (both, not either): a filter whose contact confidence is CONSISTENT
# (contact NIS/dof -> 1) and whose height does not sink.
#
# WHY THE R FLOOR. It is the only lever that has ever moved drift by a lot -- it
# flipped ContactNet from 2.5x worse than no network to 0.26x -- and it is untuned:
# 1e-4 was taken on faith from a study that only compared it against 0 and 1e-2.
# It is also the one knob that directly sets innovation covariance, and NIS/dof has
# sat at ~0.18 (5x over-covered) through every loss and every horizon we tried.
#
# WHY THESE ARMS. Ranked by accumulated vertical error, the four best cells are all
# POSITION-FREE (A@256 -0.056, C@256 -0.074, A@128 -0.083, C@128 -0.092). The
# position term is not mildly unhelpful, it is high-variance: its cells span 0.00061
# to 0.01896 on drift slope, containing both the best and the worst result measured.
# So: velocity and velocity+orientation only.
#
# WHY L=256. Both position-free arms peak there on both drift metrics and get worse
# at 512 (A: 0.00350 -> 0.00268 -> 0.00419). The horizon is spent.
#
# WHY NOT RANDOMIZED DATA. The one paired test, both scored on walking, cost 2.4x on
# drift slope and 3.4x on accumulated error. Periodic pool it is.
set -uo pipefail
cd "$(dirname "$0")/.."

FLOORS="${FLOORS:-0 3e-5 1e-4 3e-4 1e-3 3e-3}"   # 1e-2 excluded: measured to drift UPWARD on terrain
CELLS="${CELLS:-L256_A_l2vel L256_C_l2velori}"
STEPS="${STEPS:-6000}"
OUT=results/zdrift
LOG=$OUT/zdrift.log
mkdir -p "$OUT"
log(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"

log "=== z-drift pivot: floor sweep then retrain ==="

# ---- 1. evaluation-only floor sweep ---------------------------------------
# Nearly free because the floor is applied to the REPLAYED inputs
# (dataset.apply_contact_meas_floor), so an existing checkpoint can be scored at any
# floor without retraining. Train/test mismatched by construction -- the nets were
# trained at 1e-4 -- so this locates a promising operating point, it does not settle
# one. That is what stage 2 is for.
for F in ${FLOORS}; do
  out="$OUT/sweep_cmv_${F}.json"
  [ -f "$out" ] && { log "sweep ${F}: present, skipping"; continue; }
  log "sweep: contact_meas_var=${F}"
  $GPU scripts/drift_backfill.py --root results/l_ablation --pool n8fix \
     --only ${CELLS} --contact-meas-var "${F}" --out "$out" 2>&1 | tee -a "$LOG"
done

log "--- sweep summary (drift and consistency vs the floor) ---"
uv run python scripts/zdrift_summary.py 2>&1 | tee -a "$LOG"

# ---- 2. retrain at the floor the sweep likes ------------------------------
BEST=$(uv run python scripts/zdrift_summary.py --best 2>/dev/null | tail -1)
log "sweep picks contact_meas_var=${BEST}"
if [ -z "$BEST" ]; then log "!! no best floor resolved; stopping"; exit 2; fi

declare -A OBJ=( [L256_A_l2vel]=l2_velocity [L256_C_l2velori]=l2_vel_ori )
for cell in ${CELLS}; do
  dst="$OUT/${cell}_cmv${BEST}"
  [ -f "${dst}/summary.json" ] && { log "retrain ${cell}: present, skipping"; continue; }
  log "retrain ${cell} (${OBJ[$cell]}) at L=256, contact_meas_var=${BEST}"
  $GPU scripts/run_contactnet.py --pool n8fix --contacts-per-foot 4 \
     --objective "${OBJ[$cell]}" --L 256 --no-remat --contact-meas-var "${BEST}" \
     --steps "${STEPS}" --warmup-steps 100 --time-budget-s 86400 \
     --out-dir "${dst}" 2>&1 | tee -a "$LOG"
  ran=$(uv run python -c "
import json
try:  print(json.load(open('${dst}/summary.json'))['steps_run'])
except Exception: print(-1)" 2>/dev/null | tail -1)
  [ "$ran" = "${STEPS}" ] && log "retrain ${cell}: complete" \
                          || log "!! retrain ${cell} FAILED (steps_run=${ran})"
done

# ---- 3. score the retrained nets at their own floor -----------------------
log "final: drift + NIS for the retrained nets at contact_meas_var=${BEST}"
$GPU scripts/drift_backfill.py --root "$OUT" --pool n8fix \
   --contact-meas-var "${BEST}" --out "$OUT/drift_retrained.json" 2>&1 | tee -a "$LOG"

log "=== z-drift pivot complete ==="
