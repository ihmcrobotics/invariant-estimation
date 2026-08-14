#!/usr/bin/env bash
# Overnight ContactNet L2-options ladder.
#
# Collect a FRESH, fixed-terrain (B1) stratified DR pool, then train the four loss
# arms on it back-to-back:
#     A_l2vel        objective l2_velocity      (= R3b baseline, unchanged)
#     B_l2velpos     objective l2_vel_pos       (+ segment-relative position L2)
#     C_l2velori     objective l2_vel_ori       (+ SO(3)-log orientation L2)
#     D_l2velposori  objective l2_vel_pos_ori   (+ both)
#
# Deadline-aware: each arm gets (stop_by - now)/remaining_arms minus a validation
# reserve, so the whole ladder finishes before the deadline whatever collection
# costs, and the four arms get comparable training budgets. Pose-loss weights are
# auto-sized once per arm (ratio 0.5) and frozen -- see run_contactnet.py.
#
# Usage:  scripts/overnight_loss_ladder.sh [STOP_BY_HHMM]      (default 08:45)
# Env overrides: POOL_TAG CONTACTS SECONDS_PER COLLECT_SEEDS STEPS_CAP WARMUP VAL_RESERVE
set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root

STOP_BY_HHMM="${1:-08:45}"                   # finish ALL arms by this local time
POOL_TAG="${POOL_TAG:-n8fix}"
CONTACTS="${CONTACTS:-4}"                     # 4 = N=8 corner set
SECONDS_PER="${SECONDS_PER:-42}"
COLLECT_SEEDS="${COLLECT_SEEDS:-30}"
STEPS_CAP="${STEPS_CAP:-12000}"              # high cap; the time budget is the real limiter
WARMUP="${WARMUP:-100}"
VAL_RESERVE="${VAL_RESERVE:-1000}"           # s reserved per arm for held-out validation
LOG=results/overnight_loss_ladder.log

mkdir -p results
stop_by=$(date -d "today ${STOP_BY_HHMM}" +%s)
[ "$stop_by" -le "$(date +%s)" ] && stop_by=$(date -d "tomorrow ${STOP_BY_HHMM}" +%s)

log(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== L2-options ladder start; stop_by=$(date -d @${stop_by} +%F' '%H:%M) pool=${POOL_TAG} contacts_per_foot=${CONTACTS} ==="

# ---- 1. collect a fresh fixed-terrain pool (idempotent: skip if enough exist) ----
have=$(ls data/*_${POOL_TAG}_seed*.npz 2>/dev/null | wc -l)
if [ "$have" -lt 20 ]; then
  now=$(date +%s); left=$(( stop_by - now ))
  cbudget=$(( left - 10800 ))                # leave >= 3h for cache + 4 arms
  [ "$cbudget" -gt 10800 ] && cbudget=10800  # ...but cap collection at 3h
  [ "$cbudget" -lt 1800 ]  && cbudget=1800
  log "collecting up to ${COLLECT_SEEDS} rollouts (budget ${cbudget}s); currently have ${have}"
  uv run python scripts/collect_dr_pool.py --contacts-per-foot "${CONTACTS}" \
     --seeds "${COLLECT_SEEDS}" --seconds "${SECONDS_PER}" --name-tag "${POOL_TAG}" \
     --time-budget-s "${cbudget}" 2>&1 | tee -a "$LOG"
else
  log "pool already has ${have} rollouts; skipping collection"
fi
have=$(ls data/*_${POOL_TAG}_seed*.npz 2>/dev/null | wc -l)
log "pool ready: ${have} rollouts across terrains"
if [ "$have" -lt 8 ]; then log "FATAL: too few rollouts (${have}); aborting"; exit 1; fi

# ---- 2. pre-build channel caches ONCE so the four arms train on equal budgets ----
log "pre-building channel caches for the pool (one-time, ~4 min/rollout)"
uv run python -c "
import sys; sys.path.insert(0, '.')
import run_policy as rp; rp.DT = 0.001; rp.DECIMATION = 20
import invariant_estimation  # noqa: x64
from invariant_estimation.sim import collect
from invariant_estimation.contactnet import dataset
pool = sorted(collect.DATA_DIR.glob('*_${POOL_TAG}_seed*.npz'))
c = collect.build_collector(contacts_per_foot=${CONTACTS})
dataset.build_channel_cache(pool, c)
print('prebuilt channel cache for', len(pool), 'rollouts', flush=True)
" 2>&1 | tee -a "$LOG"

# ---- 3. the four arms, deadline-split ----
OBJS=(l2_velocity l2_vel_pos l2_vel_ori l2_vel_pos_ori)
TAGS=(A_l2vel B_l2velpos C_l2velori D_l2velposori)
remaining=${#OBJS[@]}
for i in "${!OBJS[@]}"; do
  obj=${OBJS[$i]}; tag=${TAGS[$i]}
  now=$(date +%s); left=$(( stop_by - now ))
  budget=$(( left / remaining - VAL_RESERVE ))
  [ "$budget" -lt 300 ] && budget=300        # floor: still trains + validates a little
  log "--- arm ${tag} (${obj}); train budget ${budget}s; ${remaining} arm(s) left; $(( left ))s to stop_by ---"
  uv run python scripts/run_contactnet.py --pool "${POOL_TAG}" \
     --contacts-per-foot "${CONTACTS}" --objective "${obj}" \
     --steps "${STEPS_CAP}" --warmup-steps "${WARMUP}" \
     --time-budget-s "${budget}" --tag "${tag}" 2>&1 | tee -a "$LOG"
  log "--- arm ${tag} finished (exit $?) ---"
  remaining=$(( remaining - 1 ))
done

log "=== ladder complete ==="
