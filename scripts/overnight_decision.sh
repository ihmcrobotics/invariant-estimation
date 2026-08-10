#!/usr/bin/env bash
# Overnight: produce the two things the go/no-go on ContactNet needs by morning.
#
#   (1) DRIFT, per existing checkpoint, plus whether drift ranks like the velocity
#       RMSE every arm was actually selected on. If those rankings disagree, every
#       conclusion drawn from the RMSE tables is about the wrong quantity.
#
#   (2) A RANDOMIZED-MOTION pool and a net trained on it, evaluated on the ordinary
#       walking held-out set. This tests the hypothesis the L-ablation pointed at:
#       four very different objectives converged to within 2.5% at L=256 and
#       vertical NEES stalled at ~2.0 regardless of horizon, which says the binding
#       constraint is upstream of both the loss and the window -- data or socket.
#       Periodic walking lets the network regress Sigma_C off gait phase (measured
#       phase R^2 = 0.942) instead of contact condition. Break the periodicity and
#       either it learns the real thing or it does not, and we know which.
#
# Train on randomized motion, evaluate on plain walking: that is the deployment
# question, not a fairness compromise.
#
# Stages are independent and each is skipped if its output already exists, so this
# can be re-run after an interruption without redoing work.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-6000}"
L="${L:-256}"                  # the horizon where the objectives converged
ARMS="${ARMS:-A_l2vel B_l2velpos}"
POOL="${POOL:-n8rand}"
SEEDS="${SEEDS:-16}"
SECONDS_PER="${SECONDS_PER:-42}"
OUT=results/rand_motion
LOG=$OUT/overnight.log
mkdir -p "$OUT"
log(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A OBJ=( [A_l2vel]=l2_velocity [B_l2velpos]=l2_vel_pos
                 [C_l2velori]=l2_vel_ori [D_l2velposori]=l2_vel_pos_ori )

log "=== overnight decision run: L=${L} steps=${STEPS} arms={${ARMS}} pool=${POOL} ==="

# ---- 1. drift on the finished L-ablation cells -----------------------------
if [ -f results/l_ablation/drift_backfill.json ]; then
  log "stage 1: drift_backfill already present, skipping"
else
  log "stage 1: measuring drift on the finished L-ablation cells"
  bash scripts/gpu_lock.sh uv run --extra gpu python scripts/drift_backfill.py --root results/l_ablation 2>&1 | tee -a "$LOG"
  log "stage 1 done (exit $?)"
fi

# ---- 2. survivability probe, then the randomized pool ----------------------
have=$(ls data/*_${POOL}_seed*.npz 2>/dev/null | wc -l)
if [ "$have" -ge 8 ]; then
  log "stage 2: pool '${POOL}' already has ${have} rollouts, skipping collection"
  PERIOD="${PERIOD:-unknown}"
else
  log "stage 2a: survivability probe (a policy that falls yields no data)"
  bash scripts/rand_motion_check.sh 0.4 0.8 2>&1 | tee -a "$LOG"
  PERIOD=""
  for P in 0.4 0.8; do            # fastest surviving period wins
    n=$(ls data/*_probe${P}_seed*.npz 2>/dev/null | wc -l)
    if [ "$n" -ge 2 ]; then PERIOD=$P; break; fi
  done
  if [ -z "$PERIOD" ]; then
    log "!! no candidate period survived. The command schedule is the binding"
    log "   constraint on what this policy can generate -- that is a POLICY problem,"
    log "   not a Sigma_C problem. Stopping stage 2; stage 1 output stands."
    exit 3
  fi
  log "stage 2b: collecting '${POOL}' at cmd_resample_s=${PERIOD} (stride is ~1.0 s)"
  uv run --extra gpu python scripts/collect_dr_pool.py \
     --contacts-per-foot 4 --seeds "${SEEDS}" --seconds "${SECONDS_PER}" \
     --cmd-resample-s "${PERIOD}" --cmd-vx 0.0 0.9 --cmd-vy 0.0 0.6 --cmd-yaw 0.0 1.5 \
     --disturb-rate-hz 1.2 --name-tag "${POOL}" --time-budget-s 10800 2>&1 | tee -a "$LOG"
fi
have=$(ls data/*_${POOL}_seed*.npz 2>/dev/null | wc -l)
log "pool '${POOL}': ${have} rollouts"
if [ "$have" -lt 8 ]; then log "!! too few rollouts (${have}); stopping"; exit 4; fi

# ---- 3. train on randomized motion -----------------------------------------
for tag in ${ARMS}; do
  cell="$OUT/L${L}_${tag}"
  if [ -f "${cell}/summary.json" ]; then log "stage 3: ${tag} done, skipping"; continue; fi
  log "stage 3: training ${tag} (${OBJ[$tag]}) at L=${L} on the randomized pool"
  bash scripts/gpu_lock.sh uv run --extra gpu python scripts/run_contactnet.py \
     --pool "${POOL}" --contacts-per-foot 4 --objective "${OBJ[$tag]}" \
     --L "${L}" --no-remat --contact-meas-var 1.0e-4 \
     --steps "${STEPS}" --warmup-steps 100 --time-budget-s 86400 \
     --out-dir "${cell}" 2>&1 | tee -a "$LOG"
  ran=$(uv run python -c "
import json
try:  print(json.load(open('${cell}/summary.json'))['steps_run'])
except Exception: print(-1)" 2>/dev/null | tail -1)
  [ "$ran" = "${STEPS}" ] && log "stage 3: ${tag} complete (${ran} steps)" \
                          || log "!! stage 3: ${tag} FAILED (steps_run=${ran})"
done

# ---- 4. drift for the randomized nets, on the WALKING held-out set ---------
# --pool n8fix deliberately: the checkpoints come from --root, the evaluation
# rollouts from --pool. Training on randomized motion and scoring on ordinary
# walking IS the deployment question. drift_backfill loads each cell's own frozen
# norm_constants.npz rather than refitting, which is what makes the cross-pool
# comparison legitimate instead of silently distribution-shifted.
log "stage 4: drift for the randomized nets, evaluated on the n8fix walking held-out set"
bash scripts/gpu_lock.sh uv run --extra gpu python scripts/drift_backfill.py --root "$OUT" --pool n8fix \
   --out "$OUT/drift_rand_on_walking.json" 2>&1 | tee -a "$LOG"

log "=== overnight complete ==="
log "  L-ablation drift : results/l_ablation/drift_backfill.json"
log "  randomized drift : $OUT/drift_rand_on_walking.json"
