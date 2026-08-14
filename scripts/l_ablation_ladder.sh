#!/usr/bin/env bash
# ContactNet L-ablation: 4 loss arms x L in {128, 256, 512}, MATCHED STEPS.
#
#     A_l2vel        l2_velocity      (the CoCo-faithful baseline)
#     B_l2velpos     l2_vel_pos       (+ segment-relative position L2)
#     C_l2velori     l2_vel_ori       (+ SO(3)-log orientation L2)
#     D_l2velposori  l2_vel_pos_ori   (+ both)
#
# Deliberately NOT deadline-aware, unlike scripts/overnight_loss_ladder.sh. That
# script split a wall-clock deadline across arms, caches warmed as the night went on,
# and the four arms got 7639..9839 steps -- so results.md could not attribute D's edge
# over B to the orientation term rather than to +29% more steps. Every cell here gets
# the SAME --steps. `--time-budget-s` is a runaway backstop only; a cell that trips it
# is reported as a FAILED CELL rather than folded into the table as a shorter arm.
#
# L is a memory axis as well as a horizon: activation memory over the BPTT scan is
# O(L). Set REMAT from scripts/remat_probe.py -- do not guess.
#
# Usage:  REMAT=on scripts/l_ablation_ladder.sh
# Env:    REMAT (on|off, REQUIRED)  LVALS  STEPS  WARMUP  POOL_TAG  CONTACTS
#         CONTACT_MEAS_VAR  TIME_BUDGET  OUT_ROOT  ARMS
set -uo pipefail
cd "$(dirname "$0")/.."                      # repo root

REMAT="${REMAT:-}"
LVALS="${LVALS:-128 256 512}"
ARMS="${ARMS:-A_l2vel B_l2velpos C_l2velori D_l2velposori}"   # subset of TAGS to run
STEPS="${STEPS:-6000}"
WARMUP="${WARMUP:-100}"
POOL_TAG="${POOL_TAG:-n8fix}"
CONTACTS="${CONTACTS:-4}"                    # 4 = N=8 corner set, as the A-D arms
CONTACT_MEAS_VAR="${CONTACT_MEAS_VAR:-1.0e-4}"
TIME_BUDGET="${TIME_BUDGET:-86400}"          # backstop only; must never bind
OUT_ROOT="${OUT_ROOT:-results/l_ablation}"
LOG="${OUT_ROOT}/ladder.log"

case "$REMAT" in
  on)  REMAT_FLAG="--remat" ;;
  off) REMAT_FLAG="--no-remat" ;;
  *)   echo "FATAL: set REMAT=on or REMAT=off (measure first: scripts/remat_probe.py)" >&2
       exit 2 ;;
esac

mkdir -p "$OUT_ROOT"
log(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== L-ablation start; L={${LVALS}} steps=${STEPS} remat=${REMAT} arms={${ARMS}} pool=${POOL_TAG} contacts_per_foot=${CONTACTS} contact_meas_var=${CONTACT_MEAS_VAR} ==="

have=$(ls data/*_${POOL_TAG}_seed*.npz 2>/dev/null | wc -l)
log "pool '${POOL_TAG}': ${have} rollouts"
if [ "$have" -lt 8 ]; then log "FATAL: too few rollouts (${have}); aborting"; exit 1; fi

# Pre-build channel caches once so no cell pays for them (this is what made the A-D
# arms unequal). Idempotent: dataset._reusable short-circuits an existing cache.
log "pre-building channel caches (one-time; no-op if already present)"
uv run --extra gpu python -c "
import sys; sys.path.insert(0, '.')
import run_policy as rp; rp.DT = 0.001; rp.DECIMATION = 20
import invariant_estimation  # noqa: x64
from invariant_estimation.sim import collect
from invariant_estimation.contactnet import dataset
pool = sorted(collect.DATA_DIR.glob('*_${POOL_TAG}_seed*.npz'))
c = collect.build_collector(contacts_per_foot=${CONTACTS}, contact_meas_var=${CONTACT_MEAS_VAR})
dataset.build_channel_cache(pool, c)
print('channel cache ready for', len(pool), 'rollouts', flush=True)
" 2>&1 | tee -a "$LOG"

OBJS=(l2_velocity l2_vel_pos l2_vel_ori l2_vel_pos_ori)
TAGS=(A_l2vel B_l2velpos C_l2velori D_l2velposori)

failed=()
# L outermost, cheapest column first: L=128 re-derives the A-D comparison at matched
# steps in ~4 h, so a surprise surfaces early instead of after two days of L=512.
for L in ${LVALS}; do
  for i in "${!OBJS[@]}"; do
    obj=${OBJS[$i]}; tag=${TAGS[$i]}
    case " ${ARMS} " in *" ${tag} "*) ;; *) continue ;; esac
    cell="${OUT_ROOT}/L${L}_${tag}"
    if [ -f "${cell}/summary.json" ]; then
      log "--- cell L=${L} ${tag}: already complete, skipping ---"
      continue
    fi
    log "--- cell L=${L} ${tag} (${obj}) -> ${cell} ---"
    uv run --extra gpu python scripts/run_contactnet.py \
       --pool "${POOL_TAG}" --contacts-per-foot "${CONTACTS}" \
       --objective "${obj}" --L "${L}" ${REMAT_FLAG} \
       --contact-meas-var "${CONTACT_MEAS_VAR}" \
       --steps "${STEPS}" --warmup-steps "${WARMUP}" \
       --time-budget-s "${TIME_BUDGET}" --out-dir "${cell}" 2>&1 | tee -a "$LOG"
    rc=$?

    # A cell counts only if it ran the FULL step budget. The 2026-08-08 study lost a
    # night to an arm that ran 1 step and still wrote a complete-looking run
    # directory; `steps_run` is the only thing that distinguishes them.
    ran=$(uv run --extra gpu python -c "
import json,sys
try:  print(json.load(open('${cell}/summary.json'))['steps_run'])
except Exception: print(-1)" 2>/dev/null | tail -1)
    if [ "$rc" -ne 0 ] || [ "$ran" != "${STEPS}" ]; then
      log "!!! FAILED CELL L=${L} ${tag}: exit=${rc} steps_run=${ran} (wanted ${STEPS})"
      failed+=("L${L}_${tag}(steps_run=${ran})")
    else
      log "--- cell L=${L} ${tag} complete: ${ran} steps ---"
    fi
  done
done

if [ ${#failed[@]} -gt 0 ]; then
  log "=== ladder complete WITH ${#failed[@]} FAILED CELL(S): ${failed[*]} ==="
  log "    Do not tabulate a failed cell as a shorter arm -- that is the confound"
  log "    this ladder exists to remove. Re-run it; completed cells are skipped."
  exit 1
fi
log "=== ladder complete: all cells ran ${STEPS} steps ==="
