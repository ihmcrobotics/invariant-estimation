#!/usr/bin/env bash
# Recovery after the 2026-08-09 OOM: a drift evaluation was launched alongside a
# training cell, JAX preallocates ~75% of the card per process, and the collision
# killed the training cell AND the evaluation. Everything here runs under
# scripts/gpu_lock.sh so that cannot recur.
#
# Order is deliberate: the A comparison comes FIRST. It is the go/no-go on the
# randomization idea (same arm, same horizon, both scored on walking) and it costs
# ~25 min, so it should not sit behind 6 h of training.
#
# C is dropped. At ~3 h/cell there is room for two more arms before morning, and C
# is the least discriminating cell in the drift table. B is kept precisely because
# it is the worst -- the only cell worse than analytic and the only one drifting
# upward -- so if randomization rescues it, that is the strongest available signal.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/rand_motion/overnight.log
log(){ echo "[$(date +%F' '%H:%M:%S)] [recover] $*" | tee -a "$LOG"; }

log "step 1/2: early drift for A on the walking held-out set"
bash scripts/gpu_lock.sh uv run --extra gpu python scripts/drift_backfill.py \
   --root results/rand_motion --pool n8fix \
   --out results/rand_motion/drift_A_rand_on_walking.json 2>&1 | tee -a "$LOG"
log "step 1 done (exit $?)"

log "step 2/2: train B and D on the randomized pool, then drift over all cells"
ARMS="B_l2velpos D_l2velposori" bash scripts/overnight_decision.sh
log "recovery chain finished (exit $?)"
