#!/usr/bin/env bash
# Can the walking policy survive faster command resampling? Measured, not assumed.
#
# The point of randomizing motion is to break the stride clock: at cmd_resample_s=3.0
# the command changes more slowly than the ~1.0 s stride, the gait settles into a
# limit cycle, and gait phase becomes a near-deterministic function of the sensor
# window -- so ContactNet can regress the phase-conditional mean of Sigma_C (measured
# phase R^2 = 0.942) instead of learning contact condition.
#
# But the 2026-08-07 DR study lost 7 of 8 rollouts to a friction tail, and crucially
# the falls needed the DR *and* the randomized command schedule TOGETHER -- a
# constant-command sim sweep cleared every arm and missed it entirely. A policy that
# falls yields no data, which trains nothing. So this probes the real collect path on
# a couple of seeds per candidate period and reports which survive, fastest first.
#
# Usage:  scripts/rand_motion_check.sh [PERIODS...]     (default: 0.4 0.8)
set -uo pipefail
cd "$(dirname "$0")/.."

PERIODS="${*:-0.4 0.8}"
SEEDS="${SEEDS:-2}"
SECONDS_PER="${SECONDS_PER:-42}"
TERRAINS="${TERRAINS:-flat hard_stepping}"
LOG=results/rand_motion/check.log
mkdir -p results/rand_motion

log(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }
log "=== survivability check; periods={${PERIODS}} seeds=${SEEDS} terrains={${TERRAINS}} ==="

for P in ${PERIODS}; do
  log "--- cmd_resample_s=${P} ---"
  # seed0 900 keeps these probe rollouts out of the training seed range, and
  # --name-tag parks them under a throwaway pool name so a fallen or short probe
  # rollout can never be picked up by a later training run.
  uv run --extra gpu python scripts/collect_dr_pool.py \
     --contacts-per-foot 4 --seeds "${SEEDS}" --seed0 900 \
     --seconds "${SECONDS_PER}" --terrains ${TERRAINS} \
     --cmd-resample-s "${P}" --cmd-vx 0.0 0.9 --cmd-vy 0.0 0.6 --cmd-yaw 0.0 1.5 \
     --disturb-rate-hz 1.2 \
     --name-tag "probe${P}" --time-budget-s 1800 2>&1 | tee -a "$LOG"
  n=$(ls data/*_probe${P}_seed*.npz 2>/dev/null | wc -l)
  log "--- cmd_resample_s=${P}: ${n}/${SEEDS} rollouts survived ---"
done

log "=== survivability summary ==="
for P in ${PERIODS}; do
  n=$(ls data/*_probe${P}_seed*.npz 2>/dev/null | wc -l)
  log "  cmd_resample_s=${P}: ${n}/${SEEDS} survived"
done
log "Pick the FASTEST period that survives all seeds; it is the one that breaks the"
log "stride clock hardest while still yielding data. If none survive, the command"
log "schedule -- not the horizon and not the loss -- is the binding constraint on"
log "what this policy can generate, and the answer is a different policy, not a"
log "different Sigma_C."
