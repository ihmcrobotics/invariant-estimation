#!/usr/bin/env bash
# Second pass on the randomized pool: arms C and D.
#
# A and B were chosen before stage 1's drift numbers existed. Those numbers changed
# which arms are interesting: B is the worst cell in the table (the only one worse
# than analytic on drift, and the only one drifting UPWARD), while D has the smallest
# drift slope of any cell (0.00061 m/s, 0.06x analytic). Training all four on the
# randomized pool gives a complete paired comparison against their periodic
# counterparts in results/l_ablation/L256_*, which is the cleanest form of the
# question: does breaking the stride clock change what the network learns, holding
# arm and horizon fixed?
#
# Collection is skipped (the pool already exists) and stage 4 re-runs over every cell
# in the directory, so the final drift table covers all four arms.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/rand_motion/overnight.log
log(){ echo "[$(date +%F' '%H:%M:%S)] [pass2] $*" | tee -a "$LOG"; }

log "waiting for the A/B pass to finish"
while pgrep -f "[o]vernight_decision.sh" > /dev/null 2>&1; do sleep 120; done
log "A/B pass finished; starting arms C and D on the randomized pool"
ARMS="C_l2velori D_l2velposori" bash scripts/overnight_decision.sh
log "pass 2 finished (exit $?)"
