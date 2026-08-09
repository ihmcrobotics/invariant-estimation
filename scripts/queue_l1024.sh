#!/usr/bin/env bash
# Chain the L=1024 column behind the running {128,256,512} ladder.
#
# WHY 1024 and why remat: vertical drift is the time-integral of vertical velocity
# error. Its DC part grows linearly in the window T = L*dt; the gait-periodic part is
# bounded by A/omega and is FLAT in T. Measured on the n8fix pool (truth v_z std
# 0.141 m/s at 2.1 Hz, stride ~1.0 s), the periodic within-window excursion is
# ~15-21 mm at every L, while the DC term at the measured ContactNet drift rate
# (0.0175 m/s) is 2.24 mm at L=128 and 17.9 mm at L=1024. The two cross at
# L ~ 1024 == one stride. Below it a DC sink and a phase-locked oscillation are not
# distinguishable functions over the window, so no objective can prefer fixing the
# drift -- which is the same wall the 2026-08-08 absolute-z study hit ("the 128 ms
# BPTT window is the binding constraint, not the objective").
#
# L=1024 is also where remat stops being optional. Measured (scripts/remat_probe.py,
# RTX 4070 12 GB): L=512 peaks at 5201 MB without remat, so L=1024 extrapolates to
# ~10.2 GB against ~11.4 GB free -- too tight to trust. With remat, ~5.1 GB.
#
# Usage:  setsid nohup scripts/queue_l1024.sh > results/l_ablation/queue1024.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-results/l_ablation}"
LOG="${OUT_ROOT}/ladder.log"
POLL="${POLL:-120}"

log(){ echo "[$(date +%F' '%H:%M:%S)] [queue1024] $*" | tee -a "$LOG"; }

log "waiting for the {128,256,512} ladder to finish (poll ${POLL}s)"
while pgrep -f "l_ablation_ladder.sh" > /dev/null 2>&1; do
  sleep "$POLL"
done
log "ladder finished; starting the L=1024 column with remat ON"

# Do NOT gate on the earlier columns succeeding: L=1024 is the column that tests the
# stride-crossover prediction, and it is worth having even if one of the shorter
# cells needs a re-run. The ladder skips completed cells, so nothing is redone.
LVALS=1024 REMAT=on STEPS="${STEPS:-6000}" OUT_ROOT="${OUT_ROOT}" \
  bash scripts/l_ablation_ladder.sh
rc=$?
log "L=1024 column finished (exit ${rc})"
exit $rc
