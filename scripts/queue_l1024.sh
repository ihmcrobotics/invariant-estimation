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

# HELD, 2026-08-09. The L=1024 column is NOT launched automatically any more.
#
# Two reasons. (1) The trimmed column (A, C) cannot test the hypothesis it exists
# for: the stride-crossover argument is about a POSITION loss term distinguishing
# accumulated DC drift from bounded gait oscillation, and neither l2_velocity nor
# l2_vel_ori has a position term at any horizon. The arms carrying the mechanism are
# B (l2_vel_pos) and D. (2) Measured returns are decaying: RMSE -14.5% then -5.4%
# per doubling, NEES_z saturated at ~2.0 between L=256 and 512, NIS/dof flat at
# ~0.18 across every L. Committing 19 h on that basis is not warranted before we
# have drift numbers.
#
# So: measure drift on the finished cells, then STOP and let a human choose. If the
# column is run, the informative design is A (control: loss blind to accumulation)
# vs B (treatment: loss that can see it) -- not A vs C.
log "grid finished; measuring drift on the completed cells"
uv run --extra gpu python scripts/drift_backfill.py --root "${OUT_ROOT}" 2>&1 | tee -a "$LOG"
rc=$?
log "drift backfill finished (exit ${rc}); L=1024 is HELD pending that result."
log "  to run it:  ARMS='A_l2vel B_l2velpos' LVALS=1024 REMAT=on bash scripts/l_ablation_ladder.sh"
exit $rc
