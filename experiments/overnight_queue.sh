#!/usr/bin/env bash
# experiments/overnight_queue.sh — run the closed-loop sweeps strictly in order.
#
# One waiter, one order, so priority is explicit and re-orderable in one place.
# Load average sits near the core count with a single closed-loop run plus the
# GPU job, so these never overlap each other.
#
# Order is by decision value: detection answers (a); the arm sweep tests whether
# drift trades against the training loss across A/B/C/D; the OOD grid answers (b);
# the lever sweep is the (d) hunt and is the acceptable thing to lose if time runs out.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "z_matrix.sh" >/dev/null 2>&1; do sleep 45; done

for stage in "a_detection.sh results/adetect" \
             "c_arms.sh      results/carms" \
             "b_ood.sh       results/bood" \
             "d_levers.sh    results/dlevers"; do
  # shellcheck disable=SC2086
  set -- $stage
  echo "### $(date +%H:%M:%S) starting $1"
  bash "experiments/$1" "$2"
done
echo "### $(date +%H:%M:%S) queue complete"
