#!/usr/bin/env bash
# Third overnight batch: the ZERO-VELOCITY MECHANISM check, not another headline.
#
# `zero-velocity-implementation-plan.md` wrote down two falsifiable predictions
# BEFORE any number was measured. Both need the per-tick history, which the metrics
# JSON does not carry:
#
#   P1  the drift signature turns from LINEAR to sqrt(t). Today's drift is measured
#       linear, i.e. bias-dominated; if it is still linear after ZV, the roll bias
#       was not handled.
#   P2  the update-DEPOSITED component shrinks. It sat at -0.18..-0.21 m across four
#       previous arms regardless of what was changed -- the fingerprint of the
#       common-mode null mode. Computed as
#           (est_p - true_p)[-1]_z  -  cumsum((est_v - true_v)_z) * dt
#
# A headline number that moved for the wrong reason is exactly what this project
# has been burned by, so these two runs are worth their 3 minutes each.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=results/zv_qc_2026-08-11/closed_loop
mkdir -p "$OUT" /tmp/zvqc

run() {
  local name="$1"; shift
  if [ -s "$OUT/$name.npz" ]; then echo "== SKIP $name"; return 0; fi
  echo "== $(date +%H:%M:%S) $name  $*"
  bash scripts/gpu_lock.sh uv run --extra gpu python scripts/record_contactnet_demo.py \
    --contacts-per-foot 4 --contact-meas-var 1e-3 "$@" \
    --out "/tmp/zvqc/$name.mp4" --history "$OUT/$name.npz" \
    > "$OUT/$name.hist.log" 2>&1
  echo "   exit=$?"
}

run hist_base                                        # the reference signature
run hist_zvK1   --zero-velocity                      # the derived trust level
run hist_zvK10  --zero-velocity --nv-scale 10        # whichever kappa looked best

echo "== $(date +%H:%M:%S) mechanism batch done"
