#!/usr/bin/env bash
# Second overnight batch: does opening the quasi-static gate help or hurt?
#
# Measured by `gravity_gate_trace.py` on this exact clip: the gate passes 0.479%
# of walking ticks. The median term is 1.7-2.9x its tolerance, so the thresholds
# are not marginally wrong -- the specific force during gait genuinely is not
# gravity (median |f_perp| = 1.47 m/s^2, which reads as 8.6 deg of apparent tilt).
# Relaxing them therefore admits a BIASED measurement; whether the anisotropic R
# absorbs that is exactly what these three runs answer.
#
# `baseT` is the same configuration as `base`, re-run only so the tilt-RMS column
# exists on both sides of the comparison.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=results/zv_qc_2026-08-11/closed_loop
mkdir -p "$OUT" /tmp/zvqc

run() {
  local name="$1"; shift
  if [ -s "$OUT/cmv_1e-3_$name.json" ]; then echo "== SKIP $name"; return 0; fi
  echo "== $(date +%H:%M:%S) $name  $*"
  bash scripts/gpu_lock.sh uv run --extra gpu python scripts/record_contactnet_demo.py \
    --contacts-per-foot 4 --contact-meas-var 1e-3 "$@" \
    --out "/tmp/zvqc/$name.mp4" --metrics "$OUT/cmv_1e-3_$name.json" \
    > "$OUT/$name.log" 2>&1
  echo "   exit=$? $(tail -3 "$OUT/$name.log" | head -1)"
}

run baseT                                    # identical to `base`, now with tilt RMS
run gateRelax --gravity-gates 0.15,0.45,1.5  # ~3x each: about the median of every term
run gateWide  --gravity-gates 2.0,5.0,20.0   # past every p99: leveling effectively always on

echo "== $(date +%H:%M:%S) gate batch done"
