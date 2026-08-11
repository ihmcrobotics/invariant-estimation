#!/usr/bin/env bash
# Closed-loop sweep of the InEKF contact measurement-noise floor.
#
# Replaces the replay-based `zdrift_tonight.sh` sweep, which invariant N1 retracted:
# replay and closed loop agree for the analytic arm and disagree by 21x on a learned
# one, so every floor chosen by replay is unusable. This measures the same axis with
# `record_contactnet_demo.py`, the harness that validated itself against the analytic
# baseline -- ~3 min per configuration, LESS than the ~8 min replay it replaces.
#
# Two arms per floor. Selection is on the ANALYTIC arm (N2: learned held-out metrics
# have been uncorrelated with drift); the learned arm is reported beside it.
#
#   FLOORS="0 1e-4 1e-3" ARMS="analytic learned" bash scripts/cl_floor_sweep.sh
#
# Resumable: a floor whose metrics JSON already exists is skipped, so an interrupted
# sweep continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

FLOORS=${FLOORS:-"0 3e-5 1e-4 3e-4 1e-3 3e-3"}
ARMS=${ARMS:-"analytic learned"}
CKPT=${CKPT:-results/zdrift/L256_A_l2vel_cmv1e-3/params.npz}
CPF=${CPF:-4}
OUT=${OUT:-results/zdrift_bexp/closed_loop}
# Clean sensors by default. MEASURED 2026-08-10: clean reproduces the committed
# reference (`results/zdrift/closed_loop/best_A_cmv1e-3.json`, +2.4884 m) to every
# digit of all seven motions, so that number is a point on this sweep's curve and the
# harness is bit-reproducible. `--imu-noise` matches the n8fix pool's collection
# conditions instead and moves the same configuration to +1.9750 m -- a 21% level
# shift, so the two settings must never be mixed within one comparison.
NOISE=${NOISE:-}
# The mp4 is a byproduct here -- the metrics JSON is the measurement -- so the video
# goes somewhere disposable.
VID=${VID:-/tmp/cl_floor_sweep}

mkdir -p "$OUT" "$VID"
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"

for F in $FLOORS; do
  for ARM in $ARMS; do
    J="$OUT/cmv_${F}_${ARM}.json"
    if [ -s "$J" ]; then echo "== skip $ARM cmv=$F (have $J)"; continue; fi
    NET=()
    [ "$ARM" = "learned" ] && NET=(--contactnet "$CKPT")
    echo "== $ARM  cmv=$F  -> $J"
    # shellcheck disable=SC2086
    $GPU scripts/record_contactnet_demo.py \
        --contacts-per-foot "$CPF" --contact-meas-var "$F" $NOISE \
        "${NET[@]}" --out "$VID/cmv_${F}_${ARM}.mp4" --metrics "$J"
  done
done

echo "== done; summarise with: uv run python scripts/cl_floor_summary.py --dir $OUT"
