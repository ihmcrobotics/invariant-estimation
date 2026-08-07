#!/usr/bin/env bash
# experiments/a_detection.sh — hypothesis (a): is contact detection causing the sink?
#
# Two things worth keeping straight, because they make the test mean different
# things in the two arms:
#
#   * ANALYTIC Sigma_C — `sensors.contact` picks stance_chol vs swing_chol, so
#     detection drives the InEKF's contact PROCESS noise directly.
#   * ContactNet     — the network OVERWRITES `contact_chol`, so detection no
#     longer reaches Sigma_C at all; it only still drives the joint-KF stance
#     anchors. A null result here and a non-null result above would localise it.
#
# `--contact-source oracle` swaps the Schmitt/dwell decision for MuJoCo's own
# contact set at zero latency. Sequential, CPU-pinned; run after the matrix.
set -uo pipefail

OUT="${1:-results/adetect}"
mkdir -p "$OUT"
LOG="$OUT/detect.log"
CKPT="results/2026-08-06_06-42-29_D_l2velposori/params.npz"
TICKS=1500

export JAX_PLATFORMS=cpu
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p "$TMPDIR"

run () {
  local tag="$1"; shift
  if [[ -f "$OUT/$tag.npz" ]]; then echo "== skip $tag" | tee -a "$LOG"; return; fi
  echo "=== $tag :: $* ===" | tee -a "$LOG"
  timeout 1200 uv run python experiments/z_budget.py \
      --ticks "$TICKS" --contacts-per-foot 4 --out "$OUT/$tag.npz" "$@" >>"$LOG" 2>&1 \
      || echo "!! $tag FAILED" | tee -a "$LOG"
}

# -- causal A/B: Schmitt/dwell vs zero-latency oracle, both Sigma_C sources -----
for seed in 0 1; do
  for src in trust oracle; do
    run "analytic_c1_${src}_s${seed}" \
        --vx 0.4 --imu-noise --noise-seed "$seed" --contact-source "$src"
    run "analytic_c2_${src}_s${seed}" \
        --vx 0.4 --yaw 0.3 --imu-noise --noise-seed "$seed" --contact-source "$src"
    run "contactnet_c1_${src}_s${seed}" \
        --vx 0.4 --imu-noise --noise-seed "$seed" --contact-source "$src" \
        --contactnet "$CKPT"
  done
done

# -- sensitivity: if drift is flat in the dwell, detection timing is not the lever
for dw in 0.0 0.02 0.08; do
  run "analytic_c1_dwell${dw}_s0" \
      --vx 0.4 --imu-noise --noise-seed 0 --contact-source trust --dwell "$dw"
done

# -- and in the thresholds
run "analytic_c1_thresh_lo_s0" --vx 0.4 --imu-noise --noise-seed 0 \
    --contact-source trust --enter 0.15 --stay 0.10
run "analytic_c1_thresh_hi_s0" --vx 0.4 --imu-noise --noise-seed 0 \
    --contact-source trust --enter 0.60 --stay 0.45

echo "=== detection sweep complete ===" | tee -a "$LOG"
