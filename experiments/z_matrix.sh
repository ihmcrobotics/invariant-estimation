#!/usr/bin/env bash
# experiments/z_matrix.sh — the overnight z-drift matrix, run STRICTLY sequentially.
#
# Three concurrent closed-loop runs saturate this machine, so this never forks. It
# is pinned to CPU (`JAX_PLATFORMS=cpu`) so the GPU stays free for ContactNet
# gradient work running alongside it; the estimator is ~1.5x slower there and
# numerically identical to three decimals.
#
#   bash experiments/z_matrix.sh <outdir>
#
# Writes one .npz per cell plus a `matrix.log` carrying the full stdout.
set -uo pipefail

OUT="${1:-results/zmatrix}"
mkdir -p "$OUT"
LOG="$OUT/matrix.log"
CKPT="results/2026-08-06_06-42-29_D_l2velposori/params.npz"
TICKS=1500

export JAX_PLATFORMS=cpu
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p "$TMPDIR"

run () {                      # run <tag> <extra args...>
  local tag="$1"; shift
  if [[ -f "$OUT/$tag.npz" ]]; then echo "== skip $tag (exists)" | tee -a "$LOG"; return; fi
  echo "=== $tag :: $* ===" | tee -a "$LOG"
  timeout 1200 uv run python experiments/z_budget.py \
      --ticks "$TICKS" --contacts-per-foot 4 --out "$OUT/$tag.npz" "$@" >>"$LOG" 2>&1 \
      || echo "!! $tag FAILED" | tee -a "$LOG"
}

# -- headline: analytic vs ContactNet, straight vs turning, 3 noise seeds --------
for seed in 0 1 2; do
  for cond in "c1: --vx 0.4" "c2: --vx 0.4 --yaw 0.3"; do
    name="${cond%%:*}"; args="${cond#*: }"
    # shellcheck disable=SC2086
    run "analytic_${name}_s${seed}"   $args --imu-noise --noise-seed "$seed"
    # shellcheck disable=SC2086
    run "contactnet_${name}_s${seed}" $args --imu-noise --noise-seed "$seed" \
        --contactnet "$CKPT"
  done
done

# -- D+1: the omitted body->world rotation of the contact measurement noise -----
for seed in 0 1 2; do
  run "analytic_c2_rot_s${seed}"   --vx 0.4 --yaw 0.3 --imu-noise --noise-seed "$seed" --rotate-R
  run "contactnet_c2_rot_s${seed}" --vx 0.4 --yaw 0.3 --imu-noise --noise-seed "$seed" \
      --rotate-R --contactnet "$CKPT"
done

echo "=== matrix complete ===" | tee -a "$LOG"
