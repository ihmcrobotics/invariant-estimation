#!/usr/bin/env bash
# experiments/c_arms.sh — does the drift track the LOSS, across the four arms?
#
# The L2-options ablation ranked the arms on held-out velocity RMSE:
#   A l2_velocity 0.0678 | B l2_vel_pos 0.0483 | C l2_vel_ori 0.0718 | D +both 0.0456
# None of it measured VERTICAL DRIFT. If drift worsens as velocity RMSE improves,
# the objective is trading one against the other, and that is a loss finding no
# gradient-alignment number can show — alignment is local, this is about optima.
#
# The analytic heuristic is the reference: it uses no network at all.
set -uo pipefail

OUT="${1:-results/carms}"
mkdir -p "$OUT"
LOG="$OUT/arms.log"
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

declare -A ARMS=(
  [A]="results/2026-08-06_02-10-15_A_l2vel/params.npz"
  [B]="results/2026-08-06_03-35-57_B_l2velpos/params.npz"
  [C]="results/2026-08-06_05-05-58_C_l2velori/params.npz"
  [D]="results/2026-08-06_06-42-29_D_l2velposori/params.npz"
)

for seed in 0 1; do
  for arm in A B C D; do
    ck="${ARMS[$arm]}"
    [[ -f "$ck" ]] || { echo "!! missing $ck" | tee -a "$LOG"; continue; }
    run "arm${arm}_c1_s${seed}" --vx 0.4 --imu-noise --noise-seed "$seed" \
        --contactnet "$ck" --contactnet-norm "$(dirname "$ck")/norm_constants.npz"
  done
done

echo "=== arm sweep complete ===" | tee -a "$LOG"
