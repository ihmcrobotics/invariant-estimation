#!/usr/bin/env bash
# experiments/d_levers.sh — Track D: the (d) candidates the question did not list.
#
#   contact_floor  — ADDITIVE (inEKF/contact.py:91) and shipped at 1e-4, while the
#                    analytic stance signal is (1e-4)^2 = 1e-8. Measured: stance
#                    Sigma_C trace is exactly 3.0e-4, i.e. 100% floor. If Sigma_C
#                    is a binary switch between the floor and the swing value, then
#                    ContactNet's expressiveness in stance is irrelevant and no
#                    objective over it can matter. Sweeping the floor says how much
#                    authority is being suppressed.
#   contact_meas_var — shipped 0.0, so R = J Sigma_q J^T ~ 1e-8..1e-6 while
#                    H P H^T is ~1e-4. S is then dominated by P and the update is
#                    near a hard constraint, compressing Sigma_C's leverage further.
#   x0             — the world-origin rotation lever, re-measured under ContactNet.
set -uo pipefail

OUT="${1:-results/dlevers}"
mkdir -p "$OUT"
LOG="$OUT/levers.log"
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

# -- D1: how much of Sigma_C's authority does the additive floor suppress? ------
for f in 1.0e-3 1.0e-5 1.0e-6; do
  run "contactnet_floor${f}_s0" --vx 0.4 --imu-noise --noise-seed 0 \
      --contact-floor "$f" --contactnet "$CKPT"
  run "analytic_floor${f}_s0"   --vx 0.4 --imu-noise --noise-seed 0 \
      --contact-floor "$f"
done

# -- D2: give R a floor, so S stops being pure H P H^T --------------------------
for v in 1.0e-6 1.0e-4; do
  run "contactnet_rfloor${v}_s0" --vx 0.4 --imu-noise --noise-seed 0 \
      --contact-meas-var "$v" --contactnet "$CKPT"
done

# -- D4: the world-origin rotation lever, under ContactNet this time ------------
run "contactnet_x50_s0" --vx 0.4 --imu-noise --noise-seed 0 --x0 50 --contactnet "$CKPT"
run "analytic_x50_s0"   --vx 0.4 --imu-noise --noise-seed 0 --x0 50

echo "=== lever sweep complete ===" | tee -a "$LOG"
