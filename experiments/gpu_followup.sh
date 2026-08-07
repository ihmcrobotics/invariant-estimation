#!/usr/bin/env bash
# experiments/gpu_followup.sh — the GPU sequence that runs once the ceiling lands.
#
# 1. END-TO-END check of the ceiling params. The ceiling optimises a 128 ms
#    segment DC; whether that transfers to 30 s of closed loop is exactly the
#    question, so the proxy is never trusted on its own.
# 2. Sigma_C on the TRAINING distribution — is the 20x stance loosening learned,
#    or only an artifact of deployment?
# 3. THE RATE TEST. ContactNet is trained at 1 kHz (run_contactnet sets
#    rp.DT=0.001) and deployed at 200 Hz (run_estimator leaves rp.DT=0.005), so
#    its H=20 window spans 19 ms in training and 95 ms in deployment and its `v`
#    channel is a first difference over a 5x longer step. If ContactNet only
#    loses to the analytic heuristic at 200 Hz, the whole result is a deployment
#    mismatch rather than a loss problem.
set -uo pipefail
cd "$(dirname "$0")/.."

ARMD=results/2026-08-06_06-42-29_D_l2velposori
NORM="$ARMD/norm_constants.npz"
OUT=results/gpufollow
mkdir -p "$OUT"
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"; mkdir -p "$TMPDIR"

while pgrep -f "z_authority.py --mode ceiling" >/dev/null 2>&1; do sleep 20; done

echo "### $(date +%H:%M:%S) 1/4 end-to-end check of the ceiling params"
if [[ -f results/zauth_ceiling.npz ]]; then
  uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
    --imu-noise --noise-seed 0 --contactnet results/zauth_ceiling.npz \
    --contactnet-norm "$NORM" --out "$OUT/ceiling_c1_s0.npz" > "$OUT/ceiling_e2e.log" 2>&1 \
    || echo "!! ceiling e2e FAILED"
else
  echo "!! results/zauth_ceiling.npz absent — ceiling did not write params"
fi

echo "### $(date +%H:%M:%S) 2/4 Sigma_C on the training distribution"
uv run python experiments/z_authority.py --mode sigma --batches 12 \
  > "$OUT/sigma_train.log" 2>&1 || echo "!! sigma mode FAILED"

echo "### $(date +%H:%M:%S) 3/4 rate test at 1 kHz (the ContactNet training rate)"
for arm in "analytic:" "contactnet:--contactnet $ARMD/params.npz --contactnet-norm $NORM"; do
  name="${arm%%:*}"; extra="${arm#*:}"
  # shellcheck disable=SC2086
  uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
    --imu-noise --noise-seed 0 --khz --out "$OUT/${name}_khz_s0.npz" $extra \
    > "$OUT/${name}_khz.log" 2>&1 || echo "!! ${name} khz FAILED"
done

echo "### $(date +%H:%M:%S) 4/4 gait control: policy reads TRUTH, so both arms walk identically"
for arm in "analytic:" "contactnet:--contactnet $ARMD/params.npz --contactnet-norm $NORM"; do
  name="${arm%%:*}"; extra="${arm#*:}"
  # shellcheck disable=SC2086
  uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
    --imu-noise --noise-seed 0 --source truth --out "$OUT/${name}_truthsrc_s0.npz" $extra \
    > "$OUT/${name}_truthsrc.log" 2>&1 || echo "!! ${name} truthsrc FAILED"
done

echo "### $(date +%H:%M:%S) gpu followup complete"
