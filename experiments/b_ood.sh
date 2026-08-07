#!/usr/bin/env bash
# experiments/b_ood.sh — hypothesis (b): is the training DR too narrow?
#
# The discriminator is not "does drift get worse out of distribution" — the whole
# filter may degrade. It is whether ContactNet degrades MORE than the analytic
# heuristic does under the same insult. Equal degradation means the filter is
# what struggles; a widening gap means the network failed to generalise, which is
# what "not enough DR" actually predicts.
#
# Axes chosen for a VERTICAL drift question:
#   friction  — pool sampled [0.45, 1.2]; 0.30 and 1.60 sit outside it
#   payload   — never randomised at all
#   push-z    — never randomised at all (training pushes are horizontal-only)
set -uo pipefail

OUT="${1:-results/bood}"
mkdir -p "$OUT"
LOG="$OUT/ood.log"
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

for cell in "indist:" \
            "mu030:--friction 0.30" "mu160:--friction 1.60" \
            "load05:--payload 5.0"  "load10:--payload 10.0" \
            "pushz060:--push-z 60"  "pushz120:--push-z 120"; do
  name="${cell%%:*}"; args="${cell#*:}"
  # shellcheck disable=SC2086
  run "analytic_${name}_s0"   --vx 0.4 --imu-noise --noise-seed 0 $args
  # shellcheck disable=SC2086
  run "contactnet_${name}_s0" --vx 0.4 --imu-noise --noise-seed 0 $args --contactnet "$CKPT"
done

echo "=== OOD grid complete ===" | tee -a "$LOG"
