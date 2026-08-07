#!/usr/bin/env bash
# experiments/combo_check.sh — do the two config levers compose?
#
# The report recommends two config-only changes: dwell 40->80 ms (worth ~3x) and
# contact_meas_var 0 -> 1e-4 (worth ~16% analytic / 31% ContactNet, n=3). Both are
# measured ALONE against the shipped baseline. Since they act on different parts of
# the same update -- the dwell decides WHEN the contact update runs, the R floor how
# hard it pulls when it does -- they might compose, or the R floor's gain might be
# entirely an artefact of trusting contact too early. Shipping advice should rest on
# the combination that would actually be shipped.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/combo "$TMPDIR"
for seed in 0 1 2; do
  out="results/combo/analytic_dw0.08_rf1e-4_s${seed}.npz"
  [[ -f "$out" ]] && continue
  echo "### $(date +%H:%M:%S) combo seed=$seed"
  uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
    --imu-noise --noise-seed "$seed" --contact-source trust --dwell 0.08 \
    --contact-meas-var 1.0e-4 --out "$out" >> results/combo/combo.log 2>&1 \
    || echo "!! combo s=$seed FAILED"
done
echo "### $(date +%H:%M:%S) combo done"
