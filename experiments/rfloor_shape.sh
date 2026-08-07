#!/usr/bin/env bash
# experiments/rfloor_shape.sh — is contact_meas_var=1e-4 a knee or just the first value tried?
#
# 1e-4 was picked because it is what flight's ConstantContactMeasurementNoiseProvider
# uses, and it measured +15.5% (analytic) / +31.1% (ContactNet) at n=3. Before that
# goes into a config recommendation, the response shape matters: if drift keeps
# improving to 1e-3 or 1e-2 the recommendation is wrong, and if 1e-5 already captures
# most of it the value is not critical. Run at the dwell we would actually ship (80 ms)
# so the answer applies to the recommended configuration, not the current one.
# 1e-4 at dwell 0.08 already exists in results/combo (n=3).
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p results/rshape "$TMPDIR"
for v in 1.0e-5 1.0e-3 1.0e-2; do
  for seed in 0 1; do
    out="results/rshape/analytic_dw0.08_rf${v}_s${seed}.npz"
    [[ -f "$out" ]] && continue
    echo "### $(date +%H:%M:%S) rf=$v seed=$seed"
    uv run python experiments/z_budget.py --ticks 1500 --vx 0.4 --contacts-per-foot 4 \
      --imu-noise --noise-seed "$seed" --contact-source trust --dwell 0.08 \
      --contact-meas-var "$v" --out "$out" >> results/rshape/rshape.log 2>&1 \
      || echo "!! rf=$v s=$seed FAILED"
  done
done
echo "### $(date +%H:%M:%S) rfloor shape sweep done"
