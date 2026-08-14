#!/usr/bin/env bash
# Fourth overnight batch: extend the N^v trust sweep to the PHYSICALLY MOTIVATED
# range, and redo the one run lost to a mid-flight edit.
#
# Why the sweep needs points above kappa = 100, and a CORRECTION to the premise.
#
# The night's brief said the analytic stance Sigma_C asserts a contact slip of
# 3.2e-3 m/s against a measured foot roll of 0.07-0.20 m/s, i.e. that kappa = 1 is
# 20-60x overconfident and the learned (looser) Sigma_C is the fix. **That is wrong
# for the deployed configuration.** It uses `stance_chol^2 = 1e-8` and omits
# `contact_floor`, which is ADDITIVE and 10000x larger:
#
#     Sigma_C = stance_chol^2 + contact_floor = 1e-8 + 1e-4 = 1.0001e-4  m^2/s
#     N^v slip sigma = sqrt(Sigma_C / dt) = sqrt(1.0001e-4 / 5e-3) = 0.1414 m/s
#
# (verified against `inEKF.contact.digest` with the shipped config, not derived on
# paper). 0.1414 m/s sits at the TOP of the measured 0.07-0.20 m/s roll range, so
# kappa = 1 is already at the physically right magnitude and the sweep's job is not
# to find the right trust level -- it is to establish whether ANY trust level makes
# the trade favourable. kappa -> inf must reproduce `base` (-0.1165 / 0.060), which
# `zvK1e6` checks, so these fill in the fade between the two known endpoints.
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

run zvK1e3  --zero-velocity --nv-scale 1e3    # sigma x32 past the measured roll
run zvK1e4  --zero-velocity --nv-scale 1e4    # sigma x100: most of the way to off

# Lost to a mid-flight edit of record_contactnet_demo.py, not to anything physical.
run zvBexp  --zero-velocity \
            --contactnet results/zdrift_bexp/L256_A_cmv1e-3_bexp/params.npz

echo "== $(date +%H:%M:%S) kappa-hi batch done"
