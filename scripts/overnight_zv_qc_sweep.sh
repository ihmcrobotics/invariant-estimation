#!/usr/bin/env bash
# Overnight 2026-08-11 diagnostic queue: process-noise sweep (lead 3), ZV + learned
# checkpoint (lead 1), and the N^v trust sweep (lead 2).
#
# STRICTLY SEQUENTIAL, every job under `gpu_lock.sh` -- N6: a second JAX process on
# this card does not run slower, it OOMs and takes the first one down.
# Clean sensors throughout (no --imu-noise): that flag is a 21% level shift and must
# never be mixed into a comparison.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=results/zv_qc_2026-08-11/closed_loop
mkdir -p "$OUT" /tmp/zvqc

run() {           # run <arm-name> <extra flags...>
  local name="$1"; shift
  if [ -s "$OUT/cmv_1e-3_$name.json" ]; then
    echo "== SKIP $name (already have it)"; return 0
  fi
  echo "== $(date +%H:%M:%S) $name  $*"
  bash scripts/gpu_lock.sh uv run --extra gpu python scripts/record_contactnet_demo.py \
    --contacts-per-foot 4 --contact-meas-var 1e-3 "$@" \
    --out "/tmp/zvqc/$name.mp4" --metrics "$OUT/cmv_1e-3_$name.json" \
    > "$OUT/$name.log" 2>&1
  echo "   exit=$? $(tail -3 "$OUT/$name.log" | head -1)"
}

# -- lead 3: untuned non-contact process noise (config gyro 1e-4, accel 1e-3) -----
# Contact NIS/dof sits at 0.005-0.010, i.e. S is far too LARGE. Sigma_C is the only
# block of Q_c anyone ever tuned; these two came from the Java config.
run base                                          # reproduces -0.117 m, now with NIS
run Qgyro0.1  --gyro-var 1e-5
run Qgyro10   --gyro-var 1e-3
run Qaccel0.1 --accel-var 1e-4
run Qaccel10  --accel-var 1e-2

# -- lead 1: zero velocity under the learned (bounded_exp) Sigma_C ----------------
# Learned stance Sigma_C is ~100x looser than analytic, which is the direction foot
# roll needs. Does that fix ZV's horizontal regression?
run bexp      --contactnet results/zdrift_bexp/L256_A_cmv1e-3_bexp/params.npz
run zvBexp    --zero-velocity \
              --contactnet results/zdrift_bexp/L256_A_cmv1e-3_bexp/params.npz

# -- lead 2: the N^v trust sweep on the analytic arm ------------------------------
# kappa=1 is the derived value (no free parameter). kappa=1e6 is the graceful
# degradation check: it must come back to `base`.
run zvK1      --zero-velocity --nv-scale 1
run zvK0.1    --zero-velocity --nv-scale 0.1
run zvK10     --zero-velocity --nv-scale 10
run zvK100    --zero-velocity --nv-scale 100
run zvK1e6    --zero-velocity --nv-scale 1e6

echo "== $(date +%H:%M:%S) queue done"
uv run python scripts/cl_floor_summary.py --dir "$OUT"
