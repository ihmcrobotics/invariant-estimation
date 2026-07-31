#!/usr/bin/env bash
# Second closed-loop sweep: the per-FOOT load source won the first one (1.81x on the sink),
# so every arm here uses it and varies what else is on top. Serial, one TMPDIR each --
# see `early_release_sweep.sh` for why.
set -u
OUT=${1:-artifacts/sweep2}
EXTRA=${2:-"--vx 0.6"}
mkdir -p "$OUT"
BASE="--policy baseline --headless --ticks 1500 --imu-noise --toe-heel $EXTRA"

run () {
  name=$1; shift
  tmp=$(mktemp -d)
  echo "=== $name : $*"
  TMPDIR=$tmp uv run python run_estimator.py $BASE "$@" --out "$OUT/$name.npz" \
      > "$OUT/$name.log" 2>&1
  grep -E "signed dz|tilt error|base velocity|base position|attitude error" "$OUT/$name.log"
  rm -rf "$tmp"
}

F="--early-release-source foot"
run F_frac03      --early-release 0.3 $F
run F_frac05_prev --early-release 0.5 $F --early-release-mode prev
run F_frac07      --early-release 0.7 $F
run CLK100        --early-release 0.0 --early-release-lead 100 $F
run CLK100_F05    --early-release 0.5 --early-release-lead 100 $F
run CLK150_F05    --early-release 0.5 --early-release-lead 150 $F
