#!/usr/bin/env bash
# Serial closed-loop sweep of `run_estimator.py --early-release`.
#
# SERIAL BY CONSTRUCTION: three concurrent `run_estimator.py` saturate 20 cores and make the
# machine unusable, and `run_policy.cycloid_forearm_urdf` writes to a FIXED temp path, so
# concurrent runs also race on it. Each arm gets its own TMPDIR and they run one at a time.
#
#   ./experiments/early_release_sweep.sh artifacts/video/sweep "--vx 0.6"
#
# Arg 1: output prefix (a directory is created). Arg 2: extra flags shared by every arm.
set -u
OUT=${1:-artifacts/sweep}
EXTRA=${2:-"--vx 0.6"}
mkdir -p "$OUT"
BASE="--policy baseline --headless --ticks 1500 --imu-noise --toe-heel $EXTRA"

run () {   # $1 = arm name, rest = arm-specific flags
  name=$1; shift
  tmp=$(mktemp -d)
  echo "=== $name : $*"
  TMPDIR=$tmp uv run python run_estimator.py $BASE "$@" \
      --out "$OUT/$name.npz" > "$OUT/$name.log" 2>&1
  tail -n 12 "$OUT/$name.log" | grep -E "signed dz|tilt error|base velocity|base position|VERTICAL"
  rm -rf "$tmp"
}

run A_baseline
run E_frac05 --early-release 0.5
run E_frac03 --early-release 0.3
run E_frac07 --early-release 0.7
run E_prev05 --early-release 0.5 --early-release-mode prev
run E_foot05 --early-release 0.5 --early-release-source foot
run E_blank75 --early-release 0.5 --early-release-blank 75
run E_blank250 --early-release 0.5 --early-release-blank 250
