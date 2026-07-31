#!/usr/bin/env bash
# One serial queue for every `run_estimator.py` / renderer job of the night.
#
# WHY A SINGLE QUEUE. Three concurrent `run_estimator.py` saturate 20 cores, and
# `run_policy.cycloid_forearm_urdf` writes a URDF to a FIXED temp path, so concurrent runs also
# race on it and die with a ParseError. Chaining "wait until no run_estimator is running" between
# separate launchers is NOT safe either -- there is a sub-second gap between arms in which the
# check passes. One script, one job at a time, each with its own TMPDIR.
set -u
cd "$(dirname "$0")/.."
mkdir -p artifacts/video artifacts/sweep2_vx06

BASE="--policy baseline --headless --ticks 1500 --imu-noise --toe-heel"
F="--early-release-source foot"

job () {   # $1 = log path, rest = run_estimator flags
  log=$1; shift
  tmp=$(mktemp -d)
  echo "=== $log : $*"
  TMPDIR=$tmp uv run python run_estimator.py $BASE "$@" > "$log" 2>&1
  grep -E "signed dz|tilt error|attitude error|base velocity|base position|VERTICAL" "$log" \
    || tail -5 "$log"
  rm -rf "$tmp"
}

# 1. The best arm known when this was queued (per-FOOT load source, 1.81x on the sink),
#    rendered FIRST so a partial night still leaves a watchable deliverable.
job artifacts/video/B_earlyrelease_foot05_vx06.log --vx 0.6 --early-release 0.5 $F \
    --ghost --video artifacts/video/B_earlyrelease_foot05_vx06.mp4 --video-fps 25 \
    --out artifacts/video/B_earlyrelease_foot05_vx06.npz

# 2. The rest of sweep 2: everything on the per-foot source.
job artifacts/sweep2_vx06/F_frac03.log      --vx 0.6 --early-release 0.3 $F --out artifacts/sweep2_vx06/F_frac03.npz
job artifacts/sweep2_vx06/F_frac07.log      --vx 0.6 --early-release 0.7 $F --out artifacts/sweep2_vx06/F_frac07.npz
job artifacts/sweep2_vx06/F_frac05_prev.log --vx 0.6 --early-release 0.5 $F --early-release-mode prev --out artifacts/sweep2_vx06/F_frac05_prev.npz
job artifacts/sweep2_vx06/CLK100.log        --vx 0.6 --early-release 0.0 --early-release-lead 100 $F --out artifacts/sweep2_vx06/CLK100.npz
job artifacts/sweep2_vx06/CLK100_F05.log    --vx 0.6 --early-release 0.5 --early-release-lead 100 $F --out artifacts/sweep2_vx06/CLK100_F05.npz
job artifacts/sweep2_vx06/CLK150_F05.log    --vx 0.6 --early-release 0.5 --early-release-lead 150 $F --out artifacts/sweep2_vx06/CLK150_F05.npz

# 3. Turning walk, the harder case dr5 was built for: baseline and the same fix.
job artifacts/video/A_analytic_n4_yaw08.log --vx 0.4 --yaw 0.8 \
    --ghost --video artifacts/video/A_analytic_n4_yaw08.mp4 --video-fps 25 \
    --out artifacts/video/A_analytic_n4_yaw08.npz
job artifacts/video/B_earlyrelease_foot05_yaw08.log --vx 0.4 --yaw 0.8 --early-release 0.5 $F \
    --ghost --video artifacts/video/B_earlyrelease_foot05_yaw08.mp4 --video-fps 25 \
    --out artifacts/video/B_earlyrelease_foot05_yaw08.npz

# 4. The best trained checkpoint, same everything else -- the third arm of the visual comparison.
job artifacts/video/C_contactnet_run7_vx06.log --vx 0.6 \
    --contactnet artifacts/contactnet_run7.npz --contactnet-norm data/dr5/norm_constants.npz \
    --ghost --video artifacts/video/C_contactnet_run7_vx06.mp4 --video-fps 25 \
    --out artifacts/video/C_contactnet_run7_vx06.npz

echo "=== QUEUE DONE"
