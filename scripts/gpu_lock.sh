#!/usr/bin/env bash
# Serialize GPU work. Usage:  scripts/gpu_lock.sh <command...>
#
# Why this exists: JAX preallocates ~75% of device memory per process
# (XLA_PYTHON_CLIENT_MEM_FRACTION default), so a second JAX process on this 12 GB
# card does not run slower -- it OOMs, and it takes the FIRST one down with it. On
# 2026-08-09 a drift evaluation launched alongside a training cell killed the cell
# 2.5 minutes in AND then OOM'd itself, losing the rest of the night's queue.
#
# The failure is silent-ish and asymmetric: the victim is whichever process next
# tries to instantiate a CUDA graph, not the one that over-committed. So "I'll just
# be careful" is not a control. This is.
#
# flock blocks rather than failing, so a queued job waits its turn instead of dying.
set -uo pipefail
exec flock /tmp/alex_gpu.lock "$@"
