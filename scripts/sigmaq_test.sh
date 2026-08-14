#!/usr/bin/env bash
# Is the contact over-coverage a STRUCTURED Sigma_q error, and can an isotropic knob
# fix it?
#
# Measured: joint-level NEES 48.5 vs a target of 9 -- the joint KF is overconfident
# ~5.4x -- and the error is structured, per-joint e^2/sigma^2 spanning 0.56 to 6.41.
# `contact_meas_var` is ISOTROPIC, and invariant I9 is "per-joint noise scaling, never
# uniform". So: correct Sigma_q three ways at FLOOR ZERO and compare.
#
#   asis      the recorded Sigma_q                          (control)
#   uniform   x5.39, matching the aggregate NEES            (what an isotropic knob can do)
#   perjoint  the measured per-joint ratios                 (what I9 asks for)
#
# If per-joint fixes NIS and uniform does not, the floor was never the right shape and
# the fix belongs in the joint KF's encoder-variance calibration, not in a contact
# noise floor. Evaluation-only, so no retraining: Sigma_q lives in the replayed inputs.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/zdrift; LOG=$OUT/zdrift.log
log(){ echo "[$(date +%F' '%H:%M:%S)] [sigmaq] $*" | tee -a "$LOG"; }
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"
CELLS="L256_A_l2vel L256_C_l2velori"
PERJOINT="2.3946 1.5288 1.1189 6.4114 1.5357 1.6054 3.4254 5.4561 0.5571"

log "control: recorded Sigma_q, floor 0"
$GPU scripts/drift_backfill.py --root results/l_ablation --pool n8fix --only ${CELLS} \
   --contact-meas-var 0 --out "$OUT/sigmaq_asis.json" 2>&1 | tee -a "$LOG"

log "uniform x5.39 (what an isotropic correction can achieve), floor 0"
$GPU scripts/drift_backfill.py --root results/l_ablation --pool n8fix --only ${CELLS} \
   --contact-meas-var 0 --sigma-q-scale 5.39 --out "$OUT/sigmaq_uniform.json" 2>&1 | tee -a "$LOG"

log "per-joint measured ratios (what I9 asks for), floor 0"
$GPU scripts/drift_backfill.py --root results/l_ablation --pool n8fix --only ${CELLS} \
   --contact-meas-var 0 --sigma-q-scale ${PERJOINT} --out "$OUT/sigmaq_perjoint.json" 2>&1 | tee -a "$LOG"

log "--- Sigma_q correction: drift and consistency ---"
uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
print(f"{'variant':10s} {'cell':17s} {'|drift_z|':>10s} {'final e_z':>10s} "
      f"{'NIS/dof':>9s} {'analytic NIS':>13s}  sign")
for v in ("asis", "uniform", "perjoint"):
    p = pathlib.Path(f"results/zdrift/sigmaq_{v}.json")
    if not p.exists():
        continue
    for r in json.loads(p.read_text()):
        per = r.get("per_rollout") or []
        agree = "yes" if len({e["final_ez"] > 0 for e in per}) == 1 else "MIXED"
        print(f"{v:10s} {r['cell']:17s} {abs(r['drift_z']):10.5f} {r['final_ez']:+10.3f} "
              f"{r['nis_over_dof']:9.3f} {r['base_nis_over_dof']:13.3f}  {agree}")
print("\nNIS/dof target is 1. If per-joint moves it and uniform does not, an isotropic")
print("floor cannot represent this error and the fix belongs upstream (I9).")
PY
log "=== sigma_q test complete ==="
