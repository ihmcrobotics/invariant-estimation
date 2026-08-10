#!/usr/bin/env bash
# Stage 2 after the sweep: settle the cancellation question, THEN retrain.
#
# The sweep found drift crossing zero between 3e-5 and 1e-4 (final e_z: +0.020/+0.061
# at 3e-5, -0.056/-0.074 at 1e-4). A floor on that crossing gives a small mean and is
# a CANCELLATION, not a fix -- the 2026-08-07 study measured exactly this parameter
# doing exactly that, its flat-ground optimum drifting upward on terrain. Selecting on
# |mean| walks into it, so: re-evaluate the two candidates recording PER-TERRAIN
# values, disqualify any floor whose terrains disagree on sign, and only then spend
# six hours retraining.
#
# The sweep also showed the two goals pulling apart: NIS/dof is monotone in the floor
# (0.272, 0.059, 0.037, 0.026 as it rises) so consistency wants the floor at ZERO,
# while drift magnitude wants it near 3e-5. Even the best NIS is 3.7x from target.
# That tension is the headline result, not a detail.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/zdrift
LOG=$OUT/zdrift.log
log(){ echo "[$(date +%F' '%H:%M:%S)] [stage2] $*" | tee -a "$LOG"; }
GPU="bash scripts/gpu_lock.sh uv run --extra gpu python"
CELLS="L256_A_l2vel L256_C_l2velori"
STEPS=6000

# re-evaluate the two candidate floors WITH per-terrain detail
for F in 3e-5 1e-4; do
  log "per-terrain re-evaluation at contact_meas_var=${F}"
  $GPU scripts/drift_backfill.py --root results/l_ablation --pool n8fix \
     --only ${CELLS} --contact-meas-var "${F}" \
     --out "$OUT/sweep_cmv_${F}.json" 2>&1 | tee -a "$LOG"
done

log "--- sweep table with the sign-consistency column ---"
uv run python scripts/zdrift_summary.py 2>&1 | tee -a "$LOG"
uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, pathlib
for p in sorted(glob.glob("results/zdrift/sweep_cmv_*.json")):
    for r in json.loads(pathlib.Path(p).read_text()):
        per = r.get("per_rollout")
        if not per: continue
        print(f"{r['contact_meas_var']:.0e}  {r['cell']:16s} " +
              "  ".join(f"{e['name'].split('/')[0][:12]}:{e['final_ez']:+.3f}" for e in per))
PY

BEST=$(uv run python scripts/zdrift_summary.py --best 2>/dev/null | tail -1)
log "selected contact_meas_var=${BEST} (mixed-sign floors disqualified)"
[ -z "$BEST" ] && { log "!! no eligible floor; stopping"; exit 2; }

declare -A OBJ=( [L256_A_l2vel]=l2_velocity [L256_C_l2velori]=l2_vel_ori )
for cell in ${CELLS}; do
  dst="$OUT/${cell}_cmv${BEST}"
  [ -f "${dst}/summary.json" ] && { log "retrain ${cell}: present, skipping"; continue; }
  log "retrain ${cell} (${OBJ[$cell]}) at L=256, contact_meas_var=${BEST}"
  $GPU scripts/run_contactnet.py --pool n8fix --contacts-per-foot 4 \
     --objective "${OBJ[$cell]}" --L 256 --no-remat --contact-meas-var "${BEST}" \
     --steps ${STEPS} --warmup-steps 100 --time-budget-s 86400 \
     --out-dir "${dst}" 2>&1 | tee -a "$LOG"
  ran=$(uv run python -c "
import json
try:  print(json.load(open('${dst}/summary.json'))['steps_run'])
except Exception: print(-1)" 2>/dev/null | tail -1)
  [ "$ran" = "${STEPS}" ] && log "retrain ${cell}: complete" || log "!! retrain ${cell} FAILED (${ran})"
done

log "final scoring of the retrained nets at contact_meas_var=${BEST}"
$GPU scripts/drift_backfill.py --root "$OUT" --pool n8fix \
   --contact-meas-var "${BEST}" --out "$OUT/drift_retrained.json" 2>&1 | tee -a "$LOG"
log "=== stage 2 complete ==="
