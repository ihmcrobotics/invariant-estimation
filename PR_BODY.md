# ContactNet: CoCo-faithful features (process socket, q̇ / F=30, stride=1) + a validated training run

Base: **`contactnet/take-two`** (do not merge to `main`). Cut from `8052c02`.

Gets ContactNet training end-to-end on the corrected feature representation and
produces a held-out number: **on held-out flat rollouts, the learned Σ_C cuts
body-frame velocity RMSE ~3× vs the analytic-heuristic `contact_chol` baseline
(0.0857 → 0.0283 m/s) and moves velocity NEES from 19.4 (overconfident) to 1.05
(target 3).** Flat terrain only — see Caveats.

## Headline validation (2 disjoint held-out flat rollouts, train seeds 0–3 / val 4–5)

| metric (target)              | analytic baseline | learned | Δ |
|------------------------------|-------------------|---------|---|
| body-frame velocity RMSE m/s | 0.0857            | 0.0283  | ~3× / −67% |
| velocity NEES (→ 3)          | 19.35             | 1.05    | overconfident → mildly conservative |
| contact NIS/dof (→ 1)        | 0.0654            | 0.0261  | both under-confident (follow-up) |

Training loss 0.0815 → 0.000958 over 300 steps; reseeds 91; `floored=[]`; wall ~2883 s.

![training](https://github.com/ihmcrobotics/invariant-estimation/raw/contactnet/coco-faithful-features/results/training.png)
![validation](https://github.com/ihmcrobotics/invariant-estimation/raw/contactnet/coco-faithful-features/results/validation.png)

## Commits → phases

| commit | phase |
|---|---|
| `8d8a36f` rebuild `ContactNetConfig` (F=30, H=20, stride=1) | Phase 1.6 + 2 |
| `5d19850` fix `features.window_indices` broadcast `(H,1)→(1,H)` | Phase 2 (bug) |
| `d36fba3` restore `torques` channel (FusedSensors + sensors.read) | Phase 1 (restore) |
| `6ccfbae` `scripts/verify.sh` Layer-1 gate | Verification |
| `d8f2524` restore `FusedOutputs.inekf_inputs` | Phase 3 (restore) |
| `dc3c6b1` port `contactnet/train.py` | Phase 4 |
| `ce3fcb9` port `sim/collect.py` (flat-only) | Phase 3 |
| `1a64e36` port `contactnet/dataset.py` | Phase 3 |
| `e2b16ee` add `optax` | env |
| `25a7300` batched `l2_velocity`; `enc_dofadr`; process-socket load; orchestrator | Phase 4 (bug) |
| `78fad63` RESULTS.md + report + run artifacts | deliverables |

`take-two` had trimmed away working code the plan assumed present; each restore is
faithful to the reference (`contact-net-integration`), committed separately for bisect.

## Key ports / fixes (permalinks)

- config F=30: [`config.py` L16](https://github.com/ihmcrobotics/invariant-estimation/blob/78fad632908f85b2844e6aa4280e2ef09bef8160/src/invariant_estimation/contactnet/config.py#L16)
- `l2_velocity` batched CoCo form (was non-batched `einsum("ji,j->i")+norm`, raised on the segment arrays): [`losses.py` L6–L24](https://github.com/ihmcrobotics/invariant-estimation/blob/78fad632908f85b2844e6aa4280e2ef09bef8160/src/invariant_estimation/contactnet/losses.py#L6-L24)
- `window_indices` broadcast fix: [`features.py` L30](https://github.com/ihmcrobotics/invariant-estimation/blob/78fad632908f85b2844e6aa4280e2ef09bef8160/src/invariant_estimation/contactnet/features.py#L30)
- process-socket wiring — ContactNet drives `InEKFInputs.contact_chol` only: [`rollout.py` L82](https://github.com/ihmcrobotics/invariant-estimation/blob/78fad632908f85b2844e6aa4280e2ef09bef8160/src/invariant_estimation/contactnet/rollout.py#L82) (and take-two's `InEKFInputs` has no `contact_meas_chol` field, so the measurement socket is structurally zero)
- `FusedOutputs.inekf_inputs` restore (collection needs the recorded inputs): [`main_estimator.py` L635](https://github.com/ihmcrobotics/invariant-estimation/blob/78fad632908f85b2844e6aa4280e2ef09bef8160/src/invariant_estimation/pipeline/main_estimator.py#L635)

## Gate output

G1 geometry + online oracle:
```
OK: geometry F=30, d_in=600, stride=1, channel order q,qd,tau; len(channel_names)==30
online↔offline (F=30): stride=8 rel_err 6.9e-15, stride=1 rel_err 1.4e-16,
    provider==offline network 1.9e-15
```

`verify.sh` (fork base `87f1987` as instructed): checks 4 & 6 PASS; check 2 is a
false-positive on a `measure_p0` **docstring** that states the invariant (take-two's
`InEKFInputs` has no `contact_meas_chol` field); check 3 is a false-positive on the
`OnlineState` ring buffer (`state.prev_p/n/buf`, sensor-history only, identical to
reference); check 1 flags take-two's *prior* test edits (against the true cut point
`8052c02`, this branch touched **zero** tracked tests); check 5 is the stale gate below.
Full analysis in `RESULTS.md`.

## Caveats (do not oversell)

- **Flat terrain only** — the flat-only `collect.py` port dropped terrain / domain
  randomisation; cross-condition generalisation is **untested**. Claim is "beats the
  analytic baseline on held-out flat rollouts," not "beats CoCo across conditions."
- **Velocity NEES 1.05** = learned is mildly *conservative*; baseline 19.4 = severely
  *overconfident*.
- **Contact NIS/dof under-confident for both** (0.026 / 0.065, target 1) — contact
  covariance looks inflated; open follow-up (l2_velocity constrains only Σ *ratios*,
  so absolute NIS calibration likely needs β-NLL).
- **300 steps** is a wall-clock cap, not a measured RMSE plateau; reseeds climb
  ~linearly to 91 (episode boundaries), divergence not proven absent.
- **`tests/contactnet/test_online.py` is RED by construction** — the untracked gate
  file is a byte-identical copy of the reference's pre-q̇ (F=24) test (`_cfg` sets
  `F=12+2·J_SUB=24`, asserts `len(channel_names())==cfg.F`, and `_sensors` never sets
  `encoders_vel`). Incompatible with the mandated F=30 set; **not edited** per the
  rules. One-line human fix: `2*J_SUB → 3*J_SUB` + populate `encoders_vel`. The
  property it guards is verified independently at F=30 (above).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
