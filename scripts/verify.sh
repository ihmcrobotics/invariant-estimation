#!/usr/bin/env bash
set -euo pipefail

# Fork point of contactnet/coco-faithful-features off contactnet/take-two.
FORK=87f1987dd09a6e7a45f35515c767db720b3785b5

# 1. No test was touched to force a pass (prohibited).
if git diff --name-only "$FORK" | grep -q '^tests/'; then
  echo "FAIL: files under tests/ were modified"; exit 1
fi

# 2. Measurement socket stays zero -- process socket only.
if grep -rn "contact_meas_chol" src/invariant_estimation/pipeline src/invariant_estimation/contactnet \
     | grep -v "zeros" | grep -v "#" | grep -q .; then
  echo "FAIL: contact_meas_chol referenced non-zero outside a zeros/comment"; exit 1
fi

# 3. No filter state leaked into the feature builders (invariance).
if grep -Enr "state\.|carry\.|\.P\b|q_hat|qhat|xhat|x_hat" \
     src/invariant_estimation/contactnet/features.py \
     src/invariant_estimation/contactnet/online.py | grep -q .; then
  echo "FAIL: a filter-state symbol appears in features.py/online.py"; exit 1
fi

# 4. Feature geometry is CoCo-faithful: F=30, d_in=600, stride=1.
uv run python - <<'PY'
from invariant_estimation.contactnet.config import ContactNetConfig
from invariant_estimation.contactnet import features
cfg = ContactNetConfig()          # defaults are the training config
assert cfg.F == 30,        f"F={cfg.F}, expected 30"
assert cfg.d_in == 600,    f"d_in={cfg.d_in}, expected 600"
assert cfg.stride == 1,    f"stride={cfg.stride}, expected 1 (window not consecutive)"
names = features.channel_names()
assert len(names) == 30,   f"channel_names has {len(names)}, expected 30"
qd = [i for i,n in enumerate(names) if n.startswith("qd_")]
q  = [i for i,n in enumerate(names) if n.startswith("q_")]
tau= [i for i,n in enumerate(names) if n.startswith("tau_")]
assert q and qd and tau and max(q) < min(qd) < max(qd) < min(tau), "q/q-dot/tau order wrong"
print("OK: geometry F=30, d_in=600, stride=1, channel order q,qd,tau")
PY

# 5. online.py and features.py agree (the load-bearing oracle).
uv run python -m pytest -q tests/contactnet/test_online.py

# 6. Normalization has a floor for every channel (no KeyError), incl. qd_.
uv run python - <<'PY'
from invariant_estimation.contactnet import features, normalize
normalize.channel_floor(features.channel_names())   # raises if any channel unfloored
print("OK: every channel has a noise floor")
PY

echo "verify.sh: all Layer-1 checks passed"
