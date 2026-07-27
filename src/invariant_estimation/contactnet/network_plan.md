# ContactNet — Learned Contact Covariance MLP: Implementation Plan

Dispatch spec for building the learned contact-covariance network and its training harness.
Target stack: **JAX + Flax (or Equinox) + Optax**, float64, trained by BPTT through a differentiable InEKF.
Deployment target: weights exported to numpy → hand-implemented forward pass in Java (IHMC stack, EJML, zero-alloc, 1 kHz).

---

## 0. Open numbers — MUST be resolved before writing layer sizes

| Symbol | Meaning | Status |
|---|---|---|
| `F` | per-contact feature count (channels × subchain joints) | **UNRESOLVED** — enumerate for Alex first; `D_in = H * F` |
| `B` | segments per training batch | **UNRESOLVED** — tie to MJX vmap env count |
| `sigma_0` | shipped constant contact std (per axis), from current Java filter | **UNRESOLVED** — read from `ContactMeasurementNoiseProvider` |
| `eps` | softplus floor on Cholesky diagonal | **UNRESOLVED** — choose so `cond(S) < 1e9` holds (see §6.2) |

Everything else below is a committed value. Do not treat `D_in ≈ 600` as final; it is `20 * F`.

---

## 1. Scope and interface boundary

The network is **one replaceable noise block**. It does not touch filter state, propagation, or the measurement Jacobian.

```
differentiable InEKF  (separate module; assumed to exist or be built in parallel)
  └── contact update: S_k = H P_k Hᵀ + N̄_k
                       N̄_k = R̂ (J_C Σ_q J_Cᵀ + Σ_C) R̂ᵀ
                                                  ▲
                                        ContactNet supplies Σ_C
```

**Contract the network must satisfy:**
- Input: sensor history only. **Never** filter state (`X̂`, `P`, or anything derived from the estimate). This is an invariance requirement, not a style preference — violating it breaks group-affinity silently while the code keeps running.
- Output: `Σ_C ∈ R^{3×3}`, SPD, per contact, per tick.
- Contacts are permanent in the filter state ⇒ **fixed `N_c`**, static shapes, constant computation graph. Do not write code that adds/removes contacts.

---

## 2. Repository layout

```
contactnet/
  config.py        # frozen dataclass: all hyperparameters + open numbers
  features.py      # per-contact feature extraction + windowing  (HAND-AUTHOR: convention-bound)
  normalize.py     # frozen per-channel standardization constants
  network.py       # MLP trunk + Cholesky head
  losses.py        # l2_velocity, beta_nll
  rollout.py       # scan-based BPTT segment rollout
  train.py         # Adam + clip + warmup training loop
  export.py        # weights → numpy → Java-consumable artifact
tests/
  test_spd.py
  test_init_parity.py
  test_grad_fd.py
  test_stopgrad.py
  test_synthetic_recovery.py
```

**Hand-author vs. agent-generate.** `features.py` is convention-bound (frame placement, which subchain joints, channel ordering) with no cheap oracle → hand-author. Everything else has a decisive oracle (§6) → agent-generate and guard with the tests.

---

## 3. Architecture

### 3.1 Shape

```
per-contact input   (H=20, F)  →  flatten  →  (D_in = 20F,)
                                    ↓ Dense(256) + GELU
                                    ↓ Dense(256) + GELU
                                    ↓ Dense(6)          # Cholesky elements
                                  Σ_C ∈ R^{3×3}
```

Monotone funnel. This is an **encoder/bottleneck**, not a classifier MLP — the input is a redundant time-window of correlated channels whose intrinsic dimension is a few latent factors (slip / load / compliance). Do not widen past `D_in`.

- **Depth: 2 hidden layers.** Network depth multiplies into the 128-step BPTT gradient chain. Do not exceed 3 without an explicit sweep.
- **Width: 256** default. Sweep grid `{128, 256, 512} × {2, 3}` later; 256×2 is the entry point.
- **Per-contact independence via `vmap`.** Define the network for a single contact; `vmap` over the contact axis. One shared weight set.

### 3.2 Activation: GELU (tanh approximation) — **not** exact-erf

Use `jax.nn.gelu(x, approximate=True)` (this is JAX's default). Required, for four reasons:

1. `C¹` loss landscape through BPTT — ReLU kinks compound multiplicatively across the unroll.
2. **No gain chatter**: `Σ(t) → S → K →` state estimate at 1 kHz. A ReLU-kinked `Σ(t)` injects rate discontinuities into the balance controller. Smoothness in the inputs is a *filter* requirement.
3. Finite-difference Jacobian oracles (§6.3) are exact only away from kinks.
4. **Java portability**: `Math.tanh`/`Math.exp` exist; `erf` does not. The tanh approximation gives exact train/deploy parity.

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

### 3.3 Output head — SPD by construction

6 outputs → lower-triangular `L` → `Σ_C = L Lᵀ`.

```
L[0,0] = softplus(o[0]) + eps
L[1,1] = softplus(o[1]) + eps
L[2,2] = softplus(o[2]) + eps
L[1,0] = o[3]        # off-diagonals unconstrained
L[2,0] = o[4]
L[2,1] = o[5]
```

`eps` is tied to the filter's conditioning gate — pick it so the resulting `S` stays under `cond(S) < 1e9` (§6.2).

### 3.4 Explicitly NOT included

- **No dropout.** (i) Wrong regime — MJX gives effectively unlimited data against ~240K params. (ii) Fights BPTT — per-timestep masks inject temporally-uncorrelated noise into a recursive filter. (iii) **Biases calibration** (disqualifying): the output *is* the uncertainty, so NLL would calibrate `Σ` against the dropout-perturbed residual distribution while deployment runs dropout-off.
- **No batch normalization.** Cross-sample coupling has no physical meaning under `vmap` over contacts (contact A's covariance would depend on contact B's inputs); deployment is batch-1; running stats are state. If internal normalization is ever needed use **LayerNorm** on hidden layers only — never the head.
- **No weight decay on the head.** Trunk-only if used at all; on the head it drags `Σ` toward init and fights the `ln det` term.

---

## 4. Initialization — the network must *start at the shipped analytic filter*

This is the single most important implementation detail. At step 0, for **every** input, the network must output the shipped constant covariance. Training is then strictly a refinement from a known-good point, and every gradient step is a NEES-checkable deviation.

| Parameter | Init |
|---|---|
| Trunk weights | He/Kaiming normal (GELU gain ≈ ReLU) |
| Trunk biases | zero |
| Head weights (all 6 outputs) | **zero** |
| Head bias, diagonal (`o[0..2]`) | `log(exp(sigma_0 - eps) - 1)`  ⟵ `softplus⁻¹(sigma_0 - eps)` |
| Head bias, off-diagonal (`o[3..5]`) | **zero** |

Zero head weights ⇒ output is the bias regardless of input. Zero off-diagonals ⇒ `Σ` is diagonal at init, matching the current isotropic-per-axis assumption. Correlation structure is *earned* during training, never hallucinated at init.

Guard this with `test_init_parity.py` (§6.1) — it is a decisive, exact oracle.

---

## 5. Training

### 5.1 BPTT is not windowed subsampling — structure the data loader accordingly

- A **sample is a trajectory segment of length `L = 128`**, not a window. The gradient at step `k` carries contributions from every later step in the unroll.
- `H = 20` (history per evaluation) and `L = 128` (filter steps traversed by the gradient) are **independent axes**. Do not conflate.
- A batch of `B` segments yields `B * L * N_c` forward passes but only `B` independent samples.
- **No shuffling within a segment.** Decide and document the boundary policy: reinitialize filter state from ground truth at segment start (recommended for run 1) vs. carry state across segments (truncated BPTT).

Batched input tensor: `(B, L, N_c, H, F)`, float64.
Memory: use `jax.checkpoint` (remat) on the per-step filter update — storing activations for 128 steps × Cholesky/inverse ops will otherwise dominate.

### 5.2 Rollout

`jax.lax.scan` over the `L` axis; `vmap` over `B` and over `N_c`. Carry = filter state `(X̂, P)`. Per step: window → normalize → network → `Σ_C` → filter update → accumulate loss terms.

### 5.3 Loss — reproduce, then calibrate

**Run 1 — L2 velocity (CoCo's objective), baseline only:**
```
L_l2 = mean_k || v_filter[k] - v_true[k] ||²
```
Purpose: prove the differentiable filter and BPTT plumbing are correct, and reproduce the published result. Known limitation: `Σ` reaches the loss only through the gain, so only *ratios* are constrained — absolute scale is free, and an L2-trained `Σ` can pass RMSE while failing NEES.

**Run 2 onward — β-NLL (the real objective):**
```
L_beta = mean_k [ stop_grad(det(S_k)**beta) * 0.5 * (nu_kᵀ S_k⁻¹ nu_k + logdet(S_k)) ]
beta = 0.5
```
- The `logdet` term is what constrains absolute scale: the two terms balance at `S_k ≈ E[nu_k nu_kᵀ]`, i.e. calibration by construction. This is the term L2 structurally lacks.
- The `stop_grad(...)` reweight cancels the `S⁻¹` gradient-shrinking that otherwise makes plain NLL inflate variance to escape hard samples. **The factor must be detached** — guard with `test_stopgrad.py`.
- `beta` is the one sweep knob: 0 = pure NLL, 1 = L2-like gradient magnitude while keeping calibration.

Implementation notes: compute `nuᵀ S⁻¹ nu` and `logdet(S)` **via Cholesky of `S`**, never an explicit inverse or `det`. `logdet(S) = 2 * sum(log(diag(chol(S))))`.

Optional composite (only if pure β-NLL drifts from the L2 baseline): `L_beta + lambda * L_l2`, small `lambda`.

### 5.4 Optimizer

```
optax.chain(
    optax.clip_by_global_norm(max_norm),     # THE stability lever — gradient crosses 128 filter steps
    optax.adamw(learning_rate=warmup_cosine, weight_decay=0.0),   # decay off head; 0.0 default
)
```
- **Gradient clipping by global norm matters more than optimizer choice here.**
- **LR warmup with a low initial LR is required**, not optional: Adam's per-parameter normalization otherwise takes full-size steps the instant a zero-initialized head weight sees any gradient, destroying the §4 "refinement from a known-good point" property.

### 5.5 Input standardization

Per-channel zero-mean/unit-std, computed **once** over a calibration set and **frozen**.
- **Not** per-window — that would strip absolute magnitude, which is exactly what signals slip (a sliding foot has larger `q̇`/`τ` excursions than a planted one).
- **Not** running statistics — adds state and creates a train/deploy mismatch.
- Store as a `(F,)` mean and `(F,)` std pair; these get baked into the Java export.

---

## 6. Decisive oracles (write these before or alongside the implementation)

### 6.1 `test_init_parity.py` — exact
For random inputs at step 0, assert `Σ_C == sigma_0² * I3` to float64 tolerance (≤1e-12 relative). Additionally: run the differentiable filter with the frozen-at-init network over a logged trajectory and assert the state trajectory matches the constant-covariance reference filter to ≤1e-10. **If this fails, nothing downstream is meaningful.**

### 6.2 `test_spd.py`
For a large batch of random and adversarial (saturating, out-of-distribution) inputs: assert `Σ_C` symmetric, `eigvals > 0`, and `cond(H P Hᵀ + N̄) < 1e9` with representative `P`. Use this test to *choose* `eps` — sweep it until the conditioning bound holds with margin.

### 6.3 `test_grad_fd.py`
Finite-difference vs. autodiff gradient of the total loss w.r.t. a sampled subset of parameters, on a **short** rollout (`L = 4`), float64, central differences. Agreement to ≤1e-6 relative. GELU's smoothness makes this an exact oracle — that is part of why it was chosen.

### 6.4 `test_stopgrad.py`
Assert `grad` of `L_beta` w.r.t. parameters *through the `S**beta` factor alone* is exactly zero — i.e. the reweight is detached. Easiest form: compare gradients with the factor computed normally vs. explicitly `lax.stop_gradient`-wrapped; they must be bitwise identical.

### 6.5 `test_synthetic_recovery.py` — the strongest calibration oracle
Generate synthetic contact measurements with **known** injected Gaussian noise covariance `Σ*` (vary it as a known function of a synthetic "slip" input channel). Train with β-NLL. Assert the network recovers `Σ*` to within a stated tolerance. This validates the entire loss/rollout/gradient path against a ground truth that does not exist in the real problem — run it before touching MJX data.

### 6.6 Consistency reporting (not a unit test)
Log per-update NIS `nuᵀ S⁻¹ nu` against the χ² band (already instrumented in the Java filter, §III-B of the writeup). Report **NEES + velocity RMSE + per-contact inference time** — the three-metric table is the deliverable.

---

## 7. Java export path

- `export.py` emits: trunk/head weight matrices and biases as float arrays, the `(F,)` normalization mean/std, `eps`, and a manifest recording `H`, `F`, layer sizes, and activation variant.
- Java side: hand-implemented forward pass, **no allocation per tick** (preallocate all intermediate buffers), EJML for the small matmuls, `Math.tanh` for the GELU approximation, warmed on startup.
- Runtime inference may be float32 (it only produces a covariance the float64 filter consumes) — treat as a *measured* optimization, verify against the JAX reference before adopting.
- Cross-language oracle: same inputs → JAX and Java outputs agree to ≤1e-6.

---

## 8. Budget (design against these)

| Quantity | Value |
|---|---|
| Control loop | 1 kHz ⇒ **1 ms/tick total** for everything |
| ContactNet inference ceiling | ~0.14 ms/tick (CoCo reference) |
| MACs per contact | ~220K at `600→256→256→6` |
| MACs per tick | ~440K (2 contacts) |
| Params | ~240K, shared across contacts; <1 MB as float32 |

**Latency-bound, not size-bound.** The weights fit anywhere; the hand-ported warm forward pass has to clear its slice of the 1 ms budget at `N_c` contacts.

---

## 9. Build order

1. `config.py` + resolve the four open numbers (§0).
2. `features.py` — **hand-author**, then freeze the channel ordering and document it.
3. `normalize.py` + calibration-set statistics.
4. `network.py` + `test_init_parity.py`, `test_spd.py`. **Do not proceed until 6.1 passes.**
5. `losses.py` + `test_stopgrad.py`.
6. `rollout.py` (scan + remat) + `test_grad_fd.py`.
7. `test_synthetic_recovery.py` — full-path validation on synthetic data with known `Σ*`.
8. `train.py` — L2 baseline run against MJX data; confirm it reproduces the CoCo-style result.
9. Switch to β-NLL; report NEES/RMSE/latency.
10. Width×depth sweep `{128,256,512} × {2,3}`; select on NEES within the inference budget.
11. `export.py` + Java forward pass + cross-language oracle.

Deferred, not in scope for this sprint: mean head (contact-velocity prior — on stationary ground zero-mean *is* the baseline), unroll-length curriculum, learned `Σ_ε` for the joint-KF stance anchor.
