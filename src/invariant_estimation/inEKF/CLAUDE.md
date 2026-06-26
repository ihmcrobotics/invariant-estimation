# Contact-Aided Invariant EKF — Design Document

> Design record for the **world-centric, right-invariant** contact-aided InEKF
> that fuses IMU + forward-kinematics into a pose/velocity/contact estimate on
> `SE_{N+2}(3)`. This is the *main filter*; the joint KF (`joint_kf/`) is its
> pre-filter, and an external ContactNet module supplies its contact-velocity
> covariances.
>
> **Scope:** the `inekf/` package only. ContactNet (the MLP and everything it
> needs) and the BPTT training loop are **separate modules documented elsewhere**
> — this file does not cover them beyond the boundary contract (what the filter
> receives and emits).
>
> **Implementation order:** JAX reference first (differentiable, BPTT-trainable
> *through* the filter), then port to the IHMC Java stack (EJML).
>
> **Hard JAX constraint (applies to every file):** everything must be
> `jax.jit`-able and differentiable. The time loop is `jax.lax.scan`, all
> per-contact / per-IMU-pair work is `jax.vmap`, **no Python `for` loops over
> traced data**, no data-dependent shapes, no `.item()` / `bool()` on traced
> values, all branching is `jax.lax.cond` / `jnp.where`. See §8.

---

## 0. Core architectural commitment

We estimate the state on the matrix Lie group `SE_{N+2}(3)` using the
**right-invariant error**, in the **world-centric** convention. This corner is
chosen deliberately:

> World-centric + right-invariant error is the only formulation in which the
> linearized propagation Jacobian `A` (hence `Φ`) and the FK observation matrix
> `H` are **both state-independent**, simultaneously.

The invariant that governs every decision here:

> **Group-affine structure must be preserved throughout.** Anything learned,
> estimated-but-uncertain (bias), or routed in from the pre-filter (`q̂`, `Σ_q`)
> enters **only** on the correction side or as a *covariance*, **never** as a term
> that makes the propagation dynamics `f_u(X)` violate the group-affine property
> (Thm 1, Hartley 2020).

Two things are kept **out of the SE_{N+2}(3) propagation**:

1. **IMU bias.** Handled upstream in the pre-filter (P-A design): the pelvis IMU
   runs through a Mahony attitude observer, and the residual gyro bias lives in
   the joint KF state, not here. Augmenting bias into the main state couples it
   into propagation via adjoint terms (`Φ_15`, `Φ_25`, …) and makes `A`
   state-dependent — the GMKF / DILIGENT-KIO failure mode. We do **not** do that.
2. **Learned contact confidence.** ContactNet predicts *covariances*, not state.
   They modulate the contact process-noise block. They never appear in `f_u(X)`.

> **ContactNet is external to this package.** This filter receives the predicted
> per-contact covariances `Σ_{C_i}` as **step inputs** (scan `xs`), exactly the
> way it receives IMU and `q̂`. The filter holds **no trainable parameters**;
> BPTT during training flows *through* the filter (a fixed, differentiable
> function) into the ContactNet module. ContactNet's internals are out of scope
> for this doc.

> Mantra from the joint-KF doc, restated:
> **"It's not what's in the state, it's what's in the propagation."**
> Augmenting the state on a *direct product* `G × ℝ^k` is fine; coupling the new
> terms into the propagation dynamics is what destroys group-affineness.

---

## 1. State representation

### 1.1 The group element

For `N` contact candidates the state is `X ∈ SE_{N+2}(3)`, a `(N+5)×(N+5)` matrix:

```
        ┌ R   v   p   d_1  …  d_N ┐
        │ 0   1   0   0    …  0   │
  X  =  │ 0   0   1   0    …  0   │
        │ 0   0   0   1    …  0   │
        │ ⋮               ⋱      │
        └ 0   0   0   0    …  1   ┘
```

in left-superscript (Traversaro/Featherstone) notation:

- `R  = {}^{W}R_{B}`     base orientation (world ← body)
- `v  = {}^{W}v_{B}`     base linear velocity in world
- `p  = {}^{W}p_{WB}`    base position in world
- `d_i = {}^{W}p_{WC_i}` world position of contact candidate `i`

**CoCo modeling choice (important for JIT):** keep **all `N` contact candidates
in the state permanently**, regardless of contact condition. No dynamic
add/remove. A candidate not in contact is expressed through a large contact
covariance, never a shape change. This gives a **constant computation graph** —
required for `jit` + `scan` + BPTT. `N` is a **static** integer fixed at
construction, never traced.

### 1.2 The error state

Right-invariant error: `η^r = X̄ X^{-1} = exp(ξ^∧)`, `ξ ∈ ℝ^{3N+9}`, with fixed
ordering (do not permute):

```
  ξ = [ ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; … ; ξ_{d_N} ]
```

Covariance `P ∈ ℝ^{(3N+9)×(3N+9)}` is the covariance of `ξ`.

### 1.3 Pytree design

```python
class InEKFState(NamedTuple):
    R: Array      # (3, 3)
    v: Array      # (3,)
    p: Array      # (3,)
    d: Array      # (N, 3)   stacked contact positions — vmap axis 0
    P: Array      # (3N+9, 3N+9)
```

- `NamedTuple` → valid JAX pytree, clean `scan` carry.
- **`N` is NOT stored** — implicit in `d.shape[0]`. Storing it forces static-int
  handling and recompiles.
- Keep `d` as one `(N, 3)` array, not a Python list — every per-contact op is a
  single `vmap`.
- Store `(R, v, p, d)`, not the dense `(N+5)×(N+5)` matrix; build the dense matrix
  on demand inside Lie ops that need it.

`InEKFParams` holds static config: `g` (gravity), `dt`, gyro/accel spectral
densities `Q_g, Q_a`, contact-noise floor, and the **precomputed constant `Φ`**.

---

## 2. Lie group operations (`inekf/group.py`)

Hand-author these; they must be `jit`-able and differentiable. `jaxlie` covers
`SO(3)` (use it for `Γ_0` / left Jacobian); `SE_{N+2}(3)` with a variable contact
count is custom on top.

| op | signature | notes |
|---|---|---|
| `skew(φ)` | `(3,)→(3,3)` | `(φ)_×` |
| `Gamma0(φ)` | `(3,)→(3,3)` | `SO(3)` exp (Rodrigues). `jnp.where` series fallback near `θ→0`. |
| `Gamma1(φ)` | `(3,)→(3,3)` | left Jacobian of `SO(3)` |
| `Gamma2(φ)` | `(3,)→(3,3)` | 2nd integral term (exact mean integration) |
| `exp_SEn3(ξ, N)` | `(3N+9,)→(N+5,N+5)` | `Γ_0(φ)` rotation; `Γ_1(φ)` applied to each of `v,p,d_i` (**vmap** the contacts) |
| `log_SEn3(X)` | `(N+5,N+5)→(3N+9,)` | inverse; for innovation mapping + re-init |
| `Adjoint(X)` | `(N+5,N+5)→(3N+9,3N+9)` | block form below |

Closed forms (`θ = ‖φ‖`):

```
  Γ_0 = I + (sinθ/θ)(φ)_× + ((1−cosθ)/θ²)(φ)_×²
  Γ_1 = I + ((1−cosθ)/θ²)(φ)_× + ((θ−sinθ)/θ³)(φ)_×²
  Γ_2 = ½I + ((θ−sinθ)/θ³)(φ)_× + ((θ²+2cosθ−2)/2θ⁴)(φ)_×²
  Γ_m(φ) = Σ_{n≥0} (φ)_×^n / (n+m)!
```

```
            ┌ R        0     0   …  0 ┐
            │ (v)_× R  R     0   …  0 │
  Ad_X  =   │ (p)_× R  0     R   …  0 │
            │ (d_1)_×R 0     0   …  0 │
            │ ⋮                ⋱     │
            └ (d_N)_×R 0     0   …  R ┘
```

**Near zero:** double-`where` trick — compute both branches and select with
`jnp.where`, and guard the `θ` inside the analytic branch so `grad` at `θ=0` is
finite. Never a Python `if` on a traced scalar.

---

## 3. Propagation (`inekf/propagate.py`)

Nonlinear **mean** on the group + linear **covariance** in the tangent. Input
`u = (ω̃, ã)` is the **bias-corrected** IMU (bias removed upstream).

### 3.1 Mean (exact integration, IMU constant over `dt`)

```
  R̄_{k+1} = R̄_k Γ_0(ω̄ dt)
  v̄_{k+1} = v̄_k + R̄_k Γ_1(ω̄ dt) ā dt + g dt
  p̄_{k+1} = p̄_k + v̄_k dt + R̄_k Γ_2(ω̄ dt) ā dt² + ½ g dt²
  d̄_{i,k+1} = d̄_{i,k}
```

`Γ_1`, `Γ_2` are the once/twice integrals of the rotating accelerometer signal
(`Γ_m → I/m!` as `ω→0`), so this is the *exact* step under constant-IMU-over-`dt`,
not Euler. Contact means are unchanged (their dynamics is all noise → §3.3).

### 3.2 Covariance — the state-independent `Φ`

With **no bias in the state**, the right-invariant error matrix is **constant**:

```
        ┌ 0      0   0   0 ┐   (R)
  A^r = │ (g)_×  0   0   0 │   (v)
        │ 0      I   0   0 │   (p)
        └ 0      0   0   0 ┘   (d: all zero)
```

`(g)_×` is the skew of constant gravity — no dependence on `R̄, v̄, p̄` or the IMU
input. `A^r` is nilpotent (`(A^r)³ = 0`), so

```
        ┌ I             0      0   0 ┐
  Φ  =  expm(A^r dt) = │ (g)_× dt      I      0   0 │
        │ ½(g)_× dt²    I dt   I   0 │
        └ 0             0      0   I ┘
```

is a **constant**, precomputed once in `InEKFParams`. **Never call `expm` in the
scan body.** The `d`-block is identity: contacts have no deterministic coupling.

### 3.3 Process noise `Q̄_d` — exact closed form (nilpotent ⇒ polynomial)

```
  P_{k+1} = Φ P_k Φᵀ + Q̄_d ,     Q̄_d = ∫₀^{dt} e^{A^r s} Q̄_c e^{A^rᵀ s} ds .
```

Because `A^r` is nilpotent, `e^{A^r s} = I + A^r s + ½(A^r)² s²` **exactly** (three
terms, no truncation), so the integrand is a degree-≤4 matrix polynomial in `s`
and `Q̄_d` is a closed-form polynomial in `dt`. **Do the exact integral.** Do
**not** use the `Φ Q̄ Φᵀ dt` approximation: there is no cost saving (everything is
closed-form) and it overshoots the cross terms (3× on position variance, 2× on
the position–velocity covariance), which corrupts NEES.

Injected continuous error-state densities (block-diagonal):

```
  Q̄_c = blkdiag( Q_g, Q_a, 0, R̄Σ_{C_1}R̄ᵀ, …, R̄Σ_{C_N}R̄ᵀ )
```

`Q_g = σ_g² I` (gyro), `Q_a = σ_a² I` (accel), both isotropic ⇒ rotation-invariant
in the error frame; contact densities are the (anisotropic) ContactNet
covariances rotated to world.

**Inertial block (ordering R, v, p), `G ≡ (g)_×`, symmetric:**

```
  [R,R] = Q_g dt
  [R,v] = −½ Q_g G dt²              [v,R] = ½ G Q_g dt²
  [R,p] = −⅙ Q_g G dt³              [p,R] = ⅙ G Q_g dt³
  [v,v] = Q_a dt − ⅓ G Q_g G dt³
  [v,p] = [p,v] = ½ Q_a dt² − ⅛ G Q_g G dt⁴
  [p,p] = ⅓ Q_a dt³ − (1/20) G Q_g G dt⁵
```

The off-diagonal blocks are the **cross terms**: gyro noise leaking R→v→p and
accel noise leaking v→p through the `A^r` coupling, accumulated over the step.
Sanity: drop `Q_g`, set `Q_a = q I` → textbook double integrator
`[[q dt³/3, q dt²/2],[q dt²/2, q dt]]`. PSD: `−G Q_g G = −σ_g²(g)_×² ⪰ 0`.

**Contact blocks** (decoupled in `A^r` ⇒ no propagation cross terms, no leakage
into base states):

```
  Q̄_d[d_i, d_i] = R̄ Σ_{C_i} R̄ᵀ dt        (vmap over i)
```

> **Frame note (the one honest caveat):** the exactly state-independent objects
> are `A^r/Φ` and `H` — **not** `Q̄_d`. The contact blocks carry `R̄`, and with
> *anisotropic* IMU noise or the full right-invariant treatment the inertial
> densities would be conjugated by `Ad_{X̄}` (frozen over the step) before
> integrating, adding extra off-diagonal coupling — still closed-form. For
> isotropic `Q_g, Q_a` that conjugation is trivial on the diagonal and the form
> above is exact. Keep it isotropic for v1; the `Ad`-conjugated refinement is
> available later.

Implement the inertial block as a direct fill from the formulas above (cheap,
exact); **vmap** the contact blocks. No `expm`, no numerical integration.

---

## 4. Correction — FK measurement (`inekf/correct.py`)

FK gives the body-frame vector base→contact. In the **world-centric** state this
is a **right-invariant observation** (`Y = X^{-1} b + V`), which is why `H` is
constant.

### 4.1 Per-contact observation

`h_{p,i}(q̂) = {}^{B}p_{BC_i} = R̄ᵀ(d̄_i − p̄)`, with joint angles `q̂` from the
joint KF. In right-invariant form:

```
  Y_i = [ h_{p,i}(q̂) ; 0 ; 1 ; −1 ]            b = [ 0_3 ; 0 ; 1 ; −1 ]
  H_i = [ 0   0   −I   …  +I(col d_i)  … ]      ∈ ℝ^{3×(3N+9)}
```

`H_i` is **constant** (`±I` in the `p` and `d_i` blocks). Precompute the fixed
sparse pattern; **vmap**/stack over contacts.

### 4.2 Measurement noise `N` — routed from the joint KF

```
  position FK noise :  N^p_i  =  J_{C_i}(q̂)  · Σ_q  · J_{C_i}(q̂)ᵀ
  velocity   noise  :  N^v_i  =  J_{Ċ_i}(q̂) · Σ_q̇ · J_{Ċ_i}(q̂)ᵀ
```

**DECIDED:** `Σ_q̇` is routed as its own term `J_{Ċ} Σ_q̇ J_{Ċ}ᵀ`, **kept separate
from** `Σ_q` (not folded in), for legibility. `N^p` is the noise on the position
FK measurement (§4.1); `N^v` is the noise on the contact zero-velocity
constraint, *if/when* that constraint is added to the measurement stack (open
item). The two never mix into one Jacobian.

- `Σ_q`, `Σ_q̇`, `q̂`, `q̇̂` come **from `joint_kf`** (`sigma_q`, `sigma_q_dot`,
  and the means).
- `J_{C_i}` is the contact-point position Jacobian; `J_{Ċ_i}` its time-derivative,
  both at `q̂`, from `robot/`.
- Stock Hartley's position block is `R̄ J_p Cov(w^α) J_pᵀ R̄ᵀ`; we replace the
  hand-set `Cov(w^α)` with the *filtered* `Σ_q` from the pre-filter.
- **vmap** both `N^p_i` and `N^v_i` over contacts.

> **Boundary contract:** joint-KF outputs enter here, on the **correction side**,
> always multiplied by a kinematic Jacobian. They never reach §3's `Φ` or the
> inertial `Q̄`.

### 4.3 Gain and Joseph update (right-invariant)

```
  S  = H P Hᵀ + N
  K  = P Hᵀ S^{-1}                          # jax.scipy.linalg.solve / cho_solve
  ξ⁺ = K · ν                                # ν_i = Π(X̄ Y_i) per contact, stacked
  P⁺ = (I − K H) P (I − K H)ᵀ + K N Kᵀ      # Joseph form — required
  X̄⁺ = exp(ξ⁺) X̄                            # right-invariant: exp on the LEFT
```

- **Joseph form mandatory:** `K` is from a linearized `H`, never exactly optimal,
  so the short `(I−KH)P` is not guaranteed PSD; Joseph is. Same `LΣLᵀ` principle
  as `Q̄_d`, `N`.
- `solve`/`cho_solve`, never `inv`. Symmetrize `P⁺ ← ½(P⁺ + P⁺ᵀ)`.
- Slice `ξ⁺` into `(φ, ξ_v, ξ_p, ξ_{d_i})`, apply `exp_SEn3` to the whole element
  so base and all contacts move consistently (off-diagonal covariance coupling is
  what lets a foot measurement sharpen the base and vice versa).

---

## 5. Contact covariance handling (`inekf/contact.py`) — consumes, never learns

This file **consumes** the per-contact covariances `Σ_{C_i}` produced by the
external ContactNet module. It contains **no learned components and does not
compute `Σ_{C_i}`**. ContactNet (MLP, features, weights, normalization) is a
separate module, out of scope for this doc.

What lives here:

- **Digest utilities for incoming covariances.** ContactNet emits a **Cholesky
  factor** `L_{C_i}` (the SPD-safe parameterization — keep it). `contact.py`
  reconstructs `Σ_{C_i} = L_{C_i} L_{C_i}ᵀ`, applies the noise floor / clamping,
  and rotates to world (`R̄ Σ_{C_i} R̄ᵀ`) for the `Q̄_d` contact block (§3.3).
  These "digest" helpers belong here so the filter owns its own input
  conditioning; they are pure, branch-free, `vmap`'d over contacts.
- **No add/remove.** All `N` candidates always in state (CoCo). Contact condition
  is expressed *only* through `Σ_{C_i}`: small (firm) / anisotropic (slip) / large
  (no contact). Constant graph ⇒ jit/scan/BPTT-safe.

> **TODO — left on the table, do NOT delete (`# TODO(re-anchor): see §5`):**
> contact-switch **re-anchoring**. When a candidate transitions to firm contact,
> snap its mean `d̄_i ← p̄ + R̄ h_{p,i}(q̂)` and reset its covariance block via the
> linear augmentation map `P ← F P Fᵀ + G Cov Gᵀ` (Hartley eq. 38), gated by a
> *soft* contact indicator through `jnp.where` (branch-free). **Not in v1.** This
> is expected to **improve estimation quality** — it prevents stale-anchor drift
> on long stances and limits covariance blow-up on lift-off — so implement it once
> the baseline filter is consistent. Leave the marker at the call site so we find
> it later.

---

## 6. Interfaces / routing summary

`joint_kf/` and ContactNet are **separate modules**; the InEKF sees only their
*outputs*, delivered as step inputs (scan `xs`). `inekf/` imports neither.

```
 joint_kf  (external) ──► q̂, q̇̂   ──► FK & Jacobian eval (§3 mean uses q̂ via FK;
                                      §4 uses q̂ for h_p and J_C)
 joint_kf  (external) ──► Σ_q, Σ_q̇ ──► N^p = J_C Σ_q J_Cᵀ ; N^v = J_Ċ Σ_q̇ J_Ċᵀ  [§4.2]
 ContactNet (external) ──► L_{C_i} ──► digest → R̄ Σ_{C_i} R̄ᵀ → Q̄_d block  [§3.3,§5]
 inekf  ──► ν, diag(P), state ──► consumed by ContactNet as trust features
                                  (feature assembly happens in ContactNet, NOT here)
 wrench/GMO (external, later) ──► correction-side only (§7)
```

**Forbidden edges:** nothing from `joint_kf`, ContactNet, or `wrench` may flow
into §3's `Φ` or the inertial part of `Q̄`. If a future change needs that, it's a
redesign, not a tweak.

---

## 7. Wrench / GMO axis (forward-looking, keep correction-side)

Sensorless contact-wrench estimation (replacing the removed F/T sensors) via a
generalized-momentum-observer residual is a **separate but coupled** contribution
with its own module. Two non-negotiables when it lands:

1. **Augment on a direct product**, if at all — never into the `SE_{N+2}(3)`
   propagation. Same rule as bias.
2. **Frame convention:** the wrench observation must be **body/contact-frame
   relative**, so the estimated base orientation does **not** enter `H`. A
   world-frame form re-introduces `R̄` into the observation matrix and kills the
   error-state independence bought in §0.

Leave a clean seam (`apply_wrench_residual(state, residual)`); do not implement
in v1.

---

## 8. JAX / JIT constraints (read before writing any code)

- **Time loop = `jax.lax.scan`.** Carry = `InEKFState`; per-step inputs (IMU, `q̂`,
  `Σ_q`, `Σ_q̇`, `L_{C_i}`) are scanned `xs`. No Python loop over timesteps.
- **Per-contact = `jax.vmap`.** Contact noise blocks, digest, `N^p_i`/`N^v_i`,
  `H_i`, exp translation columns — all `vmap` over the `N` axis.
- **Static shapes only.** `N` fixed at construction. "Not in contact" = big
  covariance, not a smaller array.
- **Branch-free.** `jnp.where` / `lax.cond` / `lax.select` for all traced
  branching (double-where in `Γ_0/1/2` near `θ→0`; soft indicator in re-anchor).
- **No `inv`.** `solve` / `cho_solve` for `S^{-1}`. Symmetrize covariances.
- **Differentiable end-to-end.** The whole `scan` must admit `jax.grad`/`jax.vjp`
  so ContactNet trains by BPTT through the filter. No host callbacks, no `.item()`,
  no NumPy on traced arrays.
- **Pytrees, not classes-with-logic.** Behavior in free functions.
- **`dt`, `g`, `Φ`, `H` are config/constants**, precomputed. Don't rebuild them in
  the loop.

---

## 9. File structure (`inekf/` only)

```
inekf/                # the filter — pure, param-free, differentiable
  state.py            # InEKFState, InEKFParams (NamedTuples); init helpers; precomputed Φ, H
  group.py            # skew, Gamma0/1/2, exp_SEn3, log_SEn3, Adjoint  (§2)
  propagate.py        # mean integration + constant-Φ cov propagation + closed-form Q̄_d (§3)
  correct.py          # FK right-invariant observation, N^p/N^v routing, Joseph update (§4)
  contact.py          # Cholesky digest of Σ_{C_i}, Q̄_d contact-block assembly,
                      #   re-anchor TODO (§5). NO network, NO weights.
  filter.py           # step() ; run() = lax.scan over step (§8)
  __init__.py
```

`inekf.step(state, inputs, params) -> (state, outputs)` is the scan body:
propagate (§3) → correct (§4) → emit `(ν, diag(P), state)`. `inputs` carry
`L_{C_i}` (from ContactNet) and `q̂, Σ_q, Σ_q̇` (from `joint_kf`).
`inekf.run = jax.lax.scan(step, init, xs)`.

> **Out of scope for this doc** (separate packages, separate design notes): the
> ContactNet module (`contact_net/` — MLP, feature assembly, weights, per-window
> normalization) and the training loop (`train/` — wires `contact_net.predict →
> inekf.run → state-error loss → grad`). `inekf/` imports nothing from them; the
> dependency points inward only (the trainer composes them). Keeping the filter
> param-free is what keeps the BPTT graph clean and the Java port a faithful
> forward-only mirror.

Build order, one file per prompt: `state.py` → `group.py` → `propagate.py` →
`correct.py` → `contact.py` → `filter.py`. Write `group.py` carefully and unit-test
exp/log/Adjoint round-trips (and finite gradients at `θ=0`) first.

---

## 10. Decisions — resolved & deferred

**Resolved:**

- **`N` velocity split:** `Σ_q̇` routed as its own term `J_{Ċ} Σ_q̇ J_{Ċ}ᵀ`,
  separate from `Σ_q` (legibility). [§4.2]
- **`Q̄_d` cross terms:** use the **exact closed-form** inertial block (nilpotent
  ⇒ polynomial); decoupled `R̄ Σ_{C_i} R̄ᵀ dt` contact blocks. No approximation.
  [§3.3]
- **`Σ_{C_i}` parameterization:** ContactNet emits a **Cholesky factor**; the
  filter keeps digest utilities (`L Lᵀ`, floor, rotate-to-world) in
  `inekf/contact.py`. The `Σ_{C_i}` *computation* is not in this repo. [§5]

**Deferred (kept on the table):**

- **Contact re-anchoring:** TODO in §5, marked at the call site. Expected to
  improve estimation quality; implement after the baseline is consistent.
- **Contact zero-velocity constraint:** whether `N^v` attaches to an explicit
  velocity constraint in the v1 measurement stack, or is staged later. Routing is
  decided either way. [§4.2]
- **NLL vs L2 training loss:** belongs to the ContactNet module, not this repo.
  Flag it when we build ContactNet (NLL may be wanted for NEES-consistent learned
  covariances). Does not affect `inekf/`.
- **`Ad`-conjugated anisotropic IMU noise:** the closed-form refinement noted in
  §3.3; only if isotropic `Q_g, Q_a` proves insufficient.

---

## 11. Standing invariants (never silently break these)

1. World-centric, right-invariant error, `SE_{N+2}(3)`. Don't switch corners.
2. No IMU bias and no learned term inside the SE_{N+2}(3) **propagation**.
3. Joint-KF / ContactNet / wrench outputs enter **only** correction-side or as
   covariances, always through a Jacobian or a `Q̄`/`N` block.
4. **The filter holds no trainable parameters.** All learned machinery is external
   (ContactNet module); `inekf/` imports nothing from it.
5. Constant computation graph: all `N` candidates always in state; no dynamic
   add/remove; static shapes.
6. `Φ` and `H` are precomputed constants; `Q̄_d` is the exact closed-form fill.
7. Joseph-form covariance update; symmetrize; `solve` not `inv`.
8. Everything `jit`-able and differentiable: `scan` for time, `vmap` for
   contacts, branch-free numerics.
9. Right-invariant update multiplies `exp(ξ)` on the **left** of `X̄`.
```
