# Contact-Aided Invariant EKF — Design Document

> Design record for the **world-centric, right-invariant** contact-aided InEKF
> that fuses IMU + forward-kinematics into a pose/velocity/contact estimate on
> `SE_{N+2}(3)`. This is the *main filter*; the joint KF (`joint_kf/`) is its
> pre-filter, and ContactNet supplies its contact-velocity covariances.
>
> **Implementation order:** JAX reference first (differentiable, BPTT-trainable
> through the filter), then port to the IHMC Java stack (EJML).
>
> **Hard JAX constraint (applies to every file):** everything must be
> `jax.jit`-able and differentiable. That means: the time loop is `jax.lax.scan`,
> all per-contact / per-IMU-pair work is `jax.vmap`, **no Python `for` loops over
> traced data**, no data-dependent shapes, no `.item()` / `bool()` on traced
> values, and all branching is `jax.lax.cond` / `jnp.where`. See §8.
>
> **Scope:** the `inekf/` package and its boundary contracts with `robot/`,
> `joint_kf/`, `contact_net/`, and (later) `wrench/`. It does **not** cover those
> modules' internals beyond their interfaces.

---

## 0. Core architectural commitment

We estimate the state on the matrix Lie group `SE_{N+2}(3)` using the
**right-invariant error**, in the **world-centric** convention. This single
corner of the design space is chosen deliberately:

> World-centric + right-invariant error is the only formulation in which the
> linearized propagation Jacobian `A` (hence `Φ`) and the FK observation matrix
> `H` are **both state-independent**, simultaneously. Robo-centric, or
> left-invariant error with FK, breaks one or the other.

The invariant that governs every design decision in this package:

> **Group-affine structure must be preserved throughout.** Anything that is
> learned (ContactNet), estimated-but-uncertain (bias), or routed in from the
> pre-filter (`q̂`, `Σ_q`) enters **only** on the correction side or as a
> *covariance*, **never** as a term that makes the propagation dynamics
> `f_u(X)` violate the group-affine property (Thm 1 of Hartley 2020).

Concretely, two things are kept **out of the SE_{N+2}(3) propagation**:

1. **IMU bias.** Handled upstream in the pre-filter (P-A design): the pelvis IMU
   is run through a Mahony attitude observer, and the residual gyro bias lives in
   the joint KF state, not here. Augmenting bias into the main state couples it
   into propagation via adjoint terms (`Φ_15`, `Φ_25`, …) and makes `A`
   state-dependent — the GMKF / DILIGENT-KIO failure mode. We do **not** do that.
2. **Learned contact confidence.** ContactNet predicts *covariances*, not state.
   Those covariances modulate the contact process-noise block and the FK
   measurement noise. They never appear in `f_u(X)` itself.

> Mantra from the joint-KF doc, restated for this filter:
> **"It's not what's in the state, it's what's in the propagation."**
> Augmenting the state on a *direct product* `G × ℝ^k` is fine; coupling the new
> terms into the propagation dynamics is what destroys group-affineness.

---

## 1. State representation

### 1.1 The group element

For `N` contact candidates the state is `X ∈ SE_{N+2}(3)`, a
`(N+5) × (N+5)` matrix:

```
        ┌ R   v   p   d_1  …  d_N ┐
        │ 0   1   0   0    …  0   │
  X  =  │ 0   0   1   0    …  0   │
        │ 0   0   0   1    …  0   │
        │ ⋮               ⋱      │
        └ 0   0   0   0    …  1   ┘
```

with, in left-superscript (Traversaro/Featherstone) notation:

- `R  = {}^{W}R_{B}`          base orientation (world ← body)
- `v  = {}^{W}v_{B}`          base linear velocity in world
- `p  = {}^{W}p_{WB}`         base position in world
- `d_i = {}^{W}p_{WC_i}`      world position of contact candidate `i`

**CoCo modeling choice (important for JIT):** following CoCo-InEKF, we keep
**all `N` contact candidates in the state permanently**, regardless of current
contact condition. We do **not** add/remove contacts dynamically. This is what
gives a **constant computation graph** with static shapes — a hard requirement
for `jit` + `scan` + BPTT. A "candidate not in contact" is expressed purely
through a large contact-velocity covariance from ContactNet, never through a
shape change. (See §5. This is the single most important deviation from stock
Hartley and the reason the filter is differentiable.)

`N` is a **static** integer fixed at construction (`functools.partial` /
closure), never a traced value.

### 1.2 The error state

Right-invariant error between truth `X` and estimate `X̄`:

```
  η^r = X̄ X^{-1}  =  exp(ξ^∧),     ξ ∈ ℝ^{3(N+2)}
```

Tangent-space ordering (fixed everywhere — do not permute):

```
  ξ = [ ξ_R ; ξ_v ; ξ_p ; ξ_{d_1} ; … ; ξ_{d_N} ]   ∈ ℝ^{3N+9}
```

The covariance `P ∈ ℝ^{(3N+9) × (3N+9)}` is the covariance of `ξ`.

### 1.3 Pytree design (mirror the joint KF conventions)

```python
class InEKFState(NamedTuple):
    R: Array      # (3, 3)
    v: Array      # (3,)
    p: Array      # (3,)
    d: Array      # (N, 3)   stacked contact positions — vmap axis 0
    P: Array      # (3N+9, 3N+9)
```

- `NamedTuple` → valid JAX pytree, no `register_pytree_node`, clean `scan` carry.
- **`N` is NOT stored in the struct** — it is implicit in `d.shape[0]` and `P`'s
  size. Storing it as a field would force static-int handling and recompiles.
- Keep `d` as a single `(N, 3)` array, *not* a Python list of vectors — this is
  what lets every per-contact operation be a single `vmap`.
- Do **not** store the full `(N+5)×(N+5)` matrix as the canonical state; store
  `(R, v, p, d)` and build the dense matrix only inside Lie ops that need it.
  Reason: the dense matrix is mostly constant structure (identity block + zeros),
  and reconstructing it on demand avoids carrying redundant constants through the
  BPTT graph.

A companion `InEKFParams` holds static config: `g` (gravity, `(3,)`), `dt`,
process-noise spectral densities for gyro/accel, and the contact-noise *floor*
(ContactNet supplies the rest at runtime).

---

## 2. Lie group operations (`inekf/lie.py`)

These are the mathematically sensitive primitives. **Hand-author them**; do not
lean on a black-box. They must be `jit`-able and differentiable. `jaxlie` covers
`SO(3)` cleanly — use it for `Γ_0` / the left Jacobian — but `SE_{N+2}(3)` with a
variable contact count needs custom code built on top.

Required (all pure `jnp`, all `vmap`-friendly):

| op | signature | notes |
|---|---|---|
| `skew(φ)` | `(3,) → (3,3)` | `(φ)_×` |
| `Gamma0(φ)` | `(3,) → (3,3)` | `SO(3)` exp = Rodrigues. Use `jnp.where`/series fallback near `‖φ‖→0`, **not** a Python `if`. |
| `Gamma1(φ)` | `(3,) → (3,3)` | left Jacobian of `SO(3)` |
| `Gamma2(φ)` | `(3,) → (3,3)` | 2nd integral term (needed in exact mean integration) |
| `exp_SEn3(ξ, N)` | `(3N+9,) → (N+5,N+5)` | block exp: `Γ_0(φ)` rotation, `Γ_1(φ)` applied to each of `v,p,d_i`. **vmap the `Γ_1 ξ_{d_i}` over the `N` contacts.** |
| `log_SEn3(X)` | `(N+5,N+5) → (3N+9,)` | inverse; needed for innovation `Π X̄ Y` mapping and for re-init only |
| `Adjoint(X)` | `(N+5,N+5) → (3N+9,3N+9)` | block form below |
| `adjoint_alg(ξ)` | small-`ad` (optional, only if you add closed-form `Q̄`) | |

Closed forms to implement directly (from the `SE_K(3)` appendix):

```
            ┌ R        0     0   …  0 ┐
            │ (v)_× R  R     0   …  0 │
  Ad_X  =   │ (p)_× R  0     R   …  0 │
            │ (d_1)_×R 0     0   …  0 │   ← block-rows for contacts have R on
            │ ⋮                ⋱     │      the diagonal, (d_i)_×R in col 1
            └ (d_N)_×R 0     0   …  R ┘
```

```
  exp(ξ)  has rotation Γ_0(φ) and translation columns Γ_1(φ) ξ_v,
          Γ_1(φ) ξ_p, Γ_1(φ) ξ_{d_i}.
```

**`Γ_0` / `Γ_1` near zero:** implement the Taylor fallback with `jnp.where` on
`θ = ‖φ‖`, computing *both* branches and selecting — never a control-flow `if` on
a traced scalar, or `grad` will see a `nan` through the dead branch. (Standard
"double-where" trick: also guard the `θ` used inside the analytic branch so the
zero-input gradient is finite.)

---

## 3. Propagation (`inekf/propagate.py`)

Two sub-steps: nonlinear **mean** on the group, linear **covariance** in the
tangent. The input `u = (ω̃, ã)` is the **bias-corrected** IMU (bias removed
upstream — see §0).

### 3.1 Mean (exact integration, IMU constant over `dt`)

```
  R̄_{k+1} = R̄_k Γ_0(ω̄ dt)
  v̄_{k+1} = v̄_k + R̄_k Γ_1(ω̄ dt) ā dt + g dt
  p̄_{k+1} = p̄_k + v̄_k dt + R̄_k Γ_2(ω̄ dt) ā dt² + ½ g dt²
  d̄_{i,k+1} = d̄_{i,k}                          (contact mean is constant)
```

- The contact mean is **unchanged** by propagation (zero-mean contact velocity;
  all the contact dynamics is noise — that goes in `Q̄`, §3.3).
- This is exact under constant-IMU-over-`dt`, not a forward-Euler approximation.
  Use `Γ_1`, `Γ_2` from `lie.py`.

### 3.2 Covariance — the state-independent `Φ`

This is the payoff of the world-centric right-invariant choice. With **no bias in
the state**, the right-invariant error-dynamics matrix is **constant**:

```
        ┌ 0      0   0   0 ┐   (R)
  A^r = │ (g)_×  0   0   0 │   (v)
        │ 0      I   0   0 │   (p)
        └ 0      0   0   0 ┘   (d-block: all zero)
```

`(g)_×` is the skew of the *constant* gravity vector — there is **no dependence
on `R̄`, `v̄`, `p̄`, or the IMU input**. Therefore

```
  Φ = expm(A^r dt)
```

is a **constant matrix**, computable **once at filter construction**, not per
step. Its analytic blocks (no-bias case):

```
        ┌ I             0      0   0 ┐
  Φ  =  │ (g)_× dt      I      0   0 │
        │ ½(g)_× dt²    I dt   I   0 │
        └ 0             0      0   I ┘
```

> **Implementation note:** precompute `Φ` in `InEKFParams` (or as a module
> constant once `g`, `dt` are known). Do **not** call `jax.scipy.linalg.expm`
> inside the scan body — bake the closed form above. The `d`-block is identity
> because contact positions have no deterministic coupling; all their dynamics is
> in `Q̄`.

### 3.3 Process noise `Q̄`

```
  P_{k+1} = Φ P_k Φᵀ  +  Q̄_d
```

`Q̄_d` is block-diagonal-ish, assembled from three sources via the `LΣLᵀ`
principle (same principle that gives `N` and the Joseph form):

- **gyro / accel** spectral densities on the `ξ_R`, `ξ_v` blocks (constant,
  body-frame, world-centric → these are clean in this corner);
- **contact velocity** on each `ξ_{d_i}` block: this is where ContactNet enters.
  The continuous contact model is `ḋ_i = -R̄ w_{C_i}`, `w_{C_i} ~ N(0, Σ_{C_i})`
  with `Σ_{C_i}` **predicted per-contact, per-step by ContactNet** (IMU-frame
  covariance, per CoCo footnote 2). The discrete block is

  ```
    Q̄_d[d_i, d_i]  =  R̄ Σ_{C_i} R̄ᵀ · dt        (vmap over i)
  ```

  > ⚠ **Honesty flag (do not overclaim):** this contact block carries an explicit
  > `R̄`. So `Q̄` is *not literally* state-independent — only the IMU-driven
  > `A`/`Φ` are. The "state-independent `Q̄`" claim holds for the inertial part;
  > the contact block legitimately rotates the ContactNet covariance into world
  > frame. Keep `R̄` here; do not "simplify" it away. (This is the right place for
  > `R̄` to appear — correction/noise side — and it does not break group-affine
  > propagation because it sits in `Q̄`, not in `f_u(X)`.)

- Use the standard discrete approximation `Q̄_d ≈ Φ Q̄_c Φᵀ dt` only if you want
  the cross terms; for a first cut the block assembly above is enough.
  **vmap** the per-contact assembly; **never** loop over `i` in Python.

---

## 4. Correction — FK measurement (`inekf/correct.py`)

Forward kinematics gives the body-frame vector from base to each contact. In the
**world-centric** state this is a **right-invariant observation** (`Y = X^{-1} b
+ V`), which is exactly why `H` comes out constant.

### 4.1 Per-contact observation

For contact `i`, with measured joint angles `α̃` (here `α̃ = q̂` from the joint
KF), FK gives `h_p,i(q̂) = {}^{B}p_{BC_i}`. Stacked into the right-invariant form,
the per-contact innovation uses

```
  Y_i = [ h_{p,i}(q̂) ; 0 ; 1 ; −1 ]    (selecting base index +1, contact i −1)
  H_i = [ 0   0   −I   …  +I(col i)  … ]        ∈ ℝ^{3 × (3N+9)}
```

`H_i` is **constant** (just `±I` in the `p` and `d_i` blocks, zeros elsewhere).
**Build the full `H` by vmapping `H_i` over contacts and stacking** — it is a
fixed sparse pattern, so you can also just precompute it once.

### 4.2 Measurement noise `N` — routed from the joint KF

This is the P-A / "B+C" routing boundary. **`N` does not come from a hand-tuned
constant here** — it is propagated from the pre-filter's joint covariance through
the contact Jacobian:

```
  N_i  =  J_{C_i}(q̂) · Σ_q · J_{C_i}(q̂)ᵀ          (+ optional velocity split
                                                    J_Ċ Σ_q̇ J_Ċᵀ)
```

- `Σ_q`, `q̂` come **from `joint_kf`** (its `sigma_q` property and mean).
- `J_{C_i}` is the contact-point Jacobian, evaluated at `q̂`, from `robot/`.
- In stock Hartley this block is `R̄ J_p Cov(w^α) J_pᵀ R̄ᵀ`; our version replaces
  the hand-set `Cov(w^α)` with the *honest, filtered* `Σ_q` from the pre-filter.
  Keep the frame handling consistent with how `J_{C_i}` is expressed.
- **vmap** `N_i` over contacts.

> **Boundary contract (must not be violated):** joint-KF outputs enter here, on
> the **correction side**, always multiplied by a kinematic Jacobian. They never
> reach §3's `Φ` or `Q̄_inertial`. This is the architectural invariant from the
> joint-KF doc, enforced at this seam.

### 4.3 Gain and Joseph update (right-invariant)

Stack contacts (block-diagonal `N`, stacked `H`, stacked innovation), then:

```
  S  = H P Hᵀ + N
  K  = P Hᵀ S^{-1}                          # use jax.scipy.linalg.solve, not inv
  ξ⁺ = K · ( stacked innovation )           # innovation = Π X̄ Y per contact
  P⁺ = (I − K H) P (I − K H)ᵀ + K N Kᵀ      # Joseph form — required
```

**Right-invariant state update** — `exp` multiplies on the **left** (world-frame
error):

```
  X̄⁺ = exp(ξ⁺) · X̄
```

i.e. update `R, v, p, d` by extracting blocks of `ξ⁺` and applying the
`SE_{N+2}(3)` exp from `lie.py`. Apply it to the whole group element at once
(rotation + all translations) so the contacts move consistently with the base.

- **Joseph form is mandatory**, not the short `(I−KH)P` form. The gain is built
  from a linearized (first-order) `H`; it is never the exact optimal gain, so the
  short form is not guaranteed PSD. Joseph stays symmetric-PSD and is the same
  `LΣLᵀ` principle as `Q_a`, `N`, and the contact block.
- Solve `S^{-1}` via `jax.scipy.linalg.solve` / `cho_solve`, never `inv`.
- Symmetrize `P⁺ = ½(P⁺ + P⁺ᵀ)` after the update (cheap numerical hygiene that
  helps the BPTT gradients too).

---

## 5. Contact management (`inekf/contact.py`)

**There is no dynamic add/remove.** All `N` candidates live in the state for all
time (CoCo). Contact state is expressed *only* through `Σ_{C_i}`:

- firm contact      → small `Σ_{C_i}`  (foot pinned, FK trusted)
- directional slip  → anisotropic `Σ_{C_i}` (large along slip direction)
- no contact        → large `Σ_{C_i}`  (candidate effectively floats)

ContactNet predicts these every step. Because the graph is structurally
constant, the filter is one fixed sequence of differentiable ops — exactly what
BPTT needs.

- **Re-anchoring** (the moral equivalent of "adding" a contact): when a candidate
  transitions to firm contact you may want to snap `d̄_i = p̄ + R̄ h_{p,i}(q̂)` and
  reset its covariance block. Do this with `jnp.where` on a (soft) contact
  indicator and a covariance-augmentation linear map `P ← F P Fᵀ + G Cov Gᵀ`
  (Hartley eq. 38 form), **all branch-free**. Never gate it with a Python `if`.
- Keep this optional for v1; the pure "all candidates always in state, ContactNet
  modulates trust" path is simpler and is the CoCo baseline. Start there.

---

## 6. Interfaces / routing summary

```
 joint_kf  ──► q̂, q̇̂        ──► FK & Jacobian eval (§3 mean uses q̂ only via FK,
                                §4 uses q̂ for h_p and J_C)
 joint_kf  ──► Σ_q, Σ_q̇     ──► N = J_C Σ_q J_Cᵀ (+ vel split)        [§4.2]
 contact_net ──► Σ_{C_i}    ──► contact block of Q̄ (§3.3) AND any N floor
 inekf     ──► ν, diag(P), (state) ──► ContactNet trust features (+ joint-KF
                                       ν, b̂_ω, diag(Σ_q))
 wrench/GMO ──► (correction-side only, §7)
```

**Forbidden edges:** nothing from `joint_kf`, `contact_net`, or `wrench`
may flow into §3's `Φ` or the inertial part of `Q̄`. If a future change needs
that, it's a redesign, not a tweak.

---

## 7. Wrench / GMO axis (forward-looking, keep correction-side)

Sensorless contact-wrench estimation (replacing the removed F/T sensors) via a
generalized-momentum-observer residual is a **separate but coupled** contribution
and will get its own module. Two non-negotiables when it lands:

1. **It augments on a direct product**, if at all — never into the `SE_{N+2}(3)`
   propagation. Same rule as bias: "it's not what's in the state, it's what's in
   the propagation."
2. **Frame convention:** the wrench observation must be expressed
   **body/contact-frame-relative**, so the estimated base orientation does **not**
   get dragged into `H`. Putting it in a world frame re-introduces an `R̄` into
   the observation matrix and kills the error-state independence we bought in §0.

Leave a clean seam (a correction-style `apply_wrench_residual(state, residual)`)
but do not implement it in v1.

---

## 8. JAX / JIT constraints (read before writing any code)

These are not stylistic — violating them breaks `jit`, `scan`, or BPTT.

- **Time loop = `jax.lax.scan`.** The scan carry is `InEKFState`; the per-step
  inputs (IMU, `q̂`, `Σ_q`, `Σ_{C_i}`) are the scanned `xs`. No Python loop over
  timesteps anywhere on the hot path.
- **Per-contact / per-IMU-pair = `jax.vmap`.** Contact noise blocks (§3.3), `N_i`
  (§4.2), `H_i`, exp translation columns — all `vmap` over the `N` axis. Zero
  Python loops over contacts.
- **Static shapes only.** `N` fixed at construction. No `dynamic_slice` with
  traced sizes, no boolean-mask indexing that changes output shape. "Not in
  contact" = big covariance, not a smaller array.
- **Branch-free.** Replace every `if` on traced data with `jnp.where` /
  `jax.lax.cond` / `jax.lax.select`. This matters most in `Γ_0`/`Γ_1` near
  `θ→0` (double-where trick) and in contact re-anchoring (§5).
- **No `inv`.** Use `solve` / `cho_solve` for `S^{-1}`. Symmetrize covariances.
- **Differentiable end-to-end.** The whole `scan` must admit `jax.grad` /
  `jax.vjp` so ContactNet trains by BPTT through the filter. Keep everything in
  `jnp`; no host callbacks, no `.item()`, no NumPy on traced arrays.
- **Pytrees, not classes-with-logic.** `InEKFState` / `InEKFParams` are
  `NamedTuple`s; put behavior in free functions, not methods that close over
  traced state.
- **`dt` and `g` are config**, baked into the precomputed constant `Φ`. Don't
  recompute `expm` in the loop.

---

## 9. File structure (`inekf/`)

```
inekf/
  state.py        # InEKFState, InEKFParams (NamedTuples); init helpers
  lie.py          # skew, Gamma0/1/2, exp_SEn3, log_SEn3, Adjoint  (§2)
  propagate.py    # mean integration + constant-Φ covariance propagation (§3)
  correct.py      # FK right-invariant observation, N routing, Joseph update (§4)
  contact.py      # candidate covariance plumbing, optional re-anchor (§5)
  filter.py       # one step() ; run() = lax.scan over step (§8)
  routing.py      # explicit seams to joint_kf / contact_net (§6) — thin
  __init__.py
```

`step(state, inputs, params) -> (state, outputs)` is the scan body:
propagate (§3) → correct (§4) → emit outputs (ν, diag(P), state) for ContactNet.
`run` is just `jax.lax.scan(step, init_state, xs)`.

Build order, one file per prompt (as with the joint KF):
`state.py` → `lie.py` → `propagate.py` → `correct.py` → `contact.py` →
`filter.py`. Write `lie.py` carefully and unit-test exp/log/Adjoint round-trips
(and finite gradients at `θ=0`) before anything depends on them.

---

## 10. Open decisions

- **`N` velocity split:** do we route `Σ_q̇` as a separate `J_Ċ Σ_q̇ J_Ċᵀ`
  measurement-noise term, or fold it into `Σ_q`? (Joint-KF doc left this open
  too; resolve consistently across both filters.)
- **Contact re-anchoring (§5):** ship the branch-free augmentation in v1, or run
  pure "always-in-state + ContactNet trust" first and add re-anchoring only if
  drift on long stance shows up?
- **`Q̄_d` cross terms:** block-assembly (simple) vs. `Φ Q̄_c Φᵀ dt` (full).
  Start simple; revisit if NEES is hot.
- **ContactNet `Σ_{C_i}` parameterization:** Cholesky-factor output → SPD is the
  safe choice for BPTT; confirm it matches the contact-net module's contract.
- **NLL vs. L2 training:** orthogonal to the filter, but if we want NEES-consistent
  covariances the contact-net loss may need an NLL term — doesn't change this
  package, just flagged.

---

## 11. Standing invariants (never silently break these)

1. World-centric, right-invariant error, `SE_{N+2}(3)`. Don't switch corners.
2. No IMU bias and no learned term inside the SE_{N+2}(3) **propagation**.
3. Joint-KF / ContactNet / wrench outputs enter **only** correction-side or as
   covariances, always through a Jacobian or a `Q̄`/`N` block.
4. Constant computation graph: all `N` candidates always in state; no dynamic
   add/remove; static shapes.
5. `Φ` is a precomputed **constant**; `H` is a fixed sparse pattern.
6. Joseph-form covariance update; symmetrize; `solve` not `inv`.
7. Everything `jit`-able and differentiable: `scan` for time, `vmap` for
   contacts, branch-free numerics.
8. Right-invariant update multiplies `exp(ξ)` on the **left** of `X̄`.
```
