# `jointKF/` — agent guide & design record

This file is the resume point for the joint-space KF pre-filter. It carries (1) a
short implementation status, (2) the standing conventions, and (3) the full design
record. It lives in `jointKF/` because the design is package-specific — the InEKF /
ContactNet appear only at the boundary.

---

## Implementation status (resume here)

| file              | status | notes |
|-------------------|--------|-------|
| `state.py`        | ✅ done (+tests) | `JointKFState` (q̂, q̇̂, b_ω, P), `JointKFParams`, `init_state`, `default_params`. State is `x = [q ; q̇ ; b_ω] ∈ R^{2n+3m}`. |
| `../robot.py`     | ✅ seam (adapter TODO) | `RobotModel` Protocol: `mass_matrix(q)` + `relative_gyro_jacobian(q) -> (m,3,n)`. IsaacLab adapter implementing it is TODO. No simulator import. |
| `noise.py`        | ✅ done (+tests) | `build_F`, `build_Q_d` / `build_process_noise` (diagonal or `σ_τ² M⁻²`), `build_R`. Takes `M` as a raw array (`M=None` → diagonal early-dev `Q_a`). |
| `predict.py`      | ✅ done (+tests) | `predict(state, params, M=None)`: `x⁻ = F x`, `P⁻ = F P Fᵀ + Q_d` (symmetrized). Raw-array `M`; `split_x` helper added to `state.py`. |
| `measurement.py`  | ✅ done (+tests) | `relative_gyro_measurement` (vmap differencing), `build_z`, `build_H`, `build_measurement`. Isolates the EKF nonlinearity; takes raw `J_omega (m,3,n)` from `robot.relative_gyro_jacobian`. |
| `update.py`       | ⏳ next | EKF update with **Joseph form** covariance update (§4). Reuses `noise.build_R` + `state.split_x`; consumes `z`/`H` from `measurement.py`. |
| `filter.py`       | ⏳ todo | Thin `predict → update` orchestrator; target for `jax.lax.scan`. |

Tests live in `tests/jointKF/test_<name>.py`. Run: `uv run pytest tests/jointKF -q`.

## Standing conventions

1. **Pytest per component.** Every new file lands with its own
   `tests/jointKF/test_<name>.py` (shapes, block layout, invariants like
   symmetry/PSD, numerical behavior) — not ad-hoc terminal checks.
2. **Strict JAX vectorization.** No Python `for`-loops over data/pair dimensions —
   use `jnp.eye`, broadcasting, `jax.vmap`. Keep everything jit-safe (static
   shapes; integer dims `n_joints`/`n_pairs` are static). Loops over the few static
   structural blocks are fine. The case to watch: `measurement.py` stacking `m`
   per-pair IMU Jacobians `J^k(q̂)` must be one `vmap` over pairs, then assembled.
3. **Mass-matrix injection seam.** Robot dynamics (`M(q̂)`, later Jacobians) come
   from a `robot.RobotModel` (the IsaacLab seam) and are passed into `noise.py` /
   `measurement.py` as raw arrays. Those modules never import a simulator.
4. The full standing principles are in §8 of the design record below.

---

# Joint KF Pre-Filter — Design Document

> Design record for the linear joint-chain Kalman filter (pre-filter) that fuses
> joint encoders with distributed IMU measurements, rejects residual gyro bias,
> and feeds joint estimates + covariances downstream to the InEKF and ContactNet.
>
> **Implementation order:** Python/JAX reference first (for ContactNet BPTT
> training), then port to Java in the IHMC stack.
>
> **Scope of this doc:** the `joint_kf/` package and its interfaces to `robot/`,
> `inekf/`, and `contact_net/`. It does NOT cover the InEKF or ContactNet
> internals beyond the contract at their boundaries.

---

## 0. Core architectural commitment

The joint KF is a **pre-filter (P-A architecture)**. Its entire reason for
existing is to improve the *quality* of the joint estimate `q̂` and produce an
*honest* joint covariance `Σ_q`, **without** any of its outputs ever entering the
InEKF process model.

**The invariant that must hold throughout:**

> Joint-KF outputs may enter the InEKF **only** through correction-side or
> learned components — always multiplied by a kinematic Jacobian (`J_C`, `J_Ċ`).
> They may **never** touch the SE₂(3) propagation `Φ` or `Q̄`.

This is what preserves the world-centric right-invariant InEKF's group-affine
structure: state-independent `H` and state-independent `Q̄` simultaneously.
Coupling joint information into propagation (the DILIGENT-KIO / GMKF failure mode)
makes `F_k` state-dependent and destroys log-linearity. We do not do that.

---

## 1. State

```
x = [ q ; q̇ ; b_ω ]  ∈ R^{2n + 3m}
```

| symbol | dim   | meaning                                                   |
|--------|-------|-----------------------------------------------------------|
| `q`    | `n`   | joint positions [rad]                                     |
| `q̇`    | `n`   | joint velocities [rad/s]                                  |
| `b_ω`  | `3m`  | **residual** relative gyro bias, one 3-vector per IMU pair |

- `n` = number of 1-DoF joints.
- `m` = number of IMU pairs being fused.
- Covariance `P ∈ R^{(2n+3m)×(2n+3m)}`.

**`b_ω` is the residual after Mahony**, not the full gyro bias. The per-IMU
Mahony filter (see §5) already does coarse bias attenuation; `b_ω` captures only
the small leftover on the *differenced* relative measurement. Its random-walk
noise and initial covariance must therefore be **tight**, or the two estimators
fight over the same error and produce a slow oscillation.

### Block layout of P

```
P = [ P_qq    P_q,q̇   P_q,b   ]
    [ P_q̇,q   P_q̇q̇    P_q̇,b   ]
    [ P_b,q   P_b,q̇   P_bb    ]
```

- `Σ_q   := P[0:n,     0:n    ]`  → InEKF position FK noise
- `Σ_q̇   := P[n:2n,   n:2n   ]`  → kinematic part of InEKF contact-velocity noise

---

## 2. Process model

Double integrator on `[q ; q̇]`, random-walk on `b_ω`. Continuous time:

```
q̇   = q̇
q̈   = w_a          (acceleration process noise)
ḃ_ω = w_b          (bias random walk)
```

### Transition matrix (EXACT — nilpotent, no truncation)

`A` is nilpotent in the `[q;q̇]` block, zero in the bias block, so
`F = exp(A·Δt) = I + A·Δt` is exact for all Δt:

```
F = [ I_n   Δt·I_n   0     ]
    [ 0     I_n      0     ]
    [ 0     0        I_3m  ]
```

### Process noise — where M(q) enters

The disturbance is physically a **torque** disturbance in joint-torque space:
`w_τ ~ N(0, Σ_τ)`. Newton's law in joint space, `M(q) q̈ = τ + …`, maps it to an
acceleration disturbance:

```
w_a = M(q)^{-1} w_τ
```

Applying the covariance transform rule `Cov(L x) = L Cov(x) Lᵀ` with `L = M(q)^{-1}`:

```
Q_a = M(q)^{-1} Σ_τ M(q)^{-T}
```

Since `M` is symmetric PD, `M^{-T} = M^{-1}`, so for isotropic torque noise
`Σ_τ = σ_τ² I`:

```
Q_a = σ_τ² · M(q)^{-2}
```

**Why this matters:** `M(q)^{-1}` is dense and reflects inertial coupling across
the kinematic tree. The sandwich `M^{-1} Σ_τ M^{-T}` therefore spreads a diagonal
torque uncertainty into **correlated** acceleration uncertainty across joints.
That off-diagonal coupling is the entire reason for using `M` instead of a
per-joint scalar — it is what makes `Σ_q` (and downstream `N`) honest about
kinematic-tree structure.

The block-sparsity of `M(q)` mirrors the tree topology (composite-rigid-body
inertia: `M_ij ≠ 0` only for ancestor/descendant joint pairs). A per-joint
independent KF produces a diagonal `Σ_q` and is provably suboptimal:
`P_∞^{per-joint} ⪰ P_∞^{tree}` in the PSD sense (Gauss-Markov).

### Discrete process noise (EXACT closed form)

For the `[q;q̇]` block (Van Loan integral with nilpotent `A`):

```
Q_d^{[q,q̇]} = [ (Δt³/3)·Q_a   (Δt²/2)·Q_a ]
              [ (Δt²/2)·Q_a   Δt·Q_a      ]
```

For the bias block:

```
Q_d^{bb} = σ_b² · Δt · I_3m       (σ_b kept TIGHT — residual after Mahony)
```

Full `Q_d = blkdiag(Q_d^{[q,q̇]}, Q_d^{bb})`.

### Early-development substitution

`Q_a = σ_τ² M(q)^{-2}` can be replaced by a hand-tuned diagonal
`Σ_q = σ_enc² I_n` during early development. This lets the InEKF and ContactNet
be validated independently before the mass-matrix coupling is switched on. The
mass-matrix shaping is a refinement, not a prerequisite.

---

## 3. Measurement model — encoder + distributed IMU fusion

Two stacked measurement blocks. This is the centerpiece of the design.

### 3a. Encoder block (linear, time-invariant)

```
z_enc = q̃ = I_n · q + v_enc,     R_enc = σ_enc² I_n
```

### 3b. IMU block (the fusion)

For each IMU pair `(a, b)` bracketing a sub-chain of joints, the per-IMU Mahony
filter supplies gravity-corrected, coarsely-debiased gyro readings and the
inter-IMU rotation `R^b_a`. We **difference** the two gyro readings, rotating
`a`'s reading into `b`'s frame, which cancels the common base motion:

```
z_{ω,ab} = ω_b^{b,w} − R^b_a · ω_a^{a,w}
```

By rigid-body kinematics, the RHS is the relative angular velocity of the
sub-chain, which depends **only** on the path joint velocities:

```
z_{ω,ab} = J^{b,a}_b(q̂) · S_ab · q̇  +  b_{ω,ab}  +  v_{ω,ab}
```

- `J^{b,a}_b(q̂)` = angular part of the relative FK Jacobian of the sub-chain.
- `S_ab` = selection matrix picking out the path joints (from the URDF / tree).
- `b_{ω,ab}` = residual relative bias (state), added linearly so the filter can
  observe and reject it whenever there is enough motion to separate `J q̇` from a
  constant offset.

**This is fusion:** each joint on the path now has *two* independent velocity
sources — encoder-differentiated `q̇̃` and the IMU-derived relative angular
velocity. The KF combines them optimally, weighted by their noise covariances.
Joints bracketed by IMUs get a much tighter velocity estimate; joints not
bracketed fall back to encoder-only.

### Stacked measurement

```
z = [ q̃        ]      H = [ I_n   0           0   ]      R = [ σ_enc² I_n   0    ]
    [ z_{ω,1}  ]          [ 0     J¹_q̇(q̂)     I_3 ]          [ 0           R_ω  ]
    [ ⋮        ]          [       ⋮               ]
    [ z_{ω,m}  ]          [ 0     Jᵐ_q̇(q̂)     I_3 ]
```

where `J^k_q̇(q̂) = J^{b_k,a_k}_{b_k}(q̂) · S_{ab,k}`.

- **`R_ω` is the Mahony-cleaned relative-gyro covariance** — smaller than a raw
  gyro because the per-IMU pre-filter already attenuated bias and noise. The
  shrinking of `R_ω` is exactly what the Mahony layer buys.
- The **only** state-dependence in `H` is `q̂` inside `J^k_q̇`. This makes the
  filter a mild **EKF**, not a strictly linear KF. The nonlinearity is contained
  entirely in the IMU block; the position dynamics stay exactly linear. Keep this
  nonlinearity isolated in the measurement module.

---

## 4. Filter recursion

Standard EKF recursion. **Use the Joseph-form covariance update** (see rationale).

```
Predict:
    x⁻ = F x
    P⁻ = F P Fᵀ + Q_d(M(q̂))

Update:
    ν  = z_meas − H x⁻                       (innovation)
    S  = H P⁻ Hᵀ + R
    K  = P⁻ Hᵀ S⁻¹
    x⁺ = x⁻ + K ν
    P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ      (Joseph form)
```

### Why Joseph form, not (I−KH)P⁻

The short form `(I−KH)P⁻` is algebraically exact **only at the optimal gain** `K`.
Our `H` contains the linearized `J^{b,a}_b(q̂)` evaluated at an estimate, so the
gain is computed from an *approximate* `H` and is never exactly optimal. The
Joseph form is the honest `LΣLᵀ` transform of the posterior error
`e⁺ = (I−KH)e⁻ − K v` over BOTH error sources (propagated prior error + injected
measurement noise), so it:

- gives the correct covariance for whatever `K` was actually applied, and
- is structurally symmetric-PSD by construction (sum of two `LΣLᵀ` sandwiches).

The extra cost is trivial at this state dimension. For an EKF with a
state-dependent measurement Jacobian, Joseph is the correct choice.

---

## 5. Mahony pre-filter (upstream, per-IMU — KEEP)

The existing IHMC per-IMU Mahony filter stays in place at the IMU-manager level.
It does something the joint KF structurally cannot: estimate each IMU's
**absolute orientation** `^W R_{I_i}` by fusing gyro with the accelerometer
gravity direction. Its role here is **preprocessing**:

- supplies gravity-corrected, coarsely-debiased `ω̂_{I_i}` → shrinks `R_ω`
- supplies the inter-IMU rotation `R^b_a` needed to express one IMU's reading in
  another's frame for the differencing in §3b

**Two-tier bias handling (be deliberate):**

- Mahony = **coarse** absolute gyro-bias attenuation, per IMU.
- Joint KF `b_ω` = **fine residual** relative bias, after Mahony.

Model `b_ω` with tight noise/initial covariance so the two layers do not fight
over the same error.

---

## 6. Outputs and routing (the B + C architecture)

The joint KF emits six quantities. **Every one lands in a correction-side or
learned component — never the propagation.**

| output            | source            | consumer                                          |
|-------------------|-------------------|---------------------------------------------------|
| `q̂`, `q̇̂`          | state mean        | InEKF FK + Jacobian evaluation: `h_Ci(q̂)`, `J_C(q̂)` |
| `Σ_q`             | `P[0:n,0:n]`      | InEKF **position** FK noise `N = J_C Σ_q J_Cᵀ`     |
| `Σ_q̇`             | `P[n:2n,n:2n]`    | kinematic part of contact-**velocity** noise `J_Ċ Σ_q̇ J_Ċᵀ` |
| innovation `ν`    | update step       | ContactNet trust feature                          |
| `b̂_ω`             | state mean        | ContactNet trust feature                          |
| `diag(Σ_q)`       | `P` diagonal      | ContactNet trust feature                          |

### B — InEKF measurement noise (structural backbone)

```
N = J_C(q̂) Σ_q J_C(q̂)ᵀ          (position FK noise — covariance pushforward)
```

This is the `LΣLᵀ` pushforward of joint uncertainty onto the contact frame with
`L = J_C`. It is anisotropic: large in contact-space directions where the
Jacobian amplifies joint error, small where kinematics constrains tightly. A
diagonal `σ_enc² I` in contact space would discard all this geometric structure.

**Velocity split (a contribution):** the contact-velocity covariance currently is
either hand-tuned or fully learned by ContactNet. Split it:

- analytical kinematic part: `J_Ċ Σ_q̇ J_Ċᵀ`
- learned residual (slip / compliance): ContactNet

This lightens ContactNet's learning burden — it only has to learn the
*non-kinematic* part — and is a cleaner division of labor.

### C — ContactNet trust features (the contribution)

The joint-KF **innovation** `ν` between the two independent velocity sources
(encoder vs IMU) is a direct, physically-grounded signal of **kinematic
consistency**. When a foot slips or the structure flexes, the rigid-body
assumption linking encoder chain and IMU chain breaks — and that disagreement
shows up *first* in the joint-KF innovation, before it propagates to the contact
point.

Append to ContactNet's feature vector `o` (currently
`Bω, Ba, q, q̇, τ, BpB→Ci, BvB→Ci`):

- per-pair innovation magnitude `‖ν_ω‖`
- residual bias estimate `b̂_ω`
- marginal joint variances `diag(Σ_q)`

These are **free** — byproducts of the fusion already being done.

> **Empirical bet, flagged:** the *value* of the C-path features is well-motivated
> theoretically but unproven. Build the plumbing so the features are available,
> then **ablate explicitly** (train with vs without the joint-KF byproducts). That
> ablation is itself a clean result regardless of outcome.

### What NOT to do

`Σ_q` / `q̂` may enter anything multiplied by a kinematic Jacobian in a
**correction or augmentation** step. They may **never** enter `Φ` or `Q̄`. In
particular, do not use `Σ_q` to drive InEKF propagation or to set the dynamics of
contact landmarks. (Covariance augmentation for a newly-added contact is fine —
its `G` carries `J_p` and sits in the measurement-derived block, not the dynamics.)

---

## 7. Proposed file structure for `joint_kf/`

Implement file by file, one at a time. Use JAX throughout; `NamedTuple` for state
(native pytree, clean `jax.lax.scan`).

| file          | responsibility                                                        |
|---------------|-----------------------------------------------------------------------|
| `state.py`    | `JointKFState` (q̂, q̇̂, b_ω, P) + `JointKFParams` + init helpers       |
| `noise.py`    | build `F`, `Q_d` (from `M(q̂)`), `R = blkdiag(R_enc, R_ω)`             |
| `predict.py`  | predict step: `x⁻ = F x`, `P⁻ = F P Fᵀ + Q_d`                         |
| `update.py`   | update step with **Joseph-form** covariance update                    |
| `measurement.py` | build stacked `z`, `H` (encoder + IMU blocks); isolates the EKF nonlinearity `J^{b,a}_b(q̂)` |
| `filter.py`   | thin orchestrator (`predict → update`); target for `jax.lax.scan`     |

### Dependencies on `robot/`

`measurement.py` and `noise.py` depend on the shared `robot/` layer:

- `M(q)` mass matrix (for `Q_a`)
- relative FK Jacobian `J^{b,a}_b(q)` and selection `S_ab` (for the IMU block)
- IMU pairing topology (which joints each pair brackets)

For initial `state.py` / `noise.py` work, the IMU pairing may be passed in as a
fixed configuration via params; wire up the real relative Jacobian when building
`measurement.py` / `update.py`.

> **Implementation note:** the shared layer lives at
> `src/invariant_estimation/robot.py` as a `RobotModel` Protocol (the IsaacLab
> seam). `noise.py` consumes `M(q̂)` as a raw `(n,n)` array (`M=None` → diagonal
> early-dev `Q_a`); `measurement.py` will extend the Protocol with the relative
> Jacobian + `S_ab`.

---

## 8. Key principles (carry these as standing constraints)

1. **Nothing the joint KF produces enters InEKF propagation.** Outputs are
   correction-side only, always via a kinematic Jacobian.
2. **It's not what's in the state, it's what's in the propagation** — the reason
   the pre-filter (P-A) preserves group-affinity is that bias estimation lives
   *here*, not in the augmented InEKF state (P-B).
3. **`M(q)` enters only the process noise**, shaping acceleration-uncertainty
   coupling across the tree. Substitutable by a diagonal during early dev.
4. **The IMU block is the only nonlinearity** (EKF), contained in `measurement.py`.
5. **Joseph form** for the covariance update — the gain is never exactly optimal
   because `H` is linearized.
6. **Mahony stays** as per-IMU coarse preprocessing; joint-KF `b_ω` is the fine
   residual with tight noise.
7. **Covariance transforms as `LΣLᵀ`** everywhere (`Q_a`, `N`, Joseph) — the
   transpose is the outer-product definition showing through, and guarantees PSD.
8. Hand-author the kinematically-sensitive ops (relative Jacobian, frame
   conventions) — high risk of silent sign/frame errors.

---

## 9. Notation

Featherstone / Traversaro style: left superscript = expressed-in frame, index
pair = reference/target. Frames: `W` world, `P` pelvis/root-joint, `I` IMU
measurement, `F_i` sole/contact of foot `i`.
