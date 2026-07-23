# MuJoCo CRBA vs. Mecano CRBA in the Alex Joint-KF Parity

*Working note — reconciling the two mass-matrix backends behind the Java
estimator and its Python/MJX port. Written 2026-07-22.*

## 0. TL;DR

Both libraries implement the same algorithm (Composite Rigid Body → joint-space
inertia `M(q)`), and on an *identical model with identical free DoFs* they agree
to machine precision. Every discrepancy we see is therefore **structural, not
numerical**. The one that produces large modeling error is that **Mecano, as
wired in `JointLevelKFPreFilter`, freezes every off-path subtree's inertia at the
construction configuration `q = 0` and never refreshes it**, whereas MuJoCo's
`mj_crb` evaluates the *whole* robot at the live `q`. When the arms/ankles are
far from zero — i.e. during any real walking or manipulation run — this is a
persistent bias of up to ~14% in `diag(Qa)`, with a sign that flips between the
spine and the legs. Matching it on the Python side (freeze off-path joints at
`qpos0`) collapses the parity error from ~3–6% to ~0.5%.

---

## 1. Both compute the same object

The joint-space mass matrix is defined by the kinetic energy form

    T(q, q̇) = ½ q̇ᵀ M(q) q̇ .

The Composite Rigid Body Algorithm (CRBA) builds `M` from body spatial inertias
`I_i` and joint motion subspaces `S_i`:

    M_ij = S_iᵀ I^c_max(i,j) S_j ,     I^c_i = Σ_{k ∈ subtree(i)} I_k ,

with the composite inertia `I^c` accumulated leaf-to-root. This is an exact,
O(n²) computation. **There is no approximation inside either implementation**, so
two correct CRBA implementations of the same articulated system return the same
`M(q)` to floating-point roundoff (~1e-12 on this problem). If MuJoCo and Mecano
disagree by percent, they are not computing CRBA of the same system.

That reframes the whole question: the differences are not in the algorithm but in
**what system each is handed**.

---

## 2. Where they *cannot* differ (and we verified it)

**Base-coordinate representation.** MuJoCo's free joint carries 6 DoF expressed
in the world frame (3 translational, then 3 rotational); Mecano's `SixDoFJoint`
uses its own ordering and frame. This is irrelevant to the filter, because the
joint-KF consumes the **Schur complement** onto the filtered joints,

    Λ = M_ff − M_fb M_bb⁻¹ M_bf ,     b = base 6 DoF ,

and `Λ` is invariant under any invertible change of the nuisance coordinates `b`:
for base change `T`,

    M_fb T (Tᵀ M_bb T)⁻¹ Tᵀ M_bf = M_fb M_bb⁻¹ M_bf .

We confirmed this numerically — `Λ` is invariant to base orientation to
**9.4e-15**. So no base-frame convention difference can leak into `Λ`, `Λ_eff`,
or `Qa`. This one is closed.

---

## 3. Where they *do* differ

### 3a. DoF scope — full model vs. considered subsystem

- **MuJoCo** builds the full `35×35` `qM` (6 base + 29 hinges) at the live `q`.
  The Python port then takes the principal submatrix on `{base 6, 9 filtered
  joints}` and Schur-eliminates the base.
- **Mecano**, via `MultiBodySystemReadOnly.toMultiBodySystemInput(jointsToConsider)`,
  builds a calculator over only the *considered* joints: the floating base plus
  the joints spanning base→filtered. Off-path joints (arms, head, ankles) are
  **not DoFs** of that calculator.

By itself this is fine and *equivalent*: locking a coordinate restricts the
kinetic-energy form to a subspace, and the restricted `M` is exactly the
principal submatrix of the full `M` on the retained DoFs. This is why the
"locked-subsystem" reading (Schur-eliminate the base only) gives ~6% error while
the "free-nuisance" reading (eliminate all 26 non-filtered DoFs) gives ~38% — the
locked reading is structurally correct.

### 3b. The stale-inertia trap — **the large error**

Off-path joints are not simply dropped. Mecano composites each ignored subtree's
spatial inertia into its parent link via `considerIgnoredSubtreesInertia`, so the
arms still contribute mass. **But that compositing runs exactly once, inside
`CompositeRigidBodyMassMatrixCalculator`'s constructor** (`updateIgnoredSubtreeInertia`,
called from the ctor and nowhere else). The result is a `SpatialInertia` stored
in the parent's body-fixed frame at whatever configuration the model held at
construction — which on hardware is the freshly-built model, **`q = 0`** — and it
is never recomputed.

The consequence:

    M_Mecano(q) = M( q_considered,  q_off-path = 0 )
    M_MuJoCo(q) = M( q_considered,  q_off-path = q_live )

On the July 17 run the arms sit near `(shoulder ≈ 0.71, elbow ≈ −1.91)` rad and
the ankles near `−0.40` rad from the first tick. Feeding those live instead of
zero moves `diag(Qa)` by up to **14.4%**. A scale sweep `q_off = s·q_live` has a
sharp unique minimum at `s = 0` (0.21% at s=0, 3.1% at s=0.1, 14.4% at s=1) —
direct evidence that Mecano is frozen at zero.

**The sign signature.** The error is not uniform: freezing the arms straightens
the torso subtree's composited inertia (moving `SPINE_Z` one way), while freezing
the ankles shortens the leg subtrees' effective inertia (moving the legs the
other way). This is exactly the fingerprint we saw — eight leg joints high (ratio
1.03–1.06) and `SPINE_Z` low (0.86). A global scale or unit error could never
produce opposite signs on one subtree; a subtree-specific stale-inertia error
does so naturally.

### 3c. Armature / reflected rotor inertia

- **MuJoCo** folds `dof_armature` into `qM`'s diagonal *before* any block is
  gathered, so `Λ` computed from MuJoCo's `qM` is *already* `Λ_eff`.
- **Mecano** CRBA knows nothing about armature; Java adds `diag(rotor)` to `Λ`
  *after* the Schur complement.

These are algebraically identical because the rotor diagonal touches neither
`M_bb` nor `M_fb`:

    (M_ff + diag(a)) − M_fb M_bb⁻¹ M_bf = Λ + diag(a) .

Not a source of error **if handled correctly**, but it is the classic double-add
trap: put armature in the MJCF *and* add rotor post-Schur and you count the
drivetrain twice, silently quadrupling distal-joint apparent inertia and starving
`Qa` ~4×. The port avoids it by putting rotor only in `armature` and adding
nothing post-Schur.

### 3d. Inertia-frame conversion (URDF → MJCF), a converter-side risk

URDF states each link's inertia tensor in the *inertial-origin* frame; MJCF
`fullinertia` is in the *body* frame and refuses an accompanying orientation. The
converter must apply `I_body = R I_urdf Rᵀ` explicitly. Done via a congruence
this is exact; done via eigendecomposition it inherits eigenvector sign/order
ambiguity and can silently rotate a link's inertia. This is a MuJoCo-side-only
hazard (Mecano reads the URDF frame directly), so it is a *divergence* risk
between the two even when both are "correct."

---

## 4. Error budget

| Source | Cancels in `Λ`? | Magnitude on Alex | Status |
|---|---|---|---|
| Base frame/order convention | Yes (proven 9.4e-15) | 0 | closed |
| CRBA numerics on identical DoF | — | ~1e-12 | closed |
| Considered-subsystem *scope* | equivalent to principal submatrix | 0 (if read as locked) | closed |
| **Off-path inertia frozen at `q=0`** | **No** | **up to 14.4%, sign-split** | **the finding** |
| Armature double-add | — | ~4× if mishandled | avoided by design |
| URDF→MJCF inertia rotation | — | link-dependent | converter must use congruence |

Everything except row 4 is either provably zero or a design pitfall we already
guard. Row 4 is the whole ~3–6% residual.

---

## 5. Is Mecano's behavior *correct*?

No — the frozen lumped inertia is **stale by construction**. The
physically-correct mass matrix would refresh the composited subtree inertia each
tick (or, cleaner, promote the arms/head to considered joints so CRBA tracks them
live). The Java estimator is using a systematically biased `M(q)` for its process
noise whenever the upper body is away from `q = 0`, which is essentially always.
Two separable conclusions follow:

1. **For parity** (the immediate goal): the port must *reproduce* the freeze —
   feed off-path joints `qpos0`, not their live angles. `MjxModel.qpos` already
   does this by construction, so only the test oracle needed the fix. This is
   what took the parity error to 0.5%.

2. **For correctness** (a real finding about the flight code): Java's process
   noise is mis-scaled by up to ~14% on the joints whose subtrees articulate, in
   a configuration-dependent way. It is bounded (the arms are lumped rigid, not
   dropped, so it never blows up), but it is a genuine latent bug in
   `JointLevelKFPreFilter`'s use of Mecano — one the parity harness surfaced.
   Whether to fix it upstream (refresh the lumped inertia, or widen the considered
   set) is a separate decision from matching it in the port.

---

## 6. Recommendations

- **Port:** keep off-path joints frozen at `qpos0` in every `M(q)` evaluation,
  matching Mecano's constructor-time freeze. Assert it (the parity oracle now
  does, at 0.5%).
- **Converter:** rotate link inertias into the body frame by congruence
  `R I Rᵀ`, never by eigendecomposition, to avoid a MuJoCo-only divergence.
- **Rotor:** armature in MJCF only; never re-add post-Schur.
- **Upstream (optional, correctness):** consider refreshing the ignored-subtree
  inertia each tick or adding the shoulders/elbows to the considered subsystem,
  and re-measuring whether the ~14% process-noise bias matters to the InEKF that
  consumes `Σ_q`. This is a change to the *Java* estimator, not the port, and
  should be decided on its own merits — the port's job is to match whatever Java
  does.

---

*Caveat on the ~3–6% cluster the port shows before the freeze fix is applied
everywhere: this note explains it as the stale-inertia difference, which the
`s=0` sweep supports strongly, but we have not independently excluded a small
model-fidelity gap (e.g. Alex001 carrying payload not in `model.sdf`). The freeze
fix landing the residual at 0.5% is the strongest evidence it is the dominant
term.*
