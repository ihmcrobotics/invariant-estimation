# ContactNet at N=4 (toe/heel), two trunk widths — do the two changes compose?

Overnight run, 2026-07-29 → 2026-07-30. Branch `contactnet/process-socket`, HEAD `3f0a1ad`.

> **STATUS: COMPLETE.** The overnight agent died at ~04:18 to a server-side API
> 529 (not a defect in the run) after finishing Steps 0-5 and the w256 in-sample
> replay. The remaining evaluation (w128 in-sample, both held-out, the closed-loop
> analysis) was completed the following morning from the artifacts it left on disk.

---

## 1. Verdict

**The two changes compose on the vertical axis and actively fight each other on the
rotational one.** `(128, 64)` at N=4 gives the best sink on record — closed-loop
final `dz` **+0.070 m** and sink rate **+0.0019 m/s**, against run 5's −0.180 m /
−0.0064 m/s — so the composition works for what it was aimed at.

**But the yaw fix does not survive training, and it fails worse than before.**
Closed-loop yaw (tail RMS) goes 0.897° for the N=4 *analytic* filter → 3.06°
(w256) → **4.71°** (w128). Run 5 at N=2 was 2.22°. So training on top of toe/heel
re-spends the orientation observability that toe/heel bought, and having four
anchors to loosen instead of two makes it *worse*, not better. Tilt regresses the
same way (0.240° → 0.437°/0.489°).

**Width: `(128, 64)` is the better choice, on generalisation, not speed.** On the
metric L2 actually optimises, w256's held-out `vel_rms` is **1.016x the analytic
baseline** — it made out-of-sample velocity *worse* while improving it in sample
(0.819x). w128 is 0.955x held out. w128 also wins held-out `|slope_e_pz|` and
`pos_rms`. The measured speed difference was **1.06x** (4567 s vs 4850 s), i.e.
negligible, exactly as predicted.

**Next run should change the objective, not the capacity or the contact count.**
`l2_velocity` has no attitude term, so nothing penalises spending yaw to buy sink.
Add an attitude term, or move to `beta_nll` whose `logdet` term charges for
covariance inflation directly. More contacts without that is likely to make the
rotational cost worse again.

---

## 2. What ran

### Step 0 — pre-flight — PASSED

```
JAX_PLATFORMS=cpu uv run pytest tests/sim/test_multipoint_contacts.py tests/pipeline -q
31 passed in 254.92s
```
10 multipoint + 21 pipeline, as expected.

### Step 1 — N=4 DR training set

```
TMPDIR=/tmp/cn_collect_dr4 uv run python -m invariant_estimation.sim.collect --toe-heel --dr \
  --seconds 60 --seeds 0 1 2 --out data/dr4
```
Collector banner: `9 filtered joints, 8 IMUs, 4 contacts, 4 off-path joints, dt=0.001` — **N=4 gate passed.**
Disk before: 463 GB free, so no need to drop seeds.

Shape gate on the first saved rollout (`data/dr4/flat_seed000.npz`):

```
inputs.contact_chol      (62000, 4, 3, 3)     <- N=4
inputs.contact_meas_chol (62000, 4, 3, 3)
sensors.contact          (62000, 2)           <- per-FOOT trust, still 2
sensors.contact_chol     (62000, 4, 3, 3)
```

**Free extra evidence:** `meta.travelled_m` for `dr4/flat_seed000` is
`4.022111511462757`, *bit-identical* to `dr/flat_seed000` at N=2. The plant is
untouched by the extra contact sites, which independently confirms the
"exact-0.0 agreement" claim in PORT_NOTES §"Toe/heel". Collection is also ~2.2x
faster than the N=2 set was (`wall_fused_s` 117 s vs 257 s) — that set was
collected on the 20-core CPU, this one on the 4070 SUPER.

**12/12 rollouts survived, zero falls.** 23:29 → 23:58, **29 min** wall (2.4 min
each: 8 s sim + 22 s read + 98 s fused filtering). 1.5 GB, 131–132 MB per rollout.
Per-rollout `travelled_m` 3.6–4.3 m (DR resamples the command and includes stands,
so short travel is expected, not a stumble); `slip` 6.8–9.3%; `tilt_max` 5.2–6.5°;
`mu` 0.47–0.85.

### Step 2 — N=4 held-out control set

```
TMPDIR=/tmp/cn_collect_ctl4 uv run python -m invariant_estimation.sim.collect \
  --toe-heel --terrain flat --seeds 0 1 2 --seconds 60 --out data/control4
```
`--dr` omitted, so this is the non-randomised control set. **3/3 survived**,
130 MB each, 23:58 → 00:07 (~9 min). `vx = 0.4` default, matching `data/control`'s
`meta`. One deliberate difference from `data/control`: `--record-slip` was not
passed, so `truth.slip_sat`/`truth.contact_fn` are absent. Nothing downstream of
`cache` reads them.

### Step 3 — cache + frozen norm

Channel gate on `data/dr4/cache/flat_seed000_feat.npz`:
```
channels (62000, 4, 24) float64      <- (T, 4, 24), gate passed
y_fk     (62000, 4, 3)  float64
names    (24,)
```
`data/control4` was cached against `--norm data/dr4/norm_constants.npz`, so the
held-out set is normalised by the constants frozen with the checkpoint.

Norm fit: **2 208 000 samples** (12 rollouts x 46 k post-warm-up ticks x 4
contacts), **zero floored channels**. Segment starts: 545 772 legal over the 12
dr4 rollouts, 136 443 over the 3 control4 ones (`L=128, H=50, stride=8,
span=0.392 s, Nyquist=62.5 Hz` — unchanged from N=2).

### Step 4 — P0, measured per dataset

Both are **21x21** (`9 + 3*4`), which is the first-order check that N=4 reached the
InEKF state and not just the data loader.

```
p0_dr4      eig [3.713e-10, 1.054]
  diag(R) = [7.100e-5, 7.082e-5, 1.0003]      <- yaw unobservable, as it must be
  diag(v) = [8.40e-4, 8.44e-4, 2.49e-4]
  diag(p) = [0.2106, 0.2081, 0.2093]
  diag(d) = same 0.2106/0.2081/0.2093 triple, x4 anchors
p0_control4 eig [3.570e-10, 1.096]
```

Worth noting for its own sake: `diag(p) = diag(d) ~ 0.209` at N=4 against **0.34**
at N=2 (PORT_NOTES, "What the loader guarantees"). Absolute position is
unobservable in both, so this is not an accuracy statement — but four anchors
settle the absolute-position block ~1.6x tighter than two. I did not chase why.

Steps 2-4 wall: 23:58 -> 01:01, **63 min**.

### Step 5 — training

Both runs identical except `--widths` / `--out`:
```
JAX_PLATFORMS=cuda uv run python -u train_contactnet.py train --toe-heel \
  --data data/dr4 --cache data/dr4/cache --norm data/dr4/norm_constants.npz \
  --steps 10000 --objective l2_velocity --B 32 --no-remat --sigma-0 1e-1 --lr 1e-4 \
  --p0 artifacts/p0_dr4.npz --log-every 50 --widths {256 256 | 128 64} \
  --out artifacts/contactnet_run6_{w256|w128}.npz
```

**w256 gates, all green.** `network: d_in=1200, widths=(256, 256), 374790 params`
— exactly run 5's shape.

```
sigma_rel_err            3.469e-16      <- ~1e-16, gate passed
trunk_grad_max           0.0
trunk_grad_exactly_zero  True           <- gate passed
head_grad_max            5.786e-3       <- > 0, gate passed
loss_at_init             1.549e-3
windows_finite           True
chains: 32 seeded + warmed in (1.0s each, episode 43s) in 29s
```

`applied = 1.00e+00` at every logged step. Loss trace over the first 600:
3.555e-3 (step 0) -> **1.171e-1 (step 50, the documented spike)** -> 2.312e-2
(100) -> 5.227e-3 (200) -> 2.393e-3 (300) -> 1.463e-3 (550). Falling well before
step 500, so the "still rising at 500" failure condition did not fire. Run 5's
spike was 1.1e-1; ours is 1.17e-1 — the same behaviour.

**Step rate measured over a 4-minute window: 500 steps / 240 s = 0.480 s/step**,
so 10 000 steps is ~80 min. Both runs therefore ran the **full 10 000 steps** — no
reduction was needed.

> **An observation I did not expect.** `loss_at_init` is **1.549e-3** here against
> run 5's starting loss of **3.96e-3** at N=2, and step 0 is 3.555e-3 against run
> 5's 3.96e-3. The `l2_velocity` loss at the *untrained* loose init is already
> ~2.6x lower at N=4. **Hypothesis** (not measured): this is the analytic toe/heel
> benefit showing up in the objective itself — four anchors constrain body-frame
> velocity better than two before any learning happens. If so, the N=4 network has
> less headroom to win, because its starting point is already better. Testable by
> comparing `loss_at_init` at N=2 and N=4 on the *same* rollouts, which I did not
> do (the datasets differ).

**w256 finished: 10 000 steps in 4850 s (80.8 min), 0.49 s/step.**

| | w256 (256, 256) |
|---|---|
| params | 374 790 |
| wall | 4850 s = 80.8 min |
| loss, median of first 50 | 1.576e-1 (the spike) |
| loss, median of last 1000 | **7.663e-4** |
| loss, min / final | 3.454e-4 / 1.105e-3 |
| plateau check: last 2000 vs prev 2000 | 7.697e-4 vs 8.322e-4 (8% — plateaued) |
| `nis_over_dof`, median of last 1000 | **2.068e-2** |
| `applied_frac` min / mean | **1.000000 / 1.000000** |
| `cond_proxy_max` peak | 3.490e4 (gate is 1e9) |
| non-finite losses | 0 |

Every gate green: not one gated update in 10 000 steps, condition proxy four
decades inside its gate, no non-finite loss.

**w128: `d_in=1200, widths=(128, 64), 162374 params`** — matches the brief. Its
`check-init` is identical to w256's, including `loss_at_init = 1.5492653875174094e-3`
against w256's `1.5492653875173804e-3` — **agreeing to 13 digits**, which is the
right answer and a free consistency check: `network.init` zeroes the output head, so
iteration 0's emitted `Σ_C` cannot depend on trunk width, and the 1e-13 residual is
float summation order alone.

Checkpoint shapes verified on disk (this matters — `train.load_params` takes tree
*structure* from a reference and would silently accept mismatched shapes):

```
w256  [(256,1200),(256,),(256,256),(256,),(6,256),(6,)]  = 374790 params
w128  [(128,1200),(128,),(64,128),(64,),(6,64),(6,)]     = 162374 params
```

### Both runs, side by side

| | w256 (256, 256) | w128 (128, 64) |
|---|---|---|
| params | 374 790 | 162 374 |
| wall | 4850 s (80.8 min) | **4567 s (76.1 min)** |
| s/step | 0.49 | 0.46 |
| loss, median of first 50 (spike) | 1.576e-1 | 2.053e-1 |
| **loss, median of last 1000** | **7.663e-4** | 8.737e-4 |
| loss min / final | 3.454e-4 / 1.105e-3 | 3.503e-4 / 1.049e-3 |
| last 2000 vs prev 2000 | 7.697e-4 vs 8.322e-4 | 8.668e-4 vs 9.171e-4 |
| **`nis_over_dof`, last 1000** | **2.068e-2** | 3.129e-2 |
| `applied_frac` min / mean | 1.000000 / 1.000000 | 1.000000 / 1.000000 |
| `cond_proxy_max` peak | 3.490e4 | 3.109e4 |
| non-finite losses | 0 | 0 |

**The speed benchmark reproduces and is even weaker than advertised: w128 is only
1.06x faster** (4567 s vs 4850 s), not the ~1.1x expected. The BPTT scan through the
InEKF dominates so completely that halving-and-quartering the trunk buys 5% of wall
time. Speed is not a reason to prefer either width, exactly as the brief said.

Both plateaued (last-2000 within 8% / 6% of the preceding 2000). Neither gated a
single update in 10 000 steps.

### Step 6 — evaluation

Protocol per checkpoint: 3 rollouts x 2 truth seeds x 20 s (20 000 ticks), matching
run 5's `3x2x20 s`. The `heuristic` column is the **N=4 analytic filter** (the
recorded Schmitt-switched `contact_chol`), i.e. every table below is
"learned Σ_C against the analytic toe/heel filter", not against N=2.

---

## 3. Results tables

### 3a. Replay, in sample (`data/dr4`, `p0_dr4`)

| metric | N=4 analytic | **w256 trained** | ratio | *N=2 analytic (run 5)* | *N=2 + run 5* |
|---|---|---|---|---|---|
| `slope_e_pz` [m/s] | −0.01019 | **+0.00178** | −0.175 | −0.03751 | +0.00356 |
| `e_vz` [m/s] | −0.00866 | +0.00303 | −0.350 | — | — |
| `ratio` [-] | 1.179 | 9.113 (degenerate, see below) | 7.730 | 1.162 | 1.063 |
| `vel_rms` [m/s] (scored by L2) | 0.04049 | **0.03318** | 0.819 | 0.0771 | 0.0378 |
| `pos_rms` [m] | 0.12008 | 0.08709 | 0.725 | — | — |
| `height_rms` [m] | 0.11662 | **0.03193** | 0.274 | 0.3993 | 0.0636 |
| `height_final` [m] | 0.20338 | 0.03928 | 0.193 | — | — |
| `tilt_deg` | 0.34943 | 0.39105 | **1.119** | 0.590 | 0.365 |

Per-arm, all six improved in both velocity and height — no arm went the wrong way.

**Three things to read off this table.**

1. **The analytic N=4 filter is already 3.7x better on the sink than analytic N=2**
   (−0.01019 vs −0.03751) and 3.4x better on `height_rms` (0.1166 vs 0.3993). Most
   of what run 5 had to learn, toe/heel supplies for free.
2. **The composed system is the best absolute number on record here**: sink
   +0.00178 against run 5's +0.00356 (2.0x smaller in magnitude), `height_rms`
   0.0319 against 0.0636 (2.0x), `vel_rms` 0.0332 against 0.0378 (1.14x).
3. **`tilt_deg` REGRESSED 1.119x** (0.349 → 0.391). This is the opposite sign from
   run 5, which *improved* tilt 1.6x (0.590 → 0.365) — but only because the N=2
   analytic baseline it started from was so much worse. Measured against a
   baseline that is already good, the learned `Σ_C` costs tilt. `l2_velocity` has
   no attitude term, so nothing was defending it.

> **The `ratio` acceptance check has gone degenerate, and I do not think it means
> anything here.** `ratio = slope(e_pz)/mean(e_vz)` was the check that residual
> drift is still a clean integrated velocity bias, and run 5 kept it at ≈1.06.
> At N=4 it reads 9.11 — but the numerator and denominator are now **+0.00178 and
> +0.00303**, both essentially zero, and the printed column is the mean of
> per-arm quotients, so a near-zero denominator on any single arm dominates. A
> ratio of two vanishing quantities is not evidence of anything. **The check
> needs redefining (e.g. gate it on `|slope| > some floor`) before it is quoted
> again.** Flagging this as a harness limitation, not a result.

Also note `replay_eval`'s own auto-verdict printed *"helps on every axis
measured"* — it is wrong, because its rule only inspects `vel_rms`, `height_rms`
and `pos_rms`. Tilt regressed. Do not trust that line.

### 3b. Replay, in sample — both widths

| metric | N=4 analytic | w256 | w128 | better |
|---|---|---|---|---|
| `slope_e_pz` [m/s] | −0.01019 | +0.00178 | **−0.00089** | w128 |
| `vel_rms` [m/s] ← scored | 0.04049 | 0.03318 | **0.03274** | w128 |
| `pos_rms` [m] | 0.12008 | 0.08709 | **0.08490** | w128 |
| `height_rms` [m] | 0.11662 | **0.03193** | 0.03237 | w256 |
| `height_final` [m] | 0.20338 | **0.03928** | 0.04172 | w256 |
| `tilt_deg` | 0.34943 | **0.39105** | 0.40166 | both regress |
| `ratio` [-] | 1.179 | 9.113 (degenerate) | 0.491 | — |

### 3c. Replay, HELD OUT (`data/control4`, never trained on, `p0_control4`)

The generalisation test, and the one that decides the width.

| metric | N=4 analytic | w256 | ratio | w128 | ratio |
|---|---|---|---|---|---|
| `slope_e_pz` [m/s] | −0.01245 | +0.00515 | −0.414 | **−0.00387** | 0.311 |
| `vel_rms` [m/s] ← scored | 0.02897 | 0.02944 | **1.016 (WORSE)** | **0.02766** | 0.955 |
| `pos_rms` [m] | 0.15594 | 0.08184 | 0.525 | **0.07658** | 0.491 |
| `height_rms` [m] | 0.15324 | **0.04850** | 0.316 | 0.06410 | 0.418 |
| `height_final` [m] | 0.25926 | **0.09029** | 0.348 | 0.09706 | 0.374 |
| `tilt_deg` | 0.29231 | 0.42840 | 1.466 | 0.44813 | 1.533 |
| `ratio` [-] | 1.233 | 0.929 | — | 1.164 | — |

**`ratio` is well-behaved held out** (0.93 and 1.16, against the degenerate 9.11 in
sample), which supports reading the in-sample 9.11 as the two-vanishing-quantities
artefact described above rather than a change of mechanism.

### 3d. Closed loop, 30 s at `vx = 0.6`, `--imu-noise`, seed 0

`N=2` columns are the previously recorded runs, re-tabulated here for comparison.

| metric | N=2 analytic | N=2 + run 5 | **N=4 analytic** | N=4 + w256 | N=4 + w128 |
|---|---|---|---|---|---|
| final signed dz [m] | −3.1936 | −0.1797 | −0.5673 | −0.2204 | **+0.0696** |
| dz RMS last half [m] | 2.4345 | 0.1432 | 0.4315 | 0.1725 | **0.0499** |
| sink rate last 20 s [m/s] | −0.1079 | −0.0064 | −0.0192 | −0.0081 | **+0.0019** |
| tilt rms [deg] | 1.2761 | 0.5460 | **0.4298** | 0.6035 | 0.6057 |
| tilt TAIL [deg] | 1.2102 | 0.3696 | **0.2398** | 0.4374 | 0.4891 |
| attitude rms [deg] | 1.5521 | 1.7560 | **0.8061** | 2.3993 | 3.5954 |
| attitude TAIL [deg] | 1.6820 | 2.2552 | **0.9286** | 3.0944 | 4.7391 |
| **YAW rms [deg]** | 0.8836 | 1.6689 | **0.6820** | 2.3221 | 3.5440 |
| **YAW TAIL [deg]** | 1.1681 | 2.2247 | **0.8971** | 3.0633 | 4.7138 |
| vel err rms [m/s] | 0.1343 | **0.0304** | 0.0347 | 0.0375 | 0.0474 |
| pos err rms [m] | 1.8846 | **0.2388** | 0.3375 | 0.2706 | 0.4740 |
| gyro err rms | 0.0116 | 0.0101 | **0.0094** | 0.0099 | 0.0096 |
| NIS tail mean | 0.6908 | 0.1994 | 0.1302 | 0.3889 | **0.7191** |

**Replay and the closed loop disagree about velocity**, and the closed loop should
be believed for deployment questions. Held-out replay says w128 is better on
`vel_rms` (0.0277 vs 0.0294); the closed loop says w256 (0.0375 vs 0.0474). The
difference is that replay runs from a truth seed with the policy driven by *truth*,
while the closed loop has the policy reacting to the estimate — so a worse estimate
changes the gait, and the two are not measuring the same system. Both are reported;
neither is dismissed.

---

## 4. Did the yaw fix survive training?

**No. It inverted, and worse than at N=2.**

| | yaw TAIL [deg] | vs N=4 analytic |
|---|---|---|
| N=2 analytic | 1.168 | — |
| N=2 + run 5 | 2.225 | — |
| **N=4 analytic** | **0.897** | best on record |
| N=4 + w256 | 3.063 | **3.4x worse** |
| N=4 + w128 | 4.714 | **5.3x worse** |

The premise of the toe/heel work was that run 5's yaw regression came from missing
orientation observability, and that two contact points per foot would restore it.
On the **analytic** filter that was correct — 0.897° is the best yaw on record. But
a network trained on top of it gives the observability straight back, and gives back
*more* than it did at N=2: 4.71° against run 5's 2.22°.

The mechanism is not mysterious. Yaw is constrained only by the contact update
(planted feet at distinct world locations), loosening `Σ_C` is how the network buys
its sink reduction, and `l2_velocity` contains **no attitude term at all** — so yaw
drift is free. Doubling the contact count doubles the anchor freedom available to
spend, and the network spends it. The monotone ordering (analytic < w256 < w128, and
the smaller net worse) is consistent with that reading: less capacity means less
ability to loosen *selectively*, so it loosens more indiscriminately.

**Stated as a hypothesis, not a result:** I did not isolate *which* contacts the
network loosens or *when*, so "it loosens indiscriminately" is inference from the
ordering plus the objective's structure, not a measurement. Logging `Σ_C` per
contact against gait phase would settle it, and is the cheapest next diagnostic.

---

## 5. Width comparison

**(128, 64) is the better width, and the reason is generalisation, not speed.**

* **Speed is a non-argument, as predicted.** 4567 s vs 4850 s = **1.06x**. The BPTT
  scan through the InEKF dominates so completely that cutting the trunk from 374 790
  to 162 374 parameters buys 5% of wall time.
* **w256 overfits on the metric it optimises.** In-sample `vel_rms` ratio 0.819
  (good); held-out **1.016** — i.e. out of sample it made velocity *worse than the
  analytic filter*. w128: 0.809 in sample, **0.955** held out. That gap between
  in-sample and held-out performance, present for the larger net and absent for the
  smaller one, is the over-parameterisation signature, and it is the strongest single
  piece of evidence in this report.
* w128 also wins held-out `|slope_e_pz|` (0.00387 vs 0.00515) and `pos_rms` (0.0766
  vs 0.0818), and is much better vertically in the closed loop.
* **w256 wins `height_rms` held out** (0.0485 vs 0.0641) and closed-loop velocity /
  position. So this is not a clean sweep, and anyone optimising for height
  specifically should prefer w256.

Caveat worth stating plainly: this is **one training seed per width**. The
in-sample/held-out gap is a real and interpretable signal, but a single pair of runs
cannot separate "the smaller net generalises better" from "this particular w256 run
was unlucky". Two more seeds per width would settle it and cost ~5 h.

---

## 6. `nis_over_dof`

| | training, last 1000 | closed-loop NIS tail |
|---|---|---|
| N=2 + run 5 | 2.175e-2 | 0.199 |
| N=4 analytic | — | 0.130 |
| N=4 + w256 | 2.068e-2 | 0.389 |
| N=4 + w128 | 3.129e-2 | **0.719** |

Calibrated is 1.0. Every arm is still **over-conservative** (`S` larger than the
innovations warrant), which is the known `l2_velocity` gap: the quadratic term
constrains only *ratios* of `S`, and the `logdet` term that fixes absolute scale is
what `beta_nll` adds.

Two observations. Training moves NIS *toward* calibration rather than away
(0.130 → 0.389 → 0.719), and **w128 is much the closest to 1.0**, which is a second
independent reason to prefer it. But at 0.72 it is still ~1.4x conservative, so
**neither checkpoint will pass G10's NIS/NEES consistency bands.** That remains
`beta_nll`'s job, and this run does not change that conclusion.

---

## 7. What had to be fixed

### `experiments/replay_eval.py` had no way to build an N=4 estimator (blocking, fixed)

`main()` hardcoded `collect.build_collector(verbose=False)` and
`ContactNetConfig(F=24, sigma_0=1e-4)`, both N=2. A rollout stores `(T, N, 3, 3)`
contact arrays, so scoring an N=4 dataset would mismatch on the first tick.
Added a `--toe-heel` flag (uncommitted; see §7 diff at the end of this report).

### `features.window` OOMs at N=4 (blocking, fixed)

`replay_eval` windowed the whole rollout up front: at N=4 that materialises
`62000 x 4 x 50 x 24 x 8` = **2.38 GB**, and the `swapaxes` needs input and output
resident simultaneously — 4.8 GB of device memory for a quantity only ever consumed
`args.ticks` rows at a time. It fit at N=2, which is why this only surfaced now.
Split by *where* rather than *what*: `boxcar` is the only arithmetic and stays on
device (47 MB); the gather and transpose are pure data movement and moved to the
host, gathered per-`t0`. Gather is elementwise-independent, so gathering sliced
indices is bit-identical to slicing a whole-rollout gather.

### Recommended, NOT done

* **An attitude term in the objective, or `beta_nll`.** This is the single change
  most likely to fix the yaw regression, and §4 is the argument for it.
* **Log `Σ_C` per contact against gait phase.** Needed to turn §4's hypothesis
  ("it loosens indiscriminately") into a measurement.
* **Redefine the `ratio` acceptance check** so it cannot go degenerate when both
  its terms approach zero — gate it on `|slope|` exceeding a floor.
* **Two more training seeds per width**, to separate the over-parameterisation
  reading from run-to-run luck.

---

## 8. Limitations

* **One training seed per width.** The central width claim rests on a single pair of
  runs (§5).
* **One closed-loop noise seed.** All closed-loop numbers are `--noise-seed 0`. The
  brief's stretch goal of a second seed was not reached because the run died at 04:18.
* **The closed loop is not bit-reproducible.** Two identical invocations differ by
  ~1.4e-6 in final `dz`; anything below ~1e-5 here is noise. Every difference quoted
  in §3d is 1.3x or larger, well clear of it.
* **Replay and the closed loop disagree on velocity** (§3d). Two different systems,
  both reported.
* **`data/dr4` has 13 rollouts and 12 caches.** 12 were collected as planned; the
  13th is a stray from the pre-flight smoke collection that shares the directory and
  is *not* in the cache, so it did not enter training. Harmless, but it should be
  deleted before anyone counts the dataset.
* **The N=4 analytic baseline is itself new**, measured in this run rather than
  independently replicated.
* **Nothing here tests slip**, which is the other thing a learned contact covariance
  is supposed to buy. Held for a separate session.
