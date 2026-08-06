# Getting the reseed / rolling-anchor work onto a branch

`reseed_rolling_anchor.patch` (repo root) applies **cleanly onto `572a4bb`**, your
current HEAD — the est-n8 commit. Verified with `git apply --check`, rc=0.

```bash
git checkout -b reseed/rolling-anchor          # off 572a4bb
git apply --index reseed_rolling_anchor.patch  # applies AND stages, nothing else swept
git commit -m "inEKF: rolling-anchor contact density + touchdown reseed"
```

`--index` stages exactly the patch's 16 files, so untracked `.claude/` and the
stray `.mp4` are left alone.

**`run_estimator.py` was rebased onto `572a4bb` by hand.** It is the one file both
changes touch — your `--contacts-per-foot` / `_check_contact_geometry` and my
`--reseed` / `--rolling` land in the same argparse block and the same
`make_estimated_loop` signature. The merged file keeps both and parses; the other
15 files never conflicted.

Then clean up the stale worktree this work was originally done in (superseded —
it is based on `1408b35` and does **not** have the est-n8 fix):

```bash
rm -f .git/worktrees/reseed/*.lock
git worktree remove --force .worktrees/reseed
git branch -D reseed/touchdown && git worktree prune
```

> Why you could not find any of this: I did it in a worktree at `.worktrees/reseed`,
> hidden two ways at once — the leading dot hides it from Finder (`Cmd+Shift+.`),
> and I added `.worktrees/` to `.git/info/exclude` so `git status` stayed clean.
> My mistake; a visible branch was the right call.

---

## What is in it

Two independent features, both **off by default** (`config/filter_cfg.yaml`), so
every recorded gate number reproduces without flags. The disabled path is
verified bit-identical to the pre-change checkout over 20000 ticks x 3 axes.

### 1. Touchdown re-seed — implemented, measured, does NOT fix the drift

`src/invariant_estimation/inEKF/reseed.py` + `tests/inEKF/test_reseed.py` (14).
Covariance congruence `P_dd = P_pp + R N R^T`, `P_theta_d = P_theta_p` under a
fire-once `TouchdownReseedLatch`. The three Java `InvariantEKFReseedTest`
properties pass, and the latch fires 47x against 46 touchdown edges.

Vertical drift: **-0.0415 -> -0.0420 m/s (1.01x).** Nothing.

It cannot work: the re-seed re-anchors onto the *current base estimate*, so it is
common-mode preserving by construction, and the damage happens at toe-off (end of
stance), not at touchdown. Kept because it is correct and cheap and is the right
thing to have once the anchor model is right. `reseed.enabled: false`.

### 2. Rolling-anchor contact density — this is the fix

`src/invariant_estimation/inEKF/contact.py::rolling_anchor_density` +
`tests/inEKF/test_rolling_anchor.py` (9).

The shipped model asserts `d_dot_i = 0`. Rigid-body kinematics plus no-slip says
`d_dot_i = omega x r_i`. `omega` is measured (base gyro + leg encoders); `r_i` is
not. Pushing an unknown `r_i ~ (0, sigma_r^2 I)` through the known map gives

```
Sigma_C += tau * sigma_r^2 * ( |omega|^2 I_3  -  omega omega^T )
```

rank 2, null along `omega`, exactly zero when the foot is not rotating. No
contact detection, no terrain model, no learned component.

| arm | flat_seed005 | flat_seed020 |
|---|---|---|
| shipped | -0.0415 m/s | -0.0551 m/s |
| touchdown reseed | -0.0420 (1.01x) | -0.0556 (1.01x) |
| **rolling anchor** | **-0.0075 (0.18x)** | **-0.0058 (0.11x)** |

`tau = 0.25 s`, `sigma_r = 0.0985 m`, both untuned — they fall out of the
geometry. Two results that magnitude-fitting cannot produce: the error flips from
`LINEAR (biased)` to `SQRT (diffusive)` on both rollouts, and the touchdown
concentration collapses 3.0x -> 0.2x.

---

## Caveats, in the order they could bite

1. **Everything above is OPEN-LOOP replay** (`experiments/reseed_drift.py`): the
   InEKF consumes a recorded sensor stream and the policy is not driven by the
   estimate. Closed loop has never been run. There is no video.
2. **N=2, flat ground, two 20 s rollouts.** `sigma_r` is the sole-centre value;
   at `--contacts-per-foot 4` it is the wrong quantity and must be re-derived.
   Do not flip `--rolling` and N=8 on together without that, or the two effects
   are confounded.
3. **`NIS/dof` gets worse** (0.315 -> 0.058). Expected — this adds process noise.
   It fixes the *ratio* `P_pp : P_dd` (attribution) and does nothing for the
   *scale* of `S` (calibration). Those are independent defects; calibration is
   now the dominant remaining one.
4. **`572a4bb` (`.claude/worktrees/est-n8`) is not merged here.** It applies
   cleanly except `run_estimator.py`, where your `--contacts-per-foot` plumbing
   and my `--reseed` / `--rolling` plumbing touch the same argparse block and the
   same `make_estimated_loop` signature. Both sides additive; mechanical merge.
   None of the numbers above are affected — every run was N=2 with the analytic
   filter and no ContactNet checkpoint, so there was no geometry to mismatch.
5. One of nine `tests/pipeline` tests never finished inside the sandbox's 45 s
   command cap. `test_step_does_not_recompile_across_conditions` (the I7
   constant-graph gate) is among those that passed.

---

## Running it

After the branch is checked out, from the repo root (`data/` is right here, which
the worktree needed a symlink for):

```bash
# closed loop + video — NEVER RUN. This is the untested next step, not a
# recording of a tested one. 1500 control ticks @ 50 Hz = 30 s, matching the
# Finding 2 window. --contacts-per-foot stays at 1: this is the ANALYTIC filter,
# no checkpoint, so there is no trained geometry to match.
uv run python run_estimator.py --video results/rolling_off.mp4 --ghost full \
    --ticks 1500 --vx 0.4 --out results/rolling_off.npz
uv run python run_estimator.py --video results/rolling_on.mp4 --ghost full \
    --ticks 1500 --vx 0.4 --rolling --out results/rolling_on.npz

# the open-loop measurement the numbers above come from, all three arms
uv run python experiments/reseed_drift.py --rollout data/flat_seed005.npz \
    --ticks 20000 --arm all

# the no-filter diagnostic: is the FK contact point actually static in stance?
uv run python experiments/anchor_static_check.py --rollout data/flat_seed005.npz
```

What to look for in the video: `sim/ghost.py` documents the drift as "the ghost
steadily sinking through the floor". In `rolling_off.mp4` it should; in
`rolling_on.mp4` it should stay inside the robot. If both sink identically, the
flag is not reaching the filter — check the banner, and that
`fused.ekf.rolling.enabled` is True.

Full write-ups: `PORT_NOTES.md` Finding 2 (CLOSED) and Finding 3, and
`DESIGN_DECISIONS.md` §3 — all inside the worktree.
