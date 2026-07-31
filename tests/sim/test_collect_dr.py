"""Domain-randomised command sampling — `collect._command_schedule` and its config.

Two obligations, and the first is the one with teeth:

1. **`mode: uniform` is bit-for-bit what it was before `hollow` existed.** `data/dr` and
   `data/dr4` were collected with it and runs 4-6 are scored against them; a shifted RNG stream
   would silently make those datasets unreproducible. The oracle is the shipped dataset itself —
   every rollout records both the `dr` config it ran under and the `cmd_schedule` it realised, so
   the sampler can be replayed against real recorded output rather than against a paraphrase of
   itself.
2. **`mode: hollow` never emits a command inside the policy's tracking deadband.** That is the
   entire point of the mode (see `config/collect_dr5.yaml`), so it is asserted directly rather
   than inferred from the parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from invariant_estimation.sim import collect as C

REPO = Path(__file__).resolve().parents[2]
CONTROL_DT = 0.02
HEIGHT_RANGE = (0.83, 0.93)          # `run_policy.POLICIES["baseline"]["height_range"]`
DEADBAND = {"vx": 0.30, "vy": 0.28, "yaw": 0.60}      # `run_policy.WALK_MIN_*`


def _schedule(dr: C.DomainRandomization, *, seed: int = 0, walk_tick: int = 100,
              total_ticks: int = 3100) -> dict[int, list[float]]:
    return C._command_schedule(dr, dr.rng(seed), walk_tick=walk_tick, total_ticks=total_ticks,
                               control_dt=CONTROL_DT, height_range=HEIGHT_RANGE)


# ---------------------------------------------------------------------------
# 1. the legacy path
# ---------------------------------------------------------------------------

def _dr_from_meta(meta_dr: dict) -> C.DomainRandomization:
    """`DomainRandomization.to_meta()` inverted — lists back to tuples, dict values too."""
    kw = {}
    for k, v in meta_dr.items():
        if k == "friction_range_by_terrain":
            kw[k] = {t: tuple(r) for t, r in (v or {}).items()}
        elif isinstance(v, list):
            kw[k] = tuple(v)
        else:
            kw[k] = v
    return C.DomainRandomization(**kw)


@pytest.mark.parametrize("rollout", ["dr/flat_seed000.npz", "dr4/flat_seed000.npz"])
def test_uniform_mode_reproduces_the_shipped_datasets_exactly(rollout):
    """Replay the recorded config through today's sampler; the schedule must match to the bit.

    This is the regression that protects runs 4-6: `data/dr` and `data/dr4` are the datasets those
    checkpoints were trained on, and their `meta` carries everything needed to redraw the commands
    — the `dr` config, the seed, the settle length. The RNG is replayed in `collect_rollout`'s own
    order (friction, then pushes, then commands) because the three share one stream, so an extra or
    missing draw anywhere upstream would shift the commands.
    """
    path = C.DATA_DIR / rollout
    if not path.exists():
        pytest.skip(f"{path} not collected on this machine (data/ is gitignored)")
    meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
    if meta.get("dr") is None:
        pytest.skip(f"{rollout} is a pre-DR rollout")

    dr = _dr_from_meta(meta["dr"])
    assert dr.command_mode == "uniform", "a pre-hollow rollout must default to the legacy mode"

    rng = dr.rng(meta["seed"])
    if dr.friction:
        mu = float(rng.uniform(*dr.friction_for(meta["terrain"])))
        assert mu == pytest.approx(meta["friction_mu"], abs=0.0), \
            "the friction draw itself must reproduce, or the stream is already out of step"
    C._push_schedule(dr, rng, meta["settle_s"], meta["settle_s"] + meta["seconds"])

    settle_ticks = int(round(meta["settle_s"] / CONTROL_DT))
    total_ticks = settle_ticks + int(round(meta["seconds"] / CONTROL_DT))
    got = C._command_schedule(dr, rng, walk_tick=settle_ticks, total_ticks=total_ticks,
                              control_dt=CONTROL_DT, height_range=HEIGHT_RANGE)

    # `meta["cmd_schedule"]` is [[t_seconds, vx, vy, yaw, stand, height], ...].
    want = {int(round(row[0] / CONTROL_DT)): list(row[1:]) for row in meta["cmd_schedule"]}
    assert sorted(got) == sorted(want), "resample instants moved"
    for k in want:
        assert got[k] == want[k], f"command at tick {k} moved: {got[k]} != {want[k]}"


def test_uniform_mode_matches_an_independent_reference_generator():
    """Self-contained twin of the dataset test: the draw ORDER is (h, stand, vx, vy, yaw, dt).

    Written out longhand so that reordering the draws inside `_command_schedule` — which changes
    every downstream rollout while leaving each marginal distribution correct, and is therefore
    invisible to a distributional test — fails here.
    """
    dr = C.DomainRandomization(command=True, seed=1234)
    rng = np.random.default_rng(99)
    lo, hi = HEIGHT_RANGE
    want, k = {}, 100
    while k < 400:
        h = float(rng.uniform(lo, hi))
        if float(rng.random()) < dr.stand_prob:
            want[k] = [0.0, 0.0, 0.0, 1.0, h]
        else:
            want[k] = [float(rng.uniform(*dr.vx_range)), float(rng.uniform(*dr.vy_range)),
                       float(rng.uniform(*dr.yaw_rate_range)), 0.0, h]
        k += max(1, int(round(float(rng.uniform(*dr.command_resample_s)) / CONTROL_DT)))

    got = C._command_schedule(dr, np.random.default_rng(99), walk_tick=100, total_ticks=400,
                              control_dt=CONTROL_DT, height_range=HEIGHT_RANGE)
    assert got == want


# ---------------------------------------------------------------------------
# 2. the hollow path
# ---------------------------------------------------------------------------

HOLLOW = C.DomainRandomization(
    command=True, command_mode="hollow", seed=20260730,
    vx_mag=(0.30, 0.90), vy_mag=(0.28, 0.50), yaw_mag=(0.60, 1.50),
    vx_zero_prob=0.15, vy_zero_prob=0.35, yaw_zero_prob=0.35, stand_prob=0.10)


def _walking(sched: dict[int, list[float]]) -> np.ndarray:
    """`(n, 3)` of the vx/vy/yaw commands on the events that are not stands."""
    ev = np.asarray([v for v in sched.values()])
    return ev[ev[:, 3] < 0.5][:, :3]


def test_hollow_never_emits_a_command_inside_the_deadband():
    """The defining property: every axis is exactly 0.0 or at/above its deadband.

    Exactly `0.0`, not "small" — a 0.1 m/s command and a 0.0 command produce the same standing
    robot, and only the second says so. That is why the assertion is on equality with zero.
    """
    v = np.concatenate([_walking(_schedule(HOLLOW, seed=s, total_ticks=20000))
                        for s in range(20)])
    assert len(v) > 2000, "not enough events to make this assertion mean anything"
    for j, (axis, mag) in enumerate(zip(("vx", "vy", "yaw"),
                                        (HOLLOW.vx_mag, HOLLOW.vy_mag, HOLLOW.yaw_mag))):
        a = np.abs(v[:, j])
        nz = a[a > 0.0]
        assert (nz >= mag[0] - 1e-12).all(), (
            f"{axis}: {(nz < mag[0]).sum()} commands landed between 0 and the deadband "
            f"{DEADBAND[axis]} — those ticks are a standing robot recorded as a walk command")
        assert (nz <= mag[1] + 1e-12).all(), f"{axis}: drew past its trained band"
        assert mag[0] >= DEADBAND[axis] - 1e-12, f"{axis}: the config's own floor is under the "
        f"policy's deadband, so the mode cannot deliver its guarantee"


def test_hollow_hits_its_zero_rates_and_is_sign_balanced():
    """Both halves matter: the zero mass is the point of the mode, the sign balance is the point
    of "side to side" and "yaw left and right" rather than a drifting bias in one direction."""
    v = np.concatenate([_walking(_schedule(HOLLOW, seed=s, total_ticks=20000))
                        for s in range(48)])
    n = len(v)
    assert n > 5000
    for j, (axis, p0) in enumerate(zip(("vx", "vy", "yaw"),
                                       (HOLLOW.vx_zero_prob, HOLLOW.vy_zero_prob,
                                        HOLLOW.yaw_zero_prob))):
        a = v[:, j]
        # 4 sigma of a binomial proportion — wide enough not to flake, tight enough that a
        # swapped or dropped probability (0.15 vs 0.35 here) cannot pass.
        tol = 4.0 * np.sqrt(p0 * (1 - p0) / n)
        assert abs((a == 0.0).mean() - p0) < tol, f"{axis}: zero rate off"
        nz = a[a != 0.0]
        assert abs((nz > 0).mean() - 0.5) < 4.0 * np.sqrt(0.25 / len(nz)), f"{axis}: sign skew"


def test_hollow_is_the_diversity_the_uniform_mode_was_missing():
    """The comparison that justifies the mode, made on the sampler rather than on a 30-minute
    collection: the fraction of walking commands the policy actually tracks, per axis.

    `data/dr4` measures 62 / 19 / 13 % at the tick level. The event-level number here is the same
    quantity before the dwell weighting, so it is compared against `uniform` recomputed the same
    way, never against the recorded tick figure.
    """
    old = C.DomainRandomization(command=True, seed=20260730, vx_range=(-0.5, 0.8),
                                vy_range=(-0.35, 0.35), yaw_rate_range=(-0.8, 0.8))
    frac = lambda v: [float((np.abs(v[:, j]) >= d).mean())        # noqa: E731
                      for j, d in enumerate(DEADBAND.values())]
    n = 20000
    f_old = frac(np.concatenate([_walking(_schedule(old, seed=s, total_ticks=n))
                                 for s in range(24)]))
    f_new = frac(np.concatenate([_walking(_schedule(HOLLOW, seed=s, total_ticks=n))
                                 for s in range(24)]))
    assert f_new[1] > 0.55 and f_old[1] < 0.30, f"lateral: {f_old[1]:.2f} -> {f_new[1]:.2f}"
    assert f_new[2] > 0.55 and f_old[2] < 0.30, f"yaw: {f_old[2]:.2f} -> {f_new[2]:.2f}"
    assert f_new[0] > 0.80, f"forward: {f_new[0]:.2f}"


def test_hollow_axes_draw_from_independent_positions_in_the_stream():
    """Changing one axis's parameters must not move another axis's draws.

    `_hollow` spends a fixed three draws per axis (zero test, magnitude, sign) precisely so this
    holds; a shortcut that returns early without consuming them would couple the axes and make a
    parameter sweep on one axis silently redraw the others.
    """
    a = _schedule(HOLLOW, seed=3, total_ticks=2000)
    b = _schedule(C.DomainRandomization(**{**HOLLOW.__dict__, "yaw_mag": (0.60, 1.20)}),
                  seed=3, total_ticks=2000)
    assert sorted(a) == sorted(b)
    va = np.asarray([a[k] for k in sorted(a)])
    vb = np.asarray([b[k] for k in sorted(b)])
    assert np.array_equal(va[:, [0, 1, 3, 4]], vb[:, [0, 1, 3, 4]]), \
        "narrowing the yaw range moved vx/vy/stand/height"
    assert not np.array_equal(va[:, 2], vb[:, 2]), "narrowing the yaw range changed nothing"


# ---------------------------------------------------------------------------
# 3. config plumbing
# ---------------------------------------------------------------------------

def test_the_shipped_dr5_config_parses_and_is_hollow():
    dr, run = C.load_dr_config(REPO / "config" / "collect_dr5.yaml")
    assert dr.command and dr.command_mode == "hollow"
    assert dr.vx_mag == (0.30, 0.90) and dr.vy_mag == (0.28, 0.50) and dr.yaw_mag == (0.60, 1.50)
    assert run["out_dir"] == "data/dr5"
    # The floors ARE the deadbands; a config that lowered one would defeat the mode silently.
    for mag, d in zip((dr.vx_mag, dr.vy_mag, dr.yaw_mag), DEADBAND.values()):
        assert mag[0] >= d - 1e-12


def test_the_shipped_dr_config_is_still_uniform():
    """`collect_dr.yaml` must keep producing `data/dr4`'s distribution — it names no mode, and the
    default has to stay `uniform` for that to hold."""
    dr, _ = C.load_dr_config(REPO / "config" / "collect_dr.yaml")
    assert dr.command_mode == "uniform"


def test_a_bad_mode_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="uniform.*hollow"):
        C.DomainRandomization.from_dict({"command": {"enabled": True, "mode": "isotropic"}})


def test_a_typo_in_a_hollow_key_is_rejected():
    with pytest.raises(KeyError, match="vx_magnitude"):
        C.DomainRandomization.from_dict({"command": {"enabled": True, "vx_magnitude": [1, 2]}})


def test_max_speed_follows_the_mode():
    """The on-field pre-flight reads this; in hollow mode `vx_range` is not what will be issued."""
    assert HOLLOW.max_speed(0.4) == pytest.approx(float(np.hypot(0.90, 0.50)))
    assert C.DomainRandomization(command=False).max_speed(0.4) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 3. mu stratification (`friction.grid_by_terrain`)
# ---------------------------------------------------------------------------

def test_the_shipped_datasets_realise_only_a_handful_of_mu():
    """The defect the grid exists to fix, asserted against the shipped data.

    Not a paraphrase of the sampler: this replays `friction_value` for every
    (terrain, seed) of the dr5 config and checks the result against the `friction_mu`
    each rollout actually recorded, then counts the distinct values.
    """
    dr, run = C.load_dr_config(REPO / "config" / "collect_dr5.yaml")
    mus = []
    for t in run["terrains"]:
        for s in run["seeds"]:
            mus.append(dr.friction_value(t, s, dr.rng(s)))
    assert len(mus) == 12
    assert len({round(m, 6) for m in mus}) <= 6, mus
    assert min(mus) > 0.44, f"dr5 was expected never to reach the slip band: min mu {min(mus)}"

    # ... and it is the mu that was really used.
    with np.load(REPO / "data" / "dr5" / "flat_seed000.npz", allow_pickle=True) as z:
        recorded = json.loads(str(z["meta"]))["friction_mu"]
    assert dr.friction_value("flat", 0, dr.rng(0)) == pytest.approx(recorded, rel=1e-12)


def test_the_grid_stratifies_mu_and_reaches_the_slip_band():
    dr, run = C.load_dr_config(REPO / "config" / "collect_dr6.yaml")
    got = {(t, s): dr.friction_value(t, s, dr.rng(s))
           for t in run["terrains"] for s in run["seeds"]}
    assert got[("flat", 0)] == pytest.approx(0.20)
    assert got[("hard_stepping", 2)] == pytest.approx(0.80)
    # `hard_stepping` FELL at 0.20 when it was measured, so the grid must floor it.
    assert min(v for (t, _), v in got.items() if t == "hard_stepping") >= 0.30
    assert len({round(v, 6) for v in got.values()}) >= 8
    assert sum(v <= 0.35 for v in got.values()) >= 5, "not enough of the set is in the slip band"


def test_a_grid_does_not_disturb_the_push_or_command_stream():
    """Turning stratification on must change friction and nothing else.

    `friction_value` consumes the uniform draw either way, so the generator that
    `_push_schedule` and `_command_schedule` then read from is at the same position.
    """
    dr5, _ = C.load_dr_config(REPO / "config" / "collect_dr5.yaml")
    dr6, _ = C.load_dr_config(REPO / "config" / "collect_dr6.yaml")
    for seed in (0, 1, 2):
        r5, r6 = dr5.rng(seed), dr6.rng(seed)
        dr5.friction_value("flat", seed, r5)
        dr6.friction_value("flat", seed, r6)
        np.testing.assert_array_equal(r5.random(64), r6.random(64))


def test_a_grid_is_not_truncated_to_a_pair():
    """A three-value grid parsed through the range coercion would silently become two."""
    dr = C.DomainRandomization.from_dict(
        {"friction": {"enabled": True, "grid_by_terrain": {"flat": [0.2, 0.3, 0.45]}}})
    assert dr.friction_grid_for("flat") == (0.2, 0.3, 0.45)


def test_an_unknown_terrain_in_a_grid_is_rejected():
    with pytest.raises(KeyError):
        C.DomainRandomization.from_dict(
            {"friction": {"enabled": True, "grid_by_terrain": {"lava": [0.2]}}})
