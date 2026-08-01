"""`sim/terrain.py` — IsaacLab's uneven terrain as a MuJoCo heightfield.

Alex's walking policies were trained on `TILED_TERRAIN_WITH_HEIGHT_CFG` (TERRAIN.md §1): four
sub-terrains — `flat`, `waves`, `stepping_stones`, `hard_stepping_stones` — over 8 x 8 m tiles at
0.1 m/px. All four are rasterised here into a single `hfield`, for the reason TERRAIN.md §4 gives:
`MeshRandomGridTerrain` is a flush axis-aligned grid of squares, which a heightfield represents
exactly, as ONE geom instead of ~324 collision candidates per foot.

Two things this module owns and nothing else should duplicate: the **rasterisers** (`flat`,
`waves`, `stepping_stones`) plus the `TERRAINS` registry, and `HeightfieldFloor`, the floor spec
`run_policy.build_sim_model(floor=...)` accepts. The model assembly — collision set, contact
parameters, position servos, solver options — stays in `run_policy` and is NOT copied here.

    uv run python experiments/terrain_stage1_walk.py     # walks the baseline policy over all four

Gotchas, all already paid for and written up in **TERRAIN.md §7** — the numbers, not the prose:
an hfield is finite (`EXTENT` = 64 m, up from the prototype's 16 m); it needs a nonzero base
thickness (`size`'s 4th component) or it is a shell; `hfield_data` is normalised to [0, 1] and
scaled by `EZ` at compile time, `EZ` being part of the MODEL while `hfield_data` is not (§2b —
that split is what makes per-env terrain batchable under `vmap`). Foot soles sit at
`ALEX_SOLE_OFFSET`, so `tests/model/test_sole_frame.py` is in this module's blast radius.
"""
import sys
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import numpy as np

# Geometry, fixed for the whole terrain family.
HSCALE = 0.1        # IsaacLab `horizontal_scale`, m/px
EXTENT = 64.0       # m of terrain, square. 16 m gave only ~21 s at the policy's 0.38 m/s, which
                    # is not enough once a warm-up prefix is dropped.
N = int(round(EXTENT / HSCALE))   # 640 -> hfield_data is 640*640 float32 = 1.6 MB
EZ = 0.15           # hfield elevation scale, m. The tallest terrain we will ever generate.
BASE_THICKNESS = 0.1  # m of solid below z = 0
TILE = 8.0          # IsaacLab sub-terrain tile size, m. The reference length for `waves`.

VSCALE = 0.005      # IsaacLab `vertical_scale`; its hfield backend snaps elevations to this


# Rasterisers: (N, N) float32 elevations in metres, in [0, EZ], in MuJoCo's hfield index order —
# axis 0 is the row index (world +y), axis 1 the column index (world +x). The robot walks +x, so
# terrain varying along the direction of travel varies along axis 1.

def flat(n=N):
    """`MeshPlaneTerrain` — the control. Zero relief."""
    return np.zeros((n, n), np.float32)


def waves(amplitude=0.10, num_waves=2.0, n=N, hscale=HSCALE, tile=TILE):
    """`HfWaveTerrain` — a sinusoidal ripple along +x.

    IsaacLab: amplitude 0.00-0.10 m, `num_waves = 2.0`. `num_waves` is a count PER TILE, so the
    wavelength is `tile / num_waves` = 4 m and does not depend on how much terrain we rasterise.

    (The Stage-1 prototype spread `num_waves` over its whole 16 m field, giving an 8 m wavelength;
    that formula at `EXTENT` = 64 m would stretch it to 32 m — a gentle ramp, not waves, and a
    silently easier terrain than the recorded result. Pass `tile=EXTENT` to reproduce it.)
    """
    x = np.arange(n) * hscale                                   # metres along +x
    z = amplitude * 0.5 * (1.0 + np.sin(2.0 * np.pi * num_waves * x / tile))
    return np.broadcast_to(z[None, :], (n, n)).astype(np.float32)


def stepping_stones(grid=0.45, hi=0.03, seed=0, platform=0.0, n=N, hscale=HSCALE):
    """`MeshRandomGridTerrain` — square stones of side `grid`, each at a random height in [0, hi].

    IsaacLab: `stepping_stones` is grid 0.45 m / hi 0.03 m / platform 0.0; the `hard_stepping_stones`
    variant is grid 0.75 m / hi 0.07 m / platform 0.5 (a flat pad at the centre, i.e. the spawn).

    `platform = 0` still clears a fixed 1.6 m pad at the centre — the robot has to have somewhere
    level to settle before it is asked to walk, or the first tick is a fall rather than a step.
    """
    blk = max(1, int(round(grid / hscale)))
    nb = int(np.ceil(n / blk))
    r = np.random.default_rng(seed)
    z = np.kron(r.uniform(0.0, hi, (nb, nb)), np.ones((blk, blk)))[:n, :n]
    c = n // 2
    p = int(platform / hscale) if platform > 0 else 8
    z[c - p:c + p, c - p:c + p] = 0.0
    return z.astype(np.float32)


# The four sub-terrains of TERRAIN.md §1 at IsaacLab's parameters, name -> zero-argument callable.
# The seeds are the Stage-1 ones, so `TERRAINS[name]()` reproduces the recorded run. All
# `partial`s, so a sweep can re-bind: `TERRAINS["waves"].func(amplitude=0.04)`.
TERRAINS = {
    "flat":            partial(flat),
    "waves":           partial(waves, amplitude=0.10, num_waves=2.0),
    "stepping_stones": partial(stepping_stones, grid=0.45, hi=0.03, seed=1, platform=0.0),
    "hard_stepping":   partial(stepping_stones, grid=0.75, hi=0.07, seed=2, platform=0.5),
}


class HeightfieldFloor:
    """A `run_policy.build_sim_model(floor=...)` spec replacing the plane with a heightfield.

    Same two-method protocol as `run_policy.PlaneFloor`: `geom_attrs` runs before compilation and
    declares the `<asset><hfield>`, `finalize` after it and writes the normalised elevation
    samples, which are model DATA, not structure.
    """

    def __init__(self, field, *, ez=EZ, extent=EXTENT, base_thickness=BASE_THICKNESS,
                 name="terrain"):
        field = np.ascontiguousarray(field, np.float32)
        if field.ndim != 2 or field.shape[0] != field.shape[1]:
            raise ValueError(f"heightfield must be square 2-D, got {field.shape}")
        lo, hi = float(field.min()), float(field.max())
        if lo < 0.0 or hi > ez:
            # Silently clipping here is how a terrain run ends up flatter than it reports.
            raise ValueError(f"elevations [{lo:.4f}, {hi:.4f}] m fall outside [0, ez={ez}]")
        self.field, self.ez, self.extent = field, float(ez), float(extent)
        self.base_thickness, self.name = float(base_thickness), name

    @property
    def relief(self):
        """Peak elevation, m — what `spawn_lift` and the run banner report."""
        return float(self.field.max())

    def normalized(self):
        """The field as MuJoCo stores it: flat, row-major, scaled into [0, 1] by `ez`."""
        return (self.field / self.ez).clip(0.0, 1.0).ravel()

    def geom_attrs(self, root, with_visuals):
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        n = self.field.shape[0]
        e = ET.SubElement(asset, "hfield")
        e.set("name", self.name)
        e.set("nrow", str(n))
        e.set("ncol", str(n))
        # size = (radius_x, radius_y, elevation_scale, base_thickness). Radii, not extent, and the
        # 4th component is what makes the field solid instead of an infinitely thin shell.
        e.set("size", f"{self.extent / 2} {self.extent / 2} {self.ez} {self.base_thickness}")
        return dict(type="hfield", hfield=self.name,
                    **({"material": "groundplane"} if with_visuals else {}))

    def finalize(self, model):
        model.hfield_data[:] = self.normalized()


def sample(field, x, y, extent=EXTENT):
    """Terrain elevation, m, at world `(x, y)` — nearest sample, no interpolation.

    MuJoCo lays an hfield out row-major over [-extent/2, extent/2]^2, column index along +x and
    row index along +y. Out-of-range queries CLAMP to the edge — the silent failure, not an error.
    """
    n = field.shape[0]
    hscale = extent / n
    c = np.clip(((np.asarray(x) + extent / 2) / hscale).astype(int), 0, n - 1)
    r = np.clip(((np.asarray(y) + extent / 2) / hscale).astype(int), 0, n - 1)
    return field[r, c]


def spawn_lift(field, clearance=0.02):
    """Metres to raise the spawn pose by so the robot starts clear of the terrain.

    `run_policy.foot_rest_height` places the feet ~1 mm above a plane at z = 0; over a heightfield
    that pose can be inside a stone. Lifting by the peak relief is crude but always safe, and the
    settle phase drops the robot onto whatever is actually under it.
    """
    return float(np.max(field)) + clearance


def _run_policy():
    """Import the root-level `run_policy` script (not a package module; pytest makes the same
    `pythonpath = ["."]` accommodation). Lazy — `run_policy` pulls in onnxruntime at import.
    """
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import run_policy
    return run_policy


def build_terrain_model(policy, field, *, with_visuals=False, with_imu_sensors=False, **floor_kw):
    """`run_policy.build_sim_model` with its floor plane swapped for `field`."""
    return _run_policy().build_sim_model(
        policy, with_visuals=with_visuals, with_imu_sensors=with_imu_sensors,
        floor=HeightfieldFloor(field, **floor_kw))
