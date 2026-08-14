"""
Heightfield rasterization for offline terrain domain randomization
"""
import numpy as np

HSCALE = 0.1 # m/px (IsaacLab horizontal scale)
EXTENT = 16.0 # m; walk-length margin over the 8m IsaacLab tile
N = int(EXTENT/HSCALE)
EZ = 0.15 # hfield elevation scale; every field is a fraction of this

def flat(seed=0):
    return np.zeros((N,N),np.float32)

def waves(seed=0, amplitude=0.10, num_waves=2.0):
    """Sinusoidal corrugation, now genuinely *seeded*.

    The previous version ignored `seed` and returned a byte-identical field for
    every seed, so `waves/seed_train` and `waves/seed_heldout` shared the exact
    same heightfield (verified: max|field(sᵢ)−field(s₀)| = 0.0 across seeds) — a
    silent train/val terrain leak (FIX_CHECKLIST B1). Now the phase, in-plane
    travel direction and a ±20% amplitude jitter all derive from `seed`, so two
    seeds give decorrelated fields.

    The relief band is preserved: `num_waves` is fixed and the corrugation
    coordinate `s` is normalised to span [0, 1] regardless of direction, so the
    spatial frequency — and hence the ~2° slope the flat-trained policy tolerates
    — is unchanged up to the mild amplitude jitter.
    """
    r = np.random.default_rng((int(seed) << 8) ^ 0x5A17)
    phase = float(r.uniform(0.0, 2.0 * np.pi))
    theta = float(r.uniform(0.0, 2.0 * np.pi))            # in-plane travel direction
    amp = amplitude * float(r.uniform(0.8, 1.2))          # mild jitter, stays in band
    u = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(u, u)                              # (N, N)
    ct, st = np.cos(theta), np.sin(theta)
    s = (ct * X + st * Y) / (abs(ct) + abs(st))           # normalised to [0, 1]
    field = amp * 0.5 * (1.0 + np.sin(2.0 * np.pi * num_waves * s + phase))
    return field.astype(np.float32)

def stepping_stones(seed=0, grid=0.45, hi=0.03, platform=0.0):
    blk = max(1, int(round(grid / HSCALE)))
    nb = int(np.ceil(N / blk))
    r = np.random.default_rng(int(seed))
    z = np.kron(r.uniform(0.0, hi, (nb, nb)), np.ones((blk, blk)))[:N, :N]
    c = N // 2
    p = int(platform / HSCALE) if platform > 0 else 8
    z[c-p:c+p, c-p:c+p] = 0.0 # flat spawn pad in the center
    return z.astype(np.float32)


def hard_stepping(seed=0):
    return stepping_stones(seed=seed,grid=0.75, hi=0.07, platform=0.5)

_RASTER = {
    "flat":flat,
    "waves":waves,
    "stepping_stones":stepping_stones,
    "hard_stepping":hard_stepping
}

def sample_field(name, seed):
    if name not in _RASTER:
        raise KeyError(f"Unknown terrain rasterization: {name}, only {tuple(_RASTER.keys())} are supported")
    return _RASTER[name](seed=int(seed))

def choose_terrain(seed, mix):
    names = [n for n, _ in mix]
    w = np.array([x for _, x in mix], float)
    i = np.random.default_rng((int(seed) << 8) ^ 0x7E44).choice(len(names), p = w / w.sum())
    return names[int(i)]
