"""Feasibility probe for per-environment heightfield terrain under vmap in MJX.

This is the experiment `TERRAIN.md` §2 rests on. Run it before trusting anything else in that
document, and re-run it after any mujoco/mjx upgrade — the whole plan depends on two facts:

  1. MJX implements `HFIELD x BOX` collisions (Alex's feet are boxes);
  2. `hfield_data` is a batchable `mjx.Model` field, so per-env terrain is DATA and never enters
     the traced graph.

Neither is a documented stability guarantee, so this probe is the contract.

    uv run python experiments/mjx_terrain_probe.py

Expected: 8 envs settle at visibly different heights over 8 different terrains, all finite, and
`traced graphs == 1` both after the first call and after swapping in completely new terrain.
"""
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

NROW = NCOL = 32
N_ENV = 8

XML = f"""
<mujoco>
  <option timestep="0.005" integrator="implicitfast" solver="Newton" iterations="10"/>
  <asset>
    <hfield name="terrain" nrow="{NROW}" ncol="{NCOL}" size="4 4 0.20 0.05"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="foot" pos="0 0 0.6">
      <freejoint/>
      <geom name="footbox" type="box" size="0.13 0.07 0.0275" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""


def collision_pairs():
    """Which geom-type pairs this MJX build actually implements."""
    from mujoco.mjx._src import collision_driver

    table = getattr(collision_driver, "_COLLISION_FUNC", None)
    if table is None:
        return None
    names = {int(getattr(mujoco.mjtGeom, n)): n.replace("mjGEOM_", "")
             for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")}
    return sorted({(names.get(int(a), a), names.get(int(b), b)) for a, b in table})


def terrain(seed, amplitude):
    """A bumpy field normalised to [0, amplitude] — the layout MuJoCo's `hfield_data` wants."""
    rng = np.random.default_rng(seed)
    z = rng.random((NROW, NCOL))
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return (z * amplitude).astype(np.float32).ravel()


def rollout(model, data, steps=200):
    def body(d, _):
        d = mjx.step(model, d)
        return d, (d.qpos[2], d._impl.ncon)

    return jax.lax.scan(body, data, None, length=steps)


def main():
    pairs = collision_pairs()
    if pairs is not None:
        hfield = [p for p in pairs if "HFIELD" in p[0] or "HFIELD" in p[1]]
        print(f"MJX implements {len(pairs)} collision pairs; HFIELD ones: {hfield}")
        assert ("HFIELD", "BOX") in pairs, "this MJX build cannot collide a box against terrain"

    model = mujoco.MjModel.from_xml_string(XML)
    print(f"hfield_data {model.hfield_data.shape}  nrow={int(model.hfield_nrow[0])} "
          f"ncol={int(model.hfield_ncol[0])}  size={model.hfield_size[0]}")

    amplitudes = np.linspace(0.0, 1.0, N_ENV)
    fields = np.stack([terrain(i, a) for i, a in enumerate(amplitudes)])

    mx = mjx.put_model(model)
    # Batch ONLY hfield_data; every other field is shared. This is the MuJoCo-Playground
    # `domain_randomize` idiom: an in_axes tree that is None everywhere except what varies.
    in_axes = jax.tree.map(lambda _: None, mx).tree_replace({"hfield_data": 0})
    step = jax.jit(jax.vmap(rollout, in_axes=(in_axes, 0)))   # jit OUTSIDE the vmap

    data = mjx.make_data(model)
    batch = jax.vmap(lambda _: data)(jnp.arange(N_ENV))

    _, (zs, ncons) = step(mx.tree_replace({"hfield_data": jnp.asarray(fields)}), batch)
    zs, ncons = np.asarray(zs), np.asarray(ncons)

    print(f"\nvmapped {N_ENV} envs x 200 steps over DIFFERENT terrain:")
    print(f"  {'env':>4}{'amp':>7}{'z_start':>10}{'z_end':>10}{'ncon':>7}")
    for i, a in enumerate(amplitudes):
        print(f"  {i:>4}{a:7.2f}{zs[i, 0]:10.4f}{zs[i, -1]:10.4f}{int(ncons[i, -1]):>7}")
    print(f"\n  resting heights differ across envs: {zs[:, -1].std():.4f} m std "
          f"(min {zs[:, -1].min():.4f}, max {zs[:, -1].max():.4f})")
    print(f"  all finite: {np.all(np.isfinite(zs))}")

    print(f"\n  traced graphs after the first call: {getattr(step, "_cache_size")()}")
    # A recompile here would mean terrain is baked into the graph, which would kill the approach.
    fields2 = np.stack([terrain(100 + i, a) for i, a in enumerate(np.linspace(0.5, 2.0, N_ENV))])
    step(mx.tree_replace({"hfield_data": jnp.asarray(fields2)}), batch)
    n = getattr(step, "_cache_size")()
    print(f"  traced graphs after NEW terrain:      {n}   "
          f"{'-> terrain is DATA, not graph structure' if n == 1 else '-> RECOMPILED, bad'}")
    assert n == 1, "swapping terrain recompiled; per-env terrain would not scale"


if __name__ == "__main__":
    main()
