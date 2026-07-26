"""TERRAIN.md Stage 1 — does the walking policy survive IsaacLab's terrain, in OUR sim?

Plain MuJoCo, no MJX: swap `run_policy`'s floor plane for a heightfield rasterised from the
IsaacLab sub-terrain parameters (TERRAIN.md §1) and walk the baseline policy across each one.

This is the cheapest gate in the plan and it PASSES — all four terrains, 0.38 m/s against a
commanded 0.4, upright for 20 s. Also doubles as the reference rasteriser for §4: `waves`,
`stepping_stones` and `hard_stepping_stones` here are the functions to lift into the MJX path.

    uv run python experiments/terrain_stage1_walk.py
"""
import sys, xml.etree.ElementTree as ET
import numpy as np, mujoco
sys.path.insert(0,'/home/llibshutz/alex/invariant-estimation')
import run_policy as rp
from invariant_estimation.pipeline import main_estimator as me

HSCALE = 0.1               # IsaacLab horizontal_scale, m/px
EXTENT = 16.0              # m of terrain (bigger than the 8 m tile so we can walk a while)
N = int(EXTENT / HSCALE)
EZ = 0.15                  # hfield elevation scale; every terrain is a fraction of this

def flat():
    return np.zeros((N, N), np.float32)

def waves(amplitude=0.10, num_waves=2.0):
    x = np.linspace(0, 1, N)
    return (amplitude * 0.5 * (1 + np.sin(2*np.pi*num_waves*x))[None, :] * np.ones((N,1))).astype(np.float32)

def stepping_stones(grid=0.45, hi=0.03, seed=0, platform=0.0):
    blk = max(1, int(round(grid / HSCALE)))
    nb = int(np.ceil(N / blk))
    r = np.random.default_rng(seed)
    z = np.kron(r.uniform(0.0, hi, (nb, nb)), np.ones((blk, blk)))[:N, :N]
    if platform > 0:                       # flat spawn pad at the centre
        p = int(platform / HSCALE); c = N // 2
        z[c-p:c+p, c-p:c+p] = 0.0
    else:
        c, p = N // 2, 8
        z[c-p:c+p, c-p:c+p] = 0.0          # always give it somewhere flat to start
    return z.astype(np.float32)

def build(field):
    urdf = rp.cycloid_forearm_urdf(rp.URDF)
    root = ET.fromstring(me.alex_spec_from_urdf(urdf).mjcf)
    opt = ET.SubElement(root, 'option')
    for k,v in dict(timestep=str(rp.DT), gravity='0 0 -9.81', integrator='implicitfast',
                    solver='Newton', iterations='25', noslip_iterations='5',
                    impratio='1', cone='pyramidal').items():
        opt.set(k, v)
    asset = ET.SubElement(root, 'asset')
    hf = ET.SubElement(asset, 'hfield')
    hf.set('name','terrain'); hf.set('nrow',str(N)); hf.set('ncol',str(N))
    hf.set('size',f'{EXTENT/2} {EXTENT/2} {EZ} 0.1')
    g = ET.SubElement(root.find('worldbody'), 'geom')
    g.set('name','floor'); g.set('type','hfield'); g.set('hfield','terrain')
    g.set('contype', rp.TERRAIN_GROUP['contype']); g.set('conaffinity', rp.TERRAIN_GROUP['conaffinity'])
    for k, v in rp.CONTACT.items():
        g.set(k, v)
    bodies={b.get('name'):b for b in root.iter('body')}
    for body,typ,size,pos,quat in rp.SCS2_COLLISION_GEOMS:
        e=ET.SubElement(bodies[body],'geom'); e.set('name',f'{body}_collision_0')
        e.set('type',typ); e.set('size',size); e.set('pos',pos); e.set('quat',quat)
        e.set('contype',rp.ROBOT_GROUP['contype']); e.set('conaffinity',rp.ROBOT_GROUP['conaffinity'])
        for k, v in rp.CONTACT.items():
            e.set(k, v)
    act=ET.SubElement(root,'actuator')
    for j in root.iter('joint'):
        n=j.get('name')
        if n in rp._FALLBACK:
            fb=rp._FALLBACK[n]
            j.set('damping', repr(POL['kd'].get(n,float(fb['kd']))))
            a=ET.SubElement(act,'position'); a.set('name',n); a.set('joint',n)
            a.set('kp',repr(POL['kp'].get(n,float(fb['kp']))))
            tau=POL['tau'].get(n,float(fb['maxEffort']))
            a.set('forcelimited','true'); a.set('forcerange',f'{-tau} {tau}')
    m = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    m.hfield_data[:] = (field / EZ).clip(0,1).ravel()          # hfield_data is normalised
    return m

POL = rp.load_policy('baseline')

def run(label, field, vx=0.4, secs=20.0):
    m = build(field); maps = rp.make_maps(m, POL)
    loop = rp.Loop(m, POL, maps)
    loop.d.qpos[2] += float(field.max()) + 0.02                # start clear of the terrain
    mujoco.mj_forward(m, loop.d)
    loop.set_height_target(loop.height_target)                 # re-seed the ramp from the new pose
    for _ in range(100):
        loop.control_tick()                                    # settle standing
    x0 = loop.d.qpos[0]; tilts=[]; zs=[]
    for k in range(int(secs/0.02)):
        loop.cmd[0:3]=(vx,0.0,0.0); loop.cmd[3]=0.0
        loop.control_tick(); tilts.append(loop.tilt_deg()); zs.append(loop.d.qpos[2])
    t=np.array(tilts); dx=loop.d.qpos[0]-x0
    fell=(t>45).any()
    print(f"  {label:34s} relief={field.max()*100:5.1f}cm  travelled={dx:+6.2f}m "
          f"({dx/secs:+.2f} m/s)  tilt_max={t.max():5.1f}  "
          + ("FELL" if fell else f"UPRIGHT {secs:.0f}s"))

print(f"hfield {N}x{N} px over {EXTENT} m at {HSCALE} m/px, elevation scale {EZ} m\n")
print("IsaacLab's four sub-terrains, walking baseline at vx=0.4 for 20 s:")
run("flat (control)",                flat())
run("waves  a=0.10 n=2",            waves(0.10, 2.0))
run("stepping_stones 0.45m/0.03m",   stepping_stones(0.45, 0.03, seed=1))
run("hard_stepping   0.75m/0.07m",   stepping_stones(0.75, 0.07, seed=2, platform=0.5))
