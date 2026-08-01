"""`sim/` — running the fused estimator inside a MuJoCo simulation (G10).

* `sensors` — the *plant → estimator* boundary. MuJoCo `gyro`/`accelerometer` sensors on
  the estimator's IMU sites, read (plus encoders and foot contact) out of an `MjData` into
  a `FusedSensors`. Plain NumPy, no JAX.
* `estimator_loop` — the closed loop. Owns the jitted `fused_step`, drives it at the physics
  rate, and feeds the policy's `base_ang_vel` / `projected_gravity` from the ESTIMATE instead
  of ground truth, which is what the real robot does.
* `estimator_thread` — the same loop off the render thread. Opt-in, viewer-only, off by
  default.
* `terrain` — the *ground*. Rasterises IsaacLab's four sub-terrains into a MuJoCo heightfield
  and hands `run_policy.build_sim_model` a floor spec, so uneven ground is a parameter of the
  one model assembly rather than a second copy of it.
* `ghost` — the estimate drawn as a translucent second robot, for seeing *where* it goes wrong.
* `collect` — terrain-randomised rollouts recorded as ContactNet training data.

Only `sensors` and `terrain` are re-exported below; the rest are imported by module.
`run_estimator.py` at the repo root is the CLI.
"""

from .sensors import (
    ContactTrust,
    IMUNoise,
    SimSensorReader,
    add_imu_sensors,
)
from .terrain import (
    TERRAINS,
    HeightfieldFloor,
    build_terrain_model,
    flat,
    sample,
    spawn_lift,
    stepping_stones,
    waves,
)

__all__ = [
    "ContactTrust", "IMUNoise", "SimSensorReader", "add_imu_sensors",
    "TERRAINS", "HeightfieldFloor", "build_terrain_model", "flat",
    "sample", "spawn_lift", "stepping_stones", "waves",
]
