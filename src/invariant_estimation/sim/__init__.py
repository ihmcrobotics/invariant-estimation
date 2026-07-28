"""`sim/` — running the fused estimator inside a MuJoCo simulation (G10).

Two pieces, deliberately separated:

* `sensors` — the *plant → estimator* boundary. Adds MuJoCo `gyro`/`accelerometer`
  sensors on the estimator's IMU sites, reads them (plus encoders and foot
  contact) out of an `MjData`, and packs a `FusedSensors`. Plain NumPy, no JAX.
* `estimator_loop` — the closed loop. Owns the jitted `fused_step`, drives it at
  the physics rate, and feeds the policy's `base_ang_vel` / `projected_gravity`
  from the ESTIMATE instead of ground truth, which is what the real robot does.
* `terrain` — the *ground*. Rasterises IsaacLab's four sub-terrains into a MuJoCo
  heightfield and hands `run_policy.build_sim_model` a floor spec, so uneven
  ground is a parameter of the one model assembly rather than a second copy of it.

`run_estimator.py` at the repo root is the CLI over both.
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
