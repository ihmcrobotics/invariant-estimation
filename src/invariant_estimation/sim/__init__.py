"""`sim/` — running the fused estimator inside a MuJoCo simulation (G10).

Two pieces, deliberately separated:

* `sensors` — the *plant → estimator* boundary. Adds MuJoCo `gyro`/`accelerometer`
  sensors on the estimator's IMU sites, reads them (plus encoders and foot
  contact) out of an `MjData`, and packs a `FusedSensors`. Plain NumPy, no JAX.
* `estimator_loop` — the closed loop. Owns the jitted `fused_step`, drives it at
  the physics rate, and feeds the policy's `base_ang_vel` / `projected_gravity`
  from the ESTIMATE instead of ground truth, which is what the real robot does.

`run_estimator.py` at the repo root is the CLI over both.
"""

from .sensors import (
    ContactTrust,
    IMUNoise,
    SimSensorReader,
    add_imu_sensors,
)

__all__ = ["ContactTrust", "IMUNoise", "SimSensorReader", "add_imu_sensors"]
