# Robot-log learning input contract v1

Implemented 2026-09-16 in `learning/log_adapter.py` and
`learning/session_model.py`. This boundary works with synthetic `LogWindow`
objects without a robot, mocap, or binary decoder. `read_session` uses the same
boundary after `replay/logsource.read_window` decodes an IHMC log.

## Required data

Names are explicit in `ChannelMap`, not inferred from dictionary iteration or
substring searches. Persist `dataclasses.asdict(channel_map)` with the capture's
configuration/provenance. `schema_version=1`; sensor values are SI float64.

| Input | Order / units / convention | Source |
|---|---|---|
| Joint positions | Every model hinge, rad; selected by name | Processed sensor output consumed by the estimator, before JointKF |
| Joint velocities | Every model hinge, rad/s | Same sensor-processing configuration; ankle columns gathered in build's unfiltered-anchor order |
| Gyros | Every `build.imu_names` entry, x/y/z, rad/s | Each IMU's measurement frame; do not subtract its learned JointKF bias here |
| Base accelerometer | x/y/z, m/s² specific force | Base IMU measurement frame; includes gravity (upright rest reads +g on z) |
| Anchor trust | One binary 0/1 channel per contact slot | Existing production decision, not a mocap label and not a smoothed probability |
| Contact probability | One [0,1] channel per slot | Logged production probability; only used by the optional covariance heuristic |
| Time and tick | Finite log-relative seconds; contiguous integer tick indices | `LogWindow`; stride must be 1 |
| Robot model | MJCF with actual IMU/sole sites, pelvis/body site, armature, explicit build | Reuse `MjxModel`; model and build joint/DoF/pair order must agree |
| IMU → body rotation | Proper 3×3 rotation | Calibration/model; checked against MJX rigid mount |
| Initial state | World-from-body R, world v/p, full 9+3N covariance | Explicit prior and source string; no implicit mocap initialization |
| Accelerometer bias | Body-frame 3-vector, or stationary prefix calibration | Fixed calibration or declared prefix with independent expected specific force |

`sensor_processing` records the processing configuration/revision. The adapter
does not emulate IHMC SensorProcessing and does not assume `raw_q_*` is the
estimator input. Resolve and inspect the correct published stage using the
existing log reader and Java configuration. Do not feed an already fused joint
estimate back as a raw sensor observation.

The first retained sensor row runs predict/update from the supplied base prior
at `t_first - dt`. Joint q and contact anchors are seeded from the first encoder
row; contact state is initialized in world coordinates as `p + R @ FK(q)`.
Joint velocity/bias use the existing default prior; previous-tick anchor trust
starts at zero. Include a warm-up interval and exclude its truth samples from
the loss if startup error is not part of the study. Every capture starts fresh.

## Time and frame alignment

`ClockMapping(source_origin_s, target_origin_ns, clock_domain)` defines:

    target_ns = target_origin_ns + round((log_time - source_origin_s) * 1e9)

The final addition uses integers so epoch-sized nanoseconds retain low bits.
Clock mapping is explicit: setting the origin to zero without measuring it is
not synchronization. This version does not estimate clock drift, resample
irregular inputs, or infer the logger's native timestamp origin. Cadence must
match BOTH filters' dt; a missing sensor tick is an error. Split discontinuous
captures into independent windows rather than compressing or filling them.

Mocap CSV timestamps must already be in the declared target clock domain, and
their poses/twists must already be registered to the declared world frame.
`PreparedSession.with_truth` checks the declared frame/clock and uses the
existing mocap reader's nearest-time tolerance/valid mask. It does not estimate
the clock or spatial transform. Record measured mappings in the capture's
`synchronization` artifact. Pelvis body origin must match the model body site.

Every sensor tick advances the filter, even during mocap dropout. `session_loss`
selects valid truth AND corresponding estimates only AFTER the complete scan.
Zero valid truth samples fail. An invalid sensor value fails input validation;
it is never interpreted as a mocap dropout.

## Model and calibration decisions

`MjxSessionModel` maps `ModelEval` to relative rotations/Jacobians and anchor
F/U Jacobians, reusing `anchor_jacobians`. All model hinges have measured q.
Each scan tick replaces filtered joints with the entering JointKF state before
evaluating its model; contact FK/J use the updated JointKF state and the live
unfiltered joints. Contact J and J-dot are differentiated from the model's FK.

The mass path follows `MjxModel.qpos`'s documented Java considered-subsystem
convention: off-path subtrees stay at construction qpos0 for M, while filtered
joints and nuisance gaps remain live. The FK/anchor path still sees live ankle
positions. Armature is already in MJX M and is not added twice.

Accelerometer correction is `R_imu_to_body @ raw_specific_force - bias_body`.
Only rotation and bias correction are implemented, matching the existing
two-stage input convention; no IMU lever-arm compensation is introduced.
The stationary option requires a prefix [0,end), independent expected
specific force, base raw-gyro norm ≤0.15 rad/s, joint speeds ≤0.05 rad/s, and
per-axis acceleration std ≤0.2 m/s² (thresholds configurable). These checks
reject obvious motion but do not prove stationarity. Save the calibration
interval, thresholds and fitted bias with the session calibration artifact.
Do not use held-out motion/truth to tune these choices.

Default contact process covariance is the existing common isotropic
`ekf.sigma_c`, compatible with artifact v1's `constant_body_isotropic` contract.
Supplying BOTH `firm_variance` and `swing_variance` opts into the existing
probability heuristic. Such a session declares `probability_body_isotropic`:
artifact v1 refuses it. Do not relabel this schedule as constant to export it.
`contact_q` is scaled once by the existing two-stage step.

## Usage

```python
from invariant_estimation.learning.log_adapter import (
    ClockMapping, InitialState, prepare_session, read_session, session_loss,
)
from invariant_estimation.learning.session_model import MjxSessionModel
from invariant_estimation.learning.noise import NoiseSpec
from invariant_estimation.learning.optimize import fit_scalars
from invariant_estimation.learning.artifact import from_fit, save_artifact

model = MjxSessionModel(mjx_model, build, body_site="pelvis", contact_sites=("leftSole", "rightSole"))
# Configure exact channel names from the real handshake before calling this.
# `window` can be synthetic or returned by replay.logsource.read_window.
session = prepare_session(
    window, channel_map, measured_clock_mapping, model, build, joint_params, ekf,
    initial_state, imu_to_body=model.imu_to_body, world_frame="registered_world",
    accel_bias_body=calibrated_bias_body,
)
session = session.with_truth(
    "pelvis.csv", "pelvisVelocity.csv", clock_domain=session.clock_domain,
    world_frame=session.world_frame, max_offset_ns=matching_tolerance_ns,
)
```

Load a verified `CaptureManifest` and build one independent prepared session
per manifest ID. Select ONLY `manifest.partition("train")` for the loss; use
validation for model selection and test sessions only for final evaluation:

```python
import jax.numpy as jnp
spec = NoiseSpec(tuple(build.imu_names), arm=7)
train = [prepared_by_id[s.session_id] for s in manifest.partition("train")]
def loss(theta):
    return jnp.mean(jnp.stack([
        session_loss(theta, spec, build, joint_params, ekf, model, s) for s in train
    ]))
fit = fit_scalars(loss, spec.initial_theta(), steps=100)
# Build context/provenance/baseline from the actual model/config/manifest.
# context["contact_process_model"] must equal each session's declared model.
artifact = from_fit(fit.theta, spec, context=context, provenance=provenance, baseline=baseline)
save_artifact("learned_noise.json", artifact)
```

The adapters do not install `ihmclog`, guess Alex's current channel names,
or generate capture metadata on the user's behalf. Full Java policy parity
(post-encoder linearization, contact/reseed/gravity ordering) remains separate.
Passing synthetic tests is not a claim of learned-matrix transfer performance.

## Software-only verification

    .venv/bin/python -m pytest tests/learning/test_log_adapter.py tests/learning/test_session_model.py -q

Tests cover reordered mappings, precise clock origins, missing/NaN channels,
gaps, trust/probability validation, rotated bias calibration, contact schedule,
CSV alignment/dropout, learning and held-out scoring, artifact round-trip,
fresh-state replay, MJX FK finite differences, unfiltered ankle motion, mass
convention, and a differentiable two-stage scan with the real MJX adapter.
IHMC binary-decoder and hardware-channel verification require a real log.
