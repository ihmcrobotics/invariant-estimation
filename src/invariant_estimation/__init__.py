"""invariant_estimation — invariant estimation pipeline in JAX.

The estimator runs in **float64**: covariance recursions (Joseph-form updates,
`F P Fᵀ + Q_d`) accumulate round-off that is harmful in float32 and compounds over
long `lax.scan` rollouts.  We enable double precision here, at package import,
before any `jax.numpy` array is built.

NOTE: `jax_enable_x64` is a *process-global* JAX setting.  Importing this package
flips it on for the whole process — including anything else that imports it, e.g.
a float32 ContactNet BPTT trainer.  This is a deliberate design decision; if a
consumer needs float32, it must manage precision explicitly around this import.
"""
import jax

jax.config.update("jax_enable_x64", True)
