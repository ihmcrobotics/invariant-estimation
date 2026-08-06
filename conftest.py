"""Repo-root pytest configuration.

**Pin the whole suite to the CPU JAX backend.**

This exists because of `pyproject.toml`'s `gpu` extra. The moment someone runs
`uv sync --extra gpu`, a `jax-cuda13-plugin` appears and `jax.devices()` starts returning CUDA by
default -- which silently moves the MJX kinematics inside `make_fused_step` onto the GPU. Nothing
about the port is wrong on GPU, but float accumulation order changes, and the tolerances in
`tests/sim`, `tests/pipeline` and `tests/replay` were all measured against CPU accumulation. The
suite would start failing on a machine that merely installed an optional dependency, which is the
worst kind of flake: it looks like a regression in the estimator.

`setdefault`, not assignment, so a deliberate parity check still works:

    JAX_PLATFORMS=cuda uv run pytest tests/sim -q

This must run before `jax` is first imported. pytest imports `conftest.py` before collecting, and
`pyproject.toml` already sets `pythonpath = ["."]`, so this file is picked up from the repo root.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
