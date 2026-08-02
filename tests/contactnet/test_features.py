from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import invariant_estimation # noqa: F401
from invariant_estimation.contactnet.features import boxcar

def test_boxcar():
    # Test boxcar function with a simple input
    data =  jnp.ones((5,1)) * 5.0
    expected = jnp.ones((5,1)) * 5.0
    result = boxcar(data, s=3)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"

#TODO: add more tests for whole package, not just here.
