# ContactNet Integration for InEKF (Training in JAX)

## Current Problems/Warnings

### `network.py`
Currently, we initialize at the analytical filter's initial state, which I'm not sure is correct.
