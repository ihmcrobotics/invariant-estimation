# `invariant-estimation`
Invariant Estimation Pipeline in JAX, Combined with Learned Module

> **Design decisions that will bite you while debugging** are collected in
> [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md). Read it before concluding the
> filter is broken — several of its behaviours are deliberate and surprising.
> Blow-by-blow reconciliation against the Java suite lives in
> [`PORT_NOTES.md`](PORT_NOTES.md); all tuning lives in
> [`config/filter_cfg.yaml`](config/filter_cfg.yaml).

## Installation

To install the repository from source, make sure you have `uv` and simply run: 

```
uv sync
```
