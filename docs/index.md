# Invariant Estimation

Theory and design notes for invariant state estimation of a humanoid robot,
implemented in JAX.

```{toctree}
:maxdepth: 2
:caption: Contents

theory/index
```

## Math & macros

Pages are written in Markdown (MyST) with LaTeX-style math. Inline math uses
`$ ... $` and display math uses `$$ ... $$`; AMS environments such as
`align` work too. All macros from `preamble.sty` are available, e.g.:

$$
\dfv{I}{p} = \angvel{I}{B} \times \fv{I}{p}, \qquad
\rotmat{I}{B} \in SO(3), \qquad
\mag{\vbf{v}}.
$$

See {doc}`theory/notation` for the full macro reference.
