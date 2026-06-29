# Notation & Macros

These macros come from `preamble.sty` and are available in every page. Write
the command inside `$...$` (inline) or `$$...$$` (display) exactly as you would
in LaTeX.

## Decorators

| Macro | Source | Renders |
|-------|--------|---------|
| `\vbs{a}`  | `$\vbs{a}$`  | $\vbs{a}$  |
| `\vbf{a}`  | `$\vbf{a}$`  | $\vbf{a}$  |
| `\hbs{a}`  | `$\hbs{a}$`  | $\hbs{a}$  |
| `\hbf{a}`  | `$\hbf{a}$`  | $\hbf{a}$  |
| `\dbf{a}`  | `$\dbf{a}$`  | $\dbf{a}$  |
| `\tbf{a}`  | `$\tbf{a}$`  | $\tbf{a}$  |
| `\bs{a}`   | `$\bs{a}$`   | $\bs{a}$   |
| `\uv{a}`   | `$\uv{a}$`   | $\uv{a}$   |
| `\wbs{a}`  | `$\wbs{a}$`  | $\wbs{a}$  |
| `\cbs{A}`  | `$\cbs{A}$`  | $\cbs{A}$  |

## Frames & kinematics

| Macro | Source | Renders |
|-------|--------|---------|
| `\iv{a}`         | `$\iv{a}$`         | $\iv{a}$         |
| `\id`            | `$\id$`           | $\id$           |
| `\dcm{A}{B}`     | `$\dcm{A}{B}$`     | $\dcm{A}{B}$     |
| `\angvel{A}{B}`  | `$\angvel{A}{B}$`  | $\angvel{A}{B}$  |
| `\rotmat{A}{B}`  | `$\rotmat{A}{B}$`  | $\rotmat{A}{B}$  |
| `\fv{A}{p}`      | `$\fv{A}{p}$`      | $\fv{A}{p}$      |
| `\dfv{A}{p}`     | `$\dfv{A}{p}$`     | $\dfv{A}{p}$     |

## Operators

| Macro | Source | Renders |
|-------|--------|---------|
| `\pd`       | `$\pd$`        | $\pd$        |
| `\mag{x}`   | `$\mag{x}$`    | $\mag{x}$    |

## Example

A short display-math example combining several macros:

$$
\dfv{I}{p} = \angvel{I}{B} \times \fv{I}{p}
+ \rotmat{I}{B}\, \fv{B}{v}, \qquad
\dcm{I}{B} \in SO(3), \qquad
\frac{\pd f}{\pd x}.
$$

And an aligned block:

$$
\begin{aligned}
\dbf{x} &= \mathbf{f}(\mathbf{x}, \mathbf{u}) \\
\mathbf{y} &= \mathbf{h}(\mathbf{x}) + \tbf{n}, \qquad \mag{\tbf{n}} \le \epsilon.
\end{aligned}
$$
