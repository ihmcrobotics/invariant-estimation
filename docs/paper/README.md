# Redundancy-observed measurement noise — draft paper

`redundancy_observed_noise.html` is the source; the PDF is rendered from it with

    weasyprint redundancy_observed_noise.html redundancy_observed_noise.pdf

WeasyPrint is the only PDF toolchain on the development machine (no LaTeX), so equations are
styled HTML with Unicode glyphs rather than MathML, which WeasyPrint renders poorly. The
`<meta charset="utf-8">` in the head is load-bearing: without it WeasyPrint decodes the file as
Latin-1 and every math glyph becomes mojibake.

Every number in the paper comes from `scripts/offaxis_adaptive_r.py` and the consistency harness in
`invariant_estimation/eval/consistency.py`, run on the recovered 2026-07-17 Alex001 overground-walking log.
Reproduce the held-out table with, for each window,

    python scripts/offaxis_adaptive_r.py eval 110.0 113.6 <start> <end> 0.366 1.797
