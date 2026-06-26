# Documentation

Sphinx docs for the invariant-estimation theory, written in MyST Markdown with
LaTeX-style math. The [Furo](https://pradyunsg.me/furo/) theme provides a dark
mode (toggle in the sidebar; follows your OS preference by default).

## Build

From the repo root, using the project's `uv` environment:

```bash
uv run docs-build      # one-off HTML build into docs/_build/html
uv run docs            # live-reload preview server (opens a browser)
```

`uv run docs` watches the sources and rebuilds on save, serving at
http://127.0.0.1:8000. Open `docs/_build/html/index.html` for the static build.

These are thin wrappers (see `src/invariant_estimation/_docs.py`). The
equivalent raw commands are `sphinx-build -b html docs docs/_build/html` and
`sphinx-autobuild docs docs/_build/html`, or `cd docs && uv run make html`.

## PDF (LaTeX)

```bash
cd docs && uv run make latexpdf   # requires a LaTeX toolchain
```

The PDF build pulls in the real `preamble.sty`, so macros render identically to
your LaTeX documents.

## Writing

- Add `.md` pages under `theory/` and list them in `theory/index.md`'s
  `toctree`.
- Math macros are defined once in `conf.py` (HTML/MathJax) and sourced from
  `preamble.sty` (PDF/LaTeX). To add a macro, update both: add an entry to
  `mathjax3_config["tex"]["macros"]` and to `preamble.sty`.
- See `theory/notation.md` for the full macro reference.
