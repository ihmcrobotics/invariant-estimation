"""Sphinx docs entry points: ``uv run docs`` (live server) / ``docs-build`` (HTML)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs"
OUTPUT = ROOT / "docs" / "_build" / "html"


def build() -> None:
    """One-off HTML build into ``docs/_build/html``."""
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "sphinx", "-b", "html", str(SOURCE), str(OUTPUT)]
        )
    )


def serve() -> None:
    """Live-reloading preview server; opens a browser at http://127.0.0.1:8000."""
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "sphinx_autobuild",
                str(SOURCE),
                str(OUTPUT),
                "--open-browser",
                "--ignore",
                "*/_build/*",
            ]
        )
    )
