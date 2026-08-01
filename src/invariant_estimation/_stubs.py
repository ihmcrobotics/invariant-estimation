"""Regenerate LSP type stubs into ``typings/`` (``uv run gen-stubs``).

mujoco's ``MjModel``/``MjData`` live in a compiled ``_structs`` extension pyright
cannot introspect; mjx is pure Python but ships no ``py.typed``, so its
re-exports (``put_data``/``put_model``) don't resolve.  ``typings/`` is gitignored
and on pyright's ``stubPath``.  mypy is installed on demand and removed
afterwards so it never lingers in the environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
TYPINGS = ROOT / "typings"

# Compiled extension modules — stubgen must import & reflect on these.
COMPILED = [
    "mujoco._structs", "mujoco._functions", "mujoco._enums", "mujoco._constants",
    "mujoco._callbacks", "mujoco._errors", "mujoco._specs",
]
# Pure-Python packages missing py.typed — AST-parsed, no import.
PACKAGES = ["mujoco.mjx"]


def _uv(*args: str) -> None:
    env = {**os.environ, "VIRTUAL_ENV": str(VENV)}
    subprocess.check_call(["uv", "pip", *args], env=env)


def main() -> None:
    """Generate pyright/nvim stubs into ``typings/`` (safe to re-run)."""
    stubgen = Path(sys.executable).with_name("stubgen")
    _uv("install", "-q", "mypy")  # dev-only; removed in the finally below
    try:
        spec: list[str] = []
        for m in COMPILED:
            spec += ["-m", m]
        for p in PACKAGES:
            spec += ["-p", p]
        subprocess.check_call([str(stubgen), "-o", str(TYPINGS), *spec])
        for pattern in ("mujoco/*_test.pyi", "mujoco/mjx/**/*_test.pyi"):
            for f in TYPINGS.glob(pattern):
                f.unlink()
    finally:
        _uv("uninstall", "-q", "mypy", "ast-serialize", "librt")
    print("Stubs written under typings/. Run :LspRestart in nvim.")
