"""Loader for ``config/filter_cfg.yaml`` — where every tuning number lives.

Each `default_*_params` factory reads its defaults from here, so retuning is a
YAML edit rather than a hunt through module globals.  Structural constants
(tangent indices, block layouts, group sizes) deliberately stay in code: those
change the math, not the tuning.  Every factory also takes keyword overrides, so
a test or a sweep can pass a value without touching the file.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root: src/invariant_estimation/config.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "filter_cfg.yaml"

_OVERRIDE: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read and parse a filter config file (uncached); default ``config/filter_cfg.yaml``."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"filter config not found: {path}")
    with path.open("r") as handle:
        parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"filter config must be a mapping, got {type(parsed).__name__}")
    _check_numeric(parsed, str(path))
    return parsed


def _check_numeric(node: Any, origin: str, trail: str = "") -> None:
    """Reject string-valued scalars that were meant to be numbers.

    PyYAML implements YAML 1.1, in which an exponent requires an explicit sign:
    ``1.0e+9`` is a float but ``1.0e9`` silently parses as the *string*
    ``"1.0e9"``.  That failure surfaces thousands of lines away as a dtype error
    inside a jitted function, so it is caught here at load time instead.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            _check_numeric(value, origin, f"{trail}.{key}" if trail else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _check_numeric(value, origin, f"{trail}[{i}]")
    elif isinstance(node, str):
        stripped = node.strip()
        try:
            float(stripped)
        except ValueError:
            return                      # a genuine string value — fine
        raise ValueError(
            f"{origin}: '{trail}' is the string {node!r}, not a number. "
            f"YAML 1.1 needs a signed exponent — write '{stripped.replace('e', 'e+')}' "
            f"if you meant a float."
        )


@lru_cache(maxsize=1)
def _cached_default() -> dict[str, Any]:
    return load_config()


def get_config() -> dict[str, Any]:
    """The active config tree (the packaged default unless `set_config` was called)."""
    return _OVERRIDE if _OVERRIDE is not None else _cached_default()


def set_config(config: dict[str, Any] | None) -> None:
    """Install a config tree process-wide; ``None`` restores the packaged default.

    Only affects factories called *after* it — parameters already built are
    plain pytrees and are not retroactively changed.
    """
    global _OVERRIDE
    _OVERRIDE = config


def section(name: str) -> dict[str, Any]:
    """One top-level section of the active config; `KeyError` on a typo, loudly at
    build time rather than a silent fall back to a hard-coded number."""
    config = get_config()
    if name not in config:
        raise KeyError(
            f"missing section '{name}' in filter config; "
            f"have {sorted(config)}"
        )
    return config[name]


def resolve(overrides: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge keyword overrides over config defaults, dropping ``None`` (= "take it
    from the config") — the pattern every `default_*_params` factory uses."""
    merged = dict(defaults)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged
