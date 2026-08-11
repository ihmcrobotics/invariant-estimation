"""Reconstruct the `ContactNetConfig` a checkpoint was TRAINED under.

Invariant N5. `train.save_params` writes bare arrays (`np.savez` of the pytree
leaves), so the checkpoint file carries no metadata at all: the parameterisation of
`diag(L)` lives only in `summary.json`, written beside it by `run_contactnet.py`.
Nothing about the weights reveals it — the same numbers are a valid head under every
choice — so a loader that constructs a default `ContactNetConfig()` silently rescales
Sigma_C and no diagnostic fires.

That is not hypothetical: until 2026-08-10 every deployment path did exactly that
(`run_estimator.build_contactnet_provider`, `evaluate_run`, `drift_backfill`,
`online_offline_oracle`), which meant the `exp` checkpoint could only ever have been
evaluated as `softplus`. `scripts/plot_contact_phase.py` was the one caller that got
it right, and only because it made the user pass `--diag-param` by hand.

Deliberately narrow: this reads the fields that change what stored weights MEAN while
loading happily anyway. Shape-bearing fields (`F`, `H`, `widths`) need no rescue —
a mismatch fails loudly in `load_params`. `contacts_per_foot` has its own check in
`run_estimator._check_contact_geometry`, which predates this module and stays there
because it compares against the deployed geometry rather than reconstructing config.
"""
import dataclasses
import json
import os

from .config import ContactNetConfig

# Fields whose value is not recoverable from the weights and which silently change
# what those weights mean. One list, so extending the contract is one line here and
# a note in N5. See `network.DIAG_PARAMS` for why the bounds travel with the kind.
RESTORED_FIELDS = ("diag_param", "diag_lo", "diag_hi")


def summary_path(ckpt: str) -> str:
    """`summary.json` beside a checkpoint. Accepts the run directory or `params.npz`."""
    d = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    return os.path.join(d, "summary.json")


def restored_fields(ckpt: str) -> dict:
    """The `RESTORED_FIELDS` recorded for `ckpt`, or `{}` for what is not recorded.

    A missing `summary.json`, an unreadable one, or a missing key all mean "trained
    before this was recorded" and fall back to the `ContactNetConfig` defaults —
    which are the pre-option behaviour (`softplus`), so legacy checkpoints keep
    loading exactly as they always did. Same tolerance as
    `run_estimator._check_contact_geometry`.
    """
    path = summary_path(ckpt)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            cfg = json.load(fh)["cfg"]
    except (KeyError, ValueError, OSError):
        return {}
    return {k: cfg[k] for k in RESTORED_FIELDS if k in cfg}


def config_for_checkpoint(ckpt: str, *, base: ContactNetConfig | None = None,
                          verbose: bool = True, **overrides) -> ContactNetConfig:
    """The config to load `ckpt` under: recorded fields restored, `overrides` on top.

    `base` supplies every field this module does not restore (defaults if omitted).
    Explicit `overrides` win over the recording — a caller that knows better, or a
    deliberate mismatch experiment, stays possible and stays visible in the log.
    """
    fields = ({f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
              if base is not None else {})
    restored = restored_fields(ckpt)
    fields.update(restored)
    fields.update(overrides)
    cfg = ContactNetConfig(**fields)
    if verbose:
        missing = [k for k in RESTORED_FIELDS if k not in restored]
        note = f"  (not recorded, using defaults: {', '.join(missing)})" if missing else ""
        print(f"ContactNet: checkpoint trained with diag_param={cfg.diag_param} "
              f"lo={cfg.diag_lo:g} hi={cfg.diag_hi:g}{note}")
    return cfg
