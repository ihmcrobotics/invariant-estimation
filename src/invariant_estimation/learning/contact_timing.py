"""Contact-timing compensation: mocap-measured detection lag -> an advanced trust channel.

The gap between the estimator and mocap ground truth decomposes into an unobservable part
(global position and yaw drift -- a random walk no module can learn) and a systematic,
onboard-observable part. Contact-detection lag is the cheapest member of the second kind:
`TouchdownLagEstimator` (in `alex`) cross-correlates the mocap ghost's marker touchdown signal
against the robot's own force-based detector over a whole log and reports, per foot, how late
the robot's detection fires. This module is the consumer of that number.

Sign convention (shared with the Java writer, stated in its javadoc): **positive `lag_seconds`
means the marker signal leads the force signal** -- the robot detects contact that much *late*.
Compensation therefore shifts the trust channel *earlier* by the measured lag. That shift reads
future samples, so it is **valid offline only**: for replays, scoring, and training. On the robot
the same number informs detector threshold tuning; it cannot be applied as a shift.

Why this lives on the `LogWindow`, before `prepare_session`, rather than inside the adapter: the
adapter treats its window as the recorded truth and validates it (binary trust, cadence, tick
contiguity). A correction is not recorded truth -- it is an explicit, documented transformation --
so it is applied as a pure function that returns a new window, and the adapter stays unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TouchdownLag:
    """One log's measured contact-detection lag, per contact name."""

    log: str
    dt: float
    lag_seconds: Mapping[str, float]
    confident: Mapping[str, bool]

    def advance_ticks(self, contact: str, dt: float, *, allow_unconfident: bool = False) -> int:
        """Ticks to advance `contact`'s trust channel, at the session's own ``dt``.

        Positive = shift earlier (the robot detected late). Refuses a flat-peak measurement
        unless explicitly overridden, because a flat correlation peak means the two signals do
        not agree well enough for the lag to mean anything -- applying it anyway would inject a
        confident-looking correction derived from noise.
        """
        if contact not in self.lag_seconds:
            raise KeyError(f"artifact carries no lag for contact {contact!r}; "
                           f"it has {sorted(self.lag_seconds)}")
        if not self.confident[contact] and not allow_unconfident:
            raise ValueError(
                f"lag for {contact!r} is marked not confident (flat correlation peak); "
                "refusing to apply it. Pass allow_unconfident=True only if you have looked at "
                "the signals and decided the number is real anyway."
            )
        if not (dt > 0.0) or not np.isfinite(dt):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        return int(round(self.lag_seconds[contact] / dt))


def load_touchdown_lag(path) -> TouchdownLag:
    """Parses the artifact `TouchdownLagEstimator --out` writes.

    The key names are a cross-language contract pinned on the Java side by
    ``TouchdownLagArtifactTest``; this loader is deliberately strict so a drifted artifact fails
    here, at load, rather than as a silently zero correction.
    """
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version: {data.get('schema_version')!r}")
    if data.get("kind") != "touchdown_lag":
        raise ValueError(f"not a touchdown-lag artifact: kind={data.get('kind')!r}")

    lag: dict[str, float] = {}
    confident: dict[str, bool] = {}
    for pair in data["pairs"]:
        contact = pair["contact"]
        if contact in lag:
            raise ValueError(f"duplicate contact in artifact: {contact!r}")
        value = float(pair["lag_seconds"])
        if not np.isfinite(value):
            raise ValueError(f"non-finite lag for {contact!r}")
        lag[contact] = value
        confident[contact] = bool(pair["confident"])

    if not lag:
        raise ValueError("artifact has no pairs")

    return TouchdownLag(log=str(data.get("log", "")), dt=float(data["dt"]),
                        lag_seconds=lag, confident=confident)


def advance_trust(window, channel_map, advance_ticks: Mapping[str, int]):
    """A new ``LogWindow`` whose trust channels are shifted by ``advance_ticks``.

    ``advance_ticks[contact] = n > 0`` shifts that contact's trust channel *earlier* by ``n``
    ticks (``trust'[t] = trust[t + n]``), holding the final recorded value over the last ``n``
    ticks; ``n < 0`` delays it, holding the first value at the head. The input window is not
    modified, untouched channels are shared rather than copied, and a contact absent from
    ``advance_ticks`` keeps its channel as recorded.

    The hold-at-the-edge choice is deliberate: the alternative -- wrapping or zero-filling --
    would fabricate a contact transition at the window boundary, and a fabricated transition is
    precisely the kind of plausible-looking artifact this pipeline keeps having to hunt down.
    """
    channels = dict(window.channels)

    for contact, ticks in advance_ticks.items():
        if contact not in channel_map.anchor_trust:
            raise KeyError(f"channel map has no trust channel for contact {contact!r}; "
                           f"it has {sorted(channel_map.anchor_trust)}")
        ticks = int(ticks)
        if ticks == 0:
            continue

        name = channel_map.anchor_trust[contact]
        old = channels[name]
        if abs(ticks) >= len(old):
            raise ValueError(f"advance of {ticks} ticks exceeds the window length {len(old)}")

        new = np.empty_like(old)
        if ticks > 0:  # detection was late -> move the signal earlier
            new[:-ticks] = old[ticks:]
            new[-ticks:] = old[-1]
        else:          # detection was early -> delay it
            new[-ticks:] = old[:ticks]
            new[:-ticks] = old[0]
        channels[name] = new

    return replace(window, channels=channels)
