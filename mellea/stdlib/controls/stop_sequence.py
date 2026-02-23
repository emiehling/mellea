"""StopSequence — add stop sequences to halt generation."""

from __future__ import annotations

from dataclasses import dataclass

from ...steering.controls import BackendControl

CONTROL_INFO = {
    "kind": "backend",
    "domain": "decoding",
    "summary": "Add additional stop sequences to halt generation.",
    "composable": True,
}


@dataclass(frozen=True)
class StopSequence(BackendControl):
    """Add additional stop sequences to halt generation.

    These are added to any stop sequences already configured in model_options.

    Fields:
        sequences: Tuple of stop sequence strings.
    """

    sequences: tuple[str, ...]
