"""Temperature — override sampling temperature for a generation."""

from __future__ import annotations

from dataclasses import dataclass

from ...steering.controls import BackendControl

CONTROL_INFO = {
    "kind": "backend",
    "domain": "decoding",
    "summary": "Override sampling temperature for this generation.",
    "composable": False,
}


@dataclass(frozen=True)
class Temperature(BackendControl):
    """Override the sampling temperature for this generation.

    Overrides any temperature set in model_options for the duration of this generation request only.

    Fields:
        temperature: The temperature value. Must be >= 0.
    """

    temperature: float

    def __post_init__(self):
        """Validate temperature is non-negative."""
        if self.temperature < 0:
            raise ValueError(f"Temperature must be >= 0, got {self.temperature}")
