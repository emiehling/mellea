"""Concrete output controls for inference-time steering.

Output controls describe decoding-time interventions. They are pure data
descriptors that modify the generation process. The backend translates
them to backend-specific parameters (e.g., OpenAI's logit_bias, stop).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...steering.controls import OutputControl


@dataclass(frozen=True)
class LogitBiasControl(OutputControl):
    """Apply static token-level logit biases during decoding.

    Fields:
        biases: Tuple of (token_id, bias_value) pairs. Positive values
            increase the probability of the token; negative values decrease it.
    """

    biases: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class StopSequenceControl(OutputControl):
    """Add additional stop sequences to halt generation.

    These are added to any stop sequences already configured in model_options.

    Fields:
        sequences: Tuple of stop sequence strings.
    """

    sequences: tuple[str, ...]


@dataclass(frozen=True)
class TemperatureOverrideControl(OutputControl):
    """Override the sampling temperature for this generation.

    Overrides any temperature set in model_options for the duration
    of this generation request only.

    Fields:
        temperature: The temperature value. Must be >= 0.
    """

    temperature: float

    def __post_init__(self):
        """Validate temperature is non-negative."""
        if self.temperature < 0:
            raise ValueError(f"Temperature must be >= 0, got {self.temperature}")
