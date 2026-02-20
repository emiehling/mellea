"""Concrete state controls for inference-time steering.

State controls describe model-internal interventions. They are pure data
descriptors — the backend reads the fields and applies the intervention
using its own internals. State controls require backend support; backends
that don't support a control type will filter it out via SteeringCapabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...steering.controls import StateControl


@dataclass(frozen=True)
class ActivationSteeringControl(StateControl):
    """Apply an activation steering vector to specific model layers.

    The vector_id references a vector in the backend's vector registry.
    The registry maps IDs to actual tensor data and manages lifecycle.
    This keeps the control serializable and decoupled from tensor storage.

    Fields:
        vector_id: Identifier for a registered steering vector.
        layers: Tuple of layer indices where the vector should be applied.
        strength: Scaling factor for the vector. Default 1.0.
    """

    vector_id: str
    layers: tuple[int, ...]
    strength: float = 1.0


@dataclass(frozen=True)
class AttentionMaskControl(StateControl):
    """Modify the attention mask for specific token spans.

    Fields:
        mask_type: Type of mask modification (e.g., "block", "causal_prefix").
            Interpretation is backend-specific.
        target_span: Optional (start, end) token positions to mask.
            If None, the backend determines the span.
    """

    mask_type: str
    target_span: tuple[int, int] | None = None
