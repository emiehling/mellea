"""Utility for resolving transformer layer lists across model architectures."""

from __future__ import annotations

from typing import Any


def get_model_layers(model: Any) -> Any:
    """Resolve the list of transformer layers from a HuggingFace model.

    Supports common architectures: ``model.model.layers`` (Llama, Granite, Mistral),
    ``model.transformer.h`` (GPT-2, GPT-J), ``model.model.decoder.layers`` (OPT, BART).

    Args:
        model: A HuggingFace ``PreTrainedModel`` instance.

    Returns:
        The model's ``nn.ModuleList`` of transformer layers.

    Raises:
        AttributeError: If the model architecture is not recognized.
    """
    # Llama, Granite, Mistral, Qwen, Phi, Gemma, etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2, GPT-J, GPT-NeoX
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # OPT, BART
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    raise AttributeError(
        f"Cannot resolve transformer layers for model architecture: "
        f"{type(model).__name__}. Add support in get_model_layers()."
    )
