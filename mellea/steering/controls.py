"""Factory functions for creating common steering controls."""

from __future__ import annotations

from typing import Any

from ..core.steering import Control, ControlCategory


def input_control(
    name: str, *, params: dict[str, Any] | None = None, artifact_ref: str | None = None
) -> Control:
    """Create an input-stage control (prompt rewriting, system prompt injection).

    Args:
        name: Identifier for this control.
        params: Control-specific configuration.
        artifact_ref: Optional reference to a steering artifact.

    Returns:
        A ``Control`` with category ``INPUT``.
    """
    return Control(
        category=ControlCategory.INPUT,
        name=name,
        params=params or {},
        artifact_ref=artifact_ref,
    )


def structural_control(
    name: str,
    *,
    adapter_ref: str,
    params: dict[str, Any] | None = None,
    model_family: str | None = None,
) -> Control:
    """Create a structural control (LoRA adapters, adapter blending).

    Args:
        name: Identifier for this control.
        adapter_ref: Reference to the adapter artifact in the registry.
        params: Control-specific configuration.
        model_family: Model family this artifact was trained for.

    Returns:
        A ``Control`` with category ``STRUCTURAL``.
    """
    return Control(
        category=ControlCategory.STRUCTURAL,
        name=name,
        params=params or {},
        artifact_ref=adapter_ref,
        model_family=model_family,
    )


def state_control(
    name: str,
    *,
    artifact_ref: str,
    layer: int | None = None,
    params: dict[str, Any] | None = None,
    model_family: str | None = None,
) -> Control:
    """Create a state control (activation steering, forward hooks).

    Args:
        name: Identifier for this control.
        artifact_ref: Reference to the steering vector or similar artifact.
        layer: Optional single layer index to target. To steer multiple layers
            with different multipliers, use multiple ``state_control``s.
        params: Control-specific configuration.
        model_family: Model family this artifact was trained for.

    Returns:
        A ``Control`` with category ``STATE``.
    """
    p: dict[str, Any] = dict(params) if params else {}
    if layer is not None:
        p["layer"] = layer
    return Control(
        category=ControlCategory.STATE,
        name=name,
        params=p,
        artifact_ref=artifact_ref,
        model_family=model_family,
    )


def static_output_control(name: str, **gen_kwargs: Any) -> Control:
    """Create a static output control (temperature, top_p, logit_bias, etc.).

    Args:
        name: Identifier for this control.
        **gen_kwargs: Generation keyword arguments to merge into model options.

    Returns:
        A ``Control`` with category ``OUTPUT``.
    """
    return Control(category=ControlCategory.OUTPUT, name=name, params=gen_kwargs)


def active_output_control(
    name: str,
    *,
    artifact_ref: str,
    params: dict[str, Any] | None = None,
    model_family: str | None = None,
) -> Control:
    """Create an active output control (reward models, custom logits processors).

    Args:
        name: Identifier for this control.
        artifact_ref: Reference to the reward model or logits processor artifact.
        params: Control-specific configuration.
        model_family: Model family this artifact was trained for.

    Returns:
        A ``Control`` with category ``OUTPUT``.
    """
    return Control(
        category=ControlCategory.OUTPUT,
        name=name,
        params=params or {},
        artifact_ref=artifact_ref,
        model_family=model_family,
    )
