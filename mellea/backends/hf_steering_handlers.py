"""HuggingFace-specific steering control handlers.

These handlers operate on HuggingFace ``PreTrainedModel`` internals and are
registered by ``LocalHFBackend`` at initialization time. They are not usable
with API-based backends.
"""

from __future__ import annotations

from typing import Any

from ..core.steering import (
    Control,
    OutputControlHandler,
    StateControlHandler,
    StructuralControlHandler,
)


class ActivationSteeringHandler(StateControlHandler):
    """Applies a steering vector to hidden states via forward hooks.

    The artifact must be a tensor with shape compatible with the model's hidden
    states. Hooks are registered on the specified layers (or all layers if none
    are specified) and add the scaled steering vector to the hidden state output.

    Expects ``control.params`` to optionally contain:

    - ``layers`` (list[int], optional): Layer indices to hook. Defaults to all layers.
    - ``coefficient`` (float, optional): Scaling factor for the steering vector.
      Defaults to ``1.0``.
    - ``token_positions`` (list[int] | str, optional): Which token positions to steer.
      ``"all"`` (default) steers all positions. A list of ints steers only those positions.
    """

    def activate(self, control: Control, model: Any, artifact: Any | None) -> Any:
        """Register forward hooks that add the steering vector to hidden states.

        Args:
            control: The control descriptor.
            model: A HuggingFace ``PreTrainedModel`` instance.
            artifact: A tensor (the steering vector).

        Returns:
            A list of hook handles for cleanup.
        """
        import torch

        steering_vector = artifact
        assert steering_vector is not None, (
            "ActivationSteeringHandler requires a steering vector artifact"
        )

        layers_param = control.params.get("layers", None)
        coefficient = control.params.get("coefficient", 1.0)
        token_positions = control.params.get("token_positions", "all")

        model_layers = _get_model_layers(model)

        if layers_param is None:
            layer_indices = list(range(len(model_layers)))
        else:
            layer_indices = layers_param

        hooks: list[torch.utils.hooks.RemovableHook] = []

        for layer_idx in layer_indices:
            layer = model_layers[layer_idx]

            def _make_hook(sv: torch.Tensor, coeff: float, positions: list[int] | str):
                def hook(module, input, output):
                    hidden_states = output[0]
                    if positions == "all":
                        hidden_states = (
                            hidden_states + sv.to(hidden_states.device) * coeff
                        )
                    else:
                        for pos in positions:
                            hidden_states[:, pos, :] = (
                                hidden_states[:, pos, :]
                                + sv.to(hidden_states.device) * coeff
                            )
                    return (hidden_states, *output[1:])

                return hook

            h = layer.register_forward_hook(
                _make_hook(steering_vector, coefficient, token_positions)
            )
            hooks.append(h)

        return hooks

    def deactivate(self, handle: Any) -> None:
        """Remove all registered forward hooks.

        Args:
            handle: The list of hook handles returned by ``activate()``.
        """
        for h in handle:
            h.remove()


class AdapterControlHandler(StructuralControlHandler):
    """Loads and activates LoRA/PEFT adapters on a HuggingFace model.

    Integrates with Mellea's existing ``load_adapter`` / ``set_adapter``
    machinery on ``LocalHFBackend``. The artifact should be the adapter's
    qualified name or path.

    Expects ``control.params`` to optionally contain:

    - ``adapter_name`` (str, optional): The name to register the adapter under.
      Defaults to ``control.name``.
    """

    def activate(self, control: Control, model: Any, artifact: Any | None) -> Any:
        """Load and activate the adapter on the model.

        Args:
            control: The control descriptor.
            model: A HuggingFace ``PreTrainedModel`` with PEFT support.
            artifact: The adapter path or qualified name (str).

        Returns:
            The adapter name (str), used as the deactivation handle.
        """
        adapter_name = control.params.get("adapter_name", control.name)
        adapter_path = artifact

        if adapter_path is not None:
            model.load_adapter(adapter_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        return adapter_name

    def deactivate(self, handle: Any) -> None:
        """Deactivation is a no-op -- adapter teardown is handled by the lock scope.

        The adapter lock in ``_generate_with_adapter_lock`` manages adapter
        state transitions. We don't unload adapters eagerly because they may
        be reused on the next sampling loop iteration.

        Args:
            handle: The adapter name (unused).
        """


class StaticOutputControlHandler(OutputControlHandler):
    """Merges static generation parameters (temperature, top_p, etc.) into gen kwargs.

    All entries in ``control.params`` are merged directly into the generation
    kwargs dict. Common params: ``temperature``, ``top_p``, ``top_k``,
    ``repetition_penalty``, ``max_new_tokens``.
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Merge control params into generation kwargs.

        Args:
            control: The control descriptor whose params are generation kwargs.
            gen_kwargs: The current generation kwargs dict.
            artifact: Unused.

        Returns:
            The updated generation kwargs dict.
        """
        gen_kwargs.update(control.params)
        return gen_kwargs


class RewardGuidedDecodingHandler(OutputControlHandler):
    """Adds a reward-model-based logits processor to generation kwargs.

    The artifact must be a callable reward model that accepts input_ids and returns
    per-token reward scores. The handler wraps it in a ``LogitsProcessor`` and appends
    it to the ``logits_processor`` list in gen kwargs.

    Expects ``control.params`` to optionally contain:

    - ``temperature`` (float, optional): Scaling factor for reward scores.
      Defaults to ``1.0``.
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Add a reward-guided logits processor to gen kwargs.

        Args:
            control: The control descriptor.
            gen_kwargs: The current generation kwargs dict.
            artifact: A callable reward model.

        Returns:
            The updated generation kwargs dict with the logits processor added.
        """
        import torch

        reward_model = artifact
        assert reward_model is not None, (
            "RewardGuidedDecodingHandler requires a reward model artifact"
        )
        temperature = control.params.get("temperature", 1.0)

        class _RewardLogitsProcessor:
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor
            ) -> torch.FloatTensor:
                with torch.no_grad():
                    rewards = reward_model(input_ids)
                return scores + rewards * temperature

        processors = gen_kwargs.get("logits_processor", [])
        # Avoid mutating a shared default list.
        if not isinstance(processors, list):
            processors = list(processors)
        else:
            processors = list(processors)
        processors.append(_RewardLogitsProcessor())
        gen_kwargs["logits_processor"] = processors
        return gen_kwargs


def _get_model_layers(model: Any) -> Any:
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
        f"{type(model).__name__}. Add support in _get_model_layers()."
    )
