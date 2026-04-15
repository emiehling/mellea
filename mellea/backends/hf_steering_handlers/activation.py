"""Activation steering via forward hooks on transformer layers."""

from __future__ import annotations

from typing import Any

from ...core.steering import Control, StateControlHandler
from .model_layers import get_model_layers


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

        model_layers = get_model_layers(model)

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
