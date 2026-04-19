"""Activation steering via forward hooks on transformer layers."""

from __future__ import annotations

from typing import Any

from ...core.steering import Control, StateControlHandler
from .model_layers import get_model_layers


class ActivationSteeringHandler(StateControlHandler):
    """Applies per-layer steering vectors to hidden states via forward hooks.

    The artifact must be a ``dict[int, Tensor]`` mapping layer indices to their
    direction vectors; each layer gets its own vector.

    Expects ``control.params`` to optionally contain:

    - ``layer`` (int, optional): A single layer index to hook. Defaults to all
      layers present in the artifact when omitted. To steer multiple layers with
      different multipliers, use multiple ``state_control``s in the policy.
    - ``multiplier`` (float, optional): Scaling factor for the steering vector.
      Defaults to ``1.0``.
    - ``token_positions`` (list[int] | str, optional): Which token positions to steer.
      ``"all"`` (default) steers all positions. A list of ints steers only those positions.
    - ``transform`` (str, optional): How the steering vector is applied to hidden states.
      Currently only ``"additive"`` is supported (default).
    """

    def activate(self, control: Control, model: Any, artifact: Any | None) -> Any:
        """Register forward hooks that add per-layer steering vectors to hidden states.

        Args:
            control: The control descriptor.
            model: A HuggingFace ``PreTrainedModel`` instance.
            artifact: A ``dict[int, Tensor]`` mapping layer indices to their
                direction vectors.

        Returns:
            A list of hook handles for cleanup.
        """
        import torch

        assert artifact is not None and isinstance(artifact, dict), (
            "ActivationSteeringHandler requires a dict[int, Tensor] artifact "
            "mapping layer indices to direction vectors"
        )

        directions: dict[int, torch.Tensor] = artifact
        layer_param: int | None = control.params.get("layer", None)
        multiplier = control.params.get("multiplier", 1.0)
        token_positions = control.params.get("token_positions", "all")
        transform = control.params.get("transform", "additive")

        if transform != "additive":
            raise ValueError(
                f"Unsupported transform: {transform!r}. "
                f"Currently supported: 'additive'."
            )

        model_layers = get_model_layers(model)

        if layer_param is not None:
            layer_indices = [layer_param]
        else:
            layer_indices = sorted(directions.keys())

        hooks: list[torch.utils.hooks.RemovableHook] = []

        for layer_idx in layer_indices:
            if layer_idx not in directions:
                raise KeyError(
                    f"No steering vector for layer {layer_idx}. "
                    f"Available layers: {sorted(directions.keys())}"
                )

            layer = model_layers[layer_idx]
            sv = directions[layer_idx]

            def _make_hook(sv: torch.Tensor, mult: float, positions: list[int] | str):
                def hook(module, input, output):
                    hidden_states = output[0]
                    if positions == "all":
                        hidden_states = (
                            hidden_states + sv.to(hidden_states.device) * mult
                        )
                    else:
                        for pos in positions:
                            hidden_states[:, pos, :] = (
                                hidden_states[:, pos, :]
                                + sv.to(hidden_states.device) * mult
                            )
                    return (hidden_states, *output[1:])

                return hook

            h = layer.register_forward_hook(_make_hook(sv, multiplier, token_positions))
            hooks.append(h)

        return hooks

    def deactivate(self, handle: Any) -> None:
        """Remove all registered forward hooks.

        Args:
            handle: The list of hook handles returned by ``activate()``.
        """
        for h in handle:
            h.remove()
