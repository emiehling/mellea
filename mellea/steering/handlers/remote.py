"""Remote-backend handlers for state controls (activation steering over HTTP)."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from ...core.steering import Control, RemoteStateControlHandler

if TYPE_CHECKING:
    import torch

# torch.bfloat16 → np.int16 view-trick map (numpy has no native bf16 dtype).
_VIEW_MAP: dict[Any, Any] | None = None


def _get_view_map() -> dict[Any, Any]:
    """Lazily build the bf16 → int16 view map after torch/numpy are importable."""
    global _VIEW_MAP
    if _VIEW_MAP is None:
        import numpy as np
        import torch

        _VIEW_MAP = {torch.bfloat16: np.dtype(np.int16)}
    return _VIEW_MAP


def _serialize_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a torch.Tensor to a JSON-safe dict.

    Mirrors vllm-lens / vllm-hook v2's wire format: zstd-compressed bytes,
    base64 string, with bfloat16 → int16 view trick (numpy has no native bf16).

    Args:
        t: The tensor to serialize.

    Returns:
        Dict with ``data``, ``dtype``, ``original_dtype``, ``shape``, and
        ``compression`` keys.
    """
    try:
        import torch as _torch
        import zstandard
    except ImportError as e:
        raise ImportError(
            "Remote state steering requires torch and zstandard. "
            "Install them with: pip install 'mellea[remote_steering]'"
        ) from e

    view_map = _get_view_map()
    t = t.detach().cpu().contiguous()
    original_dtype = str(t.dtype)
    if t.dtype in view_map:
        arr = t.view(_torch.int16).numpy()
    else:
        arr = t.numpy()
    compressed = zstandard.ZstdCompressor().compress(arr.tobytes())
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
        "compression": "zstd",
    }


class VLLMSteeringRequestHandler(RemoteStateControlHandler):
    """Packages an activation-steering control into a vLLM-compatible request payload.

    Reads the resolved artifact (a ``dict[int, Tensor]`` from ``VectorStore``),
    selects the requested layer(s), serializes the tensor(s), and appends a
    ``SteeringVector``-shaped dict to
    ``extra_body["extra_args"]["apply_steering_vectors"]``.

    Expects ``control.params`` to optionally contain:

    - ``layer`` (int): Single layer index. If omitted, all layers in the
      artifact are sent (matches ``ActivationSteeringHandler`` semantics).
    - ``multiplier`` (float): Scale factor, default ``1.0``. Sent as ``scale``.
    - ``token_positions`` (list[int] | str): Forwarded as ``position_indices``;
      ``"all"`` (default) maps to ``None``.
    - ``norm_match`` (bool): Per-token L2 norm matching, default ``False``.
    """

    def contribute_to_request(
        self, control: Control, request_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Append a serialized steering vector to ``extra_body.extra_args``.

        Args:
            control: The control descriptor.
            request_kwargs: Current request kwargs dict.
            artifact: A ``dict[int, Tensor]`` from ``VectorStore.get_raw()``.

        Returns:
            The updated request kwargs.

        Raises:
            ValueError: If the artifact is missing or not a dict.
            KeyError: If ``control.params["layer"]`` is set but absent from the artifact.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Remote state steering requires torch. "
                "Install it with: pip install 'mellea[remote_steering]'"
            ) from e

        if artifact is None or not isinstance(artifact, dict):
            raise ValueError(
                "VLLMSteeringRequestHandler requires a dict[int, Tensor] artifact"
            )

        layer_param = control.params.get("layer")
        multiplier = control.params.get("multiplier", 1.0)
        token_positions = control.params.get("token_positions", "all")
        norm_match = control.params.get("norm_match", False)

        if layer_param is not None:
            if layer_param not in artifact:
                raise KeyError(
                    f"No steering vector for layer {layer_param}; "
                    f"available: {sorted(artifact.keys())}"
                )
            layer_indices: list[int] = [int(layer_param)]
            tensors: list[torch.Tensor] = [artifact[layer_param]]
        else:
            layer_indices = sorted(int(k) for k in artifact.keys())
            tensors = [artifact[i] for i in layer_indices]

        stacked = torch.stack(tensors, dim=0)  # (n_layers, hidden_dim)

        position_indices = None if token_positions == "all" else list(token_positions)

        sv_dict = {
            "activations": _serialize_tensor(stacked),
            "layer_indices": list(layer_indices),
            "scale": float(multiplier),
            "norm_match": bool(norm_match),
            "position_indices": position_indices,
        }

        extra_body = request_kwargs.setdefault("extra_body", {})
        extra_args = extra_body.setdefault("extra_args", {})
        vectors = extra_args.setdefault("apply_steering_vectors", [])
        vectors.append(sv_dict)

        return request_kwargs
