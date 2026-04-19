"""Zarr-backed store for STATE steering vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
    import xarray as xr

    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False

from .base import ArtifactStore

_META_KEYS = ("description", "handler", "param_space")


class VectorStore(ArtifactStore):
    """Zarr-backed store for STATE steering vectors.

    Steering vectors are stored as xarray DataArrays in a zarr directory store,
    indexed by ``(model, behavior, layer)`` coordinates. Per-vector defaults
    (layers, coefficient, positions, transform) are stored as coordinate-level
    attributes.

    Requires ``xarray`` and ``zarr``.

    Args:
        root (str | Path): Path to the zarr directory store root.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize VectorStore, opening the zarr store at root."""
        if not _HAS_XARRAY:
            raise ImportError(
                "VectorStore requires xarray, zarr, and numpy. "
                "Install them with: pip install 'mellea[vectors]'"
            )
        self._root = Path(root)
        self._ds = xr.open_zarr(self._root)

    @staticmethod
    def _split_meta(meta: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
        """Split a meta dict into (description, handler, params)."""
        desc = meta.get("description", "")
        handler = meta.get("handler")
        params = {k: v for k, v in meta.items() if k not in _META_KEYS}
        return desc, handler, params

    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        """Load per-layer steering vectors by model and behavior.

        Args:
            **selectors: Must include ``model`` and ``behavior``, or a
                composite ``name`` in ``"model/behavior"`` format (split on
                the last ``/`` so model IDs containing ``/`` work correctly).

        Returns:
            ``(directions, default_params)`` where directions is a
            ``dict[int, Tensor]`` mapping layer indices to direction vectors,
            and default_params contains handler parameters.

        Raises:
            KeyError: If no vector matches the selectors.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "VectorStore.get_raw requires PyTorch. "
                "Install it with: pip install 'mellea[hf]'"
            ) from e

        model = selectors.get("model")
        behavior = selectors.get("behavior")

        if model is None and behavior is None:
            name = selectors.get("name")
            if name is not None:
                parts = name.rsplit("/", 1)
                if len(parts) == 2:
                    model, behavior = parts

        if model is None or behavior is None:
            raise KeyError(
                "VectorStore.get_raw requires 'model' and 'behavior' selectors "
                "(or a 'name' in 'model/behavior' format)"
            )

        try:
            da = self._ds["vectors"].sel(model=model, behavior=behavior)
        except KeyError:
            raise KeyError(
                f"No steering vector found for model={model!r}, behavior={behavior!r}"
            ) from None

        arr = da.values  # shape: (num_layers, hidden_dim)
        directions: dict[int, Any] = {}
        for layer_idx in range(arr.shape[0]):
            directions[layer_idx] = torch.from_numpy(arr[layer_idx]).float()

        attrs = self._ds.attrs if hasattr(self._ds, "attrs") else {}
        meta_key = f"{model}/{behavior}"
        meta = attrs.get(meta_key, {})
        if isinstance(meta, dict):
            _desc, _handler, params = self._split_meta(meta)
        else:
            params = {}

        return directions, params

    @staticmethod
    def _normalize_param_space(raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize param_space keys that were stringified by JSON round-trip."""
        by_layer = raw.get("by_layer")
        if by_layer is None:
            return raw
        return {
            **raw,
            "by_layer": {int(k): v for k, v in by_layer.items()},
        }

    def _build_info_dict(self, m: str, b: str) -> dict[str, Any]:
        """Build an info dict for a (model, behavior) pair."""
        meta_key = f"{m}/{b}"
        meta = self._ds.attrs.get(meta_key, {})
        if isinstance(meta, dict):
            desc, handler, params = self._split_meta(meta)
            raw_ps = meta.get("param_space", {})
            param_space = self._normalize_param_space(raw_ps) if raw_ps else {}
        else:
            desc, handler, params, param_space = "", None, {}, {}
        return {
            "name": f"{m}/{b}",
            "description": desc,
            "model": m,
            "handler": handler,
            "default_params": params,
            "param_space": param_space,
        }

    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        """Search for steering vectors matching a text query.

        Args:
            query: Substring to match against behavior names and descriptions.
            model: Optional model family filter.

        Returns:
            List of matching artifact metadata dicts.
        """
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        models = (
            [model]
            if model is not None
            else [str(m) for m in self._ds.coords["model"].values]
        )
        behaviors = [str(b) for b in self._ds.coords["behavior"].values]

        for m in models:
            if m not in [str(x) for x in self._ds.coords["model"].values]:
                continue
            for b in behaviors:
                info = self._build_info_dict(m, b)
                if (
                    query_lower in b.lower()
                    or query_lower in info["description"].lower()
                ):
                    results.append(info)

        return results

    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        """List available steering vectors.

        Args:
            **partial_selectors: Optional ``model`` filter.

        Returns:
            List of artifact metadata dicts.
        """
        results: list[dict[str, Any]] = []
        model_filter = partial_selectors.get("model")

        models = (
            [model_filter]
            if model_filter is not None
            else [str(m) for m in self._ds.coords["model"].values]
        )
        behaviors = [str(b) for b in self._ds.coords["behavior"].values]

        for m in models:
            if m not in [str(x) for x in self._ds.coords["model"].values]:
                continue
            for b in behaviors:
                results.append(self._build_info_dict(m, b))

        return results
