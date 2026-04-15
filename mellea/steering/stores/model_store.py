"""Manifest-backed store for OUTPUT model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import ArtifactStore


class ModelStore(ArtifactStore):
    """Manifest-backed store for OUTPUT model artifacts.

    Reads a ``manifest.yaml`` file listing available reward models and
    logits processors with their metadata and filesystem/Hub references.

    Args:
        manifest_path (str | Path): Path to the model manifest YAML file.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        """Initialize ModelStore from a manifest YAML file."""
        self._manifest_path = Path(manifest_path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load the manifest YAML and index entries by name."""
        with open(self._manifest_path) as f:
            data = yaml.safe_load(f)
        for entry in data.get("models", []):
            name = entry["name"]
            self._entries[name] = entry

    @staticmethod
    def _extract_params(entry: dict[str, Any]) -> dict[str, Any]:
        """Extract handler params from a manifest entry."""
        params: dict[str, Any] = {}
        stored = entry.get("defaults", {})
        if "temperature" in stored:
            params["temperature"] = stored["temperature"]
        return params

    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        """Load a model reference by name.

        Args:
            **selectors: Must include ``name`` (str).

        Returns:
            ``(ref, default_params)`` where ref is the filesystem path or
            HuggingFace Hub reference, and default_params contains
            handler parameters.

        Raises:
            KeyError: If no model with the given name is in the manifest.
        """
        name = selectors.get("name")
        if name is None:
            raise KeyError("ModelStore.get_raw requires a 'name' selector")

        if name not in self._entries:
            raise KeyError(f"No model found with name: {name!r}")

        entry = self._entries[name]
        ref = entry.get("ref", entry.get("path", ""))
        return ref, self._extract_params(entry)

    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        """Search for output models matching a text query.

        Args:
            query: Substring to match against model names and descriptions.
            model: Optional model family filter.

        Returns:
            List of matching artifact metadata dicts.
        """
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        for name, entry in self._entries.items():
            desc = entry.get("description", "")
            entry_model = entry.get("model_family")

            if model is not None and entry_model is not None and entry_model != model:
                continue

            if query_lower in name.lower() or query_lower in desc.lower():
                results.append(
                    {
                        "name": name,
                        "description": desc,
                        "model": entry_model,
                        "handler": entry.get("handler"),
                        "default_params": self._extract_params(entry),
                    }
                )

        return results

    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        """List available output models.

        Args:
            **partial_selectors: Optional ``model_family`` filter.

        Returns:
            List of artifact metadata dicts.
        """
        model_filter = partial_selectors.get("model_family")
        results: list[dict[str, Any]] = []

        for name, entry in self._entries.items():
            entry_model = entry.get("model_family")
            if model_filter is not None and entry_model != model_filter:
                continue

            results.append(
                {
                    "name": name,
                    "description": entry.get("description", ""),
                    "model": entry_model,
                    "handler": entry.get("handler"),
                    "default_params": self._extract_params(entry),
                }
            )

        return results
