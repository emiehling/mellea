"""YAML-backed store for INPUT prompt artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import ArtifactStore, semantic_match

_META_KEYS = ("description", "handler", "_subdir")


class PromptStore(ArtifactStore):
    """YAML-backed store for INPUT prompt artifacts.

    Manages two sub-types under one directory:

    - ``templates/``: Prompt template YAML files.
    - ``example_pools/``: ICL example pool YAML files.

    Each YAML file contains the artifact content and default metadata.
    All keys except ``description``, ``handler``, and internal bookkeeping
    are passed through as ``default_params``, so handler-specific parameters
    (e.g. ``system_prompt``, ``template``) flow through automatically.

    Args:
        root (str | Path): Path to the prompt artifacts directory.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize PromptStore, scanning the root directory for YAML artifacts."""
        self._root = Path(root)
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Scan subdirectories and load all YAML artifacts into memory."""
        for subdir in ("templates", "example_pools"):
            path = self._root / subdir
            if not path.is_dir():
                continue
            for yaml_file in sorted(path.glob("*.yaml")):
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data is None:
                    continue
                name = yaml_file.stem
                data["_subdir"] = subdir
                self._artifacts[name] = data

    @staticmethod
    def _extract_params(data: dict[str, Any]) -> dict[str, Any]:
        """Extract handler params from a YAML data dict."""
        return {k: v for k, v in data.items() if k not in _META_KEYS}

    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        """Load a prompt artifact by name.

        Args:
            **selectors: Must include ``name`` (str).

        Returns:
            ``(content, default_params)`` where content is the template string
            or example list, and default_params contains handler parameters.

        Raises:
            KeyError: If no artifact with the given name is found.
        """
        name = selectors.get("name")
        if name is None:
            raise KeyError("PromptStore.get_raw requires a 'name' selector")

        if name not in self._artifacts:
            raise KeyError(f"No prompt artifact found with name: {name!r}")

        data = self._artifacts[name]
        subdir = data.get("_subdir", "templates")

        if subdir == "example_pools":
            content = data.get("examples", [])
        else:
            content = data.get("template", "")

        return content, self._extract_params(data)

    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        """Search for prompt artifacts matching a text query.

        Args:
            query: Keyword or natural-language sentence describing the
                desired prompt artifact.
            model: Unused (prompt artifacts are model-agnostic).

        Returns:
            List of matching artifact metadata dicts.
        """
        candidates: list[str] = []
        result_dicts: list[dict[str, Any]] = []

        for name, data in self._artifacts.items():
            desc = data.get("description", "")
            candidates.append(f"{name}: {desc}")
            result_dicts.append(
                {
                    "name": name,
                    "description": desc,
                    "model": None,
                    "handler": data.get("handler"),
                    "default_params": self._extract_params(data),
                }
            )

        matched = semantic_match(query, candidates)
        return [result_dicts[i] for i in matched]

    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        """List available prompt artifacts.

        Args:
            **partial_selectors: Optional ``artifact_type`` filter
                (``"template"`` or ``"example_pool"``).

        Returns:
            List of artifact metadata dicts.
        """
        type_filter = partial_selectors.get("artifact_type")
        subdir_map = {"template": "templates", "example_pool": "example_pools"}

        results: list[dict[str, Any]] = []
        for name, data in self._artifacts.items():
            subdir = data.get("_subdir", "templates")
            if type_filter is not None and subdir_map.get(type_filter) != subdir:
                continue

            results.append(
                {
                    "name": name,
                    "description": data.get("description", ""),
                    "model": None,
                    "handler": data.get("handler"),
                    "default_params": self._extract_params(data),
                }
            )

        return results
