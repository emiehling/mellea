"""Abstract base class for per-category artifact stores."""

from __future__ import annotations

import abc
from typing import Any


class ArtifactStore(abc.ABC):
    """Abstract base for per-category artifact stores.

    Each concrete store manages persistence, indexing, and loading for one
    control category's data shape. Stores return raw dicts for default handler
    parameters that flow directly into ``Control.params``.
    """

    @abc.abstractmethod
    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        """Load an artifact and its raw defaults dict.

        Args:
            **selectors: Store-specific selectors (e.g., ``model`` and
                ``behavior`` for ``VectorStore``; ``name`` for manifest stores).

        Returns:
            A ``(payload, defaults_dict)`` tuple where payload is the loaded
            artifact and defaults_dict contains default parameter overrides.

        Raises:
            KeyError: If no matching artifact is found.
        """
        ...

    @abc.abstractmethod
    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        """Search for artifacts matching a text query.

        Args:
            query: Substring to match against artifact names and descriptions.
            model: Optional model family filter.

        Returns:
            List of dicts with at least ``name``, ``description``, and
            ``category`` keys.
        """
        ...

    @abc.abstractmethod
    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        """List artifacts, optionally filtered by partial selectors.

        Args:
            **partial_selectors: Store-specific filters to narrow results.

        Returns:
            List of dicts with at least ``name``, ``description``, and
            ``category`` keys.
        """
        ...
