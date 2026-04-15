"""Unified artifact library with per-category stores.

Provides ``ArtifactInfo`` (lightweight discovery record) and ``ArtifactLibrary``
(the unified interface over per-category ``ArtifactStore`` instances).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.steering import ControlCategory
from .stores.base import ArtifactStore


@dataclass(frozen=True)
class ArtifactInfo:
    """Lightweight discovery record returned by search and list operations.

    Does not hold the artifact itself — only metadata for discovery.

    Args:
        name (str): Artifact identifier.
        category (ControlCategory): Which pipeline stage this artifact targets.
        description (str): Human-readable description.
        model (str | None): Model family, or ``None`` if model-agnostic.
        handler (str | None): Default handler name for this artifact.
        default_params (dict[str, Any]): Default handler parameters. Keys match
            what the handler reads from ``control.params``.
    """

    name: str
    category: ControlCategory
    description: str
    model: str | None
    handler: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)


class ArtifactLibrary:
    """Unified interface over per-category artifact stores.

    Delegates ``get``, ``search``, and ``list`` calls to the appropriate
    category-specific ``ArtifactStore``. Categories without a registered store
    are unavailable and raise ``ValueError``.

    Args:
        stores (dict[ControlCategory, ArtifactStore] | None): Optional mapping
            of categories to their stores.
    """

    def __init__(
        self, stores: dict[ControlCategory, ArtifactStore] | None = None
    ) -> None:
        """Initialize the library with optional per-category stores."""
        self._stores: dict[ControlCategory, ArtifactStore] = (
            dict(stores) if stores else {}
        )

    def register_store(self, category: ControlCategory, store: ArtifactStore) -> None:
        """Register or replace a store for a control category.

        Args:
            category: The control category this store serves.
            store: The store implementation.
        """
        self._stores[category] = store

    def _get_store(self, category: ControlCategory) -> ArtifactStore:
        """Return the store for a category, or raise if not configured.

        Args:
            category: The control category.

        Returns:
            The configured ``ArtifactStore``.

        Raises:
            ValueError: If no store is configured for this category.
        """
        store = self._stores.get(category)
        if store is None:
            raise ValueError(
                f"No store configured for category {category.value!r}. "
                f"Register one via library.register_store({category!r}, store)."
            )
        return store

    def get(
        self, category: ControlCategory, **selectors: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Load an artifact and its default params from the appropriate store.

        Args:
            category: Which category to load from.
            **selectors: Store-specific keyword arguments (e.g., ``name``
                for manifest stores, ``model`` and ``behavior`` for
                ``VectorStore``).

        Returns:
            ``(artifact, default_params)`` — the loaded object and its
            default handler parameters.

        Raises:
            ValueError: If no store is configured for this category.
            KeyError: If no matching artifact is found.
        """
        store = self._get_store(category)
        return store.get_raw(**selectors)

    def search(
        self,
        query: str,
        category: ControlCategory | None = None,
        model: str | None = None,
    ) -> list[ArtifactInfo]:
        """Search across configured stores for matching artifacts.

        Args:
            query: Substring to match against artifact names and descriptions.
            category: Optional filter to search only one category's store.
            model: Optional model family filter.

        Returns:
            List of ``ArtifactInfo`` discovery records.
        """
        results: list[ArtifactInfo] = []
        stores = (
            {category: self._stores[category]}
            if category is not None and category in self._stores
            else self._stores
        )
        for cat, store in stores.items():
            for raw in store.search(query, model=model):
                results.append(
                    ArtifactInfo(
                        name=raw["name"],
                        category=cat,
                        description=raw.get("description", ""),
                        model=raw.get("model"),
                        handler=raw.get("handler"),
                        default_params=raw.get("default_params", {}),
                    )
                )
        return results

    def list(
        self, category: ControlCategory, **partial_selectors: Any
    ) -> list[ArtifactInfo]:
        """List available artifacts in a specific store.

        Args:
            category: Which category's store to list.
            **partial_selectors: Store-specific filters to narrow results.

        Returns:
            List of ``ArtifactInfo`` discovery records.

        Raises:
            ValueError: If no store is configured for this category.
        """
        store = self._get_store(category)
        results: list[ArtifactInfo] = []
        for raw in store.list_artifacts(**partial_selectors):
            results.append(
                ArtifactInfo(
                    name=raw["name"],
                    category=category,
                    description=raw.get("description", ""),
                    model=raw.get("model"),
                    handler=raw.get("handler"),
                    default_params=raw.get("default_params", {}),
                )
            )
        return results


_default_library: ArtifactLibrary | None = None


def get_default_library() -> ArtifactLibrary:
    """Return the default global artifact library, creating an empty one on first access.

    Returns:
        The singleton ``ArtifactLibrary`` instance.
    """
    global _default_library
    if _default_library is None:
        _default_library = ArtifactLibrary()
    return _default_library


def set_default_library(library: ArtifactLibrary) -> None:
    """Set the default global artifact library.

    Args:
        library: The library to use as the global default.
    """
    global _default_library
    _default_library = library
