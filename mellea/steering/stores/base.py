"""Abstract base class for per-category artifact stores."""

from __future__ import annotations

import abc
from typing import Any

_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL
_embedder: Any = None


def set_embedding_model(name: str) -> None:
    """Override the default embedding model name.

    Must be called before the first ``search()`` call.  Calling after the
    model has already been loaded raises ``RuntimeError``.
    """
    global _embedding_model_name, _embedder
    if _embedder is not None:
        raise RuntimeError(
            "Embedding model already loaded. "
            "Call set_embedding_model() before any search()."
        )
    _embedding_model_name = name


def _get_embedder() -> Any:
    """Return the shared SentenceTransformer, loading it on first call."""
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        import sentence_transformers

        _embedder = sentence_transformers.SentenceTransformer(_embedding_model_name)
        return _embedder
    except ImportError:
        raise ImportError(
            "semantic_match requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from None


def semantic_match(
    query: str, candidates: list[str], threshold: float = 0.4
) -> list[int]:
    """Return indices of candidates that semantically match the query.

    Uses cosine similarity between sentence embeddings.

    Args:
        query: The search query (keyword or natural-language sentence).
        candidates: List of candidate texts (e.g. "behavior: description").
        threshold: Minimum cosine similarity for a match.

    Returns:
        List of integer indices into ``candidates`` that match.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    if not candidates:
        return []

    import sentence_transformers

    embedder = _get_embedder()

    query_emb = embedder.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )
    cand_embs = embedder.encode(
        candidates, convert_to_tensor=True, normalize_embeddings=True
    )
    scores = sentence_transformers.util.cos_sim(query_emb, cand_embs)[0]

    return [i for i, s in enumerate(scores) if s >= threshold]


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

        Uses :func:`semantic_match` for embedding-based cosine similarity.
        Requires ``sentence-transformers``.

        Args:
            query: Keyword or natural-language sentence describing the
                desired artifact.
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
