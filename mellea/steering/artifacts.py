"""Steering artifact registry for managing pretrained interventions.

Provides ``SteeringArtifact`` (metadata about a pretrained intervention) and
``ArtifactRegistry`` (a searchable catalog of available artifacts). Backends
resolve ``Control.artifact_ref`` values against a registry instance to load
the actual artifact objects at ``attach()`` time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.steering import ControlCategory


@dataclass(frozen=True)
class SteeringArtifact:
    """Metadata about a pretrained steering intervention.

    Args:
        name (str): Unique identifier for this artifact.
        description (str): Human-readable description used for semantic search
            when matching artifacts to requirements.
        category (ControlCategory): Which stage of the pipeline this artifact
            targets.
        model_family (str | None): Model family this artifact was trained for,
            or ``None`` if model-agnostic.
        artifact_type (str): Kind of artifact (e.g., ``"steering_vector"``,
            ``"prompt_adapter"``, ``"reward_model"``).
        path_or_ref (str): Filesystem path or registry reference for loading
            the actual artifact object.
    """

    name: str
    description: str
    category: ControlCategory
    model_family: str | None
    artifact_type: str
    path_or_ref: str


class ArtifactRegistry:
    """A searchable catalog of available steering artifacts.

    Artifacts are registered with ``register()`` and retrieved via ``search()``
    with optional filtering by query text, category, or model family.
    """

    def __init__(self) -> None:
        """Initialize an empty artifact registry."""
        self._artifacts: list[SteeringArtifact] = []

    def register(self, artifact: SteeringArtifact) -> None:
        """Add an artifact to the registry.

        Args:
            artifact: The steering artifact to register.
        """
        self._artifacts.append(artifact)

    def search(
        self,
        query: str | None = None,
        category: ControlCategory | None = None,
        model_family: str | None = None,
    ) -> list[SteeringArtifact]:
        """Search the registry for matching artifacts.

        All filters are optional; when multiple are provided they are combined
        with AND semantics.

        Args:
            query: Substring to match against artifact ``name`` or ``description``.
            category: Filter to artifacts targeting this pipeline stage.
            model_family: Filter to artifacts trained for this model family, or
                model-agnostic artifacts (``model_family is None``).

        Returns:
            List of matching artifacts in registration order.
        """
        results = self._artifacts
        if category is not None:
            results = [a for a in results if a.category == category]
        if model_family is not None:
            results = [
                a
                for a in results
                if a.model_family is None or a.model_family == model_family
            ]
        if query is not None:
            query_lower = query.lower()
            results = [
                a
                for a in results
                if query_lower in a.name.lower() or query_lower in a.description.lower()
            ]
        return results

    def resolve(self, ref: str) -> Any:
        """Load the actual artifact object from the registry.

        Currently returns the path/reference string for the matched artifact.
        Backends should override or extend this to perform actual artifact loading.

        Args:
            ref: The artifact reference string (matching ``SteeringArtifact.path_or_ref``).

        Returns:
            The artifact's ``path_or_ref`` value.

        Raises:
            KeyError: If no artifact with the given reference is found.
        """
        for artifact in self._artifacts:
            if artifact.path_or_ref == ref:
                return artifact.path_or_ref
        raise KeyError(f"No artifact found with reference: {ref}")

    def __len__(self) -> int:
        """Return the number of registered artifacts."""
        return len(self._artifacts)
