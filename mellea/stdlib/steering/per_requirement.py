"""Per-requirement ``Composer`` implementation."""

from __future__ import annotations

from ...core.requirement import Requirement, ValidationResult
from ...core.steering import BackendCapabilities, Composer, Control, SteeringPolicy
from ...steering.artifacts import ArtifactRegistry


class PerRequirementComposer(Composer):
    """Simple composer that selects steering artifacts per requirement.

    For each requirement, searches the artifact registry for matching interventions
    and composes them via ``SteeringPolicy.__add__()``. This is a baseline strategy
    that partially violates DR#2 (steering should be a function of the full set)
    but is useful for simple cases.

    Args:
        registry (ArtifactRegistry): The artifact registry to search for
            matching interventions.
    """

    def __init__(self, registry: ArtifactRegistry) -> None:
        """Initialize PerRequirementComposer with an artifact registry."""
        self.registry = registry

    def compose(
        self, requirements: list[Requirement], capabilities: BackendCapabilities
    ) -> SteeringPolicy:
        """Construct a policy by searching for artifacts matching each requirement.

        Args:
            requirements: The full set of requirements for this generation call.
            capabilities: The backend's declared steering capabilities.

        Returns:
            A ``SteeringPolicy`` combining all matched artifacts.
        """
        controls: list[Control] = []
        for req in requirements:
            if req.description is None:
                continue
            artifacts = self.registry.search(query=req.description)
            for artifact in artifacts:
                if artifact.category in capabilities.supported_categories:
                    controls.append(
                        Control(
                            category=artifact.category,
                            name=artifact.name,
                            artifact_ref=artifact.path_or_ref,
                            model_family=artifact.model_family,
                        )
                    )
        return SteeringPolicy(controls=tuple(controls))

    def update(
        self,
        current_policy: SteeringPolicy,
        validation_results: list[tuple[Requirement, ValidationResult]],
        capabilities: BackendCapabilities,
    ) -> SteeringPolicy:
        """Refine the policy by searching for additional artifacts for failed requirements.

        Args:
            current_policy: The steering policy used in the previous generation.
            validation_results: Per-requirement validation outcomes.
            capabilities: The backend's declared steering capabilities.

        Returns:
            An updated ``SteeringPolicy``.
        """
        existing_refs = {
            c.artifact_ref for c in current_policy.controls if c.artifact_ref
        }
        new_controls: list[Control] = []
        for req, val in validation_results:
            if val.as_bool() or req.description is None:
                continue
            artifacts = self.registry.search(query=req.description)
            for artifact in artifacts:
                if (
                    artifact.category in capabilities.supported_categories
                    and artifact.path_or_ref not in existing_refs
                ):
                    new_controls.append(
                        Control(
                            category=artifact.category,
                            name=artifact.name,
                            artifact_ref=artifact.path_or_ref,
                            model_family=artifact.model_family,
                        )
                    )
                    existing_refs.add(artifact.path_or_ref)
        if new_controls:
            return current_policy + SteeringPolicy(controls=tuple(new_controls))
        return current_policy