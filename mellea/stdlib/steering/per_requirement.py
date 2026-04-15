"""Per-requirement ``Composer`` implementation."""

from __future__ import annotations

from ...core.requirement import Requirement, ValidationResult
from ...core.steering import BackendCapabilities, Composer, Control, SteeringPolicy
from ...steering.library import ArtifactLibrary


class PerRequirementComposer(Composer):
    """Simple composer that selects steering artifacts per requirement.

    For each requirement, searches the artifact library for matching interventions
    and composes them via ``SteeringPolicy.__add__()``. Populates
    ``Control.params`` from the artifact's default parameters so handlers receive
    artifact-level defaults without querying the library.

    Args:
        library (ArtifactLibrary): The artifact library to search for
            matching interventions.
    """

    def __init__(self, library: ArtifactLibrary) -> None:
        """Initialize PerRequirementComposer with an artifact library."""
        self.library = library

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
            infos = self.library.search(query=req.description)
            for info in infos:
                if info.category not in capabilities.supported_categories:
                    continue
                controls.append(
                    Control(
                        category=info.category,
                        name=info.handler or info.name,
                        params=dict(info.default_params),
                        artifact_ref=info.name,
                        model_family=info.model,
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
            infos = self.library.search(query=req.description)
            for info in infos:
                if (
                    info.category in capabilities.supported_categories
                    and info.name not in existing_refs
                ):
                    new_controls.append(
                        Control(
                            category=info.category,
                            name=info.handler or info.name,
                            params=dict(info.default_params),
                            artifact_ref=info.name,
                            model_family=info.model,
                        )
                    )
                    existing_refs.add(info.name)
        if new_controls:
            return current_policy + SteeringPolicy(controls=tuple(new_controls))
        return current_policy
