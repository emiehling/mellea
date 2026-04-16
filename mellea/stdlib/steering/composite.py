"""Composite ``Composer`` implementation."""

from __future__ import annotations

from ...core.requirement import Requirement, ValidationResult
from ...core.steering import BackendCapabilities, Composer, SteeringPolicy
from .per_requirement import PerRequirementComposer


class CompositeComposer(Composer):
    """Composer that analyzes the full requirement set together.

    Placeholder for more sophisticated composition strategies that consider
    requirement interactions and side-effects. Initially delegates to
    ``PerRequirementComposer``.

    Todo: extended with collaborative-filtering or recommendation-system techniques.
    """

    def __init__(self) -> None:
        """Initialize CompositeComposer."""
        self._delegate = PerRequirementComposer()

    def compose(
        self, requirements: list[Requirement], capabilities: BackendCapabilities
    ) -> SteeringPolicy:
        """Construct a policy by analyzing the full requirement set.

        Currently delegates to ``PerRequirementComposer``.

        Args:
            requirements: The full set of requirements for this generation call.
            capabilities: The backend's declared steering capabilities.

        Returns:
            A ``SteeringPolicy`` combining all matched artifacts.
        """
        return self._delegate.compose(requirements, capabilities)

    def update(
        self,
        current_policy: SteeringPolicy,
        validation_results: list[tuple[Requirement, ValidationResult]],
        capabilities: BackendCapabilities,
    ) -> SteeringPolicy:
        """Refine the policy based on validation failures.

        Currently delegates to ``PerRequirementComposer``.

        Args:
            current_policy: The steering policy used in the previous generation.
            validation_results: Per-requirement validation outcomes.
            capabilities: The backend's declared steering capabilities.

        Returns:
            An updated ``SteeringPolicy``.
        """
        return self._delegate.update(current_policy, validation_results, capabilities)
