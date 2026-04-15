"""No-op ``Composer`` implementation."""

from __future__ import annotations

from ...core.requirement import Requirement, ValidationResult
from ...core.steering import BackendCapabilities, Composer, SteeringPolicy


class NoOpComposer(Composer):
    """Passthrough composer that produces empty steering policies.

    When configured, the control flow is identical to the current unsteered flow.
    This is the default when no ``Composer`` is explicitly set, and allows existing
    repair strategies to still own the loop without interference.
    """

    def compose(
        self, requirements: list[Requirement], capabilities: BackendCapabilities
    ) -> SteeringPolicy:
        """Return an empty steering policy.

        Args:
            requirements: Ignored.
            capabilities: Ignored.

        Returns:
            An empty ``SteeringPolicy``.
        """
        return SteeringPolicy.empty()

    def update(
        self,
        current_policy: SteeringPolicy,
        validation_results: list[tuple[Requirement, ValidationResult]],
        capabilities: BackendCapabilities,
    ) -> SteeringPolicy:
        """Return the current policy unchanged.

        Args:
            current_policy: The existing steering policy.
            validation_results: Ignored.
            capabilities: Ignored.

        Returns:
            The unchanged ``current_policy``.
        """
        return current_policy
