"""Fixed composer that applies a pre-built steering policy unchanged."""

from __future__ import annotations

from ...core.requirement import Requirement, ValidationResult
from ...core.steering import BackendCapabilities, Composer, SteeringPolicy


class FixedComposer(Composer):
    """Composer that always returns a pre-built steering policy.

    The policy is returned as-is from ``compose()`` and never modified by
    ``update()``. Use this when you know exactly which controls to apply
    and do not want adaptive policy refinement.

    Args:
        policy (SteeringPolicy): The policy to apply on every generation.

    Examples:
        Build a policy manually and pass it through the sampling loop::

            from mellea.core.steering import Control, ControlCategory, SteeringPolicy
            from mellea.stdlib.steering import FixedComposer

            control = Control(category=ControlCategory.OUTPUT, name="static_output",
                              params={"temperature": 0.3})
            policy = SteeringPolicy(controls=(control,))

            m = start_session("hf", model_id, composer=FixedComposer(policy))
            result = m.instruct("Explain mutexes.", requirements=["Be concise."])
    """

    def __init__(self, policy: SteeringPolicy) -> None:
        """Initialize with a fixed steering policy."""
        self._policy = policy

    def compose(
        self, requirements: list[Requirement], capabilities: BackendCapabilities
    ) -> SteeringPolicy:
        """Return the fixed policy, ignoring requirements and capabilities.

        Args:
            requirements: Ignored.
            capabilities: Ignored.

        Returns:
            The pre-built ``SteeringPolicy``.
        """
        return self._policy

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
