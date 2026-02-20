"""SteeringOptimizer — translates requirements into a SteeringPolicy."""

from __future__ import annotations

import abc

from ..core.base import CBlock, Component, Context
from ..core.requirement import Requirement, ValidationResult
from .capabilities import SteeringCapabilities
from .policy import SteeringPolicy


class SteeringOptimizer(abc.ABC):
    """Analyzes requirements holistically and produces a steering policy.

    The optimizer is the load-bearing abstraction in the steering system.
    It ensures that controls for multiple requirements are composed
    coherently rather than naively merged.

    Two operations:
    - compile: requirements -> policy (called once before the sampling loop)
    - refine: policy + failures -> adjusted policy (called on each retry)

    Both are async to support LLM-based or I/O-bound optimizer implementations.
    """

    @abc.abstractmethod
    async def compile(
        self,
        requirements: list[Requirement],
        capabilities: SteeringCapabilities,
        ctx: Context | None = None,
        action: Component | CBlock | None = None,
    ) -> SteeringPolicy:
        """Analyze requirements and produce a steering policy.

        Args:
            requirements: The full set of requirements to optimize for.
            capabilities: The backend's declared capabilities (to ensure backend can execute).
            ctx: The current context, if available.
            action: The action component, if available. Lets the optimizer
                skip inapplicable input controls (e.g., FewShotControl for
                a non-Instruction action).

        Returns:
            A SteeringPolicy containing the compiled controls.
        """
        ...

    async def refine(
        self,
        policy: SteeringPolicy,
        validation_results: list[ValidationResult],
        requirements: list[Requirement],
        capabilities: SteeringCapabilities,
    ) -> SteeringPolicy:
        """Adjust the backend policy after a validation failure.

        Called within the sampling loop when a generation attempt fails.
        The returned policy replaces the current backend policy for
        subsequent retries.

        Only state and output controls can be adjusted; input controls
        were applied before the sampling loop and cannot be changed.
        This is structurally enforced (the strategy passes only the
        backend_policy to refine).

        The default implementation returns the policy unchanged.

        Args:
            policy: The current backend policy (state + output only).
            validation_results: Results from the failed validation.
            requirements: The full requirement set.
            capabilities: The backend's capabilities.

        Returns:
            An adjusted SteeringPolicy for the next retry.
        """
        return policy
