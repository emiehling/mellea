"""Optimizer — translates requirements into a Policy."""

from __future__ import annotations

import abc

from ..core.base import CBlock, Component, Context
from ..core.requirement import Requirement, ValidationResult
from .policy import Policy


class Optimizer(abc.ABC):
    """Analyzes full set of requirements and produces a steering policy.

    The Optimizer consists of two main operations:
    - compile: requirements -> policy (called once before the sampling loop)
    - refine: policy + (validation) failures -> adjusted policy (called on each retry)

    Both are async to support LLM-based or I/O-bound optimizer implementations.
    """

    @abc.abstractmethod
    async def compile(
        self,
        requirements: list[Requirement],
        supported_controls: frozenset[type],
        ctx: Context | None = None,
        action: Component | CBlock | None = None,
    ) -> Policy:
        """Analyze requirements and produce a steering policy.

        Args:
            requirements: The full set of requirements to optimize for.
            supported_controls: The backend's supported control types.
            ctx: The current context, if available.
            action: The action component, if available. Lets the optimizer skip inapplicable input controls 
                (e.g., FewShot for a non-Instruction action).

        Returns:
            A steering Policy containing the compiled controls.
        """
        ...

    async def refine(
        self,
        policy: Policy,
        validation_results: list[ValidationResult],
        requirements: list[Requirement],
        supported_controls: frozenset[type],
    ) -> Policy:
        """Adjust the steering policy after a validation failure.

        Called within the sampling loop when a generation attempt fails. The returned steering policy replaces the current policy for subsequent retries. 
        All control types (input and backend) may be adjusted; input controls are re-applied fresh each iteration.

        The default implementation returns the steering policy unchanged.

        Args:
            policy: The current full steering policy.
            validation_results: Results from the failed validation.
            requirements: The full requirement set.
            supported_controls: The backend's supported control types.

        Returns:
            An adjusted Policy for the next retry.
        """
        return policy
