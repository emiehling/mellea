"""SteeringCapabilities — declares what control types a backend can execute."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .controls import OutputControl, StateControl
from .policy import SteeringPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SteeringCapabilities:
    """Declares the set of steering control types a backend supports.

    Capabilities are declared as a frozenset of concrete control types
    (not strings). This is type-safe and extensible; adding a new control
    type never requires modifying existing capability declarations.

    Input controls are not checked here as they are always executable regardless 
    of backend.
    """

    supported_control_types: frozenset[type] = field(default_factory=frozenset)

    def supports(self, control: StateControl | OutputControl) -> bool:
        """Check if a specific state or output control is supported."""
        return type(control) in self.supported_control_types

    def filter_policy(self, policy: SteeringPolicy) -> SteeringPolicy:
        """Return a new policy with unsupported controls removed.

        Logs a warning for each removed control. The returned policy
        contains only state/output controls (input controls are not
        present in backend policies by construction).

        Args:
            policy: A backend policy (typically from SteeringPolicy.backend_policy).

        Returns:
            A new SteeringPolicy with unsupported controls filtered out.
        """
        supported_state: list[StateControl] = []
        supported_output: list[OutputControl] = []

        for sc in policy.state_controls:
            if self.supports(sc):
                supported_state.append(sc)
            else:
                logger.warning(
                    "Steering control %s not supported by backend; "
                    "removed from policy.",
                    type(sc).__name__,
                )

        for oc in policy.output_controls:
            if self.supports(oc):
                supported_output.append(oc)
            else:
                logger.warning(
                    "Steering control %s not supported by backend; "
                    "removed from policy.",
                    type(oc).__name__,
                )

        return SteeringPolicy(
            state_controls=tuple(supported_state),
            output_controls=tuple(supported_output),
        )
