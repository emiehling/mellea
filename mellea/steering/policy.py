"""Policy — immutable container of controls for a generation request."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .controls import BackendControl, InputControl

if TYPE_CHECKING:
    from ..core.base import CBlock, Component, Context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Policy:
    """An immutable, hashable collection of steering controls.

    New steering policies are created by constructing new Policy instances.

    EM: Note that this intentionally doesn't contain a public merge method. Mechanically combining two policies can produce conflicting controls. 
    Constructing a steering policy is done by the optimizer as a function of the whole list of requirements.
    """

    input_controls: tuple[InputControl, ...] = ()
    backend_controls: tuple[BackendControl, ...] = ()

    def is_empty(self) -> bool:
        """True if the policy contains no controls of any type."""
        return not (self.input_controls or self.backend_controls)

    @property
    def backend_policy(self) -> Policy:
        """The subset of controls requiring backend execution.

        Returns a new Policy containing only backend controls; convenience for call sites that forward controls to ``backend.generate_from_context``.
        """
        return Policy(backend_controls=self.backend_controls)

    def filter(self, supported: frozenset[type]) -> Policy:
        """Return a new steering policy with unsupported backend controls removed.

        Logs a warning for each removed control. The returned policy contains only backend controls whose type is in ``supported``.

        Args:
            supported: The set of backend control types the backend supports.

        Returns:
            A new Policy with unsupported controls filtered out.
        """
        kept: list[BackendControl] = []
        for ctrl in self.backend_controls:
            if type(ctrl) in supported:
                kept.append(ctrl)
            else:
                logger.warning(
                    "Steering control %s not supported by backend; removed from policy.",
                    type(ctrl).__name__,
                )
        return Policy(backend_controls=tuple(kept))



def apply_input_controls(
    policy: Policy, action: Component | CBlock, ctx: Context
) -> tuple[Component | CBlock, Context]:
    """Apply a steering policy's input controls to an action and context.

    Controls are applied in tuple order. Each control's ``apply`` method returns a new (action, context) pair via public builder methods on
    Components, so the original action is never mutated.

    Args:
        policy: The steering policy whose input controls to apply.
        action: The current action Component or CBlock.
        ctx: The current Context.

    Returns:
        The transformed (action, context) after all input controls.
    """
    for control in policy.input_controls:
        action, ctx = control.apply(action, ctx)
    return action, ctx
