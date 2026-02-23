"""Grounding — inject grounding context into an Instruction."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.base import CBlock, Component, Context
from ...steering.controls import InputControl
from ..components.instruction import Instruction

CONTROL_INFO = {
    "kind": "input",
    "domain": "prompting",
    "summary": "Inject additional grounding context (key-value pairs).",
    "composable": True,
}


@dataclass(frozen=True)
class Grounding(InputControl):
    """Inject additional grounding context into an Instruction.

    Calls Instruction.with_additional_grounding(). 
    No-ops for non-Instruction actions (Message, GenerativeSlot, etc.). 

    Fields:
        entries: A tuple of (key, value) pairs to add to grounding context.
    """

    entries: tuple[tuple[str, str | CBlock | Component], ...]

    def apply(
        self, action: Component | CBlock, ctx: Context
    ) -> tuple[Component | CBlock, Context]:
        """Apply the control by adding grounding entries to the action."""
        if isinstance(action, Instruction):
            return action.with_additional_grounding(dict(self.entries)), ctx
        return action, ctx
