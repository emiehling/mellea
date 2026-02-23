"""FewShot — inject in-context learning examples into an Instruction."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.base import CBlock, Component, Context
from ...steering.controls import InputControl
from ..components.instruction import Instruction

CONTROL_INFO = {
    "kind": "input",
    "domain": "prompting",
    "summary": "Inject additional in-context learning examples.",
    "composable": True,
}


@dataclass(frozen=True)
class FewShot(InputControl):
    """Inject additional in-context learning examples into an Instruction.

    Calls Instruction.with_additional_examples(). 
    No-ops for non-Instruction actions (Message, GenerativeSlot, etc.).

    Fields:
        examples: A tuple of example strings or CBlocks to append.
    """

    examples: tuple[str | CBlock, ...]

    def apply(
        self, action: Component | CBlock, ctx: Context
    ) -> tuple[Component | CBlock, Context]:
        """Apply the control by appending examples to the action."""
        if isinstance(action, Instruction):
            return action.with_additional_examples(list(self.examples)), ctx
        return action, ctx
