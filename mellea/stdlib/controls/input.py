"""Concrete input controls for inference-time steering.

Input controls transform Components before formatting. They operate at
the Mellea level and require no backend support. Each control is a frozen
dataclass with a deterministic apply() method that only calls public
builder methods on Components.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.base import CBlock, Component, Context
from ...steering.controls import InputControl
from ..components.instruction import Instruction


@dataclass(frozen=True)
class FewShotControl(InputControl):
    """Inject additional in-context learning examples into an Instruction.

    Calls Instruction.with_additional_examples(). No-ops for non-Instruction
    actions (Message, GenerativeSlot, etc.).

    Fields:
        examples: A tuple of example strings or CBlocks to append.
            Uses tuple (not list) for hashability/immutability.
    """

    examples: tuple[str | CBlock, ...]

    def apply(
        self, action: Component | CBlock, ctx: Context
    ) -> tuple[Component | CBlock, Context]:
        """Apply the control by appending examples to the action."""
        if isinstance(action, Instruction):
            return action.with_additional_examples(list(self.examples)), ctx
        return action, ctx


@dataclass(frozen=True)
class GroundingControl(InputControl):
    """Inject additional grounding context into an Instruction.

    Calls Instruction.with_additional_grounding(). No-ops for non-Instruction
    actions. Raises KeyError if any key conflicts with existing grounding
    (propagated from Instruction.with_additional_grounding).

    Fields:
        entries: A tuple of (key, value) pairs to add to grounding context.
            Uses tuple-of-tuples (not dict) for hashability/immutability.
    """

    entries: tuple[tuple[str, str | CBlock | Component], ...]

    def apply(
        self, action: Component | CBlock, ctx: Context
    ) -> tuple[Component | CBlock, Context]:
        """Apply the control by adding grounding entries to the action."""
        if isinstance(action, Instruction):
            return action.with_additional_grounding(dict(self.entries)), ctx
        return action, ctx
