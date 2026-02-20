"""This module holds shim backends used for smoke tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..core import (
    Backend,
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    GenerateLog,
    ModelOutputThunk,
)

if TYPE_CHECKING:
    from ..steering.policy import SteeringPolicy


class DummyBackend(Backend):
    """A backend for smoke testing."""

    def __init__(self, responses: list[str] | None):
        """Initializes the dummy backend, optionally with a list of dummy responses.

        Args:
            responses: If `None`, then the dummy backend always returns "dummy". Otherwise, returns the next item from responses. The generate function will throw an exception if a generate call is made after the list is exhausted.
        """
        self.responses = responses
        self.idx = 0

    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
        steering: SteeringPolicy | None = None,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """See constructor for an exmplanation of how DummyBackends work."""
        assert format is None, "The DummyBackend does not support constrained decoding."
        if self.responses is None:
            mot = ModelOutputThunk(value="dummy")
            mot._generate_log = GenerateLog(backend="DummyBackend")
            return mot, ctx.add(action).add(mot)
        elif self.idx < len(self.responses):
            return_value = ModelOutputThunk(value=self.responses[self.idx])
            return_value._generate_log = GenerateLog(backend="DummyBackend")
            self.idx += 1
            return return_value, ctx.add(action).add(return_value)
        else:
            raise Exception(
                f"DummyBackend expected no more than {len(self.responses)} calls."
            )

    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Generate from raw inputs. Returns dummy responses for each action."""
        results = []
        for _ in actions:
            mot, _ = await self.generate_from_context(
                CBlock("dummy"), ctx, format=format, model_options=model_options
            )
            results.append(mot)
        return results
