"""This module holds shim backends used for smoke tests."""

from collections.abc import Sequence

from ..core import (
    Backend,
    BackendCapabilities,
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    ModelOutputThunk,
)


class DummyBackend(Backend):
    """A backend for smoke testing.

    Returns predetermined string responses in sequence, or ``"dummy"`` if no
    responses are provided. Intended for unit tests and integration smoke tests
    where real model inference is not needed.

    Args:
        responses (list[str] | None): Ordered list of strings to return on
            successive ``generate_from_context`` calls, or ``None`` to always
            return ``"dummy"``.

    Attributes:
        idx (int): Index of the next response to return from ``responses``;
            starts at ``0`` and increments on each call.
    """

    def __init__(self, responses: list[str] | None):
        """Initialize the dummy backend with an optional list of predetermined responses."""
        super().__init__()
        self.responses = responses
        self.idx = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        """Returns empty capabilities -- the dummy backend supports no steering."""
        return BackendCapabilities()

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Return the next predetermined response for ``action`` given ``ctx``.

        If ``responses`` is ``None``, always returns the string ``"dummy"``.
        Otherwise returns the next item from ``responses`` in order.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Must be ``None``; constrained
                decoding is not supported.
            model_options (dict | None): Ignored by this backend.
            tool_calls (bool): Ignored by this backend.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the predetermined
                response and an updated context.

        Raises:
            AssertionError: If ``format`` is not ``None``.
            Exception: If all responses from ``responses`` have been consumed.
        """
        assert format is None, "The DummyBackend does not support constrained decoding."
        if self.responses is None:
            mot = ModelOutputThunk(value="dummy")
            return mot, ctx.add(action).add(mot)
        elif self.idx < len(self.responses):
            return_value = ModelOutputThunk(value=self.responses[self.idx])
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
        """Return predetermined responses for each action.

        Args:
            actions: List of actions to generate responses for.
            ctx: Context (ignored).
            format: Must be ``None``; constrained decoding is not supported.
            model_options: Ignored.
            tool_calls: Ignored.

        Returns:
            List of ``ModelOutputThunk`` instances, one per action.
        """
        results = []
        for _ in actions:
            mot, _ = await self._generate_from_context(actions[0], ctx, format=format)
            results.append(mot)
        return results
