"""Tests for steering integration in functional.py."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import Component, Context, Requirement
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import aact, act, ainstruct, instruct
from mellea.steering import InputControl, Optimizer, Policy


@dataclass(frozen=True)
class MockInputControl(InputControl):
    """A mock input control that adds text to the action."""

    text_to_add: str = "test"

    def apply(
        self, action: Component, ctx: Context
    ) -> tuple[Component, Context]:
        """Modify the action by adding grounding context."""
        if isinstance(action, Instruction):
            return action.with_additional_grounding(
                {"_mock_key": self.text_to_add}
            ), ctx
        return action, ctx


class MockOptimizer(Optimizer):
    """Mock optimizer for testing."""

    def __init__(
        self,
        input_controls: tuple[InputControl, ...] = (),
    ):
        self.input_controls = input_controls
        self.compile_call_count = 0
        self.compile_args: list[tuple] = []
        self.refine_call_count = 0

    async def compile(
        self,
        requirements: list[Requirement],
        supported_controls: frozenset[type],
        ctx: Context | None = None,
        action: Component | None = None,
    ) -> Policy:
        """Return a policy with configured controls."""
        self.compile_call_count += 1
        self.compile_args.append((requirements, supported_controls, ctx, action))
        return Policy(input_controls=self.input_controls)

    async def refine(
        self,
        policy: Policy,
        validation_results,
        requirements,
        supported_controls,
    ) -> Policy:
        """Track refine calls."""
        self.refine_call_count += 1
        return policy


class TestAactSteeringCompilation:
    """Tests for steering compilation in aact."""

    @pytest.mark.asyncio
    async def test_optimizer_compile_called_with_requirements(self):
        """compile() is called when optimizer and requirements are present."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        reqs = [Requirement("test req")]
        optimizer = MockOptimizer()

        await aact(
            action,
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=optimizer,
        )

        assert optimizer.compile_call_count == 1
        # check that requirements were passed
        call_args = optimizer.compile_args[0]
        assert call_args[0] == reqs

    @pytest.mark.asyncio
    async def test_optimizer_not_called_without_requirements(self):
        """compile() is NOT called when requirements are empty/None."""
        backend = DummyBackend(responses=["hello", "world"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        optimizer = MockOptimizer()

        # empty requirements
        await aact(
            action,
            ctx,
            backend,
            requirements=[],
            strategy=None,
            optimizer=optimizer,
        )
        assert optimizer.compile_call_count == 0

        # None requirements
        await aact(
            action,
            ctx,
            backend,
            requirements=None,
            strategy=None,
            optimizer=optimizer,
        )
        assert optimizer.compile_call_count == 0

    @pytest.mark.asyncio
    async def test_optimizer_not_called_when_none(self):
        """No error when optimizer is None (backward compat)."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        reqs = [Requirement("test req")]

        # should not raise
        result, _new_ctx = await aact(
            action,
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=None,
        )
        assert result.value == "hello"

    @pytest.mark.asyncio
    async def test_input_controls_applied_to_action(self):
        """Input controls transform the action before generation."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        reqs = [Requirement("test req")]

        mock_control = MockInputControl(text_to_add="enriched")
        optimizer = MockOptimizer(input_controls=(mock_control,))

        await aact(
            action,
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=optimizer,
        )

        assert optimizer.compile_call_count == 1

    @pytest.mark.asyncio
    async def test_input_controls_applied_in_order(self):
        """Multiple input controls are applied in tuple order."""
        applied_order: list[str] = []

        @dataclass(frozen=True)
        class OrderTrackingControl(InputControl):
            name: str = "default"

            def apply(
                self, action: Component, ctx: Context
            ) -> tuple[Component, Context]:
                applied_order.append(self.name)
                return action, ctx

        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        reqs = [Requirement("test req")]

        controls = (
            OrderTrackingControl(name="first"),
            OrderTrackingControl(name="second"),
            OrderTrackingControl(name="third"),
        )
        optimizer = MockOptimizer(input_controls=controls)

        await aact(
            action,
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=optimizer,
        )

        assert applied_order == ["first", "second", "third"]


class TestActSyncWrapper:
    """Tests for the sync act() wrapper."""

    def test_optimizer_forwarded_to_aact(self):
        """The sync act() forwards optimizer to aact()."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        reqs = [Requirement("test req")]
        optimizer = MockOptimizer()

        act(
            action,
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=optimizer,
        )

        assert optimizer.compile_call_count == 1


class TestInstructWithOptimizer:
    """Tests for instruct/ainstruct with optimizer."""

    def test_instruct_accepts_optimizer(self):
        """instruct() accepts optimizer kwarg."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        reqs = [Requirement("test req")]
        optimizer = MockOptimizer()

        instruct(
            "Test instruction",
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=optimizer,
        )

        assert optimizer.compile_call_count == 1

    @pytest.mark.asyncio
    async def test_ainstruct_accepts_optimizer(self):
        """ainstruct() accepts optimizer kwarg."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        reqs = [Requirement("test req")]
        optimizer = MockOptimizer()

        await ainstruct(
            "Test instruction",
            ctx,
            backend,
            requirements=reqs,
            strategy=None,
            optimizer=optimizer,
        )

        assert optimizer.compile_call_count == 1


class TestNoOptimizerBackwardCompat:
    """Tests for backward compatibility without optimizer."""

    @pytest.mark.asyncio
    async def test_aact_without_optimizer_unchanged(self):
        """aact behavior is identical when optimizer is None."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")

        result, _new_ctx = await aact(action, ctx, backend, strategy=None)

        assert result.value == "hello"

    def test_act_without_optimizer_unchanged(self):
        """act behavior is identical when optimizer is None."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")

        result, _new_ctx = act(action, ctx, backend, strategy=None)

        assert result.value == "hello"
