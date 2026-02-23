"""Tests for symmetric input control refinement in the sampling loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import Component, Context, Requirement
from mellea.core.requirement import ValidationResult
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import aact
from mellea.stdlib.sampling.base import RejectionSamplingStrategy
from mellea.steering import InputControl, Optimizer, Policy
from mellea.steering.policy import apply_input_controls

# --- test controls and optimizers ---


@dataclass(frozen=True)
class RecordingControl(InputControl):
    """Input control that records each apply() call."""

    tag: str = "default"
    record: list[str] = field(default_factory=list, compare=False, hash=False)

    def apply(self, action: Component, ctx: Context) -> tuple[Component, Context]:
        self.record.append(self.tag)
        return action, ctx


class AlwaysFailRequirement(Requirement):
    """Requirement that always fails validation."""

    def __init__(self):
        super().__init__(
            description="always fails",
            validation_fn=lambda ctx: ValidationResult(
                result=False, reason="forced failure"
            ),
        )


class PassOnNthRequirement(Requirement):
    """Requirement that passes on the nth validation attempt."""

    def __init__(self, pass_on: int = 2):
        self._call_count = 0
        self._pass_on = pass_on
        super().__init__(
            description="pass on nth attempt", validation_fn=self._validate
        )

    def _validate(self, ctx: Context) -> ValidationResult:
        self._call_count += 1
        if self._call_count >= self._pass_on:
            return ValidationResult(result=True)
        return ValidationResult(result=False, reason="not yet")


class FixedPolicyOptimizer(Optimizer):
    """Optimizer that returns a fixed policy from compile and refine."""

    def __init__(
        self,
        compile_policy: Policy,
        refine_policy: Policy | None = None,
    ):
        self._compile_policy = compile_policy
        self._refine_policy = refine_policy
        self.compile_call_count = 0
        self.refine_call_count = 0

    async def compile(self, requirements, supported_controls, ctx=None, action=None):
        self.compile_call_count += 1
        return self._compile_policy

    async def refine(self, policy, validation_results, requirements, supported_controls):
        self.refine_call_count += 1
        if self._refine_policy is not None:
            return self._refine_policy
        return policy


class SwappingOptimizer(Optimizer):
    """Optimizer whose refine() swaps to a different input control."""

    def __init__(self, initial_record: list[str], refined_record: list[str]):
        self._initial_control = RecordingControl(tag="initial", record=initial_record)
        self._refined_control = RecordingControl(tag="refined", record=refined_record)
        self._initial_policy = Policy(input_controls=(self._initial_control,))
        self._refined_policy = Policy(input_controls=(self._refined_control,))

    async def compile(self, requirements, supported_controls, ctx=None, action=None):
        return self._initial_policy

    async def refine(self, policy, validation_results, requirements, supported_controls):
        return self._refined_policy


# --- tests ---


class TestInputControlReapplication:
    @pytest.mark.asyncio
    async def test_control_applied_each_iteration(self):
        """Input controls are applied once per loop iteration, not once total."""
        loop_budget = 3
        record: list[str] = []
        control = RecordingControl(tag="per-iter", record=record)
        optimizer = FixedPolicyOptimizer(
            compile_policy=Policy(input_controls=(control,))
        )

        backend = DummyBackend(responses=["a", "b", "c"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        req = AlwaysFailRequirement()

        await aact(
            action,
            ctx,
            backend,
            requirements=[req],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
            return_sampling_results=True,
            optimizer=optimizer,
        )

        assert len(record) == loop_budget
        assert record == ["per-iter"] * loop_budget


class TestInputControlRefinement:
    @pytest.mark.asyncio
    async def test_refined_controls_used_after_first_iteration(self):
        """refine() can swap input controls; the new ones are used on subsequent iterations."""
        loop_budget = 3
        initial_record: list[str] = []
        refined_record: list[str] = []
        optimizer = SwappingOptimizer(initial_record, refined_record)

        backend = DummyBackend(responses=["a", "b", "c"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        req = AlwaysFailRequirement()

        await aact(
            action,
            ctx,
            backend,
            requirements=[req],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
            return_sampling_results=True,
            optimizer=optimizer,
        )

        # iteration 1 uses the compiled policy (initial control)
        assert initial_record == ["initial"]
        # iterations 2-3 use the refined policy (refined control)
        assert refined_record == ["refined", "refined"]


class TestRepairSeesBaseActions:
    @pytest.mark.asyncio
    async def test_no_key_error_on_grounding_reapplication(self):
        """GroundingControl re-application doesn't crash because repair sees the base action."""
        from mellea.stdlib.controls.grounding import Grounding

        control = Grounding(entries=(("test_key", "test_value"),))
        optimizer = FixedPolicyOptimizer(
            compile_policy=Policy(input_controls=(control,))
        )

        # two responses: first attempt fails, second passes
        backend = DummyBackend(responses=["bad", "good"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        req = PassOnNthRequirement(pass_on=2)

        # this would crash with KeyError under the old design
        result = await aact(
            action,
            ctx,
            backend,
            requirements=[req],
            strategy=RejectionSamplingStrategy(loop_budget=2),
            return_sampling_results=True,
            optimizer=optimizer,
        )

        assert result.success
        # verify the grounding control was actually applied (steered actions have the key)
        for sa in result.sample_actions:
            if isinstance(sa, Instruction):
                assert "test_key" in sa._grounding_context


class TestNoSteeringBackwardCompat:
    @pytest.mark.asyncio
    async def test_no_optimizer_no_steering(self):
        """aact without optimizer or steering behaves identically to pre-steering."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")

        result, _new_ctx = await aact(action, ctx, backend, strategy=None)

        assert result.value == "hello"

    @pytest.mark.asyncio
    async def test_no_steering_with_strategy(self):
        """Sampling loop works without steering."""
        req = PassOnNthRequirement(pass_on=1)
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")

        result = await aact(
            action,
            ctx,
            backend,
            requirements=[req],
            strategy=RejectionSamplingStrategy(loop_budget=2),
            return_sampling_results=True,
        )

        assert result.success


class TestNoStrategyInlineApplication:
    @pytest.mark.asyncio
    async def test_input_controls_applied_once_without_strategy(self):
        """With strategy=None, input controls are applied exactly once."""
        record: list[str] = []
        control = RecordingControl(tag="inline", record=record)
        optimizer = FixedPolicyOptimizer(
            compile_policy=Policy(input_controls=(control,))
        )

        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        action = Instruction(description="Test")
        req = Requirement(description="test req")

        await aact(
            action, ctx, backend, requirements=[req], strategy=None, optimizer=optimizer
        )

        assert len(record) == 1
        assert record == ["inline"]


class TestApplyInputControlsFunction:
    def test_empty_policy_is_noop(self):
        """apply_input_controls with no input controls returns args unchanged."""
        policy = Policy()
        action = Instruction(description="Test")
        ctx = SimpleContext()

        result_action, result_ctx = apply_input_controls(policy, action, ctx)

        assert result_action is action
        assert result_ctx is ctx

    def test_controls_applied_in_order(self):
        """Controls are applied in tuple order."""
        order: list[str] = []

        @dataclass(frozen=True)
        class OrderControl(InputControl):
            name: str = "default"
            record: list[str] = field(default_factory=list, compare=False, hash=False)

            def apply(self, action, ctx):
                self.record.append(self.name)
                return action, ctx

        c1 = OrderControl(name="first", record=order)
        c2 = OrderControl(name="second", record=order)
        c3 = OrderControl(name="third", record=order)
        policy = Policy(input_controls=(c1, c2, c3))

        apply_input_controls(policy, Instruction(description="Test"), SimpleContext())

        assert order == ["first", "second", "third"]
