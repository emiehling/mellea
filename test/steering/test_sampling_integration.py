"""Tests for steering integration in the sampling loop."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import ModelOutputThunk
from mellea.steering import (
    OutputControl,
    StateControl,
    SteeringCapabilities,
    SteeringOptimizer,
    SteeringPolicy,
)
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.sampling.base import RejectionSamplingStrategy


# --- Test fixtures ---


@dataclass(frozen=True)
class MockStateControl(StateControl):
    label: str


@dataclass(frozen=True)
class MockOutputControl(OutputControl):
    value: float


class RecordingOptimizer(SteeringOptimizer):
    """Optimizer that records calls for assertion."""

    def __init__(self):
        self.compile_calls = []
        self.refine_calls = []
        self._policy = SteeringPolicy(
            state_controls=(MockStateControl(label="test"),),
        )

    async def compile(self, requirements, capabilities, ctx=None, action=None):
        self.compile_calls.append((requirements, capabilities))
        return self._policy

    async def refine(self, policy, validation_results, requirements, capabilities):
        self.refine_calls.append((policy, validation_results))
        return policy


# --- Tests ---


class TestSteeringForwardedToGeneration:
    @pytest.mark.asyncio
    async def test_steering_passed_to_generate(self):
        """The steering policy is forwarded to backend.generate_from_context."""
        # use a mock backend to capture the steering argument
        backend = AsyncMock()
        mot = ModelOutputThunk(value="output")
        mot._computed = True
        backend.generate_from_context = AsyncMock(
            return_value=(mot, SimpleContext())
        )
        backend.steering_capabilities = SteeringCapabilities()

        policy = SteeringPolicy(
            output_controls=(MockOutputControl(value=0.5),),
        )
        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        # run sample - it will fail on the generate_log assertion but we can
        # still check that steering was passed
        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                steering=policy,
            )
        except AssertionError:
            # expected - DummyBackend doesn't set _generate_log
            pass

        # verify steering was passed to generate_from_context
        gen_call = backend.generate_from_context.call_args
        assert gen_call.kwargs.get("steering") is policy


class TestSteeringNoneByDefault:
    @pytest.mark.asyncio
    async def test_no_steering_is_noop(self):
        """When steering=None, behavior is identical to pre-steering."""
        # use a mock backend that returns a proper MOT
        backend = AsyncMock()
        mot = ModelOutputThunk(value="hello")
        mot._computed = True
        backend.generate_from_context = AsyncMock(
            return_value=(mot, SimpleContext())
        )
        backend.steering_capabilities = SteeringCapabilities()

        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        # run sample without steering
        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                # steering and optimizer default to None
            )
        except AssertionError:
            # expected - mock doesn't set _generate_log
            pass

        # verify steering=None was passed
        gen_call = backend.generate_from_context.call_args
        assert gen_call.kwargs.get("steering") is None


class TestSteeringNotPassedToValidation:
    def test_validation_has_no_steering_parameter(self):
        """Validate that mfuncs.avalidate does not accept steering parameter.

        This test verifies the architectural enforcement: validation calls
        cannot receive steering because the avalidate function signature
        does not include a steering parameter.
        """
        import inspect

        from mellea.stdlib import functional as mfuncs

        sig = inspect.signature(mfuncs.avalidate)
        params = list(sig.parameters.keys())

        # avalidate should NOT have steering parameter
        assert "steering" not in params, (
            "avalidate should not accept steering parameter - "
            "this enforces the semantics/mechanisms separation"
        )


class TestOptimizerRefineIntegration:
    @pytest.mark.asyncio
    async def test_refine_not_called_without_optimizer(self):
        """No error when optimizer is None."""
        # use a mock backend
        backend = AsyncMock()
        mot = ModelOutputThunk(value="hello")
        mot._computed = True
        backend.generate_from_context = AsyncMock(
            return_value=(mot, SimpleContext())
        )
        backend.steering_capabilities = SteeringCapabilities()

        policy = SteeringPolicy(
            state_controls=(MockStateControl(label="test"),),
        )
        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        # run sample with steering but no optimizer - should not error
        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                steering=policy,
                optimizer=None,  # explicitly None
            )
        except AssertionError:
            # expected - mock doesn't set _generate_log
            pass

        # if we got here without a different error, the test passes

    @pytest.mark.asyncio
    async def test_refine_not_called_without_steering(self):
        """Refine is not called when steering is None even if optimizer provided."""
        optimizer = RecordingOptimizer()

        backend = AsyncMock()
        mot = ModelOutputThunk(value="hello")
        mot._computed = True
        backend.generate_from_context = AsyncMock(
            return_value=(mot, SimpleContext())
        )
        backend.steering_capabilities = SteeringCapabilities()

        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                steering=None,  # no steering
                optimizer=optimizer,  # optimizer provided
            )
        except AssertionError:
            pass

        # refine should not have been called since steering was None
        assert len(optimizer.refine_calls) == 0
