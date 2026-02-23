"""Tests for steering integration in the sampling loop."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.sampling.base import RejectionSamplingStrategy
from mellea.steering import BackendControl, Optimizer, Policy

# --- Test fixtures ---


@dataclass(frozen=True)
class MockBackendControl(BackendControl):
    label: str


@dataclass(frozen=True)
class MockBackendControlAlt(BackendControl):
    value: float


class RecordingOptimizer(Optimizer):
    """Optimizer that records calls for assertion."""

    def __init__(self):
        self.compile_calls = []
        self.refine_calls = []
        self._policy = Policy(
            backend_controls=(MockBackendControl(label="test"),)
        )

    async def compile(self, requirements, supported_controls, ctx=None, action=None):
        self.compile_calls.append((requirements, supported_controls))
        return self._policy

    async def refine(self, policy, validation_results, requirements, supported_controls):
        self.refine_calls.append((policy, validation_results))
        return policy


# --- Tests ---


class TestPolicyForwardedToGeneration:
    @pytest.mark.asyncio
    async def test_policy_passed_to_generate(self):
        """Backend controls from the policy are forwarded to backend.generate_from_context."""
        # use a mock backend to capture the policy argument.
        # give it supported_controls that support our control so it survives filtering.
        backend = AsyncMock()
        mot = ModelOutputThunk(value="output")
        mot._computed = True
        backend.generate_from_context = AsyncMock(return_value=(mot, SimpleContext()))
        backend.supported_controls = frozenset({MockBackendControlAlt})

        policy = Policy(
            backend_controls=(MockBackendControlAlt(value=0.5),)
        )
        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        # run sample - it will fail on the generate_log assertion but we can
        # still check that policy was passed
        try:
            await strategy.sample(
                action, SimpleContext(), backend, requirements=None, policy=policy
            )
        except AssertionError:
            # expected - DummyBackend doesn't set _generate_log
            pass

        # verify the backend received the output control via the filtered backend policy
        gen_call = backend.generate_from_context.call_args
        forwarded = gen_call.kwargs.get("policy")
        assert forwarded is not None
        assert forwarded.backend_controls == (MockBackendControlAlt(value=0.5),)


class TestPolicyNoneByDefault:
    @pytest.mark.asyncio
    async def test_no_policy_is_noop(self):
        """When policy=None, behavior is identical to pre-steering."""
        # use a mock backend that returns a proper MOT
        backend = AsyncMock()
        mot = ModelOutputThunk(value="hello")
        mot._computed = True
        backend.generate_from_context = AsyncMock(return_value=(mot, SimpleContext()))
        backend.supported_controls = frozenset()

        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        # run sample without policy
        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                # policy and optimizer default to None
            )
        except AssertionError:
            # expected - mock doesn't set _generate_log
            pass

        # verify policy=None was passed
        gen_call = backend.generate_from_context.call_args
        assert gen_call.kwargs.get("policy") is None


class TestPolicyNotPassedToValidation:
    def test_validation_has_no_policy_parameter(self):
        """Validate that mfuncs.avalidate does not accept policy parameter.

        This test verifies the architectural enforcement: validation calls
        cannot receive policy because the avalidate function signature
        does not include a policy parameter.
        """
        import inspect

        from mellea.stdlib import functional as mfuncs

        sig = inspect.signature(mfuncs.avalidate)
        params = list(sig.parameters.keys())

        # avalidate should NOT have policy parameter
        assert "policy" not in params, (
            "avalidate should not accept policy parameter - "
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
        backend.generate_from_context = AsyncMock(return_value=(mot, SimpleContext()))
        backend.supported_controls = frozenset()

        policy = Policy(backend_controls=(MockBackendControl(label="test"),))
        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        # run sample with policy but no optimizer - should not error
        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                policy=policy,
                optimizer=None,  # explicitly None
            )
        except AssertionError:
            # expected - mock doesn't set _generate_log
            pass

        # if we got here without a different error, the test passes

    @pytest.mark.asyncio
    async def test_refine_not_called_without_policy(self):
        """Refine is not called when policy is None even if optimizer provided."""
        optimizer = RecordingOptimizer()

        backend = AsyncMock()
        mot = ModelOutputThunk(value="hello")
        mot._computed = True
        backend.generate_from_context = AsyncMock(return_value=(mot, SimpleContext()))
        backend.supported_controls = frozenset()

        strategy = RejectionSamplingStrategy(loop_budget=1)
        action = Instruction(description="Test")

        try:
            await strategy.sample(
                action,
                SimpleContext(),
                backend,
                requirements=None,
                policy=None,  # no policy
                optimizer=optimizer,  # optimizer provided
            )
        except AssertionError:
            pass

        # refine should not have been called since policy was None
        assert len(optimizer.refine_calls) == 0
