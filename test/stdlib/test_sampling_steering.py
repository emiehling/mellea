"""Unit tests for steering integration in the sampling loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.core.requirement import Requirement, ValidationResult
from mellea.core.steering import (
    BackendCapabilities,
    Composer,
    ControlCategory,
    SteeringPolicy,
)
from mellea.stdlib.steering.composers import NoOpComposer
from mellea.steering.controls import input_control


class _TrackingComposer(Composer):
    """Test composer that tracks calls."""

    def __init__(self) -> None:
        """Initialize tracking composer."""
        self.compose_calls: list[tuple] = []
        self.update_calls: list[tuple] = []
        self._policy = SteeringPolicy(controls=(input_control("tracking"),))

    def compose(self, requirements, capabilities):
        self.compose_calls.append((requirements, capabilities))
        return self._policy

    def update(self, current_policy, validation_results, capabilities):
        self.update_calls.append((current_policy, validation_results, capabilities))
        return current_policy


# --- SamplingResult with steering_policy ---


def test_sampling_result_steering_policy_none_by_default():
    from mellea.core import ComputedModelOutputThunk, ModelOutputThunk, SamplingResult

    mot = ModelOutputThunk(value="test")
    computed = ComputedModelOutputThunk(mot)
    result = SamplingResult(
        result_index=0,
        success=True,
        sample_generations=[computed],
        sample_validations=[],
        sample_actions=[],
        sample_contexts=[],
    )
    assert result.steering_policy is None


def test_sampling_result_with_steering_policy():
    from mellea.core import ComputedModelOutputThunk, ModelOutputThunk, SamplingResult

    mot = ModelOutputThunk(value="test")
    computed = ComputedModelOutputThunk(mot)
    policy = SteeringPolicy(controls=(input_control("test"),))
    result = SamplingResult(
        result_index=0,
        success=True,
        sample_generations=[computed],
        sample_validations=[],
        sample_actions=[],
        sample_contexts=[],
        steering_policy=policy,
    )
    assert result.steering_policy is policy
    assert len(result.steering_policy.controls) == 1


# --- Backend attach/detach ---


def test_backend_attach_detach():
    """Test attach/detach on DummyBackend."""
    from mellea.backends.dummy import DummyBackend

    backend = DummyBackend(responses=["hello"])
    policy = SteeringPolicy(controls=(input_control("test"),))

    # Initially empty
    assert not backend.steering_policy

    # Attach
    backend.attach(policy)
    assert backend.steering_policy is policy
    assert backend.steering_policy.controls[0].name == "test"

    # Detach
    detached = backend.detach()
    assert detached is policy
    assert not backend.steering_policy


def test_backend_detach_when_none():
    from mellea.backends.dummy import DummyBackend

    backend = DummyBackend(responses=None)
    detached = backend.detach()
    assert detached is None


# --- Backend capabilities ---


def test_dummy_backend_capabilities():
    from mellea.backends.dummy import DummyBackend

    backend = DummyBackend(responses=None)
    caps = backend.capabilities
    assert isinstance(caps, BackendCapabilities)
    assert caps.supported_categories == frozenset()


# --- NoOpComposer in sampling loop ---


def test_noop_composer_produces_none_steering_policy():
    """NoOpComposer composes empty policy, so steering_policy should be empty."""
    composer = NoOpComposer()
    caps = BackendCapabilities()
    policy = composer.compose([], caps)
    assert not policy  # Empty policy


# --- Tracking composer ---


def test_tracking_composer_compose():
    composer = _TrackingComposer()
    reqs = [Requirement("test")]
    caps = BackendCapabilities()

    policy = composer.compose(reqs, caps)
    assert len(composer.compose_calls) == 1
    assert composer.compose_calls[0] == (reqs, caps)
    assert policy.controls[0].name == "tracking"


def test_tracking_composer_update():
    composer = _TrackingComposer()
    caps = BackendCapabilities()
    policy = SteeringPolicy.empty()
    val_results = [(Requirement("test"), ValidationResult(False, reason="fail"))]

    updated = composer.update(policy, val_results, caps)
    assert len(composer.update_calls) == 1
    assert updated is policy
