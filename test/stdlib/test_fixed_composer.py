"""Unit tests for FixedComposer."""

from mellea.core.requirement import Requirement, ValidationResult
from mellea.core.steering import (
    BackendCapabilities,
    Control,
    ControlCategory,
    SteeringPolicy,
)
from mellea.stdlib.steering import FixedComposer


def test_compose_returns_fixed_policy():
    c = Control(category=ControlCategory.OUTPUT, name="test", params={"temperature": 0.5})
    policy = SteeringPolicy(controls=(c,))
    composer = FixedComposer(policy)

    result = composer.compose(
        [Requirement("be concise")],
        BackendCapabilities(supported_categories=frozenset(ControlCategory)),
    )
    assert result is policy
    assert len(result.controls) == 1
    assert result.controls[0].params["temperature"] == 0.5


def test_compose_ignores_requirements_and_capabilities():
    policy = SteeringPolicy(controls=(
        Control(category=ControlCategory.STATE, name="vec"),
    ))
    composer = FixedComposer(policy)

    # Even with empty capabilities that wouldn't support STATE, the
    # FixedComposer returns the policy as-is — no filtering.
    result = composer.compose([], BackendCapabilities())
    assert result is policy
    assert result.controls[0].category == ControlCategory.STATE


def test_update_returns_current_policy_unchanged():
    original = SteeringPolicy(controls=(
        Control(category=ControlCategory.INPUT, name="test"),
    ))
    composer = FixedComposer(SteeringPolicy.empty())

    # update() returns whatever it's given, not the fixed policy.
    updated = composer.update(
        original,
        [(Requirement("test"), ValidationResult(False, reason="failed"))],
        BackendCapabilities(),
    )
    assert updated is original


def test_empty_fixed_policy():
    composer = FixedComposer(SteeringPolicy.empty())
    result = composer.compose([], BackendCapabilities())
    assert not result
    assert result.controls == ()
