"""Tests for steering core abstractions."""

import logging
from dataclasses import dataclass

import pytest

from mellea.steering import (
    InputControl,
    OutputControl,
    StateControl,
    SteeringCapabilities,
    SteeringOptimizer,
    SteeringPolicy,
)

from .conftest import (
    MockInputControl,
    MockOptimizer,
    MockOutputControl,
    MockStateControl,
)


class TestSteeringPolicy:
    """SteeringPolicy construction and properties."""

    def test_empty_policy(self):
        """SteeringPolicy() with no controls is empty."""
        policy = SteeringPolicy()
        assert policy.is_empty()

    def test_policy_with_input_controls(self):
        """Policy with input controls is not empty."""
        ic = MockInputControl(tag="test")
        policy = SteeringPolicy(input_controls=(ic,))
        assert not policy.is_empty()

    def test_policy_with_state_controls(self):
        """Policy with state controls is not empty."""
        sc = MockStateControl(label="test")
        policy = SteeringPolicy(state_controls=(sc,))
        assert not policy.is_empty()

    def test_policy_with_output_controls(self):
        """Policy with output controls is not empty."""
        oc = MockOutputControl(value=0.5)
        policy = SteeringPolicy(output_controls=(oc,))
        assert not policy.is_empty()

    def test_backend_policy_excludes_input(self):
        """backend_policy strips input controls."""
        ic = MockInputControl(tag="input")
        sc = MockStateControl(label="state")
        oc = MockOutputControl(value=0.7)

        policy = SteeringPolicy(
            input_controls=(ic,), state_controls=(sc,), output_controls=(oc,)
        )
        bp = policy.backend_policy

        assert bp.input_controls == ()
        assert bp.state_controls == (sc,)
        assert bp.output_controls == (oc,)

    def test_backend_policy_of_empty(self):
        """backend_policy of empty policy is empty."""
        policy = SteeringPolicy()
        assert policy.backend_policy.is_empty()

    def test_policy_is_frozen(self):
        """Cannot mutate policy fields."""
        policy = SteeringPolicy()
        with pytest.raises(AttributeError):
            policy.input_controls = ()  # type: ignore

    def test_policy_is_hashable(self):
        """Policies can be used as dict keys / in sets."""
        sc = MockStateControl(label="test")
        policy = SteeringPolicy(state_controls=(sc,))
        # should not raise
        hash(policy)
        # should work in a set
        assert len({policy}) == 1

    def test_equal_policies_have_equal_hashes(self):
        """Two policies with identical controls are equal and have same hash."""
        sc = MockStateControl(label="test")
        p1 = SteeringPolicy(state_controls=(sc,))
        p2 = SteeringPolicy(state_controls=(sc,))

        assert p1 == p2
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1  # deduplicates in a set


class TestSteeringCapabilities:
    """SteeringCapabilities filtering and support checks."""

    def test_empty_capabilities_support_nothing(self):
        """Empty capabilities reject all state/output controls."""
        caps = SteeringCapabilities()
        sc = MockStateControl(label="test")
        oc = MockOutputControl(value=0.5)
        assert not caps.supports(sc)
        assert not caps.supports(oc)

    def test_supports_declared_type(self):
        """supports() returns True for a declared control type."""
        caps = SteeringCapabilities(
            supported_control_types=frozenset({MockStateControl})
        )
        sc = MockStateControl(label="test")
        assert caps.supports(sc)

    def test_rejects_undeclared_type(self):
        """supports() returns False for an undeclared control type."""
        caps = SteeringCapabilities(
            supported_control_types=frozenset({MockStateControl})
        )
        oc = MockOutputControl(value=0.5)
        assert not caps.supports(oc)

    def test_filter_policy_keeps_supported(self):
        """filter_policy keeps controls whose types are in capabilities."""
        caps = SteeringCapabilities(
            supported_control_types=frozenset({MockStateControl, MockOutputControl})
        )
        sc = MockStateControl(label="state")
        oc = MockOutputControl(value=0.5)
        policy = SteeringPolicy(state_controls=(sc,), output_controls=(oc,))

        filtered = caps.filter_policy(policy)

        assert filtered.state_controls == (sc,)
        assert filtered.output_controls == (oc,)

    def test_filter_policy_removes_unsupported(self):
        """filter_policy removes controls whose types are not in capabilities."""
        caps = SteeringCapabilities(
            supported_control_types=frozenset({MockStateControl})
        )
        sc = MockStateControl(label="state")
        oc = MockOutputControl(value=0.5)
        policy = SteeringPolicy(state_controls=(sc,), output_controls=(oc,))

        filtered = caps.filter_policy(policy)

        assert filtered.state_controls == (sc,)
        assert filtered.output_controls == ()

    def test_filter_policy_logs_warnings(self, caplog):
        """filter_policy logs a warning for each removed control."""
        caps = SteeringCapabilities(supported_control_types=frozenset())
        sc = MockStateControl(label="state")
        oc = MockOutputControl(value=0.5)
        policy = SteeringPolicy(state_controls=(sc,), output_controls=(oc,))

        with caplog.at_level(logging.WARNING):
            caps.filter_policy(policy)

        assert "MockStateControl" in caplog.text
        assert "MockOutputControl" in caplog.text

    def test_filter_policy_preserves_order(self):
        """Filtered controls maintain their original order."""
        caps = SteeringCapabilities(
            supported_control_types=frozenset({MockStateControl})
        )
        sc1 = MockStateControl(label="first")
        sc2 = MockStateControl(label="second")
        sc3 = MockStateControl(label="third")
        policy = SteeringPolicy(state_controls=(sc1, sc2, sc3))

        filtered = caps.filter_policy(policy)

        assert filtered.state_controls == (sc1, sc2, sc3)

    def test_filter_empty_policy(self):
        """Filtering an empty policy returns an empty policy."""
        caps = SteeringCapabilities(
            supported_control_types=frozenset({MockStateControl})
        )
        policy = SteeringPolicy()

        filtered = caps.filter_policy(policy)

        assert filtered.is_empty()


class TestInputControlABC:
    """InputControl interface contract."""

    def test_cannot_instantiate_abc(self):
        """InputControl cannot be instantiated directly."""
        with pytest.raises(TypeError):
            InputControl()  # type: ignore

    def test_concrete_subclass_must_implement_apply(self):
        """A subclass without apply() cannot be instantiated."""

        @dataclass(frozen=True)
        class IncompleteInputControl(InputControl):
            tag: str
            # missing apply method

        with pytest.raises(TypeError):
            IncompleteInputControl(tag="test")  # type: ignore

    def test_concrete_subclass_works(self):
        """A frozen dataclass subclass with apply() works."""
        ic = MockInputControl(tag="test")
        # should return (action, ctx) tuple - using None for simplicity
        result = ic.apply(None, None)  # type: ignore
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestStateControlABC:
    """StateControl is a frozen dataclass ABC."""

    def test_base_class_has_no_fields(self):
        """StateControl base class has no dataclass fields."""
        # StateControl is a dataclass ABC but has no fields itself.
        # Concrete subclasses add their own fields.
        # The ABC marker is for type checking, not runtime prevention.
        assert StateControl.__dataclass_fields__ == {}

    def test_concrete_subclass_is_frozen(self):
        """A concrete StateControl subclass is immutable."""
        sc = MockStateControl(label="test")
        with pytest.raises(AttributeError):
            sc.label = "modified"  # type: ignore

    def test_concrete_subclass_is_hashable(self):
        """A concrete StateControl subclass is hashable."""
        sc = MockStateControl(label="test")
        # should not raise
        hash(sc)
        # should work in a set
        assert len({sc}) == 1


class TestOutputControlABC:
    """OutputControl is a frozen dataclass ABC."""

    def test_base_class_has_no_fields(self):
        """OutputControl base class has no dataclass fields."""
        # OutputControl is a dataclass ABC but has no fields itself.
        # Concrete subclasses add their own fields.
        # The ABC marker is for type checking, not runtime prevention.
        assert OutputControl.__dataclass_fields__ == {}

    def test_concrete_subclass_is_frozen(self):
        """A concrete OutputControl subclass is immutable."""
        oc = MockOutputControl(value=0.5)
        with pytest.raises(AttributeError):
            oc.value = 0.9  # type: ignore

    def test_concrete_subclass_is_hashable(self):
        """A concrete OutputControl subclass is hashable."""
        oc = MockOutputControl(value=0.5)
        # should not raise
        hash(oc)
        # should work in a set
        assert len({oc}) == 1


class TestSteeringOptimizerABC:
    """SteeringOptimizer interface contract."""

    def test_cannot_instantiate_abc(self):
        """SteeringOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SteeringOptimizer()  # type: ignore

    @pytest.mark.asyncio
    async def test_default_refine_returns_policy_unchanged(self):
        """The default refine() implementation is identity."""
        sc = MockStateControl(label="test")
        policy = SteeringPolicy(state_controls=(sc,))
        optimizer = MockOptimizer()
        caps = SteeringCapabilities()

        refined = await optimizer.refine(policy, [], [], caps)

        assert refined is policy
