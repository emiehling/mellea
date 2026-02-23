"""Tests for steering core abstractions."""

import logging
from dataclasses import dataclass

import pytest

from mellea.steering import BackendControl, InputControl, Optimizer, Policy

from .conftest import (
    MockBackendControl,
    MockBackendControlAlt,
    MockInputControl,
    MockOptimizer,
)


class TestPolicy:
    """Policy construction and properties."""

    def test_empty_policy(self):
        """Policy() with no controls is empty."""
        policy = Policy()
        assert policy.is_empty()

    def test_policy_with_input_controls(self):
        """Policy with input controls is not empty."""
        ic = MockInputControl(tag="test")
        policy = Policy(input_controls=(ic,))
        assert not policy.is_empty()

    def test_policy_with_backend_controls(self):
        """Policy with backend controls is not empty."""
        bc = MockBackendControl(label="test")
        policy = Policy(backend_controls=(bc,))
        assert not policy.is_empty()

    def test_backend_policy_excludes_input(self):
        """backend_policy strips input controls."""
        ic = MockInputControl(tag="input")
        bc = MockBackendControl(label="backend")

        policy = Policy(input_controls=(ic,), backend_controls=(bc,))
        bp = policy.backend_policy

        assert bp.input_controls == ()
        assert bp.backend_controls == (bc,)

    def test_backend_policy_of_empty(self):
        """backend_policy of empty policy is empty."""
        policy = Policy()
        assert policy.backend_policy.is_empty()

    def test_policy_is_frozen(self):
        """Cannot mutate policy fields."""
        policy = Policy()
        with pytest.raises(AttributeError):
            policy.input_controls = ()  # type: ignore

    def test_policy_is_hashable(self):
        """Policies can be used as dict keys / in sets."""
        bc = MockBackendControl(label="test")
        policy = Policy(backend_controls=(bc,))
        # should not raise
        hash(policy)
        # should work in a set
        assert len({policy}) == 1

    def test_equal_policies_have_equal_hashes(self):
        """Two policies with identical controls are equal and have same hash."""
        bc = MockBackendControl(label="test")
        p1 = Policy(backend_controls=(bc,))
        p2 = Policy(backend_controls=(bc,))

        assert p1 == p2
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1  # deduplicates in a set


class TestPolicyFilter:
    """Policy.filter() support checks."""

    def test_empty_supported_rejects_all(self):
        """Empty supported set rejects all backend controls."""
        bc = MockBackendControl(label="test")
        alt = MockBackendControlAlt(value=0.5)
        policy = Policy(backend_controls=(bc, alt))

        filtered = policy.filter(frozenset())
        assert filtered.backend_controls == ()

    def test_filter_keeps_supported(self):
        """filter() keeps controls whose types are in supported."""
        supported = frozenset({MockBackendControl, MockBackendControlAlt})
        bc = MockBackendControl(label="backend")
        alt = MockBackendControlAlt(value=0.5)
        policy = Policy(backend_controls=(bc, alt))

        filtered = policy.filter(supported)

        assert filtered.backend_controls == (bc, alt)

    def test_filter_removes_unsupported(self):
        """filter() removes controls whose types are not in supported."""
        supported = frozenset({MockBackendControl})
        bc = MockBackendControl(label="backend")
        alt = MockBackendControlAlt(value=0.5)
        policy = Policy(backend_controls=(bc, alt))

        filtered = policy.filter(supported)

        assert filtered.backend_controls == (bc,)

    def test_filter_logs_warnings(self, caplog):
        """filter() logs a warning for each removed control."""
        bc = MockBackendControl(label="backend")
        alt = MockBackendControlAlt(value=0.5)
        policy = Policy(backend_controls=(bc, alt))

        with caplog.at_level(logging.WARNING):
            policy.filter(frozenset())

        assert "MockBackendControl" in caplog.text
        assert "MockBackendControlAlt" in caplog.text

    def test_filter_preserves_order(self):
        """Filtered controls maintain their original order."""
        supported = frozenset({MockBackendControl})
        bc1 = MockBackendControl(label="first")
        bc2 = MockBackendControl(label="second")
        bc3 = MockBackendControl(label="third")
        policy = Policy(backend_controls=(bc1, bc2, bc3))

        filtered = policy.filter(supported)

        assert filtered.backend_controls == (bc1, bc2, bc3)

    def test_filter_empty_policy(self):
        """Filtering an empty policy returns an empty policy."""
        supported = frozenset({MockBackendControl})
        policy = Policy()

        filtered = policy.filter(supported)

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


class TestBackendControlABC:
    """BackendControl is a frozen dataclass ABC."""

    def test_base_class_has_no_fields(self):
        """BackendControl base class has no dataclass fields."""
        assert BackendControl.__dataclass_fields__ == {}

    def test_concrete_subclass_is_frozen(self):
        """A concrete BackendControl subclass is immutable."""
        bc = MockBackendControl(label="test")
        with pytest.raises(AttributeError):
            bc.label = "modified"  # type: ignore

    def test_concrete_subclass_is_hashable(self):
        """A concrete BackendControl subclass is hashable."""
        bc = MockBackendControl(label="test")
        # should not raise
        hash(bc)
        # should work in a set
        assert len({bc}) == 1


class TestOptimizerABC:
    """Optimizer interface contract."""

    def test_cannot_instantiate_abc(self):
        """Optimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Optimizer()  # type: ignore

    @pytest.mark.asyncio
    async def test_default_refine_returns_policy_unchanged(self):
        """The default refine() implementation is identity."""
        bc = MockBackendControl(label="test")
        policy = Policy(backend_controls=(bc,))
        optimizer = MockOptimizer()

        refined = await optimizer.refine(policy, [], [], frozenset())

        assert refined is policy
