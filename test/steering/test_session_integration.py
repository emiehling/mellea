"""Tests for steering integration in session.py."""

from __future__ import annotations

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import Component, Context, Requirement
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.session import MelleaSession, start_session
from mellea.steering import SteeringCapabilities, SteeringOptimizer, SteeringPolicy


class MockOptimizer(SteeringOptimizer):
    """Mock optimizer for testing."""

    def __init__(self):
        self.compile_call_count = 0
        self.id = id(self)  # unique ID for tracking

    async def compile(
        self,
        requirements: list[Requirement],
        capabilities: SteeringCapabilities,
        ctx: Context | None = None,
        action: Component | None = None,
    ) -> SteeringPolicy:
        """Return an empty policy."""
        self.compile_call_count += 1
        return SteeringPolicy()


class TestSessionOptimizer:
    """Tests for optimizer in MelleaSession."""

    def test_session_stores_optimizer(self):
        """MelleaSession constructor stores optimizer."""
        opt = MockOptimizer()
        m = MelleaSession(
            backend=DummyBackend(responses=None),
            optimizer=opt,
        )
        assert m.optimizer is opt

    def test_session_optimizer_defaults_none(self):
        """MelleaSession optimizer defaults to None."""
        m = MelleaSession(backend=DummyBackend(responses=None))
        assert m.optimizer is None

    def test_clone_preserves_optimizer(self):
        """clone() preserves the session optimizer."""
        opt = MockOptimizer()
        m = MelleaSession(
            backend=DummyBackend(responses=None),
            optimizer=opt,
        )
        m2 = m.clone()
        assert m2.optimizer is opt
        assert m2.optimizer.id == opt.id

    def test_copy_preserves_optimizer(self):
        """__copy__ preserves the session optimizer."""
        from copy import copy

        opt = MockOptimizer()
        m = MelleaSession(
            backend=DummyBackend(responses=None),
            optimizer=opt,
        )
        m2 = copy(m)
        assert m2.optimizer is opt


class TestSessionOptimizerOverride:
    """Tests for per-call optimizer override."""

    def test_per_call_optimizer_overrides_session(self):
        """Per-call optimizer overrides session default."""
        session_opt = MockOptimizer()
        call_opt = MockOptimizer()
        reqs = [Requirement("test")]

        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=session_opt,
        )

        # call with per-call optimizer
        m.instruct("Test", requirements=reqs, strategy=None, optimizer=call_opt)

        # only call_opt should have been used
        assert call_opt.compile_call_count == 1
        assert session_opt.compile_call_count == 0

    def test_session_optimizer_used_when_per_call_none(self):
        """Session optimizer is used when per-call optimizer is None."""
        session_opt = MockOptimizer()
        reqs = [Requirement("test")]

        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=session_opt,
        )

        # call without per-call optimizer
        m.instruct("Test", requirements=reqs, strategy=None, optimizer=None)

        # session_opt should have been used
        assert session_opt.compile_call_count == 1

    def test_no_optimizer_when_both_none(self):
        """No optimizer is used when both session and per-call are None."""
        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=None,
        )

        # should not raise, just run without steering
        result = m.instruct(
            "Test",
            requirements=[Requirement("test")],
            strategy=None,
            optimizer=None,
        )
        assert result.value == "hello"


class TestSessionAsyncMethods:
    """Tests for async session methods with optimizer."""

    @pytest.mark.asyncio
    async def test_aact_uses_session_optimizer(self):
        """aact() uses session optimizer by default."""
        session_opt = MockOptimizer()
        reqs = [Requirement("test")]

        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=session_opt,
        )

        from mellea.stdlib.components.instruction import Instruction

        await m.aact(Instruction(description="Test"), requirements=reqs, strategy=None)

        assert session_opt.compile_call_count == 1

    @pytest.mark.asyncio
    async def test_aact_per_call_override(self):
        """aact() per-call optimizer overrides session."""
        session_opt = MockOptimizer()
        call_opt = MockOptimizer()
        reqs = [Requirement("test")]

        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=session_opt,
        )

        from mellea.stdlib.components.instruction import Instruction

        await m.aact(
            Instruction(description="Test"),
            requirements=reqs,
            strategy=None,
            optimizer=call_opt,
        )

        assert call_opt.compile_call_count == 1
        assert session_opt.compile_call_count == 0

    @pytest.mark.asyncio
    async def test_ainstruct_uses_session_optimizer(self):
        """ainstruct() uses session optimizer by default."""
        session_opt = MockOptimizer()
        reqs = [Requirement("test")]

        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=session_opt,
        )

        await m.ainstruct("Test", requirements=reqs, strategy=None)

        assert session_opt.compile_call_count == 1

    @pytest.mark.asyncio
    async def test_ainstruct_per_call_override(self):
        """ainstruct() per-call optimizer overrides session."""
        session_opt = MockOptimizer()
        call_opt = MockOptimizer()
        reqs = [Requirement("test")]

        m = MelleaSession(
            backend=DummyBackend(responses=["hello"]),
            optimizer=session_opt,
        )

        await m.ainstruct("Test", requirements=reqs, strategy=None, optimizer=call_opt)

        assert call_opt.compile_call_count == 1
        assert session_opt.compile_call_count == 0
