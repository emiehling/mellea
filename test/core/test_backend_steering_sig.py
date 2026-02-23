"""Verify Backend ABC accepts policy parameter."""

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import CBlock
from mellea.stdlib.context import SimpleContext


class TestBackendPolicySig:
    @pytest.mark.asyncio
    async def test_generate_from_context_accepts_policy_none(self):
        """Existing call pattern (no policy) still works."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        mot, _new_ctx = await backend.generate_from_context(CBlock("test"), ctx)
        assert mot.value == "hello"

    @pytest.mark.asyncio
    async def test_generate_from_context_accepts_policy_kwarg(self):
        """New call pattern (policy=None explicitly) works."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        mot, _new_ctx = await backend.generate_from_context(
            CBlock("test"), ctx, policy=None
        )
        assert mot.value == "hello"

    def test_supported_controls_default_empty(self):
        """Default supported_controls returns empty frozenset."""
        backend = DummyBackend(responses=None)
        assert backend.supported_controls == frozenset()
