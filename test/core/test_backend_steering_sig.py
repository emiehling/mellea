"""Verify Backend ABC accepts steering parameter."""

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core import CBlock
from mellea.stdlib.context import SimpleContext


class TestBackendSteeringSig:
    @pytest.mark.asyncio
    async def test_generate_from_context_accepts_steering_none(self):
        """Existing call pattern (no steering) still works."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        mot, _new_ctx = await backend.generate_from_context(CBlock("test"), ctx)
        assert mot.value == "hello"

    @pytest.mark.asyncio
    async def test_generate_from_context_accepts_steering_kwarg(self):
        """New call pattern (steering=None explicitly) works."""
        backend = DummyBackend(responses=["hello"])
        ctx = SimpleContext()
        mot, _new_ctx = await backend.generate_from_context(
            CBlock("test"), ctx, steering=None
        )
        assert mot.value == "hello"

    def test_steering_capabilities_default_empty(self):
        """Default steering_capabilities returns empty SteeringCapabilities."""
        backend = DummyBackend(responses=None)
        caps = backend.steering_capabilities
        assert caps.supported_control_types == frozenset()
