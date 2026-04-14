"""Unit tests ensuring every backend returns a valid BackendCapabilities."""

from mellea.backends.dummy import DummyBackend
from mellea.core.steering import BackendCapabilities, ControlCategory


def test_dummy_backend_capabilities():
    backend = DummyBackend(responses=None)
    caps = backend.capabilities
    assert isinstance(caps, BackendCapabilities)
    assert caps.supported_categories == frozenset()
    assert caps.supports_logits_processors is False
    assert caps.supports_adapter_loading is False
    assert caps.supports_forward_hooks is False
