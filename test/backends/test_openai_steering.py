"""Tests for the OpenAI backend's remote state-steering wiring.

Verifies:
  - capabilities omits STATE when ``enable_remote_state_steering`` is False (default)
  - capabilities includes STATE when the flag is True
  - the activation_steering handler is registered only when the flag is True
  - end-to-end: a state_control flows through compose -> attach -> request body
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.core.steering import Control, ControlCategory, SteeringPolicy


def _torch():
    return pytest.importorskip("torch")


def _zstd():
    return pytest.importorskip("zstandard")


# --- Capabilities: gating on the flag ---


def test_capabilities_default_no_state():
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(model_id="gpt-4", api_key="test", base_url="http://x/v1")
    caps = backend.capabilities
    assert ControlCategory.INPUT in caps.supported_categories
    assert ControlCategory.OUTPUT in caps.supported_categories
    assert ControlCategory.STATE not in caps.supported_categories
    assert caps.extra.get("remote_state_steering") is False


def test_capabilities_flag_enables_state():
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(
        model_id="gpt-4",
        api_key="test",
        base_url="http://x/v1",
        enable_remote_state_steering=True,
    )
    caps = backend.capabilities
    assert ControlCategory.STATE in caps.supported_categories
    assert caps.extra.get("remote_state_steering") is True


# --- Handler registration ---


def test_handler_registered_only_with_flag():
    from mellea.backends.openai import OpenAIBackend
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    off = OpenAIBackend(model_id="gpt-4", api_key="test", base_url="http://x/v1")
    assert "activation_steering" not in off._handler_registry

    on = OpenAIBackend(
        model_id="gpt-4",
        api_key="test",
        base_url="http://x/v1",
        enable_remote_state_steering=True,
    )
    assert isinstance(
        on._handler_registry["activation_steering"], VLLMSteeringRequestHandler
    )


# --- End-to-end mock test ---


def _state_library_with_vector(layer: int, dim: int = 64):
    """Build an ArtifactLibrary whose STATE store returns a single layer vector."""
    torch = _torch()
    from mellea.steering.library import ArtifactLibrary
    from mellea.steering.stores.base import ArtifactStore

    class _Store(ArtifactStore):
        def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
            return {layer: torch.ones(dim)}, {}

        def search(self, query, model=None, max_results=None):
            return []

        def list_artifacts(self, **partial_selectors):
            return []

    return ArtifactLibrary({ControlCategory.STATE: _Store()})


@pytest.mark.asyncio
async def test_state_control_flows_into_request_body(monkeypatch):
    """A state_control attached to the OpenAI backend should produce
    apply_steering_vectors in the outgoing chat.completions.create call.
    """
    _torch()
    _zstd()

    from mellea.backends.openai import OpenAIBackend
    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext
    from mellea.steering.library import set_default_library

    library = _state_library_with_vector(layer=15, dim=8)
    set_default_library(library)
    try:
        backend = OpenAIBackend(
            model_id="granite-4-micro",
            api_key="test",
            base_url="http://x/v1",
            enable_remote_state_steering=True,
        )
        # Attach a state_control that will resolve to the mock vector.
        policy = SteeringPolicy(
            controls=(
                Control(
                    category=ControlCategory.STATE,
                    name="activation_steering",
                    artifact_ref="dummy",
                    params={"layer": 15, "multiplier": 1.5},
                ),
            )
        )
        backend.attach(policy)

        # Mock the API response so generation doesn't actually fire.
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].message.role = "assistant"

        ctx = ChatContext()
        with patch.object(
            backend._async_client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await backend.generate_from_chat_context(CBlock(value="hi"), ctx)

            kwargs = mock_create.call_args.kwargs
            assert "extra_body" in kwargs, (
                "Steering should populate extra_body on the request"
            )
            extra_args = kwargs["extra_body"]["extra_args"]
            sv_list = extra_args["apply_steering_vectors"]
            assert len(sv_list) == 1
            sv = sv_list[0]
            assert sv["layer_indices"] == [15]
            assert sv["scale"] == 1.5
            assert sv["activations"]["shape"] == [1, 8]
            assert sv["activations"]["compression"] == "zstd"
    finally:
        set_default_library(None)


@pytest.mark.asyncio
async def test_no_state_control_means_no_steering_payload(monkeypatch):
    """Without a STATE control, the create call should not carry an
    apply_steering_vectors entry.
    """
    from mellea.backends.openai import OpenAIBackend
    from mellea.core.base import CBlock
    from mellea.stdlib.context import ChatContext

    backend = OpenAIBackend(
        model_id="granite-4-micro",
        api_key="test",
        base_url="http://x/v1",
        enable_remote_state_steering=True,
    )
    backend.attach(SteeringPolicy.empty())

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.choices[0].message.role = "assistant"

    ctx = ChatContext()
    with patch.object(
        backend._async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        await backend.generate_from_chat_context(CBlock(value="hi"), ctx)

        kwargs = mock_create.call_args.kwargs
        extra_body = kwargs.get("extra_body", {})
        extra_args = extra_body.get("extra_args", {}) if extra_body else {}
        assert "apply_steering_vectors" not in extra_args
