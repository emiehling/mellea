"""Unit tests for HuggingFace-specific steering control handlers.

Uses mock models to test handler behavior without requiring GPU resources.
"""

from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.hf_steering_handlers import (
    ActivationSteeringHandler,
    AdapterControlHandler,
    RewardGuidedDecodingHandler,
    StaticOutputControlHandler,
    _get_model_layers,
)
from mellea.core.steering import Control, ControlCategory


# --- ActivationSteeringHandler ---


def test_activation_steering_registers_hooks():
    torch = pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layers": [0, 1], "coefficient": 1.5},
    )

    # Create a mock model with 3 layers.
    mock_layers = [MagicMock() for _ in range(3)]
    mock_model = MagicMock()
    mock_model.model.layers = mock_layers

    # Each layer's register_forward_hook returns a removable hook.
    for layer in mock_layers:
        layer.register_forward_hook.return_value = MagicMock()

    steering_vector = torch.zeros(128)
    hooks = handler.activate(control, mock_model, steering_vector)

    # Should register hooks on layers 0 and 1 only.
    assert len(hooks) == 2
    mock_layers[0].register_forward_hook.assert_called_once()
    mock_layers[1].register_forward_hook.assert_called_once()
    mock_layers[2].register_forward_hook.assert_not_called()


def test_activation_steering_deactivate_removes_hooks():
    handler = ActivationSteeringHandler()
    mock_hooks = [MagicMock(), MagicMock()]
    handler.deactivate(mock_hooks)
    for h in mock_hooks:
        h.remove.assert_called_once()


def test_activation_steering_default_all_layers():
    torch = pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={},  # No layers specified → all layers.
    )

    mock_layers = [MagicMock() for _ in range(4)]
    mock_model = MagicMock()
    mock_model.model.layers = mock_layers
    for layer in mock_layers:
        layer.register_forward_hook.return_value = MagicMock()

    hooks = handler.activate(control, mock_model, torch.zeros(64))
    assert len(hooks) == 4


# --- AdapterControlHandler ---


def test_adapter_control_loads_and_sets():
    handler = AdapterControlHandler()
    control = Control(
        category=ControlCategory.STRUCTURAL,
        name="my_adapter",
        params={"adapter_name": "lora_v1"},
    )
    mock_model = MagicMock()
    handle = handler.activate(control, mock_model, "path/to/adapter")

    assert handle == "lora_v1"
    mock_model.load_adapter.assert_called_once_with(
        "path/to/adapter", adapter_name="lora_v1"
    )
    mock_model.set_adapter.assert_called_once_with("lora_v1")


def test_adapter_control_default_name():
    handler = AdapterControlHandler()
    control = Control(
        category=ControlCategory.STRUCTURAL, name="fallback_adapter", params={}
    )
    mock_model = MagicMock()
    handle = handler.activate(control, mock_model, "path/to/adapter")
    assert handle == "fallback_adapter"


def test_adapter_control_deactivate_is_noop():
    handler = AdapterControlHandler()
    handler.deactivate("lora_v1")  # Should not raise.


# --- StaticOutputControlHandler ---


def test_static_output_merges_params():
    handler = StaticOutputControlHandler()
    control = Control(
        category=ControlCategory.OUTPUT,
        name="static_output",
        params={"temperature": 0.3, "top_p": 0.9},
    )
    gen_kwargs = {"max_new_tokens": 100}
    result = handler.apply(control, gen_kwargs, None)
    assert result["temperature"] == 0.3
    assert result["top_p"] == 0.9
    assert result["max_new_tokens"] == 100


# --- RewardGuidedDecodingHandler ---


def test_reward_guided_adds_logits_processor():
    torch = pytest.importorskip("torch")

    handler = RewardGuidedDecodingHandler()
    mock_reward = MagicMock(return_value=torch.zeros(10))
    control = Control(
        category=ControlCategory.OUTPUT,
        name="reward_guided_decoding",
        params={"temperature": 2.0},
    )
    gen_kwargs = {}
    result = handler.apply(control, gen_kwargs, mock_reward)
    assert "logits_processor" in result
    assert len(result["logits_processor"]) == 1


def test_reward_guided_appends_to_existing_processors():
    torch = pytest.importorskip("torch")

    handler = RewardGuidedDecodingHandler()
    mock_reward = MagicMock(return_value=torch.zeros(10))
    control = Control(
        category=ControlCategory.OUTPUT, name="reward_guided_decoding", params={}
    )
    existing_processor = MagicMock()
    gen_kwargs = {"logits_processor": [existing_processor]}
    result = handler.apply(control, gen_kwargs, mock_reward)
    assert len(result["logits_processor"]) == 2
    # Original list should not be mutated.
    assert len(gen_kwargs["logits_processor"]) == 1


# --- _get_model_layers ---


def test_get_model_layers_llama_style():
    mock_model = MagicMock()
    mock_model.model.layers = ["layer0", "layer1"]
    assert _get_model_layers(mock_model) == ["layer0", "layer1"]


def test_get_model_layers_gpt2_style():
    mock_model = MagicMock(spec=[])
    mock_model.transformer = MagicMock()
    mock_model.transformer.h = ["h0", "h1"]
    # Ensure model.model doesn't exist (gpt2 style).
    assert not hasattr(mock_model, "model")
    assert _get_model_layers(mock_model) == ["h0", "h1"]


def test_get_model_layers_unknown_raises():
    mock_model = MagicMock(spec=[])
    with pytest.raises(AttributeError, match="Cannot resolve transformer layers"):
        _get_model_layers(mock_model)
