"""Unit tests for HuggingFace-specific steering control handlers.

Uses mock models to test handler behavior without requiring GPU resources.
"""

from unittest.mock import MagicMock

import pytest

from mellea.backends.hf_steering_handlers import (
    ActivationSteeringHandler,
    AdapterHandler,
    LogitsProcessorHandler,
    StoppingCriteriaHandler,
    get_model_layers,
)
from mellea.core.steering import Control, ControlCategory

# --- ActivationSteeringHandler ---


def _mock_model(num_layers: int) -> MagicMock:
    """Create a mock model with *num_layers* layers, each accepting hooks."""
    mock_layers = [MagicMock() for _ in range(num_layers)]
    mock_model = MagicMock()
    mock_model.model.layers = mock_layers
    for layer in mock_layers:
        layer.register_forward_hook.return_value = MagicMock()
    return mock_model


def _directions(layer_indices: list[int], dim: int = 64):
    """Build a ``dict[int, Tensor]`` with distinct vectors per layer."""
    torch = pytest.importorskip("torch")
    return {i: torch.ones(dim) * (i + 1) for i in layer_indices}


def test_activation_steering_registers_hooks():
    pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layers": [0, 1], "coefficient": 1.5},
    )

    mock_model = _mock_model(3)
    directions = _directions([0, 1, 2])
    hooks = handler.activate(control, mock_model, directions)

    assert len(hooks) == 2
    mock_model.model.layers[0].register_forward_hook.assert_called_once()
    mock_model.model.layers[1].register_forward_hook.assert_called_once()
    mock_model.model.layers[2].register_forward_hook.assert_not_called()


def test_activation_steering_deactivate_removes_hooks():
    handler = ActivationSteeringHandler()
    mock_hooks = [MagicMock(), MagicMock()]
    handler.deactivate(mock_hooks)
    for h in mock_hooks:
        h.remove.assert_called_once()


def test_activation_steering_default_all_layers():
    pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={},
    )

    mock_model = _mock_model(4)
    directions = _directions([0, 1, 2, 3])
    hooks = handler.activate(control, mock_model, directions)
    assert len(hooks) == 4


def test_activation_steering_per_layer_uses_correct_vectors():
    """Each layer's hook receives that layer's specific direction vector."""
    torch = pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layers": [0, 2], "coefficient": 1.0},
    )

    mock_model = _mock_model(3)
    directions = _directions([0, 1, 2], dim=8)
    handler.activate(control, mock_model, directions)

    # Extract the hook functions that were registered.
    hook_0 = mock_model.model.layers[0].register_forward_hook.call_args[0][0]
    hook_2 = mock_model.model.layers[2].register_forward_hook.call_args[0][0]

    hidden = torch.zeros(1, 1, 8)
    out_0 = hook_0(None, None, (hidden.clone(),))[0]
    out_2 = hook_2(None, None, (hidden.clone(),))[0]

    # Layer 0 vector is ones*1, layer 2 vector is ones*3.
    assert torch.allclose(out_0, torch.ones(1, 1, 8) * 1.0)
    assert torch.allclose(out_2, torch.ones(1, 1, 8) * 3.0)


def test_activation_steering_missing_layer_raises():
    """Requesting a layer not in the directions dict raises KeyError."""
    torch = pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layers": [5]},
    )

    mock_model = _mock_model(10)
    directions = {0: torch.zeros(64), 1: torch.zeros(64)}

    with pytest.raises(KeyError):
        handler.activate(control, mock_model, directions)


def test_activation_steering_rejects_single_tensor():
    """A single tensor (not a dict) is rejected."""
    torch = pytest.importorskip("torch")

    handler = ActivationSteeringHandler()
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layers": [0]},
    )

    mock_model = _mock_model(3)

    with pytest.raises(AssertionError, match="dict"):
        handler.activate(control, mock_model, torch.zeros(64))


# --- AdapterHandler ---


def test_adapter_loads_and_sets():
    handler = AdapterHandler()
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


def test_adapter_default_name():
    handler = AdapterHandler()
    control = Control(
        category=ControlCategory.STRUCTURAL, name="fallback_adapter", params={}
    )
    mock_model = MagicMock()
    handle = handler.activate(control, mock_model, "path/to/adapter")
    assert handle == "fallback_adapter"


def test_adapter_deactivate_is_noop():
    handler = AdapterHandler()
    handler.deactivate("lora_v1")  # Should not raise.


# --- LogitsProcessorHandler ---


def test_logits_processor_handler_appends():
    handler = LogitsProcessorHandler()
    mock_processor = MagicMock()
    control = Control(
        category=ControlCategory.OUTPUT,
        name="logits_processor",
        params={},
    )
    gen_kwargs: dict = {}
    result = handler.apply(control, gen_kwargs, mock_processor)
    assert "logits_processor" in result
    assert len(result["logits_processor"]) == 1
    assert result["logits_processor"][0] is mock_processor


def test_logits_processor_handler_appends_to_existing():
    handler = LogitsProcessorHandler()
    mock_processor = MagicMock()
    control = Control(
        category=ControlCategory.OUTPUT, name="logits_processor", params={}
    )
    existing_processor = MagicMock()
    original_list = [existing_processor]
    gen_kwargs = {"logits_processor": original_list}
    result = handler.apply(control, gen_kwargs, mock_processor)
    assert len(result["logits_processor"]) == 2
    assert result["logits_processor"][0] is existing_processor
    assert result["logits_processor"][1] is mock_processor
    # Original list should not be mutated.
    assert len(original_list) == 1


def test_logits_processor_handler_prepend():
    handler = LogitsProcessorHandler()
    mock_processor = MagicMock()
    control = Control(
        category=ControlCategory.OUTPUT,
        name="logits_processor",
        params={"priority": "prepend"},
    )
    existing_processor = MagicMock()
    gen_kwargs = {"logits_processor": [existing_processor]}
    result = handler.apply(control, gen_kwargs, mock_processor)
    assert len(result["logits_processor"]) == 2
    assert result["logits_processor"][0] is mock_processor
    assert result["logits_processor"][1] is existing_processor


def test_logits_processor_handler_requires_artifact():
    handler = LogitsProcessorHandler()
    control = Control(
        category=ControlCategory.OUTPUT, name="logits_processor", params={}
    )
    with pytest.raises(AssertionError, match="requires a LogitsProcessor artifact"):
        handler.apply(control, {}, None)


# --- StoppingCriteriaHandler ---


def test_stopping_criteria_handler_appends():
    handler = StoppingCriteriaHandler()
    mock_criteria = MagicMock()
    control = Control(
        category=ControlCategory.OUTPUT,
        name="stopping_criteria",
        params={},
    )
    gen_kwargs: dict = {}
    result = handler.apply(control, gen_kwargs, mock_criteria)
    assert "stopping_criteria" in result
    assert len(result["stopping_criteria"]) == 1
    assert result["stopping_criteria"][0] is mock_criteria


def test_stopping_criteria_handler_prepend():
    handler = StoppingCriteriaHandler()
    mock_criteria = MagicMock()
    control = Control(
        category=ControlCategory.OUTPUT,
        name="stopping_criteria",
        params={"priority": "prepend"},
    )
    existing_criteria = MagicMock()
    gen_kwargs = {"stopping_criteria": [existing_criteria]}
    result = handler.apply(control, gen_kwargs, mock_criteria)
    assert len(result["stopping_criteria"]) == 2
    assert result["stopping_criteria"][0] is mock_criteria
    assert result["stopping_criteria"][1] is existing_criteria


def test_stopping_criteria_handler_requires_artifact():
    handler = StoppingCriteriaHandler()
    control = Control(
        category=ControlCategory.OUTPUT, name="stopping_criteria", params={}
    )
    with pytest.raises(AssertionError, match="requires a StoppingCriteria artifact"):
        handler.apply(control, {}, None)


# --- get_model_layers() ---


def test_get_model_layers_llama_style():
    mock_model = MagicMock()
    mock_model.model.layers = ["layer0", "layer1"]
    assert get_model_layers(mock_model) == ["layer0", "layer1"]


def test_get_model_layers_gpt2_style():
    mock_model = MagicMock(spec=[])
    mock_model.transformer = MagicMock()
    mock_model.transformer.h = ["h0", "h1"]
    # Ensure model.model doesn't exist (gpt2 style).
    assert not hasattr(mock_model, "model")
    assert get_model_layers(mock_model) == ["h0", "h1"]


def test_get_model_layers_unknown_raises():
    mock_model = MagicMock(spec=[])
    with pytest.raises(AttributeError, match="Cannot resolve transformer layers"):
        get_model_layers(mock_model)
