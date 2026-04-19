"""Unit tests for steering control factory functions."""

from mellea.core.steering import ControlCategory
from mellea.steering.controls import (
    active_output_control,
    input_control,
    state_control,
    static_output_control,
    structural_control,
)


def test_input_control():
    c = input_control("prompt_rewrite")
    assert c.category == ControlCategory.INPUT
    assert c.name == "prompt_rewrite"
    assert c.params == {}
    assert c.artifact_ref is None


def test_input_control_with_params():
    c = input_control("rewrite", params={"style": "concise"}, artifact_ref="ref/1")
    assert c.params["style"] == "concise"
    assert c.artifact_ref == "ref/1"


def test_structural_control():
    c = structural_control(
        "lora_blend", adapter_ref="adapters/v1", model_family="granite"
    )
    assert c.category == ControlCategory.STRUCTURAL
    assert c.name == "lora_blend"
    assert c.artifact_ref == "adapters/v1"
    assert c.model_family == "granite"


def test_state_control():
    c = state_control("honesty_steering", artifact_ref="vectors/v1", layer=10)
    assert c.category == ControlCategory.STATE
    assert c.name == "honesty_steering"
    assert c.artifact_ref == "vectors/v1"
    assert c.params["layer"] == 10


def test_state_control_without_layer():
    c = state_control("base_steering", artifact_ref="vectors/v1")
    assert "layer" not in c.params


def test_static_output_control():
    c = static_output_control("low_temp", temperature=0.2, top_p=0.9)
    assert c.category == ControlCategory.OUTPUT
    assert c.name == "low_temp"
    assert c.params["temperature"] == 0.2
    assert c.params["top_p"] == 0.9
    assert c.artifact_ref is None


def test_active_output_control():
    c = active_output_control(
        "reward_model",
        artifact_ref="models/reward_v1",
        params={"threshold": 0.8},
        model_family="granite",
    )
    assert c.category == ControlCategory.OUTPUT
    assert c.name == "reward_model"
    assert c.artifact_ref == "models/reward_v1"
    assert c.params["threshold"] == 0.8
    assert c.model_family == "granite"
