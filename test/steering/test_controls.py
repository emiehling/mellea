"""Tests for concrete steering controls."""

from dataclasses import FrozenInstanceError

import pytest

from mellea.core import CBlock
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.controls import (
    ActivationSteeringControl,
    AttentionMaskControl,
    FewShotControl,
    GroundingControl,
    LogitBiasControl,
    StopSequenceControl,
    TemperatureOverrideControl,
)
from mellea.steering import InputControl, OutputControl, StateControl

# ---- Classification Tests ----


class TestControlClassification:
    def test_fewshot_is_input_control(self):
        assert isinstance(FewShotControl(examples=()), InputControl)

    def test_grounding_is_input_control(self):
        assert isinstance(GroundingControl(entries=()), InputControl)

    def test_activation_steering_is_state_control(self):
        ctrl = ActivationSteeringControl(vector_id="v1", layers=(0, 1))
        assert isinstance(ctrl, StateControl)

    def test_attention_mask_is_state_control(self):
        assert isinstance(AttentionMaskControl(mask_type="block"), StateControl)

    def test_logit_bias_is_output_control(self):
        assert isinstance(LogitBiasControl(biases=()), OutputControl)

    def test_stop_sequence_is_output_control(self):
        assert isinstance(StopSequenceControl(sequences=()), OutputControl)

    def test_temperature_is_output_control(self):
        assert isinstance(TemperatureOverrideControl(temperature=0.5), OutputControl)


# ---- Frozen / Hashable Tests ----


class TestControlImmutability:
    def test_fewshot_is_frozen(self):
        ctrl = FewShotControl(examples=("a", "b"))
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctrl.examples = ("c",)

    def test_activation_steering_is_frozen(self):
        ctrl = ActivationSteeringControl(vector_id="v", layers=(0,))
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctrl.vector_id = "other"

    def test_temperature_is_frozen(self):
        ctrl = TemperatureOverrideControl(temperature=0.7)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctrl.temperature = 0.9

    def test_all_controls_hashable(self):
        """All controls can be used as dict keys / in sets."""
        controls = [
            FewShotControl(examples=("a",)),
            GroundingControl(entries=(("k", "v"),)),
            ActivationSteeringControl(vector_id="v", layers=(0,)),
            AttentionMaskControl(mask_type="block"),
            LogitBiasControl(biases=((1, 0.5),)),
            StopSequenceControl(sequences=("stop",)),
            TemperatureOverrideControl(temperature=0.5),
        ]
        s = set(controls)
        assert len(s) == len(controls)

    def test_equal_controls_have_equal_hashes(self):
        a = TemperatureOverrideControl(temperature=0.5)
        b = TemperatureOverrideControl(temperature=0.5)
        assert a == b
        assert hash(a) == hash(b)


# ---- Input Control Apply Tests ----


class TestFewShotControlApply:
    def test_appends_examples_to_instruction(self):
        ctrl = FewShotControl(examples=("example1", "example2"))
        inst = Instruction(description="Test", icl_examples=["existing"])
        ctx = SimpleContext()

        new_action, new_ctx = ctrl.apply(inst, ctx)

        assert isinstance(new_action, Instruction)
        assert len(new_action._icl_examples) == 3
        assert new_ctx is ctx  # context unchanged

    def test_noop_for_non_instruction(self):
        ctrl = FewShotControl(examples=("example1",))
        block = CBlock("not an instruction")
        ctx = SimpleContext()

        new_action, new_ctx = ctrl.apply(block, ctx)

        assert new_action is block  # unchanged
        assert new_ctx is ctx

    def test_empty_examples_noop(self):
        ctrl = FewShotControl(examples=())
        inst = Instruction(description="Test")
        ctx = SimpleContext()

        new_action, _ = ctrl.apply(inst, ctx)
        assert len(new_action._icl_examples) == 0


class TestGroundingControlApply:
    def test_adds_grounding_to_instruction(self):
        ctrl = GroundingControl(entries=(("doc1", "content1"),))
        inst = Instruction(description="Test")
        ctx = SimpleContext()

        new_action, new_ctx = ctrl.apply(inst, ctx)

        assert isinstance(new_action, Instruction)
        assert "doc1" in new_action._grounding_context
        assert new_ctx is ctx

    def test_noop_for_non_instruction(self):
        ctrl = GroundingControl(entries=(("doc", "content"),))
        block = CBlock("not an instruction")
        ctx = SimpleContext()

        new_action, _new_ctx = ctrl.apply(block, ctx)
        assert new_action is block

    def test_raises_on_key_conflict(self):
        ctrl = GroundingControl(entries=(("existing_key", "new_val"),))
        inst = Instruction(
            description="Test",
            grounding_context={"existing_key": "old_val"},
        )
        ctx = SimpleContext()

        with pytest.raises(KeyError, match="existing_key"):
            ctrl.apply(inst, ctx)


# ---- Output Control Validation Tests ----


class TestTemperatureOverrideValidation:
    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature must be >= 0"):
            TemperatureOverrideControl(temperature=-0.1)

    def test_zero_temperature_ok(self):
        ctrl = TemperatureOverrideControl(temperature=0.0)
        assert ctrl.temperature == 0.0

    def test_positive_temperature_ok(self):
        ctrl = TemperatureOverrideControl(temperature=1.5)
        assert ctrl.temperature == 1.5
