"""Tests for concrete steering controls."""

from dataclasses import FrozenInstanceError

import pytest

from mellea.core import CBlock
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.controls import (
    FewShot,
    Grounding,
    StopSequence,
    Temperature,
    control_registry,
)
from mellea.steering import BackendControl, InputControl

# ---- Classification Tests ----


class TestControlClassification:
    def test_fewshot_is_input_control(self):
        assert isinstance(FewShot(examples=()), InputControl)

    def test_grounding_is_input_control(self):
        assert isinstance(Grounding(entries=()), InputControl)

    def test_stop_sequence_is_backend_control(self):
        assert isinstance(StopSequence(sequences=()), BackendControl)

    def test_temperature_is_backend_control(self):
        assert isinstance(Temperature(temperature=0.5), BackendControl)


# ---- Frozen / Hashable Tests ----


class TestControlImmutability:
    def test_fewshot_is_frozen(self):
        ctrl = FewShot(examples=("a", "b"))
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctrl.examples = ("c",)

    def test_temperature_is_frozen(self):
        ctrl = Temperature(temperature=0.7)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctrl.temperature = 0.9

    def test_all_controls_hashable(self):
        """All controls can be used as dict keys / in sets."""
        controls = [
            FewShot(examples=("a",)),
            Grounding(entries=(("k", "v"),)),
            StopSequence(sequences=("stop",)),
            Temperature(temperature=0.5),
        ]
        s = set(controls)
        assert len(s) == len(controls)

    def test_equal_controls_have_equal_hashes(self):
        a = Temperature(temperature=0.5)
        b = Temperature(temperature=0.5)
        assert a == b
        assert hash(a) == hash(b)


# ---- Input Control Apply Tests ----


class TestFewShotApply:
    def test_appends_examples_to_instruction(self):
        ctrl = FewShot(examples=("example1", "example2"))
        inst = Instruction(description="Test", icl_examples=["existing"])
        ctx = SimpleContext()

        new_action, new_ctx = ctrl.apply(inst, ctx)

        assert isinstance(new_action, Instruction)
        assert len(new_action._icl_examples) == 3
        assert new_ctx is ctx  # context unchanged

    def test_noop_for_non_instruction(self):
        ctrl = FewShot(examples=("example1",))
        block = CBlock("not an instruction")
        ctx = SimpleContext()

        new_action, new_ctx = ctrl.apply(block, ctx)

        assert new_action is block  # unchanged
        assert new_ctx is ctx

    def test_empty_examples_noop(self):
        ctrl = FewShot(examples=())
        inst = Instruction(description="Test")
        ctx = SimpleContext()

        new_action, _ = ctrl.apply(inst, ctx)
        assert len(new_action._icl_examples) == 0


class TestGroundingApply:
    def test_adds_grounding_to_instruction(self):
        ctrl = Grounding(entries=(("doc1", "content1"),))
        inst = Instruction(description="Test")
        ctx = SimpleContext()

        new_action, new_ctx = ctrl.apply(inst, ctx)

        assert isinstance(new_action, Instruction)
        assert "doc1" in new_action._grounding_context
        assert new_ctx is ctx

    def test_noop_for_non_instruction(self):
        ctrl = Grounding(entries=(("doc", "content"),))
        block = CBlock("not an instruction")
        ctx = SimpleContext()

        new_action, _new_ctx = ctrl.apply(block, ctx)
        assert new_action is block

    def test_raises_on_key_conflict(self):
        ctrl = Grounding(entries=(("existing_key", "new_val"),))
        inst = Instruction(
            description="Test",
            grounding_context={"existing_key": "old_val"},
        )
        ctx = SimpleContext()

        with pytest.raises(KeyError, match="existing_key"):
            ctrl.apply(inst, ctx)


# ---- Backend Control Validation Tests ----


class TestTemperatureValidation:
    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature must be >= 0"):
            Temperature(temperature=-0.1)

    def test_zero_temperature_ok(self):
        ctrl = Temperature(temperature=0.0)
        assert ctrl.temperature == 0.0

    def test_positive_temperature_ok(self):
        ctrl = Temperature(temperature=1.5)
        assert ctrl.temperature == 1.5


# ---- Registry Tests ----


class TestControlRegistry:
    def test_registry_discovers_all_controls(self):
        """control_registry() discovers all control modules."""
        registry = control_registry()

        assert "few_shot" in registry
        assert "grounding" in registry
        assert "temperature" in registry
        assert "stop_sequence" in registry

    def test_registry_has_expected_keys(self):
        """Each entry has kind, domain, summary, composable."""
        registry = control_registry()
        for name, info in registry.items():
            assert "kind" in info, f"{name} missing 'kind'"
            assert "domain" in info, f"{name} missing 'domain'"
            assert "summary" in info, f"{name} missing 'summary'"
            assert "composable" in info, f"{name} missing 'composable'"

    def test_registry_kind_values(self):
        """kind is 'input' or 'backend'."""
        registry = control_registry()
        assert registry["few_shot"]["kind"] == "input"
        assert registry["grounding"]["kind"] == "input"
        assert registry["temperature"]["kind"] == "backend"
        assert registry["stop_sequence"]["kind"] == "backend"

