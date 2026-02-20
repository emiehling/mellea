"""Tests for Instruction builder methods."""

import pytest

from mellea.core import CBlock
from mellea.stdlib.components.instruction import Instruction


class TestWithAdditionalExamples:
    def test_appends_string_examples(self):
        """String examples are appended and blockified."""
        inst = Instruction(
            description="Translate to French", icl_examples=["Hello -> Bonjour"]
        )
        inst2 = inst.with_additional_examples(["Goodbye -> Au revoir"])

        # original unchanged
        assert len(inst._icl_examples) == 1
        # new has both
        assert len(inst2._icl_examples) == 2
        assert str(inst2._icl_examples[0]) == "Hello -> Bonjour"
        assert str(inst2._icl_examples[1]) == "Goodbye -> Au revoir"

    def test_appends_cblock_examples(self):
        """CBlock examples are appended as-is."""
        inst = Instruction(description="Test")
        block = CBlock("example block")
        inst2 = inst.with_additional_examples([block])
        assert len(inst2._icl_examples) == 1

    def test_empty_list_is_noop(self):
        """Passing empty list returns copy with same examples."""
        inst = Instruction(description="Test", icl_examples=["A"])
        inst2 = inst.with_additional_examples([])
        assert len(inst2._icl_examples) == 1

    def test_does_not_mutate_original(self):
        """Original instruction is not modified."""
        inst = Instruction(description="Test", icl_examples=["A"])
        original_examples = list(inst._icl_examples)
        _ = inst.with_additional_examples(["B", "C"])
        assert inst._icl_examples == original_examples

    def test_chaining(self):
        """Multiple calls can be chained."""
        inst = (
            Instruction(description="Test")
            .with_additional_examples(["A"])
            .with_additional_examples(["B"])
        )
        assert len(inst._icl_examples) == 2

    def test_preserves_other_fields(self):
        """Other instruction fields are preserved."""
        inst = Instruction(
            description="Desc",
            icl_examples=["A"],
            grounding_context={"doc": "content"},
            prefix="Prefix",
        )
        inst2 = inst.with_additional_examples(["B"])
        assert str(inst2._description) == "Desc"
        assert "doc" in inst2._grounding_context
        assert str(inst2._prefix) == "Prefix"


class TestWithAdditionalGrounding:
    def test_adds_string_entries(self):
        """String values are blockified and added."""
        inst = Instruction(description="Summarize")
        inst2 = inst.with_additional_grounding({"doc1": "Content of doc1"})
        assert "doc1" in inst2._grounding_context
        assert str(inst2._grounding_context["doc1"]) == "Content of doc1"

    def test_adds_to_existing_grounding(self):
        """New keys are added alongside existing ones."""
        inst = Instruction(
            description="Summarize", grounding_context={"existing": "data"}
        )
        inst2 = inst.with_additional_grounding({"new": "more data"})
        assert "existing" in inst2._grounding_context
        assert "new" in inst2._grounding_context

    def test_raises_on_key_conflict(self):
        """Raises KeyError if a key already exists."""
        inst = Instruction(description="Test", grounding_context={"doc": "original"})
        with pytest.raises(KeyError, match="doc"):
            inst.with_additional_grounding({"doc": "replacement"})

    def test_empty_dict_is_noop(self):
        """Passing empty dict returns copy with same grounding."""
        inst = Instruction(description="Test", grounding_context={"a": "b"})
        inst2 = inst.with_additional_grounding({})
        assert "a" in inst2._grounding_context

    def test_does_not_mutate_original(self):
        """Original instruction is not modified."""
        inst = Instruction(description="Test")
        _ = inst.with_additional_grounding({"key": "value"})
        assert "key" not in inst._grounding_context

    def test_chaining(self):
        """Multiple calls can be chained with different keys."""
        inst = (
            Instruction(description="Test")
            .with_additional_grounding({"a": "1"})
            .with_additional_grounding({"b": "2"})
        )
        assert "a" in inst._grounding_context
        assert "b" in inst._grounding_context

    def test_preserves_other_fields(self):
        """Other instruction fields are preserved."""
        inst = Instruction(description="Desc", icl_examples=["ex"], prefix="Prefix")
        inst2 = inst.with_additional_grounding({"doc": "content"})
        assert str(inst2._description) == "Desc"
        assert len(inst2._icl_examples) == 1
        assert str(inst2._prefix) == "Prefix"

    def test_multiple_entries_at_once(self):
        """Multiple entries can be added in a single call."""
        inst = Instruction(description="Test")
        inst2 = inst.with_additional_grounding({"doc1": "content1", "doc2": "content2"})
        assert "doc1" in inst2._grounding_context
        assert "doc2" in inst2._grounding_context


class TestBuilderComposition:
    """Builders compose with each other and with copy_and_repair."""

    def test_examples_then_grounding(self):
        inst = (
            Instruction(description="Test")
            .with_additional_examples(["ex1"])
            .with_additional_grounding({"doc": "content"})
        )
        assert len(inst._icl_examples) == 1
        assert "doc" in inst._grounding_context

    def test_grounding_then_repair(self):
        inst = (
            Instruction(description="Test")
            .with_additional_grounding({"doc": "content"})
            .copy_and_repair("Fix the output")
        )
        assert "doc" in inst._grounding_context
        assert inst._repair_string == "Fix the output"

    def test_examples_survive_repair(self):
        """Additional examples persist through copy_and_repair."""
        inst = (
            Instruction(description="Test")
            .with_additional_examples(["ex1", "ex2"])
            .copy_and_repair("Try again")
        )
        assert len(inst._icl_examples) == 2
        assert inst._repair_string == "Try again"
