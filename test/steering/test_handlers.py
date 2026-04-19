"""Unit tests for built-in input control handlers."""

from mellea.core.base import CBlock
from mellea.core.steering import Control, ControlCategory, get_global_input_handler
from mellea.stdlib.components.chat import Message
from mellea.stdlib.components.instruction import Instruction
from mellea.steering.handlers import (
    ContextPrefixHandler,
    InstructionRewriteHandler,
    SystemPromptInjectionHandler,
)

# --- SystemPromptInjectionHandler ---


def test_system_prompt_injection_adds_message():
    handler = SystemPromptInjectionHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="system_prompt_injection",
        params={"system_prompt": "You are a helpful assistant."},
    )
    action = CBlock("test action")

    new_action, new_ctx = handler.apply(control, action, [], None)

    assert new_action is action
    assert len(new_ctx) == 1
    assert isinstance(new_ctx[0], Message)
    assert new_ctx[0].role == "system"
    assert new_ctx[0].content == "You are a helpful assistant."


def test_system_prompt_injection_prepends_to_existing():
    handler = SystemPromptInjectionHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="system_prompt_injection",
        params={"system_prompt": "Be concise."},
    )
    existing = [Message(role="user", content="Hello")]

    _, new_ctx = handler.apply(control, CBlock("action"), existing, None)

    assert len(new_ctx) == 2
    assert new_ctx[0].role == "system"
    assert new_ctx[1].role == "user"


def test_system_prompt_injection_registered_globally():
    handler = get_global_input_handler("system_prompt_injection")
    assert handler is not None
    assert isinstance(handler, SystemPromptInjectionHandler)


# --- InstructionRewriteHandler ---


def test_instruction_rewrite_modifies_description():
    handler = InstructionRewriteHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="instruction_rewrite",
        params={"template": "Be concise. {original}"},
    )
    instruction = Instruction(description="Write a story.")
    ctx_list: list = []

    new_action, new_ctx = handler.apply(control, instruction, ctx_list, None)

    assert isinstance(new_action, Instruction)
    assert "Be concise." in str(new_action._description)
    assert "Write a story." in str(new_action._description)
    assert new_ctx is ctx_list


def test_instruction_rewrite_leaves_non_instruction_unchanged():
    handler = InstructionRewriteHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="instruction_rewrite",
        params={"template": "Be concise. {original}"},
    )
    action = CBlock("not an instruction")
    ctx_list: list = []

    new_action, new_ctx = handler.apply(control, action, ctx_list, None)

    assert new_action is action
    assert new_ctx is ctx_list


def test_instruction_rewrite_registered_globally():
    handler = get_global_input_handler("instruction_rewrite")
    assert handler is not None
    assert isinstance(handler, InstructionRewriteHandler)


# --- ContextPrefixHandler ---


def test_context_prefix_adds_user_message():
    handler = ContextPrefixHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="context_prefix",
        params={"content": "Consider the following context."},
    )
    action = CBlock("test action")

    new_action, new_ctx = handler.apply(control, action, [], None)

    assert new_action is action
    assert len(new_ctx) == 1
    assert isinstance(new_ctx[0], Message)
    assert new_ctx[0].role == "user"
    assert new_ctx[0].content == "Consider the following context."


def test_context_prefix_with_custom_role():
    handler = ContextPrefixHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="context_prefix",
        params={"content": "System grounding.", "role": "system"},
    )
    action = CBlock("test")

    _, new_ctx = handler.apply(control, action, [], None)

    assert new_ctx[0].role == "system"


def test_context_prefix_registered_globally():
    handler = get_global_input_handler("context_prefix")
    assert handler is not None
    assert isinstance(handler, ContextPrefixHandler)
