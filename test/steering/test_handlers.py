"""Unit tests for built-in input control handlers."""

from mellea.core.base import CBlock
from mellea.core.steering import Control, ControlCategory, get_global_input_handler
from mellea.steering.handlers import (
    ContextPrefixHandler,
    InstructionRewriteHandler,
    SystemPromptInjectionHandler,
)
from mellea.stdlib.components.chat import Message
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import ChatContext


# --- SystemPromptInjectionHandler ---


def test_system_prompt_injection_adds_message():
    handler = SystemPromptInjectionHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="system_prompt_injection",
        params={"system_prompt": "You are a helpful assistant."},
    )
    action = CBlock("test action")
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, action, ctx, None)

    # Action unchanged.
    assert new_action is action
    # Context has a new entry.
    ctx_list = new_ctx.as_list()
    assert len(ctx_list) == 1
    assert isinstance(ctx_list[0], Message)
    assert ctx_list[0].role == "system"
    assert ctx_list[0].content == "You are a helpful assistant."


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
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, instruction, ctx, None)

    assert isinstance(new_action, Instruction)
    assert "Be concise." in str(new_action._description)
    assert "Write a story." in str(new_action._description)
    # Context unchanged.
    assert new_ctx is ctx


def test_instruction_rewrite_leaves_non_instruction_unchanged():
    handler = InstructionRewriteHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="instruction_rewrite",
        params={"template": "Be concise. {original}"},
    )
    action = CBlock("not an instruction")
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, action, ctx, None)

    # Action unchanged.
    assert new_action is action
    assert new_ctx is ctx


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
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, action, ctx, None)

    assert new_action is action
    ctx_list = new_ctx.as_list()
    assert len(ctx_list) == 1
    assert isinstance(ctx_list[0], Message)
    assert ctx_list[0].role == "user"
    assert ctx_list[0].content == "Consider the following context."


def test_context_prefix_with_custom_role():
    handler = ContextPrefixHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="context_prefix",
        params={"content": "System grounding.", "role": "system"},
    )
    action = CBlock("test")
    ctx = ChatContext()

    _, new_ctx = handler.apply(control, action, ctx, None)

    ctx_list = new_ctx.as_list()
    assert ctx_list[0].role == "system"


def test_context_prefix_registered_globally():
    handler = get_global_input_handler("context_prefix")
    assert handler is not None
    assert isinstance(handler, ContextPrefixHandler)
