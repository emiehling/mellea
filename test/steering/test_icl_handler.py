"""Unit tests for the ICL example selector handler."""

from mellea.core.base import CBlock
from mellea.core.steering import Control, ControlCategory, get_global_input_handler
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext
from mellea.steering.handlers import ICLExampleSelectorHandler


def test_icl_handler_registered_globally():
    handler = get_global_input_handler("icl_example_selector")
    assert handler is not None
    assert isinstance(handler, ICLExampleSelectorHandler)


def test_icl_handler_first_strategy():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 2, "strategy": "first", "role": "user"},
    )
    pool = ["example 1", "example 2", "example 3"]
    action = CBlock("test")
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, action, ctx, pool)

    assert new_action is action
    ctx_list = new_ctx.as_list()
    assert len(ctx_list) == 2
    assert isinstance(ctx_list[0], Message)
    assert ctx_list[0].content == "example 1"
    assert ctx_list[1].content == "example 2"


def test_icl_handler_random_strategy():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 2, "strategy": "random"},
    )
    pool = ["a", "b", "c", "d", "e"]
    action = CBlock("test")
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, action, ctx, pool)

    assert new_action is action
    ctx_list = new_ctx.as_list()
    assert len(ctx_list) == 2
    # All selected examples should come from the pool.
    for msg in ctx_list:
        assert isinstance(msg, Message)
        assert msg.content in pool


def test_icl_handler_defaults():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT, name="icl_example_selector", params={}
    )
    pool = ["ex1", "ex2", "ex3", "ex4"]
    action = CBlock("test")
    ctx = ChatContext()

    _new_action, new_ctx = handler.apply(control, action, ctx, pool)

    # Default count is 3, default strategy is "first", default role is "user".
    ctx_list = new_ctx.as_list()
    assert len(ctx_list) == 3
    assert ctx_list[0].content == "ex1"
    assert ctx_list[0].role == "user"


def test_icl_handler_custom_role():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 1, "role": "assistant"},
    )
    pool = ["response example"]
    action = CBlock("test")
    ctx = ChatContext()

    _, new_ctx = handler.apply(control, action, ctx, pool)

    ctx_list = new_ctx.as_list()
    assert ctx_list[0].role == "assistant"


def test_icl_handler_none_artifact():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT, name="icl_example_selector", params={"count": 3}
    )
    action = CBlock("test")
    ctx = ChatContext()

    new_action, new_ctx = handler.apply(control, action, ctx, None)

    assert new_action is action
    assert new_ctx is ctx


def test_icl_handler_count_exceeds_pool():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 10, "strategy": "first"},
    )
    pool = ["only one"]
    action = CBlock("test")
    ctx = ChatContext()

    _, new_ctx = handler.apply(control, action, ctx, pool)

    ctx_list = new_ctx.as_list()
    assert len(ctx_list) == 1
