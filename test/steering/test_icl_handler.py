"""Unit tests for the ICL example selector handler."""

from mellea.core.base import CBlock
from mellea.core.steering import Control, ControlCategory, get_global_input_handler
from mellea.stdlib.components.chat import Message
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

    new_action, new_ctx = handler.apply(control, action, [], pool)

    assert new_action is action
    assert len(new_ctx) == 2
    assert isinstance(new_ctx[0], Message)
    assert new_ctx[0].content == "example 1"
    assert new_ctx[1].content == "example 2"


def test_icl_handler_random_strategy():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 2, "strategy": "random"},
    )
    pool = ["a", "b", "c", "d", "e"]
    action = CBlock("test")

    new_action, new_ctx = handler.apply(control, action, [], pool)

    assert new_action is action
    assert len(new_ctx) == 2
    for msg in new_ctx:
        assert isinstance(msg, Message)
        assert msg.content in pool


def test_icl_handler_defaults():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT, name="icl_example_selector", params={}
    )
    pool = ["ex1", "ex2", "ex3", "ex4"]
    action = CBlock("test")

    _new_action, new_ctx = handler.apply(control, action, [], pool)

    assert len(new_ctx) == 3
    assert new_ctx[0].content == "ex1"
    assert new_ctx[0].role == "user"


def test_icl_handler_custom_role():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 1, "role": "assistant"},
    )
    pool = ["response example"]
    action = CBlock("test")

    _, new_ctx = handler.apply(control, action, [], pool)

    assert new_ctx[0].role == "assistant"


def test_icl_handler_none_artifact():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT, name="icl_example_selector", params={"count": 3}
    )
    action = CBlock("test")
    ctx_list: list = []

    new_action, new_ctx = handler.apply(control, action, ctx_list, None)

    assert new_action is action
    assert new_ctx is ctx_list


def test_icl_handler_count_exceeds_pool():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 10, "strategy": "first"},
    )
    pool = ["only one"]
    action = CBlock("test")

    _, new_ctx = handler.apply(control, action, [], pool)

    assert len(new_ctx) == 1


def test_icl_handler_with_empty_list():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 2, "strategy": "first"},
    )
    examples = ["Example 1", "Example 2", "Example 3"]

    _, new_ctx = handler.apply(control, CBlock("action"), [], examples)

    assert len(new_ctx) == 2


def test_icl_handler_preserves_existing_context():
    handler = ICLExampleSelectorHandler()
    control = Control(
        category=ControlCategory.INPUT,
        name="icl_example_selector",
        params={"count": 1, "strategy": "first"},
    )
    existing = [Message(role="user", content="existing")]

    _, new_ctx = handler.apply(control, CBlock("action"), existing, ["new example"])

    assert len(new_ctx) == 2
    assert new_ctx[0].content == "new example"
    assert new_ctx[1].content == "existing"
