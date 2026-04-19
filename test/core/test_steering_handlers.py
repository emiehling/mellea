"""Unit tests for handler ABCs, ResolvedControl, and global input handler registry."""

from typing import Any

from mellea.core.steering import (
    Control,
    ControlCategory,
    InputControlHandler,
    OutputControlHandler,
    ResolvedControl,
    StateControlHandler,
    StructuralControlHandler,
    get_global_input_handler,
    register_global_input_handler,
)


# --- Concrete handler implementations for testing ---


class _MockInputHandler(InputControlHandler):
    def apply(self, control, action, linearized_ctx, artifact):
        return action, linearized_ctx


class _MockStructuralHandler(StructuralControlHandler):
    def activate(self, control, model, artifact):
        return "structural_handle"

    def deactivate(self, handle):
        pass


class _MockStateHandler(StateControlHandler):
    def activate(self, control, model, artifact):
        return ["hook1", "hook2"]

    def deactivate(self, handle):
        pass


class _MockOutputHandler(OutputControlHandler):
    def apply(self, control, gen_kwargs, artifact):
        gen_kwargs["modified"] = True
        return gen_kwargs


# --- InputControlHandler ---


def test_input_handler_abc():
    handler = _MockInputHandler()
    c = Control(category=ControlCategory.INPUT, name="test")
    action_out, ctx_out = handler.apply(c, "action", [], None)
    assert action_out == "action"
    assert ctx_out == []


# --- StructuralControlHandler ---


def test_structural_handler_activate_deactivate():
    handler = _MockStructuralHandler()
    c = Control(category=ControlCategory.STRUCTURAL, name="test")
    handle = handler.activate(c, "model", None)
    assert handle == "structural_handle"
    handler.deactivate(handle)


# --- StateControlHandler ---


def test_state_handler_activate_deactivate():
    handler = _MockStateHandler()
    c = Control(category=ControlCategory.STATE, name="test")
    handle = handler.activate(c, "model", None)
    assert handle == ["hook1", "hook2"]
    handler.deactivate(handle)


# --- OutputControlHandler ---


def test_output_handler_apply():
    handler = _MockOutputHandler()
    c = Control(category=ControlCategory.OUTPUT, name="test")
    result = handler.apply(c, {"temperature": 0.5}, None)
    assert result["modified"] is True
    assert result["temperature"] == 0.5


# --- ResolvedControl ---


def test_resolved_control_creation():
    c = Control(category=ControlCategory.INPUT, name="test")
    handler = _MockInputHandler()
    rc = ResolvedControl(control=c, handler=handler, artifact=None)
    assert rc.control is c
    assert rc.handler is handler
    assert rc.artifact is None


def test_resolved_control_with_artifact():
    c = Control(category=ControlCategory.STATE, name="test", artifact_ref="ref/1")
    handler = _MockStateHandler()
    rc = ResolvedControl(control=c, handler=handler, artifact="loaded_vector")
    assert rc.artifact == "loaded_vector"


def test_resolved_control_is_frozen():
    c = Control(category=ControlCategory.INPUT, name="test")
    handler = _MockInputHandler()
    rc = ResolvedControl(control=c, handler=handler)
    try:
        rc.artifact = "new"  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


# --- Global input handler registry ---


def test_register_and_get_global_input_handler():
    handler = _MockInputHandler()
    register_global_input_handler("_test_handler", handler)
    retrieved = get_global_input_handler("_test_handler")
    assert retrieved is handler


def test_get_global_input_handler_not_found():
    result = get_global_input_handler("_nonexistent_handler_xyz")
    assert result is None
