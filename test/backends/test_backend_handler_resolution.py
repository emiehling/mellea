"""Unit tests for Backend handler registration, resolution, and attach/detach."""

import pytest

from mellea.backends.dummy import DummyBackend
from mellea.core.steering import (
    BackendCapabilities,
    Control,
    ControlCategory,
    InputControlHandler,
    OutputControlHandler,
    ResolvedControl,
    SteeringPolicy,
    register_global_input_handler,
)


class _TestInputHandler(InputControlHandler):
    def apply(self, control, action, context, artifact):
        return action, context


class _TestOutputHandler(OutputControlHandler):
    def apply(self, control, gen_kwargs, artifact):
        gen_kwargs["test"] = True
        return gen_kwargs


# --- register_handler / _resolve_handler ---


def test_register_and_resolve_handler():
    backend = DummyBackend(responses=None)
    handler = _TestOutputHandler()
    backend.register_handler("my_output", handler)

    control = Control(category=ControlCategory.OUTPUT, name="my_output")
    resolved = backend._resolve_handler(control)
    assert resolved is handler


def test_resolve_handler_global_fallback_for_input():
    handler = _TestInputHandler()
    register_global_input_handler("_test_global_fallback", handler)

    backend = DummyBackend(responses=None)
    control = Control(category=ControlCategory.INPUT, name="_test_global_fallback")
    resolved = backend._resolve_handler(control)
    assert resolved is handler


def test_resolve_handler_no_global_fallback_for_non_input():
    backend = DummyBackend(responses=None)
    control = Control(category=ControlCategory.OUTPUT, name="_nonexistent_output")
    with pytest.raises(ValueError, match="No handler registered"):
        backend._resolve_handler(control)


def test_resolve_handler_raises_for_unknown():
    backend = DummyBackend(responses=None)
    control = Control(category=ControlCategory.STATE, name="unknown_state_handler")
    with pytest.raises(ValueError, match="No handler registered"):
        backend._resolve_handler(control)


def test_backend_handler_overrides_global():
    """Backend-specific handler takes priority over global handler."""
    global_handler = _TestInputHandler()
    register_global_input_handler("_test_override", global_handler)

    backend_handler = _TestInputHandler()
    backend = DummyBackend(responses=None)
    backend.register_handler("_test_override", backend_handler)

    control = Control(category=ControlCategory.INPUT, name="_test_override")
    resolved = backend._resolve_handler(control)
    assert resolved is backend_handler
    assert resolved is not global_handler


# --- attach / detach ---


def test_attach_resolves_controls():
    backend = DummyBackend(responses=None)
    handler = _TestInputHandler()
    backend.register_handler("my_input", handler)

    control = Control(category=ControlCategory.INPUT, name="my_input")
    policy = SteeringPolicy(controls=(control,))
    backend.attach(policy)

    assert backend.steering_policy is policy
    assert len(backend._resolved_controls) == 1
    rc = backend._resolved_controls[0]
    assert isinstance(rc, ResolvedControl)
    assert rc.control is control
    assert rc.handler is handler
    assert rc.artifact is None


def test_attach_empty_policy():
    backend = DummyBackend(responses=None)
    policy = SteeringPolicy.empty()
    backend.attach(policy)

    assert not backend.steering_policy
    assert backend._resolved_controls == ()


def test_attach_raises_for_unresolvable_control():
    backend = DummyBackend(responses=None)
    control = Control(category=ControlCategory.STATE, name="no_handler_registered")
    policy = SteeringPolicy(controls=(control,))

    with pytest.raises(ValueError, match="No handler registered"):
        backend.attach(policy)


def test_detach_clears_resolved_controls():
    backend = DummyBackend(responses=None)
    handler = _TestInputHandler()
    backend.register_handler("my_input", handler)

    control = Control(category=ControlCategory.INPUT, name="my_input")
    policy = SteeringPolicy(controls=(control,))
    backend.attach(policy)
    assert len(backend._resolved_controls) == 1

    detached = backend.detach()
    assert detached is policy
    assert backend._resolved_controls == ()
    assert not backend.steering_policy


def test_detach_when_nothing_attached():
    backend = DummyBackend(responses=None)
    detached = backend.detach()
    assert detached is None
    assert backend._resolved_controls == ()


# --- resolved_controls_for_stage ---


def test_resolved_controls_for_stage_filters():
    backend = DummyBackend(responses=None)
    input_handler = _TestInputHandler()
    output_handler = _TestOutputHandler()
    backend.register_handler("inp", input_handler)
    backend.register_handler("out", output_handler)

    c1 = Control(category=ControlCategory.INPUT, name="inp")
    c2 = Control(category=ControlCategory.OUTPUT, name="out")
    c3 = Control(category=ControlCategory.INPUT, name="inp")
    policy = SteeringPolicy(controls=(c1, c2, c3))
    backend.attach(policy)

    input_rcs = backend.resolved_controls_for_stage(ControlCategory.INPUT)
    assert len(input_rcs) == 2
    assert all(rc.control.category == ControlCategory.INPUT for rc in input_rcs)

    output_rcs = backend.resolved_controls_for_stage(ControlCategory.OUTPUT)
    assert len(output_rcs) == 1
    assert output_rcs[0].control.category == ControlCategory.OUTPUT

    state_rcs = backend.resolved_controls_for_stage(ControlCategory.STATE)
    assert len(state_rcs) == 0
