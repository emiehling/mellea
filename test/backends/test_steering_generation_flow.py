"""Integration tests for the end-to-end steered generation flow using DummyBackend."""

from mellea.backends.dummy import DummyBackend
from mellea.core.steering import (
    BackendCapabilities,
    Control,
    ControlCategory,
    InputControlHandler,
    OutputControlHandler,
    SteeringPolicy,
)
from mellea.steering.controls import input_control


class _TrackingInputHandler(InputControlHandler):
    """Input handler that records calls for verification."""

    def __init__(self):
        self.calls: list[str] = []

    def apply(self, control, action, context, artifact):
        self.calls.append(control.name)
        return action, context


class _TrackingOutputHandler(OutputControlHandler):
    """Output handler that records calls for verification."""

    def __init__(self):
        self.calls: list[str] = []

    def apply(self, control, gen_kwargs, artifact):
        self.calls.append(control.name)
        gen_kwargs["steered"] = True
        return gen_kwargs


# --- attach populates resolved_controls ---


def test_attach_populates_resolved_controls():
    backend = DummyBackend(responses=None)
    handler = _TrackingInputHandler()
    backend.register_handler("track_input", handler)

    c = Control(category=ControlCategory.INPUT, name="track_input")
    policy = SteeringPolicy(controls=(c,))
    backend.attach(policy)

    assert len(backend._resolved_controls) == 1
    assert backend._resolved_controls[0].handler is handler


def test_empty_policy_is_noop():
    backend = DummyBackend(responses=None)
    policy = SteeringPolicy.empty()
    backend.attach(policy)
    assert backend._resolved_controls == ()
    assert not backend.steering_policy


def test_controls_for_stage_filters_correctly():
    backend = DummyBackend(responses=None)
    input_handler = _TrackingInputHandler()
    output_handler = _TrackingOutputHandler()
    backend.register_handler("inp", input_handler)
    backend.register_handler("out", output_handler)

    c1 = Control(category=ControlCategory.INPUT, name="inp")
    c2 = Control(category=ControlCategory.OUTPUT, name="out")
    policy = SteeringPolicy(controls=(c1, c2))
    backend.attach(policy)

    input_rcs = backend.resolved_controls_for_stage(ControlCategory.INPUT)
    output_rcs = backend.resolved_controls_for_stage(ControlCategory.OUTPUT)
    state_rcs = backend.resolved_controls_for_stage(ControlCategory.STATE)

    assert len(input_rcs) == 1
    assert len(output_rcs) == 1
    assert len(state_rcs) == 0


def test_attach_then_detach_cycle():
    backend = DummyBackend(responses=None)
    handler = _TrackingInputHandler()
    backend.register_handler("track", handler)

    c = Control(category=ControlCategory.INPUT, name="track")
    policy = SteeringPolicy(controls=(c,))

    # Attach
    backend.attach(policy)
    assert len(backend._resolved_controls) == 1

    # Detach
    detached = backend.detach()
    assert detached is policy
    assert backend._resolved_controls == ()
    assert not backend.steering_policy

    # Re-attach
    backend.attach(policy)
    assert len(backend._resolved_controls) == 1


def test_multiple_controls_same_category():
    backend = DummyBackend(responses=None)
    handler = _TrackingInputHandler()
    backend.register_handler("inp_a", handler)
    backend.register_handler("inp_b", handler)

    c1 = Control(category=ControlCategory.INPUT, name="inp_a")
    c2 = Control(category=ControlCategory.INPUT, name="inp_b")
    policy = SteeringPolicy(controls=(c1, c2))
    backend.attach(policy)

    input_rcs = backend.resolved_controls_for_stage(ControlCategory.INPUT)
    assert len(input_rcs) == 2
    assert input_rcs[0].control.name == "inp_a"
    assert input_rcs[1].control.name == "inp_b"


def test_dummy_backend_capabilities_unchanged():
    """DummyBackend should still report empty capabilities."""
    backend = DummyBackend(responses=None)
    caps = backend.capabilities
    assert isinstance(caps, BackendCapabilities)
    assert caps.supported_categories == frozenset()
