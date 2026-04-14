"""Unit tests for core steering abstractions."""

from mellea.core.steering import (
    BackendCapabilities,
    Composer,
    Control,
    ControlCategory,
    SteeringPolicy,
)

# --- ControlCategory ---


def test_control_category_values():
    assert ControlCategory.INPUT.value == "input"
    assert ControlCategory.STRUCTURAL.value == "structural"
    assert ControlCategory.STATE.value == "state"
    assert ControlCategory.OUTPUT.value == "output"


def test_control_category_members():
    members = list(ControlCategory)
    assert len(members) == 4


# --- Control ---


def test_control_creation():
    c = Control(category=ControlCategory.INPUT, name="test_control")
    assert c.category == ControlCategory.INPUT
    assert c.name == "test_control"
    assert c.params == {}
    assert c.artifact_ref is None
    assert c.model_family is None


def test_control_with_all_fields():
    c = Control(
        category=ControlCategory.STATE,
        name="honesty_vector",
        params={"layers": [10, 11, 12], "alpha": 0.5},
        artifact_ref="vectors/honesty_v1",
        model_family="granite",
    )
    assert c.category == ControlCategory.STATE
    assert c.name == "honesty_vector"
    assert c.params["alpha"] == 0.5
    assert c.artifact_ref == "vectors/honesty_v1"
    assert c.model_family == "granite"


def test_control_is_frozen():
    c = Control(category=ControlCategory.INPUT, name="test")
    try:
        c.name = "other"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


# --- SteeringPolicy ---


def test_steering_policy_empty():
    p = SteeringPolicy.empty()
    assert p.controls == ()
    assert not p  # __bool__ returns False for empty


def test_steering_policy_bool():
    empty = SteeringPolicy()
    assert not empty

    c = Control(category=ControlCategory.INPUT, name="test")
    nonempty = SteeringPolicy(controls=(c,))
    assert nonempty


def test_steering_policy_controls_for_stage():
    c1 = Control(category=ControlCategory.INPUT, name="c1")
    c2 = Control(category=ControlCategory.OUTPUT, name="c2")
    c3 = Control(category=ControlCategory.INPUT, name="c3")
    p = SteeringPolicy(controls=(c1, c2, c3))

    input_controls = p.controls_for_stage(ControlCategory.INPUT)
    assert len(input_controls) == 2
    assert input_controls[0].name == "c1"
    assert input_controls[1].name == "c3"

    output_controls = p.controls_for_stage(ControlCategory.OUTPUT)
    assert len(output_controls) == 1
    assert output_controls[0].name == "c2"

    state_controls = p.controls_for_stage(ControlCategory.STATE)
    assert len(state_controls) == 0


def test_steering_policy_add():
    c1 = Control(category=ControlCategory.INPUT, name="c1")
    c2 = Control(category=ControlCategory.OUTPUT, name="c2")
    p1 = SteeringPolicy(controls=(c1,))
    p2 = SteeringPolicy(controls=(c2,))

    combined = p1 + p2
    assert len(combined.controls) == 2
    assert combined.controls[0].name == "c1"
    assert combined.controls[1].name == "c2"

    # Original policies are unchanged (immutability)
    assert len(p1.controls) == 1
    assert len(p2.controls) == 1


def test_steering_policy_is_frozen():
    p = SteeringPolicy.empty()
    try:
        p.controls = ()  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


# --- BackendCapabilities ---


def test_backend_capabilities_defaults():
    caps = BackendCapabilities()
    assert caps.supported_categories == frozenset()
    assert caps.supports_logits_processors is False
    assert caps.supports_adapter_loading is False
    assert caps.supports_forward_hooks is False
    assert caps.extra == {}


def test_backend_capabilities_full():
    caps = BackendCapabilities(
        supported_categories=frozenset(ControlCategory),
        supports_logits_processors=True,
        supports_adapter_loading=True,
        supports_forward_hooks=True,
        extra={"custom_flag": True},
    )
    assert len(caps.supported_categories) == 4
    assert ControlCategory.INPUT in caps.supported_categories
    assert caps.supports_logits_processors is True
    assert caps.extra["custom_flag"] is True


def test_backend_capabilities_is_frozen():
    caps = BackendCapabilities()
    try:
        caps.supports_logits_processors = True  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
