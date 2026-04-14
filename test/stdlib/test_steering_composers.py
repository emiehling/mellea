"""Unit tests for composer implementations."""

from mellea.core.requirement import Requirement, ValidationResult
from mellea.core.steering import (
    BackendCapabilities,
    Control,
    ControlCategory,
    SteeringPolicy,
)
from mellea.stdlib.steering.composers import (
    CompositeComposer,
    NoOpComposer,
    PerRequirementComposer,
)
from mellea.steering.artifacts import ArtifactRegistry, SteeringArtifact

# --- NoOpComposer ---


def test_noop_composer_compose_returns_empty():
    composer = NoOpComposer()
    reqs = [Requirement("be concise")]
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))
    policy = composer.compose(reqs, caps)
    assert not policy
    assert policy.controls == ()


def test_noop_composer_update_returns_unchanged():
    composer = NoOpComposer()
    c = Control(category=ControlCategory.INPUT, name="test")
    original_policy = SteeringPolicy(controls=(c,))
    caps = BackendCapabilities()

    updated = composer.update(
        original_policy,
        [(Requirement("test"), ValidationResult(False, reason="failed"))],
        caps,
    )
    assert updated is original_policy
    assert len(updated.controls) == 1


# --- PerRequirementComposer ---


def _make_registry() -> ArtifactRegistry:
    reg = ArtifactRegistry()
    reg.register(
        SteeringArtifact(
            name="conciseness_adapter",
            description="be concise in output",
            category=ControlCategory.INPUT,
            model_family=None,
            artifact_type="prompt_adapter",
            path_or_ref="prompts/concise",
        )
    )
    reg.register(
        SteeringArtifact(
            name="honesty_vector",
            description="be honest in generation",
            category=ControlCategory.STATE,
            model_family="granite",
            artifact_type="steering_vector",
            path_or_ref="vectors/honesty",
        )
    )
    return reg


def test_per_requirement_composer_compose():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    reqs = [Requirement("be concise")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.OUTPUT})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].name == "conciseness_adapter"


def test_per_requirement_composer_filters_by_capabilities():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    reqs = [Requirement("be honest")]
    # Backend only supports INPUT, not STATE
    caps = BackendCapabilities(supported_categories=frozenset({ControlCategory.INPUT}))

    policy = composer.compose(reqs, caps)
    # honesty_vector is STATE, so it should be filtered out
    assert len(policy.controls) == 0


def test_per_requirement_composer_compose_with_state_support():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    reqs = [Requirement("be honest")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.STATE})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].name == "honesty_vector"


def test_per_requirement_composer_update_adds_new_controls():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    initial_policy = SteeringPolicy.empty()
    failed_results = [
        (Requirement("be concise"), ValidationResult(False, reason="too verbose"))
    ]

    updated = composer.update(initial_policy, failed_results, caps)
    assert len(updated.controls) == 1
    assert updated.controls[0].name == "conciseness_adapter"


def test_per_requirement_composer_update_no_duplicates():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    # Start with conciseness already in the policy
    existing = Control(
        category=ControlCategory.INPUT,
        name="conciseness_adapter",
        artifact_ref="prompts/concise",
    )
    initial_policy = SteeringPolicy(controls=(existing,))

    failed_results = [
        (Requirement("be concise"), ValidationResult(False, reason="too verbose"))
    ]

    updated = composer.update(initial_policy, failed_results, caps)
    # Should not add duplicate
    assert len(updated.controls) == 1


def test_per_requirement_composer_update_skips_passing_reqs():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    results = [(Requirement("be concise"), ValidationResult(True))]
    updated = composer.update(SteeringPolicy.empty(), results, caps)
    assert len(updated.controls) == 0


def test_per_requirement_composer_no_description():
    registry = _make_registry()
    composer = PerRequirementComposer(registry)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    # Requirement with no description should be skipped
    reqs = [Requirement(description=None)]
    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 0


# --- CompositeComposer ---


def test_composite_composer_delegates_to_per_requirement():
    registry = _make_registry()
    composer = CompositeComposer(registry)
    reqs = [Requirement("be concise")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.OUTPUT})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].name == "conciseness_adapter"
