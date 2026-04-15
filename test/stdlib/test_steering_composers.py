"""Unit tests for composer implementations."""

from __future__ import annotations

from typing import Any

from mellea.core.requirement import Requirement, ValidationResult
from mellea.core.steering import (
    BackendCapabilities,
    Control,
    ControlCategory,
    SteeringPolicy,
)
from mellea.stdlib.steering import (
    CompositeComposer,
    NoOpComposer,
    PerRequirementComposer,
)
from mellea.steering.library import ArtifactLibrary
from mellea.steering.stores.base import ArtifactStore

# --- Mock store for test artifacts ---


class _TestStore(ArtifactStore):
    """In-memory store for testing composers."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        name = selectors.get("name")
        for item in self._items:
            if item["name"] == name:
                return item.get("payload", name), item.get("default_params", {})
        raise KeyError(f"not found: {name}")

    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        results = []
        query_lower = query.lower()
        for item in self._items:
            desc = item.get("description", "")
            m = item.get("model")
            if model is not None and m is not None and m != model:
                continue
            if query_lower in item["name"].lower() or query_lower in desc.lower():
                results.append(item)
        return results

    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        return list(self._items)


def _make_library() -> ArtifactLibrary:
    """Create a library with test artifacts."""
    input_store = _TestStore(
        [
            {
                "name": "conciseness_adapter",
                "description": "be concise in output",
                "model": None,
                "handler": "instruction_rewrite",
                "default_params": {"role": "user"},
            }
        ]
    )
    state_store = _TestStore(
        [
            {
                "name": "honesty_vector",
                "description": "be honest in generation",
                "model": "granite",
                "handler": "activation_steering",
                "default_params": {
                    "layers": [0, 1, 2],
                    "coefficient": 1.5,
                    "transform": "additive",
                },
            }
        ]
    )
    return ArtifactLibrary(
        {ControlCategory.INPUT: input_store, ControlCategory.STATE: state_store}
    )


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


def test_per_requirement_composer_compose():
    library = _make_library()
    composer = PerRequirementComposer(library)
    reqs = [Requirement("be concise")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.OUTPUT})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    ctrl = policy.controls[0]
    assert ctrl.name == "instruction_rewrite"
    assert ctrl.artifact_ref == "conciseness_adapter"


def test_per_requirement_composer_populates_params_from_defaults():
    library = _make_library()
    composer = PerRequirementComposer(library)
    reqs = [Requirement("be honest")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.STATE})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    ctrl = policy.controls[0]
    assert ctrl.name == "activation_steering"
    assert ctrl.params["layers"] == [0, 1, 2]
    assert ctrl.params["coefficient"] == 1.5
    assert ctrl.params["transform"] == "additive"
    assert ctrl.model_family == "granite"


def test_per_requirement_composer_filters_by_capabilities():
    library = _make_library()
    composer = PerRequirementComposer(library)
    reqs = [Requirement("be honest")]
    # Backend only supports INPUT, not STATE
    caps = BackendCapabilities(supported_categories=frozenset({ControlCategory.INPUT}))

    policy = composer.compose(reqs, caps)
    # honesty_vector is STATE, so it should be filtered out
    assert len(policy.controls) == 0


def test_per_requirement_composer_compose_with_state_support():
    library = _make_library()
    composer = PerRequirementComposer(library)
    reqs = [Requirement("be honest")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.STATE})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].artifact_ref == "honesty_vector"


def test_per_requirement_composer_update_adds_new_controls():
    library = _make_library()
    composer = PerRequirementComposer(library)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    initial_policy = SteeringPolicy.empty()
    failed_results = [
        (Requirement("be concise"), ValidationResult(False, reason="too verbose"))
    ]

    updated = composer.update(initial_policy, failed_results, caps)
    assert len(updated.controls) == 1
    assert updated.controls[0].artifact_ref == "conciseness_adapter"


def test_per_requirement_composer_update_no_duplicates():
    library = _make_library()
    composer = PerRequirementComposer(library)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    # Start with conciseness already in the policy
    existing = Control(
        category=ControlCategory.INPUT,
        name="instruction_rewrite",
        artifact_ref="conciseness_adapter",
    )
    initial_policy = SteeringPolicy(controls=(existing,))

    failed_results = [
        (Requirement("be concise"), ValidationResult(False, reason="too verbose"))
    ]

    updated = composer.update(initial_policy, failed_results, caps)
    # Should not add duplicate
    assert len(updated.controls) == 1


def test_per_requirement_composer_update_skips_passing_reqs():
    library = _make_library()
    composer = PerRequirementComposer(library)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    results = [(Requirement("be concise"), ValidationResult(True))]
    updated = composer.update(SteeringPolicy.empty(), results, caps)
    assert len(updated.controls) == 0


def test_per_requirement_composer_no_description():
    library = _make_library()
    composer = PerRequirementComposer(library)
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    # Requirement with no description should be skipped
    reqs = [Requirement(description=None)]
    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 0


# --- CompositeComposer ---


def test_composite_composer_delegates_to_per_requirement():
    library = _make_library()
    composer = CompositeComposer(library)
    reqs = [Requirement("be concise")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.OUTPUT})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].artifact_ref == "conciseness_adapter"
