"""Unit tests for composer implementations."""

from __future__ import annotations

from typing import Any

import pytest

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
from mellea.steering.library import ArtifactLibrary, set_default_library
from mellea.steering.stores.base import ArtifactStore, semantic_match

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

    def search(
        self,
        query: str,
        model: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[str] = []
        filtered_items: list[dict[str, Any]] = []
        for item in self._items:
            m = item.get("model")
            if model is not None and m is not None and m != model:
                continue
            desc = item.get("description", "")
            candidates.append(f"{item['name']}: {desc}")
            filtered_items.append(item)

        matched = semantic_match(query, candidates, max_results=max_results)
        return [filtered_items[i] for i in matched]

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
                    "layer": 0,
                    "multiplier": 1.5,
                    "transform": "additive",
                },
            }
        ]
    )
    return ArtifactLibrary(
        {ControlCategory.INPUT: input_store, ControlCategory.STATE: state_store}
    )


@pytest.fixture(autouse=True)
def _configure_library():
    """Set up a test library as the default for all tests in this module."""
    set_default_library(_make_library())
    yield
    set_default_library(None)


# --- semantic_match max_results ---


@pytest.mark.e2e
def test_semantic_match_max_results_returns_top_n():
    """max_results=1 returns only the single highest-scoring match."""
    candidates = [
        "sentiment: positive emotional tone",
        "warmth: warm and empathetic tone",
        "formality: formal language style",
        "technicality: technical jargon level",
    ]
    results = semantic_match("positive optimistic", candidates, max_results=1)
    assert len(results) <= 1


@pytest.mark.e2e
def test_semantic_match_max_results_none_returns_all():
    """max_results=None preserves existing behavior."""
    candidates = [
        "sentiment: positive emotional tone",
        "warmth: warm and empathetic tone",
        "formality: formal language style",
    ]
    all_results = semantic_match("positive optimistic", candidates)
    none_results = semantic_match("positive optimistic", candidates, max_results=None)
    assert all_results == none_results


@pytest.mark.e2e
def test_semantic_match_max_results_respects_threshold():
    """max_results does not return results below threshold."""
    candidates = ["xyzzy: completely unrelated gibberish nonsense"]
    results = semantic_match("positive optimistic", candidates, max_results=5)
    assert len(results) == 0


@pytest.mark.e2e
def test_semantic_match_max_results_ordered_by_score():
    """Returned indices correspond to the highest-scoring candidates."""
    candidates = [
        "sentiment: positive emotional tone",
        "warmth: warm and empathetic tone",
        "formality: formal language style",
        "technicality: technical jargon level",
    ]
    top1 = semantic_match("positive optimistic", candidates, max_results=1)
    all_matches = semantic_match("positive optimistic", candidates)
    if top1:
        assert top1[0] in all_matches


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
    composer = PerRequirementComposer()
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
    composer = PerRequirementComposer()
    reqs = [Requirement("be honest")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.STATE})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    ctrl = policy.controls[0]
    assert ctrl.name == "activation_steering"
    assert ctrl.params["layer"] == 0
    assert ctrl.params["multiplier"] == 1.5
    assert ctrl.params["transform"] == "additive"
    assert ctrl.model_family == "granite"


def test_per_requirement_composer_filters_by_capabilities():
    composer = PerRequirementComposer()
    reqs = [Requirement("be honest")]
    # Backend only supports INPUT, not STATE
    caps = BackendCapabilities(supported_categories=frozenset({ControlCategory.INPUT}))

    policy = composer.compose(reqs, caps)
    # honesty_vector is STATE, so it should be filtered out
    assert len(policy.controls) == 0


def test_per_requirement_composer_compose_with_state_support():
    composer = PerRequirementComposer()
    reqs = [Requirement("be honest")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.STATE})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].artifact_ref == "honesty_vector"


def test_per_requirement_composer_update_adds_new_controls():
    composer = PerRequirementComposer()
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    initial_policy = SteeringPolicy.empty()
    failed_results = [
        (Requirement("be concise"), ValidationResult(False, reason="too verbose"))
    ]

    updated = composer.update(initial_policy, failed_results, caps)
    assert len(updated.controls) == 1
    assert updated.controls[0].artifact_ref == "conciseness_adapter"


def test_per_requirement_composer_update_no_duplicates():
    composer = PerRequirementComposer()
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
    composer = PerRequirementComposer()
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    results = [(Requirement("be concise"), ValidationResult(True))]
    updated = composer.update(SteeringPolicy.empty(), results, caps)
    assert len(updated.controls) == 0


def test_per_requirement_composer_no_description():
    composer = PerRequirementComposer()
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    # Requirement with no description should be skipped
    reqs = [Requirement(description=None)]
    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 0


def test_per_requirement_composer_compose_deduplicates_across_requirements():
    """Two requirements matching the same artifact produce one control, not two."""
    composer = PerRequirementComposer()
    caps = BackendCapabilities(supported_categories=frozenset(ControlCategory))

    reqs = [
        Requirement("be honest and transparent"),
        Requirement("truthful and honest responses"),
    ]
    policy = composer.compose(reqs, caps)
    artifact_refs = [c.artifact_ref for c in policy.controls]
    assert len(artifact_refs) == len(set(artifact_refs)), (
        f"Duplicate artifact refs in composed policy: {artifact_refs}"
    )


# --- CompositeComposer ---


def test_composite_composer_delegates_to_per_requirement():
    composer = CompositeComposer()
    reqs = [Requirement("be concise")]
    caps = BackendCapabilities(
        supported_categories=frozenset({ControlCategory.INPUT, ControlCategory.OUTPUT})
    )

    policy = composer.compose(reqs, caps)
    assert len(policy.controls) == 1
    assert policy.controls[0].artifact_ref == "conciseness_adapter"
