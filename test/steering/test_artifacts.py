"""Unit tests for steering artifact registry."""

import pytest

from mellea.core.steering import ControlCategory
from mellea.steering.artifacts import ArtifactRegistry, SteeringArtifact


@pytest.fixture
def registry() -> ArtifactRegistry:
    """Create a registry with some test artifacts."""
    reg = ArtifactRegistry()
    reg.register(
        SteeringArtifact(
            name="honesty_vector",
            description="Steering vector for honest generation",
            category=ControlCategory.STATE,
            model_family="granite",
            artifact_type="steering_vector",
            path_or_ref="vectors/honesty_v1.pt",
        )
    )
    reg.register(
        SteeringArtifact(
            name="conciseness_prompt",
            description="Prompt rewrite for concise output",
            category=ControlCategory.INPUT,
            model_family=None,
            artifact_type="prompt_adapter",
            path_or_ref="prompts/conciseness_v1",
        )
    )
    reg.register(
        SteeringArtifact(
            name="safety_vector",
            description="Steering vector for safe generation",
            category=ControlCategory.STATE,
            model_family="llama",
            artifact_type="steering_vector",
            path_or_ref="vectors/safety_v1.pt",
        )
    )
    return reg


def test_artifact_creation():
    a = SteeringArtifact(
        name="test",
        description="test artifact",
        category=ControlCategory.INPUT,
        model_family=None,
        artifact_type="steering_vector",
        path_or_ref="test/path",
    )
    assert a.name == "test"
    assert a.category == ControlCategory.INPUT


def test_registry_register_and_len(registry: ArtifactRegistry):
    assert len(registry) == 3


def test_registry_search_no_filters(registry: ArtifactRegistry):
    results = registry.search()
    assert len(results) == 3


def test_registry_search_by_category(registry: ArtifactRegistry):
    results = registry.search(category=ControlCategory.STATE)
    assert len(results) == 2
    assert all(a.category == ControlCategory.STATE for a in results)


def test_registry_search_by_model_family(registry: ArtifactRegistry):
    # Should match granite-specific + model-agnostic
    results = registry.search(model_family="granite")
    assert len(results) == 2
    names = {a.name for a in results}
    assert "honesty_vector" in names
    assert "conciseness_prompt" in names  # model_family is None, matches any


def test_registry_search_by_query(registry: ArtifactRegistry):
    results = registry.search(query="honest")
    assert len(results) == 1
    assert results[0].name == "honesty_vector"


def test_registry_search_combined_filters(registry: ArtifactRegistry):
    results = registry.search(category=ControlCategory.STATE, model_family="granite")
    assert len(results) == 1
    assert results[0].name == "honesty_vector"


def test_registry_resolve(registry: ArtifactRegistry):
    result = registry.resolve("vectors/honesty_v1.pt")
    assert result == "vectors/honesty_v1.pt"


def test_registry_resolve_not_found(registry: ArtifactRegistry):
    with pytest.raises(KeyError, match="No artifact found"):
        registry.resolve("nonexistent/ref")


def test_empty_registry():
    reg = ArtifactRegistry()
    assert len(reg) == 0
    assert reg.search() == []
