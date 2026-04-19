"""Unit tests for the artifact library and info types."""

from __future__ import annotations

from typing import Any

import pytest

from mellea.core.steering import ControlCategory
from mellea.steering.library import (
    ArtifactInfo,
    ArtifactLibrary,
    get_default_library,
    set_default_library,
)
from mellea.steering.stores.base import ArtifactStore

# --- ArtifactInfo ---


def test_artifact_info_creation():
    info = ArtifactInfo(
        name="test_vec",
        category=ControlCategory.STATE,
        description="a test vector",
        model="granite",
        handler="activation_steering",
        default_params={"multiplier": 1.5, "layer": 0},
    )
    assert info.name == "test_vec"
    assert info.category == ControlCategory.STATE
    assert info.model == "granite"
    assert info.handler == "activation_steering"
    assert info.default_params["multiplier"] == 1.5


def test_artifact_info_with_param_space():
    ps = {"by_layer": {15: {"multiplier": {"min": 0.8, "max": 1.8}}}}
    info = ArtifactInfo(
        name="test_vec",
        category=ControlCategory.STATE,
        description="a test vector",
        model="granite",
        param_space=ps,
    )
    assert info.param_space["by_layer"][15]["multiplier"]["max"] == 1.8


def test_artifact_info_defaults():
    info = ArtifactInfo(
        name="x", category=ControlCategory.INPUT, description="", model=None
    )
    assert info.handler is None
    assert info.default_params == {}
    assert info.param_space == {}


def test_artifact_info_frozen():
    info = ArtifactInfo(
        name="x", category=ControlCategory.INPUT, description="", model=None
    )
    with pytest.raises(AttributeError):
        info.name = "changed"  # type: ignore[misc]


# --- ArtifactLibrary ---


class _MockStore(ArtifactStore):
    """Minimal store for testing library dispatch."""

    def __init__(self, items: dict[str, tuple[Any, dict[str, Any]]]) -> None:
        self._items = items

    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        name = selectors.get("name")
        if name not in self._items:
            raise KeyError(f"not found: {name}")
        return self._items[name]

    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        results = []
        for name, (_, params) in self._items.items():
            desc = params.get("_desc", "")
            if query.lower() in name.lower() or query.lower() in desc.lower():
                results.append(
                    {
                        "name": name,
                        "description": desc,
                        "model": model,
                        "handler": params.get("_handler"),
                        "default_params": {
                            k: v for k, v in params.items() if not k.startswith("_")
                        },
                    }
                )
        return results

    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        results = []
        for name, (_, params) in self._items.items():
            results.append(
                {
                    "name": name,
                    "description": params.get("_desc", ""),
                    "model": None,
                    "handler": params.get("_handler"),
                    "default_params": {
                        k: v for k, v in params.items() if not k.startswith("_")
                    },
                }
            )
        return results


class _MockStoreWithParamSpace(ArtifactStore):
    """Mock store that includes param_space in search/list results."""

    def __init__(
        self, items: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]]
    ) -> None:
        self._items = items

    def get_raw(self, **selectors: Any) -> tuple[Any, dict[str, Any]]:
        name = selectors.get("name")
        if name not in self._items:
            raise KeyError(f"not found: {name}")
        artifact, params, _ = self._items[name]
        return artifact, {k: v for k, v in params.items() if not k.startswith("_")}

    def search(self, query: str, model: str | None = None) -> list[dict[str, Any]]:
        results = []
        for name, (_, params, ps) in self._items.items():
            desc = params.get("_desc", "")
            if query.lower() in name.lower() or query.lower() in desc.lower():
                results.append(
                    {
                        "name": name,
                        "description": desc,
                        "model": model,
                        "handler": params.get("_handler"),
                        "default_params": {
                            k: v for k, v in params.items() if not k.startswith("_")
                        },
                        "param_space": ps,
                    }
                )
        return results

    def list_artifacts(self, **partial_selectors: Any) -> list[dict[str, Any]]:
        results = []
        for name, (_, params, ps) in self._items.items():
            results.append(
                {
                    "name": name,
                    "description": params.get("_desc", ""),
                    "model": None,
                    "handler": params.get("_handler"),
                    "default_params": {
                        k: v for k, v in params.items() if not k.startswith("_")
                    },
                    "param_space": ps,
                }
            )
        return results


def test_library_get_delegates_to_store():
    store = _MockStore({"honesty": ("vector_data", {"multiplier": 1.5})})
    lib = ArtifactLibrary({ControlCategory.STATE: store})

    artifact, default_params = lib.get(ControlCategory.STATE, name="honesty")
    assert artifact == "vector_data"
    assert default_params["multiplier"] == 1.5


def test_library_get_raises_for_missing_store():
    lib = ArtifactLibrary()
    with pytest.raises(ValueError, match="No store configured"):
        lib.get(ControlCategory.STATE, name="x")


def test_library_get_raises_for_missing_artifact():
    store = _MockStore({})
    lib = ArtifactLibrary({ControlCategory.STATE: store})
    with pytest.raises(KeyError):
        lib.get(ControlCategory.STATE, name="nonexistent")


def test_library_search_single_store():
    store = _MockStore(
        {
            "honesty": ("v1", {"_desc": "honest generation"}),
            "safety": ("v2", {"_desc": "safe generation"}),
        }
    )
    lib = ArtifactLibrary({ControlCategory.STATE: store})

    results = lib.search("honest")
    assert len(results) == 1
    assert results[0].name == "honesty"
    assert results[0].category == ControlCategory.STATE


def test_library_search_across_stores():
    state_store = _MockStore({"honesty": ("v1", {"_desc": "honest vec"})})
    input_store = _MockStore({"concise": ("t1", {"_desc": "concise prompt"})})
    lib = ArtifactLibrary(
        {ControlCategory.STATE: state_store, ControlCategory.INPUT: input_store}
    )

    results = lib.search("conci")
    assert len(results) == 1
    assert results[0].name == "concise"
    assert results[0].category == ControlCategory.INPUT


def test_library_search_with_category_filter():
    state_store = _MockStore({"honesty": ("v1", {"_desc": "honest"})})
    input_store = _MockStore({"honest_prompt": ("t1", {"_desc": "honest"})})
    lib = ArtifactLibrary(
        {ControlCategory.STATE: state_store, ControlCategory.INPUT: input_store}
    )

    results = lib.search("honest", category=ControlCategory.STATE)
    assert len(results) == 1
    assert results[0].category == ControlCategory.STATE


def test_library_search_returns_handler_and_params():
    store = _MockStore(
        {"vec": ("v1", {"_desc": "test", "_handler": "act_steer", "multiplier": 2.0})}
    )
    lib = ArtifactLibrary({ControlCategory.STATE: store})

    results = lib.search("test")
    assert len(results) == 1
    assert results[0].handler == "act_steer"
    assert results[0].default_params == {"multiplier": 2.0}


def test_library_search_threads_param_space():
    store = _MockStoreWithParamSpace(
        {
            "vec": (
                "v1",
                {"_desc": "test", "_handler": "act_steer"},
                {"by_layer": {15: {"multiplier": {"min": 0.8, "max": 1.8}}}},
            )
        }
    )
    lib = ArtifactLibrary({ControlCategory.STATE: store})

    results = lib.search("test")
    assert len(results) == 1
    assert results[0].param_space["by_layer"][15]["multiplier"]["max"] == 1.8


def test_library_list_threads_param_space():
    store = _MockStoreWithParamSpace(
        {
            "vec": (
                "v1",
                {"_desc": "test"},
                {"by_layer": {10: {"multiplier": {"min": 0.5, "max": 1.0}}}},
            )
        }
    )
    lib = ArtifactLibrary({ControlCategory.STATE: store})

    results = lib.list(ControlCategory.STATE)
    assert len(results) == 1
    assert results[0].param_space["by_layer"][10]["multiplier"]["min"] == 0.5


def test_library_list():
    store = _MockStore(
        {"a": ("va", {"_desc": "artifact a"}), "b": ("vb", {"_desc": "artifact b"})}
    )
    lib = ArtifactLibrary({ControlCategory.INPUT: store})

    results = lib.list(ControlCategory.INPUT)
    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"a", "b"}


def test_library_list_raises_for_missing_store():
    lib = ArtifactLibrary()
    with pytest.raises(ValueError, match="No store configured"):
        lib.list(ControlCategory.OUTPUT)


def test_library_register_store():
    lib = ArtifactLibrary()
    store = _MockStore({"x": ("val", {"key": "value"})})
    lib.register_store(ControlCategory.STRUCTURAL, store)

    artifact, default_params = lib.get(ControlCategory.STRUCTURAL, name="x")
    assert artifact == "val"
    assert default_params == {"key": "value"}


# --- Singleton ---


def test_get_default_library_creates_empty_on_first_access():
    import mellea.steering.library as mod

    original = mod._default_library
    try:
        mod._default_library = None
        lib = get_default_library()
        assert isinstance(lib, ArtifactLibrary)
    finally:
        mod._default_library = original


def test_set_and_get_default_library():
    import mellea.steering.library as mod

    original = mod._default_library
    try:
        lib = ArtifactLibrary()
        set_default_library(lib)
        assert get_default_library() is lib
    finally:
        mod._default_library = original
