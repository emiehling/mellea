"""Unit tests for per-category artifact stores."""

from __future__ import annotations

import pytest
import yaml

from mellea.steering.stores.adapter_store import AdapterStore
from mellea.steering.stores.base import ArtifactStore, semantic_match
from mellea.steering.stores.model_store import ModelStore
from mellea.steering.stores.prompt_store import PromptStore

# --- ArtifactStore ABC ---


def test_artifact_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ArtifactStore()  # type: ignore[abstract]


# --- PromptStore ---


@pytest.fixture
def prompt_dir(tmp_path):
    """Create a temporary prompt store directory with test artifacts."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "conciseness.yaml").write_text(
        yaml.dump(
            {
                "template": "Be concise. {original}",
                "description": "Prompt rewrite for concise output",
                "handler": "instruction_rewrite",
                "default_role": "user",
            }
        )
    )
    (templates / "chain_of_thought.yaml").write_text(
        yaml.dump(
            {
                "template": "Think step by step. {original}",
                "description": "Chain of thought prompting",
                "handler": "instruction_rewrite",
            }
        )
    )

    pools = tmp_path / "example_pools"
    pools.mkdir()
    (pools / "math_reasoning.yaml").write_text(
        yaml.dump(
            {
                "examples": ["2+2=4", "3*5=15", "sqrt(16)=4"],
                "description": "Math reasoning examples",
                "handler": "icl_example_selector",
                "default_count": 2,
                "default_strategy": "first",
            }
        )
    )
    return tmp_path


def test_prompt_store_get_template(prompt_dir):
    store = PromptStore(prompt_dir)
    content, params = store.get_raw(name="conciseness")
    assert "Be concise" in content
    assert params["default_role"] == "user"
    # description and handler are metadata, not in get_raw params
    assert "description" not in params
    assert "handler" not in params


def test_prompt_store_get_example_pool(prompt_dir):
    store = PromptStore(prompt_dir)
    content, params = store.get_raw(name="math_reasoning")
    assert isinstance(content, list)
    assert len(content) == 3
    assert params["default_count"] == 2
    assert params["default_strategy"] == "first"


def test_prompt_store_get_not_found(prompt_dir):
    store = PromptStore(prompt_dir)
    with pytest.raises(KeyError, match="nonexistent"):
        store.get_raw(name="nonexistent")


def test_prompt_store_get_requires_name(prompt_dir):
    store = PromptStore(prompt_dir)
    with pytest.raises(KeyError, match="requires"):
        store.get_raw()


def test_prompt_store_search(prompt_dir):
    store = PromptStore(prompt_dir)
    results = store.search("concise")
    assert len(results) == 1
    assert results[0]["name"] == "conciseness"
    assert results[0]["handler"] == "instruction_rewrite"


def test_prompt_store_search_by_description(prompt_dir):
    store = PromptStore(prompt_dir)
    results = store.search("math")
    assert len(results) == 1
    assert results[0]["name"] == "math_reasoning"
    assert results[0]["handler"] == "icl_example_selector"


def test_prompt_store_list_all(prompt_dir):
    store = PromptStore(prompt_dir)
    results = store.list_artifacts()
    assert len(results) == 3
    names = {r["name"] for r in results}
    assert names == {"conciseness", "chain_of_thought", "math_reasoning"}


def test_prompt_store_list_by_type(prompt_dir):
    store = PromptStore(prompt_dir)
    results = store.list_artifacts(artifact_type="template")
    assert len(results) == 2
    names = {r["name"] for r in results}
    assert names == {"conciseness", "chain_of_thought"}


def test_prompt_store_list_by_type_example_pool(prompt_dir):
    store = PromptStore(prompt_dir)
    results = store.list_artifacts(artifact_type="example_pool")
    assert len(results) == 1
    assert results[0]["name"] == "math_reasoning"


# --- AdapterStore ---


@pytest.fixture
def adapter_manifest(tmp_path):
    """Create a temporary adapter manifest."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "adapters": [
                    {
                        "name": "safety_lora",
                        "description": "LoRA adapter for safe generation",
                        "model": "granite",
                        "ref": "./adapters/safety_lora/",
                        "defaults": {"adapter_name": "safety"},
                    },
                    {
                        "name": "conciseness_lora",
                        "description": "LoRA for concise output",
                        "model": "llama",
                        "ref": "hf-hub/concise-lora",
                        "defaults": {"adapter_name": "conciseness"},
                    },
                ]
            }
        )
    )
    return manifest


def test_adapter_store_get(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    ref, params = store.get_raw(name="safety_lora")
    assert ref == "./adapters/safety_lora/"
    assert params["adapter_name"] == "safety"
    assert "description" not in params


def test_adapter_store_get_not_found(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    with pytest.raises(KeyError, match="nonexistent"):
        store.get_raw(name="nonexistent")


def test_adapter_store_get_requires_name(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    with pytest.raises(KeyError, match="requires"):
        store.get_raw()


def test_adapter_store_search(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    results = store.search("safe generation")
    assert len(results) == 1
    assert results[0]["name"] == "safety_lora"
    assert results[0]["default_params"]["adapter_name"] == "safety"


def test_adapter_store_search_by_model(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    results = store.search("lora", model="granite")
    assert len(results) == 1
    assert results[0]["name"] == "safety_lora"


def test_adapter_store_list_all(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    results = store.list_artifacts()
    assert len(results) == 2


def test_adapter_store_list_by_model(adapter_manifest):
    store = AdapterStore(adapter_manifest)
    results = store.list_artifacts(model="llama")
    assert len(results) == 1
    assert results[0]["name"] == "conciseness_lora"


# --- ModelStore ---


@pytest.fixture
def model_manifest(tmp_path):
    """Create a temporary model manifest."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "models": [
                    {
                        "name": "helpfulness_reward",
                        "description": "Reward model for helpful generation",
                        "model_family": "llama",
                        "ref": "./models/helpfulness_rm/",
                        "defaults": {"temperature": 1.0},
                    },
                    {
                        "name": "safety_reward",
                        "description": "Reward model for safe output",
                        "model_family": "granite",
                        "ref": "./models/safety_rm/",
                        "defaults": {"temperature": 0.8},
                    },
                ]
            }
        )
    )
    return manifest


def test_model_store_get(model_manifest):
    store = ModelStore(model_manifest)
    ref, params = store.get_raw(name="helpfulness_reward")
    assert ref == "./models/helpfulness_rm/"
    assert params["temperature"] == 1.0
    assert "description" not in params


def test_model_store_get_not_found(model_manifest):
    store = ModelStore(model_manifest)
    with pytest.raises(KeyError, match="nonexistent"):
        store.get_raw(name="nonexistent")


def test_model_store_get_requires_name(model_manifest):
    store = ModelStore(model_manifest)
    with pytest.raises(KeyError, match="requires"):
        store.get_raw()


def test_model_store_search(model_manifest):
    store = ModelStore(model_manifest)
    results = store.search("helpful")
    assert len(results) == 1
    assert results[0]["name"] == "helpfulness_reward"
    assert results[0]["default_params"]["temperature"] == 1.0


def test_model_store_search_by_model(model_manifest):
    store = ModelStore(model_manifest)
    results = store.search("reward", model="granite")
    assert len(results) == 1
    assert results[0]["name"] == "safety_reward"


def test_model_store_list_all(model_manifest):
    store = ModelStore(model_manifest)
    results = store.list_artifacts()
    assert len(results) == 2


def test_model_store_list_by_model_family(model_manifest):
    store = ModelStore(model_manifest)
    results = store.list_artifacts(model_family="llama")
    assert len(results) == 1
    assert results[0]["name"] == "helpfulness_reward"


# --- VectorStore ---


def test_vector_store_requires_xarray():
    """VectorStore raises ImportError if xarray is not available."""
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    pytest.importorskip("numpy")
    from mellea.steering.stores.vector_store import VectorStore

    assert VectorStore is not None


def test_vector_store_get_and_list(tmp_path):
    """Test VectorStore with a real zarr store."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    from mellea.steering.stores.vector_store import VectorStore

    hidden_dim = 8
    vectors = np.random.randn(1, 1, 3, hidden_dim).astype(np.float32)
    da = xr.DataArray(
        vectors,
        dims=["model", "behavior", "layer", "hidden"],
        coords={"model": ["granite"], "behavior": ["honesty"], "layer": [0, 1, 2]},
    )
    ds = xr.Dataset(
        {"vectors": da},
        attrs={
            "granite/honesty": {
                "description": "Honesty steering vector",
                "handler": "activation_steering",
                "default_layer": 0,
                "default_multiplier": 1.5,
                "transform": "additive",
            }
        },
    )
    store_path = tmp_path / "test_vectors.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)

    # get_raw returns dict[int, Tensor] and handler params
    torch = pytest.importorskip("torch")
    directions, params = store.get_raw(model="granite", behavior="honesty")
    assert isinstance(directions, dict)
    assert set(directions.keys()) == {0, 1, 2}
    for layer_idx, vec in directions.items():
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (hidden_dim,)
    assert params["default_multiplier"] == 1.5
    assert params["transform"] == "additive"
    assert params["default_layer"] == 0
    assert "description" not in params
    assert "handler" not in params

    # search returns handler and default_params as top-level keys
    results = store.search("honest")
    assert len(results) == 1
    assert results[0]["name"] == "granite/honesty"
    assert results[0]["handler"] == "activation_steering"
    assert results[0]["default_params"]["transform"] == "additive"

    # list_artifacts
    all_results = store.list_artifacts()
    assert len(all_results) == 1

    filtered = store.list_artifacts(model="granite")
    assert len(filtered) == 1

    empty = store.list_artifacts(model="nonexistent")
    assert len(empty) == 0


def test_vector_store_search_uses_tags(tmp_path):
    """Search matches against tags, not description boilerplate."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    from mellea.steering.stores.vector_store import VectorStore

    hidden_dim = 4
    vectors = np.zeros((1, 2, 2, hidden_dim), dtype=np.float32)
    da = xr.DataArray(
        vectors,
        dims=["model", "behavior", "layer", "hidden"],
        coords={
            "model": ["granite"],
            "behavior": ["formality", "warmth"],
            "layer": [0, 1],
        },
    )
    ds = xr.Dataset(
        {"vectors": da},
        attrs={
            "granite/formality": {
                "description": "Steers responses toward more formal phrasing.",
                "handler": "activation_steering",
                "tags": ["formal", "formality", "professional"],
            },
            "granite/warmth": {
                "description": "Steers responses toward more warm phrasing.",
                "handler": "activation_steering",
                "tags": ["warm", "warmth", "empathetic"],
            },
        },
    )
    store_path = tmp_path / "tags_test.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)

    # "formal" matches formality, not warmth
    results = store.search("formal")
    assert any(r["name"] == "granite/formality" for r in results)
    assert not any(r["name"] == "granite/warmth" for r in results)

    # "warm" matches warmth, not formality
    results = store.search("warm")
    assert any(r["name"] == "granite/warmth" for r in results)
    assert not any(r["name"] == "granite/formality" for r in results)

    # Full sentence: only the relevant vector matches
    results = store.search("The response should be warm and empathetic")
    assert any(r["name"] == "granite/warmth" for r in results)


def test_vector_store_get_raw_name_fallback(tmp_path):
    """get_raw accepts a composite name= selector, splitting on the last /."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from mellea.steering.stores.vector_store import VectorStore

    hidden_dim = 4
    vectors = np.ones((1, 1, 2, hidden_dim), dtype=np.float32)
    da = xr.DataArray(
        vectors,
        dims=["model", "behavior", "layer", "hidden"],
        coords={"model": ["org/model-name"], "behavior": ["honesty"], "layer": [0, 1]},
    )
    ds = xr.Dataset(
        {"vectors": da},
        attrs={
            "org/model-name/honesty": {
                "description": "Test vector",
                "handler": "activation_steering",
                "coeff": 1.0,
            }
        },
    )
    store_path = tmp_path / "name_fallback.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)

    # name= fallback splits on last /
    dirs_by_name, params_by_name = store.get_raw(name="org/model-name/honesty")
    dirs_explicit, params_explicit = store.get_raw(
        model="org/model-name", behavior="honesty"
    )

    assert set(dirs_by_name.keys()) == set(dirs_explicit.keys())
    for k in dirs_by_name:
        assert torch.equal(dirs_by_name[k], dirs_explicit[k])
    assert params_by_name == params_explicit


def test_vector_store_param_space_round_trip(tmp_path):
    """param_space round-trips through zarr attrs with string-key normalization."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    from mellea.steering.stores.vector_store import VectorStore

    hidden_dim = 4
    vectors = np.zeros((1, 1, 2, hidden_dim), dtype=np.float32)
    da = xr.DataArray(
        vectors,
        dims=["model", "behavior", "layer", "hidden"],
        coords={"model": ["granite"], "behavior": ["technicality"], "layer": [0, 1]},
    )
    # JSON serialization in zarr attrs turns int keys to strings.
    ds = xr.Dataset(
        {"vectors": da},
        attrs={
            "granite/technicality": {
                "description": "Technicality vector",
                "handler": "activation_steering",
                "default_multiplier": 1.4,
                "param_space": {
                    "by_layer": {
                        "0": {"multiplier": {"min": 0.8, "max": 1.8}},
                        "1": {"multiplier": {"min": 0.5, "max": 1.0}},
                    }
                },
            }
        },
    )
    store_path = tmp_path / "ps_vectors.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)

    # param_space should not leak into default_params
    _, params = store.get_raw(model="granite", behavior="technicality")
    assert "param_space" not in params
    assert params["default_multiplier"] == 1.4

    # search should return param_space with int keys normalized
    results = store.search("technicality")
    assert len(results) == 1
    ps = results[0]["param_space"]
    assert 0 in ps["by_layer"]
    assert 1 in ps["by_layer"]
    assert ps["by_layer"][0]["multiplier"]["max"] == 1.8

    # list_artifacts should also carry param_space
    all_results = store.list_artifacts()
    assert all_results[0]["param_space"]["by_layer"][1]["multiplier"]["min"] == 0.5


def test_vector_store_param_space_empty_when_absent(tmp_path):
    """Vectors without param_space in attrs get an empty dict."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    from mellea.steering.stores.vector_store import VectorStore

    vectors = np.zeros((1, 1, 1, 4), dtype=np.float32)
    da = xr.DataArray(
        vectors,
        dims=["model", "behavior", "layer", "hidden"],
        coords={"model": ["granite"], "behavior": ["honesty"], "layer": [0]},
    )
    ds = xr.Dataset(
        {"vectors": da},
        attrs={
            "granite/honesty": {
                "description": "Honesty vector",
                "handler": "activation_steering",
            }
        },
    )
    store_path = tmp_path / "no_ps.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)
    results = store.search("honesty")
    assert results[0]["param_space"] == {}


def test_vector_store_get_not_found(tmp_path):
    """VectorStore raises KeyError for missing vectors."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    from mellea.steering.stores.vector_store import VectorStore

    vectors = np.zeros((1, 1, 2, 4), dtype=np.float32)
    da = xr.DataArray(
        vectors,
        dims=["model", "behavior", "layer", "hidden"],
        coords={"model": ["granite"], "behavior": ["honesty"], "layer": [0, 1]},
    )
    ds = xr.Dataset({"vectors": da})
    store_path = tmp_path / "test.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)
    with pytest.raises(KeyError, match="No steering vector"):
        store.get_raw(model="granite", behavior="nonexistent")


# --- semantic_match ---


def test_semantic_match_with_embeddings():
    """semantic_match returns correct indices via cosine similarity."""
    candidates = [
        "formality: Steers responses toward more formal phrasing.",
        "warmth: Steers responses toward more warm, empathetic phrasing.",
        "technicality: Steers responses toward more technical, precise phrasing.",
        "sentiment: Steers responses toward more optimistic, positive phrasing.",
    ]

    matched = semantic_match(
        "The response should be formal and professional", candidates
    )
    assert 0 in matched
    assert 1 not in matched

    matched = semantic_match("The response should be warm and empathetic", candidates)
    assert 1 in matched
    assert 0 not in matched


def test_semantic_match_empty_candidates():
    """semantic_match returns empty list for empty candidates."""
    assert semantic_match("anything", []) == []
