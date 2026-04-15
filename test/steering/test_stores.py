"""Unit tests for per-category artifact stores."""

from __future__ import annotations

import pytest
import yaml

from mellea.steering.stores.adapter_store import AdapterStore
from mellea.steering.stores.base import ArtifactStore
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
    results = store.search("safe")
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
                "default_layers": [0, 1],
                "default_coefficient": 1.5,
                "transform": "additive",
            }
        },
    )
    store_path = tmp_path / "test_vectors.zarr"
    ds.to_zarr(store_path)

    store = VectorStore(store_path)

    # get_raw returns only handler params (no description/handler)
    tensor, params = store.get_raw(model="granite", behavior="honesty")
    assert tensor.shape == (3, hidden_dim)
    assert params["default_coefficient"] == 1.5
    assert params["transform"] == "additive"
    assert params["default_layers"] == [0, 1]
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
