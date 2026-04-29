"""Unit tests for the VLLMSteeringRequestHandler and tensor serialization."""

import base64

import pytest

from mellea.core.steering import Control, ControlCategory


def _torch():
    return pytest.importorskip("torch")


def _zstd():
    return pytest.importorskip("zstandard")


# --- Tensor serialization roundtrip ---


def test_serialize_roundtrip_float32():
    torch = _torch()
    zstandard = _zstd()
    from mellea.steering.handlers.remote import _serialize_tensor

    t = torch.randn(2, 8, dtype=torch.float32)
    d = _serialize_tensor(t)
    raw = zstandard.ZstdDecompressor().decompress(base64.b64decode(d["data"]))
    arr = torch.frombuffer(raw, dtype=torch.float32).reshape(d["shape"])
    assert torch.allclose(arr, t)
    assert d["dtype"] == "float32"
    assert d["original_dtype"] == "torch.float32"
    assert d["compression"] == "zstd"


def test_serialize_bfloat16_view_trick():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers.remote import _serialize_tensor

    t = torch.randn(4, dtype=torch.bfloat16)
    d = _serialize_tensor(t)
    assert d["dtype"] == "int16"
    assert d["original_dtype"] == "torch.bfloat16"
    assert d["shape"] == [4]


# --- Handler behavior ---


def test_handler_single_layer():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    artifact = {15: torch.randn(4096), 16: torch.randn(4096)}
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layer": 15, "multiplier": 1.5},
    )
    handler = VLLMSteeringRequestHandler()
    rk = handler.contribute_to_request(control, {}, artifact)

    sv_list = rk["extra_body"]["extra_args"]["apply_steering_vectors"]
    assert len(sv_list) == 1
    sv = sv_list[0]
    assert sv["layer_indices"] == [15]
    assert sv["scale"] == 1.5
    assert sv["norm_match"] is False
    assert sv["position_indices"] is None
    assert sv["activations"]["shape"] == [1, 4096]


def test_handler_all_layers():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    artifact = {15: torch.randn(4096), 16: torch.randn(4096)}
    control = Control(
        category=ControlCategory.STATE, name="activation_steering", params={}
    )
    handler = VLLMSteeringRequestHandler()
    rk = handler.contribute_to_request(control, {}, artifact)

    sv = rk["extra_body"]["extra_args"]["apply_steering_vectors"][0]
    assert sv["layer_indices"] == [15, 16]
    assert sv["activations"]["shape"] == [2, 4096]


def test_handler_appends_to_existing_list():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    artifact = {15: torch.randn(4096)}
    handler = VLLMSteeringRequestHandler()
    rk = {
        "extra_body": {"extra_args": {"apply_steering_vectors": [{"existing": True}]}}
    }
    rk = handler.contribute_to_request(
        Control(
            category=ControlCategory.STATE,
            name="activation_steering",
            params={"layer": 15},
        ),
        rk,
        artifact,
    )
    vectors = rk["extra_body"]["extra_args"]["apply_steering_vectors"]
    assert len(vectors) == 2
    assert vectors[0] == {"existing": True}
    assert vectors[1]["layer_indices"] == [15]


def test_handler_norm_match_and_token_positions():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    artifact = {15: torch.randn(4096)}
    control = Control(
        category=ControlCategory.STATE,
        name="activation_steering",
        params={"layer": 15, "norm_match": True, "token_positions": [-1, -2]},
    )
    handler = VLLMSteeringRequestHandler()
    rk = handler.contribute_to_request(control, {}, artifact)

    sv = rk["extra_body"]["extra_args"]["apply_steering_vectors"][0]
    assert sv["norm_match"] is True
    assert sv["position_indices"] == [-1, -2]


def test_handler_missing_layer_raises():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    artifact = {15: torch.randn(4096)}
    control = Control(
        category=ControlCategory.STATE, name="activation_steering", params={"layer": 99}
    )
    with pytest.raises(KeyError):
        VLLMSteeringRequestHandler().contribute_to_request(control, {}, artifact)


def test_handler_rejects_non_dict_artifact():
    _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    control = Control(
        category=ControlCategory.STATE, name="activation_steering", params={}
    )
    with pytest.raises(ValueError):
        VLLMSteeringRequestHandler().contribute_to_request(control, {}, None)


def test_handler_default_multiplier_is_one():
    torch = _torch()
    _zstd()
    from mellea.steering.handlers import VLLMSteeringRequestHandler

    artifact = {15: torch.randn(4096)}
    control = Control(
        category=ControlCategory.STATE, name="activation_steering", params={"layer": 15}
    )
    rk = VLLMSteeringRequestHandler().contribute_to_request(control, {}, artifact)
    sv = rk["extra_body"]["extra_args"]["apply_steering_vectors"][0]
    assert sv["scale"] == 1.0
