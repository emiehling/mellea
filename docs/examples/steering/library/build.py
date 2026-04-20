#!/usr/bin/env python3
"""Build a zarr-backed VectorStore from raw .svec and .yaml artifacts.

Scans ``artifacts/state/`` for model subdirectories (named with slugified
HuggingFace model IDs). Each subdirectory contains ``.svec`` / ``.yaml``
pairs — one pair per steering behavior.

Usage::

    python build.py  # build vectors.zarr in this directory
    python build.py --output /tmp/store.zarr  # custom output path

The ``.svec`` format is a JSON file::

    {
      "model_type": "granitemoehybrid",
      "directions": {
        "0": [[-0.038, 0.012, ...]],
        "1": [[-0.007, 0.045, ...]],
        ...
      }
    }

Each key is a layer index (as a string). Each value is a list containing one
direction vector; shape ``(1, hidden_dim)``.

The companion ``.yaml`` file contains metadata for the artifact library::

    model: "ibm-granite/granite-4.0-micro"
    description: "Steers toward technical phrasing."
    handler: activation_steering
    default_params:
      layer: 28
      multiplier: 1.25
      ...
    param_space:
      by_layer:
        28: {multiplier: {min: -1.25, max: 1.75}}

The ``model`` field in the YAML is the canonical HuggingFace model ID used as
a coordinate in the zarr store (the folder name (slugified) is only for the filesystem).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import xarray as xr
except ImportError:
    sys.exit("xarray is required: pip install xarray zarr")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "state"
DEFAULT_OUTPUT = Path(__file__).parent / "vectors.zarr"


def load_svec(path: Path) -> tuple[str, dict[int, np.ndarray]]:
    """Load a .svec file and return (model_type, directions).

    Args:
        path: Path to the .svec JSON file.

    Returns:
        A tuple of (model_type, directions) where directions maps integer
        layer indices to numpy arrays of shape (hidden_dim,).
    """
    with open(path) as f:
        data = json.load(f)

    model_type = data["model_type"]
    directions: dict[int, np.ndarray] = {}
    for layer_str, vecs in data["directions"].items():
        # vecs is [[float, ...]] — shape (1, hidden_dim). Take the first.
        directions[int(layer_str)] = np.array(vecs[0], dtype=np.float32)

    return model_type, directions


def load_metadata(path: Path) -> dict[str, Any]:
    """Load a .yaml metadata file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed YAML as a dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def discover_artifacts(
    root: Path,
) -> list[tuple[str, str, dict[int, np.ndarray], dict[str, Any]]]:
    """Scan artifact directories for .svec/.yaml pairs.

    Args:
        root: The ``artifacts/state/`` directory.

    Returns:
        A list of (model, behavior, directions, metadata) tuples.
    """
    artifacts = []

    if not root.is_dir():
        return artifacts

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        for svec_path in sorted(model_dir.glob("*.svec")):
            behavior = svec_path.stem
            yaml_path = svec_path.with_suffix(".yaml")

            if not yaml_path.exists():
                print(f"  warning: {svec_path.name} has no companion .yaml, skipping")
                continue

            metadata = load_metadata(yaml_path)
            model = metadata.get("model")
            if model is None:
                print(
                    f"  warning: {yaml_path.name} missing 'model' field, skipping"
                )
                continue

            _model_type, directions = load_svec(svec_path)
            artifacts.append((model, behavior, directions, metadata))
            print(f"  found: {model} / {behavior} ({len(directions)} layers)")

    return artifacts


def build_zarr(
    artifacts: list[tuple[str, str, dict[int, np.ndarray], dict[str, Any]]],
    output: Path,
) -> None:
    """Build a zarr store from discovered artifacts.

    All artifacts must share the same layer count and hidden dimension.

    Args:
        artifacts: List of (model, behavior, directions, metadata) tuples.
        output: Path to write the zarr directory store.
    """
    if not artifacts:
        sys.exit("No artifacts found. Nothing to build.")

    # Determine dimensions from the first artifact.
    _model0, _beh0, dirs0, _meta0 = artifacts[0]
    num_layers = max(dirs0.keys()) + 1
    hidden_dim = next(iter(dirs0.values())).shape[0]
    print(f"\nDimensions: {num_layers} layers × {hidden_dim} hidden_dim")

    # Collect unique models and behaviors (preserving insertion order).
    models: list[str] = []
    behaviors: list[str] = []
    for model, behavior, _, _ in artifacts:
        if model not in models:
            models.append(model)
        if behavior not in behaviors:
            behaviors.append(behavior)

    # Build the data array: (models, behaviors, layers, hidden_dim).
    data = np.zeros(
        (len(models), len(behaviors), num_layers, hidden_dim), dtype=np.float32
    )
    attrs: dict[str, Any] = {}

    for model, behavior, directions, metadata in artifacts:
        mi = models.index(model)
        bi = behaviors.index(behavior)

        for layer_idx, vec in directions.items():
            if vec.shape[0] != hidden_dim:
                sys.exit(
                    f"Dimension mismatch: {model}/{behavior} layer {layer_idx} "
                    f"has dim {vec.shape[0]}, expected {hidden_dim}"
                )
            data[mi, bi, layer_idx, :] = vec

        # Store metadata in attrs, keyed by "model/behavior".
        meta_key = f"{model}/{behavior}"
        meta_for_attrs = {
            k: v
            for k, v in metadata.items()
            if k not in ("model", "default_params")
        }
        # Flatten default_params to top level
        if "default_params" in metadata:
            meta_for_attrs.update(metadata["default_params"])
        attrs[meta_key] = meta_for_attrs

    ds = xr.Dataset(
        {
            "vectors": xr.DataArray(
                data,
                dims=["model", "behavior", "layer", "hidden_dim"],
                coords={
                    "model": models,
                    "behavior": behaviors,
                },
            )
        },
        attrs=attrs,
    )

    # Remove existing store if present.
    if output.exists():
        import shutil

        shutil.rmtree(output)

    ds.to_zarr(output)
    print(f"\nWrote {output}/ ({len(models)} models, {len(behaviors)} behaviors)")


def main():
    parser = argparse.ArgumentParser(
        description="Build a zarr VectorStore from .svec/.yaml artifact pairs."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ARTIFACTS_DIR,
        help=f"Root directory to scan (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output zarr path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print(f"Scanning {args.artifacts}/")
    artifacts = discover_artifacts(args.artifacts)
    build_zarr(artifacts, args.output)


if __name__ == "__main__":
    main()