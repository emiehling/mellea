# pytest: e2e, hf, qualitative
"""A steering PoC that combines a state control, via a pretrained steering vector, with a given input and output control.

reads default_params and param_space from the zarr store.

Builds a three-control policy (INPUT + STATE + OUTPUT) using the technicality
vector's default layer/coefficient from default_params, then re-runs at
the max coefficient from param_space to show the envelope edge.
"""

from mellea import start_session
from mellea.core.steering import ControlCategory, SteeringPolicy
from mellea.steering import (
    ArtifactLibrary,
    VectorStore,
    get_default_library,
    input_control,
    set_default_library,
    state_control,
    static_output_control,
)
from mellea.stdlib.steering import FixedComposer

VECTORS_ZARR = "docs/examples/steering_artifacts/vectors.zarr"
ARTIFACT_REF = "granite-4-micro/technicality"
INSTRUCTION = "Explain what a mutex is to a new programmer."

library = ArtifactLibrary()
library.register_store(ControlCategory.STATE, VectorStore(root=VECTORS_ZARR))
set_default_library(library)

# Pull metadata (default_params, param_space) from the store.
[info] = get_default_library().list(ControlCategory.STATE, model="granite-4-micro")
defaults = info.default_params
by_layer = info.param_space["by_layer"]


def build_policy(layers: list[int], coefficient: float) -> SteeringPolicy:
    return SteeringPolicy(
        controls=(
            input_control(
                name="system_prompt_injection",
                params={"system_prompt": "Be terse. Under 60 words."},
            ),
            state_control(
                name="activation_steering",
                artifact_ref=ARTIFACT_REF,
                layers=layers,
                params={"coefficient": coefficient},
            ),
            static_output_control(
                name="static_output", 
                temperature=0.3, 
                max_new_tokens=120
            ),
        )
    )


default_layers = defaults["default_layers"]
default_coef = defaults["default_coefficient"]

# Run 1: default configuration from default_params.
m = start_session(
    "hf", "granite-4-micro",
    composer=FixedComposer(build_policy(default_layers, default_coef)),
)
print(f"[default coef={default_coef}] {m.instruct(INSTRUCTION)}")

# Run 2: push to the envelope's max coefficient across the default layers.
max_coef = max(by_layer[layer]["coefficient"]["max"] for layer in default_layers)
m = start_session(
    "hf", "granite-4-micro",
    composer=FixedComposer(build_policy(default_layers, max_coef)),
)
print(f"[max coef={max_coef}] {m.instruct(INSTRUCTION)}")