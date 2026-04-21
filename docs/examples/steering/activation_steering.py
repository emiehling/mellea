# pytest: e2e, hf, qualitative
"""Example that combines a state control, via a precomputed steering vector, with a given input and output control.

Builds a three-control policy (INPUT + STATE + OUTPUT) using the technicality vector's default layer/multiplier from 
default_params, then re-runs at the max multiplier from param_space to show maximally steered behavior.
"""

from mellea import start_session
from mellea.core.steering import ControlCategory, SteeringPolicy
from mellea.stdlib.steering import FixedComposer
from mellea.steering import (
    ArtifactLibrary,
    VectorStore,
    get_default_library,
    input_control,
    set_default_library,
    state_control,
    static_output_control,
)

MODEL = "ibm-granite/granite-4.0-micro"

VECTORS_ZARR = "docs/examples/steering/library/vectors.zarr"
ARTIFACT_REF = "ibm-granite/granite-4.0-micro/technicality"
INSTRUCTION = "Explain what a mutex is to a new programmer."

# build library; register steering vectors
library = ArtifactLibrary()
library.register_store(ControlCategory.STATE, VectorStore(root=VECTORS_ZARR))
set_default_library(library)

# pull metadata (default_params, param_space) from the store
infos = get_default_library().list(ControlCategory.STATE, model="ibm-granite/granite-4.0-micro")
info = next(i for i in infos if i.name.endswith("/technicality"))


def build_policy(layer: int, multiplier: float) -> SteeringPolicy:
    return SteeringPolicy(
        controls=(
            input_control(
                name="system_prompt_injection",
                params={"system_prompt": "Be concise."},
            ),
            state_control(
                name="activation_steering",
                artifact_ref=ARTIFACT_REF,
                layer=layer,
                params={"multiplier": multiplier},
            ),
            static_output_control(
                name="static_output", temperature=0.3, max_new_tokens=300
            ),
        )
    )


defaults = info.default_params
default_layer = defaults["layer"]

# no steering
multiplier = 0.0
m = start_session(
    "hf",
    MODEL,
    composer=FixedComposer(build_policy(default_layer, multiplier)),
)
print(f"[zeroed mulitiplier={multiplier}] {m.instruct(INSTRUCTION)}")

# default configuration from default_params
default_multiplier = defaults["multiplier"]
m = start_session(
    "hf",
    MODEL,
    composer=FixedComposer(build_policy(default_layer, default_multiplier)),
)
print(f"[default multiplier={default_multiplier}] {m.instruct(INSTRUCTION)}")

# push to the max multiplier (edge of calibration region)
by_layer = info.param_space["by_layer"]
max_multiplier = by_layer[default_layer]["multiplier"]["max"]
m = start_session(
    "hf",
    MODEL,
    composer=FixedComposer(build_policy(default_layer, max_multiplier)),
)
print(f"[max multiplier={max_multiplier}] {m.instruct(INSTRUCTION)}")
