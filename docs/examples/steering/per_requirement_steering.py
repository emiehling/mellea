# pytest: e2e, hf, qualitative
"""Automatic steering-vector discovery via PerRequirementComposer.

Requirements drive both artifact selection and LLM-as-a-Judge validation. 
Compares an unsteered baseline against a positive-sentiment steered run.

For manual, single-vector steering with explicit policies see
``activation_steering.py`` in this directory.
"""

from mellea import start_session
from mellea.core import Requirement
from mellea.core.steering import ControlCategory
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.steering import PerRequirementComposer
from mellea.steering import ArtifactLibrary, VectorStore, set_default_library

MODEL = "ibm-granite/granite-4.0-micro"
VECTORS_ZARR = "docs/examples/steering/library/vectors.zarr"
INSTRUCTION = "Should governments regulate artificial intelligence?"

# library setup
library = ArtifactLibrary()
library.register_store(ControlCategory.STATE, VectorStore(root=VECTORS_ZARR))
set_default_library(library)

available = library.list(ControlCategory.STATE, model=MODEL)
print("Available steering vectors:")
for info in available:
    print(f"- {info.name}")
print()

# session + composer
m = start_session(
    "hf", MODEL, model_options={"temperature": 0.3, "max_new_tokens": 400}
)
composer = PerRequirementComposer()
caps = m.backend.capabilities

# baseline; no requirements, no composer
result = m.instruct(INSTRUCTION, strategy=RejectionSamplingStrategy(loop_budget=5))
print(f"[baseline] {result}\n")

# positive; discovers the sentiment vector automatically
m.ctx = SimpleContext()
requirements = [
    Requirement("The response should convey a positive, optimistic perspective."),
]

policy = composer.compose(requirements, caps)
print(f"[positive] Discovered {len(policy.controls)} control(s):")
for ctrl in policy.controls:
    p = ctrl.params
    print(f"- {ctrl.artifact_ref} (layer={p['layer']}, multiplier={p['multiplier']})")

result = m.instruct(
    INSTRUCTION,
    requirements=requirements,
    strategy=RejectionSamplingStrategy(loop_budget=5),
    composer=composer,
)
print(f"[positive] {result}")
