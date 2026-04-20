# pytest: e2e, hf, qualitative
"""Automatic steering-vector discovery via PerRequirementComposer.

Demonstrates how PerRequirementComposer inspects requirement descriptions,
searches the artifact library for matching pretrained steering vectors, and
assembles a steering policy automatically — no manual Control construction
needed.

Each requirement serves two roles simultaneously:

1. **Steering selector** — the PerRequirementComposer tokenizes the
   requirement description, matches content words against artifact names
   and descriptions in the library, and builds a steering policy from the
   discovered vectors.
2. **LLM-as-a-Judge prompt** — the same description is included in the
   rendered instruction and used by the sampling strategy for validation.

Three scenarios compare identical prompts under different steering conditions:

1. **Baseline** — no requirements, no composer (unsteered generation).
2. **Technical + Formal** — discovers the *technicality* and *formality*
   vectors; validates that the output matches those qualities.
3. **Warm + Positive** — discovers the *warmth* and *sentiment* vectors;
   validates accordingly.

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
    print(f"  {info.name}")
print()

# session + composer
m = start_session(
    "hf", MODEL, model_options={"temperature": 0.3, "max_new_tokens": 400}
)
composer = PerRequirementComposer()
caps = m.backend.capabilities


def run_scenario(label: str, requirements: list[Requirement] | None = None) -> None:
    """Run a single generation scenario and print the result."""
    m.ctx = SimpleContext()

    if requirements:
        policy = composer.compose(requirements, caps)
        print(f"[{label}] Discovered {len(policy.controls)} control(s):")
        for ctrl in policy.controls:
            p = ctrl.params
            print(
                f"    {ctrl.artifact_ref}  (layer={p['layer']}, multiplier={p['multiplier']})"
            )

        result = m.instruct(
            INSTRUCTION,
            requirements=requirements,
            strategy=RejectionSamplingStrategy(loop_budget=5),
            composer=composer,
        )
    else:
        result = m.instruct(
            INSTRUCTION, strategy=RejectionSamplingStrategy(loop_budget=5)
        )

    print(f"[{label}] {result}")
    print()


run_scenario("baseline")

run_scenario("technical + formal", [
    Requirement("The response should use technical, precise language."),
    Requirement("The response should be written in a formal tone."),
])

run_scenario("warm + positive", [
    Requirement("The response should be warm and empathetic."),
    Requirement("The response should convey a positive, optimistic perspective."),
])