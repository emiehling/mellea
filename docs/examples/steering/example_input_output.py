# pytest: e2e, hf, qualitative
"""Minimal steering PoC consisting of an input control (system prompt instruction) and a simple output control (gen kwargs).

This example is to illustrate that existing interventions can be easily self-contained into a single steering policy.
"""

from mellea import start_session
from mellea.core.steering import ControlCategory, SteeringPolicy
from mellea.steering import (
    ArtifactLibrary,
    set_default_library,
    input_control,
    static_output_control,
)
from mellea.stdlib.steering import FixedComposer

# create artifact library and register controls
library = ArtifactLibrary()
set_default_library(library)

# define steering policy
steering_policy = SteeringPolicy(controls=(
    input_control(
        name="system_prompt_injection",
        params={
            "system_prompt": "Answer in one or two sentences."
        },
    ),
    static_output_control(
        name="static_output", 
        temperature=0.8, 
        max_new_tokens=400
    ),
))

# start session with a fixed composer under the given steering policy
m = start_session("hf", "ibm-granite/granite-4.0-micro", composer=FixedComposer(steering_policy))

# produce steered response
print(m.instruct("Explain what a mutex is to a new programmer."))
