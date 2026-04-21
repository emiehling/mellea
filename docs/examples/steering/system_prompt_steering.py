# pytest: e2e, hf, qualitative
"""Minimal example an input control and a simple output control.

This example is to illustrate that the composition operator is optional (no steering yields existing behavior), and that 
existing interventions (like changing the system prompt) can be easily self-contained into a single steering policy.
"""

from mellea import start_session
from mellea.core.steering import SteeringPolicy
from mellea.steering import (
    ArtifactLibrary,
    set_default_library,
    input_control,
    static_output_control,
)
from mellea.stdlib.steering import FixedComposer, NoOpComposer


MODEL = "ibm-granite/granite-4.0-micro"
PROMPT = "Explain what a mutex is to a new programmer."
TEMP = 0.8
MAX_NEW_TOKENS = 300
FIXED_OPTS = {"temperature": TEMP, "max_new_tokens": MAX_NEW_TOKENS}

# baseline; no composer
m_baseline = start_session("hf", MODEL, model_options=FIXED_OPTS)
baseline_output = m_baseline.instruct(PROMPT)

# explicit NoOpComposer
m_noop = start_session("hf", MODEL, model_options=FIXED_OPTS, composer=NoOpComposer())
noop_output = m_noop.instruct(PROMPT)

print("Baseline output:", baseline_output)
print("NoOp output:    ", noop_output)
assert str(baseline_output) == str(noop_output), (
    "NoOpComposer should produce identical output to no composer"
)
print("Unsteered behavior matches NoOpComposer behavior\n")

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
        temperature=TEMP,
        max_new_tokens=MAX_NEW_TOKENS
    ),
))

# start session with a fixed composer under the given steering policy
m = start_session("hf", MODEL, composer=FixedComposer(steering_policy))

# produce steered response
print(m.instruct(PROMPT))
