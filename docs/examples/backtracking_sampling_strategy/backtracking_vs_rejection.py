import asyncio
import time

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_options import ModelOption
from mellea.core import Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import BacktrackingSamplingStrategy, RejectionSamplingStrategy

MODEL_ID = "ibm-granite/granite-3.0-2b-instruct"
# MODEL_ID = "ibm-granite/granite-4.0-h-small"
# MODEL_ID = "ibm-granite/granite-4.0-micro"

TASK = "Explain how a computer processor executes instructions."
FORBIDDEN_WORDS = {"memory", "data", "register", "instruction", "cache"}
MODEL_OPTIONS = {
    ModelOption.MAX_NEW_TOKENS: 300,
    ModelOption.TEMPERATURE: 0.7
}
LOOP_BUDGET = 10

# requirement
no_forbidden_words = Requirement(
    description=f"Do not use any of these words: {', '.join(sorted(FORBIDDEN_WORDS))}",
    validation_fn=lambda ctx: (
        ValidationResult(result=False, reason=f"Forbidden: {[w for w in FORBIDDEN_WORDS if w in (ctx.last_output().value or '').lower()]}")
        if any(w in (ctx.last_output().value or "").lower() for w in FORBIDDEN_WORDS)
        else ValidationResult(result=True)
    ),
)

action = Instruction(TASK, requirements=[no_forbidden_words])


async def main():
    backend = LocalHFBackend(MODEL_ID)

    # rejection sampling
    rejection_strategy = RejectionSamplingStrategy(loop_budget=LOOP_BUDGET, requirements=[no_forbidden_words])

    t0 = time.perf_counter()
    rejection_result = await rejection_strategy.sample(
        action=action,
        context=ChatContext(),
        backend=backend,
        requirements=[no_forbidden_words],
        model_options=MODEL_OPTIONS,
        show_progress=False,
    )
    rejection_time = time.perf_counter() - t0

    # backtracking
    backtracking_strategy = BacktrackingSamplingStrategy(
        process_verifier=no_forbidden_words,
        backtrack_quota=15,
        backtrack_stride=10,
        check_cadence="token",
        max_token_resamples=5,
        redo_backtracked_with_argmax=False,
        loop_budget=LOOP_BUDGET,
        requirements=[no_forbidden_words],
    )

    t0 = time.perf_counter()
    backtracking_result = await backtracking_strategy.sample(
        action=action,
        context=ChatContext(),
        backend=backend,
        requirements=[no_forbidden_words],
        model_options=MODEL_OPTIONS,
    )
    backtracking_time = time.perf_counter() - t0

    # results
    backtracking_metrics = backtracking_result.result._meta.get("backtracking_metrics", {})
    rejection_chars = len(rejection_result.result.value or "")
    backtracking_chars = len(backtracking_result.result.value or "")

    print(f"\nRejectionSampling: "
          f"success={rejection_result.success}, "
          f"attempts={len(rejection_result.sample_generations)}, "
          f"time={rejection_time:.2f}s")
    print(f"BacktrackingSampling: "
          f"success={backtracking_result.success}, "
          f"attempts={len(backtracking_result.sample_generations)}, "
          f"time={backtracking_time:.2f}s, "
          f"backtracks={backtracking_metrics.get('num_backtracks', 0)}, "
          f"verifier_calls={backtracking_metrics.get('num_verifier_calls', 0)}, "
          f"resamples={backtracking_metrics.get('num_token_resamples', 0)}")

    print(f"\nRejectionSampling output ({rejection_chars} chars):\n{rejection_result.result.value}")
    print(f"\nBacktrackingSampling output ({backtracking_chars} chars):\n{backtracking_result.result.value}")


if __name__ == "__main__":
    asyncio.run(main())
