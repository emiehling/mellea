"""Backtracking Sampling Strategy.

This module implements a sampling strategy that performs bounded backtracking and local resampling during decoding,
rather than regenerating from scratch after validation failures. This approach reuses already-computed prefix state
(KV cache), avoiding repeated prompt prefill costs.

The design is inspired by "verifier-assisted generation" (Botta et al. 2025, arXiv:2502.12123).
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from ...core import (
    Backend,
    BaseModelSubclass,
    Component,
    Context,
    FancyLogger,
    ModelOutputThunk,
    Requirement,
    S,
    SamplingResult,
    SamplingStrategy,
    ValidationResult,
)
from ...stdlib import functional as mfuncs
from ..context import SimpleContext

if TYPE_CHECKING:
    from ...backends.huggingface import LocalHFBackend


@dataclasses.dataclass
class BacktrackingMetrics:
    """Metrics collected during backtracking generation."""

    num_backtracks: int = 0
    total_backtracked_tokens: int = 0
    num_verifier_calls: int = 0
    num_token_resamples: int = 0
    tokens_generated_final: int = 0


@dataclasses.dataclass
class BacktrackingConfig:
    """Configuration for backtracking generation.

    Attributes:
        backtrack_quota: Maximum backtrack events per generation.
        backtrack_stride: Tokens to remove per backtrack.
        check_cadence: When to check: "token", "newline", or "boundary".
        verifier_threshold: Threshold for score-based verifiers.
        max_token_resamples: Local resampling attempts from same logits.
        redo_backtracked_with_argmax: Force greedy after backtrack.
        argmax_length: Tokens to force greedy (defaults to stride).
    """

    backtrack_quota: int = 5
    backtrack_stride: int = 4
    check_cadence: Literal["token", "newline", "boundary"] = "newline"
    verifier_threshold: float = 0.5
    max_token_resamples: int = 3
    redo_backtracked_with_argmax: bool = True
    argmax_length: int | None = None


class BacktrackingSamplingStrategy(SamplingStrategy):
    """Sampling strategy with bounded backtracking and local resampling.

    This strategy performs bounded backtracking and local resampling during decoding. The mainn benefit is that it
    backtracks only to the problematic prefix and resamples, reducing compute for constrained generation.

    Requires a LocalHFBackend for backtracking functionality. For other backends, falls back to standard generation
    (unless require_hf_backend=True).

    """

    def __init__(
        self,
        *,
        process_verifier: Requirement | None = None,
        backtrack_quota: int = 5,
        backtrack_stride: int = 4,
        check_cadence: Literal["token", "newline", "boundary"] = "newline",
        verifier_threshold: float = 0.5,
        max_token_resamples: int = 3,
        redo_backtracked_with_argmax: bool = True,
        argmax_length: int | None = None,
        loop_budget: int = 1,
        requirements: list[Requirement] | None = None,
        require_hf_backend: bool = False,
    ):
        """Initialize the backtracking sampling strategy.

        Args:
            process_verifier: Optional prefix-time verifier (uses Requirement interface).
                This verifier is called during generation to validate prefixes.
            backtrack_quota: Maximum backtrack events per generation.
            backtrack_stride: Number of tokens to remove per backtrack.
            check_cadence: When to run the verifier:
                - "token": Check after every token (most expensive)
                - "newline": Check after newline characters (balanced)
                - "boundary": Check at sentence boundaries
            verifier_threshold: Threshold for score-based verifiers. Higher = better.
            max_token_resamples: Local resampling attempts from same logits.
            redo_backtracked_with_argmax: Force greedy decoding after backtrack.
            argmax_length: Number of tokens to force greedy (defaults to stride).
            loop_budget: Outer retry loop budget (standard Mellea pattern).
            requirements: Final requirements for validation after generation.
            require_hf_backend: If True, raise TypeError for non-HF backends.
                If False, fall back to standard generation.

        Raises:
            ValueError: If backtrack_quota <= 0 or loop_budget <= 0.
        """
        if backtrack_quota <= 0:
            raise ValueError("backtrack_quota must be greater than 0")
        if loop_budget <= 0:
            raise ValueError("loop_budget must be greater than 0")
        if backtrack_stride <= 0:
            raise ValueError("backtrack_stride must be greater than 0")

        self.process_verifier = process_verifier
        self.backtrack_quota = backtrack_quota
        self.backtrack_stride = backtrack_stride
        self.check_cadence = check_cadence
        self.verifier_threshold = verifier_threshold
        self.max_token_resamples = max_token_resamples
        self.redo_backtracked_with_argmax = redo_backtracked_with_argmax
        self.argmax_length = argmax_length if argmax_length is not None else backtrack_stride
        self.loop_budget = loop_budget
        self.requirements = requirements
        self.require_hf_backend = require_hf_backend

    def _create_prefix_context(
        self, prefix_text: str, original_ctx: Context  # noqa: ARG002
    ) -> Context:
        """Create a temporary context for process verifier evaluation.

        Args:
            prefix_text: The current prefix text to evaluate.
            original_ctx: The original context (reserved for future use).

        Returns:
            A SimpleContext with the prefix as the last output.
        """
        # Note: original_ctx is kept for potential future use where we may
        # want to preserve some state from the original context
        temp_ctx = SimpleContext()
        temp_output = ModelOutputThunk(value=prefix_text)
        return temp_ctx.add(temp_output)

    async def _evaluate_process_verifier(
        self,
        prefix_text: str,
        backend: Backend,
        original_ctx: Context,
        model_options: dict | None = None,
    ) -> tuple[bool, BacktrackingMetrics]:
        """Evaluate the process verifier on a prefix.

        This is the single source of truth for verifier evaluation logic. The backend receives a callback built from
        this method (via ``_generate_with_backtracking_internal``) instead of reimplementing the evaluation protocol.

        Args:
            prefix_text: The current prefix text to evaluate.
            backend: The backend for validation.
            original_ctx: The original context.
            model_options: Model options for validation.

        Returns:
            Tuple of (passed, updated_metrics).
        """
        if self.process_verifier is None:
            return True, BacktrackingMetrics(num_verifier_calls=0)

        metrics = BacktrackingMetrics(num_verifier_calls=1)
        temp_ctx = self._create_prefix_context(prefix_text, original_ctx)

        val_result = await self.process_verifier.validate(
            backend, temp_ctx, model_options=model_options
        )

        # handle both boolean and score-based verifiers
        if val_result.score is not None:
            passed = val_result.score >= self.verifier_threshold
        else:
            passed = val_result.as_bool()

        return passed, metrics

    def _is_hf_backend(self, backend: Backend) -> bool:
        """Check if the backend is a LocalHFBackend."""
        # avoid circular imports (and handle case where HF backend deepndencies aren't installed)
        # todo: make backtracking work under vllm
        try:
            from ...backends.huggingface import LocalHFBackend

            return isinstance(backend, LocalHFBackend)
        except ImportError:
            # HF backend dependencies not available
            return False

    async def _generate_with_backtracking_internal(
        self,
        action: Component,
        context: Context,
        backend: "LocalHFBackend",
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk, Context, BacktrackingMetrics]:
        """Internal backtracking generation implementation.

        Builds a ``verify_prefix`` callback from the process verifier and delegates the actual decode loop to the
        backend. This keeps verifier evaluation protocol in the strategy layer and decoding mechanics in the backend.

        Args:
            action: The action component.
            context: The generation context.
            backend: The LocalHFBackend for generation.
            format: Output format for structured outputs.
            model_options: Model options.
            tool_calls: Whether to use tool calls.

        Returns:
            Tuple of (result, context, metrics).
        """
        # create the backtracking config
        config = BacktrackingConfig(
            backtrack_quota=self.backtrack_quota,
            backtrack_stride=self.backtrack_stride,
            check_cadence=self.check_cadence,
            verifier_threshold=self.verifier_threshold,
            max_token_resamples=self.max_token_resamples,
            redo_backtracked_with_argmax=self.redo_backtracked_with_argmax,
            argmax_length=self.argmax_length,
        )

        # Build verify_prefix callback: wraps _evaluate_process_verifier so the backend receives a simple
        # async str -> bool callable and doesn't need to know about Requirement/threshold/context protocol.
        verify_prefix = None
        if self.process_verifier is not None:
            async def verify_prefix(prefix_text: str) -> bool:
                passed, _ = await self._evaluate_process_verifier(
                    prefix_text=prefix_text,
                    backend=backend,
                    original_ctx=context,
                    model_options=model_options,
                )
                return passed

        # delegate to backend's backtracking generation
        result, result_ctx, metrics = await backend._generate_with_backtracking(
            action=action,
            ctx=context,
            config=config,
            verify_prefix=verify_prefix,
            format=format,
            model_options=model_options,
            tool_calls=tool_calls,
        )

        return result, result_ctx, metrics

    async def sample(
        self,
        action: Component[S],
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]:
        """Sample with backtracking support.

        This method generates output using backtracking when a LocalHFBackend is available, otherwise falls back to standard generation.

        Args:
            action: The action component to sample.
            context: The session context.
            backend: The backend for generation.
            requirements: Requirements to validate against after generation.
            validation_ctx: Optional separate validation context.
            format: Output format for structured outputs.
            model_options: Model options to pass to backend.
            tool_calls: True if tool calls should be used.

        Returns:
            SamplingResult with success status and generation history.

        Raises:
            TypeError: If require_hf_backend=True and backend is not LocalHFBackend.
        """
        flog = FancyLogger.get_logger()

        # check backend type
        is_hf = self._is_hf_backend(backend)
        if not is_hf:
            if self.require_hf_backend:
                raise TypeError(
                    f"BacktrackingSamplingStrategy requires LocalHFBackend, got {type(backend).__name__}. "
                    f"Set require_hf_backend=False to fall back to standard generation."
                )
            flog.warning(
                f"BacktrackingSamplingStrategy: Backend {type(backend).__name__} does not support backtracking. "
                f"Falling back to standard generation."
            )

        # merge requirements
        reqs: list[Requirement] = []
        if self.requirements is not None:
            reqs.extend(self.requirements)
        if requirements is not None:
            reqs.extend(requirements)
        reqs = list(set(reqs))

        # state tracking
        sampled_results: list[ModelOutputThunk] = []
        sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
        sampled_actions: list[Component] = []
        sample_contexts: list[Context] = []

        # outer retry loop
        for loop_idx in range(self.loop_budget):
            flog.info(
                f"BacktrackingSamplingStrategy: Loop {loop_idx + 1}/{self.loop_budget}"
            )

            metrics = BacktrackingMetrics()

            if is_hf and self.process_verifier is not None:
                # use backtracking generation
                from ...backends.huggingface import LocalHFBackend

                assert isinstance(backend, LocalHFBackend)
                result, result_ctx, metrics = await self._generate_with_backtracking_internal(
                    action=deepcopy(action),
                    context=context,
                    backend=backend,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )
            else:
                # fall back to standard generation
                result, result_ctx = await backend.generate_from_context(
                    deepcopy(action),
                    ctx=context,
                    format=format,
                    model_options=model_options,
                    tool_calls=tool_calls,
                )
                await result.avalue()

            # store backtracking metrics in result metadata
            result._meta["backtracking_metrics"] = dataclasses.asdict(metrics)

            # parse result
            result.parsed_repr = action.parse(result)

            # validate against final requirements
            validation_target_ctx = validation_ctx if validation_ctx is not None else result_ctx
            val_scores = await mfuncs.avalidate(
                reqs=reqs,
                context=validation_target_ctx,
                backend=backend,
                output=result,
                format=None,
                model_options=model_options,
            )
            constraint_scores = list(zip(reqs, val_scores))

            # collect data
            sampled_results.append(result)
            sampled_scores.append(constraint_scores)
            sampled_actions.append(action)
            sample_contexts.append(result_ctx)

            # check success
            if all(bool(s[1]) for s in constraint_scores):
                flog.info("BacktrackingSamplingStrategy: SUCCESS")
                assert result._generate_log is not None
                result._generate_log.is_final_result = True

                return SamplingResult(
                    result_index=len(sampled_results) - 1,
                    success=True,
                    sample_generations=sampled_results,
                    sample_validations=sampled_scores,
                    sample_contexts=sample_contexts,
                    sample_actions=sampled_actions,
                )

            # log failure
            count_valid = len([s for s in constraint_scores if bool(s[1])])
            flog.info(
                f"BacktrackingSamplingStrategy: FAILED. "
                f"Valid: {count_valid}/{len(constraint_scores)}, "
                f"Backtracks: {metrics.num_backtracks}"
            )

        # all loops finished; return best attempt (last one)
        flog.info(
            f"BacktrackingSamplingStrategy: Loop budget exhausted after "
            f"{len(sampled_results)} attempts."
        )

        best_idx = -1  # return last attempt
        assert sampled_results[best_idx]._generate_log is not None
        sampled_results[best_idx]._generate_log.is_final_result = True

        return SamplingResult(
            result_index=best_idx,
            success=False,
            sample_generations=sampled_results,
            sample_validations=sampled_scores,
            sample_actions=sampled_actions,
            sample_contexts=sample_contexts,
        )
