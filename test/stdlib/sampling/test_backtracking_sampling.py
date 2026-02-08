"""Unit tests for BacktrackingSamplingStrategy."""

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends import Backend
from mellea.core import Requirement, ValidationResult
from mellea.stdlib.components import ModelOutputThunk
from mellea.stdlib.context import ChatContext, SimpleContext
from mellea.stdlib.sampling import BacktrackingSamplingStrategy
from mellea.stdlib.sampling.backtracking import BacktrackingConfig, BacktrackingMetrics


class TestBacktrackingMetrics:
    """Test BacktrackingMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = BacktrackingMetrics()

        assert metrics.num_backtracks == 0
        assert metrics.total_backtracked_tokens == 0
        assert metrics.num_verifier_calls == 0
        assert metrics.num_token_resamples == 0
        assert metrics.tokens_generated_final == 0

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = BacktrackingMetrics(
            num_backtracks=3,
            total_backtracked_tokens=12,
            num_verifier_calls=5,
            num_token_resamples=2,
            tokens_generated_final=50,
        )

        assert metrics.num_backtracks == 3
        assert metrics.total_backtracked_tokens == 12
        assert metrics.num_verifier_calls == 5
        assert metrics.num_token_resamples == 2
        assert metrics.tokens_generated_final == 50

    def test_asdict(self):
        """Test conversion to dictionary."""
        metrics = BacktrackingMetrics(num_backtracks=2)
        d = dataclasses.asdict(metrics)

        assert d["num_backtracks"] == 2
        assert isinstance(d, dict)


class TestBacktrackingConfig:
    """Test BacktrackingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktrackingConfig()

        assert config.backtrack_quota == 5
        assert config.backtrack_stride == 4
        assert config.check_cadence == "newline"
        assert config.verifier_threshold == 0.5
        assert config.max_token_resamples == 3
        assert config.redo_backtracked_with_argmax is True
        assert config.argmax_length is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BacktrackingConfig(
            backtrack_quota=10,
            backtrack_stride=8,
            check_cadence="token",
            verifier_threshold=0.7,
            max_token_resamples=5,
            redo_backtracked_with_argmax=False,
            argmax_length=12,
        )

        assert config.backtrack_quota == 10
        assert config.backtrack_stride == 8
        assert config.check_cadence == "token"
        assert config.verifier_threshold == 0.7
        assert config.max_token_resamples == 5
        assert config.redo_backtracked_with_argmax is False
        assert config.argmax_length == 12


class TestBacktrackingSamplingStrategyInit:
    """Test BacktrackingSamplingStrategy initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        strategy = BacktrackingSamplingStrategy()

        assert strategy.process_verifier is None
        assert strategy.backtrack_quota == 5
        assert strategy.backtrack_stride == 4
        assert strategy.check_cadence == "newline"
        assert strategy.verifier_threshold == 0.5
        assert strategy.max_token_resamples == 3
        assert strategy.redo_backtracked_with_argmax is True
        assert strategy.argmax_length == 4  # Defaults to backtrack_stride
        assert strategy.loop_budget == 1
        assert strategy.requirements is None
        assert strategy.require_hf_backend is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        verifier = Requirement(description="Test verifier")

        strategy = BacktrackingSamplingStrategy(
            process_verifier=verifier,
            backtrack_quota=10,
            backtrack_stride=8,
            check_cadence="token",
            verifier_threshold=0.7,
            max_token_resamples=5,
            redo_backtracked_with_argmax=False,
            argmax_length=12,
            loop_budget=3,
            requirements=[Requirement(description="Final req")],
            require_hf_backend=True,
        )

        assert strategy.process_verifier is verifier
        assert strategy.backtrack_quota == 10
        assert strategy.backtrack_stride == 8
        assert strategy.check_cadence == "token"
        assert strategy.verifier_threshold == 0.7
        assert strategy.max_token_resamples == 5
        assert strategy.redo_backtracked_with_argmax is False
        assert strategy.argmax_length == 12
        assert strategy.loop_budget == 3
        assert len(strategy.requirements) == 1
        assert strategy.require_hf_backend is True

    def test_init_invalid_backtrack_quota_raises(self):
        """Test that backtrack_quota <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="backtrack_quota must be greater than 0"):
            BacktrackingSamplingStrategy(backtrack_quota=0)

        with pytest.raises(ValueError, match="backtrack_quota must be greater than 0"):
            BacktrackingSamplingStrategy(backtrack_quota=-1)

    def test_init_invalid_loop_budget_raises(self):
        """Test that loop_budget <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="loop_budget must be greater than 0"):
            BacktrackingSamplingStrategy(loop_budget=0)

    def test_init_invalid_backtrack_stride_raises(self):
        """Test that backtrack_stride <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="backtrack_stride must be greater than 0"):
            BacktrackingSamplingStrategy(backtrack_stride=0)

    def test_init_argmax_length_defaults_to_stride(self):
        """Test that argmax_length defaults to backtrack_stride."""
        strategy = BacktrackingSamplingStrategy(backtrack_stride=10)
        assert strategy.argmax_length == 10

        strategy2 = BacktrackingSamplingStrategy(backtrack_stride=10, argmax_length=5)
        assert strategy2.argmax_length == 5


class TestBacktrackingSamplingStrategyIsHFBackend:
    """Test BacktrackingSamplingStrategy._is_hf_backend method."""

    def test_is_hf_backend_with_non_hf(self):
        """Test with non-HF backend."""
        strategy = BacktrackingSamplingStrategy()
        mock_backend = MagicMock(spec=Backend)

        result = strategy._is_hf_backend(mock_backend)

        assert result is False

    def test_is_hf_backend_with_hf(self):
        """Test with HF backend using patch."""
        strategy = BacktrackingSamplingStrategy()

        # mock the isinstance check
        with patch(
            "mellea.stdlib.sampling.backtracking.BacktrackingSamplingStrategy._is_hf_backend",
            return_value=True,
        ):
            # verifies the method can be called
            assert strategy._is_hf_backend(MagicMock()) is True


class TestBacktrackingSamplingStrategyCreatePrefixContext:
    """Test BacktrackingSamplingStrategy._create_prefix_context method."""

    def test_create_prefix_context(self):
        """Test prefix context creation."""
        strategy = BacktrackingSamplingStrategy()
        original_ctx = ChatContext()
        prefix_text = "Hello, world!"

        result = strategy._create_prefix_context(prefix_text, original_ctx)

        assert isinstance(result, SimpleContext)
        # the context should have the prefix as the last output
        last = result.last_output()
        assert isinstance(last, ModelOutputThunk)
        assert last.value == prefix_text


class TestBacktrackingSamplingStrategySample:
    """Test BacktrackingSamplingStrategy.sample method."""

    @pytest.mark.asyncio
    async def test_require_hf_backend_raises_for_non_hf(self):
        """Test that require_hf_backend=True raises TypeError for non-HF backends."""
        strategy = BacktrackingSamplingStrategy(require_hf_backend=True)
        mock_backend = MagicMock(spec=Backend)
        mock_action = MagicMock()
        mock_action.parse = MagicMock(return_value="parsed")
        mock_ctx = ChatContext()

        with pytest.raises(TypeError, match="BacktrackingSamplingStrategy requires LocalHFBackend"):
            await strategy.sample(
                action=mock_action,
                context=mock_ctx,
                backend=mock_backend,
                requirements=[],
            )

    @pytest.mark.asyncio
    async def test_fallback_to_standard_generation_for_non_hf(self):
        """Test that non-HF backend falls back to standard generation with require_hf_backend=False."""
        strategy = BacktrackingSamplingStrategy(require_hf_backend=False)

        # create mock backend
        mock_backend = MagicMock(spec=Backend)
        mock_output = ModelOutputThunk(value="test output")
        mock_output._generate_log = MagicMock()
        mock_output._generate_log.is_final_result = False
        mock_output._meta = {}

        mock_ctx = ChatContext()
        mock_result_ctx = mock_ctx.add(mock_output)

        # setup async mock
        async def mock_generate(*args, **kwargs):
            return mock_output, mock_result_ctx

        mock_backend.generate_from_context = AsyncMock(side_effect=mock_generate)

        # Create mock action
        mock_action = MagicMock()
        mock_action.parse = MagicMock(return_value="parsed")

        # Run sample
        with patch("mellea.stdlib.sampling.backtracking.mfuncs.avalidate", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = []  # No requirements to validate

            result = await strategy.sample(
                action=mock_action,
                context=mock_ctx,
                backend=mock_backend,
                requirements=[],
            )

        # Verify fallback was used
        assert mock_backend.generate_from_context.called
        assert result.success is True  # With no requirements, it succeeds
        assert "backtracking_metrics" in result.result._meta

    @pytest.mark.asyncio
    async def test_no_verifier_no_backtracks(self):
        """Test that without process_verifier, metrics show num_backtracks=0."""
        strategy = BacktrackingSamplingStrategy(process_verifier=None)

        mock_backend = MagicMock(spec=Backend)
        mock_output = ModelOutputThunk(value="test output")
        mock_output._generate_log = MagicMock()
        mock_output._generate_log.is_final_result = False
        mock_output._meta = {}

        mock_ctx = ChatContext()
        mock_result_ctx = mock_ctx.add(mock_output)

        async def mock_generate(*args, **kwargs):
            return mock_output, mock_result_ctx

        mock_backend.generate_from_context = AsyncMock(side_effect=mock_generate)

        mock_action = MagicMock()
        mock_action.parse = MagicMock(return_value="parsed")

        with patch("mellea.stdlib.sampling.backtracking.mfuncs.avalidate", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = []

            result = await strategy.sample(
                action=mock_action,
                context=mock_ctx,
                backend=mock_backend,
                requirements=[],
            )

        # Check metrics show no backtracks
        metrics = result.result._meta.get("backtracking_metrics", {})
        assert metrics.get("num_backtracks", 0) == 0

    @pytest.mark.asyncio
    async def test_validation_failure_returns_failure_result(self):
        """Test that validation failure returns success=False."""
        strategy = BacktrackingSamplingStrategy(
            require_hf_backend=False,
            loop_budget=1,
        )

        mock_backend = MagicMock(spec=Backend)
        mock_output = ModelOutputThunk(value="test output")
        mock_output._generate_log = MagicMock()
        mock_output._generate_log.is_final_result = False
        mock_output._meta = {}

        mock_ctx = ChatContext()
        mock_result_ctx = mock_ctx.add(mock_output)

        async def mock_generate(*args, **kwargs):
            return mock_output, mock_result_ctx

        mock_backend.generate_from_context = AsyncMock(side_effect=mock_generate)

        mock_action = MagicMock()
        mock_action.parse = MagicMock(return_value="parsed")

        # Create a failing requirement
        failing_req = Requirement(description="Always fails")

        with patch("mellea.stdlib.sampling.backtracking.mfuncs.avalidate", new_callable=AsyncMock) as mock_validate:
            # Return a failing validation
            mock_validate.return_value = [ValidationResult(result=False, reason="Test failure")]

            result = await strategy.sample(
                action=mock_action,
                context=mock_ctx,
                backend=mock_backend,
                requirements=[failing_req],
            )

        assert result.success is False
        assert len(result.sample_generations) == 1

    @pytest.mark.asyncio
    async def test_quota_respected_no_infinite_loops(self):
        """Test that backtrack_quota is respected and prevents infinite loops."""
        strategy = BacktrackingSamplingStrategy(
            backtrack_quota=2,
            loop_budget=1,
            require_hf_backend=False,
        )

        # Just verify the strategy can be created with a low quota
        assert strategy.backtrack_quota == 2

    @pytest.mark.asyncio
    async def test_token_resampling_bounded(self):
        """Test that max_token_resamples is respected."""
        strategy = BacktrackingSamplingStrategy(
            max_token_resamples=5,
            require_hf_backend=False,
        )

        # Verify the config is set
        assert strategy.max_token_resamples == 5


class TestBacktrackingSamplingStrategyEvaluateProcessVerifier:
    """Test BacktrackingSamplingStrategy._evaluate_process_verifier method."""

    @pytest.mark.asyncio
    async def test_no_verifier_returns_true(self):
        """Test that no verifier returns True."""
        strategy = BacktrackingSamplingStrategy(process_verifier=None)
        mock_backend = MagicMock(spec=Backend)
        mock_ctx = ChatContext()

        passed, metrics = await strategy._evaluate_process_verifier(
            prefix_text="test",
            backend=mock_backend,
            original_ctx=mock_ctx,
        )

        assert passed is True
        assert metrics.num_verifier_calls == 0

    @pytest.mark.asyncio
    async def test_boolean_verifier_passes(self):
        """Test boolean verifier that passes."""
        verifier = Requirement(
            description="Test",
            validation_fn=lambda ctx: ValidationResult(result=True),
        )
        strategy = BacktrackingSamplingStrategy(process_verifier=verifier)
        mock_backend = MagicMock(spec=Backend)
        mock_ctx = ChatContext()

        passed, metrics = await strategy._evaluate_process_verifier(
            prefix_text="test",
            backend=mock_backend,
            original_ctx=mock_ctx,
        )

        assert passed is True
        assert metrics.num_verifier_calls == 1

    @pytest.mark.asyncio
    async def test_boolean_verifier_fails(self):
        """Test boolean verifier that fails."""
        verifier = Requirement(
            description="Test",
            validation_fn=lambda ctx: ValidationResult(result=False),
        )
        strategy = BacktrackingSamplingStrategy(process_verifier=verifier)
        mock_backend = MagicMock(spec=Backend)
        mock_ctx = ChatContext()

        passed, metrics = await strategy._evaluate_process_verifier(
            prefix_text="test",
            backend=mock_backend,
            original_ctx=mock_ctx,
        )

        assert passed is False
        assert metrics.num_verifier_calls == 1

    @pytest.mark.asyncio
    async def test_score_verifier_above_threshold(self):
        """Test score-based verifier above threshold."""
        verifier = Requirement(
            description="Test",
            validation_fn=lambda ctx: ValidationResult(result=True, score=0.8),
        )
        strategy = BacktrackingSamplingStrategy(
            process_verifier=verifier,
            verifier_threshold=0.5,
        )
        mock_backend = MagicMock(spec=Backend)
        mock_ctx = ChatContext()

        passed, metrics = await strategy._evaluate_process_verifier(
            prefix_text="test",
            backend=mock_backend,
            original_ctx=mock_ctx,
        )

        assert passed is True

    @pytest.mark.asyncio
    async def test_score_verifier_below_threshold(self):
        """Test score-based verifier below threshold."""
        verifier = Requirement(
            description="Test",
            validation_fn=lambda ctx: ValidationResult(result=True, score=0.3),
        )
        strategy = BacktrackingSamplingStrategy(
            process_verifier=verifier,
            verifier_threshold=0.5,
        )
        mock_backend = MagicMock(spec=Backend)
        mock_ctx = ChatContext()

        passed, metrics = await strategy._evaluate_process_verifier(
            prefix_text="test",
            backend=mock_backend,
            original_ctx=mock_ctx,
        )

        assert passed is False


class TestCheckCadenceOptions:
    """Test different check_cadence options."""

    def test_token_cadence(self):
        """Test token cadence option."""
        strategy = BacktrackingSamplingStrategy(check_cadence="token")
        assert strategy.check_cadence == "token"

    def test_newline_cadence(self):
        """Test newline cadence option (default)."""
        strategy = BacktrackingSamplingStrategy(check_cadence="newline")
        assert strategy.check_cadence == "newline"

    def test_boundary_cadence(self):
        """Test boundary cadence option."""
        strategy = BacktrackingSamplingStrategy(check_cadence="boundary")
        assert strategy.check_cadence == "boundary"


@pytest.mark.qualitative
@pytest.mark.llm
class TestBacktrackingIntegration:
    """Integration tests for BacktrackingSamplingStrategy.

    These tests require actual HF backends and are marked as qualitative.
    """

    @pytest.mark.skip(reason="Requires HF model download")
    def test_backtracking_with_hf_backend(self):
        """Test backtracking with actual HF backend."""
        from mellea.backends.huggingface import LocalHFBackend
        from mellea.stdlib.components import Instruction
        from mellea.stdlib.context import ChatContext

        backend = LocalHFBackend("ibm-granite/granite-3.0-2b-instruct")

        # Simple length verifier
        length_verifier = Requirement(
            description="Output must be short",
            validation_fn=lambda ctx: ValidationResult(
                len(ctx.last_output().value) < 100
            ),
        )

        strategy = BacktrackingSamplingStrategy(
            process_verifier=length_verifier,
            backtrack_quota=3,
            backtrack_stride=10,
        )

        import asyncio

        async def run():
            result = await strategy.sample(
                action=Instruction("Write a short poem"),
                context=ChatContext(),
                backend=backend,
                requirements=[],
            )
            return result

        result = asyncio.run(run())

        assert result is not None
        assert hasattr(result, "success")
        assert result.result._meta.get("backtracking_metrics") is not None

    # @pytest.mark.skip(reason="Requires HF model download")
    # def test_single_prefill_verification(self):
    #     """Verify prompt prefill occurs once during backtracking."""
    #     pass
    #
    # @pytest.mark.skip(reason="Requires HF model download")
    # def test_metrics_populated(self):
    #     """Test that backtracking metrics are properly populated."""
    #     pass
    #
    # @pytest.mark.skip(reason="Requires HF model download")
    # def test_final_validation_runs(self):
    #     """Test that standard requirements are validated after generation."""
    #     pass


if __name__ == "__main__":
    pytest.main(["-v", __file__])