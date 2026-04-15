"""Reward-guided decoding handler for HuggingFace generation."""

from __future__ import annotations

from typing import Any

from ...core.steering import Control, OutputControlHandler


class RewardGuidedDecodingHandler(OutputControlHandler):
    """Adds a reward-model-based logits processor to generation kwargs.

    The artifact must be a callable reward model that accepts input_ids and returns
    per-token reward scores. The handler wraps it in a ``LogitsProcessor`` and appends
    it to the ``logits_processor`` list in gen kwargs.

    Expects ``control.params`` to optionally contain:

    - ``temperature`` (float, optional): Scaling factor for reward scores.
      Defaults to ``1.0``.
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Add a reward-guided logits processor to gen kwargs.

        Args:
            control: The control descriptor.
            gen_kwargs: The current generation kwargs dict.
            artifact: A callable reward model.

        Returns:
            The updated generation kwargs dict with the logits processor added.
        """
        import torch

        reward_model = artifact
        assert reward_model is not None, (
            "RewardGuidedDecodingHandler requires a reward model artifact"
        )
        temperature = control.params.get("temperature", 1.0)

        class _RewardLogitsProcessor:
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor
            ) -> torch.FloatTensor:
                with torch.no_grad():
                    rewards = reward_model(input_ids)
                return scores + rewards * temperature

        processors = gen_kwargs.get("logits_processor", [])
        # Avoid mutating a shared default list.
        if not isinstance(processors, list):
            processors = list(processors)
        else:
            processors = list(processors)
        processors.append(_RewardLogitsProcessor())
        gen_kwargs["logits_processor"] = processors
        return gen_kwargs
