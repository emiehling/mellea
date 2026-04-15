"""HuggingFace-specific output handlers for logits processing and stopping."""

from __future__ import annotations

from typing import Any

from ...core.steering import Control, OutputControlHandler


class LogitsProcessorHandler(OutputControlHandler):
    """Appends a LogitsProcessor to generation kwargs.

    The artifact must be a callable conforming to HuggingFace's
    LogitsProcessor interface:
        (input_ids: LongTensor, scores: FloatTensor) -> FloatTensor

    This is the fundamental OUTPUT handler for any control that modifies the
    token distribution during decoding. Specific behaviors — reward-guided
    decoding, length penalties, token biasing, classifier-free guidance —
    are determined by which LogitsProcessor is passed as the artifact, not
    by different handler classes.

    Expects control.params to optionally contain:

    - ``priority`` (str): ``"prepend"`` or ``"append"`` (default).
      Determines whether the processor is added to the front or back of
      the processor list. Order matters when processors interact.
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Append (or prepend) a LogitsProcessor to gen kwargs.

        Args:
            control: The control descriptor.
            gen_kwargs: The current generation kwargs dict.
            artifact: A callable LogitsProcessor.

        Returns:
            The updated generation kwargs dict.
        """
        assert artifact is not None, (
            "LogitsProcessorHandler requires a LogitsProcessor artifact"
        )
        priority = control.params.get("priority", "append")

        processors = gen_kwargs.get("logits_processor", [])
        # Avoid mutating a shared default list.
        processors = list(processors)
        if priority == "prepend":
            processors.insert(0, artifact)
        else:
            processors.append(artifact)
        gen_kwargs["logits_processor"] = processors
        return gen_kwargs


class StoppingCriteriaHandler(OutputControlHandler):
    """Appends a StoppingCriteria to generation kwargs.

    The artifact must be a callable conforming to HuggingFace's
    StoppingCriteria interface:
        (input_ids: LongTensor, scores: FloatTensor) -> bool

    Expects control.params to optionally contain:

    - ``priority`` (str): ``"prepend"`` or ``"append"`` (default).
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Append (or prepend) a StoppingCriteria to gen kwargs.

        Args:
            control: The control descriptor.
            gen_kwargs: The current generation kwargs dict.
            artifact: A callable StoppingCriteria.

        Returns:
            The updated generation kwargs dict.
        """
        assert artifact is not None, (
            "StoppingCriteriaHandler requires a StoppingCriteria artifact"
        )
        priority = control.params.get("priority", "append")

        criteria = gen_kwargs.get("stopping_criteria", [])
        criteria = list(criteria)
        if priority == "prepend":
            criteria.insert(0, artifact)
        else:
            criteria.append(artifact)
        gen_kwargs["stopping_criteria"] = criteria
        return gen_kwargs
