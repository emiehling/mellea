"""Portable output control handler for static generation parameter overrides."""

from __future__ import annotations

from typing import Any

from ...core.steering import Control, OutputControlHandler


class StaticOutputControlHandler(OutputControlHandler):
    """Merges static generation parameters into generation/request kwargs.

    All entries in ``control.params`` are merged directly into the kwargs dict.
    Works identically for HuggingFace ``model.generate()`` kwargs and API
    request kwargs — both are ``dict[str, Any]`` merges.

    Common params: ``temperature``, ``top_p``, ``top_k``,
    ``repetition_penalty``, ``max_new_tokens``.
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Merge control params into generation/request kwargs.

        Args:
            control: The control descriptor whose params are generation kwargs.
            gen_kwargs: The current generation kwargs dict.
            artifact: Unused.

        Returns:
            The updated generation kwargs dict.
        """
        gen_kwargs.update(control.params)
        return gen_kwargs
