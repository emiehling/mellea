"""Steering control handlers for API-based backends.

API backends can only apply input controls (via global handlers) and static
output controls (via this module). Active output controls, state controls,
and structural controls are not available through API endpoints.
"""

from __future__ import annotations

from typing import Any

from ..core.steering import Control, OutputControlHandler


class APIStaticOutputControlHandler(OutputControlHandler):
    """Merges static generation parameters into API request kwargs.

    Maps Mellea-native parameter names to backend-specific API parameter names.
    The ``control.params`` dict is merged into the generation kwargs, with
    backend-specific key translation handled by the backend's existing
    ``_simplify_and_merge`` or equivalent method.

    All entries in ``control.params`` are set directly on the kwargs dict.
    """

    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Merge control params into API request kwargs.

        Args:
            control: The control descriptor whose params are API parameters.
            gen_kwargs: The current API request kwargs dict.
            artifact: Unused.

        Returns:
            The updated kwargs dict.
        """
        gen_kwargs.update(control.params)
        return gen_kwargs
