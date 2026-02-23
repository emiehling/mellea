"""Base classes for steering controls.

Controls are classified as one of two types:
- InputControl: transforms Components/Context before formatting (Mellea-level)
- BackendControl: pure data descriptor for backend-executed interventions, e.g., activation steering, decoding-time intervenctions, etc.

Control implementations live in mellea/stdlib/controls/.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.base import CBlock, Component, Context


class InputControl(abc.ABC):
    """Base class for input controls.

    Input controls are unique among controls: they are both declarative
    (frozen dataclass fields define identity/hashability) and behavioral
    (the apply method transforms Components).

    Concrete subclasses MUST be decorated with @dataclass(frozen=True).
    The apply method MUST be deterministic given the dataclass fields
    and MUST only call public builder methods on Components.

    InputControls are always executable, i.e., they operate at the Mellea level
    before the backend is involved, so no backend capability check is needed.
    """

    @abc.abstractmethod
    def apply(
        self, action: Component | CBlock, ctx: Context
    ) -> tuple[Component | CBlock, Context]:
        """Transform the action and/or context.

        Args:
            action: The current action Component or CBlock.
            ctx: The current Context.

        Returns:
            A possibly modified (action, context) tuple.
        """
        ...


@dataclass(frozen=True)
class BackendControl(abc.ABC):
    """Base class for backend-executed steering controls.

    Backend controls are pure data descriptors — they carry no behavior.
    The backend reads the fields and applies the intervention using its
    own internals, whether that means modifying decoding parameters,
    applying activation vectors, or adjusting attention masks.

    Concrete subclasses MUST be decorated with @dataclass(frozen=True).
    """
