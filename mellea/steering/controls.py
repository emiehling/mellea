"""Base classes for steering controls.

All controls are classified as one of three types:
- InputControl: transforms Components/Context before formatting (Mellea-level)
- StateControl: describes model-internal interventions (backend-level)
- OutputControl: describes decoding-time interventions (backend-level)

Concrete control implementations live in mellea/stdlib/controls/.
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
class StateControl(abc.ABC):
    """Base class for state controls.

    State controls are pure data descriptors (no behavior). The backend
    reads the descriptor fields and applies the intervention using its
    own internals.

    Concrete subclasses MUST be decorated with @dataclass(frozen=True).
    """


@dataclass(frozen=True)
class OutputControl(abc.ABC):
    """Base class for output controls.

    Output controls are pure data descriptors that modify the decoding
    process. The backend reads the descriptor fields and translates them
    to backend-specific parameters.

    Concrete subclasses MUST be decorated with @dataclass(frozen=True).
    """
