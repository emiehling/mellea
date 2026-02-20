"""SteeringPolicy — immutable container of controls for a generation request."""

from __future__ import annotations

from dataclasses import dataclass

from .controls import InputControl, OutputControl, StateControl


@dataclass(frozen=True)
class SteeringPolicy:
    """An immutable, hashable collection of steering controls.

    New policies are created by constructing new SteeringPolicy instances.

    Note that there is intentionally no public merge method. Mechanically combining
    two policies can produce conflicting controls. The SteeringOptimizer 
    is the only sanctioned path to multi-requirement policies.
    """

    input_controls: tuple[InputControl, ...] = ()
    state_controls: tuple[StateControl, ...] = ()
    output_controls: tuple[OutputControl, ...] = ()

    def is_empty(self) -> bool:
        """True if the policy contains no controls of any type."""
        return not (self.input_controls or self.state_controls or self.output_controls)

    @property
    def backend_policy(self) -> SteeringPolicy:
        """The subset of controls requiring backend execution.

        Returns a new SteeringPolicy containing only state and output
        controls. Input controls are excluded because they are applied
        at the Mellea level before the backend is involved.
        """
        return SteeringPolicy(
            state_controls=self.state_controls, output_controls=self.output_controls
        )
