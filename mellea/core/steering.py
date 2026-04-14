"""Steering policy and composer abstractions for inference-time model interventions.

Defines the ``ControlCategory`` enum, the ``Control`` and ``SteeringPolicy`` immutable
data types that flow through the steered generation pipeline, the ``BackendCapabilities``
descriptor, and the ``Composer`` abstract base class for constructing and updating
steering policies. Start here when building a new composer or extending a backend
with steering support.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .requirement import Requirement, ValidationResult


class ControlCategory(enum.Enum):
    """Categories of steering controls, corresponding to stages of the generation pipeline.

    Attributes:
        INPUT: Controls that modify the prompt/action/context before the formatter
            (e.g., prompt rewriting, system prompt injection).
        STRUCTURAL: Controls that modify model structure before the forward pass
            (e.g., LoRA adapters, adapter blending).
        STATE: Controls that modify model state during the forward pass
            (e.g., activation steering via forward hooks, attention steering).
        OUTPUT: Controls that modify the decoding/generation process
            (e.g., temperature, logit bias, reward-driven decoding, custom logits processors).
    """

    INPUT = "input"
    STRUCTURAL = "structural"
    STATE = "state"
    OUTPUT = "output"


@dataclass(frozen=True)
class Control:
    """An immutable descriptor for a single steering intervention.

    Controls are lightweight descriptors -- they carry configuration and artifact
    references but not the artifacts themselves. The backend resolves artifact
    references against its registry at ``attach()`` time.

    Args:
        category (ControlCategory): Which stage of the generation pipeline this
            control applies to.
        name (str): Identifier for this control (e.g., ``"rewrite_for_conciseness"``,
            ``"honesty_steering_vector"``).
        params (Mapping[str, Any]): Control-specific configuration (e.g., layer indices
            for activation steering, temperature for static output controls).
        artifact_ref (str | None): Optional reference to the steering artifacts library
            for heavyweight objects (steering vectors, reward models). The backend
            resolves this against its registry.
        model_family (str | None): Model family this artifact was trained for, or
            ``None`` if model-agnostic (e.g., most input controls).
    """

    category: ControlCategory
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    artifact_ref: str | None = None
    model_family: str | None = None


@dataclass(frozen=True)
class SteeringPolicy:
    """An immutable, composed set of steering controls that biases generation.

    A steering policy is a conditional plan across the full model control surface.
    It must be attached to a backend (via ``backend.attach()``) before it can
    influence generation. Policies are immutable -- every modification returns a
    new ``SteeringPolicy``.

    Args:
        controls (tuple[Control, ...]): Ordered tuple of control descriptors.
    """

    controls: tuple[Control, ...] = ()

    def controls_for_stage(self, category: ControlCategory) -> tuple[Control, ...]:
        """Return controls filtered by category.

        Args:
            category: The control category to filter by.

        Returns:
            Tuple of controls matching the given category.
        """
        return tuple(c for c in self.controls if c.category == category)

    def __add__(self, other: SteeringPolicy) -> SteeringPolicy:
        """Compose two policies by concatenating their controls.

        Args:
            other: The policy to combine with this one.

        Returns:
            A new ``SteeringPolicy`` containing controls from both policies.
        """
        return SteeringPolicy(controls=self.controls + other.controls)

    def __bool__(self) -> bool:
        """Return ``False`` for empty policies."""
        return len(self.controls) > 0

    @classmethod
    def empty(cls) -> SteeringPolicy:
        """Return a policy with no controls, representing no steering.

        Returns:
            An empty ``SteeringPolicy``.
        """
        return cls()


@dataclass(frozen=True)
class BackendCapabilities:
    """Describes which steering interventions a backend supports.

    Backends expose a ``capabilities`` property returning this descriptor so the
    ``Composer`` can construct policies containing only runnable controls (DR#5).

    Args:
        supported_categories (frozenset[ControlCategory]): Control categories the
            backend can execute.
        supports_logits_processors (bool): Whether the backend accepts custom logits
            processors during generation.
        supports_adapter_loading (bool): Whether the backend supports dynamic
            adapter loading/blending.
        supports_forward_hooks (bool): Whether the backend supports registering
            forward hooks on model layers.
        extra (dict[str, Any]): Backend-specific capability flags.
    """

    supported_categories: frozenset[ControlCategory] = frozenset()
    supports_logits_processors: bool = False
    supports_adapter_loading: bool = False
    supports_forward_hooks: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Composer(abc.ABC):
    """Abstract base class for constructing and updating steering policies.

    A ``Composer`` is responsible for two operations:

    1. **Composing** an initial steering policy from requirements and backend
       capabilities (called once before the sampling loop).
    2. **Updating** the policy based on validation failures during repair
       (called on each repair iteration).

    When no ``Composer`` is configured, the flow is identical to the current
    unsteered flow.
    """

    @abc.abstractmethod
    def compose(
        self, requirements: list[Requirement], capabilities: BackendCapabilities
    ) -> SteeringPolicy:
        """Construct an initial steering policy from requirements and backend capabilities.

        Maps requirements + backend capabilities into a composed set of pretrained
        steering artifacts. This is called once before the sampling loop begins.

        Args:
            requirements: The full set of requirements for this generation call.
            capabilities: The backend's declared steering capabilities.

        Returns:
            The initial ``SteeringPolicy`` to attach to the backend.
        """
        ...

    @abc.abstractmethod
    def update(
        self,
        current_policy: SteeringPolicy,
        validation_results: list[tuple[Requirement, ValidationResult]],
        capabilities: BackendCapabilities,
    ) -> SteeringPolicy:
        """Refine the steering policy based on validation failures.

        Called during repair when requirements fail. Receives the current policy,
        the validation results (including failure reasons and scores), and the
        backend capabilities. Returns an updated policy for the next generation
        attempt.

        Args:
            current_policy: The steering policy used in the previous generation.
            validation_results: Per-requirement validation outcomes from the
                previous generation, as ``(Requirement, ValidationResult)`` tuples.
            capabilities: The backend's declared steering capabilities.

        Returns:
            An updated ``SteeringPolicy`` for the next generation attempt.
        """
        ...
