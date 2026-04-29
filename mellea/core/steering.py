"""Steering policy, composer, and handler abstractions for inference-time model interventions.

Defines the ``ControlCategory`` enum, the ``Control`` and ``SteeringPolicy`` immutable
data types that flow through the steered generation pipeline, the ``BackendCapabilities``
descriptor, the ``Composer`` abstract base class for constructing and updating
steering policies, category-specific handler ABCs for executing controls, and the
``ResolvedControl`` type that pairs a control with its handler and loaded artifact.
Start here when building a new composer, handler, or extending a backend with steering
support.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import CBlock, Component, Context
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
    ``Composer`` can construct policies containing only runnable controls.

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


# ---------------------------------------------------------------------------
# Handler ABCs — category-specific interpreters for Controls
# ---------------------------------------------------------------------------


class InputControlHandler(abc.ABC):
    """Transforms the action and/or linearized context before the formatter runs.

    Input controls are the most portable category — they operate on
    ``Component`` and ``CBlock`` objects, not model internals, so they work
    on every backend. Handlers are stateless and may be shared across calls.

    Handlers receive the **linearized** context (the output of
    ``Context.view_for_generation()``) so they are decoupled from the
    ``Context`` subclass used by the session.
    """

    @abc.abstractmethod
    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        linearized_ctx: list[Component | CBlock],
        artifact: Any | None,
    ) -> tuple[Component | CBlock, list[Component | CBlock]]:
        """Apply an input control to the action and linearized context.

        Args:
            control: The control descriptor with params.
            action: The current action component or content block.
            linearized_ctx: The linearized context — a list of components
                produced by ``Context.view_for_generation()``.
            artifact: The resolved artifact, or ``None`` if the control
                has no ``artifact_ref``.

        Returns:
            A ``(action, linearized_ctx)`` tuple, potentially modified.
        """
        ...


class StructuralControlHandler(abc.ABC):
    """Manages model structure modifications (adapters, merges) with scoped lifecycle.

    Structural controls modify the model's weight structure — loading LoRA adapters,
    blending adapter weights, etc. They must live inside the backend's generation
    lock scope because adapter state is global to the model instance.
    """

    @abc.abstractmethod
    def activate(self, control: Control, model: Any, artifact: Any | None) -> Any:
        """Apply structural modifications to the model.

        Called inside the generation lock, before ``model.generate()``.

        Args:
            control: The control descriptor with params.
            model: The model object to modify (e.g., a HuggingFace ``PreTrainedModel``).
            artifact: The resolved artifact (e.g., an adapter path or weight tensor),
                or ``None``.

        Returns:
            An opaque handle passed to ``deactivate()`` to reverse the modifications.
        """
        ...

    @abc.abstractmethod
    def deactivate(self, handle: Any) -> None:
        """Reverse the structural modifications applied by ``activate()``.

        Called inside the generation lock, after ``model.generate()`` completes
        (including on error, via ``finally``).

        Args:
            handle: The opaque handle returned by ``activate()``.
        """
        ...


class StateControlHandler(abc.ABC):
    """Manages forward hooks on model internals with scoped lifecycle.

    State controls modify the model's runtime behavior during the forward pass —
    activation steering, attention head masking, etc. They register hooks before
    ``model.generate()`` and remove them after. Like structural controls, they
    must live inside the generation lock scope.
    """

    @abc.abstractmethod
    def activate(self, control: Control, model: Any, artifact: Any | None) -> Any:
        """Register forward hooks on the model.

        Called inside the generation lock, before ``model.generate()``.

        Args:
            control: The control descriptor with params (e.g., layer indices,
                scaling coefficients).
            model: The model object to hook (e.g., a HuggingFace ``PreTrainedModel``).
            artifact: The resolved artifact (e.g., a steering vector tensor),
                or ``None``.

        Returns:
            An opaque handle (typically a list of hook handles) passed to
            ``deactivate()`` for cleanup.
        """
        ...

    @abc.abstractmethod
    def deactivate(self, handle: Any) -> None:
        """Remove forward hooks registered by ``activate()``.

        Called inside the generation lock, after ``model.generate()`` completes
        (including on error, via ``finally``). Must be called even if generation
        raises — leaked hooks silently corrupt subsequent generations.

        Args:
            handle: The opaque handle returned by ``activate()``.
        """
        ...


class OutputControlHandler(abc.ABC):
    """Modifies the generation/decoding process.

    Output controls come in two flavors:

    - **Static**: Simple parameter overrides (temperature, top_p, logit_bias)
      merged into generation kwargs.
    - **Active**: Callable objects (logits processors, stopping criteria, reward
      models) passed to ``model.generate()`` via its processor/criteria args.

    Both flavors are handled through the same ``apply()`` method — the handler
    is responsible for adding to the appropriate kwargs key.
    """

    @abc.abstractmethod
    def apply(
        self, control: Control, gen_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Merge control parameters into generation kwargs.

        Args:
            control: The control descriptor with params.
            gen_kwargs: The current generation keyword arguments dict. The handler
                should return a modified copy or mutate and return the same dict.
            artifact: The resolved artifact (e.g., a reward model for guided
                decoding), or ``None``.

        Returns:
            The updated generation kwargs dict.
        """
        ...


class RemoteStateControlHandler(abc.ABC):
    """Contributes a state-steering payload to a remote-backend request body.

    For backends that do not own the model in-process (OpenAI-compatible HTTP,
    vLLM serve, etc.). Where ``StateControlHandler`` registers forward hooks,
    this handler builds a JSON-serializable contribution that the backend
    merges into its request kwargs. Stateless and per-call: there is no
    activate/deactivate lifecycle because the steering payload lives in the
    request itself.
    """

    @abc.abstractmethod
    def contribute_to_request(
        self, control: Control, request_kwargs: dict[str, Any], artifact: Any | None
    ) -> dict[str, Any]:
        """Merge a steering contribution into request kwargs.

        Args:
            control: The control descriptor.
            request_kwargs: Current request kwargs dict (typically including
                ``extra_body``). The handler should mutate-and-return or
                return a new dict.
            artifact: The resolved artifact. For activation steering this is
                a ``dict[int, Tensor]`` from ``VectorStore.get_raw()``.

        Returns:
            The updated request kwargs.
        """
        ...


ControlHandler = (
    InputControlHandler
    | StructuralControlHandler
    | StateControlHandler
    | OutputControlHandler
    | RemoteStateControlHandler
)
"""Union of all handler types."""


@dataclass(frozen=True)
class ResolvedControl:
    """A control paired with its handler and resolved artifact.

    Created during ``backend.attach()`` and cached for the lifetime of the
    attached policy. The backend's generation pipeline works exclusively with
    resolved controls — never raw ``Control`` descriptors.

    Args:
        control (Control): The original control descriptor.
        handler (ControlHandler): The handler that knows how to execute this control.
        artifact (Any | None): The resolved artifact object (e.g., a steering vector
            tensor, a reward model instance, an adapter weight path). ``None`` for
            controls that don't reference external artifacts.
    """

    control: Control
    handler: ControlHandler
    artifact: Any | None = None


# ---------------------------------------------------------------------------
# Global input handler registry — backend-agnostic fallback
# ---------------------------------------------------------------------------

_global_input_handlers: dict[str, InputControlHandler] = {}


def register_global_input_handler(name: str, handler: InputControlHandler) -> None:
    """Register a backend-agnostic input control handler.

    Global handlers serve as fallbacks — if a backend does not have a handler
    registered for a given input control name, the global registry is checked.
    Only ``InputControlHandler`` instances may be registered globally, since
    input controls are the only category that operates on data (action/context)
    rather than model internals.

    Args:
        name: The control name this handler responds to.
        handler: The handler implementation.
    """
    _global_input_handlers[name] = handler


def get_global_input_handler(name: str) -> InputControlHandler | None:
    """Look up a globally registered input control handler.

    Args:
        name: The control name to look up.

    Returns:
        The handler, or ``None`` if not registered.
    """
    return _global_input_handlers.get(name)
