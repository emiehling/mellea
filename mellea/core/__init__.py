"""Core abstractions for the mellea library.

This package defines the fundamental interfaces and data structures on which every
other layer of mellea is built: the ``Backend``, ``Formatter``, and
``SamplingStrategy`` protocols; the ``Component``, ``CBlock``, ``Context``, and
``ModelOutputThunk`` data types that flow through the inference pipeline; and
``Requirement`` / ``ValidationResult`` for constrained generation. Start here when
building a new backend, formatter, or sampling strategy, or when you need the type
definitions shared across the library.
"""

from .backend import Backend, BaseModelSubclass, generate_walk
from .base import (
    C,
    CBlock,
    Component,
    ComponentParseError,
    ComputedModelOutputThunk,
    Context,
    ContextTurn,
    GenerateLog,
    GenerateType,
    ImageBlock,
    ModelOutputThunk,
    ModelToolCall,
    S,
    TemplateRepresentation,
    blockify,
)
from .formatter import Formatter
from .requirement import Requirement, ValidationResult, default_output_to_bool
from .sampling import SamplingResult, SamplingStrategy
from .steering import (
    BackendCapabilities,
    Composer,
    Control,
    ControlCategory,
    ControlHandler,
    InputControlHandler,
    OutputControlHandler,
    RemoteStateControlHandler,
    ResolvedControl,
    StateControlHandler,
    SteeringPolicy,
    StructuralControlHandler,
    get_global_input_handler,
    register_global_input_handler,
)
from .utils import FancyLogger

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BaseModelSubclass",
    "C",
    "CBlock",
    "Component",
    "ComponentParseError",
    "Composer",
    "ComputedModelOutputThunk",
    "Context",
    "ContextTurn",
    "Control",
    "ControlCategory",
    "ControlHandler",
    "FancyLogger",
    "Formatter",
    "GenerateLog",
    "GenerateType",
    "ImageBlock",
    "InputControlHandler",
    "ModelOutputThunk",
    "ModelToolCall",
    "OutputControlHandler",
    "RemoteStateControlHandler",
    "Requirement",
    "ResolvedControl",
    "S",
    "SamplingResult",
    "SamplingStrategy",
    "StateControlHandler",
    "SteeringPolicy",
    "StructuralControlHandler",
    "TemplateRepresentation",
    "ValidationResult",
    "blockify",
    "default_output_to_bool",
    "generate_walk",
    "get_global_input_handler",
    "register_global_input_handler",
]
