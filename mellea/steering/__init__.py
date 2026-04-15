"""Steering artifacts library and control factories for Mellea.

This package provides the infrastructure for managing pretrained steering artifacts
(steering vectors, prompt adapters, reward models, etc.) and factory functions for
creating common control types. The ``Composer`` implementations in
``mellea.stdlib.steering`` draw from this library during policy composition.
"""

from .artifacts import ArtifactRegistry, SteeringArtifact, get_default_registry
from .controls import (
    active_output_control,
    input_control,
    state_control,
    static_output_control,
    structural_control,
)
from .handlers import (
    ContextPrefixHandler,
    InstructionRewriteHandler,
    SystemPromptInjectionHandler,
)

__all__ = [
    "ArtifactRegistry",
    "ContextPrefixHandler",
    "InstructionRewriteHandler",
    "SteeringArtifact",
    "SystemPromptInjectionHandler",
    "active_output_control",
    "get_default_registry",
    "input_control",
    "state_control",
    "static_output_control",
    "structural_control",
]
