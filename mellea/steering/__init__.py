"""Steering artifacts library and control factories for Mellea.

This package provides the infrastructure for managing pretrained steering artifacts
(steering vectors, prompt adapters, reward models, etc.) and factory functions for
creating common control types. The ``Composer`` implementations in
``mellea.stdlib.steering`` draw from this library during policy composition.
"""

from .controls import (
    active_output_control,
    input_control,
    state_control,
    static_output_control,
    structural_control,
)
from .handlers import (
    ContextPrefixHandler,
    ICLExampleSelectorHandler,
    InstructionRewriteHandler,
    StaticOutputControlHandler,
    SystemPromptInjectionHandler,
)
from .library import (
    ArtifactInfo,
    ArtifactLibrary,
    get_default_library,
    set_default_library,
)
from .stores import AdapterStore, ArtifactStore, ModelStore, PromptStore, VectorStore

__all__ = [
    "AdapterStore",
    "ArtifactInfo",
    "ArtifactLibrary",
    "ArtifactStore",
    "ContextPrefixHandler",
    "ICLExampleSelectorHandler",
    "InstructionRewriteHandler",
    "ModelStore",
    "PromptStore",
    "StaticOutputControlHandler",
    "SystemPromptInjectionHandler",
    "VectorStore",
    "active_output_control",
    "get_default_library",
    "input_control",
    "set_default_library",
    "state_control",
    "static_output_control",
    "structural_control",
]
