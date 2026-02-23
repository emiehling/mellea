"""Concrete steering control implementations.

Each control lives in its own module with a ``CONTROL_INFO`` metadata dict.
Use ``control_registry()`` to discover all available controls at runtime.
"""

import importlib
import pkgutil
from typing import Any

from .few_shot import FewShot
from .grounding import Grounding
from .stop_sequence import StopSequence
from .temperature import Temperature

__all__ = [
    "FewShot",
    "Grounding",
    "StopSequence",
    "Temperature",
    "control_registry",
]


def control_registry() -> dict[str, dict[str, Any]]:
    """Discover all controls in this package that expose CONTROL_INFO.

    Scans sub-modules of ``mellea.stdlib.controls`` for a module-level
    ``CONTROL_INFO`` dict and returns a mapping of module name to info.

    Returns:
        A dict mapping module names to their ``CONTROL_INFO`` dicts.
    """
    registry: dict[str, dict[str, Any]] = {}
    package_path = __path__
    for finder, name, _ispkg in pkgutil.iter_modules(package_path):
        mod = importlib.import_module(f".{name}", __package__)
        info = getattr(mod, "CONTROL_INFO", None)
        if info is not None:
            registry[name] = info
    return registry
