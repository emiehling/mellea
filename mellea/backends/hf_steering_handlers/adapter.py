"""LoRA/PEFT adapter control handler for HuggingFace models."""

from __future__ import annotations

from typing import Any

from ...core.steering import Control, StructuralControlHandler


class AdapterHandler(StructuralControlHandler):
    """Loads and activates LoRA/PEFT adapters on a HuggingFace model.

    Integrates with Mellea's existing ``load_adapter`` / ``set_adapter``
    machinery on ``LocalHFBackend``. The artifact should be the adapter's
    qualified name or path.

    Expects ``control.params`` to optionally contain:

    - ``adapter_name`` (str, optional): The name to register the adapter under.
      Defaults to ``control.name``.
    """

    def activate(self, control: Control, model: Any, artifact: Any | None) -> Any:
        """Load and activate the adapter on the model.

        Args:
            control: The control descriptor.
            model: A HuggingFace ``PreTrainedModel`` with PEFT support.
            artifact: The adapter path or qualified name (str).

        Returns:
            The adapter name (str), used as the deactivation handle.
        """
        adapter_name = control.params.get("adapter_name", control.name)
        adapter_path = artifact

        if adapter_path is not None:
            model.load_adapter(adapter_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        return adapter_name

    def deactivate(self, handle: Any) -> None:
        """Deactivation is a no-op -- adapter teardown is handled by the lock scope.

        The adapter lock in ``_generate_with_adapter_lock`` manages adapter
        state transitions. We don't unload adapters eagerly because they may
        be reused on the next sampling loop iteration.

        Args:
            handle: The adapter name (unused).
        """
