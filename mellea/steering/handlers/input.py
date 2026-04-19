"""Built-in backend-agnostic input control handlers.

These handlers operate on the **linearized** context (``list[Component | CBlock]``)
before the formatter runs, making them portable across all backends and all
``Context`` subclasses. They are registered globally at module load time via
``register_global_input_handler``.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from ...core.base import CBlock, Component
from ...core.steering import Control, InputControlHandler, register_global_input_handler
from ...stdlib.components.chat import Message


class SystemPromptInjectionHandler(InputControlHandler):
    """Prepends a system message to the linearized context.

    Expects ``control.params`` to contain:

    - ``system_prompt`` (str): The system message content to inject.
    """

    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        linearized_ctx: list[Component | CBlock],
        artifact: Any | None,
    ) -> tuple[Component | CBlock, list[Component | CBlock]]:
        """Inject a system message at the front of the linearized context."""
        system_prompt = control.params["system_prompt"]
        system_msg = Message(role="system", content=system_prompt)
        return action, [system_msg, *linearized_ctx]


class InstructionRewriteHandler(InputControlHandler):
    """Rewrites the instruction description using a Python format string template.

    Expects ``control.params`` to contain:

    - ``template`` (str): A format string with an ``{original}`` placeholder
      that will be replaced with the current instruction description.

    If the action is not an ``Instruction``, it is returned unchanged.
    """

    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        linearized_ctx: list[Component | CBlock],
        artifact: Any | None,
    ) -> tuple[Component | CBlock, list[Component | CBlock]]:
        """Rewrite an Instruction's description using a template."""
        from ...stdlib.components.instruction import Instruction

        if isinstance(action, Instruction):
            template = control.params["template"]
            original_text = str(action._description) if action._description else ""
            rewritten = template.format(original=original_text)
            new_action = deepcopy(action)
            new_action._description = CBlock(rewritten)
            return new_action, linearized_ctx
        return action, linearized_ctx


class ContextPrefixHandler(InputControlHandler):
    """Prepends a message to the linearized context as grounding/priming.

    Expects ``control.params`` to contain:

    - ``content`` (str): The message content to prepend.
    - ``role`` (str, optional): The message role. Defaults to ``"user"``.
    """

    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        linearized_ctx: list[Component | CBlock],
        artifact: Any | None,
    ) -> tuple[Component | CBlock, list[Component | CBlock]]:
        """Add a context message as grounding."""
        content = control.params["content"]
        role = control.params.get("role", "user")
        msg = Message(role=role, content=content)
        return action, [msg, *linearized_ctx]


class ICLExampleSelectorHandler(InputControlHandler):
    """Selects in-context learning examples from an artifact example pool.

    The artifact should be a list of example strings (as loaded from a
    ``PromptStore`` example pool). Selects ``count`` examples using the
    specified ``strategy`` and injects them into the linearized context
    as messages.

    Expects ``control.params`` to optionally contain:

    - ``count`` (int): Number of examples to select. Defaults to ``3``.
    - ``strategy`` (str): Selection strategy — ``"first"`` (default) takes
      the first N examples, ``"random"`` samples uniformly.
    - ``role`` (str): Message role for injected examples. Defaults to ``"user"``.
    """

    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        linearized_ctx: list[Component | CBlock],
        artifact: Any | None,
    ) -> tuple[Component | CBlock, list[Component | CBlock]]:
        """Select examples from the pool and prepend them to the linearized context."""
        if artifact is None:
            return action, linearized_ctx

        pool: list[str] = artifact
        count = control.params.get("count", 3)
        strategy = control.params.get("strategy", "first")
        role = control.params.get("role", "user")

        count = min(count, len(pool))

        if strategy == "random":
            selected = random.sample(pool, count)
        else:
            selected = pool[:count]

        examples = [Message(role=role, content=ex) for ex in selected]
        return action, [*examples, *linearized_ctx]


# Register built-in handlers globally at import time.
register_global_input_handler("system_prompt_injection", SystemPromptInjectionHandler())
register_global_input_handler("instruction_rewrite", InstructionRewriteHandler())
register_global_input_handler("context_prefix", ContextPrefixHandler())
register_global_input_handler("icl_example_selector", ICLExampleSelectorHandler())
