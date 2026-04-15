"""Built-in backend-agnostic input control handlers.

These handlers operate on ``Component`` and ``CBlock`` objects before the formatter
runs, making them portable across all backends. They are registered globally at
module load time via ``register_global_input_handler``.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from ..core.base import CBlock, Component, Context
from ..core.steering import Control, InputControlHandler, register_global_input_handler
from ..stdlib.components.chat import Message


class SystemPromptInjectionHandler(InputControlHandler):
    """Prepends a system message to the context.

    Expects ``control.params`` to contain:

    - ``system_prompt`` (str): The system message content to inject.
    """

    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        context: Context,
        artifact: Any | None,
    ) -> tuple[Component | CBlock, Context]:
        """Inject a system message into the context.

        Args:
            control: The control descriptor; ``params["system_prompt"]`` is required.
            action: The current action (returned unchanged).
            context: The current generation context.
            artifact: Unused.

        Returns:
            The unchanged action and a new context with the system message added.
        """
        system_prompt = control.params["system_prompt"]
        system_msg = Message(role="system", content=system_prompt)
        return action, context.add(system_msg)


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
        context: Context,
        artifact: Any | None,
    ) -> tuple[Component | CBlock, Context]:
        """Rewrite an Instruction's description using a template.

        Args:
            control: The control descriptor; ``params["template"]`` is required.
            action: The current action. If an ``Instruction``, its description
                is rewritten; otherwise returned unchanged.
            context: The current generation context (returned unchanged).
            artifact: Unused.

        Returns:
            A ``(action, context)`` tuple with the potentially rewritten action.
        """
        from ..stdlib.components.instruction import Instruction

        if isinstance(action, Instruction):
            template = control.params["template"]
            original_text = str(action._description) if action._description else ""
            rewritten = template.format(original=original_text)
            new_action = deepcopy(action)
            new_action._description = CBlock(rewritten)
            return new_action, context
        return action, context


class ContextPrefixHandler(InputControlHandler):
    """Prepends a user message to the context as grounding/priming.

    Expects ``control.params`` to contain:

    - ``content`` (str): The message content to prepend.
    - ``role`` (str, optional): The message role. Defaults to ``"user"``.
    """

    def apply(
        self,
        control: Control,
        action: Component | CBlock,
        context: Context,
        artifact: Any | None,
    ) -> tuple[Component | CBlock, Context]:
        """Add a context message as grounding.

        Args:
            control: The control descriptor; ``params["content"]`` is required.
            action: The current action (returned unchanged).
            context: The current generation context.
            artifact: Unused.

        Returns:
            The unchanged action and a new context with the message added.
        """
        content = control.params["content"]
        role = control.params.get("role", "user")
        msg = Message(role=role, content=content)
        return action, context.add(msg)


class ICLExampleSelectorHandler(InputControlHandler):
    """Selects in-context learning examples from an artifact example pool.

    The artifact should be a list of example strings (as loaded from a
    ``PromptStore`` example pool). Selects ``count`` examples using the
    specified ``strategy`` and injects them into the context as messages.

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
        context: Context,
        artifact: Any | None,
    ) -> tuple[Component | CBlock, Context]:
        """Select examples from the pool and add them to context.

        Args:
            control: The control descriptor.
            action: The current action (returned unchanged).
            context: The current generation context.
            artifact: A list of example strings from the example pool.

        Returns:
            The unchanged action and a new context with examples prepended.
        """
        if artifact is None:
            return action, context

        pool: list[str] = artifact
        count = control.params.get("count", 3)
        strategy = control.params.get("strategy", "first")
        role = control.params.get("role", "user")

        count = min(count, len(pool))

        if strategy == "random":
            selected = random.sample(pool, count)
        else:
            selected = pool[:count]

        new_context = context
        for example in selected:
            new_context = new_context.add(Message(role=role, content=example))
        return action, new_context


# Register built-in handlers globally at import time.
register_global_input_handler("system_prompt_injection", SystemPromptInjectionHandler())
register_global_input_handler("instruction_rewrite", InstructionRewriteHandler())
register_global_input_handler("context_prefix", ContextPrefixHandler())
register_global_input_handler("icl_example_selector", ICLExampleSelectorHandler())
