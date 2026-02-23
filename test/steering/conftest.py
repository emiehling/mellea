"""Test fixtures for steering tests."""

from dataclasses import dataclass

from mellea.steering import BackendControl, InputControl, Optimizer, Policy


@dataclass(frozen=True)
class MockBackendControl(BackendControl):
    label: str


@dataclass(frozen=True)
class MockBackendControlAlt(BackendControl):
    """Second concrete type for capability filtering tests."""

    value: float


@dataclass(frozen=True)
class MockInputControl(InputControl):
    tag: str

    def apply(self, action, ctx):
        return action, ctx  # no-op for testing


class MockOptimizer(Optimizer):
    """Minimal optimizer that returns a fixed policy."""

    def __init__(self, policy: Policy | None = None):
        self._policy = policy or Policy()

    async def compile(self, requirements, supported_controls, ctx=None, action=None):
        return self._policy
