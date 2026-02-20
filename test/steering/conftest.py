"""Test fixtures for steering tests."""

from dataclasses import dataclass

from mellea.steering import (
    InputControl,
    OutputControl,
    StateControl,
    SteeringOptimizer,
    SteeringPolicy,
)


@dataclass(frozen=True)
class MockStateControl(StateControl):
    label: str


@dataclass(frozen=True)
class MockOutputControl(OutputControl):
    value: float


@dataclass(frozen=True)
class MockInputControl(InputControl):
    tag: str

    def apply(self, action, ctx):
        return action, ctx  # no-op for testing


class MockOptimizer(SteeringOptimizer):
    """Minimal optimizer that returns a fixed policy."""

    def __init__(self, policy: SteeringPolicy | None = None):
        self._policy = policy or SteeringPolicy()

    async def compile(self, requirements, capabilities, ctx=None, action=None):
        return self._policy
