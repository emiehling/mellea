"""Verify SamplingStrategy ABC accepts policy/optimizer parameters."""

import inspect

import pytest

from mellea.core import SamplingStrategy
from mellea.stdlib.sampling import RejectionSamplingStrategy


class TestSamplingStrategySig:
    def test_abc_sample_has_policy_param(self):
        """SamplingStrategy.sample ABC has policy parameter."""
        sig = inspect.signature(SamplingStrategy.sample)
        params = list(sig.parameters.keys())
        assert "policy" in params
        assert "optimizer" in params

    def test_concrete_sample_has_policy_param(self):
        """RejectionSamplingStrategy.sample has policy parameter."""
        sig = inspect.signature(RejectionSamplingStrategy.sample)
        params = list(sig.parameters.keys())
        assert "policy" in params
        assert "optimizer" in params
        # verify they have defaults
        assert sig.parameters["policy"].default is None
        assert sig.parameters["optimizer"].default is None

    def test_sample_signature_compatible(self):
        """Verify concrete implementation has compatible signature with ABC."""
        abc_sig = inspect.signature(SamplingStrategy.sample)
        concrete_sig = inspect.signature(RejectionSamplingStrategy.sample)

        # all ABC params should be in concrete
        for param in abc_sig.parameters:
            assert param in concrete_sig.parameters, (
                f"Missing parameter {param} in RejectionSamplingStrategy.sample"
            )
