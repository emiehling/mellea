"""Unit tests for the portable StaticOutputControlHandler."""

from mellea.core.steering import Control, ControlCategory
from mellea.steering.handlers import StaticOutputControlHandler


def test_static_output_merges_params():
    handler = StaticOutputControlHandler()
    control = Control(
        category=ControlCategory.OUTPUT,
        name="static_output",
        params={"temperature": 0.3, "top_p": 0.9},
    )
    gen_kwargs = {"max_new_tokens": 100}
    result = handler.apply(control, gen_kwargs, None)
    assert result["temperature"] == 0.3
    assert result["top_p"] == 0.9
    assert result["max_new_tokens"] == 100
