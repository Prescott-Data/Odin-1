"""Test-only support for bootstrap tests that deliberately stub NPLL models."""

from dataclasses import asdict
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def mock_npll_config(monkeypatch):
    import npll.bootstrap
    from npll.utils.config import NPLLConfig

    monkeypatch.setattr(
        npll.bootstrap, "asdict",
        lambda value: asdict(NPLLConfig(device="cpu")) if isinstance(value, Mock) else asdict(value),
    )
