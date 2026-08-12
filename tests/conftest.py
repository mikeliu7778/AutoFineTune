import pytest


@pytest.fixture
def fake_llm_factory():
    from autofinetune.llm.client import FakeLLMClient

    def _make(**handlers):
        return FakeLLMClient(handlers=handlers)

    return _make
