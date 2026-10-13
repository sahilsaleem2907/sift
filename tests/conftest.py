# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from sift.intelligence import capability


@pytest.fixture(autouse=True)
def _no_ollama_probe(monkeypatch):
    """Never hit the network for Ollama /api/show during tests.

    config.LLM_API_BASE comes from the developer's .env, so without this every test that
    builds LiteLLM kwargs would probe that server. Tests that exercise Ollama routing patch
    ``_ollama_show_uncached`` themselves.
    """
    monkeypatch.setattr(capability, "_ollama_show_uncached", lambda root, name, key: None)
    capability._OLLAMA_SHOW_CACHE.clear()
    capability._CACHE.clear()
    yield
    capability._OLLAMA_SHOW_CACHE.clear()
    capability._CACHE.clear()
