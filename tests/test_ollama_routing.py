"""Ollama endpoints: capability from /api/show, LiteLLM routing via /v1, no-tool final steps."""
import json
from unittest import mock
from unittest.mock import MagicMock

import pytest

from sift import config
from sift.intelligence import capability, llm_client
from sift.intelligence.capability import ModelCapability, detect
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.llm_client import ignores_tool_choice, litellm_kwargs
from sift.intelligence.passes.agentic import agentic_review
from sift.intelligence.passes.critic import _CRITIC_TOOL_MAX_STEPS, _critique_with_tools
from sift.intelligence.passes.pipeline import FileReviewInput

_SHOW = {"capabilities": ["completion", "tools", "thinking"], "context_length": 131072}
_OLLAMA_CFG = {
    "LLM_MODEL": "gpt-oss:120b-cloud",
    "LLM_API_BASE": "https://ollama.com",
    "LLM_API_KEY": "ollama-key",
    "SIFT_LLM_NUM_RETRIES": 3,
    "LLM_TIMEOUT": 300.0,
}


@pytest.fixture
def ollama_server(monkeypatch):
    """Pretend https://ollama.com is an Ollama server; every other base 404s."""
    probes: list[tuple] = []

    def _show(root, name, key):
        probes.append((root, name, key))
        return dict(_SHOW) if root == "https://ollama.com" else None

    monkeypatch.setattr(capability, "_ollama_show_uncached", _show)
    return probes


def _msg(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    m.role = "assistant"
    return m


def _tool_call(call_id, name, arguments):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _resp(message, finish="stop"):
    r = MagicMock()
    r.choices = [MagicMock(message=message, finish_reason=finish)]
    return r


# --- capability -------------------------------------------------------------

def test_detect_uses_ollama_show(ollama_server):
    cap = detect("gpt-oss:120b-cloud", None, api_base="https://ollama.com", api_key="k")
    assert cap.supports_function_calling is True
    assert cap.supports_reasoning is True
    assert cap.context_window == 131072
    assert ollama_server == [("https://ollama.com", "gpt-oss:120b-cloud", "k")]


def test_detect_strips_v1_and_routing_prefix_for_probe(ollama_server):
    detect("ollama_chat/gpt-oss:120b-cloud", None, api_base="https://ollama.com/v1/")
    assert ollama_server[0][:2] == ("https://ollama.com", "gpt-oss:120b-cloud")


def test_detect_falls_back_to_litellm_when_not_ollama(ollama_server):
    cap = detect("gpt-oss:120b-cloud", None, api_base="https://gateway.example/v1")
    assert cap.supports_function_calling is False   # LiteLLM doesn't know the id
    assert cap.context_window == 8192


def test_override_beats_ollama_show(ollama_server):
    cap = detect(
        "gpt-oss:120b-cloud",
        json.dumps({"supports_function_calling": False, "context_window": 4096}),
        api_base="https://ollama.com",
    )
    assert cap.supports_function_calling is False
    assert cap.context_window == 4096


def test_known_provider_prefixes_are_never_probed(ollama_server):
    detect("anthropic/claude-sonnet-4-6", None, api_base="https://ollama.com")
    detect("groq/openai/gpt-oss-120b", None, api_base="https://ollama.com")
    assert ollama_server == []


def test_probe_result_is_cached(ollama_server):
    capability.ollama_show("gpt-oss:120b-cloud", "https://ollama.com")
    capability.ollama_show("gpt-oss:120b-cloud", "https://ollama.com/v1")
    assert len(ollama_server) == 1


# --- routing ----------------------------------------------------------------

def test_litellm_kwargs_routes_ollama_through_v1(ollama_server):
    with mock.patch.multiple(config, **_OLLAMA_CFG):
        kw = litellm_kwargs()
    assert kw["model"] == "openai/gpt-oss:120b-cloud"
    assert kw["api_base"] == "https://ollama.com/v1"
    assert kw["api_key"] == "ollama-key"
    assert kw["allowed_openai_params"] == ["reasoning_effort"]
    assert kw["num_retries"] == 3
    assert kw["timeout"] == 300.0


def test_litellm_kwargs_critic_on_same_ollama_base_inherits_key(ollama_server):
    with mock.patch.multiple(config, **_OLLAMA_CFG):
        kw = litellm_kwargs("nemotron-3-ultra:cloud", "https://ollama.com", None)
    assert kw["model"] == "openai/nemotron-3-ultra:cloud"
    assert kw["api_key"] == "ollama-key"


def test_litellm_kwargs_keyless_local_ollama_gets_placeholder_key(monkeypatch):
    monkeypatch.setattr(
        capability, "_ollama_show_uncached", lambda root, name, key: dict(_SHOW)
    )
    cfg = {**_OLLAMA_CFG, "LLM_API_BASE": "http://localhost:11434", "LLM_API_KEY": None}
    with mock.patch.multiple(config, **cfg):
        kw = litellm_kwargs()
    assert kw["api_base"] == "http://localhost:11434/v1"
    assert kw["api_key"] == "ollama"


def test_litellm_kwargs_leaves_other_providers_untouched(ollama_server):
    cfg = {**_OLLAMA_CFG, "LLM_MODEL": "anthropic/claude-sonnet-4-6", "LLM_API_BASE": None}
    with mock.patch.multiple(config, **cfg):
        kw = litellm_kwargs()
        assert not ignores_tool_choice()
    assert kw["model"] == "anthropic/claude-sonnet-4-6"
    assert "allowed_openai_params" not in kw


@pytest.mark.asyncio
async def test_call_llm_on_ollama_uses_litellm_not_raw_http(ollama_server):
    fake = mock.AsyncMock(return_value=_resp(_msg(content="[]")))
    with mock.patch.multiple(config, **_OLLAMA_CFG), \
         mock.patch.object(llm_client, "acompletion", new=fake), \
         mock.patch.object(llm_client.httpx, "AsyncClient") as raw_http:
        out = await llm_client._call_llm("sys", "user", reasoning_effort="high")
    assert out == "[]"
    raw_http.assert_not_called()
    kw = fake.call_args.kwargs
    assert kw["model"] == "openai/gpt-oss:120b-cloud"
    assert kw["reasoning_effort"] == "high"


# --- final steps ------------------------------------------------------------

@pytest.mark.asyncio
async def test_agentic_ollama_non_emit_answer_falls_to_no_tools_text(ollama_server):
    """Ollama ignores tool_choice: the forced emit call may come back as another read
    tool. The findings must then come from a no-tools text call, not be lost."""
    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(131_072, 2048, True, True)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    read = _msg(tool_calls=[_tool_call("tc1", "get_file", '{"path": "app/a.py"}')])
    text = _msg(content='[{"line": 1, "severity": "bug", "title": "Bad", "body": "boom", "confidence": 9}]')
    calls: list[dict] = []

    async def _record(**kwargs):
        calls.append(kwargs)
        # Every tool-offering call (loop steps + the forced emit) answers with a read tool.
        return _resp(text if "tools" not in kwargs else read)

    with mock.patch.multiple(config, **{**_OLLAMA_CFG, "SIFT_AGENTIC_MAX_STEPS": 2}), \
         mock.patch("sift.intelligence.passes.agentic.acompletion", new=_record):
        findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})

    assert [f.line for f in findings] == [1]
    assert len(calls) == 4                     # 2 steps + forced emit + no-tools final
    final = calls[-1]
    assert "tools" not in final and "tool_choice" not in final
    assert "Do not call any tools" in final["messages"][-1]["content"]
    assert final["model"] == "openai/gpt-oss:120b-cloud"


@pytest.mark.asyncio
async def test_critic_final_on_ollama_sends_no_tools(ollama_server, monkeypatch):
    import litellm

    tool = _msg(tool_calls=[_tool_call("tc1", "get_file", '{"path": "app/x.py"}')])
    verdict = _msg(content='{"verdict": "keep", "reason": "ok"}')
    calls: list[dict] = []

    async def _record(**kwargs):
        calls.append(kwargs)
        return _resp(tool if len(calls) <= _CRITIC_TOOL_MAX_STEPS else verdict)

    monkeypatch.setattr(litellm, "acompletion", _record)
    with mock.patch.multiple(config, **_OLLAMA_CFG):
        out = await _critique_with_tools(
            system="sys",
            user_content="u",
            llm_kw={"model": "nemotron-3-ultra:cloud", "api_base": "https://ollama.com", "api_key": None},
            repo_root="/tmp",
            path_to_content={"app/x.py": "x = 1\n"},
            mod_funcs={},
            reasoning_effort="high",
        )

    assert out == '{"verdict": "keep", "reason": "ok"}'
    final = calls[-1]
    assert "tools" not in final and "tool_choice" not in final
    assert final["reasoning_effort"] == "high"
    assert all(c["model"] == "openai/nemotron-3-ultra:cloud" for c in calls)
    assert all(c["api_key"] == "ollama-key" for c in calls)
