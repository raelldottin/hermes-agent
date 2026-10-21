"""Fallback routes honor output ceilings without leaking them into other routes."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent import auxiliary_client as ac


def _agent(budget, entries):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            max_tokens=budget, fallback_model=entries,
        )


@pytest.mark.parametrize("budget", [None, 4096, 131072])
@pytest.mark.parametrize("restore_after_first", [True, False])
def test_main_fallback_caps_from_primary_budget_and_restores_it(budget, restore_after_first):
    entries = [
        {"provider": "google", "model": "gemini-2.5-flash", "max_tokens": 16384},
        {"provider": "openai", "model": "gpt-4.1"},
    ]
    agent = _agent(budget, entries)
    client = MagicMock(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key="test-fallback")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gemini-2.5-flash")):
        assert agent._try_activate_fallback()
        assert agent.max_tokens == (16384 if budget is None else min(budget, 16384))
        if not restore_after_first:
            assert agent._try_activate_fallback()
            assert agent.max_tokens == budget
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._restore_primary_runtime()
    assert agent.max_tokens == budget


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, float("inf"), float("nan"), "16384"])
def test_invalid_main_fallback_budget_cannot_replace_the_primary_budget(invalid):
    agent = _agent(4096, [{"provider": "google", "model": "gemini-2.5-flash", "max_tokens": invalid}])
    client = MagicMock(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key="test-fallback")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gemini-2.5-flash")):
        assert agent._try_activate_fallback()
    assert agent.max_tokens == 4096


@pytest.mark.parametrize("budget,expected", [(None, 16384), (4096, 4096), (131072, 16384)])
@pytest.mark.parametrize("model,wire", [("gemini-2.5-flash", "max_tokens"), ("gpt-5", "max_completion_tokens")])
def test_auxiliary_candidate_caps_real_wire_without_mutating_caller(budget, expected, model, wire, monkeypatch):
    monkeypatch.setattr(ac, "_replan_synchronous_cache_sections", lambda messages, tools, **kw: (messages, tools))
    entry = {"provider": "google", "model": model, "max_tokens": 16384}
    destination = ac._FallbackDestination("google", "https://generativelanguage.googleapis.com/v1beta/openai/", "chat_completions", model)
    original = dict(task=None, messages=[], tools=None, temperature=None, max_tokens=budget,
                    effective_timeout=30, effective_extra_body={}, reasoning_config=None,
                    task_config={}, apply_fast_lane=False)
    first = ac._fallback_request_kwargs(destination, fallback_entry=entry, **original)
    assert first[wire] == expected
    second = ac._fallback_request_kwargs(destination, fallback_entry={}, **original)
    assert second.get(wire) == budget
    assert original["max_tokens"] == budget
    assert entry["max_tokens"] == 16384


def test_auxiliary_global_fallback_entry_cap_reaches_candidate(monkeypatch):
    entry = {"provider": "google", "model": "gemini-2.5-flash", "max_tokens": 16384}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"fallback_providers": [entry]})
    monkeypatch.setattr(ac, "_replan_synchronous_cache_sections", lambda messages, tools, **kw: (messages, tools))
    client = SimpleNamespace(base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    destination, kwargs, rebuild = ac._plan_fallback_candidate(
        client, "gemini-2.5-flash", "fallback_providers[0](google)", task=None,
        effective_timeout=30, apply_fast_lane=False, messages=[], tools=None, temperature=None,
        max_tokens=131072, effective_extra_body={}, reasoning_config=None,
    )
    assert kwargs["max_tokens"] == 16384
    _, retried = rebuild(destination.provider, client, destination.model)
    assert retried["max_tokens"] == 16384


@pytest.mark.parametrize("budget,expected", [(32, 32), (131072, 16384)])
def test_main_native_gemini_fallback_ceiling_reaches_http_payload(budget, expected):
    from agent.chat_completion_helpers import build_api_kwargs
    from agent.gemini_native_adapter import GeminiNativeClient

    payloads = []

    class HTTP:
        def post(self, url, **kwargs):
            payloads.append(kwargs["json"])
            return SimpleNamespace(status_code=200, json=lambda: {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            })

    client = GeminiNativeClient(api_key="test-native", http_client=HTTP())
    agent = _agent(budget, [{"provider": "google", "model": "gemini-2.5-flash", "max_tokens": 16384}])
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gemini-2.5-flash")):
        assert agent._try_activate_fallback()
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}], [])
    assert kwargs.get("max_tokens") == expected
    client.chat.completions.create(**kwargs)
    generation = payloads[0]["generationConfig"]
    assert generation["maxOutputTokens"] == expected
    if expected == 32:
        assert generation["thinkingConfig"]["thinkingBudget"] == 0
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._restore_primary_runtime()
    assert agent.max_tokens == budget


def test_native_auxiliary_fallback_ceiling_reaches_http_payload(monkeypatch):
    from agent.gemini_native_adapter import GeminiNativeClient

    payloads = []

    class HTTP:
        def post(self, url, **kwargs):
            payloads.append(kwargs["json"])
            return SimpleNamespace(status_code=200, json=lambda: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(ac, "_replan_synchronous_cache_sections", lambda messages, tools, **kw: (messages, tools))
    client = GeminiNativeClient(api_key="test-native", http_client=HTTP())
    destination = ac._FallbackDestination("google", client.base_url, "chat_completions", "gemini-2.5-flash")
    kwargs = ac._fallback_request_kwargs(
        destination, task=None, messages=[{"role": "user", "content": "hello"}], tools=None,
        temperature=None, max_tokens=131072, effective_timeout=30, effective_extra_body={},
        reasoning_config={"enabled": True, "effort": "high"}, task_config={}, apply_fast_lane=False,
        fallback_entry={"provider": "google", "model": "gemini-2.5-flash", "max_tokens": 16384},
    )
    client.chat.completions.create(**kwargs)
    assert payloads[0]["generationConfig"]["maxOutputTokens"] == 16384


def test_compatible_gemini_fallback_ceiling_does_not_send_native_keyword():
    from agent.chat_completion_helpers import build_api_kwargs

    client = MagicMock(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key="test-fallback")
    agent = _agent(131072, [{"provider": "google", "model": "gemini-2.5-flash", "max_tokens": 16384}])
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gemini-2.5-flash")):
        assert agent._try_activate_fallback()
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}], [])
    assert kwargs["max_tokens"] == 16384
    assert "max_output_tokens_ceiling" not in kwargs


def test_native_fallback_preserves_ephemeral_budget_with_high_reasoning():
    from agent.chat_completion_helpers import build_api_kwargs
    from agent.gemini_native_adapter import GeminiNativeClient

    payloads = []

    class HTTP:
        def post(self, url, **kwargs):
            payloads.append(kwargs["json"])
            return SimpleNamespace(status_code=200, json=lambda: {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            })

    client = GeminiNativeClient(api_key="test-native", http_client=HTTP())
    agent = _agent(131072, [{"provider": "google", "model": "gemini-2.5-flash", "max_tokens": 16384}])
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gemini-2.5-flash")):
        assert agent._try_activate_fallback()
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    agent._ephemeral_max_output_tokens = 32
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}], [])
    client.chat.completions.create(**kwargs)
    assert payloads[0]["generationConfig"]["maxOutputTokens"] == 32
    assert payloads[0]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    assert agent.max_tokens == 16384


def test_responses_fallback_budget_is_bounded_without_mutating_request():
    from hermes_cli.fallback_config import bound_fallback_request

    request = {"max_output_tokens": 131072, "model": "test-model"}
    bounded = bound_fallback_request(request, 16384)
    assert bounded["max_output_tokens"] == 16384
    assert request["max_output_tokens"] == 131072
