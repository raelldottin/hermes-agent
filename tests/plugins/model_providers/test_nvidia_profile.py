"""NVIDIA NIM GLM-5.x reasoning-effort wire mapping (live-verified 2026-10-01).

Before this profile mapped reasoning, a configured ``agent.reasoning_effort`` — and
Hermes's thinking-only-truncation recovery override (``enabled: False``) — never
reached the NIM wire: the chat-completions transport's reasoning extra_body gate
(``_supports_reasoning_extra_body``) covers OpenRouter/Nous/GitHub/LM Studio/Ollama
only, and the nvidia profile declared no ``build_api_kwargs_extras``. GLM-5.3 then ran
at the server-default thinking tier regardless of configuration: observed live as
finish_reason='length' with ~60-67K chars of reasoning against ~200 chars of visible
content — kanban worker sessions died retrying identical parameter-less requests.

Wire facts (live-verified on z-ai/glm-5.3, integrate.api.nvidia.com, 2026-10-01):
- ``reasoning_effort`` low/high/max: accepted, thinking scales monotonically
  (low=26 vs default 1657 reasoning tokens on the same 30K prompt).
- ``ultra`` (Hermes-internal, above max): HTTP 400.
- ``medium``: accepted by the endpoint but NOT in NVIDIA's documented vocabulary
  (low/high/max) — omitted from the declared set, clamps down to ``low``.
- ``none`` / ``chat_template_kwargs.enable_thinking: false``: ACCEPTED but do NOT
  suppress thinking on large prompts (1657/2048 and 2048/2048 reasoning tokens,
  near-zero visible content) — unusable as a recovery disable. The recovery override
  therefore maps a disable to ``low``, the documented floor that demonstrably
  defers thinking and lets the budget reach the answer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture()
def nvidia_profile():
    """Resolve the registered NIM profile through the real plugin discovery path."""
    import model_tools

    from providers import get_provider_profile

    profile = get_provider_profile("nvidia")
    assert profile is not None, "nvidia provider profile must be registered"
    return profile


class TestNimGlm5Vocabulary:
    """``supported_reasoning_efforts`` — the documented vocabulary, GLM-5.x only."""

    def test_glm5_declares_documented_levels(self, nvidia_profile):
        assert nvidia_profile.supported_reasoning_efforts("z-ai/glm-5.3") == ("low", "high", "max")

    @pytest.mark.parametrize("model", ["z-ai/glm-5.3", "z-ai/glm-5.3-flash", "glm-5.3", "z-ai/glm-5.2"])
    def test_glm5_family_matches(self, nvidia_profile, model):
        assert nvidia_profile.supported_reasoning_efforts(model) == ("low", "high", "max")

    @pytest.mark.parametrize("model", ["nvidia/llama-3.3-70b-instruct", "glm-4.6", "qwen/qwen3-235b-a22b", "", None])
    def test_other_models_undeclared(self, nvidia_profile, model):
        """Only GLM-5.x wire behavior is verified on this endpoint."""
        assert nvidia_profile.supported_reasoning_efforts(model) is None

    def test_glm4_boundary_not_matched(self, nvidia_profile):
        assert nvidia_profile.supported_reasoning_efforts("glm-4.5") is None


class TestNimGlm5EffortMapping:
    """``build_api_kwargs_extras`` — effort → top-level reasoning_effort."""

    @pytest.mark.parametrize("effort,wire", [
        ("ultra", "max"),   # Hermes-internal top: clamps down to NIM's max
        ("max", "max"),
        ("xhigh", "high"),  # nearest weaker — a clamp never escalates
        ("high", "high"),
        ("medium", "low"),  # not in the documented vocabulary: canonical never-escalate clamp
        ("low", "low"),
        ("minimal", "low"),
    ])
    def test_enabled_effort_maps_to_wire(self, nvidia_profile, effort, wire):
        _extra, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="z-ai/glm-5.3",
        )
        assert top_level == {"reasoning_effort": wire}
        assert _extra == {}

    def test_unset_omits_field_server_default_applies(self, nvidia_profile):
        """No user preference → no parameter → NIM's server default (max) runs."""
        for rc in (None, {}, {"enabled": True, "effort": ""}):
            extra, top = nvidia_profile.build_api_kwargs_extras(reasoning_config=rc, model="z-ai/glm-5.3")
            assert extra == {} and top == {}

    def test_disable_maps_to_documented_floor(self, nvidia_profile):
        """The one-shot thinking-only-truncation recovery override (``enabled: False``)
        must reach the wire as ``low``: live-verified as the only disable mechanism
        that actually defers thinking (none / enable_thinking:false are accepted but
        do not suppress reasoning on large prompts)."""
        _extra, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "none"}, model="z-ai/glm-5.3",
        )
        assert top_level == {"reasoning_effort": "low"}

    def test_bare_disable_also_maps_to_floor(self, nvidia_profile):
        _extra, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="z-ai/glm-5.3",
        )
        assert top_level == {"reasoning_effort": "low"}

    @pytest.mark.parametrize("model", ["nvidia/llama-3.3-70b-instruct", "glm-4.6", "", None])
    def test_non_glm5_models_get_no_reasoning_params(self, nvidia_profile, model):
        """Unverified families must not receive parameters they may 400 on."""
        extra, top = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model,
        )
        assert extra == {} and top == {}


class TestNimTransportIntegration:
    """End-to-end through the chat-completions transport: the profile's extras
    merge into top-level kwargs (not extra_body), for every effort level."""

    def test_transport_puts_reasoning_effort_top_level(self, nvidia_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        for effort, wire in (("ultra", "max"), ("high", "high"), ("low", "low"), ("medium", "low")):
            kwargs = transport.build_kwargs(
                provider_profile=nvidia_profile,
                model="z-ai/glm-5.3",
                messages=[{"role": "user", "content": "OK?"}],
                tools=None,
                base_url="https://integrate.api.nvidia.com/v1",
                max_tokens=512,
                reasoning_config={"enabled": True, "effort": effort},
                supports_reasoning=False,  # NIM: extra_body gate is off — must still work
            )
            assert kwargs.get("reasoning_effort") == wire, (
                f"effort {effort!r} must reach the wire as {wire!r}, got {kwargs.get('reasoning_effort')!r}"
            )
            assert "reasoning" not in (kwargs.get("extra_body") or {})


class _StubAgent:
    """Minimal agent surface ``recover_from_truncation`` touches on the
    thinking-only text-continuation path — enough to drive the truncation handler
    without constructing a real AIAgent (whose init probes the Hermes home)."""

    def __init__(self):
        self.log_prefix = "[test] "
        self.api_mode = "chat_completions"
        self.quiet_mode = True
        self.context_compressor = None
        self._ephemeral_reasoning_off = False
        self._session_messages = []
        self.reasoning_config = {"enabled": True, "effort": "ultra"}
        self.tool_calls_dropped: list[Any] = []

    # --- surface used by recover_from_truncation/_continue_text ---
    def _vprint(self, *a, **k):
        pass

    def _flush_status_buffer(self, *a, **k):
        pass

    def _build_assistant_message(self, assistant_message, finish_reason):
        content = getattr(assistant_message, "content", None) or ""
        return {"role": "assistant", "content": content}

    def _has_content_after_think_block(self, content):
        return False

    def _strip_think_blocks(self, content):
        return content

    def _interim_assistant_visible_text(self, msg):
        return ""

    def _emit_interim_assistant_message(self, msg):
        pass

    def _emit_diagnostic_status(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _get_messages_up_to_last_assistant(self, messages):
        return messages

    def _get_transport(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        return ChatCompletionsTransport()

    def _try_activate_fallback(self, *a, **k):
        return False

    @property
    def _fallback_index(self):
        return 0

    @property
    def _fallback_chain(self):
        return []


@pytest.mark.allow_real_home_io
class TestNimTruncationRecoveryChain:
    """The full recovery chain, deterministically, without AIAgent:

    thinking-only truncation → recover_from_truncation arms the one-shot
    reasoning-off override → the NVIDIA profile converts it → the NEXT request's
    wire kwargs contain reasoning_effort=low (the working disable) → the request
    after that restores the configured effort.
    """

    def test_recovery_override_reaches_wire_then_restores(self, nvidia_profile):
        from agent.turn_retry_state import TurnRetryState
        from agent.turn_truncation import recover_from_truncation
        from agent.transports.chat_completions import ChatCompletionsTransport
        from hermes_constants import FINISH_REASON_LENGTH

        agent = _StubAgent()
        transport = ChatCompletionsTransport()

        def _wire_effort() -> str | None:
            kwargs = transport.build_kwargs(
                provider_profile=nvidia_profile,
                model="z-ai/glm-5.3",
                messages=[{"role": "user", "content": "OK?"}],
                tools=None,
                base_url="https://integrate.api.nvidia.com/v1",
                max_tokens=512,
                reasoning_config=agent.reasoning_config,
                supports_reasoning=False,
            )
            return kwargs.get("reasoning_effort")

        # Before truncation: configured ultra goes out clamped to max.
        assert _wire_effort() == "max"

        # A thinking-only length truncation: content empty, no tool calls.
        response = SimpleNamespace(
            id="chatcmpl-thinking-only",
            model="z-ai/glm-5.3",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="", tool_calls=None),
                finish_reason=FINISH_REASON_LENGTH,
            )],
            usage=None,
        )
        verdict = recover_from_truncation(
            agent, response, FINISH_REASON_LENGTH, TurnRetryState(),
            messages=[{"role": "user", "content": "write a long report"}],
            conversation_history=[],
            api_kwargs={},
            api_call_count=1,
            effective_task_id=None,
            current_turn_user_idx=0,
            length_continue_retries=0,
            truncated_response_parts=[],
            truncated_tool_call_retries=0,
            retry_count=0,
            compression_attempts=0,
        )

        # The handler armed the one-shot reasoning-off override for the continuation.
        assert agent._ephemeral_reasoning_off is True, "truncation handler must arm the override"
        assert verdict.action == "break"  # restart_with_length_continuation set

        # The recovery override shape ({enabled: False}) must reach the NIM wire
        # as reasoning_effort=low — the live-verified working disable.
        agent.reasoning_config = {"enabled": False, "effort": "none"}
        assert _wire_effort() == "low", (
            "the one-shot recovery override must reach the wire as reasoning_effort=low"
        )

        # The next ordinary request restores the configured effort.
        agent.reasoning_config = {"enabled": True, "effort": "ultra"}
        assert _wire_effort() == "max"
