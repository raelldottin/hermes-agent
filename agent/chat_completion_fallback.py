"""Fallback-route budget and reasoning helpers for ``chat_completion_helpers`` (fork-local).

A fallback entry's ``max_tokens`` is a ceiling: the active request never asks a fallback for more
than it allows, and the primary's own budget comes back when the next turn restores the primary.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def seed_primary_budget(agent) -> None:
    """Record the primary's output budget in ``_primary_runtime`` before the first switch."""
    primary_runtime = getattr(agent, "_primary_runtime", None)
    if isinstance(primary_runtime, dict) and "max_tokens" not in primary_runtime:
        agent._primary_runtime = {**primary_runtime, "max_tokens": getattr(agent, "max_tokens", None)}


def apply_fallback_budget(agent, entry: dict[str, Any]) -> None:
    """Cap ``agent.max_tokens`` at the fallback entry's ceiling, starting from the primary budget."""
    from hermes_cli.fallback_config import fallback_token_budget
    original = (getattr(agent, "_primary_runtime", None) or {}).get("max_tokens", getattr(agent, "max_tokens", None))
    agent.max_tokens = fallback_token_budget(original, entry)


def bound_active_fallback(agent, ephemeral_budget: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Enforce the active fallback's budget on built request kwargs (after reasoning headroom)."""
    if not getattr(agent, "_fallback_activated", False):
        return kwargs
    from agent.gemini_native_adapter import GeminiNativeClient
    from hermes_cli.fallback_config import bound_fallback_request, fallback_token_budget
    ceiling = fallback_token_budget(getattr(agent, "max_tokens", None), {"max_tokens": ephemeral_budget})
    return bound_fallback_request(kwargs, ceiling, native_gemini=isinstance(agent.client, GeminiNativeClient))


def _reresolve_fallback_reasoning_config(agent) -> None:
    """Per-model override > global reasoning_effort (YAML False = disabled); a config load
    failure keeps the current reasoning_config rather than killing the swap."""
    try:
        # Re-resolve reasoning_config for the new fallback model (Closes #21256). Wrapped in try/except
        # because a config load failure must not kill the swap.
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config
        agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
        logger.info("Fallback %s: reasoning_config resolved: %s", agent.model, agent.reasoning_config)
    except Exception as _reasoning_err:
        logger.debug("Failed to resolve reasoning_config for fallback %s; keeping current: %s", agent.model, _reasoning_err)
