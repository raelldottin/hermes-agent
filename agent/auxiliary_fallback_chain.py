"""Resolve an auxiliary fallback label back to its config entry (fork-local sibling of auxiliary_client)."""
from __future__ import annotations

import re
from typing import Any, Optional


def _task_config(task: str) -> dict[str, Any]:
    from agent.auxiliary_client import _get_auxiliary_task_config
    return _get_auxiliary_task_config(task)


def _fallback_chain_entry(task: Optional[str], fb_label: str) -> Optional[dict[str, Any]]:
    """Resolve a task ``fallback_chain`` or global ``fallback_providers`` label's entry,
    or None when the label is not a configured-chain candidate or the index no longer resolves."""
    if not fb_label:
        return None
    m = re.match(r"(fallback_chain|fallback_providers)\[(\d+)\]", fb_label)
    if not m:
        return None
    try:
        if m.group(1) == "fallback_providers":
            from hermes_cli.config import load_config_readonly
            from hermes_cli.fallback_config import get_fallback_chain
            chain = get_fallback_chain(load_config_readonly())
        else:
            chain = _task_config(task).get("fallback_chain") if task else None
        entry = chain[int(m.group(2))] if isinstance(chain, list) else None
    except Exception:  # health: allow BLE001 -- moved from auxiliary_client; a bad label or config read means "no entry", never a failed call
        return None
    return entry if isinstance(entry, dict) else None
