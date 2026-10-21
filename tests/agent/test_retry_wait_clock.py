"""Retry timers preserve their duration when the system wall clock changes."""
from types import SimpleNamespace

import pytest

from agent import turn_recovery


def test_retry_wait_uses_elapsed_time_despite_wall_clock_jump(monkeypatch):
    clock = SimpleNamespace(elapsed=0.0)
    wall_times = iter((100.0, 1000.0))
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.elapsed += seconds

    monkeypatch.setattr(turn_recovery.time, "time", lambda: next(wall_times, 1000.0))
    monkeypatch.setattr(turn_recovery.time, "monotonic", lambda: clock.elapsed)
    monkeypatch.setattr(turn_recovery.time, "sleep", sleep)
    agent = SimpleNamespace(_interrupt_requested=False, _adopt_nous_key_before_expiry=lambda: None)
    result = turn_recovery.interruptible_backoff_sleep(
        agent, 1.0, None, messages=[], conversation_history=[], api_call_count=0,
        abort_message="stopped", interrupt_text="stopped", activity_label="retry",
    )
    assert result is None
    assert sleeps
    assert 1.0 <= clock.elapsed <= 1.2


def test_retry_wait_remains_interruptible(monkeypatch):
    clock = SimpleNamespace(elapsed=0.0)
    agent = SimpleNamespace(_interrupt_requested=False, _adopt_nous_key_before_expiry=lambda: None)

    def sleep(seconds):
        clock.elapsed += seconds
        agent._interrupt_requested = True

    monkeypatch.setattr(turn_recovery.time, "monotonic", lambda: clock.elapsed)
    monkeypatch.setattr(turn_recovery.time, "sleep", sleep)
    monkeypatch.setattr(turn_recovery, "abort_turn_on_interrupt", lambda *args, **kwargs: {"interrupted": True})
    result = turn_recovery.interruptible_backoff_sleep(
        agent, 120.0, None, messages=[], conversation_history=[], api_call_count=0,
        abort_message="stopped", interrupt_text="stopped", activity_label="retry",
    )
    assert result == {"interrupted": True}
    assert clock.elapsed == 0.2


@pytest.mark.parametrize("headers,body,expected", [({"Retry-After": "17"}, {}, 17), ({}, {"error": {"retry_after": 23}}, 23)])
def test_nvidia_explicit_retry_after_wins_over_adaptive_policy(monkeypatch, headers, body, expected):
    from agent import retry_utils

    def unexpected_adaptive_policy(*args, **kwargs):
        pytest.fail("A provider's explicit reset window must win")

    monkeypatch.setattr(retry_utils, "adaptive_rate_limit_backoff", unexpected_adaptive_policy)
    error = SimpleNamespace(status_code=429, response=SimpleNamespace(headers=headers), body=body)
    agent = SimpleNamespace(
        _buffer_diagnostic_status=lambda text: None,
        _emit_diagnostic_wait=lambda text: None,
        _client_log_context=lambda: "test",
    )
    wait = turn_recovery.compute_error_backoff(
        agent, error, retry_count=1, max_retries=3, is_rate_limited=True,
        is_zai_coding_overload=False, base_url="https://integrate.api.nvidia.com/v1", model="nvidia/nemotron",
    )
    assert wait == expected
