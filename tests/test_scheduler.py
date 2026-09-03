import pytest

from generative_training_audit.scheduler import (
    audit_transitions,
    build_transitions,
    transitions_from_scheduler,
)


def test_transition_count_uses_executed_edges_not_state_count():
    trace = build_transitions([1.0, 0.5, 0.0])
    assert len(trace) == 2
    assert audit_transitions(trace, expected_steps=2) == []


def test_endpoint_mismatch_is_reported():
    trace = build_transitions([1.0, 0.5, 0.1])
    issues = audit_transitions(trace, expected_start=1.0, expected_end=0.0)
    assert issues == ["end time 0.1 != expected 0.0"]


def test_sigma_length_must_match_times():
    with pytest.raises(ValueError, match="same length"):
        build_transitions([1.0, 0.0], [1.0])


def test_diffusers_like_scheduler_adapter():
    class FakeScheduler:
        timesteps = [999, 500]
        sigmas = [1.0, 0.4, 0.0]

    trace = transitions_from_scheduler(FakeScheduler(), terminal_time=0)
    assert [(step.time_from, step.time_to) for step in trace] == [(999.0, 500.0), (500.0, 0.0)]
    assert [(step.sigma_from, step.sigma_to) for step in trace] == [(1.0, 0.4), (0.4, 0.0)]


def test_scheduler_adapter_rejects_ambiguous_sigma_contract():
    class InvalidScheduler:
        timesteps = [999, 500]
        sigmas = [1.0, 0.0]

    with pytest.raises(ValueError, match="terminal sigma"):
        transitions_from_scheduler(InvalidScheduler())
