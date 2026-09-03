import pytest

from generative_training_audit.scheduler import audit_transitions, build_transitions


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
