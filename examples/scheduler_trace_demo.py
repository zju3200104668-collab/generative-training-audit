from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from generative_training_audit.scheduler import (  # noqa: E402
    audit_transitions,
    build_transitions,
    format_trace,
)


times = [1.0, 0.55, 0.0]
sigmas = [1.0, 0.42, 0.0]
trace = build_transitions(times, sigmas)

print(format_trace(trace))
issues = audit_transitions(
    trace, expected_steps=2, expected_start=1.0, expected_end=0.0
)
print("audit:", "passed" if not issues else issues)
