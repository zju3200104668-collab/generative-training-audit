import numpy as np
import pytest

from generative_training_audit.noise import (
    assert_noise_pair,
    make_noise_record,
    sample_noise,
)


def test_identical_seed_reproduces_exact_noise():
    first = sample_noise(42, (2, 3, 4))
    second = sample_noise(42, (2, 3, 4))
    assert np.array_equal(first, second)
    assert_noise_pair(make_noise_record(42, first), second)


def test_mismatched_noise_fails_pairing_assertion():
    first = sample_noise(42, (2, 3, 4))
    second = sample_noise(43, (2, 3, 4))
    with pytest.raises(AssertionError, match="pairing failed"):
        assert_noise_pair(make_noise_record(42, first), second)
