from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from generative_training_audit.noise import (  # noqa: E402
    assert_noise_pair,
    deterministic_seed,
    make_noise_record,
    sample_noise,
)


seed = deterministic_seed(shard_id=3, sample_index=17)
cached_noise = sample_noise(seed, (2, 4, 8, 8))
record = make_noise_record(seed, cached_noise)

assert_noise_pair(record, sample_noise(seed, cached_noise.shape))
print("same seed: pairing check passed")

try:
    assert_noise_pair(record, sample_noise(seed + 1, cached_noise.shape))
except AssertionError as error:
    print(f"different seed: {error}")
