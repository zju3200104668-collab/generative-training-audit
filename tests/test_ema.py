import numpy as np

from generative_training_audit.ema import ema_step_fp32, ema_step_quantized


def test_bf16_shadow_can_freeze_while_fp32_master_moves():
    shadow = np.asarray([1.0], dtype=np.float32)
    parameter = np.asarray([1.01], dtype=np.float32)
    unsafe = ema_step_quantized(shadow, parameter, decay=0.999)
    safe = ema_step_fp32(shadow, parameter, decay=0.999)
    assert np.array_equal(unsafe, shadow)
    assert not np.array_equal(safe, shadow)
