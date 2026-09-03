from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from generative_training_audit.ema import (  # noqa: E402
    ema_step_fp32,
    ema_step_quantized,
    update_to_spacing_ratio,
)


shadow = np.asarray([1.0], dtype=np.float32)
parameter = np.asarray([1.01], dtype=np.float32)
decay = 0.999

unsafe = ema_step_quantized(shadow, parameter, decay)
safe = ema_step_fp32(shadow, parameter, decay)

print(f"shadow:                 {shadow[0]:.8f}")
print(f"BF16-rounded update:    {unsafe[0]:.8f}")
print(f"FP32-master update:     {safe[0]:.8f}")
print(f"update / BF16 spacing:  {update_to_spacing_ratio(1.0, 1.01, decay):.6f}")
print(f"unsafe changed:         {not np.array_equal(shadow, unsafe)}")
