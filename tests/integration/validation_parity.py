import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.v19 import fast_e8_quantize as f8_numpy
from src.engine.torch_kernel import fast_e8_quantize_torch as f8_torch


def validate():
    print("--- Validating PyTorch vs NumPy E8 Quantization ---")
    np.random.seed(42)
    torch.manual_seed(42)

    data_np = np.random.randn(100, 8).astype(np.float32)
    data_torch = torch.from_numpy(data_np)

    res_np = f8_numpy(data_np)
    res_torch = f8_torch(data_torch).numpy()

    diff = np.abs(res_np - res_torch)
    max_diff = np.max(diff)

    print(f"Max Difference: {max_diff}")
    if max_diff < 1e-6:
        print("[OK] PyTorch kernel is bit-exact with NumPy reference.")
    else:
        print("[FAIL] Discrepancy detected in quantization logic.")


if __name__ == "__main__":
    validate()
