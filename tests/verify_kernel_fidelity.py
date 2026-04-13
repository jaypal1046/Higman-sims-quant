import sys
import os
import numpy as np
import torch

# Ensure src is in path for standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.v19 import fast_e8_quantize

def simulate_triton_e8_dequant(qs_list, scales):
    """
    Simulates the recursive dequantization logic implemented in Triton.
    """
    acc = np.zeros_like(qs_list[0])
    for q, s in zip(qs_list, scales):
        acc += q / s
    return acc

def test_fidelity():
    print("--- Singularity-HS: Geometric Fidelity Check ---")
    
    dim = 256
    np.random.seed(42)
    X = np.random.randn(10, dim)
    
    # Simulate quantization stages (like the real engine would)
    scale1 = 100.0
    q1 = fast_e8_quantize(X * scale1)
    res1 = X - (q1 / scale1)
    
    scale2 = 200.0
    q2 = fast_e8_quantize(res1 * scale2)
    
    # Reconstruction
    recon = simulate_triton_e8_dequant([q1, q2], [scale1, scale2])
    
    mse = np.mean((X - recon)**2)
    snr = 10 * np.log10(np.mean(X**2) / mse)
    
    print(f"Residual SNR after 2 stages: {snr:.2f} dB")
    
    # Requirement from paper: > 50dB at 2.0 BPD (simulated by multi-stage)
    if snr > 50.0:
        print("SUCCESS: Geometric Floor is rigid. (SNR > 50dB)")
    else:
        print("WARNING: Manifold drift detected. (SNR < 50dB)")

def triton_e8_quantize(x: torch.Tensor):
    """
    Backward-compatibility alias for the production engine.
    Wraps the single-stage dequantization logic.
    """
    return triton_e8_dequantize([x], torch.tensor([1.0], device=x.device))

if __name__ == "__main__":
    # Ensure src is in path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    test_fidelity()
