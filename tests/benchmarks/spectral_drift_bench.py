import torch
import time
from src.engine.vllm_engine import HybridLatticeEngine

def simulate_drifting_context(total_tokens, dim, drift_factor=0.01):
    """
    Simulate a context where the semantic 'ground' slowly shifts.
    """
    print(f"Generating {total_tokens} drifting tokens...")
    # Base semantic signal (Low-rank)
    basis = torch.randn(dim, 32)
    
    X = []
    current_signal = torch.zeros(dim)
    for i in range(total_tokens):
        # Sparse activation of singular vectors
        coeffs = torch.randn(32)
        current_signal = basis @ coeffs
        # Add cumulative drift to simulate semantic shift
        basis += torch.randn_like(basis) * drift_factor
        X.append(current_signal)
        
    return torch.stack(X)

def run_spectral_bench():
    print("--- Phase 6: Spectral God-Mode (SVD-LRSN) Benchmark ---")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Running in CPU mode.")
        device = "cpu"
    else:
        device = "cuda"
        
    dim = 4096
    total_tokens = 5000
    
    # 1. Initialize Engines
    # Standard Engine (Fixed Lattice)
    engine_std = HybridLatticeEngine(in_features=dim, out_features=dim, use_triton=False).to(device)
    # Spectral Engine (Adaptive SVD)
    engine_spec = HybridLatticeEngine(in_features=dim, out_features=dim, use_triton=False).to(device)
    
    # Mock Weights
    w = torch.randn(dim, dim).to(device)
    engine_std.load_from_llama(w)
    engine_spec.load_from_llama(w)
    
    # 2. Get Drifting Data
    X = simulate_drifting_context(total_tokens, dim).to(device)
    
    # 3. Benchmark Accuracy & Stability
    print("\nEvaluating Spectral Stability...")
    start = time.time()
    
    with torch.no_grad():
        # Standard Pass
        out_std = engine_std(X, use_spectral_scout=False)
        # Spectral Pass
        out_spec = engine_spec(X, use_spectral_scout=True)
        
    end = time.time()
    
    # Compute Proxy SNR (compared to idealized matmul)
    ideal_out = torch.matmul(X, w.t())
    snr_std = 20 * torch.log10(torch.norm(ideal_out) / torch.norm(ideal_out - out_std))
    snr_spec = 20 * torch.log10(torch.norm(ideal_out) / torch.norm(ideal_out - out_spec))
    
    print(f"\nStandard SNR: {snr_std:.2f} dB")
    print(f"Spectral SNR: {snr_spec:.2f} dB")
    print(f"Signal Gain:  {snr_spec - snr_std:.2f} dB")
    print(f"Latency:      {((end-start)/total_tokens)*1000:.4f} ms/token")
    
    if snr_spec > snr_std:
        print("\n✅ PHASE 6 COMPLETE: Spectral God-Mode provides higher fidelity for drifting contexts.")

if __name__ == "__main__":
    run_spectral_bench()
