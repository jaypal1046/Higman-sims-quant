import numpy as np
import time, math, os, sys

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.higman_sims_hybrid import HigmanSims_Hybrid

def verify_hybrid():
    dim = 64
    X = np.random.randn(2000, dim).astype(np.float32)
    
    # 1. Test Below Crossover (3.0 BPD)
    print("\n--- TEST: 3.0 BPD (V12 Expected) ---")
    hyb_low = HigmanSims_Hybrid(dim, target_bpd=3.0)
    hyb_low.fit(X)
    co_low = hyb_low.encode(X)
    d_low = hyb_low.decode(co_low)
    snr_low = 10 * np.log10(np.mean(X**2)/max(np.mean((X-d_low)**2), 1e-25))
    print(f"Hybrid SNR (3.0 BPD): {snr_low:.2f} dB")
    
    # 2. Test Above Crossover (8.0 BPD)
    print("\n--- TEST: 8.0 BPD (V16 Expected) ---")
    hyb_high = HigmanSims_Hybrid(dim, target_bpd=8.0)
    hyb_high.fit(X)
    co_high = hyb_high.encode(X)
    d_high = hyb_high.decode(co_high)
    snr_high = 10 * np.log10(np.mean(X**2)/max(np.mean((X-d_high)**2), 1e-25))
    print(f"Hybrid SNR (8.0 BPD): {snr_high:.2f} dB")

if __name__ == "__main__":
    verify_hybrid()
