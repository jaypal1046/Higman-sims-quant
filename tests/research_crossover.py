import numpy as np
import time, math, os, sys

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.higman_sims_quant_v12 import Untouchable_Core as V12_Core
from src.higman_sims_quant_v16 import Final_God_V16 as V16_Core

def find_crossover():
    dim = 64
    X = np.random.randn(2000, dim).astype(np.float32)
    
    sweep = np.arange(4.0, 10.1, 0.5)
    print(f"--- BITRATE CROSSOVER SWEEP (V12 vs V16) ---")
    print(f"{'BPD':5} | {'V12 SNR':10} | {'V16 SNR':10} | {'Winner':10}")
    print("-" * 45)
    
    for bpd in sweep:
        # V12: stages = bpd (roughly)
        v12 = V12_Core(dim, stages=int(bpd))
        v12.fit(X)
        co12 = v12.encode(X)
        d12 = v12.decode(co12)
        snr12 = 10 * np.log10(np.mean(X**2)/max(np.mean((X-d12)**2), 1e-25))
        
        # V16: BPD = 4 + stages * 1.5 
        # So stages = (bpd - 4) / 1.5
        v16_stages = max(0, int((bpd - 4.0) / 1.5))
        if v16_stages > 0:
            v16 = V16_Core(dim, max_stages=v16_stages)
            v16.fit(X)
            co16 = v16.encode(X)
            d16 = v16.decode(co16)
            snr16 = 10 * np.log10(np.mean(X**2)/max(np.mean((X-d16)**2), 1e-25))
        else:
            snr16 = 0.5 # Noise floor
            
        winner = "V12" if snr12 > snr16 else "V16"
        print(f"{bpd:5.1f} | {snr12:10.2f} | {snr16:10.2f} | {winner:10}")

if __name__ == "__main__":
    find_crossover()
