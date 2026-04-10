import numpy as np
import pandas as pd
import time
from ..core.v16 import Final_God_V16
from ..core.v17 import LatticeRSN_V17
from ..core.v18 import LatticeRSN_V18
from ..core.v19 import LatticeRSN_V19

def run_statistical_comparison(trials=3, dim=256):
    results = []
    
    print(f"--- Running Performance Comparison (V16, V17, V18, V19) [Dim={dim}] ---")
    
    for i in range(trials):
        seed = 42 + i
        np.random.seed(seed)
        
        # Fresh data for each trial
        X_train = np.random.randn(1000, dim).astype(np.float32)
        X_test = np.random.randn(200, dim).astype(np.float32)
        X_test64 = X_test.astype(np.float64)
        signal = np.mean(X_test64**2)

        # 1. Test V16
        eng16 = Final_God_V16(dim, max_stages=4)
        eng16.fit(X_train)
        co16 = eng16.encode(X_test)
        d16 = eng16.decode(co16)
        snr16 = 10 * np.log10(signal / max(np.mean((X_test64 - d16.astype(np.float64))**2), 1e-30))
        results.append({"Trial": i+1, "Version": "V16", "SNR": snr16, "BPD": eng16.measure_efficiency(co16)})

        # 2. Test V17
        eng17 = LatticeRSN_V17(dim, max_stages=4)
        eng17.fit(X_train)
        co17 = eng17.encode(X_test)
        d17 = eng17.decode(co17)
        snr17 = 10 * np.log10(signal / max(np.mean((X_test64 - d17.astype(np.float64))**2), 1e-30))
        results.append({"Trial": i+1, "Version": "V17", "SNR": snr17, "BPD": eng17.measure_efficiency(co17)})

        # 3. Test V18
        eng18 = LatticeRSN_V18(dim, target_bpd=3.0, max_stages=4)
        eng18.fit(X_train)
        co18 = eng18.encode(X_test)
        d18 = eng18.decode(co18)
        snr18 = 10 * np.log10(signal / max(np.mean((X_test64 - d18.astype(np.float64))**2), 1e-30))
        results.append({"Trial": i+1, "Version": "V18", "SNR": snr18, "BPD": eng18.measure_efficiency(co18)})

        # 4. Test V19
        eng19 = LatticeRSN_V19(dim, target_bpd=3.0)
        eng19.fit(X_train)
        co19 = eng19.encode(X_test)
        d19 = eng19.decode(co19)
        snr19 = 10 * np.log10(signal / max(np.mean((X_test64 - d19.astype(np.float64))**2), 1e-30))
        results.append({"Trial": i+1, "Version": "V19", "SNR": snr19, "BPD": eng19.measure_efficiency(co19)})

        print(f"Trial {i+1} Finished.")

    df = pd.DataFrame(results)
    summary = df.groupby("Version").agg({"SNR": ["mean", "std"], "BPD": ["mean", "std"]})
    
    print("\n--- Final Performance Comparison ---")
    print(summary)
    
    return summary

if __name__ == "__main__":
    run_statistical_comparison()
