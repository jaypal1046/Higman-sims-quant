import numpy as np
from higman_sims_quant_v16 import Final_God_V16, fast_e8_quantize
import pandas as pd

def run_ablation():
    np.random.seed(42)
    dim = 64
    X_train = np.random.randn(1000, dim).astype(np.float32)
    X_test = np.random.randn(200, dim).astype(np.float32)
    
    results = []

    # 1. Full Lattice-RSN (4 stages)
    eng = Final_God_V16(dim, max_stages=4)
    eng.fit(X_train)
    co = eng.encode(X_test)
    d = eng.decode(co)
    snr = 10 * np.log10(np.mean(X_test**2) / np.mean((X_test.astype(np.float64)-d)**2))
    bpd = eng.measure_efficiency(co)
    results.append({"Method": "Lattice-RSN (4-stage)", "SNR (dB)": snr, "BPD": bpd})

    # 2. RSN (1 stage)
    eng_1 = Final_God_V16(dim, max_stages=1)
    eng_1.fit(X_train)
    co_1 = eng_1.encode(X_test)
    d_1 = eng_1.decode(co_1)
    snr_1 = 10 * np.log10(np.mean(X_test**2) / np.mean((X_test.astype(np.float64)-d_1)**2))
    bpd_1 = eng_1.measure_efficiency(co_1)
    results.append({"Method": "Lattice-RSN (1-stage)", "SNR (dB)": snr_1, "BPD": bpd_1})

    # 3. No RSN (Pure E8 Global Scaling - Simulated by 0-stage or bypass)
    # If we disable RSN, we just use global mean/std
    m_global = np.mean(X_train)
    std_global = np.std(X_train)
    X_norm = (X_test - m_global) / std_global
    q = fast_e8_quantize(X_norm)
    d_no_rsn = (q * std_global) + m_global
    snr_no = 10 * np.log10(np.mean(X_test**2) / np.mean((X_test.astype(np.float64)-d_no_rsn)**2))
    # BPD without metadata tax (approx 4.0 BPD)
    results.append({"Method": "Global E8 (No RSN)", "SNR (dB)": snr_no, "BPD": 4.0})

    df = pd.DataFrame(results)
    print("\n--- Ablation Study Results ---")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    run_ablation()
