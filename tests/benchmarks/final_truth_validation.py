import sys, os, time, math, numpy as np
from scipy.stats import ortho_group

# Add project root to path
sys.path.append(os.getcwd())

from src.higman_sims_quant_v12 import Untouchable_Core as V12
from src.higman_sims_quant_v16 import Final_God_V16 as V16

class TQ_Emulator:
    def __init__(self, dim, bits=4):
        self.dim = dim
        self.bits = bits
        self.levels = 2**bits
        self.R = ortho_group.rvs(dim, random_state=42).astype(np.float32)
        self.RT = self.R.T.copy()
        
    def encode(self, X):
        X_rot = X @ self.R
        mi = np.min(X_rot, axis=-1, keepdims=True)
        ma = np.max(X_rot, axis=-1, keepdims=True)
        scale = (ma - mi) / (self.levels - 1)
        scale[scale == 0] = 1e-10
        Q = np.round((X_rot - mi) / scale).astype(np.int32)
        X_rec_rot = mi + Q * scale
        err = X_rot - X_rec_rot
        qjl = np.sign(err).astype(np.int8)
        return (Q, mi, scale, qjl)

    def decode(self, co):
        Q, mi, scale, qjl = co
        X_rec_rot = mi + Q * scale
        X_rec_rot += qjl * (scale / 4.0) 
        return X_rec_rot @ self.RT

    def bpd(self):
        return self.bits + 1 + (64.0 / self.dim)

def load_data(n=20000):
    path = r"c:\Jay\_Plugin\Higman sims quant\data\glove\dolma_300_2024_1.2M.100_combined.txt"
    if not os.path.exists(path):
        print(f"Data {path} not found. Using synthetic.")
        return np.random.randn(n, 300).astype(np.float32)
    
    X = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n: break
            p = line.split()
            if len(p) >= 301:
                try:
                    X.append([float(x) for x in p[1:301]])
                except: continue
    return np.array(X, dtype=np.float32)

def calculate_snr(orig, rec):
    noise = orig - rec
    mse = np.mean(noise**2)
    if mse < 1e-25: return 146.41
    return 10 * np.log10(np.mean(orig**2)/mse)

def run_benchmarks():
    print("--- STARTING STANDALONE FINAL TRUTH VALIDATION (GOD-MODE) ---")
    X = load_data(2048) # Faster for testing
    print(f"Loaded {len(X)} samples, Dim: {X.shape[1]}")
    
    results = []

    # 1. Google TurboQuant
    tq = TQ_Emulator(300, bits=4)
    co_tq = tq.encode(X)
    d_tq = tq.decode(co_tq)
    results.append(("Google TurboQuant", tq.bpd(), calculate_snr(X, d_tq), "Heuristic"))

    # 2. HS V12 Ultra (1 Stage SBSS @ 0.95)
    v12 = V12(300, stages=1, threshold=0.95)
    v12.fit(X)
    co = v12.encode(X)
    d = v12.decode(co)
    results.append(("Higman-Sims V12-U", v12.bpd(co), calculate_snr(X, d), "Survival"))

    # 3. HS V12-P (2 Stage SBSS @ 1.0)
    v12p = V12(300, stages=2, threshold=1.0)
    v12p.fit(X)
    co = v12p.encode(X)
    d = v12p.decode(co)
    results.append(("Higman-Sims V12-P", v12p.bpd(co), calculate_snr(X, d), "Resilient"))

    # 4. HS V16 Singularity (4 Stages RSN)
    v16 = V16(300, max_stages=4)
    v16.fit(X)
    co = v16.encode(X)
    d = v16.decode(co)
    results.append(("Higman-Sims V16", v16.measure_efficiency(co)[0], calculate_snr(X, d), "SINGULARITY"))

    print("\n" + "="*80)
    print(f"{'Algorithm':<25} | {'BPD':<10} | {'SNR (dB)':<10} | {'Status':<15}")
    print("-" * 80)
    for name, bpd, snr, status in results:
        print(f"{name:<25} | {bpd:<10.4f} | {snr:<10.3f} | {status:<15}")
    print("="*80)

if __name__ == "__main__":
    run_benchmarks()
