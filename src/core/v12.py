"""
Higman-Sims Quantizer V12 (THE-UNTOUCHABLE)
===========================================
Adaptive Syndrome-Lattice Hybrid with SBSS.
"Survival at 1.5 - 4.0 BPD."
V12: The Resilient Foundation.
"""

import math, time, os, numpy as np

def build_e8():
    v = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    x = np.zeros(8); x[i], x[j] = si, sj; v.append(x)
    for m in range(256):
        s = np.array([0.5 if ((m >> k) & 1) == 0 else -0.5 for k in range(8)])
        if np.sum(s < 0) % 2 == 0: v.append(s)
    return np.array(v) / math.sqrt(2.0)

class Untouchable_Core:
    """Extreme Fidelity Lattice Sync (Adaptive SBSS)."""
    def __init__(self, dim, stages=8, threshold=0.95):
        self.dim, self.stages = int(dim), int(stages)
        self.nch = int(math.ceil(dim/8)); self.pw = self.nch*8
        self.CB8 = build_e8(); self.CB8T = self.CB8.T.copy()
        self.m = None; self.md = None
        self.rngs = [] # (lo, hi) per stage
        self.threshold = threshold # Sparse masking (SBSS)

    def fit(self, X):
        nv = len(X)
        self.m = np.median(X, 0); self.md = np.median(np.abs(X - self.m), 0) + 1e-12
        Xn = (X - self.m) / self.md
        res = np.pad(Xn, ((0,0),(0,self.pw-self.dim))).reshape(nv, self.nch, 8)
        
        self.rngs = []
        for s in range(self.stages):
            # Calculate energy to find "Hot Chunks" (Outliers) - axis 2 is the 8D block
            energy = np.mean(res**2, axis=2)
            mask = energy > np.quantile(energy, self.threshold) if self.threshold < 1.0 else np.ones_like(energy, dtype=bool)
            
            # Projection only on active chunks (SBSS logic)
            active_res = res[mask] # active_res is 2D: (M, 8)
            if len(active_res) == 0: break
            
            # Use axis=1 for active_res (since it's 2D)
            idx = np.argmax(active_res @ self.CB8T, axis=1)
            dots = np.sum(active_res * self.CB8[idx], axis=1)
            
            lo, hi = float(np.quantile(dots, 0.001)), float(np.quantile(dots, 0.999))
            self.rngs.append((lo, hi, mask)) 
            
            sp = max(hi - lo, 1e-6); lvls = 15 
            q = np.floor(np.clip((dots - lo) / sp, 0, 0.9999) * lvls).astype(int) + 1
            h = lo + (q.astype(float) - 0.5) / lvls * sp
            
            # In-place update using boolean indexing
            res[mask] -= self.CB8[idx] * (h[..., None])

    def encode(self, X):
        nv = len(X); Xn = (X - self.m) / self.md
        res = np.pad(Xn, ((0,0),(0,self.pw-self.dim))).reshape(nv, self.nch, 8)
        
        a_idx, a_q, a_mask = [], [], []
        for s in range(len(self.rngs)):
            lo, hi, _ = self.rngs[s]
            
            energy = np.mean(res**2, axis=2)
            mask = energy > np.quantile(energy, self.threshold) if self.threshold < 1.0 else np.ones_like(energy, dtype=bool)
            
            active_res = res[mask]
            if len(active_res) == 0:
                a_idx.append(None); a_q.append(None); a_mask.append(mask)
                continue
            
            idx = np.argmax(active_res @ self.CB8T, axis=1) # Axis 1 for 2D
            dots = np.sum(active_res * self.CB8[idx], axis=1)
            
            sp = max(hi - lo, 1e-6); lvls = 15
            q = np.floor(np.clip((dots - lo) / sp, 0, 0.9999) * lvls).astype(int) + 1
            h = lo + (q.astype(float) - 0.5) / lvls * sp
            
            res[mask] -= self.CB8[idx] * (h[..., None])
            a_idx.append(idx); a_q.append(q); a_mask.append(mask)
            
        return a_idx, a_q, a_mask

    def decode(self, co):
        idx, qs, masks = co
        nv = masks[0].shape[0]
        res = np.zeros((nv, self.nch, 8))
        
        for s in range(len(self.rngs)):
            if idx[s] is None: continue
            
            lo, hi, _ = self.rngs[s]; sp = hi - lo; lvls = 15
            mask = masks[s]
            h = lo + (qs[s].astype(float) - 0.5) / lvls * sp
            
            res[mask] += self.CB8[idx[s]] * (h[..., None])
            
        return res.reshape(nv, -1)[:, :self.dim] * self.md + self.m

    def bpd(self, co=None) -> float:
        """Calculate bit-accurate BPD using the sparse masks."""
        if co is None:
            # Theoretical baseline (1 stage, 100% density)
            return (64.0 / self.dim) + (1.5 * self.stages)
        
        _, qs, masks = co
        nv = masks[0].shape[0]
        total_bits = 64.0 # Mean/Std metadata (FP32 per dim)
        
        for s in range(len(masks)):
            mask = masks[s]
            num_active = np.sum(mask)
            # 1 bit per chunk (mask) + (8 bits index + 4 bits scalar) per active chunk
            total_bits += (self.nch * 1.0) # Mask overhead
            total_bits += (num_active * 12.0) / nv # Active data
            
        return float(total_bits / self.dim)

if __name__ == "__main__":
    X = np.random.randn(500, 300).astype(np.float32)
    eng = Untouchable_Core(300, stages=1, threshold=0.95)
    eng.fit(X)
    c = eng.encode(X)
    d = eng.decode(c)
    print(f"BPD: {eng.bpd(c):.4f}")
