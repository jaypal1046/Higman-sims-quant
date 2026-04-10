"""
Lattice-RSN V18 (Singularity-Void)
==================================
God-Mode Architecture: Sparse Syndrome Bit-Stealing (SBSS) + H-RSN.
Targeting < 3.0 BPD with high-precision core refinement.
"""

import math, time, os, numpy as np
from scipy.stats import entropy as scipy_entropy

def fast_e8_quantize(x):
    """Fast nearest point in E8 lattice (8D). Multi-dimensional."""
    sh = x.shape
    x = x.reshape(-1, 8)
    def decode_dn(x):
        y = np.round(x)
        s = np.sum(y, axis=-1)
        bad = (s % 2 != 0)
        if np.any(bad):
            diff = x - y
            idx = np.argmax(np.abs(diff[bad]), axis=-1)
            rows = np.where(bad)[0]
            sign = np.sign(diff[bad, idx])
            sign[sign == 0] = 1.0
            y[rows, idx] += sign
        return y
    y0 = decode_dn(x)
    y1 = decode_dn(x - 0.5) + 0.5
    d0 = np.sum((x - y0)**2, axis=-1)
    d1 = np.sum((x - y1)**2, axis=-1)
    mask = (d1 < d0)
    res = y0.copy()
    res[mask] = y1[mask]
    return res.reshape(sh)

class LatticeRSN_V18:
    """V18: Singularity-Void. Sparse Lattice Refinement."""
    def __init__(self, dim, target_bpd=3.0, max_stages=4):
        self.dim = int(dim)
        self.target_bpd = target_bpd
        self.max_stages = max_stages
        
        self.nch_8 = int(math.ceil(dim/8))
        self.pw_8 = self.nch_8 * 8
        self.nch_16 = int(math.ceil(dim/16))
        self.pw_16 = self.nch_16 * 16
        
        # Rigorous Metadata Math (BPD)
        # Global(32) + Local(N/16*8) + Mask(N/8*1)
        self.metadata_bpd = (32 + (self.nch_16 * 8) + (self.nch_8 * 1)) / self.dim
        self.lattice_budget = max(0.1, self.target_bpd - self.metadata_bpd)
        
        self.scales = [] 
        self.density = 0.5 # Fraction of chunks to receive refinement

    def fit(self, X):
        """Fit phase determines scales and rigorously calibrates density to hit target_bpd."""
        nv = len(X)
        X_p = np.pad(X, ((0,0),(0,self.pw_16-self.dim))).astype(np.float64)
        
        # 1. Hierarchical Normalization
        gm = np.mean(X_p, axis=1, keepdims=True)
        gs = np.std(X_p - gm, axis=1, keepdims=True) + 1e-12
        res = (X_p - gm) / gs
        res_16 = res.reshape(nv, self.nch_16, 16)
        ls = np.std(res_16, axis=2, keepdims=True) + 1e-12
        res_8 = (res_16 / ls).reshape(nv, self.nch_8, 8)
        
        # 2. Stage Calibration & Actual Entropy Measurement
        # We need to know EXACTLY how many bits each stage takes on average
        stage_entropies = []
        for s in range(self.max_stages):
            rms = np.sqrt(np.mean(res_8**2)) + 1e-15
            scale = 100.0 / rms
            self.scales.append(scale)
            
            q = fast_e8_quantize(res_8 * scale)
            unique_rows, counts = np.unique(q.reshape(-1, 8), axis=0, return_counts=True)
            entropy = scipy_entropy(counts / np.sum(counts), base=2)
            # Entropy is bits per 8D block. BPD-equivalent is entropy/8.
            stage_entropies.append(entropy / 8.0)
            
            res_8 -= q / scale
            
        # 3. Proper Density Solver (Proper Math)
        # Goal: Meta + Base_Stage + density * Sum(Refinement_Stages) = target_bpd
        # Let stage_entropies[0] be the base stage.
        meta = self.metadata_bpd
        base = stage_entropies[0]
        refinement_sum = sum(stage_entropies[1:])
        
        remaining_budget = self.target_bpd - meta - base
        
        if remaining_budget <= 0:
            # We can't even afford a full Stage 0. 
            # Recalculate density for Stage 0 itself.
            self.density = 0.0
            # We must also partially apply stage 0 (this is a fallback)
            self.base_density = max(0.1, (self.target_bpd - meta) / base)
        else:
            self.base_density = 1.0
            self.density = min(1.0, remaining_budget / refinement_sum)

        print(f"V18 Calibration (Proper Math):")
        print(f" - Metadata: {meta:.4f} BPD")
        print(f" - Base Stage (100%): {base:.4f} BPD")
        print(f" - Refinement Sum: {refinement_sum:.4f} BPD")
        print(f" - Target: {self.target_bpd} BPD")
        print(f" - Calculated Refinement Density: {self.density:.2%}")

    def encode(self, X):
        """Sparse Hierarchical RSN: Dynamic Density Layering."""
        nv = len(X)
        X_p = np.pad(X, ((0,0),(0,self.pw_16-self.dim))).astype(np.float64)
        
        gm = np.mean(X_p, axis=1, keepdims=True)
        gs = np.std(X_p - gm, axis=1, keepdims=True) + 1e-12
        res = (X_p - gm) / gs
        
        res_16 = res.reshape(nv, self.nch_16, 16)
        ls = np.std(res_16, axis=2, keepdims=True) + 1e-12
        res_8 = (res_16 / ls).reshape(nv, self.nch_8, 8)
        
        # 1. Base Layer (Stage 0) with density control
        all_qs = []
        energy_0 = np.mean(res_8**2, axis=2)
        thresh_0 = np.quantile(energy_0, 1.0 - self.base_density) if self.base_density < 1.0 else -1.0
        mask_0 = (energy_0 >= thresh_0)
        
        q_0 = np.zeros_like(res_8)
        q_0[mask_0] = fast_e8_quantize(res_8[mask_0] * self.scales[0])
        res_8[mask_0] -= q_0[mask_0] / self.scales[0]
        all_qs.append(q_0)
        
        # 2. Refinement Layers (Stage 1..N)
        energy_r = np.mean(res_8**2, axis=2)
        thresh_r = np.quantile(energy_r, 1.0 - self.density) if self.density < 1.0 else -1.0
        mask_r = (energy_r >= thresh_r)
        
        for s in range(1, len(self.scales)):
            q = np.zeros_like(res_8)
            if self.density > 0:
                q[mask_r] = fast_e8_quantize(res_8[mask_r] * self.scales[s])
                res_8[mask_r] -= q[mask_r] / self.scales[s]
            all_qs.append(q)
            
        return {"qs": all_qs, "gm": gm, "gs": gs, "ls": ls, "mask_0": mask_0, "mask_r": mask_r}

    def decode(self, co):
        all_qs = co["qs"]
        gm, gs, ls = co["gm"], co["gs"], co["ls"]
        nv, nch_8, _ = all_qs[0].shape
        
        res_8 = np.zeros((nv, nch_8, 8), dtype=np.float64)
        for s, q in enumerate(all_qs):
            res_8 += q.astype(np.float64) / self.scales[s]
            
        res_16 = res_8.reshape(nv, self.nch_16, 16)
        res_16 *= ls 
        res_full = res_16.reshape(nv, -1)
        res_full = res_full * gs + gm
        
        return res_full[:, :self.dim]

    def measure_efficiency(self, co):
        all_qs, mask_0, mask_r = co["qs"], co["mask_0"], co["mask_r"]
        nv = len(co["gm"])
        
        metadata_bits = self.metadata_bpd * nv * self.dim
        total_lattice_bits = 0
        
        for s, q in enumerate(all_qs):
            if s == 0:
                q_active = q[mask_0]
            else:
                q_active = q[mask_r] 
                
            if len(q_active) == 0: continue
            unique_rows, counts = np.unique(q_active, axis=0, return_counts=True)
            probs = counts / len(q_active)
            entropy_per_chunk = scipy_entropy(probs, base=2)
            total_lattice_bits += entropy_per_chunk * len(q_active)
            
        return (metadata_bits + total_lattice_bits) / (nv * self.dim)

if __name__ == "__main__":
    np.random.seed(42)
    dim = 256
    
    X_train = np.random.randn(1000, dim).astype(np.float32)
    X_test = np.random.randn(200, dim).astype(np.float32)
    
    print(f"--- V18 (Singularity-Void) Performance Verification ---")
    eng = LatticeRSN_V18(dim, target_bpd=3.0)
    eng.fit(X_train)
    
    co = eng.encode(X_test)
    d = eng.decode(co)
    
    X_test64 = X_test.astype(np.float64)
    mse = np.mean((X_test64 - d.astype(np.float64))**2)
    snr = 10 * np.log10(np.mean(X_test64**2) / max(mse, 1e-30))
    bpd = eng.measure_efficiency(co)
    
    print(f"SNR: {snr:.2f} dB | Real Entropy BPD: {bpd:.2f}")
    if bpd <= 3.0:
        print("✅ SUCCESS: V18 breached the 3.0 BPD barrier.")
    else:
        print(f"⚠️ WARNING: BPD {bpd:.2f} > 3.0.")
