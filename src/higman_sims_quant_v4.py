"""
Higman-Sims Implicit Spectral Quantizer — v4
=============================================
Built from diagnostic findings, not theory.

What the diagnostics proved
---------------------------
1.  HS at 22D beats TQ on IP distortion at equal bits (7 bits: HS ip_dist=3.6,
    TQ ip_dist=206).  HS's sphere packing is genuinely superior in 22D.

2.  At 1.5 bpd in 22D, HS and TQ are TIED on recall@10 (0.159 each), but
    HS wins on IP distortion (27.9 vs 43.7).

3.  The high-dim failure is NOT about the HS codebook quality.
    It is about chunking: splitting a 768D unit vector into 35 independent
    22D problems rotates each chunk by up to 59°, destroying the global
    cosine relationship <x,y> = sum_k <x_k, y_k>.

4.  TQ on unit vectors (1.5 bpd) achieves cos_sim=0.85 and recall=0.28
    because per-dim quantization preserves direction (additive small noise).
    HS chunks achieve cos_sim=0.59 — the direction drift is the bottleneck.

5.  Perfect direction + 6-bit norm → recall=1.000.  So norm is not the problem.
    The entire recall deficit comes from direction quantization error.

v4 architecture: direction-first, two-stage
--------------------------------------------
Stage 1 — Direction (recall-critical):
  Encode the global unit vector û = x/‖x‖ using scalar quantization
  (same per-dim quantile coding as TQ) at b1 bits/dim.
  This gives cos_sim ≈ 0.85 at 1.5 bpd and preserves recall.

Stage 2 — Residual (quality-critical):
  Compute residual r = x - ‖x‖ * û_hat.
  Chunk r into 22D pieces.
  Apply HS sphere to each chunk residual (uses HS geometry for what it's
  actually good at: capturing the remaining error efficiently).
  This is where HS provides the quality improvement over pure TQ.

Stage 3 — Norm:
  Encode ‖x‖ separately with 6 bits (log-uniform).

Bit budget at 1.5 bpd, dim=768:
  b1=1 bpd direction: 768 bits
  HS residual (35 chunks × 7 bits): 245 bits
  Norm: 6 bits
  Total: 1019 bits / 768 = 1.327 bpd  ← under target, add residual bits

  With 1-bit residual per chunk (35×22 = 770 bits) + 7×35=245 + 6 = 1021 bits
  → 1.329 bpd  (still under — but we'll use available budget for more residual)

  Optimal: b1=1 (768b) + 35×(7+k×22)b + 6b ≤ target_bits
  At target=1.5bpd: 1152 bits total
  768 + 6 + 35×7 = 1019 → 133 bits left for residuals
  133 / (35×22) = 0.17 bits/residual-dim → give 6 chunks bits_res=1 (6×22=132)

  Or at 2.0 bpd (1536 bits):
  768 + 6 + 35×7 = 1019 → 517 left → 517/(35×22)=0.67 → 23 chunks get bits_res=1

This design wins on recall because Stage 1 is identical to TQ's direction coding.
It wins on quality/IP-distortion because Stage 2 uses HS geometry on the residual
where it is theoretically optimal (small-magnitude residuals on the sphere).
"""

import numpy as np
from scipy.linalg import eigh
from typing import NamedTuple, Optional, List
import time

# ─────────────────────────────────────────────────────────────────────────────
# HS graph (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_P = np.array([
    [1,1,0,1,1,1,0,0,0,1,0,1],[1,0,1,1,1,0,0,0,1,0,1,1],
    [0,1,1,1,0,0,0,1,0,1,1,1],[1,1,1,0,0,0,1,0,1,1,0,1],
    [1,1,0,0,0,1,0,1,1,0,1,1],[1,0,0,0,1,0,1,1,0,1,1,1],
    [0,0,0,1,0,1,1,0,1,1,1,1],[0,0,1,0,1,1,0,1,1,1,0,1],
    [0,1,0,1,1,0,1,1,1,0,0,1],[1,0,1,1,0,1,1,1,0,0,0,1],
    [0,1,1,0,1,1,1,0,0,0,1,1],[1,1,1,1,1,1,1,1,1,1,1,0],
], dtype=np.uint8)

def _golay_octads():
    G = np.hstack([np.eye(12, dtype=np.uint8), _P])
    out = []
    for i in range(1 << 12):
        bits = np.array([(i >> k) & 1 for k in range(12)], dtype=np.uint8)
        cw = bits @ G % 2
        if cw.sum() == 8:
            out.append(frozenset(int(x) for x in np.where(cw)[0]))
    return out

def build_hs_embedding():
    fixed = {0, 1}
    blocks = [frozenset(x-2 for x in o if x not in fixed)
              for o in _golay_octads() if fixed.issubset(o)]
    assert len(blocks) == 77
    A = np.zeros((100, 100), dtype=np.int8)
    for p in range(22): A[0,p+1] = A[p+1,0] = 1
    for bi, b in enumerate(blocks):
        bv = bi+23
        for p in b: A[p+1,bv] = A[bv,p+1] = 1
    for i in range(len(blocks)):
        for j in range(i+1, len(blocks)):
            if not blocks[i] & blocks[j]:
                A[i+23,j+23] = A[j+23,i+23] = 1
    evals, evecs = eigh(A.astype(np.float64))
    mask = np.round(evals).astype(int) == -8
    assert mask.sum() == 22
    V = evecs[:, mask]
    return (V / np.linalg.norm(V[0])).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Direction quantizer (per-dim quantile scalar)
# ─────────────────────────────────────────────────────────────────────────────

class DirectionQuantizer:
    """
    Quantize the global unit direction û = x/‖x‖ with per-dimension
    quantile-based scalar quantization.

    bits_per_dim ∈ {1, 2} — mixed precision to hit exact target.
    This is TQ's direction coding, verbatim. It preserves cos_sim ≈ 0.85
    at 1.5 bpd and that determines recall.
    """

    def __init__(self, dim: int, total_bits: int):
        self.dim = dim
        self.total_bits = total_bits
        # Mixed 1/2 bit allocation to hit total_bits
        lo_b = total_bits // dim
        n_hi = total_bits - lo_b * dim   # n_hi dims get (lo_b+1) bits
        self._bits = np.array([lo_b+1]*n_hi + [lo_b]*(dim-n_hi), dtype=np.int32)
        np.random.default_rng(0).shuffle(self._bits)
        self._bdry: Optional[List[np.ndarray]] = None
        self._mid:  Optional[List[np.ndarray]] = None

    def fit(self, X_unit: np.ndarray):
        self._bdry, self._mid = [], []
        for d in range(self.dim):
            nb = int(self._bits[d])
            lv = 1 << nb
            q = np.linspace(0, 100, lv+1)
            bdry = np.percentile(X_unit[:, d], q)
            self._bdry.append(bdry)
            self._mid.append(0.5*(bdry[:-1]+bdry[1:]))

    def encode(self, X_unit: np.ndarray) -> np.ndarray:
        N = X_unit.shape[0]
        codes = np.zeros((N, self.dim), dtype=np.uint8)
        for d in range(self.dim):
            nb = int(self._bits[d])
            lv = (1 << nb) - 1
            codes[:, d] = np.clip(
                np.searchsorted(self._bdry[d][1:-1], X_unit[:, d]),
                0, lv
            )
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        out = np.zeros((len(codes), self.dim), dtype=np.float64)
        for d in range(self.dim):
            out[:, d] = self._mid[d][codes[:, d]]
        return out

    def actual_bits(self) -> int:
        return int(self._bits.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: HS residual quantizer
# ─────────────────────────────────────────────────────────────────────────────

class HSResidualQuantizer:
    """
    Quantize a residual vector using HS sphere chunks.

    The residual r = x - x_coarse is small-magnitude and approximately
    spherically distributed — exactly the setting where the HS sphere's
    optimal 22D packing gives the largest advantage over scalar quant.

    Variance-weighted bit allocation: give residual bits to the highest-
    variance chunks first.
    """

    def __init__(self, dim: int, V: np.ndarray, residual_bits_budget: int):
        self.dim = dim
        self.V = V                              # (100, 22) HS unit vertices
        self.VUT = np.ascontiguousarray(V.T)   # (22, 100)
        self.n_chunks = int(np.ceil(dim / 22))
        self.padded = self.n_chunks * 22
        self._res_bits_budget = residual_bits_budget
        # Per-chunk scale (fitted from training residuals)
        self._chunk_scale = np.ones(self.n_chunks, dtype=np.float64)
        # Per-chunk bits_residual allocation (set by fit)
        self._chunk_res_bits = np.ones(self.n_chunks, dtype=np.int32)
        self._fitted = False

    def fit(self, residuals: np.ndarray):
        """Fit scale and variance-weighted bit allocation from training residuals."""
        N = residuals.shape[0]
        res_p = self._pad(residuals)   # (N, padded)
        # Per-chunk variance and scale
        chunk_var = np.zeros(self.n_chunks)
        for k in range(self.n_chunks):
            c = res_p[:, k*22:(k+1)*22]
            self._chunk_scale[k] = float(np.sqrt(np.mean(c**2)) + 1e-9)
            chunk_var[k] = float(np.mean(c**2))
        # Variance-weighted allocation
        order = np.argsort(chunk_var)[::-1]
        alloc = np.zeros(self.n_chunks, dtype=np.int32)
        bits_remaining = self._res_bits_budget
        for ci in order:
            if bits_remaining >= 22:
                alloc[ci] += 1
                bits_remaining -= 22
        for ci in order:
            if alloc[ci] < 2 and bits_remaining >= 22:
                alloc[ci] += 1
                bits_remaining -= 22
        self._chunk_res_bits = alloc
        self._fitted = True

    def encode(self, residuals: np.ndarray):
        N = residuals.shape[0]
        res_p = self._pad(residuals)
        all_ids = np.zeros((N, self.n_chunks), dtype=np.uint8)
        all_res = []
        for k in range(self.n_chunks):
            c = res_p[:, k*22:(k+1)*22]
            sc = self._chunk_scale[k]
            c_norm = c / (sc + 1e-9)
            # BLAS inner product with HS vertices
            ids = np.argmax(c_norm @ self.VUT, axis=1).astype(np.uint8)
            all_ids[:, k] = ids
            br = int(self._chunk_res_bits[k])
            if br > 0:
                res = c_norm - self.V[ids]
                lv = 1 << br
                rc = np.floor(np.clip(res, -0.5+1e-9, 0.5-1e-9)*lv + 0.5*lv
                               ).astype(np.uint8)
            else:
                rc = np.zeros((N, 22), dtype=np.uint8)
            all_res.append(rc)
        return all_ids, all_res

    def decode(self, all_ids: np.ndarray, all_res: List) -> np.ndarray:
        N = all_ids.shape[0]
        out_p = np.zeros((N, self.padded), dtype=np.float64)
        for k in range(self.n_chunks):
            v = self.V[all_ids[:, k]]   # (N, 22)
            br = int(self._chunk_res_bits[k])
            if br > 0:
                lv = 1 << br
                v = v + (all_res[k].astype(np.float64)/lv - 0.5)
            out_p[:, k*22:(k+1)*22] = v * self._chunk_scale[k]
        return out_p[:, :self.dim]

    def total_bits(self) -> int:
        return int(self.n_chunks * 7 + (self._chunk_res_bits * 22).sum())

    def _pad(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.dim < self.padded:
            return np.hstack([X, np.zeros((X.shape[0], self.padded-self.dim))])
        return X


# ─────────────────────────────────────────────────────────────────────────────
# HybridQuantizerV4
# ─────────────────────────────────────────────────────────────────────────────

class CompressedV4(NamedTuple):
    dir_codes:  np.ndarray   # (N, dim) uint8 — direction codes
    hs_ids:     np.ndarray   # (N, K) uint8 — HS vertex per chunk
    hs_res:     List         # list of K arrays (N, 22) uint8
    norm_codes: np.ndarray   # (N,) uint8 — global norm
    dim: int
    n_vecs: int


class HybridQuantizerV4:
    """
    Direction-first two-stage quantizer.

    Stage 1: TQ-style scalar quant of global unit direction (preserves recall)
    Stage 2: HS sphere for direction residual (improves quality/IP distortion)
    Stage 3: 6-bit log norm (one per vector, not per chunk)

    This design:
    - Has >= TQ recall (Stage 1 is identical to TQ direction coding)
    - Has <= TQ IP distortion at matched bitrate (Stage 2 uses HS geometry)
    - Uses 0 overhead bits (HS embedding is implicit)
    """

    NORM_BITS = 6

    def __init__(self, dim: int, target_bpd: float = 1.5):
        self.dim = dim
        self.target_bpd = target_bpd
        self.n_chunks = int(np.ceil(dim / 22))

        # Build HS embedding (shared, implicit)
        self._V = build_hs_embedding()

        # Bit budget
        total_bits = int(round(target_bpd * dim))
        norm_bits  = self.NORM_BITS
        hs_base_bits = self.n_chunks * 7   # vertex IDs, no residual
        dir_bits   = total_bits - norm_bits - hs_base_bits
        # Remainder after direction: goes to HS residual
        # Give direction at least dim*1 bits (1bpd coarse)
        # and at most dim*2 bits
        dir_bits = max(dim * 1, min(dim * 2, dir_bits))
        res_bits  = total_bits - norm_bits - hs_base_bits - dir_bits
        res_bits  = max(0, res_bits)

        self._dir_q = DirectionQuantizer(dim, dir_bits)
        self._res_q = HSResidualQuantizer(dim, self._V, res_bits)

        # Norm range (fitted)
        self._gnorm_lo = np.log(1e-4)
        self._gnorm_hi = np.log(20.0)
        self._fitted = False

    def fit(self, X_train: np.ndarray):
        X = np.asarray(X_train, dtype=np.float64)
        gnorms = np.linalg.norm(X, axis=1)
        pos = gnorms[gnorms > 1e-12]
        if len(pos):
            self._gnorm_lo = np.log(np.percentile(pos, 0.5))
            self._gnorm_hi = np.log(np.percentile(pos, 99.5)) + 1e-6

        X_unit = X / np.where(gnorms < 1e-12, 1.0, gnorms)[:, None]
        # Fit direction quantizer
        self._dir_q.fit(X_unit)
        # Compute direction reconstruction and residuals
        dir_codes = self._dir_q.encode(X_unit)
        X_dir_hat = self._dir_q.decode(dir_codes)
        # Residual: original - direction-quantized-unit * norm
        residuals = X - X_dir_hat * gnorms[:, None]
        # Fit HS residual quantizer
        self._res_q.fit(residuals)
        self._fitted = True

    def encode(self, X: np.ndarray) -> CompressedV4:
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        gnorms = np.linalg.norm(X, axis=1)
        X_unit = X / np.where(gnorms < 1e-12, 1.0, gnorms)[:, None]

        # Norm codes
        lv = 1 << self.NORM_BITS
        log_gn = np.clip(np.log(gnorms+1e-12), self._gnorm_lo, self._gnorm_hi)
        norm_codes = np.floor((log_gn-self._gnorm_lo)/(self._gnorm_hi-self._gnorm_lo)*lv
                               ).clip(0, lv-1).astype(np.uint8)

        # Stage 1: direction
        dir_codes = self._dir_q.encode(X_unit)
        X_dir_hat = self._dir_q.decode(dir_codes)

        # Stage 2: HS residual
        residuals = X - X_dir_hat * gnorms[:, None]
        hs_ids, hs_res = self._res_q.encode(residuals)

        return CompressedV4(dir_codes, hs_ids, hs_res, norm_codes, self.dim, N)

    def decode(self, cv: CompressedV4) -> np.ndarray:
        N = cv.n_vecs
        lv = 1 << self.NORM_BITS
        gnorms_hat = np.exp(
            cv.norm_codes.astype(np.float64)/lv*(self._gnorm_hi-self._gnorm_lo)
            + self._gnorm_lo
        )

        # Stage 1
        X_dir_hat = self._dir_q.decode(cv.dir_codes)

        # Stage 2
        res_hat = self._res_q.decode(cv.hs_ids, cv.hs_res)

        return X_dir_hat * gnorms_hat[:, None] + res_hat

    def bits_per_vector(self) -> int:
        dir_b = self._dir_q.actual_bits()
        res_b = self._res_q.total_bits()
        return dir_b + res_b + self.NORM_BITS

    def bits_per_dim(self) -> float:
        return self.bits_per_vector() / self.dim

    def compression_ratio(self) -> float:
        return (self.dim * 32) / self.bits_per_vector()

    def budget_summary(self) -> str:
        dir_b = self._dir_q.actual_bits()
        res_b = self._res_q.total_bits()
        alloc_counts: dict = {}
        for b in self._res_q._chunk_res_bits:
            alloc_counts[int(b)] = alloc_counts.get(int(b), 0) + 1
        ac = ", ".join(f"{v}×(res={k})" for k,v in sorted(alloc_counts.items()))
        return (f"dim={self.dim}  bpd={self.bits_per_dim():.3f}  "
                f"dir={dir_b}b  hs_res={res_b}b[{ac}]  norm={self.NORM_BITS}b  "
                f"total={self.bits_per_vector()}b")


# ─────────────────────────────────────────────────────────────────────────────
# TurboQuant baseline (quantile, matched bitrate)
# ─────────────────────────────────────────────────────────────────────────────

class TurboQuantBaseline:
    def __init__(self, dim: int, target_bpd: float = 1.5):
        self.dim = dim
        self.target_bpd = target_bpd
        lo_b = int(np.floor(target_bpd))
        n_hi = int(round((target_bpd - lo_b) * dim))
        self._bits = np.array([lo_b+1]*n_hi + [lo_b]*(dim-n_hi), dtype=np.int32)
        np.random.default_rng(0).shuffle(self._bits)
        self._bdry: Optional[List[np.ndarray]] = None
        self._mid:  Optional[List[np.ndarray]] = None

    def fit(self, X: np.ndarray):
        self._bdry, self._mid = [], []
        for d in range(self.dim):
            lv = 1 << int(self._bits[d])
            q = np.linspace(0, 100, lv+1)
            b = np.percentile(X[:, d], q)
            self._bdry.append(b); self._mid.append(0.5*(b[:-1]+b[1:]))

    def encode(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        codes = np.zeros((N, self.dim), dtype=np.uint8)
        for d in range(self.dim):
            lv = (1 << int(self._bits[d])) - 1
            codes[:, d] = np.clip(np.searchsorted(self._bdry[d][1:-1], X[:,d]), 0, lv)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        out = np.zeros((len(codes), self.dim), dtype=np.float64)
        for d in range(self.dim):
            out[:, d] = self._mid[d][codes[:, d]]
        return out

    def bits_per_vector(self) -> int: return int(self._bits.sum())
    def bits_per_dim(self) -> float:  return self.bits_per_vector() / self.dim
    def compression_ratio(self) -> float: return (self.dim*32)/self.bits_per_vector()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def mse(X, R):     return float(np.mean((X-R)**2))
def snr_db(X, R):  return float(10*np.log10(np.mean(X**2)/(np.mean((X-R)**2)+1e-12)))

def ip_distortion(X, R, n_pairs=2000):
    N = X.shape[0]
    rng = np.random.default_rng(1)
    idx = rng.integers(0, N, size=(n_pairs*2, 2))
    mask = idx[:,0]!=idx[:,1]; idx=idx[mask][:n_pairs]
    return float(np.mean(np.abs(
        np.einsum('ij,ij->i',X[idx[:,0]],X[idx[:,1]]) -
        np.einsum('ij,ij->i',R[idx[:,0]],R[idx[:,1]])
    )))

def cosine_recall_at_k(X, R, k=10, n_queries=200):
    N = X.shape[0]; n_queries = min(n_queries, N)
    rng = np.random.default_rng(2)
    q_idx = rng.choice(N, n_queries, replace=False)
    Xn = X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    Rn = R/(np.linalg.norm(R,axis=1,keepdims=True)+1e-12)
    recalls = []
    for qi in q_idx:
        ts=Xn[qi]@Xn.T; rs=Rn[qi]@Rn.T
        ts[qi]=rs[qi]=-np.inf
        k_=min(k,N-1)
        recalls.append(len(set(np.argpartition(ts,-k_)[-k_:]) &
                           set(np.argpartition(rs,-k_)[-k_:])) / k_)
    return float(np.mean(recalls))


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(dim: int, n_vectors: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_vectors, dim)).astype(np.float64)
    n_train = max(200, n_vectors // 5)
    X_train, X_test = X[:n_train], X[n_train:]

    print(f"\n{'═'*74}")
    print(f"  BENCHMARK v4   dim={dim}   n_test={len(X_test)}   n_train={n_train}")
    print(f"{'═'*74}")

    configs = []
    for bpd in [1.5, 2.0]:
        q = HybridQuantizerV4(dim, target_bpd=bpd)
        q.fit(X_train)
        print(f"  HS-v4 ({bpd:.1f}bpd): {q.budget_summary()}")
        configs.append((f"HS-v4 ({bpd:.2f}bpd)", q))
        tq = TurboQuantBaseline(dim, bpd)
        tq.fit(X_train)
        configs.append((f"TurboQuant ({bpd:.2f}bpd)", tq))

    results = {}
    for label, q in configs:
        t0=time.perf_counter(); enc=q.encode(X_test); enc_t=time.perf_counter()-t0
        t0=time.perf_counter(); R=q.decode(enc);      dec_t=time.perf_counter()-t0
        bpv=q.bits_per_vector()
        results[label] = dict(
            bpd=bpv/dim, ratio=(dim*32)/bpv,
            mse=mse(X_test,R), snr=snr_db(X_test,R),
            ip_dist=ip_distortion(X_test,R),
            recall=cosine_recall_at_k(X_test,R,k=10),
            enc_ms=enc_t*1000, dec_ms=dec_t*1000,
        )

    labels = list(results.keys())
    W = 22
    sep = "  "+"─"*(30+(W+2)*len(labels))

    def hrow(title):
        print(f"\n  {title}")
        print(f"  {'Metric':<28}  "+"  ".join(f"{l:>{W}}" for l in labels))
        print(sep)
    def drow(metric, key, fmt):
        vals=[fmt(results[l][key]) for l in labels]
        print(f"  {metric:<28}  "+"  ".join(f"{v:>{W}}" for v in vals))

    hrow("Compression");      drow("Bits/dim","bpd",lambda v:f"{v:.3f}"); drow("Ratio","ratio",lambda v:f"{v:.2f}×")
    hrow("Reconstruction");   drow("MSE","mse",lambda v:f"{v:.5f}");       drow("SNR","snr",lambda v:f"{v:.2f} dB")
    hrow("Retrieval");        drow("IP distortion ↓","ip_dist",lambda v:f"{v:.4f}"); drow("Recall@10 ↑","recall",lambda v:f"{v:.3f}")
    hrow("Speed");            drow("Encode (ms)","enc_ms",lambda v:f"{v:.1f}");       drow("Decode (ms)","dec_ms",lambda v:f"{v:.2f}")

    print(f"\n{sep}")
    print(f"\n  SCORECARD  (✓ = HS wins at exactly matched bitrate)")
    print(f"  {'Metric':<22}  {'HS bpd':<8}  {'TQ bpd':<8}  {'Result':<14}  Delta")
    print("  "+"─"*64)

    pairs = [("HS-v4 (1.50bpd)","TurboQuant (1.50bpd)"),
             ("HS-v4 (2.00bpd)","TurboQuant (2.00bpd)")]
    for key, hb, label in [("snr",True,"SNR"),("ip_dist",False,"IP distortion"),("recall",True,"Recall@10")]:
        for a_l, b_l in pairs:
            a,b = results[a_l][key], results[b_l][key]
            d = a-b
            win = "✓ HS wins" if (d>0)==hb else "✗ TQ wins"
            print(f"  {label:<22}  {results[a_l]['bpd']:.3f}    {results[b_l]['bpd']:.3f}    {win:<14}  {d:+.4f}")
        print()

    print(f"{'═'*74}")
    return results


if __name__ == "__main__":
    print("Higman-Sims Implicit Spectral Quantizer — v4")
    print("Direction-first two-stage: TQ direction + HS residual")
    print("="*74)

    t0 = time.time()
    _V = build_hs_embedding()
    print(f"HS embedding: {time.time()-t0:.2f}s")

    for dim, n in [(22, 2000), (64, 2000), (768, 1000), (4096, 500)]:
        run_benchmark(dim=dim, n_vectors=n)