import numpy as np
from scipy.linalg import eigh

# HS verified. Now compute the spectral embedding and test eigenvalues.
def golay_codewords_weight_8():
    G = np.array([
        [1,0,0,0,0,0,0,0,0,0,0,0, 1,1,0,1,1,1,0,0,0,1,0,1],
        [0,1,0,0,0,0,0,0,0,0,0,0, 1,0,1,1,1,0,0,0,1,0,1,1],
        [0,0,1,0,0,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,1,0,1,1,1],
        [0,0,0,1,0,0,0,0,0,0,0,0, 1,1,1,0,0,0,1,0,1,1,0,1],
        [0,0,0,0,1,0,0,0,0,0,0,0, 1,1,0,0,0,1,0,1,1,0,1,1],
        [0,0,0,0,0,1,0,0,0,0,0,0, 1,0,0,0,1,0,1,1,0,1,1,1],
        [0,0,0,0,0,0,1,0,0,0,0,0, 0,0,0,1,0,1,1,0,1,1,1,1],
        [0,0,0,0,0,0,0,1,0,0,0,0, 0,0,1,0,1,1,0,1,1,1,0,1],
        [0,0,0,0,0,0,0,0,1,0,0,0, 0,1,0,1,1,0,1,1,1,0,0,1],
        [0,0,0,0,0,0,0,0,0,1,0,0, 1,0,1,1,0,1,1,1,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,1,0, 0,1,1,0,1,1,1,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1,1,1,0],
    ], dtype=np.int8)
    octads = []
    for i in range(1 << 12):
        bits = np.array([(i >> k) & 1 for k in range(12)], dtype=np.int8)
        cw = bits @ G % 2
        if cw.sum() == 8:
            octads.append(frozenset(np.where(cw)[0]))
    return octads

def build_hs():
    octads = golay_codewords_weight_8()
    fixed = {0, 1}
    blocks = []
    for oct in octads:
        if fixed.issubset(oct):
            blocks.append(frozenset(x-2 for x in oct if x not in fixed))
    
    n = 100
    A = np.zeros((n, n), dtype=np.int8)
    POINTS = list(range(1, 23))
    BLOCKS = list(range(23, 100))
    
    for p in POINTS: A[0, p] = A[p, 0] = 1
    for bi, b in enumerate(blocks):
        bv = BLOCKS[bi]
        for p in b:
            pv = POINTS[p]
            A[pv, bv] = A[bv, pv] = 1
    for i in range(len(blocks)):
        for j in range(i+1, len(blocks)):
            if len(blocks[i] & blocks[j]) == 0:
                A[BLOCKS[i], BLOCKS[j]] = A[BLOCKS[j], BLOCKS[i]] = 1
    return A

A = build_hs()
print("Computing eigendecomposition...")
eigenvalues, eigenvectors = eigh(A.astype(np.float64))
eigenvalues_rounded = np.round(eigenvalues).astype(int)
from collections import Counter
ev_counts = Counter(eigenvalues_rounded)
print("Eigenvalue multiplicities:", dict(sorted(ev_counts.items())))
# Expected: {-8: 22, 2: 77, 22: 1}

# The spectral embedding uses the eigenvectors for eigenvalue -8 (22 eigenvectors)
# This is the "22-dimensional eigenspace" corresponding to the non-trivial representation
# Reference: Delsarte scheme theory - the -8 eigenspace gives the optimal 22D embedding

neg8_idx = np.where(eigenvalues_rounded == -8)[0]
print(f"\nEigenvalue -8 indices: {len(neg8_idx)} eigenvectors (should be 22)")

eig2_idx = np.where(eigenvalues_rounded == 2)[0]
print(f"Eigenvalue  2 indices: {len(eig2_idx)} eigenvectors (should be 77)")

eig22_idx = np.where(eigenvalues_rounded == 22)[0]
print(f"Eigenvalue 22 indices: {len(eig22_idx)} eigenvector (should be 1)")

# The 22D embedding: project each vertex onto the -8 eigenspace
V_neg8 = eigenvectors[:, neg8_idx]  # shape (100, 22)
print(f"\nEmbedding matrix shape: {V_neg8.shape}")
print("Norms of embedded vertices (should be uniform):")
norms = np.linalg.norm(V_neg8, axis=1)
print(f"  min={norms.min():.6f}, max={norms.max():.6f}, std={norms.std():.8f}")
print("All norms equal?", np.allclose(norms, norms[0]))

# Inner products between adjacent vs non-adjacent vertices
adj_ips, non_adj_ips = [], []
for i in range(100):
    for j in range(i+1, 100):
        ip = V_neg8[i] @ V_neg8[j]
        if A[i,j]: adj_ips.append(ip)
        else: non_adj_ips.append(ip)

print(f"\nInner products - adjacent vertices: mean={np.mean(adj_ips):.6f}, std={np.std(adj_ips):.8f}")
print(f"Inner products - non-adjacent: mean={np.mean(non_adj_ips):.6f}, std={np.std(non_adj_ips):.8f}")
print("Adjacent IP unique values:", set(np.round(adj_ips, 6)))
print("Non-adjacent IP unique values:", set(np.round(non_adj_ips, 6)))
