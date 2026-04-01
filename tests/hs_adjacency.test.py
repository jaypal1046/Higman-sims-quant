import numpy as np
from itertools import combinations

# S(3,6,22) is verified. Now build the HS graph from it.
# Construction (from Higman-Sims 1968):
# Vertices: {inf} union {22 points} union {77 blocks} = 100 total
# Vertex numbering: inf=0, points=1..22, blocks=23..99
#
# Edges:
#   inf ~ all 22 points
#   point p ~ point q: NEVER
#   point p ~ block B: p IN B  (6 edges per point from this, giving degree 1+6+? - let's check)
#   block B ~ block C: |B ∩ C| = 0 (disjoint)
#
# Degree of inf: 22 ✓
# Degree of point p: 1 (inf) + 0 (points) + |{B: p in B}|
#   Each point in 3-(22,6,1) is in λ = 21 blocks (standard calculation)
#   So degree = 1 + 21 = 22 ✓
# Degree of block B: |B| (to points) + |{C: |B∩C|=0}|
#   = 6 + |{C: B,C disjoint}|
#   Need this to be 22, so |{C: disjoint}| = 16

# Let's verify
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

octads = golay_codewords_weight_8()
fixed = {0, 1}
s3_blocks = []
for oct in octads:
    if fixed.issubset(oct):
        remaining = frozenset(x-2 for x in oct if x not in fixed)
        s3_blocks.append(remaining)

points = list(range(22))
blocks = s3_blocks  # list of frozensets

# Check point-in-block counts
point_block_count = {p: sum(1 for b in blocks if p in b) for p in points}
print("Blocks per point (should all be 21):", set(point_block_count.values()))

# Check disjoint block counts
disjoint_count = {i: sum(1 for j,c in enumerate(blocks) if i!=j and len(b & c)==0) 
                  for i,b in enumerate(blocks)}
print("Disjoint blocks per block (should all be 16):", set(disjoint_count.values()))

# Great! Now build the HS adjacency matrix
n = 100
A = np.zeros((n, n), dtype=np.int8)

INF = 0
POINTS = list(range(1, 23))  # vertex i+1 = point i
BLOCKS = list(range(23, 100)) # vertex j+23 = block j

# inf ~ all points
for p in POINTS:
    A[INF, p] = A[p, INF] = 1

# point p ~ block B if p in B
for bi, b in enumerate(blocks):
    bv = BLOCKS[bi]  # vertex index of block
    for p in b:
        pv = POINTS[p]  # vertex index of point p
        A[pv, bv] = A[bv, pv] = 1

# block B ~ block C if disjoint
for i in range(len(blocks)):
    for j in range(i+1, len(blocks)):
        if len(blocks[i] & blocks[j]) == 0:
            A[BLOCKS[i], BLOCKS[j]] = A[BLOCKS[j], BLOCKS[i]] = 1

# Verify
deg = A.sum(axis=1)
print("Degree sequence (should all be 22):", np.unique(deg))

A2 = A.astype(np.int32) @ A.astype(np.int32)
lam_set, mu_set = set(), set()
for i in range(100):
    for j in range(i+1, 100):
        v = int(A2[i,j])
        if A[i,j]: lam_set.add(v)
        else: mu_set.add(v)
print("Lambda (adj pairs, should be {0}):", lam_set)
print("Mu (non-adj pairs, should be {6}):", mu_set)
print("HS srg(100,22,0,6) VERIFIED:", lam_set=={0} and mu_set=={6} and set(deg)=={22})
