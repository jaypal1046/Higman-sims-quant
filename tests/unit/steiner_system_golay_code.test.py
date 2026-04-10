import numpy as np
import networkx as nx

# Build HS graph from scratch using the known explicit construction.
# Reference: Brouwer, Cohen, Neumaier "Distance-Regular Graphs" (1989)
# and the explicit construction via two-graph switching.
#
# THE DEFINITIVE CONSTRUCTION for srg(100, 22, 0, 6):
#
# Use vertices labelled as elements of Z_2^2 x Z_5^2 union {infty}...
# 
# Actually the clearest known construction:
# Vertices = {0,...,99} represented as pairs (row, col) in 10x10
# with a specific Cayley-like adjacency derived from the group Z_5 x Z_5.
#
# From: "The construction of the Higman-Sims graph"
# using the unique self-complementary strongly regular tournament on 22 points.
#
# Let's use the explicit method: 
# 100 vertices = GF(4)^2 union {extra}, built via the known 100-graph.
#
# SIMPLEST VERIFIED APPROACH: use the two known orbits of 
# the Mathieu group M22 action. We'll implement the "Sims graph"
# construction directly.

# HS graph construction from the unique 2-(22, 6, 5) design:
# Points: {0..21}
# The design has 77 blocks (6-subsets)
# Vertices of HS: {inf} + 22 points + 77 blocks = 100
# 
# Edges:
#   inf ~ all 22 points
#   point p ~ point q: NEVER (points are independent)  
#   point p ~ block B: p in B (each block has 6 points, each point in 21 blocks... wait)
#   Actually in the (22,6,5) design: 77 blocks, each point in (77*6/22) = 21 blocks
#   point p ~ block B: p NOT in B
#   block B ~ block C: |B ∩ C| = 0 or |B ∩ C| = 1? Let's derive.
#
# Check degrees:
#   inf: 22 edges to points ✓
#   point p: 0 (to inf) + 0 (to points) + (77-21) = 56 to non-incident blocks → too many
#
# That's not right. Let me use the correct HS construction.
#
# CORRECT construction (from Higman-Sims original paper 1968):
# Take the unique (22, 3, 1)-design (Steiner triple system S(3,6,22) - no that's S(3,6,22))
# Actually it's S(3, 6, 22): 3-(22, 6, 1) design, which has 77 blocks.
# 
# Vertices: {∞} ∪ P ∪ B, |P|=22, |B|=77
# Edges:
#   ∞ ~ all of P
#   p ∈ P, q ∈ P: p≁q (no edges within P)  
#   p ∈ P, B ∈ B: p~B iff p ∉ B
#   B, C ∈ B: B~C iff |B ∩ C| = 0
#
# Check degree of p ∈ P:
#   1 (to ∞) + 0 (within P) + |{B: p∉B}| = 1 + (77 - blocks_through_p)
#   In 3-(22,6,1): blocks through any point = C(21,2)/C(5,2) = 210/10 = 21
#   degree(p) = 1 + 0 + (77-21) = 57 ← WRONG, need 22
#
# That construction is wrong. Let me use the correct one.
# 
# CORRECT: from Brouwer's website and the original construction:
# Use S(3,6,22) but with p~B iff p IN B (not outside).
# degree(p) = 1 + 0 + 21 = 22 ✓
# 
# degree(B): need 22 edges from B.
#   B has 6 points (all adjacent to B... wait, from points-to-blocks direction, 
#   edges go both ways): edges from B to points: 6 (the points IN B)
#   edges from B to blocks C: need 22 - 6 = 16
#   |{C ∈ B: |B ∩ C| = 0 or some value}|
#   In 3-(22,6,1): two blocks B,C: |B∩C| can be 0,1,2,3
#   We need B~C iff |B∩C| = 1: let's check count.
#   Given B, number of blocks C with |B∩C|=1:
#   Choose 1 of 6 points from B (6 ways), choose 5 points from remaining 16 points.
#   But in 3-(22,6,1), C is determined by any 3 of its points.
#   C(6,1)*C(16,5)/C(6,5) ... this requires careful computation.
#   
#   Alternative: B~C iff |B∩C| = 2. Let's just verify all options programmatically.

# We need the actual blocks of the 3-(22,6,1) Steiner system.
# The unique 3-(22,6,1) design is related to the Mathieu group M22.
# Its blocks are the 77 hexads of the extended binary Golay code restricted to 22 coordinates.

# Let's construct the S(3,6,22) design computationally.
# We know it comes from the S(5,8,24) Witt design by fixing 3 points.

# Since implementing the full Golay code is complex, let's use a different 
# verified approach: construct HS as the Cayley graph of a specific group.

# VERIFIED APPROACH: Use the known fact that HS is vertex-transitive
# and its connection set in terms of the automorphism group.

# Best practical approach: use sage-like construction via the known
# adjacency from the complement of the Kneser graph K(22,2) ... no.

# Let's just directly use the Gewirtz graph construction:
# The Gewirtz graph is unique srg(56,10,0,2). 
# HS = the unique graph where we can partition vertices into:
# {cone point} + {56 Gewirtz vertices} + {other 43 vertices}
# -- this is getting complex.

# PRACTICAL SOLUTION: encode the HS graph as the following
# known edge list (verified from multiple sources):

# From Brouwer's database + own verification, HS = srg(100,22,0,6) 
# with the following known symmetric difference representation.
# We implement it via the "star" construction over GF(4).

# After research: the cleanest programmable construction is via
# the Mathieu group M22 + its action on 22 points.
# We'll use a direct orbit construction.

# THE CONSTRUCTION WE'LL USE (from Weisstein/Wolfram + Brouwer):
# Label vertices 0..99. The edges are determined by:
# - 1 vertex at "level 0" (vertex 0)
# - 22 vertices at "level 1" (vertices 1-22): all adjacent to vertex 0
# - 77 vertices at "level 2" (vertices 23-99)
# Level 1 vertices form independent set
# Level 1 i ~ Level 2 j based on the (22,6,5)-design incidence
# Level 2 i ~ Level 2 j based on design block intersection

# We'll construct the blocks of the 2-(22,6,5) design
# (which is derived from the 3-(22,6,1) by dropping one repeat constraint)
# Hmm, actually the block size matters.

# Let's pivot to a FULLY VERIFIED construction:
# Use the known fact that HS contains the Petersen graph,
# and construct it via the known 100-vertex 5-arc-transitive structure.

# FINAL DECISION: Implement via the known M22-orbit construction
# using the combinatorial characterization as the unique
# graph with intersection array {22; 21; 1, 6}.

# Actually: let's just verify the srg parameters for a known
# computer-generated HS adjacency by using the following 
# hardcoded neighbour lists (first 5 vertices to verify structure):

# I'll use the construction from:
# https://mathworld.wolfram.com/Higman-SimsGraph.html
# HS = complement of the unique srg(100, 77, 60, 56)
# which is the "Sims graph" / "triangular graph" related construction.

# The most efficient path forward: implement HS via its
# KNOWN SPECTRUM + a random matrix projection, then 
# build the implementation around verified srg(100,22,0,6) properties.
# 
# For the spectral embedding, what matters is the SPECTRUM, 
# which we know exactly: eigenvalues {22, 2, -8} with multiplicities {1, 77, 22}.

# ==== IMPLEMENT THE STEINER SYSTEM S(3,6,22) ====
# We'll derive it from the extended ternary Golay code... 
# Actually let's use the direct known block list from the 
# small Witt design construction:

# The 77 blocks of S(3,6,22) can be computed from the
# S(5,8,24) (large Witt design) by fixing 3 points.
# We implement S(5,8,24) from the (binary) extended Golay code.

def golay_codewords_weight_8():
    """Generate the 759 weight-8 codewords of the [24,12,8] extended Golay code."""
    # Generator matrix of the extended binary Golay code
    # (systematic form, from standard reference)
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
            octads.append(tuple(np.where(cw)[0]))
    return octads

print("Computing Golay code weight-8 codewords (759 expected)...")
octads = golay_codewords_weight_8()
print(f"Found {len(octads)} octads")

# S(5,8,24): the 759 octads are the blocks of the large Witt design
# Fix 3 points (say 0,1,2) to get S(3,6,22) restricted to points {3..23}
# Relabelled as {0..21}
print("Deriving S(3,6,22) blocks...")
s3_blocks = []
for oct in octads:
    oct_set = set(oct)
    if 0 in oct_set and 1 in oct_set and 2 in oct_set:
        # Remaining 5 points form... wait, 8-3=5 ≠ 6. Need different fixing.
        pass

# Fix 2 points: get S(3,6,22) with 22 points (indices in {0..23} \ {fixed two})
# Fix points 0 and 1: get S(3,6,22) on {2..23} (22 points) 
# by taking octads containing both 0 and 1: 8 choose 2 pairs ... 
# Number of such octads = 77 ✓ (standard result)

fixed = {0, 1}
s3_blocks_raw = []
for oct in octads:
    if fixed.issubset(set(oct)):
        remaining = tuple(sorted(x for x in oct if x not in fixed))
        s3_blocks_raw.append(remaining)

print(f"S(3,6,22) blocks (77 expected): {len(s3_blocks_raw)}")

# Relabel points {2..23} -> {0..21}
s3_blocks = [tuple(x-2 for x in b) for b in s3_blocks_raw]
print("Block size (should all be 6):", set(len(b) for b in s3_blocks))
print("First 3 blocks:", s3_blocks[:3])

# Verify it's a 3-(22,6,1) design
# Every 3-element subset of {0..21} should appear in exactly 1 block
from itertools import combinations
triple_count = {}
for b in s3_blocks:
    for triple in combinations(b, 3):
        triple_count[triple] = triple_count.get(triple, 0) + 1
counts = set(triple_count.values())
print("Triple coverage (should be {1}):", counts)
print("Number of triples covered:", len(triple_count), "of", len(list(combinations(range(22), 3))))
