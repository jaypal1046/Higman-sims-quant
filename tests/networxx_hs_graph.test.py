# The circulant set was wrong. Let's find the correct one.
# srg(100,22,0,6) - we need to construct it properly.
# 
# Best approach: use the known construction from the Hoffman-Singleton complement
# or directly from Brouwer's database.
# 
# Let's verify what Paley(25) gives and build from the known HS construction.

import numpy as np

# The Higman-Sims graph can be constructed as follows (Coolsaet 2006):
# Take the vertices as pairs (b, P) where b is a block and P a point of PG(2,4)
# incident to b, plus some special vertices. This is complex.
#
# ALTERNATIVE: Use the known construction from the Sims approach:
# 100 vertices = 1 + 22 + 77
# where 22 = points of a projective plane minus a conic,
# and 77 = blocks of the associated design.
#
# Let's use the simplest known explicit construction:
# HS from the 22-element Steiner system.
#
# Actually, the most reliable programmatic construction:
# use the known 100 edges from the "doubly regular tournament" method.

# DEFINITIVE: Construct from the Gewirtz graph.
# Gewirtz graph = unique srg(56, 10, 0, 2)
# HS = cone over complement of Gewirtz graph ... no that's not right either.

# The HS graph has the following KNOWN explicit construction:
# Vertices: the 100 subsets of {1..7} of size 0, 3, or 4 ... no.

# Let's use the DIRECT matrix construction from the known spectral properties:
# Build a symmetric {0,1} matrix with the right spectrum and verify.

# Actually the most reliable method in numpy/scipy:
# Use the networkx library if available, or implement from 
# the known orbit structure.

# Let's try networkx
import subprocess
r = subprocess.run(['pip', 'install', 'networkx', '--break-system-packages', '-q'], capture_output=True)
import networkx as nx
print("networkx version:", nx.__version__)

# NetworkX has HS graph built in
G = nx.higman_sims_graph()
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
A = nx.to_numpy_array(G, dtype=np.int8)
deg = A.sum(axis=1)
print("Degree (should all be 22):", np.unique(deg))
A2 = (A.astype(np.int32)) @ (A.astype(np.int32))
lam_vals, mu_vals = set(), set()
for i in range(100):
    for j in range(i+1, 100):
        v = A2[i,j]
        if A[i,j]: lam_vals.add(v)
        else: mu_vals.add(v)
print("Lambda (adj, should be {0}):", lam_vals)
print("Mu (non-adj, should be {6}):", mu_vals)
print("srg(100,22,0,6) VERIFIED:", lam_vals=={0} and mu_vals=={6})