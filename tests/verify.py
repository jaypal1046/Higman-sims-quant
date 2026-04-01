import numpy as np
# Test that we can construct the HS graph adjacency matrix
# HS graph: 100 vertices, strongly regular srg(100,22,0,6)
# We'll construct it via the known Higman-Sims construction
# Verify numpy/scipy available
from scipy.linalg import eigh
print('numpy', np.__version__)
import scipy; print('scipy', scipy.__version__)
