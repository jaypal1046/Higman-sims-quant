import traceback, sys, numpy as np
import os

# Add current directory to path for imports
sys.path.append(os.getcwd())

try:
    from src.higman_sims_quant_v12 import Untouchable_Core
    from src.higman_sims_quant_v16 import Final_God_V16
    
    # Use exact 300 dimensions as in the paper
    X = np.random.randn(10, 300).astype(np.float32)
    
    print("Testing V12 fit...")
    eng12 = Untouchable_Core(300)
    # The error likely happens here
    eng12.fit(X)
    print("V12 fit OK.")
    
    print("Testing V16 fit...")
    eng16 = Final_God_V16(300)
    # Or here
    eng16.fit(X)
    print("V16 fit OK.")
    
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
