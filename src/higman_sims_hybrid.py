"""
Higman-Sims Hybrid Engine (V12 + V16)
=====================================
The "Singularity Crossover" Engine. 
- < 5.5 BPD: Uses V12 (Global Scaling) for baseline survival.
- >= 5.5 BPD: Uses V16 (Local BAS) for bit-exact Singularity.
"""

import math
from src.higman_sims_quant_v12 import Untouchable_Core as V12_Core
from src.higman_sims_quant_v16 import Final_God_V16 as V16_Core

class HigmanSims_Hybrid:
    """The Ultimate Hybrid: Switching logic at 5.5 BPD."""
    def __init__(self, dim, target_bpd=8.0):
        self.dim = dim
        self.target_bpd = target_bpd
        
        # Crossover logic
        if target_bpd < 5.5:
            # V12: BPD = stages (roughly)
            self.mode = "V12_GLOBAL"
            self.engine = V12_Core(dim, stages=int(target_bpd))
        else:
            # V16: BPD = 4 + stages * 1.5
            self.mode = "V16_SINGULARITY"
            stages = max(0, int((target_bpd - 4.0) / 1.5))
            self.engine = V16_Core(dim, max_stages=stages)
            
        print(f"Hybrid Engine Initialized: Mode={self.mode} | Target BPD={target_bpd}")

    def fit(self, X):
        return self.engine.fit(X)

    def encode(self, X):
        return self.engine.encode(X)

    def decode(self, co):
        return self.engine.decode(co)

    def measure_efficiency(self, co):
        if self.mode == "V16_SINGULARITY":
            return self.engine.measure_efficiency(co)
        else:
            # V12 proxy
            b = len(co) if isinstance(co, list) else 0.0
            return b, b
