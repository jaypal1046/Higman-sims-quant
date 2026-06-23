"""Stable public hybrid API used by tests and research scripts."""

from .core.v12 import Untouchable_Core as V12_Core
from .core.v16 import Final_God_V16 as V16_Core


class HigmanSims_Hybrid:
    """Switch between the low-BPD V12 path and the high-fidelity V16 path."""

    def __init__(self, dim, target_bpd=8.0):
        self.dim = int(dim)
        self.target_bpd = float(target_bpd)

        if self.target_bpd < 5.5:
            self.mode = "V12_GLOBAL"
            stages = max(1, int(self.target_bpd))
            self.engine = V12_Core(self.dim, stages=stages)
        else:
            self.mode = "V16_SINGULARITY"
            stages = max(1, int((self.target_bpd - 4.0) / 1.5))
            self.engine = V16_Core(self.dim, max_stages=stages)

        print(
            f"Hybrid Engine Initialized: Mode={self.mode} | "
            f"Target BPD={self.target_bpd}"
        )

    def fit(self, X):
        return self.engine.fit(X)

    def encode(self, X):
        return self.engine.encode(X)

    def decode(self, co):
        return self.engine.decode(co)

    def measure_efficiency(self, co):
        if hasattr(self.engine, "measure_efficiency"):
            measured_bpd = self.engine.measure_efficiency(co)
        else:
            measured_bpd = self.engine.bpd(co)

        return measured_bpd, measured_bpd


__all__ = ["HigmanSims_Hybrid"]
