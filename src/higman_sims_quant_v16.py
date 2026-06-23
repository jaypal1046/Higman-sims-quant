"""Compatibility wrapper for the legacy V16 import path."""

from .core.v16 import Final_God_V16 as _Final_God_V16


class Final_God_V16(_Final_God_V16):
    """Preserve the legacy tuple-shaped efficiency API for older scripts."""

    def measure_efficiency(self, co):
        entropy_bpd = super().measure_efficiency(co)
        return entropy_bpd, entropy_bpd


__all__ = ["Final_God_V16"]
