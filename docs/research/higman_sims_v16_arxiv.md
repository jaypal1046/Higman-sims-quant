# Higman-Sims V16: Recursive Block-Wise Normalization With E8 Lattice Projection

**Authors:** Jayprakash Pal  
**Date:** April 2026  
**Status:** Pre-print style project note

## Abstract

V16 extends the earlier Higman-Sims quantization work by adding recursive block-wise normalization before E8 projection. The goal is to improve alignment between local activation statistics and the lattice geometry. In local experiments in this repository, V16 reaches very high reconstruction fidelity at higher bitrates. These results are encouraging, but they should not yet be treated as a final claim about broad model-level superiority.

## Main Claim Supported By Current Evidence

The safest claim is that local normalization appears to help in the high-fidelity regime when the metadata overhead is affordable.

## Open Questions

- How well does the pattern transfer across models?
- How stable is it at much longer contexts?
- What does the trade-off look like when measured against strong baselines under one protocol?

## Conclusion

V16 is best viewed as a promising high-fidelity research variant rather than a settled universal solution.
