# Differential Manifold and Syndrome-Lattice Hybrid Quantization (Higman-Sims V12)

**Author:** Jayprakash Pal  
**Date:** April 2026

## Abstract

This note summarizes the V12 branch of the Higman-Sims quantization work. V12 focuses on low-bitrate compression using sparse refinement and limited metadata. In local vector-level experiments, it shows interesting trade-offs between bitrate, distortion, and retrieval behavior. The document should be read as a research note rather than a conclusive benchmark paper.

## What V12 Contributes

- a low-metadata comparison point against later variants
- a sparse refinement strategy for high-error chunks
- a structured E8-based quantization path that is easy to inspect analytically

## Current Limits

The local V12 results are useful, but they do not establish that the approach is best overall. End-to-end model quality, long-context behavior, and runtime measurements still need broader evaluation.

## Conclusion

V12 remains valuable inside the project because it makes the low-bitrate regime concrete and helps explain why later variants moved toward stronger local normalization.
