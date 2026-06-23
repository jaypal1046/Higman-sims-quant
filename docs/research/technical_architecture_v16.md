# Technical Architecture: Higman-Sims V16

## Summary

V16 is a higher-fidelity variant in this repository. Its main change relative to earlier versions is moving normalization from the whole tensor to smaller local blocks before E8 projection.

## Design Goal

- Explore whether local normalization helps high-bitrate reconstruction quality.
- Accept extra metadata overhead in exchange for lower reconstruction error.
- Provide a cleaner comparison point against lower-bitrate variants such as V12.

## Pipeline

1. Partition the tensor into 8D blocks.
2. Compute local mean and scale for each block.
3. Normalize each block.
4. Project onto the E8 lattice.
5. Apply recursive residual refinement across multiple stages.
6. Reconstruct using the stored local statistics.

## Interpretation

The working hypothesis behind V16 is that local centering reduces the mismatch between the activation distribution and the lattice geometry. In local experiments inside this repo, that tends to help reconstruction quality once the bitrate budget is high enough to absorb the metadata cost.

## Caveat

The repo contains strong reconstruction results for V16, but those results should not be read as proof of broad model-level superiority. They are best treated as evidence that the design is worth further evaluation.

## Status

V16 is a research implementation with useful local validation coverage. It is not yet a production deployment recommendation on its own.
