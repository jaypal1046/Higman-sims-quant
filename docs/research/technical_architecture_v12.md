# Technical Architecture: Higman-Sims V12

## Summary

V12 is an earlier low-bitrate-oriented variant that uses global statistics, sparse refinement, and E8-based chunk quantization.

## Design Goal

- minimize metadata overhead
- reserve more bits for active chunks
- study whether sparse recursive refinement helps retrieval-style behavior

## Pipeline

1. Apply global centering or scaling.
2. Partition the residual into 8D chunks.
3. Quantize chunks using E8 structure.
4. Apply sparse refinement to the highest-error chunks.
5. Reconstruct from the stored sparse codes and global statistics.

## Interpretation

V12 is useful because it highlights a different trade-off from V16. It does not spend as much on local metadata, which makes it relevant when the bitrate budget is tight.

## Status

V12 should be treated as a research baseline and exploratory prototype, not as a final production recommendation.
