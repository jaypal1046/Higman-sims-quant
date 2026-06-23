# Higman-Sims V12: Adaptive Syndrome-Lattice Hybrid for Vector Compression

**Authors:** Jayprakash Pal  
**Date:** April 2026  
**Status:** Pre-print style project note

## Abstract

This note describes V12, an early Higman-Sims quantization variant aimed at aggressive compression. V12 combines E8-based chunk quantization with sparse recursive refinement and limited metadata overhead. In local experiments on vector datasets used in this repository, V12 shows an interesting bitrate versus distortion trade-off and supports retrieval-style validation. End-to-end LLM evaluation was limited at the time this note was first written, so the conclusions should be read as preliminary.

## Key Points

- V12 favors low metadata overhead.
- Sparse refinement is used to spend bits on high-error chunks.
- The design is especially useful as a comparison point for later local-normalization variants.

## Limits

The V12 results here do not establish broad superiority over other quantizers. They mainly show that the design is worth investigating further.

## Conclusion

V12 remains a useful research baseline inside this repository because it highlights the low-bitrate regime clearly and helps motivate later variants such as V16 and V19.
