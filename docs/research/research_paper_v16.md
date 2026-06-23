# Recursive Block Normalization in E8 Lattices (Higman-Sims V16)

**Author:** Jayprakash Pal  
**Date:** April 2026

## Abstract

This document describes the V16 variant of the Higman-Sims quantization work. V16 explores whether recursive block-wise normalization improves reconstruction quality at moderate and high bitrates by aligning each local 8D block before E8 projection. In local experiments in this repository, V16 reaches very high reconstruction fidelity at higher bitrates. These results are encouraging, but they should be interpreted as reconstruction-focused evidence rather than a final claim about end-to-end model quality.

## Main Idea

Earlier variants spend fewer bits on metadata and therefore behave differently in low-bitrate settings. V16 introduces local mean and scale metadata per block. That increases overhead, but it can also reduce residual energy and make staged refinement more effective.

## Local Results

The repository contains local runs where V16 reaches very high SNR at bitrates around 8.5 BPD. Those results support the idea that local normalization is useful when the bitrate budget is not extremely tight.

## Limits Of The Current Evidence

The V16 results in this repo do not by themselves establish that the method is best overall. Remaining work includes:

- evaluation across more models
- longer context lengths
- standardized datasets
- stronger baseline comparisons
- real runtime and storage measurements

## Conclusion

V16 is a meaningful step in the project because it demonstrates a strong high-fidelity regime. The safest conclusion is that local normalization appears promising and deserves broader evaluation.
