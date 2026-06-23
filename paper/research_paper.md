# Hybrid E8 Lattice Quantization With Recursive Block-Wise Normalization

## Abstract

This manuscript draft summarizes the central idea behind the repository: use E8 lattice projection together with staged normalization and residual refinement to study KV-cache compression for transformer inference. The current evidence in the repo is strongest on reconstruction fidelity and early end-to-end cache experiments. It should not yet be read as a final comparative paper.

## Working Hypothesis

Different bitrate regimes may favor different quantization strategies:

- lower-bitrate settings may benefit from limited metadata
- higher-bitrate settings may benefit from stronger local normalization

The repository explores that hypothesis across several variants rather than claiming one final design.

## Current Evidence

- high-fidelity reconstruction in local experiments
- kernel parity checks between NumPy and PyTorch implementations
- preliminary end-to-end autoregressive evaluation with quantized caches

## Remaining Work

- standardized baselines
- longer context lengths
- larger model coverage
- runtime and serialized storage measurements

## Conclusion

The project supports a credible research direction and now has a real end-to-end evaluation path. The appropriate public framing is that the method looks promising, not that it has already been proven best.
