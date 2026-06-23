# Project Roadmap: From Research Prototype Toward Runtime Integration

## Why This Roadmap Exists

The repository has enough working pieces to justify integration experiments, but it is still in the research stage. This roadmap is about narrowing the gap between reconstruction experiments and practical inference systems.

## Near-Term Goals

### Phase A: GPU Kernel Work

- move the hot E8 projection path into Triton or another fused kernel
- measure actual throughput and latency
- compare runtime overhead against the memory saved

### Phase B: Inference Integration

- test cleaner integration with Hugging Face and vLLM-style runtimes
- quantify the effect of quantized caches on decoding quality
- separate research-mode code paths from runtime-safe code paths

### Phase C: Better Evaluation

- run longer-context experiments
- compare against stronger baselines
- measure serialized size rather than only entropy-style estimates

## Current Position

The math and prototype code are far enough along to justify integration work, but not far enough to claim production readiness.
